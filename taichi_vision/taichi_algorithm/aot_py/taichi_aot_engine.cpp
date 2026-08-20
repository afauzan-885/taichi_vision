#include <cstring>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cwchar>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <taichi/cpp/taichi.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>
#include <condition_variable>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif


#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define EXPORT __declspec(dllexport)
#include <wincodec.h>
#include <windows.h>
#include <GL/gl.h>
#ifndef GLsizeiptr
typedef ptrdiff_t GLsizeiptr;
#endif
#include <glad/egl.h>
#include <taichi/taichi_opengl.h>
#include "raw_icd_gl_dispatch.h"

// CUDA-only bridge bundles intentionally do not link Taichi's OpenGL C-API
// object.  Keep the common bridge source linkable for that profile while
// returning a deterministic error if a caller tries to select OpenGL on a
// CUDA-only bundle.  Desktop CPU/Vulkan/OpenGL bridges leave this disabled
// and resolve the real C-API implementation.
#if defined(PIXEL_REFINE_AOT_DISABLE_OPENGL_INTEROP)
extern "C" TiRuntime TI_API_CALL
ti_import_opengl_runtime(TiOpenglRuntimeInteropInfo *, bool) {
  return nullptr;
}
#endif

#pragma comment(lib, "windowscodecs.lib")
#else
#define EXPORT
#endif

// Forward declaration for WIC factory (global for performance)
#ifdef _WIN32
static IWICImagingFactory *g_wic_factory = nullptr;
static void init_wic() {
  if (!g_wic_factory) {
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
    CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER,
                     IID_PPV_ARGS(&g_wic_factory));
  }
}
#endif


static bool is_debug_logging_enabled() {
    static bool checked = false;
    static bool enabled = false;
    if (!checked) {
        const char *env = std::getenv("AOT_ENGINE_ENABLE_DEBUG");
        if (env && std::string(env) == "1") {
            enabled = true;
        }
        checked = true;
    }
    return enabled;
}

// Match the saturating narrowing used by the SIMD conversion paths for every
// scalar tail and baseline bridge.  The old scalar casts were implementation
// defined for values outside [0, 1] (and for NaN), which could make CPU/ARM
// and GPU host-buffer normalization disagree at the dtype boundary.
static inline uint8_t ti_normalized_to_u8(float value) noexcept {
  if (!(value > 0.0f))
    return 0;
  if (value >= 1.0f)
    return 255;
  return static_cast<uint8_t>(value * 255.0f + 0.5f);
}

static inline uint16_t ti_normalized_to_u16(float value) noexcept {
  if (!(value > 0.0f))
    return 0;
  if (value >= 1.0f)
    return 65535;
  return static_cast<uint16_t>(value * 65535.0f + 0.5f);
}

// Keep signed 16-bit host-buffer casts defined across MSVC, Clang, and the
// ARM toolchains.  A floating-point value outside the representable range is
// clamped instead of relying on implementation-defined narrowing behavior.
static inline int16_t ti_float_to_i16(float value) noexcept {
  if (!(value > -32768.0f))
    return -32768;
  if (value >= 32767.0f)
    return 32767;
  return static_cast<int16_t>(value);
}

// -----------------------------------------------------------------------
// Dynamic Argument Structure
// -----------------------------------------------------------------------
struct DynamicArg {
  const char *name;
  int arg_type; // 0: ndarray, 1: scalar
  // Keep 0..3 stable for existing callers; 4/5 are compact CPU extensions.
  // This private struct is layout-compatible with the Python ctypes mirror.
  int dtype;    // 0: f32, 1: i32, 2: u8, 3: u16, 4: i16, 5: f16
  int dim_count;
  int32_t shape[8];
  int elem_dim_count;
  int32_t elem_shape[8];
  int is_vector;
  int vector_dim;
  uint64_t val_u64;
};

struct EngineContext;

// -----------------------------------------------------------------------
// Pipeline Structures (Global)
// -----------------------------------------------------------------------
struct GraphDispatch {
  void *module_ctx;
  std::string graph_name;
  std::vector<DynamicArg> args;
  std::vector<std::string> arg_names; // Storage for name pointers
};

struct Pipeline {
  std::vector<GraphDispatch> steps;
};

// Fallback store for legacy calls that do not provide an EngineContext.
static std::unordered_map<std::string, Pipeline> global_pipelines;
static std::mutex pipelines_mutex;

// -----------------------------------------------------------------------
// Internal Cache for Graphics Objects
// -----------------------------------------------------------------------
struct ModuleContext {
  EngineContext *owner;
  ti::AotModule *module;
  std::unordered_map<std::string, ti::ComputeGraph> graph_cache;
  std::mutex cache_mutex;
  std::mutex lifetime_mutex;
  std::condition_variable lifetime_cv;
  uint32_t active_calls = 0;
  bool destroying = false;
};

struct RawIcdContextTag;

struct GpuAllocationRecord {
  uint64_t size = 0;
  bool mapped = false;
};

struct EngineContext {
  TiArch arch;
  ti::Runtime *runtime;
  // Allocation handles are session-local.  Keeping the byte capacity and
  // mapped state beside the handle lets every memory entry point reject a
  // foreign handle or an out-of-range transfer before it reaches Taichi.
  std::unordered_map<TiMemory, GpuAllocationRecord> allocations;
  std::unordered_set<ModuleContext *> modules;
  std::unordered_map<std::string, Pipeline> pipelines;
  std::mutex mutex;
  std::condition_variable lifecycle_cv;
  uint32_t active_calls = 0;
  std::string last_error;
  bool destroying;
  uint64_t session_id;
  std::string device_name;
#ifdef _WIN32
  // Direct Windows OpenGL ICD mode. This path uses the vendor ICD exports
  // (DrvCreateContext/DrvSetContext) and never asks the system OpenGL loader
  // to pick an adapter.
  bool icd_mode = false;
  HWND icd_window = nullptr;
  HDC icd_dc = nullptr;
  HMODULE icd_module = nullptr;
  std::string icd_library_path;
  struct RawIcdContextTag *icd_context = nullptr;
  BOOL (WINAPI *icdSetPixelFormat)(HDC, int) = nullptr;
  struct RawIcdContextTag *(WINAPI *icdCreateContext)(HDC) = nullptr;
  const void *(WINAPI *icdSetContext)(HDC, struct RawIcdContextTag *,
                                      void (APIENTRY *)(const void *)) = nullptr;
  BOOL (WINAPI *icdDeleteContext)(struct RawIcdContextTag *) = nullptr;
  void (WINAPI *icdReleaseContext)(struct RawIcdContextTag *) = nullptr;
  PROC (WINAPI *icdGetProcAddress)(LPCSTR) = nullptr;
  const void *icd_table = nullptr;
  // EGL is the device-selectable, window-system-independent path. It is kept
  // separate from the direct ICD state so the bridge can make an OpenGL
  // context current on any worker thread without relying on the desktop
  // window/display adapter chosen by Windows.
  bool egl_mode = false;
  HMODULE egl_module = nullptr;
  std::string egl_library_path;
  PFNEGLGETDISPLAYPROC eglGetDisplay = nullptr;
  PFNEGLGETPROCADDRESSPROC eglGetProcAddress = nullptr;
  PFNEGLINITIALIZEPROC eglInitialize = nullptr;
  PFNEGLTERMINATEPROC eglTerminate = nullptr;
  PFNEGLBINDAPIPROC eglBindAPI = nullptr;
  PFNEGLCHOOSECONFIGPROC eglChooseConfig = nullptr;
  PFNEGLCREATEPBUFFERSURFACEPROC eglCreatePbufferSurface = nullptr;
  PFNEGLDESTROYSURFACEPROC eglDestroySurface = nullptr;
  PFNEGLCREATECONTEXTPROC eglCreateContext = nullptr;
  PFNEGLDESTROYCONTEXTPROC eglDestroyContext = nullptr;
  PFNEGLMAKECURRENTPROC eglMakeCurrent = nullptr;
  PFNEGLGETCURRENTDISPLAYPROC eglGetCurrentDisplay = nullptr;
  PFNEGLGETCURRENTSURFACEPROC eglGetCurrentSurface = nullptr;
  PFNEGLGETCURRENTCONTEXTPROC eglGetCurrentContext = nullptr;
  PFNEGLGETERRORPROC eglGetError = nullptr;
  PFNEGLQUERYSTRINGPROC eglQueryString = nullptr;
  PFNEGLGETPLATFORMDISPLAYPROC eglGetPlatformDisplay = nullptr;
  PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = nullptr;
  PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT = nullptr;
  PFNEGLQUERYDEVICESTRINGEXTPROC eglQueryDeviceStringEXT = nullptr;
  EGLDisplay egl_display = EGL_NO_DISPLAY;
  EGLSurface egl_surface = EGL_NO_SURFACE;
  EGLContext egl_context = EGL_NO_CONTEXT;
  std::recursive_mutex gl_context_mutex;
#endif
};

static void invalidate_pipelines_for_module(ModuleContext *ctx) {
  if (!ctx || !ctx->owner)
    return;
  std::lock_guard<std::mutex> engine_lock(ctx->owner->mutex);
  for (auto pipeline_it = ctx->owner->pipelines.begin();
       pipeline_it != ctx->owner->pipelines.end();) {
    auto &steps = pipeline_it->second.steps;
    steps.erase(std::remove_if(steps.begin(), steps.end(),
                               [ctx](const GraphDispatch &step) {
                                 return step.module_ctx == ctx;
                               }),
                steps.end());
    if (steps.empty())
      pipeline_it = ctx->owner->pipelines.erase(pipeline_it);
    else
      ++pipeline_it;
  }
}

#ifdef _WIN32
// The public Windows ICD ABI uses a 336-entry GL 1.1 dispatch table followed
// by the vendor's modern entry points returned by DrvGetProcAddress.
struct RawIcdGlcltProcTable {
  int cEntries;
  void *dispatch[336];
};
static_assert(offsetof(RawIcdGlcltProcTable, dispatch) == sizeof(void *),
              "Unexpected Windows ICD dispatch table layout");
struct RawIcdContextTag {};
using RawIcdSetProcTable = void(APIENTRY *)(const RawIcdGlcltProcTable *);
static EngineContext *g_raw_icd_context = nullptr;

static void APIENTRY raw_icd_set_proc_table(const RawIcdGlcltProcTable *) {}
#endif

static std::unordered_set<EngineContext *> engine_contexts;
static std::mutex engine_contexts_mutex;
static std::unordered_set<ModuleContext *> module_contexts;
static std::mutex module_contexts_mutex;
static uint64_t next_session_id = 1;
static std::mutex init_error_mutex;
static std::string last_init_error;

static void invalidate_pipelines_for_module(ModuleContext *ctx);

static void set_last_init_error(const std::string &message) {
  std::lock_guard<std::mutex> lock(init_error_mutex);
  last_init_error = message;
}

class EngineLease {
public:
  explicit EngineLease(void *runtime) {
    // The registry lock is held while taking the context lock and incrementing
    // active_calls.  Destruction uses the same order, so a pointer cannot pass
    // validation and then be deleted before this operation is pinned.
    std::unique_lock<std::mutex> registry_lock(engine_contexts_mutex);
    auto it = engine_contexts.find((EngineContext *)runtime);
    if (it == engine_contexts.end() || !*it)
      return;
    EngineContext *candidate = *it;
    std::lock_guard<std::mutex> context_lock(candidate->mutex);
    if (candidate->destroying)
      return;
    candidate->active_calls++;
    ctx_ = candidate;
  }

  EngineLease(const EngineLease &) = delete;
  EngineLease &operator=(const EngineLease &) = delete;

  ~EngineLease() {
    if (!ctx_)
      return;
    std::lock_guard<std::mutex> lock(ctx_->mutex);
    if (ctx_->active_calls > 0)
      ctx_->active_calls--;
    if (ctx_->active_calls == 0)
      ctx_->lifecycle_cv.notify_all();
  }

  EngineContext *get() const { return ctx_; }
  explicit operator bool() const { return ctx_ != nullptr; }

private:
  EngineContext *ctx_ = nullptr;
};

class ModuleLease {
public:
  explicit ModuleLease(void *module_ctx) {
    // As with EngineLease, validate and pin while holding the registry lock.
    // A concurrent destroy therefore either waits for this lease or removes
    // the pointer before this operation can dereference it.
    std::unique_lock<std::mutex> registry_lock(module_contexts_mutex);
    auto it = module_contexts.find((ModuleContext *)module_ctx);
    if (it == module_contexts.end() || !*it)
      return;
    ModuleContext *candidate = *it;
    std::lock_guard<std::mutex> lifetime_lock(candidate->lifetime_mutex);
    if (candidate->destroying)
      return;
    candidate->active_calls++;
    ctx_ = candidate;
  }

  ModuleLease(const ModuleLease &) = delete;
  ModuleLease &operator=(const ModuleLease &) = delete;

  ~ModuleLease() {
    if (!ctx_)
      return;
    std::lock_guard<std::mutex> lock(ctx_->lifetime_mutex);
    if (ctx_->active_calls > 0)
      ctx_->active_calls--;
    if (ctx_->active_calls == 0)
      ctx_->lifetime_cv.notify_all();
  }

  ModuleContext *get() const { return ctx_; }
  explicit operator bool() const { return ctx_ != nullptr; }

private:
  ModuleContext *ctx_ = nullptr;
};

static ModuleContext *begin_module_destroy(void *module_ctx) {
  std::unique_lock<std::mutex> registry_lock(module_contexts_mutex);
  auto it = module_contexts.find((ModuleContext *)module_ctx);
  if (it == module_contexts.end() || !*it)
    return nullptr;
  ModuleContext *ctx = *it;
  std::lock_guard<std::mutex> lifetime_lock(ctx->lifetime_mutex);
  if (ctx->destroying)
    return nullptr;
  ctx->destroying = true;
  module_contexts.erase(it);
  invalidate_pipelines_for_module(ctx);
  return ctx;
}

static void finish_module_destroy(ModuleContext *ctx) {
  if (!ctx)
    return;
  {
    std::unique_lock<std::mutex> lock(ctx->lifetime_mutex);
    ctx->lifetime_cv.wait(lock, [ctx] { return ctx->active_calls == 0; });
  }
  if (ctx->module)
    delete ctx->module;
  delete ctx;
}

static ti::Runtime *engine_runtime(EngineContext *ctx) {
  if (!ctx || ctx->destroying)
    return nullptr;
  return ctx->runtime;
}

static void set_engine_error(EngineContext *ctx, const std::string &message) {
  if (!ctx)
    return;
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->last_error = message;
}

static void clear_engine_error(EngineContext *ctx) {
  if (!ctx)
    return;
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->last_error.clear();
}

static bool validate_gpu_allocation_locked(
    EngineContext *ctx, TiMemory memory, uint64_t requested_size,
    const char *operation, bool reject_mapped = true) {
  if (!ctx || !memory) {
    if (ctx)
      ctx->last_error = std::string(operation) +
                        ": null GPU allocation handle";
    return false;
  }
  auto it = ctx->allocations.find(memory);
  if (it == ctx->allocations.end()) {
    ctx->last_error = std::string(operation) +
                      ": GPU allocation does not belong to this runtime";
    return false;
  }
  if (requested_size > it->second.size) {
    ctx->last_error = std::string(operation) +
                      ": transfer exceeds GPU allocation capacity (requested=" +
                      std::to_string(requested_size) +
                      ", capacity=" + std::to_string(it->second.size) + ")";
    return false;
  }
  if (reject_mapped && it->second.mapped) {
    ctx->last_error = std::string(operation) +
                      ": GPU allocation is already mapped";
    return false;
  }
  return true;
}

static bool checked_mul_u64(uint64_t left, uint64_t right, uint64_t *out) {
  if (!out || (right != 0 && left > std::numeric_limits<uint64_t>::max() / right))
    return false;
  *out = left * right;
  return true;
}

static size_t dynamic_arg_dtype_size(int dtype) {
  switch (dtype) {
  case 0:
    return sizeof(float);
  case 1:
    return sizeof(int32_t);
  case 2:
    return sizeof(uint8_t);
  case 3:
    return sizeof(uint16_t);
  case 4:
    return sizeof(int16_t);
  case 5:
    return sizeof(uint16_t); // f16 storage
  default:
    return 0;
  }
}

static bool validate_dynamic_arg_allocation(EngineContext *engine,
                                            const DynamicArg &dyn_arg,
                                            const char *operation) {
  if (!engine || dyn_arg.arg_type != 0)
    return true;
  uint64_t elements = 1;
  for (int d = 0; d < dyn_arg.dim_count; ++d) {
    if (!checked_mul_u64(elements, static_cast<uint64_t>(dyn_arg.shape[d]),
                         &elements))
      return false;
  }
  for (int d = 0; d < dyn_arg.elem_dim_count; ++d) {
    if (!checked_mul_u64(elements,
                         static_cast<uint64_t>(dyn_arg.elem_shape[d]),
                         &elements))
      return false;
  }
  const size_t element_size = dynamic_arg_dtype_size(dyn_arg.dtype);
  uint64_t required_bytes = 0;
  if (!element_size ||
      !checked_mul_u64(elements, static_cast<uint64_t>(element_size),
                       &required_bytes))
    return false;

  std::lock_guard<std::mutex> lock(engine->mutex);
  if (!validate_gpu_allocation_locked(engine, (TiMemory)dyn_arg.val_u64,
                                      required_bytes, operation))
    return false;
  return true;
}

static std::string consume_ti_last_error() {
  std::string last_error;
  for (int attempt = 0; attempt < 16; ++attempt) {
    uint64_t msg_size = 0;
    ti_get_last_error(&msg_size, nullptr);
    if (msg_size <= 1)
      break;
    std::vector<char> msg(msg_size);
    ti_get_last_error(&msg_size, msg.data());
    if (!msg.empty() && msg[0] != '\0')
      last_error = std::string(msg.data());
  }
  return last_error;
}

#ifdef _WIN32
static std::string current_opengl_renderer(EngineContext *ctx);

static std::string lower_ascii(std::string value) {
  for (char &ch : value)
    if (ch >= 'A' && ch <= 'Z')
      ch = static_cast<char>(ch - 'A' + 'a');
  return value;
}

static bool text_matches(const std::string &value, const std::string &needle) {
  return needle.empty() || lower_ascii(value).find(lower_ascii(needle)) !=
                              std::string::npos;
}

static const RawIcdGlcltProcTable *raw_icd_table(EngineContext *ctx) {
  return ctx && ctx->icd_table
             ? reinterpret_cast<const RawIcdGlcltProcTable *>(ctx->icd_table)
             : nullptr;
}

static void *resolve_raw_icd_proc(EngineContext *ctx, const char *name) {
  if (!ctx || !ctx->icd_mode || !name)
    return nullptr;
  if (const auto *table = raw_icd_table(ctx)) {
    for (std::size_t i = 0; i < kRawIcdGlProcEntryCount; ++i) {
      if (std::strcmp(table ? kRawIcdGlProcEntries[i].name : "", name) == 0) {
        const std::size_t index = kRawIcdGlProcEntries[i].index;
        if (table->cEntries >= 336 && index < 336)
          return table->dispatch[index];
      }
    }
  }
  if (ctx->icdGetProcAddress)
    return reinterpret_cast<void *>(ctx->icdGetProcAddress(name));
  return nullptr;
}

static void *APIENTRY raw_icd_get_proc_addr(const char *name) {
  return resolve_raw_icd_proc(g_raw_icd_context, name);
}

static void append_registry_icd_candidates(std::vector<std::string> &result) {
  // Some Windows drivers register an ICD DLL whose filename does not contain
  // "icd", "ogl", or "opengl". Consult both registry views so those drivers
  // remain discoverable without relying on opengl32's adapter selection.
  constexpr const char *kOpenGLDriversKey =
      "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\OpenGLDrivers";
  const REGSAM views[] = {KEY_WOW64_64KEY, KEY_WOW64_32KEY};
  auto append_value = [&](HKEY key, const char *value_name) {
    DWORD type = 0;
    DWORD byte_count = 0;
    if (RegQueryValueExA(key, value_name, nullptr, &type, nullptr,
                         &byte_count) != ERROR_SUCCESS ||
        (type != REG_SZ && type != REG_EXPAND_SZ) || byte_count == 0 ||
        byte_count > 32 * 1024)
      return;
    std::vector<char> storage(byte_count + 2, '\0');
    if (RegQueryValueExA(key, value_name, nullptr, &type,
                         reinterpret_cast<LPBYTE>(storage.data()),
                         &byte_count) != ERROR_SUCCESS)
      return;
    std::string value(storage.data());
    if (lower_ascii(value).find(".dll") == std::string::npos)
      return;
    std::vector<char> expanded(32768, '\0');
    DWORD expanded_length = ExpandEnvironmentStringsA(
        value.c_str(), expanded.data(), static_cast<DWORD>(expanded.size()));
    if (expanded_length > 0 && expanded_length < expanded.size())
      value.assign(expanded.data(), expanded_length - 1);
    result.emplace_back(value);
  };
  for (REGSAM view : views) {
    HKEY root = nullptr;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, kOpenGLDriversKey, 0,
                      KEY_READ | view, &root) != ERROR_SUCCESS)
      continue;
    DWORD value_count = 0;
    DWORD subkey_count = 0;
    DWORD max_value_name = 0;
    DWORD max_subkey_name = 0;
    RegQueryInfoKeyA(root, nullptr, nullptr, nullptr, &subkey_count,
                     &max_subkey_name, nullptr, &value_count,
                     &max_value_name, nullptr, nullptr, nullptr);
    std::vector<char> value_name(max_value_name + 2, '\0');
    for (DWORD i = 0; i < value_count; ++i) {
      DWORD name_length = static_cast<DWORD>(value_name.size() - 1);
      if (RegEnumValueA(root, i, value_name.data(), &name_length, nullptr,
                        nullptr, nullptr, nullptr) == ERROR_SUCCESS) {
        value_name[name_length] = '\0';
        append_value(root, value_name.data());
      }
    }
    std::vector<char> subkey_name(max_subkey_name + 2, '\0');
    for (DWORD i = 0; i < subkey_count; ++i) {
      DWORD name_length = static_cast<DWORD>(subkey_name.size() - 1);
      FILETIME last_write{};
      if (RegEnumKeyExA(root, i, subkey_name.data(), &name_length, nullptr,
                        nullptr, nullptr, &last_write) != ERROR_SUCCESS)
        continue;
      subkey_name[name_length] = '\0';
      HKEY child = nullptr;
      if (RegOpenKeyExA(root, subkey_name.data(), 0, KEY_READ, &child) !=
          ERROR_SUCCESS)
        continue;
      DWORD child_value_count = 0;
      DWORD child_max_value_name = 0;
      RegQueryInfoKeyA(child, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, &child_value_count, &child_max_value_name,
                       nullptr, nullptr, nullptr);
      std::vector<char> child_value_name(child_max_value_name + 2, '\0');
      for (DWORD j = 0; j < child_value_count; ++j) {
        DWORD child_name_length =
            static_cast<DWORD>(child_value_name.size() - 1);
        if (RegEnumValueA(child, j, child_value_name.data(),
                          &child_name_length, nullptr, nullptr, nullptr,
                          nullptr) == ERROR_SUCCESS) {
          child_value_name[child_name_length] = '\0';
          append_value(child, child_value_name.data());
        }
      }
      RegCloseKey(child);
    }
    RegCloseKey(root);
  }
}

static std::vector<std::string> raw_icd_library_candidates() {
  std::vector<std::string> result;
  const auto narrow_path = [](const std::wstring &value) {
    if (value.empty())
      return std::string();
    const int length = WideCharToMultiByte(CP_UTF8, 0, value.c_str(),
                                           static_cast<int>(value.size()),
                                           nullptr, 0, nullptr, nullptr);
    std::string output(static_cast<std::size_t>(length), '\0');
    if (length > 0)
      WideCharToMultiByte(CP_UTF8, 0, value.c_str(),
                          static_cast<int>(value.size()), output.data(), length,
                          nullptr, nullptr);
    return output;
  };
  const char *override_path = std::getenv("PIXEL_REFINE_OPENGL_ICD_LIBRARY");
  if (override_path && override_path[0] != '\0')
    result.emplace_back(override_path);
  const char *override_paths = std::getenv("PIXEL_REFINE_OPENGL_ICD_PATHS");
  if (override_paths && override_paths[0] != '\0') {
    std::stringstream list(override_paths);
    std::string path;
    while (std::getline(list, path, ';'))
      if (!path.empty())
        result.emplace_back(path);
  }
  std::vector<std::string> names = {"nvoglv64.dll", "ig11icd64.dll",
                                   "ig10icd64.dll", "ig9icd64.dll",
                                   "atio6axx.dll", "atio6axx64.dll"};
  const char *expected_vendor =
      std::getenv("PIXEL_REFINE_OPENGL_EXPECTED_VENDOR");
  const std::string vendor = lower_ascii(expected_vendor ? expected_vendor : "");
  if (vendor.find("nvidia") != std::string::npos)
    names = {"nvoglv64.dll"};
  else if (vendor.find("intel") != std::string::npos)
    names = {"ig11icd64.dll", "ig9icd64.dll"};
  wchar_t system_dir[MAX_PATH] = {};
  const UINT system_length = GetSystemDirectoryW(system_dir, MAX_PATH);
  if (system_length > 0 && system_length < MAX_PATH) {
    for (const auto &name : names) {
      const std::wstring repository = std::wstring(system_dir, system_length) +
                                      L"\\DriverStore\\FileRepository\\";
      const std::wstring pattern = repository + L"*";
      WIN32_FIND_DATAW data{};
      HANDLE handle = FindFirstFileW(pattern.c_str(), &data);
      if (handle != INVALID_HANDLE_VALUE) {
        do {
          if ((data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 &&
              std::wcscmp(data.cFileName, L".") != 0 &&
              std::wcscmp(data.cFileName, L"..") != 0) {
            const std::wstring path = repository + data.cFileName + L"\\" +
                                      std::wstring(name.begin(), name.end());
            if (GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES)
              result.emplace_back(narrow_path(path));
          }
        } while (FindNextFileW(handle, &data));
        FindClose(handle);
      }
      const std::wstring system_path =
          std::wstring(system_dir, system_length) + L"\\" +
          std::wstring(name.begin(), name.end());
      result.emplace_back(narrow_path(system_path));
    }
    // Older Intel packages use ig7/ig8/ig9 names and several third-party
    // ICDs use a vendor-specific suffix. Enumerate common OpenGL/ICD filename
    // families as a final vendor-neutral discovery pass. The export checks in
    // initialize_native_icd() reject ordinary DLLs (including Microsoft's
    // opengl32.dll) without loading them as a context provider.
    const std::wstring repository = std::wstring(system_dir, system_length) +
                                    L"\\DriverStore\\FileRepository\\";
    WIN32_FIND_DATAW directory_data{};
    HANDLE directory_handle =
        FindFirstFileW((repository + L"*").c_str(), &directory_data);
    if (directory_handle != INVALID_HANDLE_VALUE) {
      do {
        if ((directory_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0 ||
            std::wcscmp(directory_data.cFileName, L".") == 0 ||
            std::wcscmp(directory_data.cFileName, L"..") == 0)
          continue;
        const std::wstring directory = repository + directory_data.cFileName + L"\\";
        const std::vector<std::wstring> patterns = {
            L"*icd*.dll", L"*ogl*.dll", L"*opengl*.dll"};
        for (const auto &pattern : patterns) {
          WIN32_FIND_DATAW icd_data{};
          HANDLE icd_handle =
              FindFirstFileW((directory + pattern).c_str(), &icd_data);
          if (icd_handle == INVALID_HANDLE_VALUE)
            continue;
          do {
            if ((icd_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
              const std::wstring path = directory + icd_data.cFileName;
              result.emplace_back(narrow_path(path));
            }
          } while (FindNextFileW(icd_handle, &icd_data));
          FindClose(icd_handle);
        }
      } while (FindNextFileW(directory_handle, &directory_data));
      FindClose(directory_handle);
    }
  }
  append_registry_icd_candidates(result);
  return result;
}

static bool bind_raw_icd_context(EngineContext *ctx) {
  if (!ctx || !ctx->icd_mode || !ctx->icdSetContext || !ctx->icd_dc ||
      !ctx->icd_context)
    return false;
  auto set_context = reinterpret_cast<const RawIcdGlcltProcTable *(WINAPI *)(
      HDC, RawIcdContextTag *, RawIcdSetProcTable)>(ctx->icdSetContext);
  const auto *table = set_context(ctx->icd_dc, ctx->icd_context,
                                  raw_icd_set_proc_table);
  if (!table || table->cEntries < 336)
    return false;
  ctx->icd_table = table;
  g_raw_icd_context = ctx;
  return true;
}

static void release_native_icd(EngineContext *ctx) {
  if (!ctx || (!ctx->icd_mode && !ctx->icd_module && !ctx->icd_window))
    return;
  if (ctx->icdReleaseContext && ctx->icd_context)
    ctx->icdReleaseContext(ctx->icd_context);
  if (ctx->icdDeleteContext && ctx->icd_context)
    ctx->icdDeleteContext(ctx->icd_context);
  if (ctx->icd_window && ctx->icd_dc)
    ReleaseDC(ctx->icd_window, ctx->icd_dc);
  if (ctx->icd_window)
    DestroyWindow(ctx->icd_window);
  if (ctx->icd_module)
    FreeLibrary(ctx->icd_module);
  if (g_raw_icd_context == ctx)
    g_raw_icd_context = nullptr;
  ctx->icd_mode = false;
  ctx->icd_window = nullptr;
  ctx->icd_dc = nullptr;
  ctx->icd_module = nullptr;
  ctx->icd_library_path.clear();
  ctx->icd_context = nullptr;
  ctx->icd_table = nullptr;
}

static bool initialize_native_icd(EngineContext *ctx) {
  if (!ctx)
    return false;
  const std::string expected_name =
      std::getenv("PIXEL_REFINE_OPENGL_EXPECTED_NAME")
          ? std::getenv("PIXEL_REFINE_OPENGL_EXPECTED_NAME")
          : "";
  const std::string expected_vendor =
      std::getenv("PIXEL_REFINE_OPENGL_EXPECTED_VENDOR")
          ? std::getenv("PIXEL_REFINE_OPENGL_EXPECTED_VENDOR")
          : "";
  for (const auto &candidate : raw_icd_library_candidates()) {
    HMODULE module = LoadLibraryA(candidate.c_str());
    if (!module)
      continue;
    auto set_pixel_format = reinterpret_cast<BOOL(WINAPI *)(HDC, int)>(
        GetProcAddress(module, "DrvSetPixelFormat"));
    auto create_context = reinterpret_cast<RawIcdContextTag *(WINAPI *)(HDC)>(
        GetProcAddress(module, "DrvCreateContext"));
    auto set_context = reinterpret_cast<const void *(WINAPI *)(
        HDC, RawIcdContextTag *, void(APIENTRY *)(const void *))>(
        GetProcAddress(module, "DrvSetContext"));
    auto delete_context = reinterpret_cast<BOOL(WINAPI *)(RawIcdContextTag *)>(
        GetProcAddress(module, "DrvDeleteContext"));
    auto release_context = reinterpret_cast<void(WINAPI *)(RawIcdContextTag *)>(
        GetProcAddress(module, "DrvReleaseContext"));
    auto get_proc = reinterpret_cast<PROC(WINAPI *)(LPCSTR)>(
        GetProcAddress(module, "DrvGetProcAddress"));
    if (!set_pixel_format || !create_context || !set_context ||
        !delete_context || !get_proc) {
      FreeLibrary(module);
      continue;
    }
    HWND window = CreateWindowExA(0, "STATIC", "PixelRefineRawICD", 0, 0, 0,
                                  1, 1, nullptr, nullptr, GetModuleHandleA(nullptr),
                                  nullptr);
    HDC dc = window ? GetDC(window) : nullptr;
    PIXELFORMATDESCRIPTOR pfd{};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    pfd.cStencilBits = 8;
    pfd.iLayerType = PFD_MAIN_PLANE;
    const int format = dc ? ChoosePixelFormat(dc, &pfd) : 0;
    bool ok = dc && format > 0 && SetPixelFormat(dc, format, &pfd) == TRUE &&
              set_pixel_format(dc, format) == TRUE;
    RawIcdContextTag *raw_context = ok ? create_context(dc) : nullptr;
    const auto set_fn = reinterpret_cast<const RawIcdGlcltProcTable *(WINAPI *)(
        HDC, RawIcdContextTag *, RawIcdSetProcTable)>(set_context);
    const auto *table = raw_context ? set_fn(dc, raw_context, raw_icd_set_proc_table)
                                     : nullptr;
    if (table && table->cEntries >= 336) {
      ctx->icd_mode = true;
      ctx->icd_module = module;
      ctx->icd_library_path = candidate;
      ctx->icd_window = window;
      ctx->icd_dc = dc;
      ctx->icd_context = raw_context;
      ctx->icdSetPixelFormat = set_pixel_format;
      ctx->icdCreateContext = create_context;
      ctx->icdSetContext = set_context;
      ctx->icdDeleteContext = delete_context;
      ctx->icdReleaseContext = release_context;
      ctx->icdGetProcAddress = get_proc;
      ctx->icd_table = table;
      g_raw_icd_context = ctx;
      const std::string renderer = current_opengl_renderer(ctx);
      const std::string vendor = [&]() {
        using GetString = const GLubyte *(APIENTRY *)(GLenum);
        auto fn = reinterpret_cast<GetString>(resolve_raw_icd_proc(ctx, "glGetString"));
        const GLubyte *value = fn ? fn(GL_VENDOR) : nullptr;
        return value ? std::string(reinterpret_cast<const char *>(value)) : "";
      }();
      if (text_matches(renderer, expected_name) &&
          text_matches(vendor, expected_vendor)) {
        std::cout << "[AOTEngine ICD] Native OpenGL ICD context initialized from "
                  << candidate << " (" << renderer << ")" << std::endl;
        return true;
      }
      release_native_icd(ctx);
      continue;
    }
    if (raw_context)
      delete_context(raw_context);
    if (dc && window)
      ReleaseDC(window, dc);
    if (window)
      DestroyWindow(window);
    FreeLibrary(module);
  }
  set_engine_error(ctx, "OpenGL native ICD initialization failed: no matching vendor driver context was created");
  return false;
}
#endif

static std::string current_opengl_renderer(EngineContext *ctx = nullptr) {
#ifdef _WIN32
  if (ctx && ctx->icd_mode) {
    using GetString = const GLubyte *(APIENTRY *)(GLenum);
    auto fn = reinterpret_cast<GetString>(resolve_raw_icd_proc(ctx, "glGetString"));
    const GLubyte *renderer = fn ? fn(GL_RENDERER) : nullptr;
    const GLubyte *vendor = fn ? fn(GL_VENDOR) : nullptr;
    if (!renderer)
      return "";
    std::string name(reinterpret_cast<const char *>(renderer));
    if (vendor && name.find(reinterpret_cast<const char *>(vendor)) == std::string::npos)
      name = std::string(reinterpret_cast<const char *>(vendor)) + " - " + name;
    return name;
  }
  if (ctx && ctx->egl_mode && ctx->eglGetProcAddress) {
    using GetString = const GLubyte *(APIENTRY *)(GLenum);
    auto get_string = reinterpret_cast<GetString>(ctx->eglGetProcAddress("glGetString"));
    const GLubyte *renderer = get_string ? get_string(GL_RENDERER) : nullptr;
    const GLubyte *vendor = get_string ? get_string(GL_VENDOR) : nullptr;
    if (!renderer)
      return "";
    std::string name(reinterpret_cast<const char *>(renderer));
    if (vendor && name.find(reinterpret_cast<const char *>(vendor)) == std::string::npos)
      name = std::string(reinterpret_cast<const char *>(vendor)) + " - " + name;
    return name;
  }
  return "";
#else
  return "";
#endif
}

#ifdef _WIN32
template <typename Proc>
static Proc load_egl_proc(EngineContext *ctx, const char *name) {
  if (!ctx || !ctx->egl_module || !name)
    return nullptr;
  auto address = GetProcAddress(ctx->egl_module, name);
  if (!address && ctx->eglGetProcAddress)
    address = reinterpret_cast<decltype(address)>(
        ctx->eglGetProcAddress(name));
  return reinterpret_cast<Proc>(address);
}

static std::string egl_error_string(EngineContext *ctx,
                                    const char *operation) {
  std::ostringstream out;
  out << operation;
  if (ctx && ctx->eglGetError) {
    EGLint error = ctx->eglGetError();
    out << " (EGL error 0x" << std::hex << error << std::dec << ")";
  }
  return out.str();
}

static void release_native_egl(EngineContext *ctx) {
  if (!ctx || (!ctx->egl_mode && !ctx->egl_module &&
               ctx->egl_display == EGL_NO_DISPLAY &&
               ctx->egl_surface == EGL_NO_SURFACE &&
               ctx->egl_context == EGL_NO_CONTEXT))
    return;
  if (ctx->eglMakeCurrent && ctx->egl_display != EGL_NO_DISPLAY) {
    ctx->eglMakeCurrent(ctx->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                        EGL_NO_CONTEXT);
  }
  if (ctx->eglDestroyContext && ctx->egl_display != EGL_NO_DISPLAY &&
      ctx->egl_context != EGL_NO_CONTEXT) {
    ctx->eglDestroyContext(ctx->egl_display, ctx->egl_context);
  }
  if (ctx->eglDestroySurface && ctx->egl_display != EGL_NO_DISPLAY &&
      ctx->egl_surface != EGL_NO_SURFACE) {
    ctx->eglDestroySurface(ctx->egl_display, ctx->egl_surface);
  }
  if (ctx->eglTerminate && ctx->egl_display != EGL_NO_DISPLAY)
    ctx->eglTerminate(ctx->egl_display);
  if (ctx->egl_module)
    FreeLibrary(ctx->egl_module);
  ctx->egl_module = nullptr;
  ctx->egl_library_path.clear();
  ctx->egl_display = EGL_NO_DISPLAY;
  ctx->egl_surface = EGL_NO_SURFACE;
  ctx->egl_context = EGL_NO_CONTEXT;
  ctx->egl_mode = false;
}

static bool initialize_native_egl(EngineContext *ctx, uint32_t requested_device) {
  if (!ctx)
    return false;

  const char *override_path =
      std::getenv("PIXEL_REFINE_OPENGL_EGL_LIBRARY");
  std::vector<std::string> library_candidates;
  if (override_path && override_path[0] != '\0')
    library_candidates.emplace_back(override_path);

  // Prefer a self-contained provider shipped next to the renderer bridge.
  // This keeps native EGL selection independent of PATH and of host GUI
  // libraries.  Companion DLLs are resolved relative to this module by the
  // Windows loader.
  HMODULE bridge_module = nullptr;
  if (GetModuleHandleExA(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCSTR>(&initialize_native_egl), &bridge_module)) {
    char module_path[MAX_PATH] = {};
    DWORD length = GetModuleFileNameA(bridge_module, module_path,
                                      static_cast<DWORD>(sizeof(module_path)));
    if (length > 0 && length < sizeof(module_path)) {
      std::string directory(module_path, length);
      const size_t separator = directory.find_last_of("\\/");
      if (separator != std::string::npos)
        directory.resize(separator);
      library_candidates.emplace_back(directory + "\\egl\\libEGL.dll");
      library_candidates.emplace_back(directory + "\\libEGL.dll");
    }
  }
  library_candidates.emplace_back("libEGL.dll");
  library_candidates.emplace_back("EGL.dll");
  for (const auto &candidate : library_candidates) {
    ctx->egl_module = LoadLibraryA(candidate.c_str());
    if (ctx->egl_module) {
      ctx->egl_library_path = candidate;
      break;
    }
  }
  if (!ctx->egl_module) {
    set_engine_error(
        ctx,
        "OpenGL EGL initialization failed: no vendor libEGL.dll was found; "
        "native ICD/EGL selection is unavailable");
    return false;
  }
  if (is_debug_logging_enabled())
    std::cerr << "[AOTEngine EGL] Loaded provider " << ctx->egl_library_path
              << std::endl;

  ctx->eglGetProcAddress = load_egl_proc<PFNEGLGETPROCADDRESSPROC>(
      ctx, "eglGetProcAddress");
  ctx->eglGetError = load_egl_proc<PFNEGLGETERRORPROC>(ctx, "eglGetError");
  ctx->eglQueryString =
      load_egl_proc<PFNEGLQUERYSTRINGPROC>(ctx, "eglQueryString");
  ctx->eglGetDisplay = load_egl_proc<PFNEGLGETDISPLAYPROC>(ctx, "eglGetDisplay");
  ctx->eglInitialize = load_egl_proc<PFNEGLINITIALIZEPROC>(ctx, "eglInitialize");
  ctx->eglTerminate = load_egl_proc<PFNEGLTERMINATEPROC>(ctx, "eglTerminate");
  ctx->eglBindAPI = load_egl_proc<PFNEGLBINDAPIPROC>(ctx, "eglBindAPI");
  ctx->eglChooseConfig =
      load_egl_proc<PFNEGLCHOOSECONFIGPROC>(ctx, "eglChooseConfig");
  ctx->eglCreatePbufferSurface = load_egl_proc<PFNEGLCREATEPBUFFERSURFACEPROC>(
      ctx, "eglCreatePbufferSurface");
  ctx->eglDestroySurface =
      load_egl_proc<PFNEGLDESTROYSURFACEPROC>(ctx, "eglDestroySurface");
  ctx->eglCreateContext =
      load_egl_proc<PFNEGLCREATECONTEXTPROC>(ctx, "eglCreateContext");
  ctx->eglDestroyContext =
      load_egl_proc<PFNEGLDESTROYCONTEXTPROC>(ctx, "eglDestroyContext");
  ctx->eglMakeCurrent =
      load_egl_proc<PFNEGLMAKECURRENTPROC>(ctx, "eglMakeCurrent");
  ctx->eglGetCurrentDisplay = load_egl_proc<PFNEGLGETCURRENTDISPLAYPROC>(
      ctx, "eglGetCurrentDisplay");
  ctx->eglGetCurrentSurface = load_egl_proc<PFNEGLGETCURRENTSURFACEPROC>(
      ctx, "eglGetCurrentSurface");
  ctx->eglGetCurrentContext = load_egl_proc<PFNEGLGETCURRENTCONTEXTPROC>(
      ctx, "eglGetCurrentContext");
  ctx->eglGetPlatformDisplay =
      load_egl_proc<PFNEGLGETPLATFORMDISPLAYPROC>(ctx, "eglGetPlatformDisplay");
  ctx->eglGetPlatformDisplayEXT =
      load_egl_proc<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
          ctx, "eglGetPlatformDisplayEXT");
  ctx->eglQueryDevicesEXT =
      load_egl_proc<PFNEGLQUERYDEVICESEXTPROC>(ctx, "eglQueryDevicesEXT");
  ctx->eglQueryDeviceStringEXT = load_egl_proc<PFNEGLQUERYDEVICESTRINGEXTPROC>(
      ctx, "eglQueryDeviceStringEXT");

  if (!ctx->eglGetDisplay || !ctx->eglInitialize || !ctx->eglTerminate ||
      !ctx->eglBindAPI || !ctx->eglChooseConfig ||
      !ctx->eglCreatePbufferSurface || !ctx->eglDestroySurface ||
      !ctx->eglCreateContext || !ctx->eglDestroyContext ||
      !ctx->eglMakeCurrent) {
    set_engine_error(ctx, "OpenGL EGL initialization failed: required EGL entry points are missing");
    release_native_egl(ctx);
    return false;
  }

  // ANGLE's libEGL is intentionally not accepted as the native OpenGL
  // provider.  ANGLE translates OpenGL through D3D/Vulkan and would make the
  // backend appear device-independent while violating the native-driver
  // requirement.  It may be enabled explicitly for diagnostics only.
  if (ctx->eglQueryString) {
    const char *extensions = ctx->eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
    const char *allow_angle = std::getenv("PIXEL_REFINE_OPENGL_ALLOW_ANGLE");
    if (extensions && std::strstr(extensions, "EGL_ANGLE") &&
        (!allow_angle || std::string(allow_angle) != "1")) {
      set_engine_error(ctx,
                       "OpenGL EGL provider is ANGLE (translation), not a native vendor EGL implementation");
      release_native_egl(ctx);
      return false;
    }
  }

  const char *expected_name = std::getenv("PIXEL_REFINE_OPENGL_EXPECTED_NAME");
  const std::string expected = expected_name ? expected_name : "";
  EGLDeviceEXT selected_device = nullptr;
  EGLint device_count = 0;
  if (ctx->eglQueryDevicesEXT &&
      (ctx->eglGetPlatformDisplayEXT || ctx->eglGetPlatformDisplay)) {
    EGLDeviceEXT devices[32] = {};
    if (ctx->eglQueryDevicesEXT(32, devices, &device_count) == EGL_TRUE &&
        device_count > 0) {
      int selected_index = -1;
      for (EGLint i = 0; i < device_count; ++i) {
        const char *vendor = ctx->eglQueryDeviceStringEXT
                                 ? ctx->eglQueryDeviceStringEXT(devices[i], EGL_VENDOR)
                                 : nullptr;
        const char *renderer = ctx->eglQueryDeviceStringEXT
                                   ? ctx->eglQueryDeviceStringEXT(devices[i], EGL_RENDERER_EXT)
                                   : nullptr;
        std::string description;
        if (vendor)
          description += vendor;
        if (renderer) {
          if (!description.empty())
            description += " - ";
          description += renderer;
        }
        if (is_debug_logging_enabled())
          std::cerr << "[AOTEngine EGL] device[" << i << "] " << description << std::endl;
        if (!expected.empty() &&
            ((vendor && std::string(vendor).find(expected) != std::string::npos) ||
             (renderer && std::string(renderer).find(expected) != std::string::npos))) {
          selected_index = i;
          selected_device = devices[i];
          break;
        }
      }
      if (!selected_device && expected.empty()) {
        selected_index = static_cast<int>(requested_device < static_cast<uint32_t>(device_count)
                                              ? requested_device
                                              : 0);
        selected_device = devices[selected_index];
      }
      if (!selected_device && !expected.empty()) {
        set_engine_error(ctx, "OpenGL EGL device enumeration did not expose the requested renderer '" +
                                  expected + "'");
        release_native_egl(ctx);
        return false;
      }
      if (selected_device)
        if (ctx->eglGetPlatformDisplayEXT) {
          ctx->egl_display = ctx->eglGetPlatformDisplayEXT(
              EGL_PLATFORM_DEVICE_EXT, selected_device, nullptr);
        } else {
          ctx->egl_display = ctx->eglGetPlatformDisplay(
              EGL_PLATFORM_DEVICE_EXT, selected_device, nullptr);
        }
    }
  }
  if (ctx->egl_display == EGL_NO_DISPLAY) {
    if (!expected.empty() && device_count > 0) {
      set_engine_error(ctx, "OpenGL EGL could not create a display for the requested device '" +
                                expected + "'");
      release_native_egl(ctx);
      return false;
    }
    ctx->egl_display = ctx->eglGetDisplay(EGL_DEFAULT_DISPLAY);
  }
  if (ctx->egl_display == EGL_NO_DISPLAY) {
    set_engine_error(ctx, egl_error_string(ctx, "OpenGL EGL eglGetDisplay failed"));
    release_native_egl(ctx);
    return false;
  }

  EGLint major = 0, minor = 0;
  if (ctx->eglInitialize(ctx->egl_display, &major, &minor) != EGL_TRUE ||
      ctx->eglBindAPI(EGL_OPENGL_API) != EGL_TRUE) {
    set_engine_error(ctx, egl_error_string(ctx, "OpenGL EGL display initialization failed"));
    release_native_egl(ctx);
    return false;
  }
  const EGLint config_attrs[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                                 EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                                 EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8,
                                 EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE};
  EGLConfig config = nullptr;
  EGLint config_count = 0;
  if (ctx->eglChooseConfig(ctx->egl_display, config_attrs, &config, 1,
                           &config_count) != EGL_TRUE || config_count == 0) {
    set_engine_error(ctx, egl_error_string(ctx, "OpenGL EGL eglChooseConfig failed"));
    release_native_egl(ctx);
    return false;
  }
  const EGLint surface_attrs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
  ctx->egl_surface = ctx->eglCreatePbufferSurface(ctx->egl_display, config,
                                                   surface_attrs);
  if (ctx->egl_surface == EGL_NO_SURFACE) {
    set_engine_error(ctx, egl_error_string(ctx, "OpenGL EGL pbuffer creation failed"));
    release_native_egl(ctx);
    return false;
  }
  const EGLint context_attrs[] = {EGL_CONTEXT_MAJOR_VERSION_KHR, 4,
                                  EGL_CONTEXT_MINOR_VERSION_KHR, 3,
                                  EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR,
                                  EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR,
                                  EGL_NONE};
  ctx->egl_context = ctx->eglCreateContext(ctx->egl_display, config,
                                            EGL_NO_CONTEXT, context_attrs);
  if (ctx->egl_context == EGL_NO_CONTEXT) {
    const EGLint fallback_attrs[] = {EGL_NONE};
    ctx->egl_context = ctx->eglCreateContext(ctx->egl_display, config,
                                              EGL_NO_CONTEXT, fallback_attrs);
  }
  if (ctx->egl_context == EGL_NO_CONTEXT ||
      ctx->eglMakeCurrent(ctx->egl_display, ctx->egl_surface, ctx->egl_surface,
                          ctx->egl_context) != EGL_TRUE) {
    set_engine_error(ctx, egl_error_string(ctx, "OpenGL EGL context creation/bind failed"));
    release_native_egl(ctx);
    return false;
  }
  ctx->egl_mode = true;
  if (!expected.empty()) {
    const std::string selected_renderer = current_opengl_renderer(ctx);
    if (selected_renderer.find(expected) == std::string::npos) {
      set_engine_error(ctx,
                       "OpenGL EGL selected renderer '" + selected_renderer +
                           "', not the requested '" + expected + "'");
      release_native_egl(ctx);
      return false;
    }
  }
  std::cout << "[AOTEngine EGL] Native EGL context initialized (EGL " << major
            << "." << minor << ")" << std::endl;
  return true;
}
#endif

// Serializes an OpenGL runtime and migrates its native context to the calling
// thread for the duration of one bridge operation. Both supported providers
// (direct ICD and EGL) are window-system independent.
class ScopedOpenGLContext {
 public:
  explicit ScopedOpenGLContext(EngineContext *ctx) : ctx_(ctx) {
#ifdef _WIN32
    if (!ctx_ || ctx_->arch != TI_ARCH_OPENGL)
      return;
    lock_ = std::unique_lock<std::recursive_mutex>(ctx_->gl_context_mutex);
    if (ctx_->icd_mode) {
      icd_scope_ = true;
      ready_ = bind_raw_icd_context(ctx_);
      if (!ready_)
        set_engine_error(ctx_, "DrvSetContext failed while binding the native OpenGL ICD to the worker thread");
      return;
    }
    if (ctx_->egl_mode) {
      egl_scope_ = true;
      if (!ctx_->eglMakeCurrent || !ctx_->eglGetCurrentDisplay ||
          !ctx_->eglGetCurrentSurface || !ctx_->eglGetCurrentContext ||
          ctx_->egl_display == EGL_NO_DISPLAY ||
          ctx_->egl_context == EGL_NO_CONTEXT) {
        ready_ = false;
        set_engine_error(ctx_, "OpenGL EGL runtime has no valid native context");
        return;
      }
      previous_egl_display_ = ctx_->eglGetCurrentDisplay();
      previous_egl_draw_ = ctx_->eglGetCurrentSurface(EGL_DRAW);
      previous_egl_read_ = ctx_->eglGetCurrentSurface(EGL_READ);
      previous_egl_context_ = ctx_->eglGetCurrentContext();
      if (previous_egl_context_ == ctx_->egl_context &&
          previous_egl_display_ == ctx_->egl_display) {
        ready_ = true;
        return;
      }
      ready_ = ctx_->eglMakeCurrent(ctx_->egl_display, ctx_->egl_surface,
                                    ctx_->egl_surface, ctx_->egl_context) == EGL_TRUE;
      if (!ready_)
        set_engine_error(ctx_, egl_error_string(ctx_,
            "eglMakeCurrent failed while binding the OpenGL runtime to the worker thread"));
      return;
    }
    ready_ = false;
    set_engine_error(ctx_, "OpenGL runtime has no native ICD or EGL context");
#endif
  }

  ~ScopedOpenGLContext() {
#ifdef _WIN32
    if (!ctx_ || ctx_->arch != TI_ARCH_OPENGL || !ready_)
      return;
    if (egl_scope_) {
      if (previous_egl_context_ == ctx_->egl_context &&
          previous_egl_display_ == ctx_->egl_display)
        return;
      if (ctx_->eglMakeCurrent) {
        if (previous_egl_display_ == EGL_NO_DISPLAY ||
            previous_egl_context_ == EGL_NO_CONTEXT) {
          ctx_->eglMakeCurrent(EGL_NO_DISPLAY, EGL_NO_SURFACE, EGL_NO_SURFACE,
                               EGL_NO_CONTEXT);
        } else {
          ctx_->eglMakeCurrent(previous_egl_display_, previous_egl_draw_,
                               previous_egl_read_, previous_egl_context_);
        }
      }
      return;
    }
    if (icd_scope_) {
      // DrvSetContext is re-issued for every worker operation.  Do not call
      // DrvReleaseContext here: the runtime may issue deferred GL work after
      // the bridge scope returns, and the context is destroyed only at the
      // explicit engine teardown boundary.
      return;
    }
#endif
  }

  bool ready() const {
#ifdef _WIN32
    return !ctx_ || ctx_->arch != TI_ARCH_OPENGL || ready_;
#else
    return true;
#endif
  }

 private:
  EngineContext *ctx_ = nullptr;
#ifdef _WIN32
  std::unique_lock<std::recursive_mutex> lock_;
  EGLDisplay previous_egl_display_ = EGL_NO_DISPLAY;
  EGLSurface previous_egl_draw_ = EGL_NO_SURFACE;
  EGLSurface previous_egl_read_ = EGL_NO_SURFACE;
  EGLContext previous_egl_context_ = EGL_NO_CONTEXT;
  bool egl_scope_ = false;
  bool icd_scope_ = false;
  bool ready_ = true;
#endif
};

// -----------------------------------------------------------------------
// Runtime & Module Management
// -----------------------------------------------------------------------
extern "C" {

EXPORT const char *get_last_init_error() {
  static thread_local std::string snapshot;
  std::lock_guard<std::mutex> lock(init_error_mutex);
  snapshot = last_init_error;
  return snapshot.c_str();
}

EXPORT const char *scan_vulkan_devices() {
  static thread_local std::string device_list;
  device_list = "";
#ifdef _WIN32
  FILE *pipe = _popen("vulkaninfo --summary", "r");
#else
  FILE *pipe = popen("vulkaninfo --summary", "r");
#endif
  if (!pipe)
    return "";
  char buffer[256];
  while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
    std::string line(buffer);
    if (line.find("deviceName") != std::string::npos) {
      size_t pos = line.find("=");
      if (pos != std::string::npos) {
        std::string name = line.substr(pos + 1);
        // Trim whitespace and newlines
        name.erase(0, name.find_first_not_of(" \t"));
        name.erase(name.find_last_not_of(" \t\n\r") + 1);
        if (!device_list.empty())
          device_list += ";";
        device_list += name;
      }
    }
  }
#ifdef _WIN32
  _pclose(pipe);
#else
  pclose(pipe);
#endif
  return device_list.c_str();
}

EXPORT void *init_aot_engine(int arch_id, int device_id) {
  set_last_init_error("");
  TiArch arch = TI_ARCH_VULKAN;
  if (arch_id == 1)
    arch = TI_ARCH_CUDA;
  else if (arch_id == 2)
    arch = TI_ARCH_X64;
  else if (arch_id == 3)
    arch = TI_ARCH_OPENGL;
  else if (arch_id == 4)
    arch = TI_ARCH_GLES;
  try {
    // Use specified device_id
    auto ctx = std::make_unique<EngineContext>();
    ctx->arch = arch;
    if (arch == TI_ARCH_OPENGL) {
#ifdef _WIN32
      const char *egl_only = std::getenv("PIXEL_REFINE_OPENGL_EGL_ONLY");
      const char *context_mode = std::getenv("PIXEL_REFINE_OPENGL_CONTEXT");
      const char *icd_only = std::getenv("PIXEL_REFINE_OPENGL_ICD_ONLY");
      const bool strict_icd =
          (icd_only && std::string(icd_only) == "1") ||
          (context_mode && std::string(context_mode) == "icd");
      const bool strict_egl =
          (egl_only && std::string(egl_only) == "1") ||
          (context_mode && std::string(context_mode) == "egl");
      if (context_mode && std::string(context_mode) == "wgl") {
        set_last_init_error(
            "OpenGL legacy context mode has been removed; use the native vendor ICD or EGL provider");
        return nullptr;
      }
      if (strict_icd) {
        if (!initialize_native_icd(ctx.get())) {
          set_last_init_error(ctx->last_error);
          return nullptr;
        }
      }
      const bool try_icd_first = !strict_egl && !strict_icd;
      if (try_icd_first)
        initialize_native_icd(ctx.get());
      // Native ICD is preferred on Windows because it selects the vendor
      // driver directly. EGL remains the second native path for systems that
      // ship a vendor EGL provider. There is no legacy window-system fallback.
      if (!strict_icd && !ctx->icd_mode) {
        if (!initialize_native_egl(ctx.get(), static_cast<uint32_t>(device_id))) {
          std::cerr << "[AOTEngine EGL] " << ctx->last_error << std::endl;
          const std::string native_error =
              "OpenGL native ICD/EGL initialization failed; legacy context providers are not supported. ";
          set_last_init_error(native_error + ctx->last_error);
          return nullptr;
        }
      }
#endif
    }
    try {
      // Import the already-current EGL context through Taichi's supported C
      // interop API.  This is important: merely setting an environment flag
      // still lets Taichi create a second EGL display/context and can lose the
      // explicit physical-device selection.  The imported proc-address
      // callback makes the RHI operate on exactly the context created above.
#ifdef _WIN32
      if (arch == TI_ARCH_OPENGL && (ctx->egl_mode || ctx->icd_mode)) {
        TiOpenglRuntimeInteropInfo interop{};
        interop.get_proc_addr = ctx->icd_mode
                                    ? reinterpret_cast<void *>(raw_icd_get_proc_addr)
                                    : reinterpret_cast<void *>(ctx->eglGetProcAddress);
        TiRuntime imported = ti_import_opengl_runtime(&interop, false);
        if (!imported)
          throw std::runtime_error("ti_import_opengl_runtime returned a null runtime");
        try {
          ctx->runtime = new ti::Runtime(arch, imported, true);
        } catch (...) {
          ti_destroy_runtime(imported);
          throw;
        }
      } else {
        ctx->runtime = new ti::Runtime(arch, (uint32_t)device_id);
      }
#else
      ctx->runtime = new ti::Runtime(arch, (uint32_t)device_id);
#endif
    } catch (...) {
#ifdef _WIN32
      release_native_egl(ctx.get());
      release_native_icd(ctx.get());
#endif
      throw;
    }
    if (arch == TI_ARCH_OPENGL) {
#ifdef _WIN32
      if (ctx->icd_mode) {
        ctx->device_name = current_opengl_renderer(ctx.get());
      } else if (ctx->egl_mode) {
        ctx->device_name = current_opengl_renderer(ctx.get());
        // Do not leave the context owned by the initialization/UI thread.
        // ScopedOpenGLContext will bind it around every native operation.
        ctx->eglMakeCurrent(EGL_NO_DISPLAY, EGL_NO_SURFACE, EGL_NO_SURFACE,
                            EGL_NO_CONTEXT);
      }
#else
      ctx->device_name = current_opengl_renderer();
#endif
    }
    ctx->destroying = false;
    ctx->session_id = next_session_id++;
    {
      std::lock_guard<std::mutex> lock(engine_contexts_mutex);
      engine_contexts.insert(ctx.get());
    }
    return (void *)ctx.release();
  } catch (...) {
    // Never disguise a failed GPU initialization as the requested backend:
    // doing so makes CPU/Vulkan parity tests compare CPU against itself. Keep
    // the historical fallback available only as an explicit opt-in for legacy
    // callers that deliberately accept it.
    const char *allow_fallback =
        std::getenv("PIXEL_REFINE_AOT_ALLOW_CPU_FALLBACK");
    if (arch == TI_ARCH_X64 || !allow_fallback ||
        std::string(allow_fallback) != "1")
      return nullptr;
    try {
      auto ctx = std::make_unique<EngineContext>();
      // The compatibility path is a negotiated CPU runtime, never the
      // originally requested GPU identity.  Populate every identity field so
      // diagnostics and the Python-side arch probe cannot masquerade it as
      // the failed backend.
      ctx->arch = TI_ARCH_X64;
      ctx->runtime = new ti::Runtime(TI_ARCH_X64, 0);
      ctx->device_name = "CPU (legacy fallback)";
      ctx->last_error = "requested backend failed; explicit CPU fallback was enabled";
      ctx->destroying = false;
      ctx->session_id = next_session_id++;
      {
        std::lock_guard<std::mutex> lock(engine_contexts_mutex);
        engine_contexts.insert(ctx.get());
      }
      return (void *)ctx.release();
    } catch (...) {
      return nullptr;
    }
  }
}

EXPORT const char *get_runtime_device_name(void *runtime) {
  EngineLease lease(runtime);
  EngineContext *ctx = lease.get();
  static thread_local std::string snapshot;
  if (!ctx)
    return snapshot.c_str();
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    snapshot = ctx->device_name;
  }
  return snapshot.c_str();
}

EXPORT const char *get_runtime_context_backend(void *runtime) {
  EngineLease lease(runtime);
  EngineContext *ctx = lease.get();
  static thread_local std::string backend;
  // GLES is a separate Taichi architecture (and has its own target-qualified
  // bridge/artifacts), but it still belongs to the native graphics family for
  // diagnostics.  Returning an empty string here made mobile diagnostics look
  // like an uninitialised runtime even when the GLES bridge was loaded.
  if (!ctx) {
    backend.clear();
    return backend.c_str();
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  if (ctx->arch != TI_ARCH_OPENGL && ctx->arch != TI_ARCH_GLES) {
    backend.clear();
    return backend.c_str();
  }
#ifdef _WIN32
  if (ctx->arch == TI_ARCH_GLES) {
    backend = "GLES-native";
    return backend.c_str();
  }
  // Legacy window-system contexts are intentionally not supported in the
  // native bridge. Keep the diagnostic value explicit when initialization did
  // not establish an ICD/EGL provider instead of reporting a backend that is
  // never actually used.
  backend = ctx->icd_mode ? "ICD"
                          : (ctx->egl_mode ? "EGL" : "native-unavailable");
#else
  backend = ctx->arch == TI_ARCH_GLES ? "GLES-native" : "native";
#endif
  return backend.c_str();
}

EXPORT int get_runtime_arch_id(void *runtime) {
  EngineLease lease(runtime);
  EngineContext *ctx = lease.get();
  if (!ctx)
    return -1;
  std::lock_guard<std::mutex> lock(ctx->mutex);
  if (ctx->arch == TI_ARCH_CUDA)
    return 1;
  if (ctx->arch == TI_ARCH_X64)
    return 2;
  if (ctx->arch == TI_ARCH_OPENGL)
    return 3;
  if (ctx->arch == TI_ARCH_GLES)
    return 4;
  if (ctx->arch == TI_ARCH_VULKAN)
    return 0;
  return -1;
}

EXPORT const char *get_last_engine_error(void *runtime) {
  EngineLease lease(runtime);
  EngineContext *ctx = lease.get();
  static thread_local std::string snapshot;
  if (!ctx)
    return snapshot.c_str();
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    snapshot = ctx->last_error;
  }
  return snapshot.c_str();
}

EXPORT void clear_last_engine_error(void *runtime) {
  EngineLease lease(runtime);
  clear_engine_error(lease.get());
  consume_ti_last_error();
}

EXPORT void *load_aot_module(void *runtime, const char *tcm_path) {
  EngineLease lease(runtime);
  EngineContext *engine = lease.get();
  ScopedOpenGLContext gl_scope(engine);
  if (!gl_scope.ready())
    return nullptr;
  ti::Runtime *rt = engine_runtime(engine);
  if (is_debug_logging_enabled()) {
    FILE *log = fopen("engine_debug.log", "a");
    if (log) { fprintf(log, "[C++ Engine] ENTER load_aot_module path=%s runtime=%p\\n", tcm_path ? tcm_path : "<null>", (void *)rt); fclose(log); }
  }
  if (!rt || !tcm_path || tcm_path[0] == '\0') {
    if (engine)
      set_engine_error(engine, "load_aot_module: empty module path");
    return nullptr;
  }
  try {
    clear_engine_error(engine);
    consume_ti_last_error();
    ModuleContext *ctx = new ModuleContext();
    ctx->owner = engine;

    // ti_load_aot_module() accepts the legacy directory representation.
    // Pixel Refine distributes packed .tcm artifacts, which must instead be
    // deserialized with ti_create_aot_module(). Passing a .tcm path to the
    // directory loader can surface as an access violation rather than a
    // normal Taichi error, so select the API before entering Taichi.
    const std::string module_path(tcm_path);
    if (module_path.ends_with(".tcm")) {
      std::ifstream input(module_path, std::ios::binary | std::ios::ate);
      if (!input) {
        delete ctx;
        set_engine_error(engine,
                         std::string("load_aot_module: cannot open ") +
                             module_path);
        return nullptr;
      }

      const std::streamsize size = input.tellg();
      if (size <= 0) {
        delete ctx;
        set_engine_error(engine,
                         std::string("load_aot_module: empty .tcm artifact: ") +
                             module_path);
        return nullptr;
      }
      input.seekg(0, std::ios::beg);
      std::vector<uint8_t> tcm(static_cast<size_t>(size));
      if (!input.read(reinterpret_cast<char *>(tcm.data()), size)) {
        delete ctx;
        set_engine_error(engine,
                         std::string("load_aot_module: failed to read ") +
                             module_path);
        return nullptr;
      }
      if (is_debug_logging_enabled()) {
        FILE *log = fopen("engine_debug.log", "a");
        if (log) { fprintf(log, "[C++ Engine] CREATE_AOT bytes=%lld\\n", (long long)size); fclose(log); }
      }
      // Intel's native Vulkan driver is sensitive to outstanding async
      // submissions while the AOT module creates its pipelines. Drain the
      // runtime queue before deserialization to avoid fence/semaphore reuse.
      if (engine->arch == TI_ARCH_VULKAN) {
        rt->flush();
        rt->wait();
      }
      ctx->module = new ti::AotModule(rt->create_aot_module(tcm));
    } else {
      ctx->module = new ti::AotModule(rt->load_aot_module(module_path));
    }
    std::string load_error = consume_ti_last_error();
    if (!ctx->module || !ctx->module->is_valid() || !load_error.empty()) {
      if (ctx->module)
        delete ctx->module;
      delete ctx;
      set_engine_error(engine, std::string("load_aot_module: ") +
                                   (load_error.empty()
                                        ? "Taichi returned an invalid module"
                                        : load_error));
      return nullptr;
    }
    {
      std::lock_guard<std::mutex> lock(module_contexts_mutex);
      module_contexts.insert(ctx);
    }
    {
      std::lock_guard<std::mutex> lock(engine->mutex);
      engine->modules.insert(ctx);
    }
    return (void *)ctx;
  } catch (const std::exception &e) {
    set_engine_error(engine, std::string("load_aot_module: ") + e.what());
    return nullptr;
  } catch (...) {
    set_engine_error(engine, "load_aot_module: unknown exception");
    return nullptr;
  }
}

EXPORT void destroy_aot_module(void *module_ctx) {
  ModuleContext *ctx = begin_module_destroy(module_ctx);
  if (!ctx)
    return;
  EngineContext *owner = ctx->owner;
  EngineLease owner_lease(owner);
  if (owner_lease) {
    std::lock_guard<std::mutex> lock(owner_lease.get()->mutex);
    owner_lease.get()->modules.erase(ctx);
  }
  if (owner_lease) {
    ScopedOpenGLContext gl_scope(owner_lease.get());
    // Module admission is closed and active module leases are drained by the
    // finalizer even if an optional graphics rebind is unavailable.
    finish_module_destroy(ctx);
    return;
  }
  finish_module_destroy(ctx);
}

EXPORT void destroy_aot_engine(void *runtime) {
  EngineContext *ctx = nullptr;
  {
    // Close admission before touching any context-owned graphics/runtime
    // state.  EngineLease takes these locks in the same order, making the
    // registry check and lifetime pin atomic with respect to destruction.
    std::unique_lock<std::mutex> registry_lock(engine_contexts_mutex);
    auto it = engine_contexts.find((EngineContext *)runtime);
    if (it == engine_contexts.end() || !*it)
      return;
    ctx = *it;
    std::unique_lock<std::mutex> context_lock(ctx->mutex);
    ctx->destroying = true;
    engine_contexts.erase(it);
  }

  {
    std::unique_lock<std::mutex> lock(ctx->mutex);
    ctx->lifecycle_cv.wait(lock, [ctx] { return ctx->active_calls == 0; });
  }

  ScopedOpenGLContext gl_scope(ctx);
  // The context is already closed to new callers and all leases have drained.
  // Continue cleanup even if an optional graphics-context rebind is
  // unavailable; returning here would leak the runtime after removing it
  // from the live registry.

  try {
    if (ctx->runtime)
      ctx->runtime->wait();
  } catch (...) {
  }

  std::vector<ModuleContext *> modules;
  std::vector<TiMemory> allocations;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    for (auto *mod : ctx->modules)
      modules.push_back(mod);
    for (const auto &entry : ctx->allocations)
      allocations.push_back(entry.first);
    ctx->modules.clear();
    ctx->allocations.clear();
    ctx->pipelines.clear();
  }

  for (auto *mod : modules) {
    try {
      ModuleContext *retired = begin_module_destroy(mod);
      finish_module_destroy(retired);
    } catch (...) {
    }
  }

  for (auto mem : allocations) {
    try {
      if (ctx->runtime && mem)
        ti_free_memory(ctx->runtime->runtime(), mem);
    } catch (...) {
    }
  }

  try {
    if (ctx->runtime)
      delete ctx->runtime;
  } catch (...) {
  }
  ctx->runtime = nullptr;
#ifdef _WIN32
  release_native_egl(ctx);
  release_native_icd(ctx);
#endif
  delete ctx;
}

// -----------------------------------------------------------------------
// Memory Management
// -----------------------------------------------------------------------
EXPORT void *allocate_gpu_buffer(void *runtime, uint64_t size,
                                 int host_accessible) {
  EngineLease lease(runtime);
  EngineContext *engine = lease.get();
  clear_engine_error(engine);
  ScopedOpenGLContext gl_scope(engine);
  if (!gl_scope.ready())
    return nullptr;
  ti::Runtime *rt = engine_runtime(engine);
  if (!rt)
    return nullptr;
  TiMemoryAllocateInfo allocate_info = {};
  allocate_info.size = size;
  allocate_info.usage = TI_MEMORY_USAGE_STORAGE_BIT;
  if (host_accessible) {
    allocate_info.host_write = true;
    // Upload buffers only need host-write visibility. Requesting both
    // directions can make Vulkan reject the allocation on discrete/Dozen
    // drivers even when a host-visible storage heap is available.
    allocate_info.host_read = false;
  }
  TiMemory mem = ti_allocate_memory(rt->runtime(), &allocate_info);
  if (mem) {
    std::lock_guard<std::mutex> lock(engine->mutex);
    engine->allocations.emplace(mem, GpuAllocationRecord{size, false});
  }
  return (void *)mem;
}

EXPORT void free_gpu_buffer(void *runtime, void *memory) {
  EngineLease lease(runtime);
  EngineContext *engine = lease.get();
  clear_engine_error(engine);
  ScopedOpenGLContext gl_scope(engine);
  if (!gl_scope.ready())
    return;
  ti::Runtime *rt = engine_runtime(engine);
  if (rt && memory) {
    {
      std::lock_guard<std::mutex> lock(engine->mutex);
      if (!validate_gpu_allocation_locked(engine, (TiMemory)memory, 0,
                                          "free_gpu_buffer"))
        return;
      // Reject freeing a mapped allocation.  The caller must make the
      // map/unmap transition explicit so a later owner cannot observe a
      // stale mapped state or an already-freed native handle.
      engine->allocations.erase((TiMemory)memory);
    }
    ti_free_memory(rt->runtime(), (TiMemory)memory);
  }
}

EXPORT void write_to_gpu_buffer(void *runtime, void *memory, void *data,
                                uint64_t size) {
  EngineLease lease(runtime);
  EngineContext *engine = lease.get();
  clear_engine_error(engine);
  ScopedOpenGLContext gl_scope(engine);
  if (!gl_scope.ready())
    return;
  ti::Runtime *rt = engine_runtime(engine);
  if (!rt || !memory || !data)
    return;
  std::lock_guard<std::mutex> lock(engine->mutex);
  if (!validate_gpu_allocation_locked(engine, (TiMemory)memory, size,
                                      "write_to_gpu_buffer"))
    return;
  void *ptr = ti_map_memory(rt->runtime(), (TiMemory)memory);
  if (ptr) {
    memcpy(ptr, data, size);
    ti_unmap_memory(rt->runtime(), (TiMemory)memory);
  }
}

EXPORT void read_from_gpu_buffer(void *runtime, void *memory, void *data,
                                 uint64_t size) {
  EngineLease lease(runtime);
  EngineContext *engine = lease.get();
  clear_engine_error(engine);
  ScopedOpenGLContext gl_scope(engine);
  if (!gl_scope.ready())
    return;
  ti::Runtime *rt = engine_runtime(engine);
  if (!rt || !memory || !data)
    return;
  std::lock_guard<std::mutex> lock(engine->mutex);
  if (!validate_gpu_allocation_locked(engine, (TiMemory)memory, size,
                                      "read_from_gpu_buffer"))
    return;
  rt->wait(); // Ensure all kernels are done before reading
  void *ptr = ti_map_memory(rt->runtime(), (TiMemory)memory);
  if (ptr) {
    memcpy(data, ptr, size);
    ti_unmap_memory(rt->runtime(), (TiMemory)memory);
  }
}

EXPORT void *map_gpu_buffer(void *runtime, void *memory) {
  EngineLease lease(runtime);
  EngineContext *engine = lease.get();
  clear_engine_error(engine);
  ScopedOpenGLContext gl_scope(engine);
  if (!gl_scope.ready())
    return nullptr;
  ti::Runtime *rt = engine_runtime(engine);
  if (!rt || !memory)
    return nullptr;
  std::lock_guard<std::mutex> lock(engine->mutex);
  if (!validate_gpu_allocation_locked(engine, (TiMemory)memory, 0,
                                      "map_gpu_buffer"))
    return nullptr;
  void *ptr = ti_map_memory(rt->runtime(), (TiMemory)memory);
  if (ptr) {
    auto it = engine->allocations.find((TiMemory)memory);
    if (it != engine->allocations.end())
      it->second.mapped = true;
  }
  return ptr;
}

EXPORT void unmap_gpu_buffer(void *runtime, void *memory) {
  EngineLease lease(runtime);
  EngineContext *engine = lease.get();
  clear_engine_error(engine);
  ScopedOpenGLContext gl_scope(engine);
  if (!gl_scope.ready())
    return;
  ti::Runtime *rt = engine_runtime(engine);
  if (rt && memory) {
    std::lock_guard<std::mutex> lock(engine->mutex);
    if (!validate_gpu_allocation_locked(engine, (TiMemory)memory, 0,
                                        "unmap_gpu_buffer", false))
      return;
    auto it = engine->allocations.find((TiMemory)memory);
    if (!it->second.mapped) {
      engine->last_error =
          "unmap_gpu_buffer: GPU allocation is not currently mapped";
      return;
    }
    ti_unmap_memory(rt->runtime(), (TiMemory)memory);
    it->second.mapped = false;
  }
}

EXPORT void copy_gpu_buffer(void *runtime, void *src, void *dst,
                            uint64_t size) {
  EngineLease lease(runtime);
  EngineContext *engine = lease.get();
  clear_engine_error(engine);
  ScopedOpenGLContext gl_scope(engine);
  if (!gl_scope.ready())
    return;
  ti::Runtime *rt = engine_runtime(engine);
  if (!rt || !src || !dst)
    return;

  std::lock_guard<std::mutex> lock(engine->mutex);
  if (!validate_gpu_allocation_locked(engine, (TiMemory)src, size,
                                      "copy_gpu_buffer") ||
      !validate_gpu_allocation_locked(engine, (TiMemory)dst, size,
                                      "copy_gpu_buffer"))
    return;

  TiMemorySlice src_slice = {};
  src_slice.memory = (TiMemory)src;
  src_slice.offset = 0;
  src_slice.size = size;

  TiMemorySlice dst_slice = {};
  dst_slice.memory = (TiMemory)dst;
  dst_slice.offset = 0;
  dst_slice.size = size;

  ti_copy_memory_device_to_device(rt->runtime(), &dst_slice, &src_slice);
  rt->wait(); // Synchronize to prevent race conditions
}

EXPORT void sync_runtime(void *runtime) {
  EngineLease lease(runtime);
  EngineContext *engine = lease.get();
  ScopedOpenGLContext gl_scope(engine);
  if (!gl_scope.ready())
    return;
  ti::Runtime *rt = engine_runtime(engine);
  if (rt)
    rt->wait();
}

// -----------------------------------------------------------------------
// High-Performance Image IO (Smart Loader)
// -----------------------------------------------------------------------
static bool checked_image_geometry(int width, int height, int channels,
                                   int bit_depth, uint64_t *stride_out,
                                   uint64_t *size_out) {
  if (width <= 0 || height <= 0 || (channels != 1 && channels != 3) ||
      (bit_depth != 8 && bit_depth != 16) || !stride_out || !size_out)
    return false;
  const uint64_t max_value = std::numeric_limits<uint64_t>::max();
  const uint64_t bytes_per_channel = static_cast<uint64_t>(bit_depth / 8);
  uint64_t stride = static_cast<uint64_t>(width);
  if (stride > max_value / static_cast<uint64_t>(channels))
    return false;
  stride *= static_cast<uint64_t>(channels);
  if (stride > max_value / bytes_per_channel)
    return false;
  stride *= bytes_per_channel;
  if (stride > max_value / static_cast<uint64_t>(height))
    return false;
  *stride_out = stride;
  *size_out = stride * static_cast<uint64_t>(height);
  return true;
}

static bool image_io_test_failure(const char *stage) {
#ifdef _WIN32
  char configured[128] = {};
  const DWORD length = GetEnvironmentVariableA(
      "PIXEL_REFINE_AOT_TEST_FAIL_IMAGE_IO", configured,
      static_cast<DWORD>(sizeof(configured)));
  return length > 0 && length < sizeof(configured) && stage &&
         std::strcmp(configured, stage) == 0;
#else
  const char *configured = std::getenv("PIXEL_REFINE_AOT_TEST_FAIL_IMAGE_IO");
  return configured && stage && std::strcmp(configured, stage) == 0;
#endif
}

EXPORT void *ti_imread_to_gpu(void *runtime, const char *path, int *out_width,
                              int *out_height, int *out_channels,
                              int *out_bit_depth) {
  EngineLease lease(runtime);
  EngineContext *engine = lease.get();
  clear_engine_error(engine);
  ScopedOpenGLContext gl_scope(engine);
  if (!gl_scope.ready())
    return nullptr;
  ti::Runtime *rt = engine_runtime(engine);
  if (!rt || !path || !out_width || !out_height || !out_channels ||
      !out_bit_depth)
    return nullptr;
  *out_width = 0;
  *out_height = 0;
  *out_channels = 0;
  *out_bit_depth = 0;

#ifdef _WIN32
  init_wic();
  if (!g_wic_factory)
    return nullptr;

  IWICBitmapDecoder *decoder = nullptr;
  wchar_t w_path[MAX_PATH];
  if (!MultiByteToWideChar(CP_UTF8, 0, path, -1, w_path, MAX_PATH))
    return nullptr;

  if (FAILED(g_wic_factory->CreateDecoderFromFilename(
          w_path, NULL, GENERIC_READ, WICDecodeMetadataCacheOnDemand,
          &decoder))) {
    return nullptr;
  }

  IWICBitmapFrameDecode *frame = nullptr;
  if (FAILED(decoder->GetFrame(0, &frame)) || !frame) {
    decoder->Release();
    return nullptr;
  }

  UINT w = 0, h = 0;
  if (FAILED(frame->GetSize(&w, &h)) || w == 0 || h == 0 ||
      w > static_cast<UINT>(std::numeric_limits<int>::max()) ||
      h > static_cast<UINT>(std::numeric_limits<int>::max())) {
    frame->Release();
    decoder->Release();
    return nullptr;
  }
  *out_width = (int)w;
  *out_height = (int)h;

  WICPixelFormatGUID pixel_format;
  if (FAILED(frame->GetPixelFormat(&pixel_format))) {
    frame->Release();
    decoder->Release();
    return nullptr;
  }

  // Determine channels and bit depth
  int channels = 1;
  int bit_depth = 8;
  WICPixelFormatGUID target_format = GUID_WICPixelFormat8bppGray;

  if (pixel_format == GUID_WICPixelFormat8bppGray) {
    channels = 1;
    bit_depth = 8;
    target_format = GUID_WICPixelFormat8bppGray;
  } else if (pixel_format == GUID_WICPixelFormat16bppGray) {
    channels = 1;
    bit_depth = 16;
    target_format = GUID_WICPixelFormat16bppGray;
  } else if (pixel_format == GUID_WICPixelFormat24bppBGR ||
             pixel_format == GUID_WICPixelFormat32bppBGRA) {
    channels = 3;
    bit_depth = 8;
    target_format = GUID_WICPixelFormat24bppBGR;
  } else if (pixel_format == GUID_WICPixelFormat48bppBGR ||
             pixel_format == GUID_WICPixelFormat64bppBGRA) {
    channels = 3;
    bit_depth = 16;
    target_format = GUID_WICPixelFormat48bppBGR;
  } else {
    // WIC conversion is allowed for formats with a supported target, but all
    // conversion results below are checked.  Never expose an uninitialized
    // allocation when a codec/format operation fails.
    channels = 3;
    bit_depth = 8;
    target_format = GUID_WICPixelFormat24bppBGR;
  }

  *out_channels = channels;
  *out_bit_depth = bit_depth;

  // Allocate GPU memory
  uint64_t row_bytes = 0;
  uint64_t size_bytes = 0;
  if (!checked_image_geometry((int)w, (int)h, channels, bit_depth, &row_bytes,
                              &size_bytes)) {
    frame->Release();
    decoder->Release();
    return nullptr;
  }
  TiMemoryAllocateInfo allocate_info = {};
  allocate_info.size = size_bytes;
  allocate_info.usage = TI_MEMORY_USAGE_STORAGE_BIT;
  allocate_info.host_write = true; // Required for CopyPixels map path

  TiMemory gpu_mem = ti_allocate_memory(rt->runtime(), &allocate_info);
  if (!gpu_mem) {
    frame->Release();
    decoder->Release();
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(engine->mutex);
    engine->allocations.emplace(gpu_mem,
                                GpuAllocationRecord{size_bytes, false});
  }

  // Copy pixels directly to GPU (using mapped memory if possible, or
  // intermediate buffer)
  std::unique_lock<std::mutex> allocation_lock(engine->mutex);
  auto allocation_it = engine->allocations.find(gpu_mem);
  if (allocation_it == engine->allocations.end()) {
    engine->last_error =
        "ti_imread_to_gpu: allocation was not registered with this runtime";
    allocation_lock.unlock();
    ti_free_memory(rt->runtime(), gpu_mem);
    frame->Release();
    decoder->Release();
    return nullptr;
  }
  void *gpu_ptr = ti_map_memory(rt->runtime(), gpu_mem);
  if (!gpu_ptr) {
    engine->allocations.erase(allocation_it);
    allocation_lock.unlock();
    ti_free_memory(rt->runtime(), gpu_mem);
    frame->Release();
    decoder->Release();
    return nullptr;
  }
  allocation_it->second.mapped = true;

  bool copy_ok = false;
  if (pixel_format != target_format) {
      // Need conversion
      IWICFormatConverter *converter = nullptr;
      if (SUCCEEDED(g_wic_factory->CreateFormatConverter(&converter)) &&
          converter &&
          SUCCEEDED(converter->Initialize(
              frame, target_format, WICBitmapDitherTypeNone, NULL, 0.0,
              WICBitmapPaletteTypeCustom)) &&
          size_bytes <= std::numeric_limits<UINT>::max()) {
        copy_ok = SUCCEEDED(converter->CopyPixels(
            NULL, static_cast<UINT>(row_bytes), static_cast<UINT>(size_bytes),
            (BYTE *)gpu_ptr));
      }
      if (converter)
        converter->Release();
  } else if (size_bytes <= std::numeric_limits<UINT>::max()) {
      copy_ok = SUCCEEDED(frame->CopyPixels(
          NULL, static_cast<UINT>(row_bytes), static_cast<UINT>(size_bytes),
          (BYTE *)gpu_ptr));
  }
  ti_unmap_memory(rt->runtime(), gpu_mem);
  allocation_it->second.mapped = false;

  if (!copy_ok) {
    engine->allocations.erase(allocation_it);
    allocation_lock.unlock();
    ti_free_memory(rt->runtime(), gpu_mem);
    frame->Release();
    decoder->Release();
    *out_width = 0;
    *out_height = 0;
    *out_channels = 0;
    *out_bit_depth = 0;
    return nullptr;
  }

  allocation_lock.unlock();
  frame->Release();
  decoder->Release();
  return (void *)gpu_mem;

#else
  // TODO: Implement for Linux/Android using stb_image or similar
  return nullptr;
#endif
}

EXPORT bool ti_imwrite_from_gpu(void *runtime, const char *path, void *gpu_mem,
                                int width, int height, int channels,
                                int bit_depth) {
  EngineLease lease(runtime);
  EngineContext *engine = lease.get();
  clear_engine_error(engine);
  ScopedOpenGLContext gl_scope(engine);
  if (!gl_scope.ready()) {
    set_engine_error(engine, "ti_imwrite_from_gpu: native graphics context is not ready");
    return false;
  }
  ti::Runtime *rt = engine_runtime(engine);
  if (!rt || !path || !gpu_mem) {
    set_engine_error(engine, "ti_imwrite_from_gpu: invalid runtime, path, or GPU handle");
    return false;
  }

#ifdef _WIN32
  init_wic();
  if (!g_wic_factory) {
    set_engine_error(engine, "ti_imwrite_from_gpu: WIC factory initialization failed");
    return false;
  }

  uint64_t stride64 = 0;
  uint64_t size64 = 0;
  if (!checked_image_geometry(width, height, channels, bit_depth, &stride64,
                              &size64) ||
      stride64 > std::numeric_limits<UINT>::max() ||
      size64 > std::numeric_limits<UINT>::max()) {
    set_engine_error(engine,
                     "ti_imwrite_from_gpu: invalid image geometry or byte-size overflow");
    return false;
  }

  std::unique_lock<std::mutex> allocation_lock(engine->mutex);
  if (!validate_gpu_allocation_locked(engine, (TiMemory)gpu_mem, size64,
                                      "ti_imwrite_from_gpu"))
    return false;

  // Map the source before creating or truncating the destination file.  A
  // failed map must be fail-closed without leaving an empty/partial output at
  // the requested path.
  void *gpu_ptr = ti_map_memory(rt->runtime(), (TiMemory)gpu_mem);
  if (!gpu_ptr) {
    engine->last_error = "ti_imwrite_from_gpu: GPU map failed";
    return false;
  }
  auto allocation_it = engine->allocations.find((TiMemory)gpu_mem);
  if (allocation_it == engine->allocations.end()) {
    ti_unmap_memory(rt->runtime(), (TiMemory)gpu_mem);
    engine->last_error =
        "ti_imwrite_from_gpu: allocation disappeared during GPU map";
    return false;
  }
  allocation_it->second.mapped = true;

  IWICStream *stream = nullptr;
  IWICBitmapEncoder *encoder = nullptr;
  IWICBitmapFrameEncode *frame = nullptr;
  std::vector<wchar_t> temp_w;
  auto cleanup = [&]() {
    if (gpu_ptr) {
      ti_unmap_memory(rt->runtime(), (TiMemory)gpu_mem);
      gpu_ptr = nullptr;
      auto it = engine->allocations.find((TiMemory)gpu_mem);
      if (it != engine->allocations.end())
        it->second.mapped = false;
    }
    if (frame) {
      frame->Release();
      frame = nullptr;
    }
    if (encoder) {
      encoder->Release();
      encoder = nullptr;
    }
    if (stream) {
      stream->Release();
      stream = nullptr;
    }
  };

  auto fail = [&](const char *operation, HRESULT hr) {
    cleanup();
    if (!temp_w.empty())
      DeleteFileW(temp_w.data());
    std::ostringstream message;
    message << "ti_imwrite_from_gpu: " << operation << " failed";
    if (hr != S_OK)
      message << " (HRESULT=0x" << std::hex << static_cast<unsigned long>(hr)
              << ")";
    engine->last_error = message.str();
    return false;
  };

  if (image_io_test_failure("map"))
    return fail("injected GPU map failure", E_FAIL);

  const int target_chars = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);
  if (target_chars <= 0)
    return fail("UTF-8 path conversion", HRESULT_FROM_WIN32(GetLastError()));
  std::vector<wchar_t> target_w(static_cast<size_t>(target_chars));
  if (!MultiByteToWideChar(CP_UTF8, 0, path, -1, target_w.data(), target_chars))
    return fail("UTF-8 path conversion", HRESULT_FROM_WIN32(GetLastError()));

  static std::atomic<uint64_t> temp_sequence{0};
  const uint64_t sequence = ++temp_sequence;
  std::string temp_path = std::string(path) + ".pixelrefine.tmp." +
                          std::to_string(GetCurrentProcessId()) + "." +
                          std::to_string(sequence);
  const int temp_chars =
      MultiByteToWideChar(CP_UTF8, 0, temp_path.c_str(), -1, nullptr, 0);
  if (temp_chars <= 0)
    return fail("temporary path conversion", HRESULT_FROM_WIN32(GetLastError()));
  temp_w.resize(static_cast<size_t>(temp_chars));
  if (!MultiByteToWideChar(CP_UTF8, 0, temp_path.c_str(), -1, temp_w.data(),
                           temp_chars))
    return fail("temporary path conversion", HRESULT_FROM_WIN32(GetLastError()));

  HANDLE temp_handle = CreateFileW(temp_w.data(), GENERIC_WRITE, 0, nullptr,
                                    CREATE_NEW, FILE_ATTRIBUTE_TEMPORARY, nullptr);
  if (temp_handle == INVALID_HANDLE_VALUE)
    return fail("temporary file creation", HRESULT_FROM_WIN32(GetLastError()));
  CloseHandle(temp_handle);

  HRESULT hr = g_wic_factory->CreateStream(&stream);
  if (FAILED(hr) || !stream)
    return fail("WIC stream creation", FAILED(hr) ? hr : E_FAIL);

  hr = stream->InitializeFromFilename(temp_w.data(), GENERIC_WRITE);
  if (FAILED(hr))
    return fail("temporary stream initialization", hr);

  // Auto-detect encoder based on extension
  GUID encoder_guid = GUID_ContainerFormatPng;
  std::string s_path = path;
  if (s_path.find(".jpg") != std::string::npos ||
      s_path.find(".jpeg") != std::string::npos) {
    encoder_guid = GUID_ContainerFormatJpeg;
  } else if (s_path.find(".tif") != std::string::npos) {
    encoder_guid = GUID_ContainerFormatTiff;
  }

  if (image_io_test_failure("encoder"))
    return fail("injected WIC encoder failure", E_FAIL);
  hr = g_wic_factory->CreateEncoder(encoder_guid, NULL, &encoder);
  if (FAILED(hr) || !encoder)
    return fail("WIC encoder creation", FAILED(hr) ? hr : E_FAIL);

  hr = encoder->Initialize(stream, WICBitmapEncoderNoCache);
  if (FAILED(hr))
    return fail("WIC encoder initialization", hr);

  hr = encoder->CreateNewFrame(&frame, NULL);
  if (FAILED(hr) || !frame)
    return fail("WIC frame creation", FAILED(hr) ? hr : E_FAIL);
  hr = frame->Initialize(NULL);
  if (FAILED(hr))
    return fail("WIC frame initialization", hr);
  hr = frame->SetSize(width, height);
  if (FAILED(hr))
    return fail("WIC frame sizing", hr);

  WICPixelFormatGUID format_guid = GUID_WICPixelFormat8bppGray;
  if (bit_depth == 8) {
    format_guid = (channels == 1) ? GUID_WICPixelFormat8bppGray
                                  : GUID_WICPixelFormat24bppBGR;
  } else {
    format_guid = (channels == 1) ? GUID_WICPixelFormat16bppGray
                                  : GUID_WICPixelFormat48bppBGR;
  }

  hr = frame->SetPixelFormat(&format_guid);
  if (FAILED(hr))
    return fail("WIC pixel-format selection", hr);

  HRESULT write_result = frame->WritePixels(
      static_cast<UINT>(height), static_cast<UINT>(stride64),
      static_cast<UINT>(size64), (BYTE *)gpu_ptr);
  ti_unmap_memory(rt->runtime(), (TiMemory)gpu_mem);
  gpu_ptr = nullptr;
  auto mapped_it = engine->allocations.find((TiMemory)gpu_mem);
  if (mapped_it != engine->allocations.end())
    mapped_it->second.mapped = false;
  if (FAILED(write_result))
    return fail("WIC pixel write", write_result);
  if (image_io_test_failure("frame_commit"))
    return fail("injected WIC frame commit failure", E_FAIL);
  hr = frame->Commit();
  if (FAILED(hr))
    return fail("WIC frame commit", hr);
  if (image_io_test_failure("encoder_commit"))
    return fail("injected WIC encoder commit failure", E_FAIL);
  hr = encoder->Commit();
  if (FAILED(hr))
    return fail("WIC encoder commit", hr);

  if (image_io_test_failure("replace"))
    return fail("injected final file replace failure", E_FAIL);
  cleanup();
  if (!MoveFileExW(temp_w.data(), target_w.data(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    const HRESULT move_error = HRESULT_FROM_WIN32(GetLastError());
    DeleteFileW(temp_w.data());
    std::ostringstream message;
    message << "ti_imwrite_from_gpu: final file replace failed (HRESULT=0x"
            << std::hex << static_cast<unsigned long>(move_error) << ")";
    engine->last_error = message.str();
    return false;
  }

  return true;
#else
  set_engine_error(engine,
                   "ti_imwrite_from_gpu: native image writer is unavailable on this platform");
  return false;
#endif
}

EXPORT bool ti_cast_buffer(void *src_ptr, void *dst_ptr,
                           int num_elements, int src_type, int dst_type) {
  if (!src_ptr || !dst_ptr)
    return false;
  if (num_elements < 0)
    return false;

  // i16 is part of the private dtype ABI used by compact CPU/ARM graphs.  It
  // is handled before the ISA-specific branches so every bridge (baseline,
  // AVX2, NEON, Vulkan, OpenGL, and CUDA) has identical defined semantics.
  // The hot image-normalization conversions below retain their SIMD paths;
  // these signed-16 conversions are primarily used for compact intermediate
  // buffers and avoid an unnecessary NumPy round-trip.
  if (src_type == 4 && dst_type == 4) {
    std::memmove(dst_ptr, src_ptr,
                 static_cast<size_t>(num_elements) * sizeof(int16_t));
    return true;
  }

#if defined(__aarch64__) || defined(_M_ARM64)
  if (src_type == 0 && dst_type == 4) { // f32 -> i16 (NEON)
    const float *s = static_cast<const float *>(src_ptr);
    int16_t *d = static_cast<int16_t *>(dst_ptr);
    const float32x4_t lo_bound = vdupq_n_f32(-32768.0f);
    const float32x4_t hi_bound = vdupq_n_f32(32767.0f);
    int i = 0;
    for (; i <= num_elements - 8; i += 8) {
      float32x4_t f0 = vld1q_f32(s + i);
      float32x4_t f1 = vld1q_f32(s + i + 4);
      // Replace NaN with the lower bound before clamping.  This matches the
      // scalar helper and keeps ARM/x86 tails deterministic.
      f0 = vbslq_f32(vceqq_f32(f0, f0), f0, lo_bound);
      f1 = vbslq_f32(vceqq_f32(f1, f1), f1, lo_bound);
      f0 = vmaxq_f32(lo_bound, vminq_f32(hi_bound, f0));
      f1 = vmaxq_f32(lo_bound, vminq_f32(hi_bound, f1));
      const int32x4_t i0 = vcvtq_s32_f32(f0);
      const int32x4_t i1 = vcvtq_s32_f32(f1);
      vst1q_s16(d + i, vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1)));
    }
    for (; i < num_elements; ++i)
      d[i] = ti_float_to_i16(s[i]);
    return true;
  }
  if (src_type == 4 && dst_type == 0) { // i16 -> f32 (NEON)
    const int16_t *s = static_cast<const int16_t *>(src_ptr);
    float *d = static_cast<float *>(dst_ptr);
    int i = 0;
    for (; i <= num_elements - 8; i += 8) {
      const int16x8_t v = vld1q_s16(s + i);
      vst1q_f32(d + i, vcvtq_f32_s32(vmovl_s16(vget_low_s16(v))));
      vst1q_f32(d + i + 4, vcvtq_f32_s32(vmovl_s16(vget_high_s16(v))));
    }
    for (; i < num_elements; ++i)
      d[i] = static_cast<float>(s[i]);
    return true;
  }
#elif (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && defined(PIXEL_REFINE_AOT_BASELINE)
  // SSE2 is mandatory on x86-64 and is the baseline bridge's safe SIMD
  // option.  Keep this separate from the AVX2 block below so an older CPU
  // never decodes an AVX instruction merely because the DLL was loaded.
  if (src_type == 0 && dst_type == 4) { // f32 -> i16 (SSE2)
    const float *s = static_cast<const float *>(src_ptr);
    int16_t *d = static_cast<int16_t *>(dst_ptr);
    const __m128 lo_bound = _mm_set1_ps(-32768.0f);
    const __m128 hi_bound = _mm_set1_ps(32767.0f);
    const __m128i zero_i = _mm_setzero_si128();
    int i = 0;
    for (; i <= num_elements - 4; i += 4) {
      __m128 f = _mm_loadu_ps(s + i);
      const __m128 valid = _mm_cmpord_ps(f, f);
      // Match ti_float_to_i16(): NaN becomes the lower bound.
      f = _mm_or_ps(_mm_and_ps(valid, f),
                    _mm_andnot_ps(valid, lo_bound));
      f = _mm_max_ps(lo_bound, _mm_min_ps(hi_bound, f));
      const __m128i v = _mm_cvttps_epi32(f);
      _mm_storel_epi64(reinterpret_cast<__m128i *>(d + i),
                       _mm_packs_epi32(v, zero_i));
    }
    for (; i < num_elements; ++i)
      d[i] = ti_float_to_i16(s[i]);
    return true;
  }
  if (src_type == 4 && dst_type == 0) { // i16 -> f32 (SSE2)
    const int16_t *s = static_cast<const int16_t *>(src_ptr);
    float *d = static_cast<float *>(dst_ptr);
    const __m128i zero_i = _mm_setzero_si128();
    int i = 0;
    for (; i <= num_elements - 4; i += 4) {
      const __m128i v = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(s + i));
      const __m128i sign = _mm_cmpgt_epi16(zero_i, v);
      const __m128i widened = _mm_unpacklo_epi16(v, sign);
      _mm_storeu_ps(d + i, _mm_cvtepi32_ps(widened));
    }
    for (; i < num_elements; ++i)
      d[i] = static_cast<float>(s[i]);
    return true;
  }
#elif (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && !defined(PIXEL_REFINE_AOT_BASELINE)
  if (src_type == 0 && dst_type == 4) { // f32 -> i16 (AVX2)
    const float *s = static_cast<const float *>(src_ptr);
    int16_t *d = static_cast<int16_t *>(dst_ptr);
    const __m256 lo_bound = _mm256_set1_ps(-32768.0f);
    const __m256 hi_bound = _mm256_set1_ps(32767.0f);
    int i = 0;
    for (; i <= num_elements - 8; i += 8) {
      __m256 f = _mm256_loadu_ps(s + i);
      // _mm256_max/min_ps select the numeric operand for NaN, so a NaN is
      // normalized to the lower bound just like ti_float_to_i16().
      f = _mm256_max_ps(lo_bound, _mm256_min_ps(hi_bound, f));
      const __m256i v = _mm256_cvttps_epi32(f);
      const __m128i signed_packed = _mm_packs_epi32(
          _mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(d + i), signed_packed);
    }
    for (; i < num_elements; ++i)
      d[i] = ti_float_to_i16(s[i]);
    return true;
  }
  if (src_type == 4 && dst_type == 0) { // i16 -> f32 (AVX2)
    const int16_t *s = static_cast<const int16_t *>(src_ptr);
    float *d = static_cast<float *>(dst_ptr);
    int i = 0;
    for (; i <= num_elements - 8; i += 8) {
      const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(s + i));
      const __m256i widened = _mm256_cvtepi16_epi32(v);
      _mm256_storeu_ps(d + i, _mm256_cvtepi32_ps(widened));
    }
    for (; i < num_elements; ++i)
      d[i] = static_cast<float>(s[i]);
    return true;
  }
#endif

  if (src_type == 4 && dst_type == 0) { // i16 -> f32
    const int16_t *s = static_cast<const int16_t *>(src_ptr);
    float *d = static_cast<float *>(dst_ptr);
    for (int i = 0; i < num_elements; ++i)
      d[i] = static_cast<float>(s[i]);
    return true;
  }
  if (src_type == 0 && dst_type == 4) { // f32 -> i16
    const float *s = static_cast<const float *>(src_ptr);
    int16_t *d = static_cast<int16_t *>(dst_ptr);
    for (int i = 0; i < num_elements; ++i)
      d[i] = ti_float_to_i16(s[i]);
    return true;
  }

#if defined(PIXEL_REFINE_AOT_BASELINE)
  // The bridge is also used on machines that predate AVX2.  SSE2 is the
  // guaranteed x86-64 baseline, so use it for bulk host-buffer conversion
  // while retaining scalar tails for exact edge handling.  AOT kernels remain
  // responsible for their own target features; this function only
  // moves/normalizes host buffers.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  if (src_type == 0 && dst_type == 2) { // f32 -> u8 (SSE2)
    const float *s = static_cast<const float *>(src_ptr);
    uint8_t *d = static_cast<uint8_t *>(dst_ptr);
    const __m128 zero = _mm_setzero_ps();
    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 scale = _mm_set1_ps(255.0f);
    const __m128 half = _mm_set1_ps(0.5f);
    const __m128i zero_i = _mm_setzero_si128();
    int i = 0;
    for (; i <= num_elements - 4; i += 4) {
      __m128 f = _mm_loadu_ps(s + i);
      const __m128 valid = _mm_cmpord_ps(f, f);
      f = _mm_and_ps(f, valid);  // NaN -> zero
      f = _mm_max_ps(zero, _mm_min_ps(one, f));
      const __m128i v = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(f, scale), half));
      const __m128i packed16 = _mm_packs_epi32(v, zero_i);
      const __m128i packed8 = _mm_packus_epi16(packed16, zero_i);
      const uint32_t value = static_cast<uint32_t>(_mm_cvtsi128_si32(packed8));
      std::memcpy(d + i, &value, sizeof(value));
    }
    for (; i < num_elements; ++i)
      d[i] = ti_normalized_to_u8(s[i]);
  } else if (src_type == 2 && dst_type == 0) { // u8 -> f32 (SSE2)
    const uint8_t *s = static_cast<const uint8_t *>(src_ptr);
    float *d = static_cast<float *>(dst_ptr);
    const __m128 scale = _mm_set1_ps(1.0f / 255.0f);
    const __m128i zero_i = _mm_setzero_si128();
    int i = 0;
    for (; i <= num_elements - 4; i += 4) {
      uint32_t value = 0;
      std::memcpy(&value, s + i, sizeof(value));
      const __m128i bytes = _mm_cvtsi32_si128(static_cast<int>(value));
      const __m128i u16 = _mm_unpacklo_epi8(bytes, zero_i);
      const __m128i u32 = _mm_unpacklo_epi16(u16, zero_i);
      _mm_storeu_ps(d + i, _mm_mul_ps(_mm_cvtepi32_ps(u32), scale));
    }
    for (; i < num_elements; ++i)
      d[i] = static_cast<float>(s[i]) / 255.0f;
  } else if (src_type == 0 && dst_type == 3) { // f32 -> u16 (SSE2)
    const float *s = static_cast<const float *>(src_ptr);
    uint16_t *d = static_cast<uint16_t *>(dst_ptr);
    const __m128 zero = _mm_setzero_ps();
    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 scale = _mm_set1_ps(65535.0f);
    const __m128 half = _mm_set1_ps(0.5f);
    const __m128i zero_i = _mm_setzero_si128();
    const __m128i sign_bit = _mm_set1_epi16(static_cast<short>(0x8000));
    const __m128i offset = _mm_set1_epi32(32768);
    int i = 0;
    for (; i <= num_elements - 4; i += 4) {
      __m128 f = _mm_loadu_ps(s + i);
      const __m128 valid = _mm_cmpord_ps(f, f);
      f = _mm_and_ps(f, valid);  // NaN -> zero
      f = _mm_max_ps(zero, _mm_min_ps(one, f));
      __m128i v = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(f, scale), half));
      // SSE2 has no packus_epi32.  Translate [0,65535] to the signed
      // interval before packing, then restore the high bit.
      v = _mm_sub_epi32(v, offset);
      v = _mm_xor_si128(_mm_packs_epi32(v, zero_i), sign_bit);
      _mm_storel_epi64(reinterpret_cast<__m128i *>(d + i), v);
    }
    for (; i < num_elements; ++i)
      d[i] = ti_normalized_to_u16(s[i]);
  } else if (src_type == 3 && dst_type == 0) { // u16 -> f32 (SSE2)
    const uint16_t *s = static_cast<const uint16_t *>(src_ptr);
    float *d = static_cast<float *>(dst_ptr);
    const __m128 scale = _mm_set1_ps(1.0f / 65535.0f);
    const __m128i zero_i = _mm_setzero_si128();
    int i = 0;
    for (; i <= num_elements - 4; i += 4) {
      const __m128i u16 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(s + i));
      const __m128i u32 = _mm_unpacklo_epi16(u16, zero_i);
      _mm_storeu_ps(d + i, _mm_mul_ps(_mm_cvtepi32_ps(u32), scale));
    }
    for (; i < num_elements; ++i)
      d[i] = static_cast<float>(s[i]) / 65535.0f;
  } else
#endif
  if (src_type == 0 && dst_type == 2) { // f32 -> u8
    const float *s = (const float *)src_ptr;
    uint8_t *d = (uint8_t *)dst_ptr;
    for (int i = 0; i < num_elements; ++i)
      d[i] = ti_normalized_to_u8(s[i]);
  } else if (src_type == 2 && dst_type == 0) { // u8 -> f32
    const uint8_t *s = (const uint8_t *)src_ptr;
    float *d = (float *)dst_ptr;
    for (int i = 0; i < num_elements; ++i)
      d[i] = (float)s[i] / 255.0f;
  } else if (src_type == 0 && dst_type == 3) { // f32 -> u16
    const float *s = (const float *)src_ptr;
    uint16_t *d = (uint16_t *)dst_ptr;
    for (int i = 0; i < num_elements; ++i)
      d[i] = ti_normalized_to_u16(s[i]);
  } else if (src_type == 3 && dst_type == 0) { // u16 -> f32
    const uint16_t *s = (const uint16_t *)src_ptr;
    float *d = (float *)dst_ptr;
    for (int i = 0; i < num_elements; ++i)
      d[i] = (float)s[i] / 65535.0f;
  } else if (src_type == 1 && dst_type == 3) { // i32 -> u16
    const int32_t *s = (const int32_t *)src_ptr;
    uint16_t *d = (uint16_t *)dst_ptr;
    for (int i = 0; i < num_elements; ++i)
      d[i] = (uint16_t)s[i];
  } else if (src_type == 1 && dst_type == 2) { // i32 -> u8
    const int32_t *s = (const int32_t *)src_ptr;
    uint8_t *d = (uint8_t *)dst_ptr;
    for (int i = 0; i < num_elements; ++i)
      d[i] = (uint8_t)s[i];
  }
#elif defined(__aarch64__) || defined(_M_ARM64)
  // ARMv8-A always provides NEON/ASIMD.  Keep this path in the ARM bridge
  // rather than relying on the scalar fallback: host/device transfers are
  // frequently the dominant cost for CPU AOT on mobile.  The arithmetic and
  // saturating narrowing match the x86 AVX2 path for normalized image data.
  if (src_type == 0 && dst_type == 2) { // f32 -> u8
    const float *s = (const float *)src_ptr;
    uint8_t *d = (uint8_t *)dst_ptr;
    const float32x4_t scale = vdupq_n_f32(255.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    int i = 0;
    for (; i <= num_elements - 16; i += 16) {
      const float32x4_t f0 = vaddq_f32(vmulq_f32(vld1q_f32(s + i), scale), half);
      const float32x4_t f1 = vaddq_f32(vmulq_f32(vld1q_f32(s + i + 4), scale), half);
      const float32x4_t f2 = vaddq_f32(vmulq_f32(vld1q_f32(s + i + 8), scale), half);
      const float32x4_t f3 = vaddq_f32(vmulq_f32(vld1q_f32(s + i + 12), scale), half);
      const uint16x8_t p16 = vcombine_u16(
          vqmovun_s32(vcvtq_s32_f32(f0)), vqmovun_s32(vcvtq_s32_f32(f1)));
      const uint16x8_t p16_hi = vcombine_u16(
          vqmovun_s32(vcvtq_s32_f32(f2)), vqmovun_s32(vcvtq_s32_f32(f3)));
      vst1q_u8(d + i, vcombine_u8(vqmovn_u16(p16), vqmovn_u16(p16_hi)));
    }
    for (; i < num_elements; ++i)
      d[i] = ti_normalized_to_u8(s[i]);
  } else if (src_type == 2 && dst_type == 0) { // u8 -> f32
    const uint8_t *s = (const uint8_t *)src_ptr;
    float *d = (float *)dst_ptr;
    const float32x4_t scale = vdupq_n_f32(1.0f / 255.0f);
    int i = 0;
    for (; i <= num_elements - 8; i += 8) {
      const uint16x8_t u16 = vmovl_u8(vld1_u8(s + i));
      const uint32x4_t lo = vmovl_u16(vget_low_u16(u16));
      const uint32x4_t hi = vmovl_u16(vget_high_u16(u16));
      vst1q_f32(d + i, vmulq_f32(vcvtq_f32_u32(lo), scale));
      vst1q_f32(d + i + 4, vmulq_f32(vcvtq_f32_u32(hi), scale));
    }
    for (; i < num_elements; ++i)
      d[i] = (float)s[i] / 255.0f;
  } else if (src_type == 0 && dst_type == 3) { // f32 -> u16
    const float *s = (const float *)src_ptr;
    uint16_t *d = (uint16_t *)dst_ptr;
    const float32x4_t scale = vdupq_n_f32(65535.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    int i = 0;
    for (; i <= num_elements - 8; i += 8) {
      const float32x4_t f0 = vaddq_f32(vmulq_f32(vld1q_f32(s + i), scale), half);
      const float32x4_t f1 = vaddq_f32(vmulq_f32(vld1q_f32(s + i + 4), scale), half);
      vst1q_u16(d + i, vcombine_u16(vqmovun_s32(vcvtq_s32_f32(f0)),
                                   vqmovun_s32(vcvtq_s32_f32(f1))));
    }
    for (; i < num_elements; ++i)
      d[i] = ti_normalized_to_u16(s[i]);
  } else if (src_type == 3 && dst_type == 0) { // u16 -> f32
    const uint16_t *s = (const uint16_t *)src_ptr;
    float *d = (float *)dst_ptr;
    const float32x4_t scale = vdupq_n_f32(1.0f / 65535.0f);
    int i = 0;
    for (; i <= num_elements - 8; i += 8) {
      const uint16x8_t u16 = vld1q_u16(s + i);
      const uint32x4_t lo = vmovl_u16(vget_low_u16(u16));
      const uint32x4_t hi = vmovl_u16(vget_high_u16(u16));
      vst1q_f32(d + i, vmulq_f32(vcvtq_f32_u32(lo), scale));
      vst1q_f32(d + i + 4, vmulq_f32(vcvtq_f32_u32(hi), scale));
    }
    for (; i < num_elements; ++i)
      d[i] = (float)s[i] / 65535.0f;
  } else if (src_type == 1 && dst_type == 3) { // i32 -> u16
    const int32_t *s = (const int32_t *)src_ptr;
    uint16_t *d = (uint16_t *)dst_ptr;
    for (int i = 0; i < num_elements; ++i)
      d[i] = (uint16_t)s[i];
  } else if (src_type == 1 && dst_type == 2) { // i32 -> u8
    const int32_t *s = (const int32_t *)src_ptr;
    uint8_t *d = (uint8_t *)dst_ptr;
    for (int i = 0; i < num_elements; ++i)
      d[i] = (uint8_t)s[i];
  }
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  if (src_type == 0 && dst_type == 2) { // f32 -> u8
    float *s = (float *)src_ptr;
    uint8_t *d = (uint8_t *)dst_ptr;
    int i = 0;
    __m256 scale = _mm256_set1_ps(255.0f);
    __m256 half = _mm256_set1_ps(0.5f);
    for (; i <= num_elements - 8; i += 8) {
      __m256 f = _mm256_loadu_ps(s + i);
      f = _mm256_mul_ps(f, scale);
      f = _mm256_add_ps(f, half);
      __m256i i32 = _mm256_cvttps_epi32(f);
      __m128i lo = _mm256_castsi256_si128(i32);
      __m128i hi = _mm256_extracti128_si256(i32, 1);
      __m128i packed = _mm_packus_epi32(lo, hi);
      __m128i packed_bytes = _mm_packus_epi16(packed, _mm_setzero_si128());
      uint64_t val = _mm_cvtsi128_si64(packed_bytes);
      std::memcpy(d + i, &val, 8);
    }
    for (; i < num_elements; ++i)
      d[i] = ti_normalized_to_u8(s[i]);
  } else if (src_type == 2 && dst_type == 0) { // u8 -> f32
    uint8_t *s = (uint8_t *)src_ptr;
    float *d = (float *)dst_ptr;
    int i = 0;
    __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
    for (; i <= num_elements - 8; i += 8) {
      uint64_t val;
      std::memcpy(&val, s + i, 8);
      __m128i u8_vec = _mm_cvtsi64_si128(val);
      __m256i i32 = _mm256_cvtepu8_epi32(u8_vec);
      __m256 f = _mm256_cvtepi32_ps(i32);
      f = _mm256_mul_ps(f, scale);
      _mm256_storeu_ps(d + i, f);
    }
    for (; i < num_elements; ++i)
      d[i] = (float)s[i] / 255.0f;
  } else if (src_type == 0 && dst_type == 3) { // f32 -> u16
    float *s = (float *)src_ptr;
    uint16_t *d = (uint16_t *)dst_ptr;
    int i = 0;
    __m256 scale = _mm256_set1_ps(65535.0f);
    __m256 half = _mm256_set1_ps(0.5f);
    for (; i <= num_elements - 8; i += 8) {
      __m256 f = _mm256_loadu_ps(s + i);
      f = _mm256_mul_ps(f, scale);
      f = _mm256_add_ps(f, half);
      __m256i i32 = _mm256_cvttps_epi32(f);
      __m128i lo = _mm256_castsi256_si128(i32);
      __m128i hi = _mm256_extracti128_si256(i32, 1);
      __m128i packed = _mm_packus_epi32(lo, hi);
      _mm_storeu_si128((__m128i*)(d + i), packed);
    }
    for (; i < num_elements; ++i)
      d[i] = ti_normalized_to_u16(s[i]);
  } else if (src_type == 3 && dst_type == 0) { // u16 -> f32
    uint16_t *s = (uint16_t *)src_ptr;
    float *d = (float *)dst_ptr;
    int i = 0;
    __m256 scale = _mm256_set1_ps(1.0f / 65535.0f);
    for (; i <= num_elements - 8; i += 8) {
      __m128i u16_vec = _mm_loadu_si128((const __m128i*)(s + i));
      __m256i i32 = _mm256_cvtepu16_epi32(u16_vec);
      __m256 f = _mm256_cvtepi32_ps(i32);
      f = _mm256_mul_ps(f, scale);
      _mm256_storeu_ps(d + i, f);
    }
    for (; i < num_elements; ++i)
      d[i] = (float)s[i] / 65535.0f;
  } else if (src_type == 1 && dst_type == 3) { // i32 -> u16
    int32_t *s = (int32_t *)src_ptr;
    uint16_t *d = (uint16_t *)dst_ptr;
    for (int i = 0; i < num_elements; ++i)
      d[i] = (uint16_t)s[i];
  } else if (src_type == 1 && dst_type == 2) { // i32 -> u8
    int32_t *s = (int32_t *)src_ptr;
    uint8_t *d = (uint8_t *)dst_ptr;
    for (int i = 0; i < num_elements; ++i)
      d[i] = (uint8_t)s[i];
  }
#else
  // Keep an explicit portable path for any future non-x86/non-ARM target.
  // This branch is intentionally conservative; target-specific bridges can
  // add their own SIMD implementation without changing the exported ABI.
  if (src_type == 0 && dst_type == 2) {
    const float *s = (const float *)src_ptr;
    uint8_t *d = (uint8_t *)dst_ptr;
    for (int i = 0; i < num_elements; ++i)
      d[i] = ti_normalized_to_u8(s[i]);
  } else if (src_type == 2 && dst_type == 0) {
    const uint8_t *s = (const uint8_t *)src_ptr;
    float *d = (float *)dst_ptr;
    for (int i = 0; i < num_elements; ++i)
      d[i] = (float)s[i] / 255.0f;
  } else if (src_type == 0 && dst_type == 3) {
    const float *s = (const float *)src_ptr;
    uint16_t *d = (uint16_t *)dst_ptr;
    for (int i = 0; i < num_elements; ++i)
      d[i] = ti_normalized_to_u16(s[i]);
  } else if (src_type == 3 && dst_type == 0) {
    const uint16_t *s = (const uint16_t *)src_ptr;
    float *d = (float *)dst_ptr;
    for (int i = 0; i < num_elements; ++i)
      d[i] = (float)s[i] / 65535.0f;
  }
#endif
  return true;
}

// -----------------------------------------------------------------------
// Internal Helper for Argument Mapping
// -----------------------------------------------------------------------
static bool _fill_ti_arg(TiNamedArgument &arg, const DynamicArg &dyn_arg,
                         int i, EngineContext *engine = nullptr,
                         const char *operation = "dynamic argument") {
  arg.name = dyn_arg.name;
  /*
  if (dyn_arg.elem_dim_count > 0) {
      printf("[C++ Engine] Arg %s: type=%d, dtype=%d, dim_count=%d,
  elem_dim_count=%d, elem_shape[0]=%d\n", dyn_arg.name, dyn_arg.arg_type,
  dyn_arg.dtype, dyn_arg.dim_count, dyn_arg.elem_dim_count,
  dyn_arg.elem_shape[0]); } else { printf("[C++ Engine] Arg %s: type=%d,
  dtype=%d, dim_count=%d, elem_dim_count=%d\n", dyn_arg.name, dyn_arg.arg_type,
  dyn_arg.dtype, dyn_arg.dim_count, dyn_arg.elem_dim_count);
  }
  */

  if (dyn_arg.arg_type != 0 && dyn_arg.arg_type != 1)
    return false;
  if (dyn_arg.arg_type == 1) { // Scalar
    if (dyn_arg.dtype != 0 && dyn_arg.dtype != 1)
      return false;
    if (dyn_arg.dtype == 0) {  // f32
      arg.argument.type = TI_ARGUMENT_TYPE_F32;
      union {
        uint64_t u;
        float f;
      } converter;
      converter.u = dyn_arg.val_u64;
      arg.argument.value.f32 = converter.f;

      FILE *f = is_debug_logging_enabled() ? fopen("engine_debug.log", "a") : nullptr;
      if (f) {
        fprintf(f, "  [Arg %d] Scalar F32: %f (raw 0x%llx)\n", i, converter.f,
                (unsigned long long)dyn_arg.val_u64);
        fclose(f);
      }
    } else { // i32 or others
      arg.argument.type = TI_ARGUMENT_TYPE_I32;
      arg.argument.value.i32 = (int32_t)dyn_arg.val_u64;

      FILE *f = is_debug_logging_enabled() ? fopen("engine_debug.log", "a") : nullptr;
      if (f) {
        fprintf(f, "  [Arg %d] Scalar I32: %d (raw 0x%llx)\n", i,
                arg.argument.value.i32, (unsigned long long)dyn_arg.val_u64);
        fclose(f);
      }
    }
  } else { // NDArray
    if (dyn_arg.dtype < 0 || dyn_arg.dtype > 5 || dyn_arg.val_u64 == 0 ||
        dyn_arg.dim_count < 1 || dyn_arg.dim_count > 8 ||
        dyn_arg.elem_dim_count < 0 || dyn_arg.elem_dim_count > 8 ||
        (dyn_arg.is_vector != 0 && dyn_arg.is_vector != 1) ||
        (dyn_arg.is_vector &&
         (dyn_arg.vector_dim < 2 || dyn_arg.vector_dim > 4)))
      return false;
    if (!validate_dynamic_arg_allocation(engine, dyn_arg, operation))
      return false;
    for (int d = 0; d < dyn_arg.dim_count; d++) {
      if (dyn_arg.shape[d] <= 0)
        return false;
    }
    for (int d = 0; d < dyn_arg.elem_dim_count; d++) {
      if (dyn_arg.elem_shape[d] <= 0)
        return false;
    }
    arg.argument.type = TI_ARGUMENT_TYPE_NDARRAY;
    arg.argument.value.ndarray.memory = (TiMemory)dyn_arg.val_u64;

    TiDataType ti_dt = TI_DATA_TYPE_F32;
    if (dyn_arg.dtype == 1)
      ti_dt = TI_DATA_TYPE_I32;
    else if (dyn_arg.dtype == 2)
      ti_dt = TI_DATA_TYPE_U8;
    else if (dyn_arg.dtype == 3)
      ti_dt = TI_DATA_TYPE_U16;
    else if (dyn_arg.dtype == 4)
      ti_dt = TI_DATA_TYPE_I16;
    else if (dyn_arg.dtype == 5)
      ti_dt = TI_DATA_TYPE_F16;

    arg.argument.value.ndarray.elem_type = ti_dt;
    arg.argument.value.ndarray.shape.dim_count = dyn_arg.dim_count;
    for (int d = 0; d < dyn_arg.dim_count; d++) {
      arg.argument.value.ndarray.shape.dims[d] = dyn_arg.shape[d];
    }
    arg.argument.value.ndarray.elem_shape.dim_count = dyn_arg.elem_dim_count;
    for (int d = 0; d < dyn_arg.elem_dim_count; d++) {
      arg.argument.value.ndarray.elem_shape.dims[d] = dyn_arg.elem_shape[d];
    }
  }
  return true;
}

// -----------------------------------------------------------------------
// Generic Graph Execution with Fast Cache
// -----------------------------------------------------------------------
EXPORT void run_aot_graph(void *runtime, void *module_ctx,
                          const char *graph_name, DynamicArg *args_array,
                          int num_args) {
  EngineLease lease(runtime);
  EngineContext *engine = lease.get();
  ScopedOpenGLContext gl_scope(engine);
  if (!gl_scope.ready())
    return;
  ti::Runtime *rt = engine_runtime(engine);
  ModuleLease module_lease(module_ctx);
  ModuleContext *ctx = module_lease.get();
  if (!rt || !ctx || !ctx->module || !args_array)
    return;
  if (ctx->owner != engine) {
    set_engine_error(engine,
                     "run_aot_graph: module belongs to a different runtime");
    return;
  }

  try {
    FILE *entry_log = is_debug_logging_enabled() ? fopen("engine_debug.log", "a") : nullptr;
    if (entry_log) {
      fprintf(entry_log, "[C++ Engine] ENTER run_aot_graph graph=%s args=%d runtime=%p module=%p\\n",
              graph_name ? graph_name : "<null>", num_args, (void *)rt,
              (void *)ctx->module);
      fclose(entry_log);
    }
    clear_engine_error(engine);
    std::lock_guard<std::mutex> lock(ctx->cache_mutex);
    std::string gname(graph_name);
    auto it = ctx->graph_cache.find(gname);
    if (it == ctx->graph_cache.end()) {
      FILE *lookup_log = is_debug_logging_enabled() ? fopen("engine_debug.log", "a") : nullptr;
      if (lookup_log) {
        fprintf(lookup_log, "[C++ Engine] GET_GRAPH %s\\n", graph_name);
        fclose(lookup_log);
      }
      ti::ComputeGraph g = ctx->module->get_compute_graph(graph_name);
      it = ctx->graph_cache.emplace(std::move(gname), std::move(g)).first;
    }
    ti::ComputeGraph &graph = it->second;

    std::vector<TiNamedArgument> ti_args;
    ti_args.reserve(num_args);
    for (int i = 0; i < num_args; i++) {
      TiNamedArgument arg = {};
      if (!_fill_ti_arg(arg, args_array[i], i, engine, "run_aot_graph")) {
        set_engine_error(engine, "run_aot_graph: invalid DynamicArg descriptor");
        return;
      }
      ti_args.push_back(arg);
    }

    // Clear any stale errors before launch
    uint64_t junk_size = 0;
    ti_get_last_error(&junk_size, nullptr);

    {
      FILE *f = is_debug_logging_enabled() ? fopen("engine_debug.log", "a") : nullptr;
      if (f) {
        fprintf(f, "[C++ Engine] Launching graph '%s' with %d args...\n",
                graph_name, num_args);
        fclose(f);
      }
    }

    graph.launch(ti_args);

    {
      FILE *f = is_debug_logging_enabled() ? fopen("engine_debug.log", "a") : nullptr;
      if (f) {
        fprintf(f, "[C++ Engine] Graph '%s' launched successfully.\n",
                graph_name);
        fclose(f);
      }
    }

    // Check for launch errors
    uint64_t msg_size = 0;
    ti_get_last_error(&msg_size, nullptr);
    if (msg_size > 1) { // 1 because sometimes it might be just \0
      std::vector<char> msg(msg_size);
      ti_get_last_error(&msg_size, msg.data());
      if (msg[0] != '\0') {
        set_engine_error(engine, std::string("run_aot_graph: ") + msg.data());
        printf("[C++ Engine] ERROR in run_aot_graph launch: %s\n", msg.data());
      }
    }
    fflush(stdout);
  } catch (const std::exception &e) {
    set_engine_error(engine, std::string("run_aot_graph exception: ") + e.what());
    printf("[C++ Engine] EXCEPTION in run_aot_graph: %s\n", e.what());
    fflush(stdout);
  } catch (...) {
    set_engine_error(engine, "run_aot_graph unknown exception");
    printf("[C++ Engine] UNKNOWN EXCEPTION in run_aot_graph\n");
    fflush(stdout);
  }
}

// -----------------------------------------------------------------------
// Pipeline Recording & Execution
// -----------------------------------------------------------------------

EXPORT void clear_pipeline(void *module_ctx, const char *pipeline_name) {
  ModuleLease module_lease(module_ctx);
  ModuleContext *mod = module_lease.get();
  if (mod && mod->owner) {
    std::lock_guard<std::mutex> lock(mod->owner->mutex);
    mod->owner->pipelines.erase(pipeline_name);
    return;
  }

  {
    std::lock_guard<std::mutex> lock(pipelines_mutex);
    global_pipelines.erase(pipeline_name);
  }
  std::lock_guard<std::mutex> lock(engine_contexts_mutex);
  for (auto *ctx : engine_contexts) {
    if (!ctx)
      continue;
    std::lock_guard<std::mutex> ctx_lock(ctx->mutex);
    ctx->pipelines.erase(pipeline_name);
  }
}

EXPORT void clear_pipeline_for_engine(void *runtime, const char *pipeline_name) {
  EngineLease engine_lease(runtime);
  EngineContext *ctx = engine_lease.get();
  if (!ctx || !pipeline_name)
    return;
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->pipelines.erase(pipeline_name);
}

EXPORT void add_to_pipeline(void *module_ctx, const char *pipeline_name,
                            const char *graph_name, DynamicArg *args_array,
                            int num_args) {
  ModuleLease module_lease(module_ctx);
  ModuleContext *mod = module_lease.get();
  if (!args_array || !mod)
    return;

  GraphDispatch dispatch;
  dispatch.module_ctx = module_ctx;
  dispatch.graph_name = graph_name;
  dispatch.args.reserve(num_args);
  dispatch.arg_names.reserve(num_args);

  for (int i = 0; i < num_args; i++) {
    DynamicArg arg = args_array[i];
    TiNamedArgument validation = {};
    if (!_fill_ti_arg(validation, arg, i, mod->owner, "add_to_pipeline")) {
      set_engine_error(mod->owner, "add_to_pipeline: invalid DynamicArg descriptor");
      return;
    }
    // Allocate storage for name string to keep it alive
    dispatch.arg_names.push_back(args_array[i].name);
    arg.name = dispatch.arg_names.back().c_str();
    dispatch.args.push_back(arg);
  }

  if (mod->owner) {
    std::lock_guard<std::mutex> lock(mod->owner->mutex);
    mod->owner->pipelines[pipeline_name].steps.push_back(std::move(dispatch));
  } else {
    std::lock_guard<std::mutex> lock(pipelines_mutex);
    global_pipelines[pipeline_name].steps.push_back(std::move(dispatch));
  }
}

EXPORT void run_pipeline(void *runtime, const char *pipeline_name,
                         uint64_t *old_handles, DynamicArg *new_args,
                         int num_overrides) {
  EngineLease lease(runtime);
  EngineContext *engine = lease.get();
  ScopedOpenGLContext gl_scope(engine);
  if (!gl_scope.ready())
    return;
  ti::Runtime *rt = engine_runtime(engine);
  if (!rt)
    return;

  Pipeline pipe;
  {
    bool found = false;
    if (engine) {
      std::lock_guard<std::mutex> lock(engine->mutex);
      auto it_pipe = engine->pipelines.find(pipeline_name);
      if (it_pipe != engine->pipelines.end()) {
        pipe = it_pipe->second;
        found = true;
      }
    }
    if (!found) {
      std::lock_guard<std::mutex> lock(pipelines_mutex);
      auto it_pipe = global_pipelines.find(pipeline_name);
      if (it_pipe != global_pipelines.end()) {
        pipe = it_pipe->second;
        found = true;
      }
    }
    if (!found) {
      set_engine_error(engine, std::string("Pipeline not found: ") + pipeline_name);
      printf("[C++ Engine] ERROR: Pipeline '%s' not found!\n", pipeline_name);
      fflush(stdout);
      return;
    }
  }
  if (pipe.steps.empty()) {
    printf("[C++ Engine] WARNING: Pipeline '%s' has 0 steps.\n", pipeline_name);
    fflush(stdout);
    return;
  }

  try {
    clear_engine_error(engine);
    for (auto &step : pipe.steps) {
      ModuleLease module_lease(step.module_ctx);
      ModuleContext *ctx = module_lease.get();
      if (!ctx) {
        set_engine_error(engine, "run_pipeline: module handle is stale");
        return;
      }
      if (ctx->owner != engine) {
        set_engine_error(engine,
                         "run_pipeline: module belongs to a different runtime");
        return;
      }

      std::lock_guard<std::mutex> lock(ctx->cache_mutex);
      auto it_g = ctx->graph_cache.find(step.graph_name);
      if (it_g == ctx->graph_cache.end()) {
        ti::ComputeGraph g =
            ctx->module->get_compute_graph(step.graph_name.c_str());
        it_g = ctx->graph_cache.emplace(step.graph_name, std::move(g)).first;
      }
      ti::ComputeGraph &graph = it_g->second;

      std::vector<TiNamedArgument> ti_args;
      ti_args.reserve(step.args.size());

      int arg_idx = 0;
      for (const auto &base_arg : step.args) {
        TiNamedArgument arg = {};

        // Check for overrides by memory handle identity
        const DynamicArg *final_arg = &base_arg;
        if (base_arg.arg_type == 0) {
          uint64_t current_handle = base_arg.val_u64;
          for (int j = 0; j < num_overrides; j++) {
            if (old_handles[j] == current_handle) {
              final_arg = &new_args[j];
              break;
            }
          }
        }
        if (!_fill_ti_arg(arg, *final_arg, arg_idx++, engine, "run_pipeline")) {
          set_engine_error(engine, "run_pipeline: invalid DynamicArg descriptor");
          return;
        }

        // CRITICAL: Always use the original name from the recorded step.
        // The override argument from Python might have a generic name like
        // "override".
        arg.name = base_arg.name;

        ti_args.push_back(arg);
      }

      graph.launch(ti_args);
    }

    // Final synchronization is optional, but we'll remove it to allow
    // pipelining. rt->wait();

    // Check for error once after the whole pipeline launch
    uint64_t msg_size = 0;
    ti_get_last_error(&msg_size, nullptr);
    if (msg_size > 1) {
      std::vector<char> msg(msg_size);
      ti_get_last_error(&msg_size, msg.data());
      if (msg[0] != '\0') {
        set_engine_error(engine, std::string("run_pipeline: ") + msg.data());
        printf("[C++ Engine] ERROR in pipeline '%s': %s\n", pipeline_name,
               msg.data());
      }
    }
    fflush(stdout);
  } catch (const std::exception &e) {
    set_engine_error(engine, std::string("run_pipeline exception: ") + e.what());
    printf("[C++ Engine] EXCEPTION in run_pipeline: %s\n", e.what());
    fflush(stdout);
  } catch (...) {
    set_engine_error(engine, "run_pipeline unknown exception");
    printf("[C++ Engine] UNKNOWN EXCEPTION in run_pipeline\n");
    fflush(stdout);
  }
}

} // extern "C"
