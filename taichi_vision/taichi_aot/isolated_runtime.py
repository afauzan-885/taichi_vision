"""Process-owned native runtime proxy.

The native Taichi C API exposes process-local pointers for engines, modules,
and allocations.  A Python thread timeout cannot cancel a driver call, so the
only real containment boundary is a child process that owns the complete
native lifecycle.  This module deliberately has no dependency on ``engine``;
the worker can therefore start without constructing the module-level default
engine in the parent process.

The parent side exposes integer-like logical handles.  Payloads cross the
wire as bounded JSON/base64 values; native pointers never cross the process
boundary.
"""

from __future__ import annotations

import base64
import ctypes
import json
import os
import queue
import subprocess
import sys
import threading
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any


_PROTOCOL_VERSION = 1
_DEFAULT_TIMEOUT = max(1.0, float(os.environ.get("AOT_INIT_TIMEOUT", "30")))
# A 12MP float32 RGB readback is 144 MiB before base64/JSON framing.  Keep a
# bounded, explicit ceiling large enough for the supported image contract;
# this is still finite and prevents unbounded IPC allocation.
_MAX_MESSAGE_BYTES = 512 * 1024 * 1024
_MAX_DIAGNOSTIC_BYTES = 2048
_SHARED_MEMORY_THRESHOLD = 1 * 1024 * 1024


class IsolatedRuntimeError(RuntimeError):
    """Bounded error returned by the process-owned runtime."""


class RuntimeHandle(int):
    """Integer-like logical runtime handle retained by the parent."""

    def __new__(cls, value: int, session: "IsolatedRuntime"):
        obj = int.__new__(cls, int(value))
        obj.session = session
        return obj


class ModuleHandle(int):
    """Integer-like logical module handle retained by the parent."""

    def __new__(cls, value: int, session: "IsolatedRuntime"):
        obj = int.__new__(cls, int(value))
        obj.session = session
        return obj


class BufferHandle(int):
    """Integer-like logical allocation handle retained by the parent."""

    def __new__(cls, value: int, session: "IsolatedRuntime"):
        obj = int.__new__(cls, int(value))
        obj.session = session
        return obj


class _MappedRegion:
    __slots__ = ("session", "handle", "storage", "size")

    def __init__(self, session, handle, storage, size):
        self.session = session
        self.handle = handle
        self.storage = storage
        self.size = int(size)


def _bounded_text(value: Any, limit: int = _MAX_DIAGNOSTIC_BYTES) -> str:
    return str(value or "")[:limit]


def _pointer_value(value: Any) -> int:
    if isinstance(value, ctypes.c_void_p):
        return int(value.value or 0)
    raw = getattr(value, "value", value)
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _buffer_address(shm: shared_memory.SharedMemory) -> int:
    """Return a stable writable address while the shared mapping is open."""

    return ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))


def _encode_bytes(value: bytes) -> str:
    return base64.b64encode(bytes(value)).decode("ascii")


def _decode_bytes(value: str, limit: int = _MAX_MESSAGE_BYTES) -> bytes:
    data = base64.b64decode(str(value or ""), validate=True)
    if len(data) > limit:
        raise IsolatedRuntimeError("IPC payload exceeds the bounded size limit")
    return data


def _arg_to_wire(arg) -> dict[str, Any]:
    """Copy a DynamicArg without crossing any parent pointer fields."""

    name = getattr(arg, "name", None)
    if isinstance(name, bytes):
        name_text = name.decode("utf-8", errors="replace")
    else:
        name_text = str(name or "")
    return {
        "name": name_text[:256],
        "arg_type": int(arg.arg_type),
        "dtype": int(arg.dtype),
        "dim_count": int(arg.dim_count),
        "shape": [int(arg.shape[i]) for i in range(min(8, int(arg.dim_count)))],
        "elem_dim_count": int(arg.elem_dim_count),
        "elem_shape": [
            int(arg.elem_shape[i]) for i in range(min(8, int(arg.elem_dim_count)))
        ],
        "is_vector": int(arg.is_vector),
        "vector_dim": int(arg.vector_dim),
        "val_u64": int(arg.val_u64),
    }


def _args_to_wire(args, count: int) -> list[dict[str, Any]]:
    return [_arg_to_wire(args[i]) for i in range(max(0, int(count)))]


class IsolatedRuntime:
    """Synchronous parent-side RPC client for one child-owned runtime."""

    def __init__(self, process: subprocess.Popen, timeout: float):
        self.process = process
        self.timeout = max(1.0, float(timeout))
        self.runtime = None
        self._request_id = 0
        self._responses: queue.Queue = queue.Queue()
        self._write_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._dead = False
        self._reader = threading.Thread(
            target=self._read_responses,
            name=f"AOT-IsolatedRuntime-{process.pid}",
            daemon=True,
        )
        self._reader.start()

    @classmethod
    def start(
        cls,
        bridge_path: str,
        arch_id: int,
        device_id: int,
        *,
        backend: str,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> "IsolatedRuntime":
        bridge_path = os.path.abspath(os.fspath(bridge_path))
        if not os.path.isfile(bridge_path):
            raise IsolatedRuntimeError(
                f"isolated runtime bridge does not exist: {bridge_path}"
            )
        env = os.environ.copy()
        env["AOT_ISOLATED_WORKER"] = "1"
        env["AOT_ISOLATED_BACKEND"] = str(backend)
        env["AOT_ISOLATED_BRIDGE"] = bridge_path
        env["AOT_ISOLATED_ARCH_ID"] = str(int(arch_id))
        env["AOT_ISOLATED_DEVICE_ID"] = str(int(device_id))
        bridge_dir = os.path.dirname(bridge_path)
        if os.name == "nt":
            env["PATH"] = bridge_dir + os.pathsep + env.get("PATH", "")
        process = subprocess.Popen(
            # Execute this stdlib-only module by file path.  ``python -m``
            # would import ``taichi_vision.taichi_aot.__init__`` first, which
            # constructs the module-level engine before the worker can enter
            # its IPC loop and would recurse into another worker.
            [sys.executable, os.fspath(Path(__file__).resolve()), "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=env,
            cwd=os.getcwd(),
            bufsize=1,
            text=True,
        )
        client = cls(process, timeout)
        try:
            result = client._request(
                "init",
                {
                    "bridge": bridge_path,
                    "backend": str(backend),
                    "arch_id": int(arch_id),
                    "device_id": int(device_id),
                },
                timeout=timeout,
            )
            token = int(result.get("runtime", 0) or 0)
            if not token:
                raise IsolatedRuntimeError("worker returned an empty runtime handle")
            client.runtime = RuntimeHandle(token, client)
            return client
        except Exception:
            client.close(force=True)
            raise

    @property
    def alive(self) -> bool:
        return not self._dead and self.process.poll() is None

    def _read_responses(self) -> None:
        stream = self.process.stdout
        if stream is None:
            self._responses.put(None)
            return
        try:
            for line in stream:
                if len(line) > _MAX_MESSAGE_BYTES:
                    self._responses.put(
                        IsolatedRuntimeError("IPC response exceeds the bounded size limit")
                    )
                    break
                try:
                    payload = json.loads(line)
                except (TypeError, ValueError):
                    # Native/runtime diagnostics are allowed on stdout.  The
                    # protocol only consumes valid JSON response objects.
                    continue
                if isinstance(payload, dict) and payload.get("protocol") == _PROTOCOL_VERSION:
                    self._responses.put(payload)
        finally:
            self._responses.put(None)

    def _mark_dead(self, reason: str) -> None:
        with self._state_lock:
            self._dead = True
        raise IsolatedRuntimeError(_bounded_text(reason))

    def _request(self, opcode: str, payload: dict[str, Any], *, timeout=None) -> dict[str, Any]:
        timeout = self.timeout if timeout is None else max(0.1, float(timeout))
        with self._write_lock:
            with self._state_lock:
                if self._dead or self.process.poll() is not None:
                    self._dead = True
                    raise IsolatedRuntimeError("isolated AOT runtime worker is not alive")
                self._request_id += 1
                request_id = self._request_id
                request = {
                    "protocol": _PROTOCOL_VERSION,
                    "request_id": request_id,
                    "opcode": str(opcode),
                    "payload": payload,
                }
                encoded = json.dumps(request, separators=(",", ":"))
                if len(encoded.encode("utf-8")) > _MAX_MESSAGE_BYTES:
                    raise IsolatedRuntimeError("IPC request exceeds the bounded size limit")
                try:
                    assert self.process.stdin is not None
                    self.process.stdin.write(encoded + "\n")
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError) as exc:
                    self._dead = True
                    raise IsolatedRuntimeError(f"isolated runtime IPC write failed: {exc}") from exc

            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.close(force=True)
                    raise IsolatedRuntimeError(
                        f"isolated AOT runtime operation '{opcode}' timed out after {timeout:.1f}s"
                    )
                try:
                    # Do not wait for the full operation timeout when the
                    # child has already exited without flushing a protocol
                    # response.  The reader thread normally publishes EOF,
                    # but process polling closes this race deterministically.
                    response = self._responses.get(timeout=min(remaining, 0.05))
                except queue.Empty:
                    code = self.process.poll()
                    if code is not None:
                        self._dead = True
                        raise IsolatedRuntimeError(
                            f"isolated AOT runtime worker exited unexpectedly (code={code})"
                        )
                    # The child is still alive; keep polling until the
                    # operation deadline rather than treating the 50 ms
                    # responsiveness slice as the full timeout.
                    continue
                if response is None:
                    self._dead = True
                    code = self.process.poll()
                    raise IsolatedRuntimeError(
                        f"isolated AOT runtime worker exited unexpectedly (code={code})"
                    )
                if isinstance(response, Exception):
                    self._dead = True
                    raise response
                if int(response.get("request_id", -1)) != request_id:
                    continue
                if not response.get("ok", False):
                    raise IsolatedRuntimeError(
                        _bounded_text(response.get("error", "isolated runtime operation failed"))
                    )
                return dict(response.get("result") or {})

    def _session_for(self, value: Any) -> bool:
        return isinstance(value, (RuntimeHandle, ModuleHandle, BufferHandle)) and getattr(
            value, "session", None
        ) is self

    def _require_runtime(self, value: Any = None) -> RuntimeHandle:
        if value is None:
            value = self.runtime
        if not self._session_for(value):
            raise IsolatedRuntimeError("logical handle belongs to another isolated runtime")
        return value

    def _module(self, token: int) -> ModuleHandle:
        return ModuleHandle(token, self)

    def _buffer(self, token: int) -> BufferHandle:
        return BufferHandle(token, self)

    def load_module(self, runtime, path: str) -> ModuleHandle:
        self._require_runtime(runtime)
        return self._module(int(self._request("load_module", {"path": os.fspath(path)})["module"]))

    def destroy_module(self, module) -> None:
        if not self._session_for(module):
            raise IsolatedRuntimeError("logical module belongs to another runtime")
        self._request("destroy_module", {"module": int(module)})

    def allocate(self, runtime, size: int, host_accessible: int) -> BufferHandle:
        self._require_runtime(runtime)
        result = self._request(
            "allocate", {"size": int(size), "host_accessible": int(host_accessible)}
        )
        return self._buffer(int(result["buffer"]))

    def free(self, runtime, buffer) -> None:
        self._require_runtime(runtime)
        self._request("free", {"buffer": int(buffer)})

    def write(self, runtime, buffer, host_ptr, size: int) -> None:
        self._require_runtime(runtime)
        size = int(size)
        if size >= _SHARED_MEMORY_THRESHOLD:
            shm = shared_memory.SharedMemory(create=True, size=size)
            try:
                ctypes.memmove(_buffer_address(shm), _pointer_value(host_ptr), size)
                self._request(
                    "write_shm",
                    {"buffer": int(buffer), "name": shm.name, "size": size},
                )
            finally:
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            return
        data = ctypes.string_at(_pointer_value(host_ptr), size)
        self._request("write", {"buffer": int(buffer), "data": _encode_bytes(data)})

    def read(self, runtime, buffer, host_ptr, size: int) -> None:
        self._require_runtime(runtime)
        size = int(size)
        if size >= _SHARED_MEMORY_THRESHOLD:
            shm = shared_memory.SharedMemory(create=True, size=size)
            try:
                self._request(
                    "read_shm",
                    {"buffer": int(buffer), "name": shm.name, "size": size},
                )
                ctypes.memmove(_pointer_value(host_ptr), _buffer_address(shm), size)
            finally:
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            return
        result = self._request("read", {"buffer": int(buffer), "size": size})
        data = _decode_bytes(result.get("data", ""), _MAX_MESSAGE_BYTES)
        if len(data) != size:
            raise IsolatedRuntimeError("isolated read returned an unexpected byte count")
        ctypes.memmove(_pointer_value(host_ptr), data, len(data))

    def map(self, runtime, buffer) -> int:
        self._require_runtime(runtime)
        result = self._request("read", {"buffer": int(buffer), "size": int(result_size := 0)})
        raise IsolatedRuntimeError("map requires the parent allocation size")

    def map_with_size(self, runtime, buffer, size: int) -> int:
        self._require_runtime(runtime)
        result = self._request("read", {"buffer": int(buffer), "size": int(size)})
        data = _decode_bytes(result.get("data", ""), _MAX_MESSAGE_BYTES)
        storage = ctypes.create_string_buffer(data, len(data))
        region = _MappedRegion(self, int(buffer), storage, int(size))
        _MAPPED_BY_ADDRESS[ctypes.addressof(storage)] = region
        _MAPPED_BY_BUFFER[(id(self), int(buffer))] = region
        return ctypes.addressof(storage)

    def unmap(self, runtime, buffer) -> None:
        self._require_runtime(runtime)
        key = (id(self), int(buffer))
        region = _MAPPED_BY_BUFFER.pop(key, None)
        if region is None:
            raise IsolatedRuntimeError("logical buffer is not mapped")
        _MAPPED_BY_ADDRESS.pop(ctypes.addressof(region.storage), None)
        data = ctypes.string_at(ctypes.addressof(region.storage), region.size)
        self._request("write", {"buffer": int(buffer), "data": _encode_bytes(data)})

    def copy(self, runtime, src, dst, size: int) -> None:
        self._require_runtime(runtime)
        self._request("copy", {"src": int(src), "dst": int(dst), "size": int(size)})

    def run_graph(self, runtime, module, graph: bytes, args, count: int) -> None:
        self._require_runtime(runtime)
        if not self._session_for(module):
            raise IsolatedRuntimeError("logical module belongs to another runtime")
        self._request(
            "run_graph",
            {"module": int(module), "graph": bytes(graph).decode("utf-8", errors="replace"), "args": _args_to_wire(args, count)},
        )

    def add_pipeline(self, module, pipeline: bytes, graph: bytes, args, count: int) -> None:
        if not self._session_for(module):
            raise IsolatedRuntimeError("logical module belongs to another runtime")
        self._request(
            "add_pipeline",
            {"module": int(module), "pipeline": bytes(pipeline).decode("utf-8", errors="replace"), "graph": bytes(graph).decode("utf-8", errors="replace"), "args": _args_to_wire(args, count)},
        )

    def run_pipeline(self, runtime, pipeline: bytes, handles, args, count: int) -> None:
        self._require_runtime(runtime)
        self._request(
            "run_pipeline",
            {"pipeline": bytes(pipeline).decode("utf-8", errors="replace"), "handles": [int(handles[i]) for i in range(int(count))], "args": _args_to_wire(args, count)},
        )

    def clear_pipeline_for_module(self, module, name: bytes) -> None:
        if not self._session_for(module):
            raise IsolatedRuntimeError("logical module belongs to another runtime")
        self._request("clear_pipeline_module", {"module": int(module), "name": bytes(name).decode("utf-8", errors="replace")})

    def clear_pipeline_for_runtime(self, runtime, name: bytes) -> None:
        self._require_runtime(runtime)
        self._request("clear_pipeline_runtime", {"name": bytes(name).decode("utf-8", errors="replace")})

    def sync(self, runtime) -> None:
        self._require_runtime(runtime)
        self._request("sync", {})

    def imread(self, runtime, path: bytes) -> dict[str, Any]:
        self._require_runtime(runtime)
        result = self._request("imread", {"path": bytes(path).decode("utf-8", errors="replace")})
        result["buffer"] = self._buffer(int(result["buffer"]))
        return result

    def imwrite(self, runtime, path: bytes, buffer, width, height, channels, depth) -> bool:
        self._require_runtime(runtime)
        result = self._request(
            "imwrite",
            {"path": bytes(path).decode("utf-8", errors="replace"), "buffer": int(buffer), "width": int(width), "height": int(height), "channels": int(channels), "depth": int(depth)},
        )
        return bool(result.get("ok", False))

    def last_error(self) -> bytes:
        try:
            result = self._request("last_error", {})
            return _bounded_text(result.get("error", "")).encode("utf-8", errors="replace")
        except Exception as exc:
            return _bounded_text(exc).encode("utf-8", errors="replace")

    def clear_error(self) -> None:
        try:
            self._request("clear_error", {})
        except Exception:
            pass

    def runtime_name(self) -> bytes:
        result = self._request("runtime_info", {})
        return str(result.get("device_name", "")).encode("utf-8", errors="replace")

    def runtime_context_backend(self) -> bytes:
        result = self._request("runtime_info", {})
        return str(result.get("context_backend", "")).encode("utf-8", errors="replace")

    def runtime_arch(self) -> int:
        return int(self._request("runtime_info", {}).get("arch_id", -1))

    def destroy(self, *, force=False) -> None:
        with self._state_lock:
            if self._dead:
                return
        if not force:
            try:
                self._request("destroy", {}, timeout=min(self.timeout, 5.0))
            except Exception:
                pass
        with self._state_lock:
            self._dead = True
        if self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=2.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        self.runtime = None

    close = destroy


_MAPPED_BY_ADDRESS: dict[int, _MappedRegion] = {}
_MAPPED_BY_BUFFER: dict[tuple[int, int], _MappedRegion] = {}


class BridgeRouter:
    """Route logical handles to isolated sessions and raw calls to ctypes."""

    def __init__(self, native):
        self.native = native
        self._sessions: set[IsolatedRuntime] = set()

    def register(self, session: IsolatedRuntime) -> None:
        self._sessions.add(session)

    def _session(self, value: Any) -> IsolatedRuntime | None:
        session = getattr(value, "session", None)
        if isinstance(session, IsolatedRuntime):
            return session
        return None

    def __getattr__(self, name):
        return getattr(self.native, name)

    def init_aot_engine(self, *args):
        return self.native.init_aot_engine(*args)

    def destroy_aot_engine(self, runtime):
        session = self._session(runtime)
        if session is None:
            return self.native.destroy_aot_engine(runtime)
        session.destroy()

    def get_runtime_arch_id(self, runtime):
        session = self._session(runtime)
        return session.runtime_arch() if session else self.native.get_runtime_arch_id(runtime)

    def get_runtime_device_name(self, runtime):
        session = self._session(runtime)
        return session.runtime_name() if session else self.native.get_runtime_device_name(runtime)

    def get_runtime_context_backend(self, runtime):
        session = self._session(runtime)
        return session.runtime_context_backend() if session else self.native.get_runtime_context_backend(runtime)

    def get_last_engine_error(self, runtime):
        session = self._session(runtime)
        return session.last_error() if session else self.native.get_last_engine_error(runtime)

    def clear_last_engine_error(self, runtime):
        session = self._session(runtime)
        return session.clear_error() if session else self.native.clear_last_engine_error(runtime)

    def load_aot_module(self, runtime, path):
        session = self._session(runtime)
        if session:
            if isinstance(path, ctypes.c_char_p):
                path = ctypes.string_at(path)
            if isinstance(path, bytes):
                path = path.decode("utf-8", errors="surrogateescape")
            return session.load_module(runtime, path)
        return self.native.load_aot_module(runtime, path)

    def destroy_aot_module(self, module):
        session = self._session(module)
        return session.destroy_module(module) if session else self.native.destroy_aot_module(module)

    def allocate_gpu_buffer(self, runtime, size, host_accessible):
        session = self._session(runtime)
        return session.allocate(runtime, int(size), int(host_accessible)) if session else self.native.allocate_gpu_buffer(runtime, size, host_accessible)

    def free_gpu_buffer(self, runtime, buffer):
        session = self._session(runtime)
        return session.free(runtime, buffer) if session else self.native.free_gpu_buffer(runtime, buffer)

    def write_to_gpu_buffer(self, runtime, buffer, host_ptr, size):
        session = self._session(runtime)
        return session.write(runtime, buffer, host_ptr, int(size)) if session else self.native.write_to_gpu_buffer(runtime, buffer, host_ptr, size)

    def read_from_gpu_buffer(self, runtime, buffer, host_ptr, size):
        session = self._session(runtime)
        return session.read(runtime, buffer, host_ptr, int(size)) if session else self.native.read_from_gpu_buffer(runtime, buffer, host_ptr, size)

    def map_gpu_buffer(self, runtime, buffer):
        session = self._session(runtime)
        if session is None:
            return self.native.map_gpu_buffer(runtime, buffer)
        # ``TaichiGPUBuffer`` supplies capacity only through its wrapper.  The
        # router receives the logical handle, so use the bounded read request
        # size recorded by the parent-side map helper when available.  The
        # engine patches this method through ``map_gpu_buffer_with_size``.
        raise IsolatedRuntimeError("isolated map requires map_gpu_buffer_with_size")

    def map_gpu_buffer_with_size(self, runtime, buffer, size: int):
        session = self._session(runtime)
        return session.map_with_size(runtime, buffer, int(size)) if session else self.native.map_gpu_buffer(runtime, buffer)

    def unmap_gpu_buffer(self, runtime, buffer):
        session = self._session(runtime)
        return session.unmap(runtime, buffer) if session else self.native.unmap_gpu_buffer(runtime, buffer)

    def copy_gpu_buffer(self, runtime, src, dst, size):
        session = self._session(runtime)
        return session.copy(runtime, src, dst, int(size)) if session else self.native.copy_gpu_buffer(runtime, src, dst, size)

    def run_aot_graph(self, runtime, module, graph, args, count):
        session = self._session(runtime)
        return session.run_graph(runtime, module, graph, args, int(count)) if session else self.native.run_aot_graph(runtime, module, graph, args, count)

    def add_to_pipeline(self, module, pipeline, graph, args, count):
        session = self._session(module)
        return session.add_pipeline(module, pipeline, graph, args, int(count)) if session else self.native.add_to_pipeline(module, pipeline, graph, args, count)

    def run_pipeline(self, runtime, pipeline, handles, args, count):
        session = self._session(runtime)
        return session.run_pipeline(runtime, pipeline, handles, args, int(count)) if session else self.native.run_pipeline(runtime, pipeline, handles, args, count)

    def clear_pipeline(self, owner, name):
        session = self._session(owner)
        if session:
            return session.clear_pipeline_for_module(owner, name)
        return self.native.clear_pipeline(owner, name)

    def clear_pipeline_for_engine(self, runtime, name):
        session = self._session(runtime)
        return session.clear_pipeline_for_runtime(runtime, name) if session else self.native.clear_pipeline_for_engine(runtime, name)

    def sync_runtime(self, runtime):
        session = self._session(runtime)
        return session.sync(runtime) if session else self.native.sync_runtime(runtime)

    def ti_imread_to_gpu(self, runtime, path, wp, hp, cp, dp):
        session = self._session(runtime)
        if session is None:
            return self.native.ti_imread_to_gpu(runtime, path, wp, hp, cp, dp)
        if isinstance(path, ctypes.c_char_p):
            path = ctypes.string_at(path)
        if isinstance(path, str):
            path = path.encode("utf-8", errors="surrogateescape")
        result = session.imread(runtime, path)
        for pointer, key in ((wp, "width"), (hp, "height"), (cp, "channels"), (dp, "depth")):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = int(result[key])
        return result["buffer"]

    def ti_imwrite_from_gpu(self, runtime, path, buffer, width, height, channels, depth):
        session = self._session(runtime)
        if not session:
            return self.native.ti_imwrite_from_gpu(runtime, path, buffer, width, height, channels, depth)
        if isinstance(path, ctypes.c_char_p):
            path = ctypes.string_at(path)
        if isinstance(path, str):
            path = path.encode("utf-8", errors="surrogateescape")
        return session.imwrite(runtime, path, buffer, width, height, channels, depth)

    def ti_cast_buffer(self, src_ptr, dst_ptr, count, src_dtype, dst_dtype):
        src_address = _pointer_value(src_ptr)
        dst_address = _pointer_value(dst_ptr)
        src = _MAPPED_BY_ADDRESS.get(src_address)
        dst = _MAPPED_BY_ADDRESS.get(dst_address)
        if src is None or dst is None:
            return self.native.ti_cast_buffer(src_ptr, dst_ptr, count, src_dtype, dst_dtype)
        import numpy as np

        dtype_map = {0: np.float32, 1: np.int32, 2: np.uint8, 3: np.uint16, 4: np.int16, 5: np.float16}
        source = np.frombuffer(src.storage, dtype=dtype_map[int(src_dtype)], count=int(count))
        target = np.frombuffer(dst.storage, dtype=dtype_map[int(dst_dtype)], count=int(count))
        if (int(src_dtype), int(dst_dtype)) == (0, 4):
            source = np.nan_to_num(source, nan=-32768.0, posinf=32767.0, neginf=-32768.0)
            target[:] = np.clip(source, -32768.0, 32767.0).astype(np.int16)
        else:
            target[:] = source.astype(dtype_map[int(dst_dtype)])
        return True


def install_bridge_router(native, session: IsolatedRuntime) -> BridgeRouter:
    router = native if isinstance(native, BridgeRouter) else BridgeRouter(native)
    router.register(session)
    return router


# ---------------------------------------------------------------------------
# Child worker.  This section intentionally uses only ctypes and stdlib.
# ---------------------------------------------------------------------------


class _DynamicArg(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("arg_type", ctypes.c_int),
        ("dtype", ctypes.c_int),
        ("dim_count", ctypes.c_int),
        ("shape", ctypes.c_int * 8),
        ("elem_dim_count", ctypes.c_int),
        ("elem_shape", ctypes.c_int * 8),
        ("is_vector", ctypes.c_int),
        ("vector_dim", ctypes.c_int),
        ("val_u64", ctypes.c_uint64),
    ]


class _Worker:
    def __init__(self, bridge: str, backend: str, arch_id: int, device_id: int):
        bridge = os.path.abspath(bridge)
        bridge_dir = os.path.dirname(bridge)
        if os.name == "nt":
            try:
                os.add_dll_directory(bridge_dir)
            except (AttributeError, OSError):
                pass
        self.lib = ctypes.CDLL(bridge)
        self._configure()
        self.backend = str(backend)
        self.arch_id = int(arch_id)
        self.device_id = int(device_id)
        self.runtime = None
        self.modules: dict[int, Any] = {}
        self.buffers: dict[int, Any] = {}
        self.next_module = 1
        self.next_buffer = 1
        self.last_error = ""

    def _configure(self):
        lib = self.lib
        lib.init_aot_engine.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.init_aot_engine.restype = ctypes.c_void_p
        lib.destroy_aot_engine.argtypes = [ctypes.c_void_p]
        lib.destroy_aot_engine.restype = None
        for name, args, result in [
            ("get_last_engine_error", [ctypes.c_void_p], ctypes.c_char_p),
            ("clear_last_engine_error", [ctypes.c_void_p], None),
            ("get_runtime_device_name", [ctypes.c_void_p], ctypes.c_char_p),
            ("get_runtime_context_backend", [ctypes.c_void_p], ctypes.c_char_p),
            ("get_runtime_arch_id", [ctypes.c_void_p], ctypes.c_int),
        ]:
            fn = getattr(lib, name, None)
            if fn is not None:
                fn.argtypes = args
                fn.restype = result
        lib.load_aot_module.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.load_aot_module.restype = ctypes.c_void_p
        lib.destroy_aot_module.argtypes = [ctypes.c_void_p]
        lib.destroy_aot_module.restype = None
        lib.allocate_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
        lib.allocate_gpu_buffer.restype = ctypes.c_void_p
        lib.free_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.free_gpu_buffer.restype = None
        lib.write_to_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        lib.write_to_gpu_buffer.restype = None
        lib.read_from_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        lib.read_from_gpu_buffer.restype = None
        lib.map_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.map_gpu_buffer.restype = ctypes.c_void_p
        lib.unmap_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.unmap_gpu_buffer.restype = None
        lib.copy_gpu_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        lib.copy_gpu_buffer.restype = None
        lib.run_aot_graph.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(_DynamicArg), ctypes.c_int]
        lib.run_aot_graph.restype = None
        lib.sync_runtime.argtypes = [ctypes.c_void_p]
        lib.sync_runtime.restype = None
        lib.clear_pipeline.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.clear_pipeline.restype = None
        if hasattr(lib, "clear_pipeline_for_engine"):
            lib.clear_pipeline_for_engine.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            lib.clear_pipeline_for_engine.restype = None
        lib.add_to_pipeline.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(_DynamicArg), ctypes.c_int]
        lib.add_to_pipeline.restype = None
        lib.run_pipeline.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(_DynamicArg), ctypes.c_int]
        lib.run_pipeline.restype = None
        lib.ti_imread_to_gpu.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        lib.ti_imread_to_gpu.restype = ctypes.c_void_p
        lib.ti_imwrite_from_gpu.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.ti_imwrite_from_gpu.restype = ctypes.c_bool

    def _ptr(self, token: int, table: dict[int, Any]) -> ctypes.c_void_p:
        value = table.get(int(token))
        if not value:
            raise IsolatedRuntimeError("unknown logical native handle")
        return ctypes.c_void_p(int(value.value if hasattr(value, "value") else value))

    def _capture_error(self):
        if self.runtime is None or not hasattr(self.lib, "get_last_engine_error"):
            return
        raw = self.lib.get_last_engine_error(self.runtime)
        if raw:
            self.last_error = _bounded_text(raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw)

    def _args(self, items: list[dict[str, Any]]):
        args = (_DynamicArg * len(items))()
        names: list[bytes] = []
        for index, item in enumerate(items):
            name = str(item.get("name", "")).encode("utf-8")[:256]
            names.append(name)
            arg = args[index]
            arg.name = name
            arg.arg_type = int(item.get("arg_type", 0))
            arg.dtype = int(item.get("dtype", 0))
            arg.dim_count = int(item.get("dim_count", 0))
            for j, value in enumerate(item.get("shape", ())[:8]):
                arg.shape[j] = int(value)
            arg.elem_dim_count = int(item.get("elem_dim_count", 0))
            for j, value in enumerate(item.get("elem_shape", ())[:8]):
                arg.elem_shape[j] = int(value)
            arg.is_vector = int(item.get("is_vector", 0))
            arg.vector_dim = int(item.get("vector_dim", 0))
            value = int(item.get("val_u64", 0))
            if arg.arg_type == 0:
                value = int(self._ptr(value, self.buffers).value or 0)
            arg.val_u64 = value
        return args, names

    def init(self):
        self.runtime = self.lib.init_aot_engine(self.arch_id, self.device_id)
        self._capture_error()
        if not self.runtime:
            raise IsolatedRuntimeError(self.last_error or "native isolated runtime initialization failed")
        return {"runtime": 1}

    def dispatch(self, opcode: str, payload: dict[str, Any]) -> dict[str, Any]:
        if opcode == "init":
            return self.init()
        if self.runtime is None and opcode not in {"last_error", "clear_error"}:
            raise IsolatedRuntimeError("native isolated runtime is not initialized")
        lib = self.lib
        if opcode == "runtime_info":
            name = lib.get_runtime_device_name(self.runtime) or b""
            backend = lib.get_runtime_context_backend(self.runtime) if hasattr(lib, "get_runtime_context_backend") else b""
            arch = lib.get_runtime_arch_id(self.runtime) if hasattr(lib, "get_runtime_arch_id") else self.arch_id
            return {"device_name": name.decode(errors="replace") if isinstance(name, bytes) else str(name), "context_backend": backend.decode(errors="replace") if isinstance(backend, bytes) else str(backend or ""), "arch_id": int(arch)}
        if opcode == "last_error":
            self._capture_error()
            return {"error": self.last_error}
        if opcode == "clear_error":
            self.last_error = ""
            if hasattr(lib, "clear_last_engine_error"):
                lib.clear_last_engine_error(self.runtime)
            return {}
        if opcode == "load_module":
            ptr = lib.load_aot_module(self.runtime, str(payload["path"]).encode())
            self._capture_error()
            if not ptr:
                raise IsolatedRuntimeError(self.last_error or "native module load failed")
            token = self.next_module; self.next_module += 1; self.modules[token] = ptr
            return {"module": token}
        if opcode == "destroy_module":
            token = int(payload["module"]); ptr = self._ptr(token, self.modules)
            lib.destroy_aot_module(ptr); self.modules.pop(token, None); self._capture_error(); return {}
        if opcode == "allocate":
            ptr = lib.allocate_gpu_buffer(self.runtime, int(payload["size"]), int(payload["host_accessible"]))
            self._capture_error()
            if not ptr: raise IsolatedRuntimeError(self.last_error or "native allocation failed")
            token = self.next_buffer; self.next_buffer += 1; self.buffers[token] = ptr
            return {"buffer": token}
        if opcode == "free":
            token = int(payload["buffer"]); ptr = self._ptr(token, self.buffers); lib.free_gpu_buffer(self.runtime, ptr); self.buffers.pop(token, None); self._capture_error(); return {}
        if opcode in {"write", "read"}:
            token = int(payload["buffer"]); ptr = self._ptr(token, self.buffers)
            if opcode == "write":
                data = _decode_bytes(payload["data"]); storage = ctypes.create_string_buffer(data); lib.write_to_gpu_buffer(self.runtime, ptr, ctypes.cast(storage, ctypes.c_void_p), len(data)); self._capture_error(); return {}
            size = int(payload["size"]); storage = ctypes.create_string_buffer(size); lib.read_from_gpu_buffer(self.runtime, ptr, ctypes.cast(storage, ctypes.c_void_p), size); self._capture_error(); return {"data": _encode_bytes(ctypes.string_at(storage, size))}
        if opcode in {"write_shm", "read_shm"}:
            token = int(payload["buffer"])
            ptr = self._ptr(token, self.buffers)
            size = int(payload["size"])
            if size < 0 or size > _MAX_MESSAGE_BYTES:
                raise IsolatedRuntimeError("shared-memory transfer exceeds the bounded size limit")
            shm = shared_memory.SharedMemory(name=str(payload["name"]))
            try:
                address = _buffer_address(shm)
                native_ptr = ctypes.cast(address, ctypes.c_void_p)
                if opcode == "write_shm":
                    lib.write_to_gpu_buffer(self.runtime, ptr, native_ptr, size)
                else:
                    lib.read_from_gpu_buffer(self.runtime, ptr, native_ptr, size)
                self._capture_error()
            finally:
                shm.close()
            return {}
        if opcode == "copy":
            lib.copy_gpu_buffer(self.runtime, self._ptr(payload["src"], self.buffers), self._ptr(payload["dst"], self.buffers), int(payload["size"])); self._capture_error(); return {}
        if opcode in {"run_graph", "add_pipeline"}:
            args, names = self._args(payload.get("args", []))
            if opcode == "run_graph": lib.run_aot_graph(self.runtime, self._ptr(payload["module"], self.modules), str(payload["graph"]).encode(), args, len(args))
            else: lib.add_to_pipeline(self._ptr(payload["module"], self.modules), str(payload["pipeline"]).encode(), str(payload["graph"]).encode(), args, len(args))
            self._capture_error(); return {}
        if opcode == "run_pipeline":
            args, names = self._args(payload.get("args", [])); handles = (ctypes.c_uint64 * len(payload.get("handles", [])))()
            for index, token in enumerate(payload.get("handles", [])): handles[index] = int(self._ptr(token, self.buffers).value or 0)
            lib.run_pipeline(self.runtime, str(payload["pipeline"]).encode(), handles, args, len(args)); self._capture_error(); return {}
        if opcode == "clear_pipeline_module": lib.clear_pipeline(self._ptr(payload["module"], self.modules), str(payload["name"]).encode()); self._capture_error(); return {}
        if opcode == "clear_pipeline_runtime":
            if hasattr(lib, "clear_pipeline_for_engine"): lib.clear_pipeline_for_engine(self.runtime, str(payload["name"]).encode())
            else: lib.clear_pipeline(None, str(payload["name"]).encode())
            self._capture_error(); return {}
        if opcode == "sync": lib.sync_runtime(self.runtime); self._capture_error(); return {}
        if opcode == "imread":
            w,h,c,d=(ctypes.c_int(),ctypes.c_int(),ctypes.c_int(),ctypes.c_int()); ptr=lib.ti_imread_to_gpu(self.runtime,str(payload["path"]).encode(),ctypes.byref(w),ctypes.byref(h),ctypes.byref(c),ctypes.byref(d)); self._capture_error()
            if not ptr: raise IsolatedRuntimeError(self.last_error or "native image read failed")
            token=self.next_buffer; self.next_buffer+=1; self.buffers[token]=ptr; return {"buffer":token,"width":w.value,"height":h.value,"channels":c.value,"depth":d.value}
        if opcode == "imwrite":
            ok=lib.ti_imwrite_from_gpu(self.runtime,str(payload["path"]).encode(),self._ptr(payload["buffer"],self.buffers),int(payload["width"]),int(payload["height"]),int(payload["channels"]),int(payload["depth"])); self._capture_error()
            if not ok: raise IsolatedRuntimeError(self.last_error or "native image write failed")
            return {"ok":True}
        if opcode == "destroy":
            for ptr in list(self.modules.values()):
                try: lib.destroy_aot_module(ptr)
                except Exception: pass
            self.modules.clear()
            try: lib.sync_runtime(self.runtime)
            except Exception: pass
            if self.runtime:
                lib.destroy_aot_engine(self.runtime)
            self.runtime=None; self.buffers.clear(); return {}
        raise IsolatedRuntimeError(f"unknown isolated runtime opcode: {opcode}")


def _worker_main() -> int:
    worker = None
    for line in sys.stdin:
        if len(line.encode("utf-8", errors="ignore")) > _MAX_MESSAGE_BYTES:
            continue
        try:
            request = json.loads(line)
            if int(request.get("protocol", -1)) != _PROTOCOL_VERSION:
                raise IsolatedRuntimeError("unsupported isolated runtime protocol")
            opcode = str(request.get("opcode", "")); payload = dict(request.get("payload") or {})
            if opcode == "init":
                worker = _Worker(payload["bridge"], payload["backend"], int(payload["arch_id"]), int(payload["device_id"]))
            if worker is None:
                raise IsolatedRuntimeError("worker is not initialized")
            result = worker.dispatch(opcode, payload)
            response = {"protocol":_PROTOCOL_VERSION,"request_id":int(request.get("request_id",-1)),"ok":True,"result":result}
        except Exception as exc:
            response = {"protocol":_PROTOCOL_VERSION,"request_id":int(request.get("request_id",-1)) if isinstance(locals().get("request"),dict) else -1,"ok":False,"error":_bounded_text(exc)}
        print(json.dumps(response, separators=(",", ":")), flush=True)
        if opcode == "destroy":
            break
    return 0


if __name__ == "__main__" and "--worker" in sys.argv:
    raise SystemExit(_worker_main())
