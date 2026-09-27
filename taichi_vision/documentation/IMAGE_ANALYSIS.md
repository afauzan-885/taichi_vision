# Image Analysis

Image analysis provides small, bounded measurements for image and burst
selection. The public facade is `taichi_vision.image_analysis`; callers do not
choose a backend or pass compiler/runtime options to each analysis function.
Taichi Vision uses the application's already-selected target and its
target-qualified AOT artifact.

## Exposure analysis

```python
from taichi_vision.image_analysis import analyze_exposure

result = analyze_exposure(linear_rgb)
print(result.score, result.median_luminance)
```

`linear_rgb` must be a floating-point RGB NumPy array with shape
`(height, width, 3)` and values in `[0, 1]`. Integer images and sRGB-encoded
images must be converted by the caller first. Only the deterministic uniform
sample is checked for finite, in-range values; this avoids scanning an
additional full-resolution array.

The result is an `ExposureAnalysis` with:

- `score`: exposure suitability heuristic in `[0, 1]` used to compare frames;
- `median_luminance`: median of sampled linear luminance;
- `exposure_suitability` and `usable_fraction`: score components;
- `sample_count`: number of pixels included.

The analyzer samples at most 65,536 pixels from a frame and uses 256 histogram
bins. The AOT input is a fixed 262,144-byte `float32` array; the histogram is a
1,024-byte `int32` array. These accelerator buffers do not scale with image
resolution. The source frame and small CPU preprocessing temporaries still
occupy host RAM. Sampling makes large-frame scores approximate; this is a
ranking aid, not a calibrated exposure value or a guarantee of image quality.

If the active target has no compatible `exposure_analyzer` artifact, the call
raises the AOT loader error. It does not run a hidden CPU histogram fallback.
The current artifact has compile evidence for Taichi 1.7.4 / LLVM 20.1.5 on
Windows x86-64 Vulkan. The graph has not yet received runtime, parity, or
application validation, so exposure analysis remains **EXPERIMENTAL**.

## Design rules for additional analyzers

The family is intended to grow through independent calls with small result
objects and no per-call backend configuration. Planned analysis areas are:

- **Sharpness and focus:** report edge/detail sharpness with noise-aware
  confidence. A noise estimate must prevent full-noise images from scoring as
  highly detailed; burst scoring should account for exposure and compare
  corresponding scene regions when alignment is available.
- **Scene context:** estimate indoor/outdoor or related scene cues without
  using overall luminance as the deciding signal. A semantic classifier needs
  an explicit model, training/evaluation set, confidence and an `unknown`
  outcome; a luminance or color heuristic alone must not be called image
  understanding.
- **Burst motion and shake:** use a small coarse representation and
  exposure-resistant structure to separate global camera movement from local
  motion, scene changes, and low-confidence matches. Single-frame analysis
  should report only blur/shake cues it can observe, not infer temporal motion.

Every implementation should bound sample dimensions, scratch buffers, and
number of retained frames; reuse the selected Taichi AOT target; avoid full
resolution copies; and return confidence/validity with heuristic scores.
Memory, latency, robustness under low/high noise and exposure variation, and
runtime backend evidence must be measured before an analyzer is marked
qualified. These additional analyzers are design targets, not implemented APIs.
