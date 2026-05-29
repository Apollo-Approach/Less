# QNN SDK Native Libraries — Drop Zone

Place Qualcomm AI Engine Direct SDK shared libraries here for NPU acceleration.

## Required Libraries

Download from the [Qualcomm AI Engine Direct SDK](https://www.qualcomm.com/developer/software/neural-processing-sdk):

| File | Purpose |
|------|---------|
| `libQnnTFLiteDelegate.so` | LiteRT → QNN bridge delegate |
| `libQnnHtp.so` | HTP (Hexagon Tensor Processor / NPU) backend |
| `libQnnHtpPrepare.so` | HTP graph preparation & compilation |
| `libQnnHtpV73Stub.so` | HTP v73 skeleton (SoC-specific — see below) |
| `libQnnSystem.so` | QNN system interface |

## Optional Libraries

| File | Purpose |
|------|---------|
| `libQnnGpu.so` | Adreno GPU backend (fallback before XNNPACK CPU) |

## SoC-Specific Stubs

The `libQnnHtpVxxStub.so` file must match the target Snapdragon chipset:

- **V73**: Snapdragon 8 Gen 2 (SM8550)
- **V75**: Snapdragon 8 Gen 3 (SM8650)
- **V68**: Snapdragon 888 / 8 Gen 1

## How the Delegate Chain Works

1. The C++ `NoiseSuppressor::initialize()` attempts `dlopen("libQnnTFLiteDelegate.so")`
2. If found → creates QNN delegate → attaches to LiteRT interpreter
3. HTP backend runs inference on the Hexagon DSP/NPU (sub-0.5ms)
4. If QNN unavailable → falls back to GPU (Adreno) → CPU (XNNPACK)
5. The active backend is reported via `getActiveBackend()` JNI call

## Note

These libraries are **not** open source. They are proprietary Qualcomm binaries
distributed under the Qualcomm AI Engine Direct SDK license. They must be
obtained from the Qualcomm developer portal and are NOT committed to git.
