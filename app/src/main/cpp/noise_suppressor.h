// =============================================================================
// noise_suppressor.h — LiteRT Inference with Qualcomm NPU Acceleration
// =============================================================================
// Production noise suppression using the LiteRT C API with the Qualcomm AI
// Engine Direct Delegate (QNN HTP backend) for NPU offload.
//
// Phase 7 Upgrade: ADPF Thermal-Aware Dynamic Backend Switching
// ==============================================================
// Integrates the ThermalManager to dynamically switch between NPU, CPU,
// and spectral gate backends based on the device's predicted thermal
// headroom. This prevents OS-level thermal throttling which would cause
// audio glitches and dropped frames.
//
// Phase 4 Upgrade: Qualcomm NPU (HTP) Delegation
// ================================================
// The Snapdragon HTP (Hexagon Tensor Processor) is a dedicated AI compute
// unit that runs ML inference 3-5× faster than CPU XNNPACK while consuming
// ~60% less power. This is critical for:
//
//   1. Thermal budget — open-ear glasses heat up fast; NPU is thermally isolated
//   2. Battery life  — NPU consumes ~0.5W vs ~2W for CPU neural inference
//   3. Latency       — HTP runs DTLN in ~0.4ms vs ~1.2ms on CPU (XNNPACK)
//   4. CPU headroom  — frees the Cortex-A cores for audio I/O and UI
//
// Delegation hierarchy (fallback chain):
//   1. QNN HTP (NPU) — preferred, lowest latency and power
//   2. QNN GPU        — fallback if HTP rejects ops
//   3. XNNPACK (CPU)  — universal fallback
//   4. Spectral Gate  — if no .tflite model at all
//
// Model architecture: DTLN (Dual-Signal Transformation LSTM Network)
//   - Two-stage: magnitude STFT → LSTM → mask → iSTFT
//   - Causal (no future lookahead) — suitable for real-time
//   - ~2MB quantized .tflite model
//
// Contract:
//   - process() MUST complete in < 3ms for 480-sample frames @ 48kHz
//   - process() MUST be non-blocking: ZERO allocations, ZERO locks, ZERO I/O
//   - All memory (tensors, buffers) pre-allocated in initialize()
//   - TfLiteInterpreterInvoke() is the only heavy operation in the hot path
//
// Adapter Hot-Swap (Phase 2):
//   loadAdapterWeights() is called from the JNI thread (NOT the audio thread).
//   It atomically updates the adapter weight pointer so the next process()
//   call picks up the new weights without any lock contention.
// =============================================================================

#pragma once

#include <cstdint>
#include <atomic>
#include <vector>
#include <memory>
#include <string>

#include "thermal_manager.h"

// LiteRT (TensorFlow Lite) C API — header-only, no C++ runtime dependency
// This gives us maximum control over memory layout and avoids hidden allocations
// that the C++ Interpreter class can trigger.
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_types.h"

namespace less {

// Delegate backend used for inference — reported to Kotlin for diagnostics
enum class DelegateBackend {
    kNone,        // No model loaded (spectral gate fallback)
    kCpu,         // XNNPACK on ARM64 Cortex-A cores
    kGpu,         // QNN GPU delegate (Adreno)
    kNpu          // QNN HTP delegate (Hexagon NPU) — preferred
};

// LoRA adapter weight storage — unchanged from prototype
struct AdapterWeights {
    static constexpr int32_t kHiddenDim = 256;
    static constexpr int32_t kRank = 8;
    static constexpr int32_t kTotalParams = 2 * kHiddenDim * kRank;  // 4096

    float down[kHiddenDim * kRank];  // encoder → bottleneck
    float up[kRank * kHiddenDim];    // bottleneck → decoder

    AdapterWeights() {
        for (int i = 0; i < kHiddenDim * kRank; ++i) {
            down[i] = (i % (kRank + 1) == 0) ? 0.01f : 0.0f;
        }
        for (int i = 0; i < kRank * kHiddenDim; ++i) {
            up[i] = (i % (kHiddenDim + 1) == 0) ? 0.01f : 0.0f;
        }
    }
};

class NoiseSuppressor {
public:
    NoiseSuppressor();
    ~NoiseSuppressor();

    // Non-copyable, non-movable (owns TFLite resources)
    NoiseSuppressor(const NoiseSuppressor&) = delete;
    NoiseSuppressor& operator=(const NoiseSuppressor&) = delete;

    // =========================================================================
    // Initialization — allocates ALL memory upfront
    // =========================================================================
    // @param sampleRate   Device native sample rate (expected 48000)
    // @param modelPath    Absolute path to the DTLN .tflite model
    //                     If null/empty, falls back to spectral gate stub
    // @param qnnLibDir    Path to QNN shared libs (libQnnHtp.so, etc.)
    //                     If null/empty, skips NPU delegation (CPU fallback)
    // @return true if model loaded and tensors allocated successfully
    bool initialize(
        int32_t sampleRate,
        const char* modelPath = nullptr,
        const char* qnnLibDir = nullptr
    );

    // =========================================================================
    // The core DSP function — called from the real-time audio thread
    // =========================================================================
    // CRITICAL: This function is ALLOCATION-FREE and LOCK-FREE.
    // The only significant work is TfLiteInterpreterInvoke(), which:
    //   - Uses pre-allocated tensor buffers (no malloc)
    //   - Runs on the calling thread (no thread sync)
    //   - Executes deterministic compute (no branching on I/O)
    //
    // @param input      Raw mic samples (float, mono)
    // @param output     Processed output samples (float, mono)
    // @param numFrames  Number of samples in this callback
    // @param level      Suppression intensity [0.0 = passthrough, 1.0 = max]
    void process(
        const float* input,
        float* output,
        int32_t numFrames,
        float level
    );

    // =========================================================================
    // Adapter hot-swap — called from JNI thread, NOT the audio thread
    // =========================================================================
    bool loadAdapterWeights(const char* path);

    // Query whether a TFLite model is loaded (vs spectral gate fallback)
    bool isModelLoaded() const { return mModelLoaded; }

    // Query which delegate backend is active
    DelegateBackend getActiveBackend() const { return mActiveBackend; }

    // Human-readable delegate name (for logging / dashboard)
    static const char* backendName(DelegateBackend backend);

    // =========================================================================
    // Phase 7: ADPF Thermal Management
    // =========================================================================
    // The thermal manager runs on a separate monitoring thread.
    // Call pollThermalHeadroom() periodically (e.g., every 2 seconds) from
    // a non-audio thread. On API 36+ (callback mode), this is a no-op that
    // returns the latest callback-provided headroom — the OS drives updates.
    // On API 31-35, this polls AThermal_getThermalHeadroom().
    //
    // The process() function reads the thermal state atomically and switches
    // to the appropriate backend.

    /// Initialize the thermal manager (call once after initialize())
    bool initializeThermalManager();

    /// Poll ADPF thermal headroom — no-op in callback mode (API 36+)
    float pollThermalHeadroom(int32_t forecastSeconds = 10);

    /// Get current thermal state for dashboard display
    ThermalState getThermalState() const;

    /// Get raw thermal headroom (0.0=cold, 1.0=throttling)
    float getThermalHeadroom() const;

    /// Get forecast headroom from callback (API 36+ only)
    float getForecastHeadroom() const;

    /// Whether ADPF is available on this device
    bool isThermalMonitoringAvailable() const;

    /// Whether the async callback path is active (API 36+)
    bool isThermalCallbackActive() const;

private:
    int32_t mSampleRate{0};
    bool mInitialized{false};
    bool mModelLoaded{false};
    DelegateBackend mActiveBackend{DelegateBackend::kNone};

    // Phase 7: The backend the model was INITIALIZED with (before thermal override)
    DelegateBackend mInitializedBackend{DelegateBackend::kNone};
    // Phase 7: The backend currently ACTIVE (may differ from initialized due to thermal)
    // This is set atomically to allow lock-free read from the audio thread
    std::atomic<int32_t> mEffectiveBackend{static_cast<int32_t>(DelegateBackend::kNone)};
    // Phase 7: Thermal-forced fallback to spectral gate
    std::atomic<bool> mThermalForceFallback{false};

    // =========================================================================
    // LiteRT (TensorFlow Lite) C API handles
    // =========================================================================
    TfLiteModel* mModel{nullptr};
    TfLiteInterpreterOptions* mOptions{nullptr};
    TfLiteInterpreter* mInterpreter{nullptr};

    // QNN delegate handle — must be kept alive until interpreter is destroyed
    // Stored as opaque pointer since we load QNN via dlopen()
    void* mQnnDelegateHandle{nullptr};
    TfLiteOpaqueDelegate* mQnnDelegate{nullptr};

    // Pre-resolved tensor pointers — avoid per-frame lookup
    float* mInputTensorData{nullptr};
    float* mOutputTensorData{nullptr};
    int32_t mInputTensorSize{0};
    int32_t mOutputTensorSize{0};

    // LSTM hidden state tensors (if model uses stateful RNNs)
    float* mLstmState1{nullptr};
    float* mLstmState2{nullptr};

    // =========================================================================
    // Frame buffering for model input alignment
    // =========================================================================
    static constexpr int32_t kModelFrameSize = 512;   // DTLN input size
    static constexpr int32_t kModelHopSize = 256;      // 50% overlap
    static constexpr int32_t kNumBins = kModelFrameSize / 2 + 1;

    // Pre-allocated frame buffers (no allocation in process())
    std::vector<float> mWindow;
    std::vector<float> mRingBuffer;
    std::vector<float> mOverlapBuffer;
    int32_t mRingWritePos{0};
    int32_t mRingReadPos{0};
    int32_t mFramesAccumulated{0};

    // =========================================================================
    // Spectral gate fallback (used when no .tflite model available)
    // =========================================================================
    bool mUseFallback{false};
    std::vector<float> mNoiseFloor;
    float mNoiseAlpha{0.98f};
    bool mNoiseEstimateReady{false};
    int32_t mNoiseEstimateFrames{0};
    static constexpr int32_t kNoiseEstimatePeriod = 50;

    // =========================================================================
    // Adapter Weights (Phase 2)
    // =========================================================================
    std::shared_ptr<AdapterWeights> mActiveAdapter;
    std::atomic<bool> mAdapterLoaded{false};

    // =========================================================================
    // Internal methods
    // =========================================================================
    void computeHannWindow();
    void processModelFrame(const float* windowedInput, float* output);
    void processSpectralGate(const float* windowedInput, float* output, float level);
    void applyAdapter(float* spectrum, int32_t numBins);
    void destroyModel();

    // Phase 4: Delegate setup
    bool tryAttachQnnDelegate(const char* qnnLibDir);
    bool tryAttachXnnpackDelegate();

    // Benchmark: measure single-frame inference latency (called once at init)
    double benchmarkInferenceMs();

    // =========================================================================
    // Phase 8: Waveform Snapshot Buffer (Lock-Free Double Buffer)
    // =========================================================================
    // The audio thread writes waveform data into one buffer while the UI
    // thread reads from the other. An atomic index controls which is active.
    // This ensures ZERO lock contention on the real-time audio path.
    static constexpr int32_t kWaveformSamples = 64;

    float mInputWaveform[2][kWaveformSamples]{};
    float mOutputWaveform[2][kWaveformSamples]{};
    std::atomic<float> mInputRms{0.0f};
    std::atomic<float> mOutputRms{0.0f};
    std::atomic<int32_t> mWaveformWriteIndex{0};  // audio thread writes to this index
    // UI thread reads from (1 - mWaveformWriteIndex)

public:
    /// Called from JNI monitoring thread — returns [inputRms, outputRms,
    /// input[64], output[64]] = 130 floats total
    void getAudioLevels(float* outBuffer, int32_t bufferSize) const;

private:
    // =========================================================================
    // Phase 7: Thermal Manager
    // =========================================================================
    ThermalManager mThermalManager;
};

} // namespace less
