// =============================================================================
// audio_engine.h — LESS Real-Time Audio Pipeline
// =============================================================================
// Full-duplex Oboe/AAudio engine for Meta Ray-Ban Gen 2 smart glasses.
//
// Architecture:
//   BLE Mic Input → Oboe MMAP (Exclusive) → DSP Callback → ML Inference
//   → Oboe MMAP Output → BLE Speaker
//
// Key constraints:
//   - PerformanceMode::LowLatency — request kernel MMAP fast path
//   - SharingMode::Exclusive — bypass AudioFlinger mixer entirely
//   - Usage::Game — hint OS for priority BT LC3 packet dispatch
//   - Native sample rate (no resampling) — typically 48000 Hz
//   - Non-blocking callback — no allocations, no locks, no syscalls
// =============================================================================

#pragma once

#include <oboe/Oboe.h>
#include <atomic>
#include <memory>
#include <string>

#include "noise_suppressor.h"
#include "masking_engine.h"
#include "gsc_filter.h"
#include "alert_detector.h"
#include "environment_classifier.h"
#include "vision_synth.h"

namespace less {

// Processing modes — switch between DSP strategies
enum class ProcessingMode : int32_t {
    kVoiceIsolate = 0,  // DTLN neural net strips background, preserves speech
    kComfortMask  = 1,  // Psychoacoustic mask over detected tonal noise
    kVisionMusic  = 2   // Generative music driven by camera scene analysis
};

// Engine states — atomic transitions only, no mutex in the audio path
enum class EngineState {
    kStopped,
    kStarting,
    kRunning,
    kPaused,    // Phase 3: glasses doffed/folded — streams paused, state preserved
    kStopping,
    kError
};

class LessAudioEngine : public oboe::AudioStreamDataCallback,
                         public oboe::AudioStreamErrorCallback {
public:
    LessAudioEngine();
    ~LessAudioEngine();

    // Lifecycle — called from JNI on the main/UI thread
    bool start();
    void stop();

    // Phase 3: Session lifecycle — called when glasses are doffed/folded/donned
    // Pauses/resumes Oboe streams without full teardown to prevent buffer underruns
    void pause();
    void resume();

    // Set the .tflite model path before calling start()
    void setModelPath(const char* path);

    // Phase 4: Set the QNN shared library directory for NPU delegation
    // Path to the directory containing libQnnHtp.so, libQnnTFLiteDelegate.so, etc.
    void setQnnLibDir(const char* path);

    bool isRunning() const;
    
    // Phase 9: Lock-free stream error polling
    int getStreamErrorStatus() const;

    // Query the active delegate backend
    const char* getActiveBackendName() const;

    // Runtime tuning — safe to call from any thread
    void setSuppressionLevel(float level);  // 0.0 = passthrough, 1.0 = max suppression

    // Latency reporting
    double getLatencyMs() const;

    // Phase 2: hot-swap adapter weights from overnight training
    bool reloadAdapterWeights(const char* adapterPath);

    // =========================================================================
    // Phase 7 + 8.1: Thermal Management — ADPF Integration
    // =========================================================================
    // These delegate to NoiseSuppressor's ThermalManager.
    // Call initializeThermalManager() once after start().
    // On API 36+, the callback drives updates automatically.
    // On API 31-35, call pollThermalHeadroom() periodically from monitoring thread.

    bool initializeThermalManager();
    float pollThermalHeadroom(int32_t forecastSeconds = 10);
    int32_t getThermalState() const;
    float getThermalHeadroom() const;
    float getForecastHeadroom() const;
    bool isThermalMonitoringAvailable() const;
    bool isThermalCallbackActive() const;

    // Phase 8: Audio level snapshot for UI waveform visualizer
    void getAudioLevels(float* outBuffer, int32_t bufferSize) const;

    // =========================================================================
    // Phase 11: Psychoacoustic Masking + Environment Awareness
    // =========================================================================
    void setProcessingMode(int32_t mode);
    int32_t getProcessingMode() const;

    // Mask control
    bool isMaskActive() const;
    void setMaskTexture(int32_t texture);  // 0=brown, 1=pink, 2=white, 3=nature
    int32_t getMaskTexture() const;

    // Phase 15: Vision-to-Music
    void updateMusicalParameters(float density, float valence,
                                 float arousal, float timbre);
    void applyNeuralData(const float* data, int32_t len);
    void setMusicInterpretation(int32_t interpretation);
    int32_t getMusicInterpretation() const;
    float getMusicBpm() const;
    int32_t getMusicKey() const;
    void setSynthQuality(int32_t quality);
    int32_t getSynthQuality() const;

    // Phase 19: Synesthesia and Transitions
    void updateSynesthesiaParams(float hue, float saturation, float value);
    void flushGenerativeState();

    // Environment
    int32_t getDetectedEnvironment() const;
    const char* getDetectedEnvironmentName() const;

    // Alert
    bool isAlertActive() const;

    // =========================================================================
    // oboe::AudioStreamDataCallback — THE hot loop
    // =========================================================================
    // Called on a real-time priority thread. MUST be non-blocking.
    // No allocations, no locks, no JNI calls, no logging.
    oboe::DataCallbackResult onAudioReady(
        oboe::AudioStream* outputStream,
        void* audioData,
        int32_t numFrames
    ) override;

    // =========================================================================
    // oboe::AudioStreamErrorCallback — recovery
    // =========================================================================
    void onErrorAfterClose(
        oboe::AudioStream* stream,
        oboe::Result error
    ) override;

private:
    // Stream builders — configure the Oboe pipeline
    oboe::AudioStreamBuilder createInputStreamBuilder();
    oboe::AudioStreamBuilder createOutputStreamBuilder();

    // Oboe stream handles
    std::shared_ptr<oboe::AudioStream> mInputStream;
    std::shared_ptr<oboe::AudioStream> mOutputStream;

    // DSP processor — pluggable noise suppression
    std::unique_ptr<NoiseSuppressor> mSuppressor;

    // Phase 11: Psychoacoustic masking components
    std::unique_ptr<MaskingEngine> mMaskEngine;
    std::unique_ptr<GscFilter> mGscFilter;
    std::unique_ptr<AlertDetector> mAlertDetector;

    // Phase 15: Vision-to-Music generative synthesizer
    std::unique_ptr<VisionSynth> mVisionSynth;
    std::unique_ptr<EnvironmentClassifier> mEnvClassifier;

    // Intermediate buffer for reading input frames
    // Sized to the output stream's framesPerCallback
    std::vector<float> mInputBuffer;

    // Phase 11: Pre-allocated intermediate buffers
    std::vector<float> mMaskBuffer;   // Mask generator output
    std::vector<float> mCleanBuffer;  // GSC-cleaned input (ambient only)
    std::vector<float> mPrevMaskBuffer; // Previous frame's mask for GSC reference

    // Engine state machine — atomic, no locks
    std::atomic<EngineState> mState{EngineState::kStopped};

    // Runtime-tunable suppression level
    std::atomic<float> mSuppressionLevel{0.7f};

    // Phase 11: Processing mode — 0=VoiceIsolate, 1=ComfortMask
    std::atomic<int32_t> mProcessingMode{0};

    // Measured latency (updated each callback for reporting)
    std::atomic<double> mMeasuredLatencyMs{0.0};

    // Phase 9: Stream error status for lock-free Kotlin polling
    // 0 = OK, non-zero = oboe::Result error code
    std::atomic<int> mStreamErrorStatus{0};

    // Detected native sample rate from the device
    int32_t mSampleRate{0};

    // Frames-per-callback (dictated by Oboe for lowest latency)
    int32_t mFramesPerCallback{0};

    // Phase 11: Environment classification counter (classify every ~500ms)
    int32_t mEnvClassifyCounter{0};
    int32_t mEnvClassifyInterval{0};  // Set from sample rate

    // Path to the DTLN .tflite model (set via setModelPath before start())
    std::string mModelPath;

    // Phase 4: Path to QNN shared libs (set via setQnnLibDir before start())
    std::string mQnnLibDir;

    // =========================================================================
    // Phase 8: Waveform Snapshot Buffer (Lock-Free Double Buffer)
    // =========================================================================
    // Moved from NoiseSuppressor so visualizer works across ALL processing modes.
    static constexpr int32_t kWaveformSamples = 128;

    float mInputWaveform[2][kWaveformSamples]{};
    float mOutputWaveform[2][kWaveformSamples]{};
    std::atomic<float> mInputRms{0.0f};
    std::atomic<float> mOutputRms{0.0f};
    std::atomic<int32_t> mWaveformWriteIndex{0};
};

} // namespace less
