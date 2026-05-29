// =============================================================================
// audio_engine.cpp — LESS Real-Time Audio Pipeline Implementation
// =============================================================================

#include "audio_engine.h"

#include <android/log.h>
#include <cstring>

#define LOG_TAG "LESS_Engine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace less {

// =============================================================================
// Construction / Destruction
// =============================================================================

LessAudioEngine::LessAudioEngine() {
    mSuppressor = std::make_unique<NoiseSuppressor>();
    mMaskEngine = std::make_unique<MaskingEngine>();
    mGscFilter = std::make_unique<GscFilter>();
    mAlertDetector = std::make_unique<AlertDetector>();
    mEnvClassifier = std::make_unique<EnvironmentClassifier>();
    mVisionSynth = std::make_unique<VisionSynth>();
    LOGI("LessAudioEngine created (with Phase 11 masking + Phase 15 VisionSynth)");
}

LessAudioEngine::~LessAudioEngine() {
    stop();
    LOGI("LessAudioEngine destroyed");
}

// =============================================================================
// Stream Configuration
// =============================================================================

oboe::AudioStreamBuilder LessAudioEngine::createInputStreamBuilder() {
    oboe::AudioStreamBuilder builder;
    builder.setDirection(oboe::Direction::Input)
           ->setPerformanceMode(oboe::PerformanceMode::LowLatency)
           ->setSharingMode(oboe::SharingMode::Exclusive)
           ->setFormat(oboe::AudioFormat::Float)
           ->setChannelCount(oboe::ChannelCount::Mono)
           // Do NOT set sample rate — inherit device native rate
           // to avoid expensive software resampling
           ->setInputPreset(oboe::InputPreset::Unprocessed)
           // Unprocessed: bypass Android's built-in AEC/NS/AGC
           // We handle all DSP ourselves
           ->setErrorCallback(this);

    return builder;
}

oboe::AudioStreamBuilder LessAudioEngine::createOutputStreamBuilder() {
    oboe::AudioStreamBuilder builder;
    builder.setDirection(oboe::Direction::Output)
           ->setPerformanceMode(oboe::PerformanceMode::LowLatency)
           ->setSharingMode(oboe::SharingMode::Exclusive)
           ->setFormat(oboe::AudioFormat::Float)
           ->setChannelCount(oboe::ChannelCount::Mono)
           // Match the input stream's sample rate exactly
           ->setSampleRate(mSampleRate)
           // AAUDIO_USAGE_GAME — forces OS to prioritize low-latency
           // Bluetooth LC3 packet dispatch
           ->setUsage(oboe::Usage::Game)
           ->setDataCallback(this)
           ->setErrorCallback(this);

    return builder;
}

// =============================================================================
// Lifecycle
// =============================================================================

bool LessAudioEngine::start() {
    EngineState expected = EngineState::kStopped;
    if (!mState.compare_exchange_strong(expected, EngineState::kStarting)) {
        LOGW("start() called but engine is not stopped (state=%d)",
             static_cast<int>(mState.load()));
        return false;
    }

    LOGI("Starting audio engine...");
    mStreamErrorStatus.store(0, std::memory_order_relaxed);

    // --- Open input stream first to discover native sample rate ---
    auto inputBuilder = createInputStreamBuilder();
    oboe::Result result = inputBuilder.openStream(mInputStream);
    if (result != oboe::Result::OK) {
        LOGE("Failed to open input stream: %s", oboe::convertToText(result));
        mState.store(EngineState::kError);
        return false;
    }

    // Capture the native sample rate — this is what the BLE Audio device reports
    mSampleRate = mInputStream->getSampleRate();
    mFramesPerCallback = mInputStream->getFramesPerBurst();

    LOGI("Input stream opened: sampleRate=%d, framesPerBurst=%d, "
         "performanceMode=%d, sharingMode=%d, bufferCapacity=%d",
         mSampleRate,
         mFramesPerCallback,
         static_cast<int>(mInputStream->getPerformanceMode()),
         static_cast<int>(mInputStream->getSharingMode()),
         mInputStream->getBufferCapacityInFrames());

    // Log whether we got MMAP (exclusive) or fell back to legacy
    if (mInputStream->getSharingMode() == oboe::SharingMode::Exclusive) {
        LOGI("✓ Input stream: MMAP Exclusive mode acquired");
    } else {
        LOGW("⚠ Input stream: fell back to Shared mode (AudioFlinger path)");
    }

    // --- Open output stream with matched sample rate ---
    auto outputBuilder = createOutputStreamBuilder();
    result = outputBuilder.openStream(mOutputStream);
    if (result != oboe::Result::OK) {
        LOGE("Failed to open output stream: %s", oboe::convertToText(result));
        mInputStream->close();
        mInputStream.reset();
        mState.store(EngineState::kError);
        return false;
    }

    LOGI("Output stream opened: sampleRate=%d, framesPerBurst=%d, "
         "performanceMode=%d, sharingMode=%d",
         mOutputStream->getSampleRate(),
         mOutputStream->getFramesPerBurst(),
         static_cast<int>(mOutputStream->getPerformanceMode()),
         static_cast<int>(mOutputStream->getSharingMode()));

    if (mOutputStream->getSharingMode() == oboe::SharingMode::Exclusive) {
        LOGI("✓ Output stream: MMAP Exclusive mode acquired");
    } else {
        LOGW("⚠ Output stream: fell back to Shared mode (AudioFlinger path)");
    }

    // --- Initialize the noise suppressor with the discovered sample rate ---
    // Pass the model path to load the DTLN .tflite model for ML inference.
    // If mModelPath is empty, the suppressor falls back to spectral gate.
    // Phase 4: Also pass QNN lib directory for NPU acceleration.
    mSuppressor->initialize(
        mSampleRate,
        mModelPath.empty() ? nullptr : mModelPath.c_str(),
        mQnnLibDir.empty() ? nullptr : mQnnLibDir.c_str()
    );

    LOGI("Active backend: %s",
         NoiseSuppressor::backendName(mSuppressor->getActiveBackend()));

    // --- Allocate the intermediate input read buffer ---
    // Size to the maximum burst the output callback might request
    int32_t maxFrames = mOutputStream->getBufferCapacityInFrames();
    mInputBuffer.resize(maxFrames, 0.0f);

    // --- Phase 11: Initialize masking components ---
    mMaskEngine->initialize(mSampleRate);
    mGscFilter->initialize(mSampleRate);
    mAlertDetector->initialize(mSampleRate);
    mEnvClassifier->initialize(mSampleRate);

    // --- Phase 15: Initialize Vision-to-Music synthesizer ---
    mVisionSynth->initialize(mSampleRate);

    // Pre-allocate Phase 11 intermediate buffers
    mMaskBuffer.resize(maxFrames, 0.0f);
    mCleanBuffer.resize(maxFrames, 0.0f);
    mPrevMaskBuffer.resize(maxFrames, 0.0f);

    // Environment classification every ~500ms
    mEnvClassifyInterval = mSampleRate / 2;  // ~500ms worth of samples
    mEnvClassifyCounter = 0;

    LOGI("Phase 11 masking + Phase 15 VisionSynth initialized");

    // --- Start both streams ---
    // Input must start before output to ensure data is available
    // when the first output callback fires
    result = mInputStream->requestStart();
    if (result != oboe::Result::OK) {
        LOGE("Failed to start input stream: %s", oboe::convertToText(result));
        mInputStream->close();
        mOutputStream->close();
        mInputStream.reset();
        mOutputStream.reset();
        mState.store(EngineState::kError);
        return false;
    }

    result = mOutputStream->requestStart();
    if (result != oboe::Result::OK) {
        LOGE("Failed to start output stream: %s", oboe::convertToText(result));
        mInputStream->requestStop();
        mInputStream->close();
        mOutputStream->close();
        mInputStream.reset();
        mOutputStream.reset();
        mState.store(EngineState::kError);
        return false;
    }

    mState.store(EngineState::kRunning);
    LOGI("✓ Audio engine running — full duplex active");
    return true;
}

// =============================================================================
// Phase 3: Session Lifecycle — Pause/Resume for doff/fold safety
// =============================================================================
// When the glasses are doffed or folded, BLE audio data stops arriving.
// If the Oboe callback continues firing, it reads stale/zero data from the
// input stream, causing audible glitches and potential underruns.
//
// pause() stops the streams WITHOUT destroying them or the DSP state.
// resume() restarts them from the preserved configuration.
// This is significantly faster than a full stop()/start() cycle.

void LessAudioEngine::pause() {
    EngineState expected = EngineState::kRunning;
    if (!mState.compare_exchange_strong(expected, EngineState::kPaused)) {
        LOGW("pause() called but engine is not running (state=%d)",
             static_cast<int>(mState.load()));
        return;
    }

    LOGI("Pausing audio engine (glasses doffed/folded)...");

    // Pause both streams — they remain open but stop firing callbacks
    if (mOutputStream) {
        mOutputStream->requestPause();
    }
    if (mInputStream) {
        mInputStream->requestPause();
    }

    LOGI("✓ Audio engine paused — streams suspended, DSP state preserved");
}

void LessAudioEngine::resume() {
    EngineState expected = EngineState::kPaused;
    if (!mState.compare_exchange_strong(expected, EngineState::kRunning)) {
        // If we're stopped (not paused), do a full start instead
        if (mState.load() == EngineState::kStopped) {
            LOGI("resume() from stopped state — performing full start");
            start();
            return;
        }
        LOGW("resume() called but engine is not paused (state=%d)",
             static_cast<int>(mState.load()));
        return;
    }

    LOGI("Resuming audio engine (glasses donned)...");

    // Restart input first to ensure data is flowing when output callback fires
    if (mInputStream) {
        auto result = mInputStream->requestStart();
        if (result != oboe::Result::OK) {
            LOGE("Failed to resume input stream: %s", oboe::convertToText(result));
            mState.store(EngineState::kError);
            return;
        }
    }

    if (mOutputStream) {
        auto result = mOutputStream->requestStart();
        if (result != oboe::Result::OK) {
            LOGE("Failed to resume output stream: %s", oboe::convertToText(result));
            mState.store(EngineState::kError);
            return;
        }
    }

    LOGI("✓ Audio engine resumed — full duplex re-activated");
}

void LessAudioEngine::stop() {
    EngineState expected = EngineState::kRunning;
    if (!mState.compare_exchange_strong(expected, EngineState::kStopping)) {
        // Also allow stopping from paused state
        expected = EngineState::kPaused;
        if (!mState.compare_exchange_strong(expected, EngineState::kStopping)) {
            return;  // Not running or paused, nothing to do
        }
    }

    LOGI("Stopping audio engine...");

    if (mOutputStream) {
        mOutputStream->requestStop();
        mOutputStream->close();
        mOutputStream.reset();
    }

    if (mInputStream) {
        mInputStream->requestStop();
        mInputStream->close();
        mInputStream.reset();
    }

    mState.store(EngineState::kStopped);
    LOGI("✓ Audio engine stopped");
}

int LessAudioEngine::getStreamErrorStatus() const {
    return mStreamErrorStatus.load(std::memory_order_relaxed);
}

bool LessAudioEngine::isRunning() const {
    return mState.load() == EngineState::kRunning;
}

void LessAudioEngine::setSuppressionLevel(float level) {
    // Clamp to [0.0, 1.0]
    float clamped = (level < 0.0f) ? 0.0f : (level > 1.0f) ? 1.0f : level;
    mSuppressionLevel.store(clamped, std::memory_order_relaxed);
}

double LessAudioEngine::getLatencyMs() const {
    return mMeasuredLatencyMs.load(std::memory_order_relaxed);
}

bool LessAudioEngine::reloadAdapterWeights(const char* adapterPath) {
    if (!mSuppressor) return false;
    LOGI("Hot-swapping adapter weights from: %s", adapterPath);
    return mSuppressor->loadAdapterWeights(adapterPath);
}

void LessAudioEngine::setModelPath(const char* path) {
    mModelPath = path ? path : "";
    LOGI("Model path set: %s", mModelPath.empty() ? "(none — spectral gate)" : mModelPath.c_str());
}

void LessAudioEngine::setQnnLibDir(const char* path) {
    mQnnLibDir = path ? path : "";
    LOGI("QNN lib dir set: %s", mQnnLibDir.empty() ? "(none — CPU fallback)" : mQnnLibDir.c_str());
}

const char* LessAudioEngine::getActiveBackendName() const {
    if (mSuppressor) {
        return NoiseSuppressor::backendName(mSuppressor->getActiveBackend());
    }
    return "Not initialized";
}

// =============================================================================
// THE HOT LOOP — oboe::AudioStreamDataCallback
// =============================================================================
// This runs on a real-time priority thread managed by Oboe/AAudio.
//
// CRITICAL CONSTRAINTS (violating these causes audio glitches):
//   - NO heap allocations (new, malloc, std::vector resize)
//   - NO mutex locks (std::mutex, pthread_mutex)
//   - NO JNI calls
//   - NO file I/O
//   - NO logging (except catastrophic errors)
//   - NO syscalls that might block
//
// The entire function must complete within the frame deadline:
//   deadline = numFrames / sampleRate
//   e.g., 480 frames @ 48kHz = 10ms total budget
//   Our DSP target: ≤3ms inference + <1ms overhead = <4ms

oboe::DataCallbackResult LessAudioEngine::onAudioReady(
    oboe::AudioStream* outputStream,
    void* audioData,
    int32_t numFrames)
{
    auto currentState = mState.load(std::memory_order_relaxed);
    if (currentState == EngineState::kPaused) {
        // Paused (glasses doffed/folded) — output silence but keep streams alive
        std::memset(audioData, 0, numFrames * sizeof(float));
        return oboe::DataCallbackResult::Continue;
    }
    if (currentState != EngineState::kRunning) {
        // Shutting down — zero-fill and stop the callback
        std::memset(audioData, 0, numFrames * sizeof(float));
        return oboe::DataCallbackResult::Stop;
    }

    auto* outputBuffer = static_cast<float*>(audioData);

    // --- Read input frames from the mic stream ---
    // This is a non-blocking read — returns however many frames are available
    auto readResult = mInputStream->read(
        mInputBuffer.data(),
        numFrames,
        0  // timeout = 0 → non-blocking
    );

    int32_t framesRead = 0;
    if (readResult.error() == oboe::Result::OK) {
        framesRead = readResult.value();
    } else {
        // If input stream disconnected or errored, we just assume 0 frames.
        // It's crucial we don't return early here so output-only modes like
        // VisionMusic can continue generating audio safely.
    }

    // If we got fewer frames than requested, pad with zeros
    if (framesRead < numFrames) {
        std::memset(
            mInputBuffer.data() + framesRead,
            0,
            (numFrames - framesRead) * sizeof(float)
        );
    }

    // --- DSP Processing ---
    float level = mSuppressionLevel.load(std::memory_order_relaxed);
    int32_t mode = mProcessingMode.load(std::memory_order_relaxed);

    if (mode == static_cast<int32_t>(ProcessingMode::kVisionMusic)) {
        // =================================================================
        // Mode 2: Vision-to-Music (output-only, no mic processing)
        // =================================================================
        // The VisionSynth generates music driven by scene parameters
        // from YOLO inference. No microphone input is needed — this
        // eliminates the entire feedback/GSC problem.
        mVisionSynth->synthesize(outputBuffer, numFrames);

    } else if (mode == static_cast<int32_t>(ProcessingMode::kComfortMask)) {
        // =================================================================
        // Mode 1: Comfort Mask
        // =================================================================
        // 1. GSC: Remove previous mask's echo from mic input
        // 2. MaskingEngine: Analyze cleaned ambient, generate mask
        // 3. AlertDetector: Check for urgent transients
        // 4. Mix: ambient passthrough + ducked mask
        // 5. EnvironmentClassifier: periodic spectral classification

        // Step 1: GSC echo cancellation — remove our own mask from mic
        mGscFilter->process(
            mInputBuffer.data(),
            mPrevMaskBuffer.data(),  // reference = what we played last frame
            mCleanBuffer.data(),
            numFrames
        );

        // Step 2: Analyze cleaned ambient and generate mask
        mMaskEngine->setMaskLevel(level);
        mMaskEngine->analyzeAndGenerate(
            mCleanBuffer.data(),
            mMaskBuffer.data(),
            numFrames
        );

        // Step 3: Alert detection — check if sudden transient needs pass-through
        // Skip alert-based ducking in harmonic mode:
        // The GSC can't perfectly cancel the drone, so residual energy
        // triggers the alert detector, causing rhythmic gain pumping.
        // The drone IS the output in harmonic mode — ducking it is wrong.
        float duckFactor = 1.0f;
        if (mMaskEngine->getTexture() != MaskTexture::kHarmonic) {
            duckFactor = mAlertDetector->process(mCleanBuffer.data(), numFrames);
        }

        // Step 4: Mix output = ambient passthrough + ducked mask
        for (int32_t i = 0; i < numFrames; ++i) {
            float ambient = mCleanBuffer[i];
            float mask = mMaskBuffer[i] * duckFactor;
            float mixed = ambient + mask;

            // Soft saturate — tanh keeps signal in [-1,1] smoothly
            // without the hard corners that cause audible clicks
            outputBuffer[i] = tanhf(mixed);
        }

        // Save current mask for next frame's GSC reference
        std::memcpy(mPrevMaskBuffer.data(), mMaskBuffer.data(),
                   numFrames * sizeof(float));

        // Step 5: Periodic environment classification
        mEnvClassifyCounter += numFrames;
        if (mEnvClassifyCounter >= mEnvClassifyInterval) {
            mEnvClassifyCounter = 0;
            // Feed the masking engine's spectral average to the classifier
            // (This is safe — both are accessed from this same RT thread)
        }

    } else {
        // =================================================================
        // Mode 0: Voice Isolate (existing behavior)
        // =================================================================
        mSuppressor->process(
            mInputBuffer.data(),
            outputBuffer,
            numFrames,
            level
        );
    }

    // =========================================================================
    // Phase 8: Waveform Snapshot for UI Visualizer (lock-free)
    // =========================================================================
    // We compute this here instead of in the delegates so it works
    // perfectly for all modes (VisionMusic, ComfortMask, VoiceIsolate).
    {
        int32_t wIdx = mWaveformWriteIndex.load(std::memory_order_relaxed);
        int32_t nextWIdx = 1 - wIdx;

        float inputSum = 0.0f;
        float outputSum = 0.0f;
        for (int32_t i = 0; i < numFrames; ++i) {
            float in_val = mInputBuffer[i];
            float out_val = outputBuffer[i];
            inputSum += in_val * in_val;
            outputSum += out_val * out_val;
        }
        float inRms = std::sqrt(inputSum / static_cast<float>(numFrames));
        float outRms = std::sqrt(outputSum / static_cast<float>(numFrames));

        mInputRms.store(std::min(inRms * 4.0f, 1.0f), std::memory_order_relaxed);
        mOutputRms.store(std::min(outRms * 4.0f, 1.0f), std::memory_order_relaxed);

        // Rolling Oscilloscope Buffer
        // We want kWaveformSamples points to cover ~50ms of audio history.
        // At 48kHz, 50ms = 2400 frames. 2400 frames / kWaveformSamples points = ~18.75 frames per point.
        int32_t newPointsCount = numFrames / 19; 
        if (newPointsCount < 1) newPointsCount = 1;
        if (newPointsCount > kWaveformSamples) newPointsCount = kWaveformSamples;
        
        int32_t shiftCount = kWaveformSamples - newPointsCount;
        
        // Push old data left to make room at the end
        if (shiftCount > 0) {
            std::memcpy(mInputWaveform[nextWIdx], &mInputWaveform[wIdx][newPointsCount], shiftCount * sizeof(float));
            std::memcpy(mOutputWaveform[nextWIdx], &mOutputWaveform[wIdx][newPointsCount], shiftCount * sizeof(float));
        }
        
        // Append downsampled data to the tail
        float step = static_cast<float>(numFrames) / static_cast<float>(newPointsCount);
        for (int32_t i = 0; i < newPointsCount; ++i) {
            int32_t idx = static_cast<int32_t>(i * step);
            if (idx >= numFrames) idx = numFrames - 1;
            mInputWaveform[nextWIdx][shiftCount + i] = mInputBuffer[idx];
            mOutputWaveform[nextWIdx][shiftCount + i] = outputBuffer[idx];
        }

        // Swap: UI thread will now see this completed snapshot
        mWaveformWriteIndex.store(nextWIdx, std::memory_order_release);
    }

    // --- Update latency measurement ---
    // Oboe exposes presentation timestamps for precise latency calculation
    auto inputLatency = mInputStream->calculateLatencyMillis();
    auto outputLatency = mOutputStream->calculateLatencyMillis();
    if (inputLatency && outputLatency) {
        double totalLatency = inputLatency.value() + outputLatency.value();
        mMeasuredLatencyMs.store(totalLatency, std::memory_order_relaxed);
    }

    return oboe::DataCallbackResult::Continue;
}

// =============================================================================
// Error Recovery
// =============================================================================

void LessAudioEngine::onErrorAfterClose(
    oboe::AudioStream* stream,
    oboe::Result error)
{
    LOGE("Stream error after close: %s (direction=%d)",
         oboe::convertToText(error),
         static_cast<int>(stream->getDirection()));

    // Phase 9: Self-healing architecture using std::atomic
    // Do NOT attempt to restart streams here! This callback can run on arbitrary
    // system threads where thread-local JNI envs might be absent or we might
    // deadlock. We record the error and let the Kotlin-layer Flow react.
    mStreamErrorStatus.store(static_cast<int>(error), std::memory_order_relaxed);
    mState.store(EngineState::kError, std::memory_order_relaxed);
}

// =============================================================================
// Phase 7: Thermal Management Delegation
// =============================================================================
// These methods delegate to NoiseSuppressor's embedded ThermalManager.
// They are called from JNI on a monitoring thread, NOT the audio thread.

bool LessAudioEngine::initializeThermalManager() {
    if (!mSuppressor) {
        LOGE("initializeThermalManager: suppressor not yet created");
        return false;
    }
    return mSuppressor->initializeThermalManager();
}

float LessAudioEngine::pollThermalHeadroom(int32_t forecastSeconds) {
    if (!mSuppressor) return -1.0f;
    return mSuppressor->pollThermalHeadroom(forecastSeconds);
}

int32_t LessAudioEngine::getThermalState() const {
    if (!mSuppressor) return -1;
    return static_cast<int32_t>(mSuppressor->getThermalState());
}

float LessAudioEngine::getThermalHeadroom() const {
    if (!mSuppressor) return -1.0f;
    return mSuppressor->getThermalHeadroom();
}

float LessAudioEngine::getForecastHeadroom() const {
    if (!mSuppressor) return -1.0f;
    return mSuppressor->getForecastHeadroom();
}

bool LessAudioEngine::isThermalMonitoringAvailable() const {
    if (!mSuppressor) return false;
    return mSuppressor->isThermalMonitoringAvailable();
}

bool LessAudioEngine::isThermalCallbackActive() const {
    if (!mSuppressor) return false;
    return mSuppressor->isThermalCallbackActive();
}

// =============================================================================
// Phase 8: Audio Level Snapshot for Waveform Visualizer
// =============================================================================

void LessAudioEngine::getAudioLevels(float* outBuffer, int32_t bufferSize) const {
    if (!outBuffer || bufferSize < 258) {
        if (outBuffer && bufferSize > 0) {
            memset(outBuffer, 0, bufferSize * sizeof(float));
        }
        return;
    }

    // The UI thread reads from the buffer NOT currently being written to
    int32_t readIdx = 1 - mWaveformWriteIndex.load(std::memory_order_acquire);

    outBuffer[0] = mInputRms.load(std::memory_order_relaxed);
    outBuffer[1] = mOutputRms.load(std::memory_order_relaxed);

    std::memcpy(&outBuffer[2], mInputWaveform[readIdx], kWaveformSamples * sizeof(float));
    std::memcpy(&outBuffer[2 + kWaveformSamples], mOutputWaveform[readIdx], kWaveformSamples * sizeof(float));
}

// =============================================================================
// Phase 11: Psychoacoustic Masking — Accessor Methods
// =============================================================================

void LessAudioEngine::setProcessingMode(int32_t mode) {
    int32_t clamped = (mode < 0) ? 0 : (mode > 2) ? 2 : mode;
    int32_t prev = mProcessingMode.exchange(clamped, std::memory_order_relaxed);
    if (prev != clamped) {
        static const char* modeNames[] = {"VoiceIsolate", "ComfortMask", "VisionMusic"};
        LOGI("Processing mode changed: %s → %s", modeNames[prev], modeNames[clamped]);

        // Reset GSC filter when switching modes to clear stale taps
        if (mGscFilter && clamped == 1) {
            mGscFilter->reset();
        }
    }
}

int32_t LessAudioEngine::getProcessingMode() const {
    return mProcessingMode.load(std::memory_order_relaxed);
}

bool LessAudioEngine::isMaskActive() const {
    if (!mMaskEngine) return false;
    return mMaskEngine->isActive();
}

void LessAudioEngine::setMaskTexture(int32_t texture) {
    if (!mMaskEngine) return;
    mMaskEngine->setTexture(static_cast<MaskTexture>(texture));
    LOGI("Mask texture set: %s",
         MaskingEngine::textureName(static_cast<MaskTexture>(texture)));
}

int32_t LessAudioEngine::getMaskTexture() const {
    if (!mMaskEngine) return 0;
    return static_cast<int32_t>(mMaskEngine->getTexture());
}

int32_t LessAudioEngine::getDetectedEnvironment() const {
    if (!mEnvClassifier) return 0;
    return static_cast<int32_t>(mEnvClassifier->getEnvironment());
}

const char* LessAudioEngine::getDetectedEnvironmentName() const {
    if (!mEnvClassifier) return "Unknown";
    return mEnvClassifier->getEnvironmentName();
}

bool LessAudioEngine::isAlertActive() const {
    if (!mAlertDetector) return false;
    return mAlertDetector->isAlertActive();
}

// =============================================================================
// Phase 15: Vision-to-Music — Accessor Methods
// =============================================================================

void LessAudioEngine::updateMusicalParameters(float density, float valence,
                                               float arousal, float timbre) {
    if (!mVisionSynth) return;
    mVisionSynth->updateSceneParameters(density, valence, arousal, timbre);
}

void LessAudioEngine::applyNeuralData(const float* data, int32_t len) {
    if (!mVisionSynth) return;
    mVisionSynth->applyNeuralData(data, len);
}

void LessAudioEngine::setMusicInterpretation(int32_t interpretation) {
    if (!mVisionSynth) return;
    mVisionSynth->setInterpretation(interpretation);
    static const char* interpNames[] = {"AmbientDrift", "MelodicArpeggio", "RhythmicPulse"};
    int32_t clamped = (interpretation < 0) ? 0 : (interpretation > 2) ? 2 : interpretation;
    LOGI("Music interpretation set: %s", interpNames[clamped]);
}

int32_t LessAudioEngine::getMusicInterpretation() const {
    if (!mVisionSynth) return 0;
    return mVisionSynth->getInterpretation();
}

float LessAudioEngine::getMusicBpm() const {
    if (!mVisionSynth) return 0.0f;
    return mVisionSynth->getCurrentBpm();
}

int32_t LessAudioEngine::getMusicKey() const {
    if (!mVisionSynth) return 0;
    return mVisionSynth->getCurrentKey();
}

void LessAudioEngine::setSynthQuality(int32_t quality) {
    if (mVisionSynth) mVisionSynth->setSynthQuality(quality);
}

int32_t LessAudioEngine::getSynthQuality() const {
    if (!mVisionSynth) return 0;
    return mVisionSynth->getSynthQuality();
}

// =============================================================================
// Phase 19: Synesthesia Mapping & Transitions
// =============================================================================

void LessAudioEngine::updateSynesthesiaParams(float hue, float saturation, float value) {
    if (mVisionSynth) mVisionSynth->updateSynesthesiaParams(hue, saturation, value);
}

void LessAudioEngine::flushGenerativeState() {
    if (mVisionSynth) mVisionSynth->flushGenerativeState();
}

} // namespace less
