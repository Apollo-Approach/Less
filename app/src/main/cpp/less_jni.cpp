// =============================================================================
// less_jni.cpp — JNI Bridge for the LESS Audio Engine
// =============================================================================
// Thin JNI layer connecting Kotlin lifecycle calls to the native C++ engine.
//
// Threading model:
//   - All JNI methods are called from the Android main/UI thread
//   - The audio engine's callback runs on a separate real-time thread
//   - Communication between them uses atomics (no mutex in the audio path)
//
// Engine handle:
//   Stored as a jlong on the Kotlin side for lifecycle safety.
//   This prevents use-after-free if the Activity is recreated.
// =============================================================================

#include <jni.h>
#include <android/log.h>

#include "audio_engine.h"

#define LOG_TAG "LESS_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Global engine instance — singleton for the process lifetime
// (only one audio pipeline can run at a time due to Exclusive mode)
static less::LessAudioEngine* gEngine = nullptr;

extern "C" {

// =============================================================================
// Engine lifecycle
// =============================================================================

JNIEXPORT jlong JNICALL
Java_com_less_audio_NativeAudioBridge_nativeCreateEngine(
    JNIEnv* env, jobject /* this */)
{
    if (gEngine != nullptr) {
        LOGW("Engine already exists — returning existing handle");
        return reinterpret_cast<jlong>(gEngine);
    }

    gEngine = new less::LessAudioEngine();
    LOGI("Native engine created: handle=%p", gEngine);
    return reinterpret_cast<jlong>(gEngine);
}

JNIEXPORT void JNICALL
Java_com_less_audio_NativeAudioBridge_nativeSetModelPath(
    JNIEnv* env, jobject /* this */, jlong engineHandle, jstring modelPath)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) {
        LOGE("nativeSetModelPath: null engine handle");
        return;
    }

    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (path) {
        engine->setModelPath(path);
        env->ReleaseStringUTFChars(modelPath, path);
    }
}

JNIEXPORT jboolean JNICALL
Java_com_less_audio_NativeAudioBridge_nativeStartEngine(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) {
        LOGE("nativeStartEngine: null engine handle");
        return JNI_FALSE;
    }
    return engine->start() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_less_audio_NativeAudioBridge_nativeStopEngine(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) {
        LOGE("nativeStopEngine: null engine handle");
        return;
    }
    engine->stop();
}

JNIEXPORT void JNICALL
Java_com_less_audio_NativeAudioBridge_nativeDestroyEngine(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (engine) {
        engine->stop();
        delete engine;
        if (gEngine == engine) {
            gEngine = nullptr;
        }
        LOGI("Native engine destroyed");
    }
}

// =============================================================================
// Runtime tuning
// =============================================================================

JNIEXPORT void JNICALL
Java_com_less_audio_NativeAudioBridge_nativeSetSuppressionLevel(
    JNIEnv* env, jobject /* this */, jlong engineHandle, jfloat level)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (engine) {
        engine->setSuppressionLevel(level);
    }
}

JNIEXPORT jdouble JNICALL
Java_com_less_audio_NativeAudioBridge_nativeGetLatencyMs(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) return -1.0;
    return static_cast<jdouble>(engine->getLatencyMs());
}

JNIEXPORT jboolean JNICALL
Java_com_less_audio_NativeAudioBridge_nativeIsRunning(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) return JNI_FALSE;
    return engine->isRunning() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jint JNICALL
Java_com_less_audio_NativeAudioBridge_nativeGetStreamErrorStatus(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) return -1;
    return engine->getStreamErrorStatus();
}

// =============================================================================
// Phase 2: Adapter hot-swap
// =============================================================================

JNIEXPORT jboolean JNICALL
Java_com_less_audio_NativeAudioBridge_nativeReloadAdapter(
    JNIEnv* env, jobject /* this */, jlong engineHandle, jstring adapterPath)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) {
        LOGE("nativeReloadAdapter: null engine handle");
        return JNI_FALSE;
    }

    const char* path = env->GetStringUTFChars(adapterPath, nullptr);
    if (!path) {
        LOGE("nativeReloadAdapter: null path string");
        return JNI_FALSE;
    }

    bool success = engine->reloadAdapterWeights(path);
    env->ReleaseStringUTFChars(adapterPath, path);

    return success ? JNI_TRUE : JNI_FALSE;
}

// =============================================================================
// Phase 3: Session lifecycle — pause/resume for doff/fold safety
// =============================================================================
// Called from BluetoothAudioRouter when the Wearables SDK reports
// SessionState.PAUSED (glasses doffed) or SessionState.STOPPED (folded).
// Must execute FAST — BLE audio data may stop within milliseconds.

JNIEXPORT void JNICALL
Java_com_less_audio_NativeAudioBridge_nativePauseEngine(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) {
        LOGE("nativePauseEngine: null engine handle");
        return;
    }
    engine->pause();
}

JNIEXPORT void JNICALL
Java_com_less_audio_NativeAudioBridge_nativeResumeEngine(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) {
        LOGE("nativeResumeEngine: null engine handle");
        return;
    }
    engine->resume();
}

// =============================================================================
// Phase 4: NPU Configuration
// =============================================================================

JNIEXPORT void JNICALL
Java_com_less_audio_NativeAudioBridge_nativeSetQnnLibDir(
    JNIEnv* env, jobject /* this */, jlong engineHandle, jstring qnnLibDir)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) {
        LOGE("nativeSetQnnLibDir: null engine handle");
        return;
    }

    const char* path = env->GetStringUTFChars(qnnLibDir, nullptr);
    if (path) {
        engine->setQnnLibDir(path);
        env->ReleaseStringUTFChars(qnnLibDir, path);
    }
}

JNIEXPORT jstring JNICALL
Java_com_less_audio_NativeAudioBridge_nativeGetActiveBackend(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) {
        return env->NewStringUTF("Not initialized");
    }
    return env->NewStringUTF(engine->getActiveBackendName());
}

// =============================================================================
// Phase 7: Thermal Management JNI Bridge
// =============================================================================

JNIEXPORT jboolean JNICALL
Java_com_less_audio_NativeAudioBridge_nativeInitThermalManager(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) {
        LOGE("nativeInitThermalManager: null engine handle");
        return JNI_FALSE;
    }
    return engine->initializeThermalManager() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jfloat JNICALL
Java_com_less_audio_NativeAudioBridge_nativePollThermalHeadroom(
    JNIEnv* env, jobject /* this */, jlong engineHandle, jint forecastSeconds)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) return -1.0f;
    return engine->pollThermalHeadroom(forecastSeconds);
}

JNIEXPORT jint JNICALL
Java_com_less_audio_NativeAudioBridge_nativeGetThermalState(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) return -1;
    return static_cast<jint>(engine->getThermalState());
}

JNIEXPORT jfloat JNICALL
Java_com_less_audio_NativeAudioBridge_nativeGetThermalHeadroom(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) return -1.0f;
    return engine->getThermalHeadroom();
}

// =============================================================================
// Phase 8: Audio Level + Waveform Snapshot for UI Visualizer
// =============================================================================
// Returns a FloatArray of 258 elements:
//   [0]       = input RMS (0.0–1.0)
//   [1]       = output RMS (0.0–1.0)
//   [2..129]   = input waveform (128 downsampled samples, -1.0 to 1.0)
//   [130..257] = output waveform (128 downsampled samples, -1.0 to 1.0)

JNIEXPORT jfloatArray JNICALL
Java_com_less_audio_NativeAudioBridge_nativeGetAudioLevels(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    constexpr int32_t kBufferSize = 258;  // 2 RMS + 128 + 128 waveform
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);

    jfloatArray result = env->NewFloatArray(kBufferSize);
    if (!result) return nullptr;

    float buffer[kBufferSize];
    if (engine) {
        engine->getAudioLevels(buffer, kBufferSize);
    } else {
        memset(buffer, 0, sizeof(buffer));
    }

    env->SetFloatArrayRegion(result, 0, kBufferSize, buffer);
    return result;
}

// =============================================================================
// Phase 11: Psychoacoustic Masking — JNI Bindings
// =============================================================================

JNIEXPORT void JNICALL
Java_com_less_audio_NativeAudioBridge_nativeSetProcessingMode(
    JNIEnv* env, jobject /* this */, jlong engineHandle, jint mode)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (engine) engine->setProcessingMode(mode);
}

JNIEXPORT jint JNICALL
Java_com_less_audio_NativeAudioBridge_nativeGetProcessingMode(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) return 0;
    return engine->getProcessingMode();
}

JNIEXPORT jboolean JNICALL
Java_com_less_audio_NativeAudioBridge_nativeIsMaskActive(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) return JNI_FALSE;
    return engine->isMaskActive() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_less_audio_NativeAudioBridge_nativeSetMaskTexture(
    JNIEnv* env, jobject /* this */, jlong engineHandle, jint texture)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (engine) engine->setMaskTexture(texture);
}

JNIEXPORT jint JNICALL
Java_com_less_audio_NativeAudioBridge_nativeGetMaskTexture(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) return 0;
    return engine->getMaskTexture();
}

JNIEXPORT jint JNICALL
Java_com_less_audio_NativeAudioBridge_nativeGetDetectedEnvironment(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) return 0;
    return engine->getDetectedEnvironment();
}

JNIEXPORT jboolean JNICALL
Java_com_less_audio_NativeAudioBridge_nativeIsAlertActive(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) return JNI_FALSE;
    return engine->isAlertActive() ? JNI_TRUE : JNI_FALSE;
}

// =============================================================================
// Phase 15: Vision-to-Music
// =============================================================================

JNIEXPORT void JNICALL
Java_com_less_audio_NativeAudioBridge_nativeUpdateMusicalParameters(
    JNIEnv* env, jobject /* this */, jlong engineHandle,
    jfloat density, jfloat valence, jfloat arousal, jfloat timbre)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (engine) engine->updateMusicalParameters(density, valence, arousal, timbre);
}

JNIEXPORT void JNICALL
Java_com_less_audio_NativeAudioBridge_nativePushTinyMusicianData(
    JNIEnv* env, jobject /* this */, jlong engineHandle, jfloatArray jData)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine || !jData) return;

    jsize len = env->GetArrayLength(jData);
    jfloat* data = env->GetFloatArrayElements(jData, nullptr);
    if (data) {
        engine->applyNeuralData(data, len);
        env->ReleaseFloatArrayElements(jData, data, JNI_ABORT);
    }
}

JNIEXPORT void JNICALL
Java_com_less_audio_NativeAudioBridge_nativeSetMusicInterpretation(
    JNIEnv* env, jobject /* this */, jlong engineHandle, jint interpretation)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (engine) engine->setMusicInterpretation(interpretation);
}

JNIEXPORT jint JNICALL
Java_com_less_audio_NativeAudioBridge_nativeGetMusicInterpretation(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) return 0;
    return engine->getMusicInterpretation();
}

JNIEXPORT jfloat JNICALL
Java_com_less_audio_NativeAudioBridge_nativeGetMusicBpm(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) return 0.0f;
    return engine->getMusicBpm();
}

JNIEXPORT jint JNICALL
Java_com_less_audio_NativeAudioBridge_nativeGetMusicKey(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) return 0;
    return engine->getMusicKey();
}

JNIEXPORT void JNICALL
Java_com_less_audio_NativeAudioBridge_nativeSetSynthQuality(
    JNIEnv* env, jobject /* this */, jlong engineHandle, jint quality)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (engine) engine->setSynthQuality(quality);
}

JNIEXPORT jint JNICALL
Java_com_less_audio_NativeAudioBridge_nativeGetSynthQuality(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (!engine) return 0;
    return engine->getSynthQuality();
}

JNIEXPORT void JNICALL
Java_com_less_audio_NativeAudioBridge_nativeUpdateSynesthesiaParams(
    JNIEnv* env, jobject /* this */, jlong engineHandle,
    jfloat hue, jfloat saturation, jfloat value)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (engine) engine->updateSynesthesiaParams(hue, saturation, value);
}

JNIEXPORT void JNICALL
Java_com_less_audio_NativeAudioBridge_nativeFlushGenerativeState(
    JNIEnv* env, jobject /* this */, jlong engineHandle)
{
    auto* engine = reinterpret_cast<less::LessAudioEngine*>(engineHandle);
    if (engine) engine->flushGenerativeState();
}

} // extern "C"
