// =============================================================================
// training_jni.cpp — JNI Bridge for the Offline Training Pipeline
// =============================================================================
//
// Phase 6: Added nativeFeedDecryptedCorpus() and nativeFinalizeCorpus() for
// the encrypted corpus data flow. The Kotlin layer decrypts EncryptedFile
// data and passes float32 samples via direct ByteBuffer. This replaces the
// file-based loadCorpus() path which cannot read AES-GCM encrypted files.

#include <jni.h>
#include <android/log.h>
#include <string>
#include <memory>

#include "adapter_trainer.h"
#include "data_pipeline.h"

#define LOG_TAG "LESS_TrainingJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Global trainer instance — only one training run at a time
static std::unique_ptr<less::training::AdapterTrainer> gTrainer;

// Global data pipeline for the encrypted corpus feed path
static std::unique_ptr<less::training::DataPipeline> gPipeline;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_less_audio_NativeAudioBridge_nativeStartTraining(
    JNIEnv* env, jobject /* this */,
    jstring corpusDir,
    jstring baseModelPath,
    jstring adapterPath,
    jint epochs,
    jfloat learningRate)
{
    if (gTrainer) {
        LOGW("Training already in progress — ignoring start request");
        return JNI_FALSE;
    }

    const char* corpus = env->GetStringUTFChars(corpusDir, nullptr);
    const char* model = env->GetStringUTFChars(baseModelPath, nullptr);
    const char* adapter = env->GetStringUTFChars(adapterPath, nullptr);

    less::training::TrainingConfig config;
    config.epochs = epochs;
    config.learningRate = learningRate;
    config.checkpointDir = std::string(adapter).substr(
        0, std::string(adapter).find_last_of('/'));

    gTrainer = std::make_unique<less::training::AdapterTrainer>(config);

    std::string corpusStr(corpus);
    std::string modelStr(model);
    std::string adapterStr(adapter);

    env->ReleaseStringUTFChars(corpusDir, corpus);
    env->ReleaseStringUTFChars(baseModelPath, model);
    env->ReleaseStringUTFChars(adapterPath, adapter);

    // NOTE: This call BLOCKS — it must be called from a background thread
    // (WorkManager CoroutineWorker dispatches on Dispatchers.Default)
    bool success = gTrainer->train(corpusStr, modelStr, adapterStr);

    // Training complete — clean up
    gTrainer.reset();
    gPipeline.reset();

    return success ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_less_audio_NativeAudioBridge_nativeStopTraining(
    JNIEnv* env, jobject /* this */)
{
    if (gTrainer) {
        gTrainer->cancel();
        LOGI("Training cancellation signal sent");
    }
}

JNIEXPORT jfloatArray JNICALL
Java_com_less_audio_NativeAudioBridge_nativeGetTrainingProgress(
    JNIEnv* env, jobject /* this */)
{
    // Return [epoch, totalEpochs, currentLoss, bestLoss, framesProcessed,
    //         totalFrames, isRunning, isComplete]
    jfloatArray result = env->NewFloatArray(8);
    if (!result) return nullptr;

    float data[8] = {0};
    if (gTrainer) {
        auto progress = gTrainer->getProgress();
        data[0] = static_cast<float>(progress.currentEpoch);
        data[1] = static_cast<float>(progress.totalEpochs);
        data[2] = progress.currentLoss;
        data[3] = progress.bestLoss;
        data[4] = static_cast<float>(progress.framesProcessed);
        data[5] = static_cast<float>(progress.totalFrames);
        data[6] = progress.isRunning ? 1.0f : 0.0f;
        data[7] = progress.isComplete ? 1.0f : 0.0f;
    }

    env->SetFloatArrayRegion(result, 0, 8, data);
    return result;
}

// =============================================================================
// Phase 6: Encrypted Corpus Data Feed
// =============================================================================
//
// These methods implement the Kotlin → C++ data bridge for encrypted corpus
// files. The flow is:
//   1. Kotlin decrypts EncryptedFile via Android Keystore
//   2. Kotlin reads decrypted bytes into a direct ByteBuffer
//   3. nativeFeedDecryptedCorpus() receives the raw float32 samples
//   4. C++ DataPipeline segments into frames, computes SNR weights
//   5. nativeFinalizeCorpus() builds the shuffled index for training
//   6. nativeStartTraining() runs SPSA on the pre-loaded corpus

JNIEXPORT jint JNICALL
Java_com_less_audio_NativeAudioBridge_nativeFeedDecryptedCorpus(
    JNIEnv* env, jobject /* this */,
    jobject buffer,         // direct ByteBuffer
    jint sampleCount,
    jstring sourceId)
{
    // Lazy-init the pipeline
    if (!gPipeline) {
        gPipeline = std::make_unique<less::training::DataPipeline>();
        LOGI("DataPipeline created for encrypted corpus feed");
    }

    // Get direct buffer address — zero-copy access to Kotlin's ByteBuffer
    auto* bufferAddr = static_cast<float*>(
        env->GetDirectBufferAddress(buffer));
    if (!bufferAddr) {
        LOGE("Failed to get direct buffer address — is the ByteBuffer direct?");
        return 0;
    }

    const char* srcId = env->GetStringUTFChars(sourceId, nullptr);
    std::string srcIdStr(srcId);
    env->ReleaseStringUTFChars(sourceId, srcId);

    // Feed the decrypted samples directly into the pipeline's frame buffer.
    // This reuses the SNR weighting and frame segmentation logic from
    // DataPipeline::loadPCMFile() but operates on in-memory data.
    int32_t framesAdded = gPipeline->feedDecryptedSamples(
        bufferAddr, sampleCount, srcIdStr);

    LOGI("Fed %d samples from '%s' — %d frames extracted",
         sampleCount, srcIdStr.c_str(), framesAdded);

    return static_cast<jint>(framesAdded);
}

JNIEXPORT jint JNICALL
Java_com_less_audio_NativeAudioBridge_nativeFinalizeCorpus(
    JNIEnv* env, jobject /* this */)
{
    if (!gPipeline) {
        LOGW("nativeFinalizeCorpus called with no pipeline — nothing to finalize");
        return 0;
    }

    int32_t totalFrames = gPipeline->finalizeIngestion();
    LOGI("Corpus finalized: %d total training frames", totalFrames);

    return static_cast<jint>(totalFrames);
}

} // extern "C"
