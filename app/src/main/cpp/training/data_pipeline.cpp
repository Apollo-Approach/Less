// =============================================================================
// data_pipeline.cpp — Corpus Reader and Training Data Batcher
// =============================================================================

#include "data_pipeline.h"

#include <android/log.h>
#include <dirent.h>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>

#define LOG_TAG "LESS_DataPipeline"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace less {
namespace training {

DataPipeline::DataPipeline(const DataPipelineConfig& config)
    : mConfig(config), mSNRPredictor(SNRConfig{}) {
    // Match SNR predictor config to pipeline config
    SNRConfig snrCfg;
    snrCfg.sampleRate = config.sampleRate;
    snrCfg.frameSize = config.frameSize;
    mSNRPredictor = SNRPredictor(snrCfg);
}

DataPipeline::~DataPipeline() = default;

// =============================================================================
// Corpus Loading
// =============================================================================

int32_t DataPipeline::loadCorpus(const std::string& corpusDir) {
    LOGI("Loading corpus from: %s", corpusDir.c_str());

    mFrames.clear();

    // List all .pcm files in the corpus directory
    DIR* dir = opendir(corpusDir.c_str());
    if (!dir) {
        LOGE("Failed to open corpus directory: %s", corpusDir.c_str());
        return 0;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name(entry->d_name);
        if (name.size() > 4 && name.substr(name.size() - 4) == ".pcm") {
            std::string fullPath = corpusDir + "/" + name;
            if (!loadPCMFile(fullPath)) {
                LOGW("Failed to load: %s", name.c_str());
            }
        }
    }
    closedir(dir);

    // Compute statistics
    if (!mFrames.empty()) {
        float totalWeight = 0.0f;
        for (const auto& frame : mFrames) {
            totalWeight += frame.weight;
        }
        mMeanWeight = totalWeight / static_cast<float>(mFrames.size());
    }

    LOGI("Corpus loaded: %zu frames across directory, mean weight=%.3f",
         mFrames.size(), mMeanWeight);

    // Build initial shuffled indices
    buildShuffledIndices();

    return static_cast<int32_t>(mFrames.size());
}

bool DataPipeline::loadPCMFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOGE("Failed to open PCM file: %s", path.c_str());
        return false;
    }

    // Read the entire file into memory
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t numSamples = fileSize / sizeof(float);
    if (numSamples < static_cast<size_t>(mConfig.frameSize)) {
        LOGW("PCM file too small: %s (%zu samples)", path.c_str(), numSamples);
        return false;
    }

    std::vector<float> audio(numSamples);
    file.read(reinterpret_cast<char*>(audio.data()), fileSize);

    // Reset SNR predictor state for each file
    mSNRPredictor.reset();

    // Segment into frames and estimate SNR weights
    int32_t numFrames = static_cast<int32_t>(numSamples) / mConfig.frameSize;
    int32_t framesAdded = 0;

    for (int32_t f = 0; f < numFrames; ++f) {
        const float* frameData = audio.data() + f * mConfig.frameSize;
        float weight = mSNRPredictor.estimateWeight(frameData, mConfig.frameSize);

        // Filter out frames with very low SNR weight
        if (weight >= mConfig.minWeight) {
            TrainingFrame frame;
            frame.audio.assign(frameData, frameData + mConfig.frameSize);
            frame.weight = weight;
            mFrames.push_back(std::move(frame));
            framesAdded++;
        }
    }

    LOGI("Loaded %s: %d/%d frames passed SNR filter",
         path.c_str(), framesAdded, numFrames);
    return true;
}

// =============================================================================
// Batching
// =============================================================================

void DataPipeline::buildShuffledIndices() {
    mShuffledIndices.resize(mFrames.size());
    std::iota(mShuffledIndices.begin(), mShuffledIndices.end(), 0);

    if (mConfig.shuffle) {
        std::mt19937 rng(mConfig.seed);
        std::shuffle(mShuffledIndices.begin(), mShuffledIndices.end(), rng);
    }

    mCurrentIndex = 0;
}

bool DataPipeline::nextBatch(std::vector<TrainingFrame>& batchOut) {
    batchOut.clear();

    if (mCurrentIndex >= static_cast<int32_t>(mShuffledIndices.size())) {
        return false;  // epoch exhausted
    }

    int32_t remaining = static_cast<int32_t>(mShuffledIndices.size()) - mCurrentIndex;
    int32_t batchSize = std::min(mConfig.batchSize, remaining);

    batchOut.reserve(batchSize);
    for (int32_t i = 0; i < batchSize; ++i) {
        int32_t idx = mShuffledIndices[mCurrentIndex + i];
        batchOut.push_back(mFrames[idx]);
    }

    mCurrentIndex += batchSize;
    return true;
}

void DataPipeline::reset() {
    mConfig.seed++;  // different shuffle each epoch
    buildShuffledIndices();
}

float DataPipeline::corpusDurationSeconds() const {
    return static_cast<float>(mFrames.size() * mConfig.frameSize) /
           static_cast<float>(mConfig.sampleRate);
}

// =============================================================================
// Phase 6: Encrypted Corpus Data Feed
// =============================================================================

int32_t DataPipeline::feedDecryptedSamples(
    const float* samples, int32_t sampleCount, const std::string& sourceId) {

    if (!samples || sampleCount <= 0) {
        LOGE("feedDecryptedSamples: invalid input (null=%d, count=%d)",
             samples == nullptr, sampleCount);
        return 0;
    }

    if (sampleCount < mConfig.frameSize) {
        LOGW("feedDecryptedSamples: buffer from '%s' too small (%d < %d)",
             sourceId.c_str(), sampleCount, mConfig.frameSize);
        return 0;
    }

    // Reset SNR predictor state for each source chunk
    mSNRPredictor.reset();

    // Segment into frames and estimate SNR weights — identical logic
    // to loadPCMFile() but operating on in-memory data.
    int32_t numFrames = sampleCount / mConfig.frameSize;
    int32_t framesAdded = 0;

    for (int32_t f = 0; f < numFrames; ++f) {
        const float* frameData = samples + f * mConfig.frameSize;
        float weight = mSNRPredictor.estimateWeight(frameData, mConfig.frameSize);

        // Filter out frames with very low SNR weight
        if (weight >= mConfig.minWeight) {
            TrainingFrame frame;
            frame.audio.assign(frameData, frameData + mConfig.frameSize);
            frame.weight = weight;
            mFrames.push_back(std::move(frame));
            framesAdded++;
        }
    }

    LOGI("feedDecryptedSamples '%s': %d/%d frames passed SNR filter",
         sourceId.c_str(), framesAdded, numFrames);
    return framesAdded;
}

int32_t DataPipeline::finalizeIngestion() {
    // Compute aggregate statistics
    if (!mFrames.empty()) {
        float totalWeight = 0.0f;
        for (const auto& frame : mFrames) {
            totalWeight += frame.weight;
        }
        mMeanWeight = totalWeight / static_cast<float>(mFrames.size());
    }

    LOGI("Corpus finalized: %zu frames, mean weight=%.3f, duration=%.1fs",
         mFrames.size(), mMeanWeight, corpusDurationSeconds());

    // Build shuffled indices for training iteration
    buildShuffledIndices();

    return static_cast<int32_t>(mFrames.size());
}

} // namespace training
} // namespace less
