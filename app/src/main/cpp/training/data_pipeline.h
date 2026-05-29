// =============================================================================
// data_pipeline.h — Corpus Reader and Training Data Batcher
// =============================================================================
// Reads raw PCM files from the profiling corpus, segments into frames,
// applies SNR weighting, shuffles, and batches for training.
// =============================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <string>

#include "snr_predictor.h"

namespace less {
namespace training {

// A single training example: one frame of audio with its SNR weight
struct TrainingFrame {
    std::vector<float> audio;    // frameSize samples
    float weight;                // SNR-derived training weight ∈ [0, 1]
};

struct DataPipelineConfig {
    int32_t frameSize = 480;         // 10ms @ 48kHz
    int32_t sampleRate = 48000;
    float minWeight = 0.05f;          // discard frames below this threshold
    int32_t batchSize = 32;
    bool shuffle = true;
    uint32_t seed = 42;
};

class DataPipeline {
public:
    explicit DataPipeline(const DataPipelineConfig& config = DataPipelineConfig{});
    ~DataPipeline();

    // Load all .pcm files from the corpus directory (plaintext path — legacy)
    // Returns total number of valid training frames
    int32_t loadCorpus(const std::string& corpusDir);

    // Phase 6: Feed decrypted audio samples directly from Kotlin ByteBuffer.
    // Segments into frames, computes SNR weights, and appends to the corpus.
    // Returns the number of training frames extracted from this buffer.
    int32_t feedDecryptedSamples(const float* samples, int32_t sampleCount,
                                 const std::string& sourceId);

    // Phase 6: Finalize corpus ingestion after all chunks have been fed.
    // Computes aggregate statistics and builds the shuffled index.
    // Returns the total number of training frames.
    int32_t finalizeIngestion();

    // Get the next batch of training frames
    // Returns false if no more batches (call reset() to start new epoch)
    bool nextBatch(std::vector<TrainingFrame>& batchOut);

    // Reset the iterator for a new epoch (re-shuffles if enabled)
    void reset();

    // Total frames in the corpus (after filtering)
    int32_t totalFrames() const { return static_cast<int32_t>(mFrames.size()); }

    // Corpus statistics
    float meanWeight() const { return mMeanWeight; }
    float corpusDurationSeconds() const;

private:
    DataPipelineConfig mConfig;
    SNRPredictor mSNRPredictor;

    // All loaded and weighted frames
    std::vector<TrainingFrame> mFrames;

    // Shuffled index for iteration
    std::vector<int32_t> mShuffledIndices;
    int32_t mCurrentIndex{0};

    // Statistics
    float mMeanWeight{0.0f};

    // Internal helpers
    bool loadPCMFile(const std::string& path);
    void buildShuffledIndices();
};

} // namespace training
} // namespace less
