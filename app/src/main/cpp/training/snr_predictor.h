// =============================================================================
// snr_predictor.h — Frame-Level SNR Estimation for Self-Supervised Training
// =============================================================================
// Estimates the Signal-to-Noise Ratio of each audio frame using spectral
// heuristics. No clean reference required — works on raw noisy recordings.
//
// Method:
//   1. Spectral Flatness: flat spectrum = noise, peaked = speech
//      SF = exp(mean(log(S))) / mean(S)    ∈ [0, 1]
//
//   2. Harmonic-to-Noise Ratio (HNR): detects voiced speech harmonics
//      via autocorrelation peak analysis
//
//   3. Combined SNR estimate mapped to a training weight via sigmoid:
//      w = sigmoid((snr - threshold) / temperature)
//
// The weight vector is used by AdapterTrainer to scale the self-supervised
// loss — high-SNR frames (clear speech) dominate the gradient, while
// low-SNR frames (pure noise) contribute near-zero gradient.
// =============================================================================

#pragma once

#include <cstdint>
#include <vector>

namespace less {
namespace training {

struct SNRConfig {
    int32_t sampleRate = 48000;
    int32_t frameSize = 480;         // 10ms @ 48kHz
    int32_t fftSize = 512;           // next power of 2

    // Weight mapping parameters
    float snrThreshold = 5.0f;       // dB — center of sigmoid
    float temperature = 3.0f;        // sigmoid steepness
    float minWeight = 0.0f;          // floor clamp
    float maxWeight = 1.0f;          // ceiling clamp

    // Noise floor tracking
    float noiseAlpha = 0.95f;        // EMA smoothing for noise estimate

    // Weight smoothing
    int32_t smoothingWindow = 5;     // frames for moving average
};

class SNRPredictor {
public:
    explicit SNRPredictor(const SNRConfig& config = SNRConfig{});
    ~SNRPredictor();

    // Estimate SNR for a single frame and return a training weight ∈ [0, 1]
    float estimateWeight(const float* frame, int32_t numSamples);

    // Batch process: estimate weights for an entire corpus buffer
    // Returns a weight vector parallel to the input frames
    void estimateWeights(
        const float* audio,
        int32_t totalSamples,
        float* weightsOut,       // one weight per frame
        int32_t* numFramesOut    // total frames processed
    );

    // Get the raw SNR estimate (in dB) for the last processed frame
    float getLastSNR() const { return mLastSNR; }

    // Reset internal state (call between corpus files)
    void reset();

private:
    SNRConfig mConfig;

    // FFT working buffers (pre-allocated)
    std::vector<float> mFftRe;
    std::vector<float> mFftIm;
    std::vector<float> mMagnitude;
    std::vector<float> mWindow;

    // Noise floor estimate per bin
    std::vector<float> mNoiseFloor;
    bool mNoiseEstimateReady{false};
    int32_t mFrameCount{0};
    static constexpr int32_t kNoiseEstimateFrames = 30;

    // Weight smoothing buffer
    std::vector<float> mWeightHistory;
    int32_t mWeightHistoryPos{0};

    // Last computed SNR (for diagnostics)
    float mLastSNR{0.0f};

    // Internal methods
    float computeSpectralFlatness(const float* magnitude, int32_t numBins);
    float computeHNR(const float* frame, int32_t numSamples);
    float snrToWeight(float snrDb);
    float smoothWeight(float rawWeight);
    void computeHannWindow();
};

} // namespace training
} // namespace less
