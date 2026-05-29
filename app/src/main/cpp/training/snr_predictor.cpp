// =============================================================================
// snr_predictor.cpp — Frame-Level SNR Estimation Implementation
// =============================================================================

#include "snr_predictor.h"

#include <android/log.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

#define LOG_TAG "LESS_SNR"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace less {
namespace training {

// Reuse the same radix-2 FFT from noise_suppressor.cpp
// In production, extract to a shared dsp_utils.h
namespace {

void fft_radix2(float* re, float* im, int n, bool inverse) {
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float angle = (inverse ? 2.0f : -2.0f) * M_PI / static_cast<float>(len);
        float wRe = cosf(angle), wIm = sinf(angle);
        for (int i = 0; i < n; i += len) {
            float curRe = 1.0f, curIm = 0.0f;
            for (int j = 0; j < len / 2; ++j) {
                float tRe = re[i+j+len/2]*curRe - im[i+j+len/2]*curIm;
                float tIm = re[i+j+len/2]*curIm + im[i+j+len/2]*curRe;
                re[i+j+len/2] = re[i+j] - tRe;
                im[i+j+len/2] = im[i+j] - tIm;
                re[i+j] += tRe;
                im[i+j] += tIm;
                float newCurRe = curRe*wRe - curIm*wIm;
                curIm = curRe*wIm + curIm*wRe;
                curRe = newCurRe;
            }
        }
    }
    if (inverse) {
        float invN = 1.0f / static_cast<float>(n);
        for (int i = 0; i < n; ++i) { re[i] *= invN; im[i] *= invN; }
    }
}

} // anonymous namespace

// =============================================================================
// Construction / Initialization
// =============================================================================

SNRPredictor::SNRPredictor(const SNRConfig& config) : mConfig(config) {
    int32_t fftSize = mConfig.fftSize;
    int32_t numBins = fftSize / 2 + 1;

    mFftRe.resize(fftSize, 0.0f);
    mFftIm.resize(fftSize, 0.0f);
    mMagnitude.resize(numBins, 0.0f);
    mWindow.resize(fftSize, 0.0f);
    mNoiseFloor.resize(numBins, 0.0f);

    mWeightHistory.resize(mConfig.smoothingWindow, 0.5f);
    mWeightHistoryPos = 0;

    computeHannWindow();

    LOGI("SNRPredictor initialized: frameSize=%d, fftSize=%d, threshold=%.1f dB",
         mConfig.frameSize, fftSize, mConfig.snrThreshold);
}

SNRPredictor::~SNRPredictor() = default;

void SNRPredictor::computeHannWindow() {
    for (int32_t i = 0; i < mConfig.fftSize; ++i) {
        mWindow[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (mConfig.fftSize - 1)));
    }
}

void SNRPredictor::reset() {
    std::fill(mNoiseFloor.begin(), mNoiseFloor.end(), 0.0f);
    std::fill(mWeightHistory.begin(), mWeightHistory.end(), 0.5f);
    mNoiseEstimateReady = false;
    mFrameCount = 0;
    mWeightHistoryPos = 0;
    mLastSNR = 0.0f;
}

// =============================================================================
// Spectral Flatness
// =============================================================================
// SF = exp(mean(log(S))) / mean(S)
//
// For white noise (flat spectrum): SF ≈ 1.0
// For tonal/voiced speech (peaked spectrum): SF ≈ 0.0
//
// We invert: lower flatness → higher SNR estimate

float SNRPredictor::computeSpectralFlatness(const float* magnitude, int32_t numBins) {
    float logSum = 0.0f;
    float linSum = 0.0f;
    int32_t validBins = 0;

    // Skip DC (bin 0) and very high frequencies
    int32_t startBin = 1;
    int32_t endBin = numBins - 1;

    for (int32_t b = startBin; b < endBin; ++b) {
        float mag = magnitude[b] + 1e-10f;  // avoid log(0)
        logSum += logf(mag);
        linSum += mag;
        validBins++;
    }

    if (validBins == 0) return 1.0f;  // degenerate case

    float geometricMean = expf(logSum / static_cast<float>(validBins));
    float arithmeticMean = linSum / static_cast<float>(validBins);

    float flatness = geometricMean / (arithmeticMean + 1e-10f);
    return std::clamp(flatness, 0.0f, 1.0f);
}

// =============================================================================
// Harmonic-to-Noise Ratio via Autocorrelation
// =============================================================================
// For voiced speech, the autocorrelation has a strong peak at the fundamental
// frequency period. For noise, autocorrelation decays monotonically.
//
// HNR = 10 * log10(peak / (1 - peak))  [in dB]

float SNRPredictor::computeHNR(const float* frame, int32_t numSamples) {
    // Search for pitch in the range 80-500 Hz
    int32_t minLag = mConfig.sampleRate / 500;  // 500 Hz upper bound
    int32_t maxLag = mConfig.sampleRate / 80;   // 80 Hz lower bound
    maxLag = std::min(maxLag, numSamples / 2);

    if (minLag >= maxLag || maxLag >= numSamples) return 0.0f;

    // Compute normalized autocorrelation at lag 0
    float energy = 0.0f;
    for (int32_t i = 0; i < numSamples; ++i) {
        energy += frame[i] * frame[i];
    }
    if (energy < 1e-10f) return 0.0f;  // silence

    // Find peak autocorrelation in the pitch range
    float bestCorr = 0.0f;
    for (int32_t lag = minLag; lag < maxLag; ++lag) {
        float corr = 0.0f;
        float denomA = 0.0f, denomB = 0.0f;
        int32_t count = numSamples - lag;
        for (int32_t i = 0; i < count; ++i) {
            corr += frame[i] * frame[i + lag];
            denomA += frame[i] * frame[i];
            denomB += frame[i + lag] * frame[i + lag];
        }
        float denom = sqrtf(denomA * denomB) + 1e-10f;
        float normalizedCorr = corr / denom;
        if (normalizedCorr > bestCorr) {
            bestCorr = normalizedCorr;
        }
    }

    // Convert to HNR in dB
    bestCorr = std::clamp(bestCorr, 0.01f, 0.99f);
    float hnr = 10.0f * log10f(bestCorr / (1.0f - bestCorr));
    return hnr;
}

// =============================================================================
// SNR → Training Weight
// =============================================================================

float SNRPredictor::snrToWeight(float snrDb) {
    // Sigmoid mapping: w = 1 / (1 + exp(-(snr - threshold) / temperature))
    float x = (snrDb - mConfig.snrThreshold) / mConfig.temperature;
    float w = 1.0f / (1.0f + expf(-x));
    return std::clamp(w, mConfig.minWeight, mConfig.maxWeight);
}

float SNRPredictor::smoothWeight(float rawWeight) {
    mWeightHistory[mWeightHistoryPos] = rawWeight;
    mWeightHistoryPos = (mWeightHistoryPos + 1) % mConfig.smoothingWindow;

    float sum = 0.0f;
    for (float w : mWeightHistory) sum += w;
    return sum / static_cast<float>(mConfig.smoothingWindow);
}

// =============================================================================
// Single-Frame Estimation
// =============================================================================

float SNRPredictor::estimateWeight(const float* frame, int32_t numSamples) {
    int32_t fftSize = mConfig.fftSize;
    int32_t numBins = fftSize / 2 + 1;

    // Window and zero-pad
    int32_t frameSamples = std::min(numSamples, fftSize);
    for (int32_t i = 0; i < frameSamples; ++i) {
        mFftRe[i] = frame[i] * mWindow[i];
        mFftIm[i] = 0.0f;
    }
    for (int32_t i = frameSamples; i < fftSize; ++i) {
        mFftRe[i] = 0.0f;
        mFftIm[i] = 0.0f;
    }

    // Forward FFT
    fft_radix2(mFftRe.data(), mFftIm.data(), fftSize, false);

    // Compute magnitude spectrum
    for (int32_t b = 0; b < numBins; ++b) {
        mMagnitude[b] = sqrtf(mFftRe[b] * mFftRe[b] + mFftIm[b] * mFftIm[b]);
    }

    // Update noise floor estimate
    mFrameCount++;
    if (!mNoiseEstimateReady) {
        for (int32_t b = 0; b < numBins; ++b) {
            mNoiseFloor[b] += mMagnitude[b];
        }
        if (mFrameCount >= kNoiseEstimateFrames) {
            for (int32_t b = 0; b < numBins; ++b) {
                mNoiseFloor[b] /= static_cast<float>(kNoiseEstimateFrames);
            }
            mNoiseEstimateReady = true;
        }
        return 0.5f;  // neutral weight during estimation
    } else {
        // EMA update: track slowly-varying noise floor
        for (int32_t b = 0; b < numBins; ++b) {
            float minVal = std::min(mMagnitude[b], mNoiseFloor[b] * 2.0f);
            mNoiseFloor[b] = mConfig.noiseAlpha * mNoiseFloor[b] +
                            (1.0f - mConfig.noiseAlpha) * minVal;
        }
    }

    // --- Compute SNR indicators ---

    // 1. Spectral flatness (inverted: low flatness = speech-like)
    float flatness = computeSpectralFlatness(mMagnitude.data(), numBins);
    float flatnessSNR = (1.0f - flatness) * 20.0f;  // map [0,1] → [0,20] dB

    // 2. Harmonic-to-Noise Ratio
    float hnr = computeHNR(frame, numSamples);

    // 3. Frame energy relative to noise floor
    float signalEnergy = 0.0f;
    float noiseEnergy = 0.0f;
    for (int32_t b = 0; b < numBins; ++b) {
        signalEnergy += mMagnitude[b] * mMagnitude[b];
        noiseEnergy += mNoiseFloor[b] * mNoiseFloor[b];
    }
    float energySNR = (noiseEnergy > 1e-10f) ?
        10.0f * log10f(signalEnergy / noiseEnergy) : 0.0f;

    // --- Combine SNR indicators ---
    // Weight each heuristic and average
    float combinedSNR = 0.4f * energySNR + 0.3f * flatnessSNR + 0.3f * hnr;
    mLastSNR = combinedSNR;

    // Map to training weight
    float rawWeight = snrToWeight(combinedSNR);
    return smoothWeight(rawWeight);
}

// =============================================================================
// Batch Processing
// =============================================================================

void SNRPredictor::estimateWeights(
    const float* audio,
    int32_t totalSamples,
    float* weightsOut,
    int32_t* numFramesOut)
{
    int32_t frameSize = mConfig.frameSize;
    int32_t numFrames = totalSamples / frameSize;

    for (int32_t f = 0; f < numFrames; ++f) {
        const float* frame = audio + f * frameSize;
        weightsOut[f] = estimateWeight(frame, frameSize);
    }

    if (numFramesOut) {
        *numFramesOut = numFrames;
    }

    LOGI("Batch SNR estimation complete: %d frames processed", numFrames);
}

} // namespace training
} // namespace less
