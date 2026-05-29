// =============================================================================
// gsc_filter.cpp — Generalized Sidelobe Canceller Implementation
// =============================================================================
// Phase 11: NLMS Adaptive Echo Cancellation for mask feedback prevention.
//
// The algorithm is the standard Normalized LMS:
//   1. Build a reference vector from the speaker signal history
//   2. Compute the predicted echo: y_hat = dot(taps, refBuffer)
//   3. Compute the error: e = micInput - y_hat (this is the clean output)
//   4. Update taps: taps += μ * e * refBuffer / (||refBuffer||² + ε)
//
// The filter converges within ~50ms and continuously adapts to changes
// in the acoustic path (head movement, different wearers, etc.)
// =============================================================================

#include "gsc_filter.h"

#include <android/log.h>
#include <cstring>
#include <cmath>
#include <algorithm>

#define LOG_TAG "LESS_GSC"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace less {

// =============================================================================
// Construction / Destruction
// =============================================================================

GscFilter::GscFilter() = default;
GscFilter::~GscFilter() = default;

// =============================================================================
// Initialization
// =============================================================================

bool GscFilter::initialize(int32_t sampleRate) {
    mSampleRate = sampleRate;

    // Pre-allocate filter taps (initialized to zero — no echo estimate yet)
    mFilterTaps.resize(kFilterLength, 0.0f);

    // Pre-allocate reference buffer (circular)
    mRefBuffer.resize(kFilterLength, 0.0f);
    mRefWritePos = 0;

    mMu = 0.05f;  // Default step size

    mInitialized = true;
    LOGI("GscFilter initialized: sampleRate=%d, filterLength=%d taps (%.2f ms), "
         "mu=%.4f",
         sampleRate, kFilterLength,
         1000.0f * kFilterLength / static_cast<float>(sampleRate),
         mMu);

    return true;
}

// =============================================================================
// Configuration
// =============================================================================

void GscFilter::setAdaptationRate(float mu) {
    mMu = std::clamp(mu, 0.001f, 0.5f);
}

float GscFilter::getAdaptationRate() const {
    return mMu;
}

void GscFilter::reset() {
    if (!mInitialized) return;
    std::fill(mFilterTaps.begin(), mFilterTaps.end(), 0.0f);
    std::fill(mRefBuffer.begin(), mRefBuffer.end(), 0.0f);
    mRefWritePos = 0;
    LOGI("GscFilter reset: all taps cleared");
}

// =============================================================================
// Hot Path — process()
// =============================================================================
// ALLOCATION-FREE. LOCK-FREE. Real-time safe.
//
// For each sample:
//   1. Push the speaker reference sample into the circular buffer
//   2. Compute the dot product of filter taps with reference history
//   3. Subtract the predicted echo from the mic input → clean output
//   4. Update filter taps via NLMS rule

void GscFilter::process(
    const float* micInput,
    const float* speakerReference,
    float* cleanOutput,
    int32_t numFrames)
{
    if (!mInitialized) {
        // Pass-through if not initialized
        std::memcpy(cleanOutput, micInput, numFrames * sizeof(float));
        return;
    }

    for (int32_t n = 0; n < numFrames; ++n) {
        // Step 1: Write new reference sample into circular buffer
        mRefBuffer[mRefWritePos] = speakerReference[n];

        // Step 2: Compute predicted echo (dot product of taps × reference)
        // and reference power for normalization
        float echoEstimate = 0.0f;
        float refPower = 0.0f;

        for (int32_t k = 0; k < kFilterLength; ++k) {
            // Read from circular buffer (newest → oldest)
            int32_t refIdx = (mRefWritePos - k + kFilterLength) % kFilterLength;
            float refSample = mRefBuffer[refIdx];

            echoEstimate += mFilterTaps[k] * refSample;
            refPower += refSample * refSample;
        }

        // Step 3: Compute error (mic - predicted echo = clean ambient)
        float error = micInput[n] - echoEstimate;
        cleanOutput[n] = error;

        // Step 4: Update filter taps (NLMS rule)
        // Δw = μ * error * x / (||x||² + ε)
        float normFactor = mMu / (refPower + kEpsilon);

        for (int32_t k = 0; k < kFilterLength; ++k) {
            int32_t refIdx = (mRefWritePos - k + kFilterLength) % kFilterLength;
            mFilterTaps[k] += normFactor * error * mRefBuffer[refIdx];
        }

        // Advance circular buffer write position
        mRefWritePos = (mRefWritePos + 1) % kFilterLength;
    }
}

} // namespace less
