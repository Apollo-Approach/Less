// =============================================================================
// alert_detector.cpp — Smart Alert Pass-Through Implementation
// =============================================================================
// Phase 11: Detects sudden transient energy spikes and returns a duck
// factor for temporarily reducing mask volume.
//
// The detector uses dual-timescale RMS tracking:
//   - Short-term (~50ms): captures sudden spikes
//   - Long-term (~2s): tracks the ambient noise floor baseline
//
// When short-term / long-term > threshold, an alert fires.
// =============================================================================

#include "alert_detector.h"

#include <android/log.h>
#include <cmath>
#include <algorithm>

#define LOG_TAG "LESS_Alert"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)

namespace less {

AlertDetector::AlertDetector() = default;
AlertDetector::~AlertDetector() = default;

bool AlertDetector::initialize(int32_t sampleRate) {
    mSampleRate = sampleRate;

    // Short-term window: ~50ms
    float shortMs = 50.0f;
    float shortSamples = shortMs * 0.001f * static_cast<float>(sampleRate);
    mShortTermAlpha = 1.0f - expf(-1.0f / shortSamples);

    // Long-term baseline: ~2s
    float longMs = 2000.0f;
    float longSamples = longMs * 0.001f * static_cast<float>(sampleRate);
    mLongTermAlpha = 1.0f - expf(-1.0f / longSamples);

    // Duck envelope coefficients
    // Attack: 20ms (fast drop)
    float attackMs = 20.0f;
    float attackSamples = attackMs * 0.001f * static_cast<float>(sampleRate);
    mDuckAttackCoeff = 1.0f - expf(-1.0f / attackSamples);

    // Release: 500ms (slow recovery)
    float releaseMs = 500.0f;
    float releaseSamples = releaseMs * 0.001f * static_cast<float>(sampleRate);
    mDuckReleaseCoeff = 1.0f - expf(-1.0f / releaseSamples);

    // Hold: 200ms minimum duck duration
    mHoldSamples = static_cast<int32_t>(200.0f * 0.001f *
                                         static_cast<float>(sampleRate));
    mHoldCounter = 0;

    // Default sensitivity threshold
    mThresholdRatio = 4.0f;  // 12dB above baseline

    mShortTermRms = 0.0f;
    mLongTermRms = 0.001f;  // Small nonzero to prevent div-by-zero
    mDuckFactor = 1.0f;
    mDuckTarget = 1.0f;

    mInitialized = true;
    LOGI("AlertDetector initialized: sampleRate=%d, thresholdRatio=%.1f, "
         "holdMs=200", sampleRate, mThresholdRatio);

    return true;
}

void AlertDetector::setSensitivity(float sensitivity) {
    // sensitivity 0.5 → very sensitive (threshold = 2.0)
    // sensitivity 1.0 → default (threshold = 4.0)
    // sensitivity 3.0 → only loud alerts (threshold = 12.0)
    mThresholdRatio = std::clamp(sensitivity, 0.5f, 3.0f) * 4.0f;
}

bool AlertDetector::isAlertActive() const {
    return mAlertActive.load(std::memory_order_relaxed);
}

float AlertDetector::process(const float* input, int32_t numFrames) {
    if (!mInitialized) return 1.0f;  // Pass-through

    for (int32_t i = 0; i < numFrames; ++i) {
        float sample = input[i];
        float energy = sample * sample;

        // Update dual-timescale RMS trackers
        mShortTermRms += mShortTermAlpha * (energy - mShortTermRms);
        mLongTermRms  += mLongTermAlpha  * (energy - mLongTermRms);

        // Prevent long-term from going too low (avoid false triggers in silence)
        float baseline = std::max(mLongTermRms, 1e-6f);

        // Check for transient spike
        float ratio = mShortTermRms / baseline;

        if (ratio > mThresholdRatio) {
            // ALERT: sudden energy spike detected
            mDuckTarget = 0.1f;  // Duck to 10% mask volume
            mHoldCounter = mHoldSamples;
            mAlertActive.store(true, std::memory_order_relaxed);
        } else if (mHoldCounter > 0) {
            // Still in hold period
            mHoldCounter--;
        } else {
            // No alert — release
            mDuckTarget = 1.0f;
            mAlertActive.store(false, std::memory_order_relaxed);
        }

        // Smooth duck envelope
        if (mDuckFactor > mDuckTarget) {
            mDuckFactor += mDuckAttackCoeff * (mDuckTarget - mDuckFactor);
        } else {
            mDuckFactor += mDuckReleaseCoeff * (mDuckTarget - mDuckFactor);
        }
    }

    return std::clamp(mDuckFactor, 0.0f, 1.0f);
}

} // namespace less
