// =============================================================================
// environment_classifier.cpp — Acoustic Environment Auto-Detect
// =============================================================================
// Phase 11: Band-energy-ratio classifier for environment identification.
//
// The classifier divides the spectrum into four bands and uses their
// energy ratios to distinguish environments:
//
//   Quiet:   total energy < threshold
//   Office:  dominant sub-bass/bass (HVAC), low high-freq
//   Cafe:    strong mid-band (speech), moderate broadband
//   Transit: very strong sub-bass (rumble), periodic mid-band
//   Outdoor: strong sub-bass (wind), moderate everything else
//
// A stability counter prevents rapid switching — the environment must
// be consistently classified for 4 consecutive analyses (~2s) before
// the reported value changes.
// =============================================================================

#include "environment_classifier.h"

#include <android/log.h>
#include <cmath>
#include <algorithm>

#define LOG_TAG "LESS_Env"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace less {

EnvironmentClassifier::EnvironmentClassifier() = default;
EnvironmentClassifier::~EnvironmentClassifier() = default;

const char* EnvironmentClassifier::environmentName(Environment env) {
    switch (env) {
        case Environment::kQuiet:   return "Quiet";
        case Environment::kOffice:  return "Office";
        case Environment::kCafe:    return "Cafe";
        case Environment::kTransit: return "Transit";
        case Environment::kOutdoor: return "Outdoor";
        default: return "Unknown";
    }
}

bool EnvironmentClassifier::initialize(int32_t sampleRate) {
    mSampleRate = sampleRate;

    // Compute bin boundaries for a 512-point FFT
    // Bin frequency = bin_index * sampleRate / fftSize
    // fftSize = 512 (from MaskingEngine), so bin width = sampleRate / 512
    float binWidth = static_cast<float>(sampleRate) / 512.0f;

    mSubBassCutoff = static_cast<int32_t>(100.0f / binWidth);   // ~100Hz
    mBassCutoff    = static_cast<int32_t>(500.0f / binWidth);   // ~500Hz
    mMidCutoff     = static_cast<int32_t>(2000.0f / binWidth);  // ~2000Hz

    mCandidateEnv = static_cast<int32_t>(Environment::kQuiet);
    mCandidateCount = 0;

    mInitialized = true;
    LOGI("EnvironmentClassifier initialized: sampleRate=%d, "
         "subBassBin=%d, bassBin=%d, midBin=%d",
         sampleRate, mSubBassCutoff, mBassCutoff, mMidCutoff);

    return true;
}

void EnvironmentClassifier::classify(const float* spectralAverage,
                                      int32_t numBins) {
    if (!mInitialized || numBins < mMidCutoff) return;

    // Compute energy in each band
    float subBassEnergy = 0.0f;
    float bassEnergy = 0.0f;
    float midEnergy = 0.0f;
    float highEnergy = 0.0f;
    float totalEnergy = 0.0f;

    for (int32_t b = 0; b < numBins; ++b) {
        float e = spectralAverage[b] * spectralAverage[b];  // Power
        totalEnergy += e;

        if (b < mSubBassCutoff) {
            subBassEnergy += e;
        } else if (b < mBassCutoff) {
            bassEnergy += e;
        } else if (b < mMidCutoff) {
            midEnergy += e;
        } else {
            highEnergy += e;
        }
    }

    // Prevent division by zero
    if (totalEnergy < 1e-10f) {
        // Essentially silence
        updateAndStabilize(static_cast<int32_t>(Environment::kQuiet), 0.9f);
        return;
    }

    // Compute band ratios
    float subBassRatio = subBassEnergy / totalEnergy;
    float bassRatio    = bassEnergy / totalEnergy;
    float midRatio     = midEnergy / totalEnergy;
    float highRatio    = highEnergy / totalEnergy;
    float lowRatio     = subBassRatio + bassRatio;

    // Classification heuristics
    Environment detected = Environment::kQuiet;
    float confidence = 0.5f;

    if (totalEnergy < 1e-6f) {
        // Very low energy = quiet
        detected = Environment::kQuiet;
        confidence = 0.9f;
    } else if (subBassRatio > 0.45f && midRatio < 0.15f) {
        // Very strong sub-bass rumble, weak mid = transit
        detected = Environment::kTransit;
        confidence = 0.7f + 0.3f * std::min(subBassRatio / 0.6f, 1.0f);
    } else if (subBassRatio > 0.30f && highRatio < 0.10f) {
        // Strong low-end, weak high = outdoor/wind
        detected = Environment::kOutdoor;
        confidence = 0.6f + 0.3f * subBassRatio;
    } else if (midRatio > 0.30f && highRatio > 0.15f) {
        // Strong mid (speech) + high (clinking/chatter) = cafe
        detected = Environment::kCafe;
        confidence = 0.6f + 0.3f * midRatio;
    } else if (lowRatio > 0.40f && midRatio < 0.25f) {
        // Moderate low-end hum, weak mid = office HVAC
        detected = Environment::kOffice;
        confidence = 0.6f + 0.3f * lowRatio;
    } else {
        // Mixed profile — default to office
        detected = Environment::kOffice;
        confidence = 0.4f;
    }

    updateAndStabilize(static_cast<int32_t>(detected), confidence);
}

void EnvironmentClassifier::updateAndStabilize(int32_t envValue,
                                                 float confidence) {
    if (envValue == mCandidateEnv) {
        mCandidateCount++;
    } else {
        mCandidateEnv = envValue;
        mCandidateCount = 1;
    }

    // Only commit after consistent classification
    if (mCandidateCount >= kStabilityCount) {
        int32_t current = mEnvironment.load(std::memory_order_relaxed);
        if (current != envValue) {
            mEnvironment.store(envValue, std::memory_order_relaxed);
            mConfidence.store(confidence, std::memory_order_relaxed);
            LOGI("Environment changed: %s (confidence=%.2f)",
                 environmentName(static_cast<Environment>(envValue)),
                 confidence);
        } else {
            mConfidence.store(confidence, std::memory_order_relaxed);
        }
    }
}

Environment EnvironmentClassifier::getEnvironment() const {
    return static_cast<Environment>(
        mEnvironment.load(std::memory_order_relaxed));
}

const char* EnvironmentClassifier::getEnvironmentName() const {
    return environmentName(getEnvironment());
}

float EnvironmentClassifier::getConfidence() const {
    return mConfidence.load(std::memory_order_relaxed);
}

} // namespace less
