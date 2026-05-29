// =============================================================================
// environment_classifier.h — Acoustic Environment Auto-Detect
// =============================================================================
// Phase 11: Automatically classifies the acoustic environment based on
// spectral fingerprinting. Uses the rolling spectral data already
// computed by the MaskingEngine's tonal detector.
//
// Environment Types:
//   - Quiet:   low overall energy, no dominant tonal components
//   - Office:  moderate energy, HVAC hum (50-200Hz), keyboard clicks
//   - Cafe:    high speech energy, clinking, broadband chatter
//   - Transit: rumble under 100Hz, periodic transients (announcements)
//   - Outdoor: wind noise (sub-100Hz broadband), traffic
//
// Algorithm:
//   Simple energy-band classifier — no ML model needed.
//   Divides the spectrum into 4 bands (sub-bass, bass, mid, high)
//   and classifies based on energy ratios. Updated every ~500ms.
//
// CRITICAL: All memory pre-allocated. Zero allocations on the hot path.
// =============================================================================

#pragma once

#include <cstdint>
#include <atomic>

namespace less {

enum class Environment : int32_t {
    kQuiet   = 0,
    kOffice  = 1,
    kCafe    = 2,
    kTransit = 3,
    kOutdoor = 4
};

class EnvironmentClassifier {
public:
    EnvironmentClassifier();
    ~EnvironmentClassifier();

    EnvironmentClassifier(const EnvironmentClassifier&) = delete;
    EnvironmentClassifier& operator=(const EnvironmentClassifier&) = delete;

    // =========================================================================
    // Initialization
    // =========================================================================
    bool initialize(int32_t sampleRate);

    // =========================================================================
    // Analysis — call periodically with aggregated spectral data
    // =========================================================================
    // Accepts the spectral average from MaskingEngine (kNumBins floats)
    // and classifies the current environment.
    //
    // Call ~2x per second (every 500ms) from a non-RT thread, or from
    // the audio thread if the data is already available.
    void classify(const float* spectralAverage, int32_t numBins);

    // =========================================================================
    // Query
    // =========================================================================
    Environment getEnvironment() const;
    const char* getEnvironmentName() const;

    // Get confidence [0.0, 1.0] of the current classification
    float getConfidence() const;

    static const char* environmentName(Environment env);

private:
    int32_t mSampleRate{0};
    bool mInitialized{false};

    // Band boundaries (bin indices, computed from sample rate)
    int32_t mSubBassCutoff{0};   // 0  - 100Hz
    int32_t mBassCutoff{0};      // 100 - 500Hz
    int32_t mMidCutoff{0};       // 500 - 2000Hz
    // Above mMidCutoff: high band (2000Hz+)

    // Classification state (atomic for cross-thread reads)
    std::atomic<int32_t> mEnvironment{
        static_cast<int32_t>(Environment::kQuiet)};
    std::atomic<float> mConfidence{0.0f};

    // Smoothing: track consecutive classifications to debounce
    static constexpr int32_t kStabilityCount = 4;
    int32_t mCandidateEnv{0};
    int32_t mCandidateCount{0};

    void updateAndStabilize(int32_t envValue, float confidence);
};

} // namespace less
