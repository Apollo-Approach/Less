// =============================================================================
// alert_detector.h — Smart Alert Pass-Through Detector
// =============================================================================
// Phase 11: Detects sudden, high-energy transient sounds that should
// punch through the mask even during heavy masking — fire alarms,
// car horns, someone shouting, breaking glass, etc.
//
// Architecture:
//   Maintains a rolling RMS baseline of the ambient noise floor.
//   When a sudden energy spike exceeds the baseline by >12dB within
//   a 50ms window, the detector fires and returns a "duck" factor
//   that the audio engine uses to temporarily reduce mask volume.
//
//   The duck factor ramps from 1.0 (full mask) → 0.1 (90% mask reduction)
//   over ~20ms, holds for the duration of the alert, then smoothly
//   recovers over ~500ms.
//
// CRITICAL: All memory pre-allocated. Zero allocations on the hot path.
// =============================================================================

#pragma once

#include <cstdint>
#include <atomic>
#include <vector>

namespace less {

class AlertDetector {
public:
    AlertDetector();
    ~AlertDetector();

    AlertDetector(const AlertDetector&) = delete;
    AlertDetector& operator=(const AlertDetector&) = delete;

    // =========================================================================
    // Initialization
    // =========================================================================
    bool initialize(int32_t sampleRate);

    // =========================================================================
    // Hot-path function — called from onAudioReady()
    // =========================================================================
    // Analyzes `input` for sudden transient energy spikes.
    // Returns a "duck factor" in [0.0, 1.0]:
    //   1.0 = no alert, mask at full volume
    //   0.0 = max alert, mask completely ducked
    //
    // ALLOCATION-FREE. LOCK-FREE. Safe for real-time audio thread.
    float process(const float* input, int32_t numFrames);

    // =========================================================================
    // Query
    // =========================================================================
    bool isAlertActive() const;

    // Sensitivity: higher = more sensitive to transients
    // Range: 0.5 (very sensitive) to 3.0 (only very loud alerts)
    void setSensitivity(float sensitivity);

private:
    int32_t mSampleRate{0};
    bool mInitialized{false};

    // =========================================================================
    // Energy Tracking
    // =========================================================================
    // Short-term energy window (~50ms) vs long-term baseline (~2s)
    float mShortTermRms{0.0f};
    float mLongTermRms{0.0f};

    float mShortTermAlpha{0.0f};   // Smoothing for 50ms window
    float mLongTermAlpha{0.0f};    // Smoothing for 2s baseline

    // Alert threshold: spike must exceed baseline by this ratio
    // Default: 4.0 (≈12dB above baseline)
    float mThresholdRatio{4.0f};

    // =========================================================================
    // Duck Envelope
    // =========================================================================
    float mDuckFactor{1.0f};       // Current duck level
    float mDuckTarget{1.0f};       // Target duck level
    float mDuckAttackCoeff{0.0f};  // Per-sample ramp down (fast, ~20ms)
    float mDuckReleaseCoeff{0.0f}; // Per-sample ramp up (slow, ~500ms)

    // Hold counter: keep ducking for at least N samples after spike
    int32_t mHoldCounter{0};
    int32_t mHoldSamples{0};  // ~200ms hold

    // =========================================================================
    // State
    // =========================================================================
    std::atomic<bool> mAlertActive{false};
};

} // namespace less
