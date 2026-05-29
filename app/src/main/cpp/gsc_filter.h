// =============================================================================
// gsc_filter.h — Generalized Sidelobe Canceller (Reference-Signal AEC Mode)
// =============================================================================
// Phase 11: Prevents the psychoacoustic mask audio from feeding back into
// the microphone input, which would cause the mask to be re-detected as
// tonal noise and create a runaway feedback loop.
//
// Architecture (Mono BT Stream):
//   The Meta Wearables SDK currently exposes a single mixed mono stream
//   via Bluetooth LC3. Full 5-mic spatial null-steering requires per-mic
//   access (future SDK version). For now, we operate in reference-signal
//   Acoustic Echo Cancellation (AEC) mode:
//
//     Primary input:   Mono BT mic stream (contains ambient + speaker leak)
//     Reference signal: Our mask output (the known speaker signal)
//     Output:          Cleaned ambient-only signal
//
//   This is functionally identical to how phone AEC removes far-end echo,
//   except our "far end" is our own mask playing through the glasses speakers.
//
// Algorithm: Normalized LMS (NLMS) Adaptive Filter
//   - 128 taps @ 48kHz ≈ 2.7ms capture range
//   - Sufficient for speaker-to-mic propagation within the glasses frame
//   - Step size μ = 0.05 (convergence/stability tradeoff)
//   - Power normalization prevents divergence on low-energy signals
//
// CRITICAL: All memory pre-allocated. Zero allocations on the hot path.
// =============================================================================

#pragma once

#include <cstdint>
#include <vector>

namespace less {

class GscFilter {
public:
    GscFilter();
    ~GscFilter();

    // Non-copyable
    GscFilter(const GscFilter&) = delete;
    GscFilter& operator=(const GscFilter&) = delete;

    // =========================================================================
    // Initialization — ALL memory allocated here
    // =========================================================================
    bool initialize(int32_t sampleRate);

    // =========================================================================
    // Hot-path function — called from onAudioReady()
    // =========================================================================
    // Takes the raw mic input and the known speaker output (our mask),
    // and returns a cleaned signal with the speaker component removed.
    //
    // @param micInput         Raw mono microphone input (ambient + speaker leak)
    // @param speakerReference The mask signal we are currently playing
    // @param cleanOutput      Output: ambient-only signal (speaker removed)
    // @param numFrames        Number of samples
    //
    // ALLOCATION-FREE. LOCK-FREE. Safe for real-time audio thread.
    void process(
        const float* micInput,
        const float* speakerReference,
        float* cleanOutput,
        int32_t numFrames
    );

    // =========================================================================
    // Configuration
    // =========================================================================
    // NLMS step size (μ). Higher = faster convergence, lower = more stable.
    // Typical range: 0.01–0.1
    void setAdaptationRate(float mu);
    float getAdaptationRate() const;

    // Reset the filter taps (e.g., after mode change)
    void reset();

private:
    int32_t mSampleRate{0};
    bool mInitialized{false};

    // =========================================================================
    // NLMS Adaptive Filter
    // =========================================================================
    // Filter length: 128 taps covers ~2.7ms at 48kHz.
    // This is sufficient for the speaker-to-mic acoustic path within
    // the rigid glasses frame (physical distance ≈ 5-8cm, ~0.2ms).
    // Extra taps handle reflections from the wearer's head/ears.
    static constexpr int32_t kFilterLength = 128;

    // Regularization constant to prevent division by zero
    static constexpr float kEpsilon = 1e-8f;

    // Adaptive filter taps (pre-allocated, updated each sample)
    std::vector<float> mFilterTaps;

    // Circular buffer for the reference signal history
    std::vector<float> mRefBuffer;
    int32_t mRefWritePos{0};

    // NLMS step size
    float mMu{0.05f};
};

} // namespace less
