// =============================================================================
// masking_engine.h — Psychoacoustic Mask Generator
// =============================================================================
// Phase 11: Instead of cancelling noise (physically impossible over BT),
// we trick the brain into ignoring it by generating a pleasant acoustic
// mask targeted at detected tonal noise frequencies.
//
// Architecture:
//   1. Rolling spectral analysis detects persistent tonal components
//      (AC hum, fan whine, fridge drone) via exponential-decay averaging
//   2. A filtered noise synthesizer generates band-limited masking audio
//      covering only the detected tonal regions
//   3. A smooth gain envelope (attack 200ms, release 1500ms) prevents
//      jarring on/off transitions
//   4. A hard safety limiter at -12 dBFS protects the wearer's hearing
//
// Texture Profiles:
//   - Brown noise: -6dB/octave roll-off, warm and deep
//   - Pink noise:  -3dB/octave roll-off, balanced and natural
//   - White noise: flat spectrum, bright masking
//   - Nature:      filtered brown noise with subtle amplitude modulation
//                  to simulate rain/ocean texture
//   - Harmonic:    Detects the room's tonal frequency and generates a
//                  slowly drifting chord progression using just-intonation
//                  ratios, turning ambient drone into generative ambient music
//
// CRITICAL: All memory pre-allocated in initialize(). The generate()
// function is ALLOCATION-FREE and LOCK-FREE for real-time audio safety.
// =============================================================================

#pragma once

#include <cstdint>
#include <atomic>
#include <vector>
#include <cmath>

namespace less {

// Mask texture profiles
enum class MaskTexture : int32_t {
    kBrownNoise = 0,   // Warm, deep — best for low-frequency drones
    kPinkNoise  = 1,   // Balanced — good general-purpose masking
    kWhiteNoise = 2,   // Bright — effective for high-frequency whines
    kNature     = 3,   // Modulated brown noise — rain/ocean texture
    kHarmonic   = 4    // Generative chord progression from room drone
};

class MaskingEngine {
public:
    MaskingEngine();
    ~MaskingEngine();

    // Non-copyable
    MaskingEngine(const MaskingEngine&) = delete;
    MaskingEngine& operator=(const MaskingEngine&) = delete;

    // =========================================================================
    // Initialization — ALL memory allocated here
    // =========================================================================
    bool initialize(int32_t sampleRate);

    // =========================================================================
    // Hot-path function — called from onAudioReady()
    // =========================================================================
    // Analyzes `input` spectrum to detect tonal noise, then writes a
    // masking signal into `maskOutput`. Both buffers are `numFrames` long.
    //
    // ALLOCATION-FREE. LOCK-FREE. Safe for real-time audio thread.
    void analyzeAndGenerate(
        const float* input,
        float* maskOutput,
        int32_t numFrames
    );

    // =========================================================================
    // Runtime configuration — safe from any thread (atomic)
    // =========================================================================
    void setMaskLevel(float level);    // 0.0 = off, 1.0 = full mask
    float getMaskLevel() const;

    // Get the detected fundamental tonal frequency (Hz), 0 if none
    float getDetectedFundamental() const;

    void setTexture(MaskTexture texture);
    MaskTexture getTexture() const;

    // Query whether the mask is currently generating sound
    bool isActive() const;

    // Get detected environment tonal profile (for UI visualization)
    // Returns the number of active tonal bands (0 = no tonal noise detected)
    int32_t getActiveTonalBandCount() const;

    static const char* textureName(MaskTexture texture);

private:
    int32_t mSampleRate{0};
    bool mInitialized{false};

    // =========================================================================
    // Tonal Noise Detector
    // =========================================================================
    // Rolling spectral average with exponential decay (τ ≈ 2 seconds).
    // Bins whose magnitude stays above a threshold for > 500ms are
    // classified as "tonal" (continuous noise worth masking).
    static constexpr int32_t kFftSize = 512;
    static constexpr int32_t kNumBins = kFftSize / 2 + 1;
    static constexpr int32_t kHopSize = 256;

    // Persistence threshold: bin must stay above noise floor for this
    // many consecutive analysis frames to be classified as tonal
    static constexpr int32_t kPersistenceFrames = 24;  // ~500ms at 48kHz/256hop

    std::vector<float> mSpectralAverage;     // Exponential running average per bin
    std::vector<float> mSpectralPeak;        // Peak tracker for tonal detection
    std::vector<int32_t> mPersistenceCount;  // Consecutive frames above threshold
    std::vector<bool>  mTonalMask;           // Which bins are classified as tonal

    float mDecayAlpha{0.0f};    // Exponential decay coefficient
    float mTonalThreshold{0.0f}; // Adaptive threshold for tonal classification

    // FFT scratch buffers (pre-allocated)
    std::vector<float> mFftReal;
    std::vector<float> mFftImag;
    std::vector<float> mWindowCoeffs;

    // Ring buffer for accumulating input frames
    std::vector<float> mInputRing;
    int32_t mRingWritePos{0};
    int32_t mFramesAccumulated{0};

    // =========================================================================
    // Noise Synthesizer
    // =========================================================================
    // Generates the masking noise texture. The raw noise is spectrally
    // shaped according to the selected profile, then band-passed to
    // cover only the detected tonal regions.

    // PRNG state for noise generation (xorshift32 — fast, no syscalls)
    uint32_t mRngState{0x12345678};

    // Brown noise state (integrator)
    float mBrownState{0.0f};

    // Nature texture modulation (slow LFO for rain/ocean feel)
    float mNatureLfoPhase{0.0f};
    static constexpr float kNatureLfoFreq = 0.3f;  // 0.3 Hz modulation

    // =========================================================================
    // Harmonic Drone Synthesizer (v3 — environment-reactive portamento)
    // =========================================================================
    // Listens to the environment via mic analysis to detect persistent
    // tonal drones (AC units, engine hum, appliance buzz).
    // Every ~5 seconds, snapshots the detected frequency, octave-folds
    // it to a pleasant low register, and builds a consonant chord on top.
    // Portamento glides voices smoothly to new targets = zero clicks.
    //
    // Feedback prevention: The GSC (NLMS adaptive) filter in audio_engine.cpp
    // subtracts our known speaker output from the mic input before it reaches
    // this engine, ensuring we only detect ambient noise, not our own output.
    //
    // Design:
    //   1. Mic → GSC cleanup → FFT → tonal detection runs continuously
    //   2. Every kListenInterval seconds, snapshot mDetectedFundamental
    //   3. Octave-fold to 80-220 Hz (warm register)
    //   4. Pick consonant chord, set target frequencies
    //   5. Portamento glides voices there over 3-5 seconds

    static constexpr int32_t kDroneVoices = 4;     // root, 3rd, 5th, sub
    static constexpr int32_t kNumChords = 8;

    // How often to re-sample the environment (seconds)
    static constexpr float kListenInterval = 5.0f;

    // How long each chord voicing lasts before changing quality
    static constexpr float kChordChangeMin = 20.0f;
    static constexpr float kChordChangeMax = 40.0f;

    // Portamento smoothing per sample — lower = slower glide.
    // At 48kHz, 0.00003 gives ~3-5 second glide time.
    static constexpr float kPortamentoRate = 0.00003f;

    // Fallback root when no tonal noise is detected
    static constexpr float kFallbackRoot = 130.81f;  // C3

    // Just-intonation voicing ratios: [root, 3rd, 5th, sub/octave]
    static constexpr float kChordRatios[kNumChords][kDroneVoices] = {
        { 1.0f,  1.25f,   1.5f,   0.5f  },  // I:   maj triad + sub
        { 1.0f,  1.2f,    1.5f,   0.5f  },  // i:   min triad + sub
        { 1.0f,  1.3333f, 1.5f,   0.5f  },  // IV:  sus4 + sub
        { 1.0f,  1.25f,   1.5f,   2.0f  },  // I8:  maj triad + octave
        { 1.0f,  1.5f,    2.0f,   0.5f  },  // 5:   open 5th + oct + sub
        { 1.0f,  1.2f,    1.8f,   0.5f  },  // i7:  min7 color + sub
        { 1.0f,  1.25f,   1.875f, 0.5f  },  // Imaj7: warm major 7 + sub
        { 1.0f,  1.3333f, 2.0f,   0.5f  },  // sus8: p4 + oct + sub
    };

    struct DroneVoice {
        float phase{0.0f};
        float currentFreq{130.81f};    // Smoothed toward target each sample
        float targetFreq{130.81f};     // Where we're gliding to
        float amplitude{0.0f};
        float vibratoPhase{0.0f};
        float vibratoRate{1.5f};       // Very slow LFO (0.8-2.8 Hz)
        float vibratoDepth{2.0f};      // Subtle ±2 cents
    };

    DroneVoice mDroneVoices[kDroneVoices]{};
    float mDetectedFundamental{0.0f};      // Hz — updated by tonal analysis
    float mLockedRoot{130.81f};            // Octave-folded root, updated every kListenInterval
    int32_t mCurrentChordIndex{0};
    int32_t mPrevChordIndex{-1};
    float mListenTimer{0.0f};              // Seconds since last environment sample
    float mChordTimer{0.0f};               // Seconds since last chord quality change
    float mCurrentChordDuration{25.0f};
    void synthesizeHarmonic(float* output, int32_t numFrames);
    void updateFundamental();
    void advanceChordProgression(int32_t numFrames);

    // Per-bin gain for band-limited masking (computed from tonal mask)
    std::vector<float> mBandGain;

    // Synthesis output buffer (pre-allocated)
    std::vector<float> mSynthBuffer;
    std::vector<float> mSynthFftReal;
    std::vector<float> mSynthFftImag;

    // =========================================================================
    // Dynamic Gain Envelope
    // =========================================================================
    // Smooth attack/release to prevent jarring transitions
    float mCurrentGain{0.0f};
    float mTargetGain{0.0f};
    float mAttackCoeff{0.0f};   // Per-sample attack coefficient
    float mReleaseCoeff{0.0f};  // Per-sample release coefficient

    // =========================================================================
    // Safety Limiter
    // =========================================================================
    static constexpr float kSafetyLimitDb = -12.0f;
    float mSafetyLimitLinear{0.0f};  // Precomputed linear limit

    // =========================================================================
    // Atomic runtime controls
    // =========================================================================
    std::atomic<float> mMaskLevel{0.7f};
    std::atomic<int32_t> mTexture{static_cast<int32_t>(MaskTexture::kBrownNoise)};
    std::atomic<bool> mIsActive{false};
    std::atomic<int32_t> mActiveTonalBands{0};

    // =========================================================================
    // Internal methods
    // =========================================================================
    void computeWindow();
    void analyzeSpectrum(const float* windowedFrame);
    void updateTonalMask();
    void synthesizeMask(float* output, int32_t numFrames);
    float generateWhiteNoise();
    float generateBrownNoise();
    float generatePinkNoise();
    float applyNatureModulation(float sample);
    void applyBandLimiting(float* fftReal, float* fftImag);
    void applySafetyLimiter(float* buffer, int32_t numFrames);
};

} // namespace less
