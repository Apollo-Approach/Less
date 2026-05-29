// =============================================================================
// vision_synth.h — Generative Music Synthesizer for Vision-to-Music Mode
// =============================================================================
// Transforms scene parameters (density, valence, arousal, timbre) into
// coherent, flowing generative music through the existing Oboe pipeline.
//
// Architecture:
//   Scene parameters arrive every ~30 seconds from YOLO inference on the
//   Kotlin side. The synth maintains musical continuity between updates
//   using key memory, melodic contour tracking, and portamento glides.
//
// Music Theory Engine:
//   - Key center persists across scene updates (only shifts when valence
//     changes significantly, using the circle of fifths for smooth modulation)
//   - Chord progressions follow voice-leading rules (closest-note movement)
//   - Melody notes are chosen from the active scale via contour shaping
//     (tendency toward stepwise motion with occasional leaps)
//   - Rhythm patterns phase-lock to an internal pulse grid
//
// Three Interpretations:
//   0 = Ambient Drift   — warm evolving pads, slow glides, reverb tail
//   1 = Melodic Arpeggio — plucked/chimed arpeggiated patterns
//   2 = Rhythmic Pulse   — pulsing bass + rhythmic gate patterns
//
// Constraints:
//   - Lock-free, allocation-free in the audio callback
//   - All float arithmetic (no double)
//   - No stdlib calls that allocate (no std::string, no new/delete)
//   - Output is mono float, tanhf-saturated
// =============================================================================

#pragma once

#include <cstdint>
#include <cmath>
#include <atomic>

namespace less {

// ─────────────────────────────────────────────────────────────────────────────
// Musical Interpretation — selectable by user
// ─────────────────────────────────────────────────────────────────────────────
enum class MusicInterpretation : int32_t {
    kAmbientDrift    = 0,
    kMelodicArpeggio = 1,
    kRhythmicPulse   = 2
};

class VisionSynth {
public:
    VisionSynth();
    ~VisionSynth() = default;

    // Must be called once with the audio system's sample rate
    void initialize(int32_t sampleRate);

    // Called from any thread — atomically pushes new scene parameters.
    // The synth smoothly transitions to the new musical state over the
    // next few seconds — never snaps, never clicks.
    void updateSceneParameters(float density, float valence,
                               float arousal, float timbre);

    // Phase 19: Synesthesia and Transitions
    void updateSynesthesiaParams(float hue, float saturation, float value);
    void flushGenerativeState();

    // Apply neural inference output
    void applyNeuralData(const float* data, int32_t len);

    // Switch interpretation (Ambient / Melodic / Rhythmic)
    void setInterpretation(int32_t interpretation);
    int32_t getInterpretation() const;

    // Switch synthesis quality (0 = Battery Saver, 1 = Beautiful/High-Headroom)
    void setSynthQuality(int32_t qualityLevel);
    int32_t getSynthQuality() const;

    // Set master gain (0.0 – 1.0)
    void setLevel(float level);

    // Called on the Oboe real-time audio thread.
    // Writes numFrames of mono float output into `output`.
    // MUST be allocation-free, lock-free, no syscalls.
    void synthesize(float* output, int32_t numFrames);

    // Query for UI visualization
    float getCurrentBpm() const;
    int32_t getCurrentKey() const;       // 0=C, 1=C#, ... 11=B
    const char* getCurrentChordName() const;

private:
    int32_t mSampleRate{48000};
    float mInvSampleRate{1.0f / 48000.0f};
    bool mInitialized{false};

    // ═════════════════════════════════════════════════════════════════════════
    // PRNG — xorshift32, deterministic, no syscalls
    // ═════════════════════════════════════════════════════════════════════════
    uint32_t mRng{0xDEADBEEF};
    float randomFloat();       // [0.0, 1.0)
    float randomBipolar();     // [-1.0, 1.0)
    int32_t randomInt(int32_t max);  // [0, max)

    // ═════════════════════════════════════════════════════════════════════════
    // Scene Parameters — smoothed toward targets
    // ═════════════════════════════════════════════════════════════════════════
    // Atomic targets (written from any thread)
    std::atomic<float> mTargetDensity{0.5f};
    std::atomic<float> mTargetValence{0.5f};
    std::atomic<float> mTargetArousal{0.3f};
    std::atomic<float> mTargetTimbre{0.5f};
    
    // Neural Tweaker offsets
    std::atomic<float> mNeuralTweakDensity{0.0f};
    std::atomic<float> mNeuralTweakValence{0.0f};
    std::atomic<float> mNeuralTweakArousal{0.0f};
    std::atomic<float> mNeuralTweakTimbre{0.0f};
    
    // Neural Composer weights (16 notes)
    std::atomic<float> mNeuralComposerNotes[16];
    std::atomic<float> mTargetLevel{0.7f};
    std::atomic<int32_t> mInterpretation{0};
    std::atomic<int32_t> mSynthQuality{1}; // defaults to beautiful

    // Phase 19: Synesthesia Targets
    std::atomic<float> mTargetHue{0.0f};
    std::atomic<float> mTargetSaturation{0.0f};
    std::atomic<float> mTargetValue{0.0f};

    // Phase 19: Sequence invalidation trigger via lock-free atomic atomic
    std::atomic<bool> mPhraseNeedsFlush{false};

    // Micro-benchmarking metadata
    uint64_t mLastSynthDurationUs{0};

    // Smoothed current values (only touched on audio thread)
    float mDensity{0.5f};
    float mValence{0.5f};
    float mArousal{0.3f};
    float mTimbre{0.5f};
    float mLevel{0.7f};

    // Smoothing rate per sample (~3 second transition at 48kHz)
    static constexpr float kParamSmooth = 0.00001f;

    void smoothParameters();

    // ═════════════════════════════════════════════════════════════════════════
    // Music Theory Engine
    // ═════════════════════════════════════════════════════════════════════════

    // --- Key Management ---
    // Key center (0-11, semitones above C). Persists across scene updates.
    // Only shifts when valence changes by > 0.3, using circle-of-fifths
    // modulation (up a 5th for brighter, down a 5th for darker).
    int32_t mCurrentKey{0};         // 0=C
    float mKeyChangeThreshold{0.3f};
    float mLastValenceAtKeyChange{0.5f};

    void evaluateKeyChange();

    // --- Scale System ---
    // Scales are chosen based on valence:
    //   High valence (>0.7) → major / lydian
    //   Mid valence  (0.3-0.7) → mixolydian / dorian
    //   Low valence  (<0.3) → minor / phrygian
    enum class ScaleType : int32_t {
        kMajor = 0,      // 1 0 1 0 1 1 0 1 0 1 0 1
        kMixolydian,      // 1 0 1 0 1 1 0 1 0 1 1 0
        kDorian,          // 1 0 1 1 0 1 0 1 0 1 1 0
        kMinor,           // 1 0 1 1 0 1 0 1 1 0 1 0
        kPhrygian,        // 1 1 0 1 0 1 0 1 1 0 1 0
        kLydian,          // 1 0 1 0 1 0 1 1 0 1 0 1
        kPentatonicMaj,   // 1 0 1 0 1 0 0 1 0 1 0 0
        kPentatonicMin,   // 1 0 0 1 0 1 0 1 0 0 1 0
        kNumScales
    };

    static constexpr int32_t kScalePatterns[8][12] = {
        {1,0,1,0,1,1,0,1,0,1,0,1},  // Major (Ionian)
        {1,0,1,0,1,1,0,1,0,1,1,0},  // Mixolydian
        {1,0,1,1,0,1,0,1,0,1,1,0},  // Dorian
        {1,0,1,1,0,1,0,1,1,0,1,0},  // Natural Minor (Aeolian)
        {1,1,0,1,0,1,0,1,1,0,1,0},  // Phrygian
        {1,0,1,0,1,0,1,1,0,1,0,1},  // Lydian
        {1,0,1,0,1,0,0,1,0,1,0,0},  // Pentatonic Major
        {1,0,0,1,0,1,0,1,0,0,1,0},  // Pentatonic Minor
    };

    ScaleType mCurrentScale{ScaleType::kMajor};

    // Cached scale degree → MIDI note mappings for current key+scale
    // This avoids repeated computation in the audio callback.
    // mScaleNotes[octave * scaleLen + degree] = MIDI note number
    static constexpr int32_t kMaxScaleNotes = 64;
    int32_t mScaleNotes[kMaxScaleNotes]{};
    int32_t mScaleNoteCount{0};

    void rebuildScaleTable();
    ScaleType selectScaleForValence(float valence) const;
    bool isInScale(int32_t midiNote) const;
    int32_t nearestScaleNote(int32_t midiNote) const;

    // MIDI note → frequency
    static float midiToFreq(int32_t note);

    // --- Chord Progression ---
    // Chords are built from scale degrees using standard triads/7ths.
    // Progressions follow common functional harmony patterns.
    static constexpr int32_t kMaxChordNotes = 6;

    struct Chord {
        int32_t notes[kMaxChordNotes]{};  // MIDI notes
        int32_t noteCount{0};
        int32_t rootDegree{0};            // Scale degree (0-6)
        bool isBorrowed{false};           // Tracks modal interchange 
    };

    // Common chord progressions (as scale degree sequences)
    // Each progression is 4-8 chords long, looping.
    static constexpr int32_t kMaxProgLen = 8;

    struct ChordProgression {
        int32_t degrees[kMaxProgLen];     // Root scale degrees (0-based)
        int32_t length;
        // Chord quality override per step: 0=auto (from scale), 1=force major,
        // 2=force minor, 3=force7, 4=sus4, 5=force maj7
        int32_t qualities[kMaxProgLen];
    };

    // Progressions indexed by mood — valence selects which bank
    static constexpr int32_t kNumProgressions = 12;
    static const ChordProgression kProgressions[kNumProgressions];

    int32_t mProgIndex{0};
    int32_t mProgStep{0};
    Chord mCurrentChord{};
    Chord mNextChord{};
    float mChordTimer{0.0f};
    float mChordDuration{4.0f};     // Beats per chord
    float mChordCrossfade{0.0f};    // 0 = current, 1 = next
    static constexpr float kChordCrossfadeDuration = 0.5f; // seconds

    void advanceProgression();
    Chord buildChord(int32_t rootDegree, int32_t quality, int32_t octave);
    void selectProgression();

    // --- Melody Engine ---
    // Generates a melodic line over the chord progression using
    // contour-based pitch selection: prefers stepwise motion,
    // resolves to chord tones on strong beats, allows passing tones.
    int32_t mMelodyNote{60};        // Current MIDI note
    int32_t mMelodyTarget{60};      // Where melody is heading
    float mMelodyPhase{0.0f};       // Sub-beat phase for note timing
    float mMelodyGatePhase{0.0f};   // For note-on/off envelope
    float mMelodyNoteDuration{0.25f}; // In beats
    int32_t mMelodyContour{0};      // -1 = descending, 0 = static, 1 = ascending
    int32_t mMelodyOctave{4};       // Current octave region

    // Motif Memory
    int32_t mMotifBuffer[64]{};     // Max buffer length
    int32_t mMotifIndex{0};         // Current index
    int32_t mCurrentMotifLength{8}; // Dynamic length (4-64)
    float mValenceVelocity{0.0f};   // Monitor for stability to lock motif

    int32_t chooseMelodyNote();
    bool isChordTone(int32_t midiNote) const;

    // --- Tempo / Pulse ---
    float mBpm{72.0f};
    float mTargetBpm{72.0f};
    float mBeatPhase{0.0f};       // 0-1 within current beat
    float mBarPhase{0.0f};        // 0-1 within current bar (4 beats)
    int32_t mBeatCount{0};        // Global beat counter
    float mSwing{0.0f};           // 0 = straight, 0.3 = subtle swing

    // Euclidean rhythm state tracking
    int32_t mRhythmicStep[2]{0};

    void advanceClock(int32_t numFrames);
    float getSecondsPerBeat() const;

    // ═════════════════════════════════════════════════════════════════════════
    // Voice Architecture
    // ═════════════════════════════════════════════════════════════════════════
    // 8 multi-purpose voices. Each interpretation uses them differently:
    //   Ambient:  all voices as slow-evolving pad layers
    //   Melodic:  2 pad voices + 4 arpeggio voices + 1 bass + 1 melody
    //   Rhythmic: 2 bass pulse voices + 4 rhythmic gate voices + 2 pad

    static constexpr int32_t kMaxVoices = 8;

    struct Voice {
        float phase{0.0f};
        float currentFreq{0.0f};
        float targetFreq{0.0f};
        float amplitude{0.0f};
        float targetAmplitude{0.0f};
        float pan{0.0f};            // -1.0 left, +1.0 right
        
        // Filter states
        float filterState1{0.0f};
        float filterState2{0.0f};
        float filterState3{0.0f};
        float filterState4{0.0f};
        float filterCutoff{1.0f};   // 0.0 = fully closed, 1.0 = open

        // Portamento speed (per-sample, exponential)
        float portamento{0.00005f};

        // Envelope
        float envPhase{0.0f};       // 0 = attack start, 1 = sustain, 2 = release
        float envLevel{0.0f};
        float attackRate{0.001f};   // Per-sample increment
        float releaseRate{0.0005f}; // Per-sample decrement

        // Vibrato
        float vibratoPhase{0.0f};
        float vibratoRate{2.0f};
        float vibratoDepth{3.0f};   // Cents

        // Voice role hint (for interpretation routing)
        int32_t role{0};            // 0=pad, 1=arp, 2=bass, 3=melody
    };

    Voice mVoices[kMaxVoices]{};

    // Waveform generation
    float oscillator(float phase, float timbre) const;
    float filterResonant(Voice& voice, float input) const;
    float filterLowpass(Voice& voice, float input) const;

    // Per-interpretation synthesis
    void synthesizeAmbient(float* output, int32_t numFrames);
    void synthesizeMelodic(float* output, int32_t numFrames);
    void synthesizeRhythmic(float* output, int32_t numFrames);

    // Voice management
    void assignVoicesForInterpretation();
    void updateVoiceTargets();

    // ═════════════════════════════════════════════════════════════════════════
    // Effects
    // ═════════════════════════════════════════════════════════════════════════

    // Simple feedback delay for ambient depth
    static constexpr int32_t kDelayMaxSamples = 48000; // 1 second at 48kHz
    float mDelayBuffer[kDelayMaxSamples]{};
    int32_t mDelayWritePos{0};
    float mDelayTime{0.4f};         // Seconds
    float mDelayFeedback{0.3f};     // 0-0.9
    float mDelayMix{0.2f};          // Wet/dry

    float readDelay(float delaySec) const;
    void writeDelay(float sample);

    // Soft saturation
    static float saturate(float x);
};

} // namespace less
