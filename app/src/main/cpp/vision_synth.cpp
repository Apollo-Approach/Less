// =============================================================================
// vision_synth.cpp — Generative Music Synthesizer for Vision-to-Music Mode
// =============================================================================
// Transforms scene parameters into coherent, flowing generative music.
//
// Music Theory Design:
//   - Key center uses circle-of-fifths modulation (smooth transitions)
//   - Chord progressions use functional harmony with voice-leading
//   - Melody uses contour-based selection (stepwise + chord-tone resolution)
//   - Tempo maps to arousal via smooth BPM glide
//   - All transitions use portamento/crossfade — zero discontinuities
//
// This file is large because it contains the full music engine. Sections:
//   1. Initialization & PRNG
//   2. Music Theory (scales, chords, progressions, melody)
//   3. Synthesis (oscillators, voices, effects)
//   4. Three Interpretations (Ambient, Melodic, Rhythmic)
// =============================================================================

#define _USE_MATH_DEFINES
#include "vision_synth.h"

#include <cstring>
#include <android/log.h>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define LOG_TAG "LESS_VisionSynth"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)

namespace less {

// =============================================================================
// Section 1: Chord Progression Definitions
// =============================================================================
// Each progression encodes root scale degrees (0-based) and optional
// quality overrides. quality=0 means "auto" (derived from scale).
//
// Grouped by mood:
//   [0-3]  = Bright / High Valence  (major keys, uplifting movement)
//   [4-7]  = Neutral / Mid Valence  (modal, ambiguous, floating)
//   [8-11] = Dark / Low Valence     (minor keys, tension, gravity)

const VisionSynth::ChordProgression VisionSynth::kProgressions[kNumProgressions] = {
    // ── Bright ────────────────────────────────────────────────────────────
    // I → V → vi → IV  (the "pop" progression, universally pleasing)
    {{0, 4, 5, 3,  0, 0, 0, 0}, 4, {0, 0, 0, 0,  0, 0, 0, 0}},
    // I → IV → V → I  (authentic cadence loop)
    {{0, 3, 4, 0,  0, 0, 0, 0}, 4, {0, 0, 0, 0,  0, 0, 0, 0}},
    // I → iii → vi → IV → I → V → IV → V  (extended bright)
    {{0, 2, 5, 3,  0, 4, 3, 4}, 8, {0, 0, 0, 0,  0, 0, 0, 0}},
    // I → V → IV → V  (triumphant)
    {{0, 4, 3, 4,  0, 0, 0, 0}, 4, {0, 0, 0, 0,  0, 0, 0, 0}},

    // ── Neutral ───────────────────────────────────────────────────────────
    // I → IV → I → V  (open, floating)
    {{0, 3, 0, 4,  0, 0, 0, 0}, 4, {4, 0, 4, 0,  0, 0, 0, 0}},
    // ii → V → I → vi  (jazz turnaround)
    {{1, 4, 0, 5,  0, 0, 0, 0}, 4, {0, 0, 5, 0,  0, 0, 0, 0}},
    // I → bVII → IV → I  (mixolydian vamp)
    {{0, 6, 3, 0,  0, 0, 0, 0}, 4, {0, 1, 0, 0,  0, 0, 0, 0}},
    // vi → IV → I → V  (deceptive, bittersweet)
    {{5, 3, 0, 4,  0, 0, 0, 0}, 4, {0, 0, 0, 0,  0, 0, 0, 0}},

    // ── Dark ──────────────────────────────────────────────────────────────
    // i → VI → III → VII  (natural minor cycle)
    {{0, 5, 2, 6,  0, 0, 0, 0}, 4, {2, 1, 1, 1,  0, 0, 0, 0}},
    // i → iv → v → i  (pure minor)
    {{0, 3, 4, 0,  0, 0, 0, 0}, 4, {2, 2, 2, 2,  0, 0, 0, 0}},
    // i → VII → VI → V  (Andalusian cadence)
    {{0, 6, 5, 4,  0, 0, 0, 0}, 4, {2, 1, 1, 1,  0, 0, 0, 0}},
    // i → iv → VII → III → VI → ii° → V → i  (extended dark)
    {{0, 3, 6, 2,  5, 1, 4, 0}, 8, {2, 2, 1, 1,  1, 0, 1, 2}},
};

// =============================================================================
// Section 2: Initialization
// =============================================================================

VisionSynth::VisionSynth() {
    // Initialize voices with staggered phases to avoid phase coherence artifacts
    for (int32_t i = 0; i < kMaxVoices; ++i) {
        mVoices[i].phase = static_cast<float>(i) / kMaxVoices;
        mVoices[i].pan = -0.7f + 1.4f * (static_cast<float>(i) / (kMaxVoices - 1));
        mVoices[i].vibratoPhase = static_cast<float>(i) * 0.13f;
    }
}

void VisionSynth::initialize(int32_t sampleRate) {
    mSampleRate = sampleRate;
    mInvSampleRate = 1.0f / static_cast<float>(sampleRate);
    mInitialized = true;

    // Build initial scale table
    rebuildScaleTable();

    // Initial voice assignment
    assignVoicesForInterpretation();

    // Build first chord
    selectProgression();
    mCurrentChord = buildChord(
        kProgressions[mProgIndex].degrees[0],
        kProgressions[mProgIndex].qualities[0], 3);
    mNextChord = mCurrentChord;

    // Prime delay buffer
    std::memset(mDelayBuffer, 0, sizeof(mDelayBuffer));

    LOGI("VisionSynth initialized: sr=%d, key=C, scale=Major", sampleRate);
}

// =============================================================================
// Section 3: PRNG
// =============================================================================

float VisionSynth::randomFloat() {
    mRng ^= mRng << 13;
    mRng ^= mRng >> 17;
    mRng ^= mRng << 5;
    return static_cast<float>(mRng & 0xFFFF) / 65536.0f;
}

float VisionSynth::randomBipolar() {
    return randomFloat() * 2.0f - 1.0f;
}

int32_t VisionSynth::randomInt(int32_t max) {
    if (max <= 0) return 0;
    mRng ^= mRng << 13;
    mRng ^= mRng >> 17;
    mRng ^= mRng << 5;
    return static_cast<int32_t>(mRng % static_cast<uint32_t>(max));
}

// =============================================================================
// Section 4: Parameter Smoothing
// =============================================================================

void VisionSynth::updateSceneParameters(float density, float valence,
                                         float arousal, float timbre) {
    mTargetDensity.store(density, std::memory_order_relaxed);
    mTargetValence.store(valence, std::memory_order_relaxed);
    mTargetArousal.store(arousal, std::memory_order_relaxed);
    mTargetTimbre.store(timbre, std::memory_order_relaxed);
}

void VisionSynth::updateSynesthesiaParams(float hue, float saturation, float value) {
    mTargetHue.store(hue, std::memory_order_relaxed);
    mTargetSaturation.store(saturation, std::memory_order_relaxed);
    mTargetValue.store(value, std::memory_order_relaxed);
}

void VisionSynth::flushGenerativeState() {
    mPhraseNeedsFlush.store(true, std::memory_order_release);
}


void VisionSynth::setSynthQuality(int32_t qualityLevel) {
    mSynthQuality.store(qualityLevel, std::memory_order_relaxed);
}

int32_t VisionSynth::getSynthQuality() const {
    return mSynthQuality.load(std::memory_order_relaxed);
}

void VisionSynth::applyNeuralData(const float* data, int32_t len) {
    if (!data) return;
    
    // Parse Tweaker Offsets (0-3)
    if (len >= 4) {
        mNeuralTweakDensity.store(data[0], std::memory_order_relaxed);
        mNeuralTweakValence.store(data[1], std::memory_order_relaxed);
        mNeuralTweakArousal.store(data[2], std::memory_order_relaxed);
        mNeuralTweakTimbre.store(data[3], std::memory_order_relaxed);
    }
    
    // Parse Composer Notes (4-19)
    int32_t maxNotes = (len - 4 < 16) ? (len - 4) : 16;
    for (int32_t i = 0; i < maxNotes; ++i) {
        mNeuralComposerNotes[i].store(data[4 + i], std::memory_order_relaxed);
    }
}

void VisionSynth::setInterpretation(int32_t interp) {
    int32_t clamped = (interp < 0) ? 0 : (interp > 2) ? 2 : interp;
    mInterpretation.store(clamped, std::memory_order_relaxed);
}

int32_t VisionSynth::getInterpretation() const {
    return mInterpretation.load(std::memory_order_relaxed);
}

void VisionSynth::setLevel(float level) {
    mTargetLevel.store(level, std::memory_order_relaxed);
}

void VisionSynth::smoothParameters() {
    float tweakD = mNeuralTweakDensity.load(std::memory_order_relaxed);
    float tweakV = mNeuralTweakValence.load(std::memory_order_relaxed);
    float tweakA = mNeuralTweakArousal.load(std::memory_order_relaxed);
    float tweakT = mNeuralTweakTimbre.load(std::memory_order_relaxed);

    float td = mTargetDensity.load(std::memory_order_relaxed) + tweakD;
    float tv = mTargetValence.load(std::memory_order_relaxed) + tweakV;
    float ta = mTargetArousal.load(std::memory_order_relaxed) + tweakA;
    float tt = mTargetTimbre.load(std::memory_order_relaxed) + tweakT;

    // Phase 19: Map Synesthesia HSV to targets over time
    // Hue (0-360) -> Timbre (color of sound) -> we map (Hue / 360) to tt
    // Saturation (0-1) -> Arousal / Density
    // Value (0-1) -> Density
    float hue = mTargetHue.load(std::memory_order_relaxed);
    float sat = mTargetSaturation.load(std::memory_order_relaxed);
    float val = mTargetValue.load(std::memory_order_relaxed);

    if (val > 0.01f) {
        tt = (tt + (hue / 360.0f)) * 0.5f;
        ta = (ta + sat) * 0.5f;
        td = (td + val) * 0.5f;
    }
    
    // Clamp targets after tweaks
    td = fmaxf(0.0f, fminf(1.0f, td));
    tv = fmaxf(0.0f, fminf(1.0f, tv));
    ta = fmaxf(0.0f, fminf(1.0f, ta));
    tt  = fmaxf(0.0f, fminf(1.0f, tt));

    float tl = mTargetLevel.load(std::memory_order_relaxed);

    mDensity  += (td - mDensity)  * kParamSmooth;
    mValence  += (tv - mValence)  * kParamSmooth;
    mArousal  += (ta - mArousal)  * kParamSmooth;
    mTimbre   += (tt - mTimbre)   * kParamSmooth;
    mLevel    += (tl - mLevel)    * kParamSmooth;
}

// =============================================================================
// Section 5: Music Theory — Key Management
// =============================================================================

void VisionSynth::evaluateKeyChange() {
    float valenceShift = mValence - mLastValenceAtKeyChange;

    // Only modulate key if valence shifted significantly
    if (fabsf(valenceShift) < mKeyChangeThreshold) return;

    mLastValenceAtKeyChange = mValence;

    // Circle of fifths modulation:
    //   Brighter (valence up) → modulate UP a fifth (+7 semitones)
    //   Darker  (valence down) → modulate DOWN a fifth (-7 semitones, = +5)
    if (valenceShift > 0) {
        mCurrentKey = (mCurrentKey + 7) % 12;  // Up a 5th (brighter)
    } else {
        mCurrentKey = (mCurrentKey + 5) % 12;  // Down a 5th (= up a 4th, darker)
    }

    // Update scale type based on new valence
    ScaleType newScale = selectScaleForValence(mValence);
    if (newScale != mCurrentScale) {
        mCurrentScale = newScale;
    }

    rebuildScaleTable();
    selectProgression();

    LOGI("Key change → %d (%s), scale=%d, valence=%.2f",
         mCurrentKey,
         (const char*[]){"C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"}[mCurrentKey],
         static_cast<int32_t>(mCurrentScale), mValence);
}

VisionSynth::ScaleType VisionSynth::selectScaleForValence(float valence) const {
    if (valence > 0.8f)  return ScaleType::kLydian;
    if (valence > 0.65f) return ScaleType::kMajor;
    if (valence > 0.5f)  return ScaleType::kMixolydian;
    if (valence > 0.35f) return ScaleType::kDorian;
    if (valence > 0.2f)  return ScaleType::kMinor;
    return ScaleType::kPhrygian;
}

void VisionSynth::rebuildScaleTable() {
    const int32_t* pattern = kScalePatterns[static_cast<int32_t>(mCurrentScale)];
    mScaleNoteCount = 0;

    // Build scale from MIDI 36 (C2) to MIDI 96 (C7) — 5 octaves
    for (int32_t midi = 36; midi <= 96 && mScaleNoteCount < kMaxScaleNotes; ++midi) {
        int32_t degree = (midi - mCurrentKey + 120) % 12;
        if (pattern[degree]) {
            mScaleNotes[mScaleNoteCount++] = midi;
        }
    }
}

bool VisionSynth::isInScale(int32_t midiNote) const {
    int32_t degree = (midiNote - mCurrentKey + 120) % 12;
    return kScalePatterns[static_cast<int32_t>(mCurrentScale)][degree] != 0;
}

int32_t VisionSynth::nearestScaleNote(int32_t midiNote) const {
    if (mScaleNoteCount == 0) return midiNote;

    int32_t best = mScaleNotes[0];
    int32_t bestDist = 128;
    for (int32_t i = 0; i < mScaleNoteCount; ++i) {
        int32_t dist = abs(mScaleNotes[i] - midiNote);
        if (dist < bestDist) {
            bestDist = dist;
            best = mScaleNotes[i];
        }
    }
    return best;
}

float VisionSynth::midiToFreq(int32_t note) {
    return 440.0f * powf(2.0f, (static_cast<float>(note) - 69.0f) / 12.0f);
}

// =============================================================================
// Section 6: Chord Progression Engine
// =============================================================================

void VisionSynth::selectProgression() {
    // Choose progression bank based on valence
    int32_t bank;
    if (mValence > 0.6f) bank = 0;       // Bright (indices 0-3)
    else if (mValence > 0.3f) bank = 4;   // Neutral (indices 4-7)
    else bank = 8;                         // Dark (indices 8-11)

    // Pick a random progression from the bank
    mProgIndex = bank + randomInt(4);
    mProgStep = 0;

    LOGI("Progression selected: index=%d, valence=%.2f", mProgIndex, mValence);
}

VisionSynth::Chord VisionSynth::buildChord(int32_t rootDegree, int32_t quality,
                                            int32_t octave) {
    Chord chord{};

    // Find the root note in our scale
    // rootDegree = scale degree (0 = tonic, 1 = 2nd, etc.)
    // We need to map this to the actual scale notes
    int32_t scaleDegreesPerOctave = 0;
    const int32_t* pattern = kScalePatterns[static_cast<int32_t>(mCurrentScale)];
    for (int32_t i = 0; i < 12; ++i) scaleDegreesPerOctave += pattern[i];

    // Find the MIDI note for this scale degree in the target octave
    int32_t targetMidi = (octave + 1) * 12 + mCurrentKey; // Root of octave in key
    int32_t degreesFound = 0;
    int32_t rootMidi = targetMidi;
    for (int32_t offset = 0; offset < 24 && degreesFound <= rootDegree; ++offset) {
        int32_t midi = targetMidi + offset;
        int32_t deg = (midi - mCurrentKey + 120) % 12;
        if (pattern[deg]) {
            if (degreesFound == rootDegree) {
                rootMidi = midi;
                break;
            }
            degreesFound++;
        }
    }

    chord.rootDegree = rootDegree;

    if (quality == 0) {
        // Auto: build triad from scale (root, 3rd, 5th degree above root)
        chord.notes[0] = rootMidi;
        chord.notes[1] = nearestScaleNote(rootMidi + 3); // ~minor 3rd
        chord.notes[2] = nearestScaleNote(rootMidi + 7); // ~perfect 5th

        // Ensure 3rd is actually a 3rd (3 or 4 semitones above root)
        int32_t thirdInterval = chord.notes[1] - chord.notes[0];
        if (thirdInterval < 3) chord.notes[1] = nearestScaleNote(rootMidi + 4);

        chord.noteCount = 3;

        // Add 7th for richer voicings (subtle, low amplitude)
        int32_t seventh = nearestScaleNote(rootMidi + 10);
        if (seventh > chord.notes[2]) {
            chord.notes[3] = seventh;
            chord.noteCount = 4;
        }
    } else if (quality == 1) {
        // Force major triad
        chord.notes[0] = rootMidi;
        chord.notes[1] = rootMidi + 4;   // Major 3rd
        chord.notes[2] = rootMidi + 7;   // Perfect 5th
        chord.noteCount = 3;
    } else if (quality == 2) {
        // Force minor triad
        chord.notes[0] = rootMidi;
        chord.notes[1] = rootMidi + 3;   // Minor 3rd
        chord.notes[2] = rootMidi + 7;   // Perfect 5th
        chord.noteCount = 3;
    } else if (quality == 3) {
        // Dominant 7th
        chord.notes[0] = rootMidi;
        chord.notes[1] = rootMidi + 4;   // Major 3rd
        chord.notes[2] = rootMidi + 7;   // Perfect 5th
        chord.notes[3] = rootMidi + 10;  // Minor 7th
        chord.noteCount = 4;
    } else if (quality == 4) {
        // Sus4
        chord.notes[0] = rootMidi;
        chord.notes[1] = rootMidi + 5;   // Perfect 4th
        chord.notes[2] = rootMidi + 7;   // Perfect 5th
        chord.noteCount = 3;
    } else if (quality == 5) {
        // Major 7th
        chord.notes[0] = rootMidi;
        chord.notes[1] = rootMidi + 4;   // Major 3rd
        chord.notes[2] = rootMidi + 7;   // Perfect 5th
        chord.notes[3] = rootMidi + 11;  // Major 7th
        chord.noteCount = 4;
    }

    // Add bass note (root down an octave)
    if (chord.noteCount < kMaxChordNotes) {
        chord.notes[chord.noteCount] = rootMidi - 12;
        chord.noteCount++;
    }

    return chord;
}

void VisionSynth::advanceProgression() {
    const auto& prog = kProgressions[mProgIndex];

    // Move to next chord in progression
    mProgStep = (mProgStep + 1) % prog.length;

    bool useBorrowed = false;
    // Modal interchange: borrow a parallel minor chord if highly aroused but valence suddenly drops
    if (mArousal > 0.6f && mValence < 0.45f && mCurrentScale == ScaleType::kMajor) {
        if (randomFloat() < 0.25f) { // 25% chance to borrow when conditions met
            useBorrowed = true;
        }
    }

    // Build the next chord — use voice leading (find closest octave)
    int32_t rootDegree = prog.degrees[mProgStep];
    int32_t quality = prog.qualities[mProgStep];

    if (useBorrowed) {
        quality = 2; // Force to minor chord
    }

    // Determine octave that results in smallest voice movement
    // (voice leading principle: minimize total pitch distance)
    int32_t bestOctave = 3;
    int32_t bestDistance = 999;

    for (int32_t oct = 2; oct <= 4; ++oct) {
        Chord candidate = buildChord(rootDegree, quality, oct);
        int32_t totalDist = 0;
        int32_t compareCount = (candidate.noteCount < mCurrentChord.noteCount)
                                   ? candidate.noteCount
                                   : mCurrentChord.noteCount;
        for (int32_t n = 0; n < compareCount; ++n) {
            totalDist += abs(candidate.notes[n] - mCurrentChord.notes[n]);
        }
        if (totalDist < bestDistance) {
            bestDistance = totalDist;
            bestOctave = oct;
        }
    }

    mNextChord = buildChord(rootDegree, quality, bestOctave);
    mNextChord.isBorrowed = useBorrowed;
    mChordCrossfade = 0.0f;
}

// =============================================================================
// Section 7: Melody Engine
// =============================================================================

bool VisionSynth::isChordTone(int32_t midiNote) const {
    int32_t pc = midiNote % 12;
    for (int32_t i = 0; i < mCurrentChord.noteCount; ++i) {
        if ((mCurrentChord.notes[i] % 12) == pc) return true;
    }
    return false;
}

int32_t VisionSynth::chooseMelodyNote() {
    bool strongBeat = (mBeatCount % 2 == 0);

    // Track stability for Motif lock
    float currentValence = mTargetValence.load(std::memory_order_relaxed);
    float valenceDiff = currentValence - mValence;
    mValenceVelocity = mValenceVelocity * 0.9f + valenceDiff * 0.1f;
    bool isStable = fabsf(mValenceVelocity) < 0.05f;

    // Determine motif length (varies by arousal)
    mCurrentMotifLength = 4 + static_cast<int32_t>(mArousal * 60.0f); // 4 to 64 limit
    if (mCurrentMotifLength > 64) mCurrentMotifLength = 64;

    if (isStable && mMotifBuffer[mMotifIndex] != 0) {
        // Recall from memory
        int32_t target = mMotifBuffer[mMotifIndex];
        mMotifIndex = (mMotifIndex + 1) % mCurrentMotifLength;

        // Optionally invert motif if moody (low arousal)
        if (mArousal < 0.25f) {
            target = 60 + (60 - target); // Invert around C4
        }
        
        // Quantize back into scale if inverted out of bounds
        if (!isInScale(target)) target = nearestScaleNote(target);
        
        return target;
    }

    // Step size: mostly 1-2 scale degrees, occasionally 3-4
    int32_t maxStep = (randomFloat() < 0.7f) ? 2 : 4;

    // Choose direction based on contour and range constraints
    // If melody is too high (>78 = F#5), bias downward
    // If melody is too low (<55 = G3), bias upward
    if (mMelodyNote > 78) mMelodyContour = -1;
    else if (mMelodyNote < 55) mMelodyContour = 1;
    else if (randomFloat() < 0.15f) {
        // 15% chance of contour reversal for interest
        mMelodyContour = -mMelodyContour;
    }

    int32_t direction = mMelodyContour;
    if (direction == 0) direction = (randomFloat() < 0.5f) ? 1 : -1;

    // Find the target note by stepping through scale degrees
    int32_t step = 1 + randomInt(maxStep);
    int32_t target = mMelodyNote;
    int32_t stepsFound = 0;

    for (int32_t offset = 1; offset < 24 && stepsFound < step; ++offset) {
        int32_t candidate = mMelodyNote + (direction * offset);
        if (candidate < 48 || candidate > 84) break;  // Keep in singable range
        if (isInScale(candidate)) {
            target = candidate;
            stepsFound++;
        }
    }

    // On strong beats, snap to nearest chord tone
    if (strongBeat && !isChordTone(target)) {
        int32_t bestChordTone = target;
        int32_t bestDist = 128;
        for (int32_t i = 0; i < mCurrentChord.noteCount; ++i) {
            // Try octave equivalents near our target
            for (int32_t oct = -1; oct <= 1; ++oct) {
                int32_t ct = mCurrentChord.notes[i] + oct * 12;
                int32_t dist = abs(ct - target);
                if (dist < bestDist && ct >= 48 && ct <= 84) {
                    bestDist = dist;
                    bestChordTone = ct;
                }
            }
        }
        // Only snap if chord tone is within 3 semitones
        if (bestDist <= 3) target = bestChordTone;
    }

    // Save generated note into Motif Buffer
    mMotifBuffer[mMotifIndex] = target;
    mMotifIndex = (mMotifIndex + 1) % mCurrentMotifLength;

    return target;
}

// =============================================================================
// Section 8: Tempo Clock
// =============================================================================

float VisionSynth::getSecondsPerBeat() const {
    return 60.0f / mBpm;
}

float VisionSynth::getCurrentBpm() const {
    return mBpm;
}

int32_t VisionSynth::getCurrentKey() const {
    return mCurrentKey;
}

const char* VisionSynth::getCurrentChordName() const {
    static const char* names[] = {
        "I", "ii", "iii", "IV", "V", "vi", "vii"
    };
    int32_t deg = mCurrentChord.rootDegree;
    if (deg < 0 || deg > 6) return "?";
    return names[deg];
}

void VisionSynth::advanceClock(int32_t numFrames) {
    float dt = static_cast<float>(numFrames) * mInvSampleRate;

    // Smooth BPM toward target
    // Arousal maps to BPM: 0.0→60, 0.5→80, 1.0→120
    mTargetBpm = 60.0f + mArousal * 60.0f;
    mBpm += (mTargetBpm - mBpm) * 0.0001f;

    // Advance beat phase
    float beatsElapsed = dt / getSecondsPerBeat();
    mBeatPhase += beatsElapsed;

    // Check for beat crossing
    while (mBeatPhase >= 1.0f) {
        mBeatPhase -= 1.0f;
        mBeatCount++;

        // Bar boundary (every 4 beats)
        if (mBeatCount % 4 == 0) {
            mBarPhase = 0.0f;

            // Chord changes happen at bar boundaries
            mChordTimer += 1.0f;

            // Duration varies: ambient=long, rhythmic=shorter
            float chordBeats = mChordDuration;

            if (mChordTimer >= chordBeats) {
                mChordTimer = 0.0f;

                // Crossfade: old chord becomes "current", prepare next
                mCurrentChord = mNextChord;
                advanceProgression();
                updateVoiceTargets();

                // Periodically re-evaluate key and scale
                evaluateKeyChange();
            }
        }
    }

    // Update bar phase
    mBarPhase = static_cast<float>(mBeatCount % 4) / 4.0f + mBeatPhase / 4.0f;

    // Chord crossfade ramp
    if (mChordCrossfade < 1.0f) {
        mChordCrossfade += dt / kChordCrossfadeDuration;
        if (mChordCrossfade > 1.0f) mChordCrossfade = 1.0f;
    }

    // Chord duration adapts to interpretation
    int32_t interp = mInterpretation.load(std::memory_order_relaxed);
    switch (static_cast<MusicInterpretation>(interp)) {
        case MusicInterpretation::kAmbientDrift:
            mChordDuration = 4.0f + (1.0f - mArousal) * 4.0f;  // 4-8 bars
            break;
        case MusicInterpretation::kMelodicArpeggio:
            mChordDuration = 2.0f + (1.0f - mArousal) * 2.0f;  // 2-4 bars
            break;
        case MusicInterpretation::kRhythmicPulse:
            mChordDuration = 1.0f + (1.0f - mArousal) * 2.0f;  // 1-3 bars
            break;
    }
}

// =============================================================================
// Section 9: Oscillator & Effects
// =============================================================================

float VisionSynth::oscillator(float phase, float timbre) const {
    // Morph between sine, triangle, and soft saw based on timbre (0-1)
    // 0.0 = pure sine (warm, organic)
    // 0.5 = triangle (clear, bell-like)
    // 1.0 = soft saw (bright, rich harmonics)

    float twoPiPhase = 2.0f * static_cast<float>(M_PI) * phase;

    if (timbre < 0.5f) {
        // Sine → Triangle crossfade
        float t = timbre * 2.0f;  // 0–1
        float sine = sinf(twoPiPhase);
        float tri = (phase < 0.5f)
                         ? (4.0f * phase - 1.0f)
                         : (3.0f - 4.0f * phase);
        return sine * (1.0f - t) + tri * t;
    } else {
        // Triangle → Soft Saw crossfade
        float t = (timbre - 0.5f) * 2.0f;  // 0–1
        float tri = (phase < 0.5f)
                         ? (4.0f * phase - 1.0f)
                         : (3.0f - 4.0f * phase);
        // Polyblep-smoothed saw to avoid aliasing
        float saw = 2.0f * phase - 1.0f;
        // Simple anti-aliasing: smooth the discontinuity at phase=0/1
        float dt = mInvSampleRate * 100.0f;  // Approximate frequency-dependent
        if (phase < dt) {
            float t2 = phase / dt;
            saw -= (2.0f * t2 - t2 * t2 - 1.0f);
        } else if (phase > 1.0f - dt) {
            float t2 = (phase - 1.0f + dt) / dt;
            saw -= (2.0f * t2 - t2 * t2 - 1.0f);
        }
        return tri * (1.0f - t) + saw * t * 0.7f; // Saw slightly attenuated
    }
}

float VisionSynth::filterLowpass(Voice& voice, float input) const {
    // Basic 1-pole lowpass
    float c = (voice.filterCutoff < 0.001f) ? 0.001f : (voice.filterCutoff > 1.0f) ? 1.0f : voice.filterCutoff;
    voice.filterState1 += c * (input - voice.filterState1);
    return voice.filterState1;
}

float VisionSynth::filterResonant(Voice& voice, float input) const {
    // 4-pole Moog-style resonant lowpass approximation
    // Cutoff mapped via Arousal inside the synthesize functions
    
    // To prevent Euler Moog explosion, we must strictly limit f based on the math
    float f = voice.filterCutoff;
    if (f < 0.001f) f = 0.001f;
    if (f > 0.85f) f = 0.85f; // Hard stability limit for this topology
    
    float res = 0.5f + mTargetArousal.load(std::memory_order_relaxed) * 2.5f; // Resonance scales with arousal
    if (res > 3.8f) res = 3.8f; // 4.0 is self-oscillation, 3.8 is the safe upper bound

    float fb = res * (voice.filterState4 - voice.filterState1);
    
    // Add mild saturation to the feedback loop to prevent explosive squeals
    float drive = input - fb;
    // Fast soft-clip approximation: x / (1 + |x|)
    drive = drive / (1.0f + std::abs(drive));

    voice.filterState1 += f * (drive - voice.filterState1);
    voice.filterState2 += f * (voice.filterState1 - voice.filterState2);
    voice.filterState3 += f * (voice.filterState2 - voice.filterState3);
    voice.filterState4 += f * (voice.filterState3 - voice.filterState4);
    
    // NaN poisoning guard
    if (std::isnan(voice.filterState4) || std::isinf(voice.filterState4)) {
        voice.filterState1 = voice.filterState2 = voice.filterState3 = voice.filterState4 = 0.0f;
    }

    return voice.filterState4;
}

float VisionSynth::readDelay(float delaySec) const {
    float delaySamples = delaySec * static_cast<float>(mSampleRate);
    int32_t readPos = mDelayWritePos - static_cast<int32_t>(delaySamples);
    if (readPos < 0) readPos += kDelayMaxSamples;
    if (readPos >= kDelayMaxSamples) readPos -= kDelayMaxSamples;
    return mDelayBuffer[readPos];
}

void VisionSynth::writeDelay(float sample) {
    mDelayBuffer[mDelayWritePos] = sample;
    mDelayWritePos = (mDelayWritePos + 1) % kDelayMaxSamples;
}

float VisionSynth::saturate(float x) {
    return tanhf(x);
}

// =============================================================================
// Section 10: Voice Management
// =============================================================================

void VisionSynth::assignVoicesForInterpretation() {
    int32_t interp = mInterpretation.load(std::memory_order_relaxed);

    switch (static_cast<MusicInterpretation>(interp)) {
        case MusicInterpretation::kAmbientDrift:
            // All 8 voices as pad layers with wide stereo
            for (int32_t i = 0; i < kMaxVoices; ++i) {
                mVoices[i].role = 0;  // pad
                mVoices[i].portamento = 0.00002f;  // Very slow glide
                mVoices[i].attackRate = 0.00005f;   // ~0.4s attack
                mVoices[i].releaseRate = 0.00003f;  // ~0.7s release
                mVoices[i].vibratoRate = 0.5f + randomFloat() * 2.0f;
                mVoices[i].vibratoDepth = 2.0f + randomFloat() * 3.0f;
                mVoices[i].pan = -0.8f + 1.6f * (static_cast<float>(i) / 7.0f);
            }
            break;

        case MusicInterpretation::kMelodicArpeggio:
            // Voice 0: bass
            mVoices[0].role = 2;
            mVoices[0].portamento = 0.00008f;
            mVoices[0].attackRate = 0.001f;
            mVoices[0].releaseRate = 0.0005f;
            mVoices[0].pan = 0.0f;

            // Voices 1-2: pad layer
            for (int32_t i = 1; i <= 2; ++i) {
                mVoices[i].role = 0;
                mVoices[i].portamento = 0.00003f;
                mVoices[i].attackRate = 0.0001f;
                mVoices[i].releaseRate = 0.00005f;
                mVoices[i].pan = (i == 1) ? -0.5f : 0.5f;
            }

            // Voices 3-6: arpeggio voices
            for (int32_t i = 3; i <= 6; ++i) {
                mVoices[i].role = 1;  // arp
                mVoices[i].portamento = 0.005f;  // Fast for arpeggios
                mVoices[i].attackRate = 0.01f;   // Snappy
                mVoices[i].releaseRate = 0.002f;
                float spread = (static_cast<float>(i - 3) / 3.0f) * 2.0f - 1.0f;
                mVoices[i].pan = spread * 0.6f;
            }

            // Voice 7: melody lead
            mVoices[7].role = 3;
            mVoices[7].portamento = 0.003f;
            mVoices[7].attackRate = 0.005f;
            mVoices[7].releaseRate = 0.001f;
            mVoices[7].pan = 0.0f;
            break;

        case MusicInterpretation::kRhythmicPulse:
            // Voices 0-1: pulsing bass
            for (int32_t i = 0; i <= 1; ++i) {
                mVoices[i].role = 2;
                mVoices[i].portamento = 0.0005f;
                mVoices[i].attackRate = 0.01f;
                mVoices[i].releaseRate = 0.003f;
                mVoices[i].vibratoDepth = 0.0f;
                mVoices[i].pan = (i == 0) ? -0.1f : 0.1f;
            }

            // Voices 2-5: rhythmic gate chords
            for (int32_t i = 2; i <= 5; ++i) {
                mVoices[i].role = 1;  // rhythmic arp
                mVoices[i].portamento = 0.001f;
                mVoices[i].attackRate = 0.02f;    // Very snappy
                mVoices[i].releaseRate = 0.005f;
                float spread = (static_cast<float>(i - 2) / 3.0f) * 2.0f - 1.0f;
                mVoices[i].pan = spread * 0.7f;
            }

            // Voices 6-7: pad sustain
            for (int32_t i = 6; i <= 7; ++i) {
                mVoices[i].role = 0;
                mVoices[i].portamento = 0.00003f;
                mVoices[i].attackRate = 0.0001f;
                mVoices[i].releaseRate = 0.00005f;
                mVoices[i].pan = (i == 6) ? -0.6f : 0.6f;
            }
            break;
    }
}

void VisionSynth::updateVoiceTargets() {
    int32_t interp = mInterpretation.load(std::memory_order_relaxed);
    const Chord& chord = mCurrentChord;

    switch (static_cast<MusicInterpretation>(interp)) {
        case MusicInterpretation::kAmbientDrift: {
            // Spread chord tones across all 8 voices in different octaves
            for (int32_t i = 0; i < kMaxVoices; ++i) {
                int32_t chordIdx = i % chord.noteCount;
                int32_t octaveOffset = (i / chord.noteCount) * 12;
                // Alternate octave direction for richness
                if (i >= 4) octaveOffset = -12;

                int32_t note = chord.notes[chordIdx] + octaveOffset;
                // Clamp to reasonable range
                while (note < 36) note += 12;
                while (note > 84) note -= 12;

                mVoices[i].targetFreq = midiToFreq(note);
                mVoices[i].targetAmplitude = 0.12f + mDensity * 0.08f;

                // Higher voices are quieter for natural balance
                if (i >= 4) mVoices[i].targetAmplitude *= 0.6f;
            }
            break;
        }

        case MusicInterpretation::kMelodicArpeggio: {
            // Voice 0: bass = chord root, down an octave
            mVoices[0].targetFreq = midiToFreq(chord.notes[0] - 12);
            mVoices[0].targetAmplitude = 0.15f;

            // Voices 1-2: pad = chord root + 5th
            if (chord.noteCount >= 3) {
                mVoices[1].targetFreq = midiToFreq(chord.notes[0]);
                mVoices[2].targetFreq = midiToFreq(chord.notes[2]);
            }
            mVoices[1].targetAmplitude = 0.08f;
            mVoices[2].targetAmplitude = 0.08f;

            // Voices 3-6: arp notes = chord tones, will be triggered rhythmically
            for (int32_t i = 3; i <= 6; ++i) {
                int32_t chordIdx = (i - 3) % chord.noteCount;
                int32_t note = chord.notes[chordIdx];
                // Spread across octaves for interest
                if (i == 4 || i == 6) note += 12;
                while (note > 84) note -= 12;
                mVoices[i].targetFreq = midiToFreq(note);
                mVoices[i].targetAmplitude = 0.10f + mDensity * 0.06f;
            }

            // Voice 7: melody — will be updated by melody engine
            break;
        }

        case MusicInterpretation::kRhythmicPulse: {
            // Voices 0-1: bass = root + octave
            mVoices[0].targetFreq = midiToFreq(chord.notes[0] - 12);
            mVoices[1].targetFreq = midiToFreq(chord.notes[0] - 24);
            mVoices[0].targetAmplitude = 0.18f;
            mVoices[1].targetAmplitude = 0.12f;

            // Voices 2-5: rhythmic chord stabs
            for (int32_t i = 2; i <= 5; ++i) {
                int32_t chordIdx = (i - 2) % chord.noteCount;
                mVoices[i].targetFreq = midiToFreq(chord.notes[chordIdx]);
                mVoices[i].targetAmplitude = 0.10f + mDensity * 0.05f;
            }

            // Voices 6-7: sustain pad
            if (chord.noteCount >= 3) {
                mVoices[6].targetFreq = midiToFreq(chord.notes[0] + 12);
                mVoices[7].targetFreq = midiToFreq(chord.notes[2] + 12);
            }
            mVoices[6].targetAmplitude = 0.06f;
            mVoices[7].targetAmplitude = 0.06f;
            break;
        }
    }
}

// =============================================================================
// Section 11: Main Synthesis Entry Point
// =============================================================================

void VisionSynth::synthesize(float* output, int32_t numFrames) {
    auto t1 = std::chrono::steady_clock::now();

    if (!mInitialized) {
        std::memset(output, 0, numFrames * sizeof(float));
        return;
    }

    // Smooth parameters toward targets (per-block, not per-sample for efficiency)
    for (int32_t s = 0; s < 8; ++s) smoothParameters();

    // Advance musical clock
    advanceClock(numFrames);

    // Phase 19: Phrase Transistion State Flush
    if (mPhraseNeedsFlush.exchange(false, std::memory_order_acquire)) {
        LOGI("VisionSynth: Flushing generative state for Phase Transition");
        // Force key modulation to a relative or contrasting key
        mCurrentKey = (mCurrentKey + 5 + randomInt(4)) % 12; 
        
        // Pick new random progression
        mProgIndex = randomInt(kNumProgressions);
        mProgStep = 0;
        
        // Reset rhythmic clocks to downbeat
        mBeatPhase = 0.0f;
        mBarPhase = 0.0f;
        mBeatCount = 0;
        
        // Optionally mutate interpretation
        int32_t currentInterp = mInterpretation.load(std::memory_order_relaxed);
        int32_t nextInterp = (currentInterp + 1 + randomInt(2)) % 3;
        mInterpretation.store(nextInterp, std::memory_order_relaxed);
    }

    // Check if interpretation changed
    static int32_t lastInterp = -1;
    int32_t curInterp = mInterpretation.load(std::memory_order_relaxed);
    if (curInterp != lastInterp) {
        lastInterp = curInterp;
        assignVoicesForInterpretation();
        updateVoiceTargets();
    }

    // Route to interpretation-specific synthesis
    switch (static_cast<MusicInterpretation>(curInterp)) {
        case MusicInterpretation::kAmbientDrift:
            synthesizeAmbient(output, numFrames);
            break;
        case MusicInterpretation::kMelodicArpeggio:
            synthesizeMelodic(output, numFrames);
            break;
        case MusicInterpretation::kRhythmicPulse:
            synthesizeRhythmic(output, numFrames);
            break;
    }

    // Apply delay effect
    float delayTime = 0.3f + (1.0f - mArousal) * 0.4f;  // 0.3-0.7s
    float delayFb = 0.2f + (1.0f - mArousal) * 0.3f;     // 0.2-0.5
    float delayWet = 0.15f + (1.0f - mDensity) * 0.15f;   // 0.15-0.30

    for (int32_t i = 0; i < numFrames; ++i) {
        float dry = output[i];
        float delayed = readDelay(delayTime);
        float wet = delayed * delayWet;
        writeDelay(dry + delayed * delayFb);
        output[i] = saturate((dry + wet) * mLevel);
    }

    auto t2 = std::chrono::steady_clock::now();
    uint64_t dur = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    mLastSynthDurationUs = dur;

    // Log benchmark occasionally (~once per second)
    static int32_t logCounter = 0;
    if (++logCounter >= (mSampleRate / numFrames)) {
        logCounter = 0;
        int32_t quality = getSynthQuality();
        LOGI("VisionSynth RT Benchmark: %llu us processing %d frames (Quality: %d)", dur, numFrames, quality);
    }
}

// =============================================================================
// Section 12: Ambient Drift Interpretation
// =============================================================================
// All 8 voices as slowly evolving pads. Rich, warm, floating.
// Voices glide between chord tones over seconds. Deep vibrato.
// Timbre controls waveform brightness. Density adds higher partials.

void VisionSynth::synthesizeAmbient(float* output, int32_t numFrames) {
    for (int32_t i = 0; i < numFrames; ++i) {
        float sample = 0.0f;

        for (int32_t v = 0; v < kMaxVoices; ++v) {
            auto& voice = mVoices[v];

            // Portamento toward target
            voice.currentFreq += (voice.targetFreq - voice.currentFreq) * voice.portamento;
            voice.amplitude += (voice.targetAmplitude - voice.amplitude) * 0.00005f;

            // Vibrato
            voice.vibratoPhase += voice.vibratoRate * mInvSampleRate;
            if (voice.vibratoPhase >= 1.0f) voice.vibratoPhase -= 1.0f;
            float vibrato = sinf(2.0f * static_cast<float>(M_PI) * voice.vibratoPhase);
            float vibratoRatio = powf(2.0f, vibrato * voice.vibratoDepth / 1200.0f);

            float freq = voice.currentFreq * vibratoRatio;

            // Phase accumulation
            voice.phase += freq * mInvSampleRate;
            if (voice.phase >= 1.0f) voice.phase -= 1.0f;

            // Waveform with timbre morphing
            float wave = oscillator(voice.phase, mTimbre * 0.6f);

            // Gentle lowpass filter — timbre also controls brightness
            float cutoff = 0.05f + mTimbre * 0.4f;
            voice.filterCutoff = cutoff;
            wave = getSynthQuality() == 1 ? filterResonant(voice, wave) : filterLowpass(voice, wave);

            sample += wave * voice.amplitude;
        }

        output[i] = sample;
    }
}

// =============================================================================
// Section 13: Melodic Arpeggio Interpretation
// =============================================================================
// Bass (voice 0): sustained root
// Pad (voices 1-2): gentle chord sustain
// Arpeggio (voices 3-6): triggered in sequence on subdivisions
// Melody (voice 7): contour-based lead line on strong beats

void VisionSynth::synthesizeMelodic(float* output, int32_t numFrames) {
    float beatsPerSample = (mBpm / 60.0f) * mInvSampleRate;

    for (int32_t i = 0; i < numFrames; ++i) {
        float sample = 0.0f;

        // Advance per-sample melody phase
        mMelodyPhase += beatsPerSample;

        // Determine note duration from density (more dense = faster notes)
        float notesPerBeat = 2.0f + mDensity * 6.0f;  // 2-8 notes per beat
        float noteDur = 1.0f / notesPerBeat;

        // --- Arpeggio triggering ---
        // Check if a new arp note should fire
        static float arpPhase = 0.0f;
        arpPhase += beatsPerSample;
        if (arpPhase >= noteDur) {
            arpPhase -= noteDur;

            // Cycle through arp voices 3-6
            static int32_t arpIndex = 0;
            int32_t voiceIdx = 3 + (arpIndex % 4);
            arpIndex++;

            // Pick a chord tone for this arp hit
            int32_t chordIdx = arpIndex % mCurrentChord.noteCount;
            int32_t note = mCurrentChord.notes[chordIdx];
            // Spread upward through chord for ascending arp
            int32_t octShift = (arpIndex / mCurrentChord.noteCount) % 3;
            note += octShift * 12;
            while (note > 84) note -= 12;

            mVoices[voiceIdx].targetFreq = midiToFreq(note);
            mVoices[voiceIdx].envPhase = 0.0f;  // Re-trigger envelope
            mVoices[voiceIdx].envLevel = 0.0f;
        }

        // --- Melody triggering (every half beat) ---
        static float melodyPhaseAccum = 0.0f;
        melodyPhaseAccum += beatsPerSample;
        if (melodyPhaseAccum >= 0.5f) {
            melodyPhaseAccum -= 0.5f;

            int32_t note = chooseMelodyNote();
            mMelodyNote = note;
            mVoices[7].targetFreq = midiToFreq(note);
            mVoices[7].envPhase = 0.0f;
            mVoices[7].envLevel = 0.0f;
        }

        // --- Render all voices ---
        for (int32_t v = 0; v < kMaxVoices; ++v) {
            auto& voice = mVoices[v];

            // Portamento
            voice.currentFreq += (voice.targetFreq - voice.currentFreq) * voice.portamento;
            voice.amplitude += (voice.targetAmplitude - voice.amplitude) * 0.001f;

            // Envelope (attack-sustain-release)
            if (voice.role == 1 || voice.role == 3) {
                // Arp and melody voices: AR envelope
                if (voice.envPhase < 1.0f) {
                    voice.envLevel += voice.attackRate;
                    if (voice.envLevel >= 1.0f) {
                        voice.envLevel = 1.0f;
                        voice.envPhase = 1.0f;
                    }
                } else {
                    voice.envLevel -= voice.releaseRate;
                    if (voice.envLevel < 0.0f) voice.envLevel = 0.0f;
                }
            } else {
                // Pad and bass: always sustained
                voice.envLevel = 1.0f;
            }

            // Vibrato (subtle on arp, more on pads)
            voice.vibratoPhase += voice.vibratoRate * mInvSampleRate;
            if (voice.vibratoPhase >= 1.0f) voice.vibratoPhase -= 1.0f;
            float vibrato = sinf(2.0f * static_cast<float>(M_PI) * voice.vibratoPhase);
            float vibratoRatio = powf(2.0f, vibrato * voice.vibratoDepth / 1200.0f);

            float freq = voice.currentFreq * vibratoRatio;
            voice.phase += freq * mInvSampleRate;
            if (voice.phase >= 1.0f) voice.phase -= 1.0f;

            // Timbre: bass=warm sine, arp=triangle, melody=bright
            float voiceTimbre = mTimbre;
            if (voice.role == 2) voiceTimbre *= 0.2f;       // Bass = warm
            else if (voice.role == 1) voiceTimbre *= 0.5f;   // Arp = clear
            else if (voice.role == 3) voiceTimbre *= 0.7f;   // Melody = present

            float wave = oscillator(voice.phase, voiceTimbre);

            // Filter: bass heavily filtered, arp medium, melody open
            float cutoff = 0.1f + mTimbre * 0.5f;
            if (voice.role == 2) cutoff *= 0.3f;
            voice.filterCutoff = cutoff;
            wave = getSynthQuality() == 1 ? filterResonant(voice, wave) : filterLowpass(voice, wave);

            sample += wave * voice.amplitude * voice.envLevel;
        }

        output[i] = sample;
    }
}

// =============================================================================
// Section 14: Rhythmic Pulse Interpretation
// =============================================================================
// Bass (voices 0-1): pulsing on beat, sidechained feel
// Rhythmic gates (voices 2-5): chord stabs on subdivisions
// Pad (voices 6-7): sustained atmosphere behind the rhythm

void VisionSynth::synthesizeRhythmic(float* output, int32_t numFrames) {
    float beatsPerSample = (mBpm / 60.0f) * mInvSampleRate;
    int32_t synthQuality = getSynthQuality();

    // Euclidean mappings
    // Bass: k1 pulses in 8 steps
    int32_t bassK = 2 + static_cast<int32_t>(mDensity * 4.0f); // 2 to 6
    if (bassK > 8) bassK = 8;
    int32_t bassN = 8;
    
    // Gates: k2 pulses in 16 steps
    int32_t gateK = 3 + static_cast<int32_t>(mArousal * 10.0f); // 3 to 13
    if (gateK > 16) gateK = 16;
    int32_t gateN = 16;

    for (int32_t i = 0; i < numFrames; ++i) {
        float sample = 0.0f;
        
        // 16th note subdivision duration
        float subdivDur = 0.25f; // relative to beat
        static float stepPhase = 0.0f;
        stepPhase += beatsPerSample;

        // --- Euclidean Rhythm Triggers ---
        if (stepPhase >= subdivDur) {
            stepPhase -= subdivDur;
            
            // Advance Euclidean steps
            mRhythmicStep[0] = (mRhythmicStep[0] + 1) % bassN;
            mRhythmicStep[1] = (mRhythmicStep[1] + 1) % gateN;

            // Trigger Bass E(bassK, bassN)
            if ((mRhythmicStep[0] * bassK) % bassN < bassK) {
                mVoices[0].envPhase = 0.0f;
                mVoices[0].envLevel = 0.0f;
                mVoices[1].envPhase = 0.0f;
                mVoices[1].envLevel = 0.0f;
            }

            // Trigger Gate E(gateK, gateN)
            if ((mRhythmicStep[1] * gateK) % gateN < gateK) {
                int32_t voiceIdx = 2 + randomInt(4);
                mVoices[voiceIdx].envPhase = 0.0f;
                mVoices[voiceIdx].envLevel = 0.0f;
                float velocity = 0.7f + randomFloat() * 0.3f;
                mVoices[voiceIdx].targetAmplitude = (0.08f + mDensity * 0.06f) * velocity;
            }
        }

        // --- Sidechain pump effect on pad and rhythm voices ---
        float sidechain = 1.0f;
        if (mRhythmicStep[0] == 0 && stepPhase < subdivDur) {
            // Quick duck on the downbeat step
            sidechain = 0.2f + 0.8f * (stepPhase / subdivDur);
        }

        // --- Render all voices ---
        for (int32_t v = 0; v < kMaxVoices; ++v) {
            auto& voice = mVoices[v];

            // Portamento
            voice.currentFreq += (voice.targetFreq - voice.currentFreq) * voice.portamento;
            voice.amplitude += (voice.targetAmplitude - voice.amplitude) * 0.001f;

            // Envelope
            if (voice.role == 2 || voice.role == 1) {
                // Bass and rhythmic: AR envelope
                if (voice.envPhase < 1.0f) {
                    voice.envLevel += voice.attackRate;
                    if (voice.envLevel >= 1.0f) {
                        voice.envLevel = 1.0f;
                        voice.envPhase = 1.0f;
                    }
                } else {
                    voice.envLevel -= voice.releaseRate;
                    if (voice.envLevel < 0.0f) voice.envLevel = 0.0f;
                }
            } else {
                // Pad: always on, but modulated by sidechain
                voice.envLevel = sidechain;
            }

            // Vibrato (minimal for rhythmic)
            voice.vibratoPhase += voice.vibratoRate * mInvSampleRate;
            if (voice.vibratoPhase >= 1.0f) voice.vibratoPhase -= 1.0f;
            float vibrato = sinf(2.0f * static_cast<float>(M_PI) * voice.vibratoPhase);
            float vibratoRatio = powf(2.0f, vibrato * voice.vibratoDepth / 1200.0f);

            float freq = voice.currentFreq * vibratoRatio;
            voice.phase += freq * mInvSampleRate;
            if (voice.phase >= 1.0f) voice.phase -= 1.0f;

            // Timbre: bass = sub sine, rhythm = punchy, pad = warm
            float voiceTimbre = mTimbre;
            if (voice.role == 2) voiceTimbre = 0.0f;          // Pure sine bass
            else if (voice.role == 1) voiceTimbre *= 0.6f;

            float wave = oscillator(voice.phase, voiceTimbre);

            // Filter
            float cutoff = 0.08f + mTimbre * 0.4f;
            if (voice.role == 2) cutoff = 0.05f + mArousal * 0.1f; // Bass = very filtered
            voice.filterCutoff = cutoff;
            wave = synthQuality == 1 ? filterResonant(voice, wave) : filterLowpass(voice, wave);

            sample += wave * voice.amplitude * voice.envLevel;
        }

        output[i] = sample;
    }
}

} // namespace less
