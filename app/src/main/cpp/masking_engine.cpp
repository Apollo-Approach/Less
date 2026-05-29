// =============================================================================
// masking_engine.cpp — Psychoacoustic Mask Generator Implementation
// =============================================================================
// Phase 11: Tonal noise detection + generative acoustic masking.
//
// The mask operates in the frequency domain:
//   1. FFT the incoming mic audio
//   2. Track which frequency bins persist above the noise floor (tonal)
//   3. Generate noise shaped to the selected texture profile
//   4. Band-limit the noise to only cover tonal regions
//   5. IFFT back to time domain
//   6. Apply smooth gain envelope + safety limiter
//
// All computation is deterministic with pre-allocated buffers.
// =============================================================================

#include "masking_engine.h"

#include <algorithm>
#include <android/log.h>
#include <cstring>

#define LOG_TAG "LESS_Mask"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace less {

// =============================================================================
// FFT (reused from noise_suppressor — identical radix-2 implementation)
// =============================================================================
namespace {

void fft_radix2(float *re, float *im, int n, bool inverse) {
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
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
        float tRe = re[i + j + len / 2] * curRe - im[i + j + len / 2] * curIm;
        float tIm = re[i + j + len / 2] * curIm + im[i + j + len / 2] * curRe;
        re[i + j + len / 2] = re[i + j] - tRe;
        im[i + j + len / 2] = im[i + j] - tIm;
        re[i + j] += tRe;
        im[i + j] += tIm;
        float newCurRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newCurRe;
      }
    }
  }
  if (inverse) {
    float invN = 1.0f / static_cast<float>(n);
    for (int i = 0; i < n; ++i) {
      re[i] *= invN;
      im[i] *= invN;
    }
  }
}

} // anonymous namespace

// =============================================================================
// Texture Name
// =============================================================================

const char *MaskingEngine::textureName(MaskTexture texture) {
  switch (texture) {
  case MaskTexture::kBrownNoise:
    return "Brown Noise";
  case MaskTexture::kPinkNoise:
    return "Pink Noise";
  case MaskTexture::kWhiteNoise:
    return "White Noise";
  case MaskTexture::kNature:
    return "Nature";
  case MaskTexture::kHarmonic:
    return "Harmonic";
  default:
    return "Unknown";
  }
}

float MaskingEngine::getDetectedFundamental() const {
  return mDetectedFundamental;
}

// =============================================================================
// Construction / Destruction
// =============================================================================

MaskingEngine::MaskingEngine() = default;
MaskingEngine::~MaskingEngine() = default;

// =============================================================================
// Initialization — ALL allocation happens here
// =============================================================================

bool MaskingEngine::initialize(int32_t sampleRate) {
  mSampleRate = sampleRate;

  // --- Tonal detector buffers ---
  mSpectralAverage.resize(kNumBins, 0.0f);
  mSpectralPeak.resize(kNumBins, 0.0f);
  mPersistenceCount.resize(kNumBins, 0);
  mTonalMask.resize(kNumBins, false);

  // Exponential decay: τ ≈ 2 seconds
  // α = 1 - exp(-hopSize / (τ * sampleRate))
  float tau = 2.0f;
  mDecayAlpha = 1.0f - expf(-static_cast<float>(kHopSize) /
                            (tau * static_cast<float>(sampleRate)));

  // Initial tonal threshold (adaptive, updated during analysis)
  mTonalThreshold = 0.001f;

  // FFT scratch
  mFftReal.resize(kFftSize, 0.0f);
  mFftImag.resize(kFftSize, 0.0f);
  mWindowCoeffs.resize(kFftSize, 0.0f);
  computeWindow();

  // Input ring buffer
  mInputRing.resize(kFftSize * 2, 0.0f);
  mRingWritePos = 0;
  mFramesAccumulated = 0;

  // --- Synthesizer buffers ---
  mBandGain.resize(kNumBins, 0.0f);
  mSynthBuffer.resize(kFftSize, 0.0f);
  mSynthFftReal.resize(kFftSize, 0.0f);
  mSynthFftImag.resize(kFftSize, 0.0f);

  // --- Gain envelope ---
  // Attack: 200ms → per-sample coefficient
  float attackMs = 200.0f;
  float attackSamples = attackMs * 0.001f * static_cast<float>(sampleRate);
  mAttackCoeff = 1.0f - expf(-1.0f / attackSamples);

  // Release: 1500ms → per-sample coefficient
  float releaseMs = 1500.0f;
  float releaseSamples = releaseMs * 0.001f * static_cast<float>(sampleRate);
  mReleaseCoeff = 1.0f - expf(-1.0f / releaseSamples);

  mCurrentGain = 0.0f;
  mTargetGain = 0.0f;

  // --- Safety limiter ---
  // -12 dBFS → linear
  mSafetyLimitLinear = powf(10.0f, kSafetyLimitDb / 20.0f); // ≈ 0.25

  // --- RNG seed from sample rate for deterministic-ish startup ---
  mRngState = static_cast<uint32_t>(sampleRate) ^ 0xDEADBEEF;
  mBrownState = 0.0f;
  mNatureLfoPhase = 0.0f;

  mInitialized = true;
  LOGI("MaskingEngine initialized: sampleRate=%d, safetyLimit=%.3f, "
       "attackCoeff=%.6f, releaseCoeff=%.6f",
       sampleRate, mSafetyLimitLinear, mAttackCoeff, mReleaseCoeff);

  return true;
}

void MaskingEngine::computeWindow() {
  for (int32_t i = 0; i < kFftSize; ++i) {
    mWindowCoeffs[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (kFftSize - 1)));
  }
}

// =============================================================================
// Runtime Configuration (atomic, thread-safe)
// =============================================================================

void MaskingEngine::setMaskLevel(float level) {
  mMaskLevel.store(std::clamp(level, 0.0f, 1.0f), std::memory_order_relaxed);
}

float MaskingEngine::getMaskLevel() const {
  return mMaskLevel.load(std::memory_order_relaxed);
}

void MaskingEngine::setTexture(MaskTexture texture) {
  mTexture.store(static_cast<int32_t>(texture), std::memory_order_relaxed);
}

MaskTexture MaskingEngine::getTexture() const {
  return static_cast<MaskTexture>(mTexture.load(std::memory_order_relaxed));
}

bool MaskingEngine::isActive() const {
  return mIsActive.load(std::memory_order_relaxed);
}

int32_t MaskingEngine::getActiveTonalBandCount() const {
  return mActiveTonalBands.load(std::memory_order_relaxed);
}

// =============================================================================
// Hot Path — analyzeAndGenerate()
// =============================================================================
// ALLOCATION-FREE. LOCK-FREE. Called from the real-time audio thread.

void MaskingEngine::analyzeAndGenerate(const float *input, float *maskOutput,
                                       int32_t numFrames) {
  if (!mInitialized) {
    std::memset(maskOutput, 0, numFrames * sizeof(float));
    return;
  }

  float level = mMaskLevel.load(std::memory_order_relaxed);
  if (level < 0.01f) {
    std::memset(maskOutput, 0, numFrames * sizeof(float));
    mIsActive.store(false, std::memory_order_relaxed);
    return;
  }

  // --- Accumulate input into ring buffer ---
  int32_t ringSize = static_cast<int32_t>(mInputRing.size());
  for (int32_t i = 0; i < numFrames; ++i) {
    mInputRing[mRingWritePos] = input[i];
    mRingWritePos = (mRingWritePos + 1) % ringSize;
    mFramesAccumulated++;
  }

  // --- Analyze complete frames ---
  while (mFramesAccumulated >= kFftSize) {
    // Window the frame
    int32_t readPos = (mRingWritePos - kFftSize + ringSize) % ringSize;
    for (int32_t i = 0; i < kFftSize; ++i) {
      int32_t idx = (readPos + i) % ringSize;
      mFftReal[i] = mInputRing[idx] * mWindowCoeffs[i];
      mFftImag[i] = 0.0f;
    }

    // Forward FFT
    fft_radix2(mFftReal.data(), mFftImag.data(), kFftSize, false);

    // Analyze spectrum for tonal components
    analyzeSpectrum(mFftReal.data());

    // Update tonal classification
    updateTonalMask();

    mFramesAccumulated -= kHopSize;
  }

  // --- Generate mask audio ---
  auto texture = static_cast<MaskTexture>(
      mTexture.load(std::memory_order_relaxed));
  synthesizeMask(maskOutput, numFrames);

  // --- Apply gain envelope ---
  if (texture == MaskTexture::kHarmonic) {
    // Harmonic drone always plays — no tonal-dependent gating.
    // The drone IS the feature, not a reactive mask. Using fixed
    // target gain prevents gain pumping when tonal count fluctuates.
    mTargetGain = level;
  } else {
    // Other textures: only play when tonal noise is detected
    int32_t tonalCount = mActiveTonalBands.load(std::memory_order_relaxed);
    mTargetGain = (tonalCount > 0) ? level : 0.0f;
  }

  for (int32_t i = 0; i < numFrames; ++i) {
    // Smooth envelope tracking
    if (mCurrentGain < mTargetGain) {
      mCurrentGain += mAttackCoeff * (mTargetGain - mCurrentGain);
    } else {
      mCurrentGain += mReleaseCoeff * (mTargetGain - mCurrentGain);
    }
    maskOutput[i] *= mCurrentGain;
  }

  // --- Safety limiter ---
  applySafetyLimiter(maskOutput, numFrames);

  // Update active state
  mIsActive.store(mCurrentGain > 0.001f, std::memory_order_relaxed);
}

// =============================================================================
// Spectral Analysis — detect tonal (persistent) noise components
// =============================================================================

void MaskingEngine::analyzeSpectrum(const float *fftData) {
  // Compute magnitude spectrum and update rolling average
  float totalEnergy = 0.0f;

  for (int32_t b = 0; b < kNumBins; ++b) {
    float mag = sqrtf(mFftReal[b] * mFftReal[b] + mFftImag[b] * mFftImag[b]);

    // Exponential moving average
    mSpectralAverage[b] =
        mDecayAlpha * mag + (1.0f - mDecayAlpha) * mSpectralAverage[b];

    // Peak tracker (slower decay for persistence detection)
    if (mag > mSpectralPeak[b]) {
      mSpectralPeak[b] = mag;
    } else {
      mSpectralPeak[b] *= 0.995f; // Slow peak decay
    }

    totalEnergy += mag;
  }

  // Adaptive threshold: mean energy * scale factor
  // Tonal components are bins significantly above the spectral mean
  float meanEnergy = totalEnergy / static_cast<float>(kNumBins);
  mTonalThreshold = meanEnergy * 3.0f; // 3× mean = "significantly above"
}

// =============================================================================
// Tonal Mask Update — classify persistent frequency bins
// =============================================================================

void MaskingEngine::updateTonalMask() {
  int32_t tonalCount = 0;

  for (int32_t b = 0; b < kNumBins; ++b) {
    // A bin is "tonal" if:
    //   1. Its average magnitude exceeds the adaptive threshold
    //   2. Its peak is close to its average (steady, not transient)
    //   3. It has persisted for kPersistenceFrames consecutive analyses

    bool isAboveThreshold = mSpectralAverage[b] > mTonalThreshold;
    bool isSteady = (mSpectralPeak[b] > 0.001f) &&
                    (mSpectralAverage[b] / mSpectralPeak[b] > 0.5f);

    if (isAboveThreshold && isSteady) {
      mPersistenceCount[b] =
          std::min(mPersistenceCount[b] + 1, kPersistenceFrames + 10);
    } else {
      mPersistenceCount[b] = std::max(mPersistenceCount[b] - 1, 0);
    }

    bool wasTonal = mTonalMask[b];
    mTonalMask[b] = mPersistenceCount[b] >= kPersistenceFrames;

    if (mTonalMask[b]) {
      // Set band gain proportional to the tonal energy
      // (mask louder noise with proportionally louder masking)
      mBandGain[b] = std::min(mSpectralAverage[b] * 2.0f, 1.0f);
      tonalCount++;
    } else {
      // Smooth fade-out for bands losing tonal status
      mBandGain[b] *= 0.95f;
    }
  }

  mActiveTonalBands.store(tonalCount, std::memory_order_relaxed);

  // Update detected fundamental for harmonic synthesis
  updateFundamental();
}

// =============================================================================
// Noise Synthesizer — generate the masking texture
// =============================================================================

void MaskingEngine::synthesizeMask(float *output, int32_t numFrames) {
  auto texture =
      static_cast<MaskTexture>(mTexture.load(std::memory_order_relaxed));

  // Harmonic texture bypasses the noise→FFT→bandlimit pipeline entirely
  if (texture == MaskTexture::kHarmonic) {
    synthesizeHarmonic(output, numFrames);
    return;
  }

  // Generate raw noise in the time domain
  for (int32_t i = 0; i < kFftSize; ++i) {
    float sample;
    switch (texture) {
    case MaskTexture::kBrownNoise:
      sample = generateBrownNoise();
      break;
    case MaskTexture::kPinkNoise:
      sample = generatePinkNoise();
      break;
    case MaskTexture::kWhiteNoise:
      sample = generateWhiteNoise();
      break;
    case MaskTexture::kNature:
      sample = applyNatureModulation(generateBrownNoise());
      break;
    default:
      sample = generateBrownNoise();
      break;
    }
    mSynthFftReal[i] = sample;
    mSynthFftImag[i] = 0.0f;
  }

  // Forward FFT the noise
  fft_radix2(mSynthFftReal.data(), mSynthFftImag.data(), kFftSize, false);

  // Band-limit to tonal regions only
  applyBandLimiting(mSynthFftReal.data(), mSynthFftImag.data());

  // Inverse FFT back to time domain
  fft_radix2(mSynthFftReal.data(), mSynthFftImag.data(), kFftSize, true);

  // Copy to output (may need to truncate or pad)
  int32_t copySize = std::min(numFrames, kFftSize);
  std::memcpy(output, mSynthFftReal.data(), copySize * sizeof(float));

  if (copySize < numFrames) {
    std::memset(output + copySize, 0, (numFrames - copySize) * sizeof(float));
  }
}

// =============================================================================
// Noise Generators — deterministic, no syscalls
// =============================================================================

float MaskingEngine::generateWhiteNoise() {
  // xorshift32 PRNG — fast, seedable, no system calls
  mRngState ^= mRngState << 13;
  mRngState ^= mRngState >> 17;
  mRngState ^= mRngState << 5;

  // Convert to float in [-1.0, 1.0]
  return static_cast<float>(static_cast<int32_t>(mRngState)) *
         (1.0f / 2147483648.0f);
}

float MaskingEngine::generateBrownNoise() {
  // Integrate white noise with leaky integrator
  // Brown noise has -6dB/octave roll-off (warm, deep sound)
  float white = generateWhiteNoise();
  mBrownState = 0.99f * mBrownState + 0.01f * white;

  // Normalize (brown noise has lower amplitude than white)
  return mBrownState * 8.0f;
}

float MaskingEngine::generatePinkNoise() {
  // Paul Kellet's economy pink noise approximation
  // -3dB/octave roll-off — balanced between brown and white
  static float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f, b4 = 0.0f, b5 = 0.0f,
               b6 = 0.0f;

  float white = generateWhiteNoise();
  b0 = 0.99886f * b0 + white * 0.0555179f;
  b1 = 0.99332f * b1 + white * 0.0750759f;
  b2 = 0.96900f * b2 + white * 0.1538520f;
  b3 = 0.86650f * b3 + white * 0.3104856f;
  b4 = 0.55000f * b4 + white * 0.5329522f;
  b5 = -0.7616f * b5 - white * 0.0168980f;

  float pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362f;
  b6 = white * 0.115926f;

  return pink * 0.11f; // Normalize amplitude
}

float MaskingEngine::applyNatureModulation(float sample) {
  // Slow amplitude modulation to create rain/ocean texture
  // LFO at ~0.3Hz with slight randomization
  mNatureLfoPhase += kNatureLfoFreq / static_cast<float>(mSampleRate);
  if (mNatureLfoPhase >= 1.0f)
    mNatureLfoPhase -= 1.0f;

  // Sine LFO with slight noise for organic feel
  float lfo = 0.5f + 0.5f * sinf(2.0f * M_PI * mNatureLfoPhase);
  float noise = generateWhiteNoise() * 0.1f; // 10% randomization
  float mod = std::clamp(lfo + noise, 0.0f, 1.0f);

  // Modulate between 30% and 100% amplitude
  return sample * (0.3f + 0.7f * mod);
}

// =============================================================================
// Band Limiting — apply tonal mask to synthesized noise spectrum
// =============================================================================

void MaskingEngine::applyBandLimiting(float *fftReal, float *fftImag) {
  // Zero out bins that don't correspond to detected tonal regions
  for (int32_t b = 0; b < kNumBins; ++b) {
    float gain = mBandGain[b];
    fftReal[b] *= gain;
    fftImag[b] *= gain;

    // Mirror for real-valued signal
    if (b > 0 && b < kNumBins - 1) {
      fftReal[kFftSize - b] *= gain;
      fftImag[kFftSize - b] *= gain;
    }
  }
}

// =============================================================================
// Safety Limiter — hard clip at -12 dBFS (non-negotiable)
// =============================================================================

void MaskingEngine::applySafetyLimiter(float *buffer, int32_t numFrames) {
  // Soft-knee tanh saturation instead of hard clipping.
  // Hard clipping creates sharp waveform corners = audible distortion.
  // tanh(x) smoothly rolls off peaks toward ±1.0, preserving
  // waveform continuity and eliminating clipping artifacts.
  float invLimit = 1.0f / mSafetyLimitLinear;
  for (int32_t i = 0; i < numFrames; ++i) {
    // Scale into [-1,1] range relative to limit, saturate, scale back
    buffer[i] = mSafetyLimitLinear * tanhf(buffer[i] * invLimit);
  }
}

// =============================================================================
// Harmonic Drone — Fundamental Detection (kept for tonal mask, not for synth)
// =============================================================================

void MaskingEngine::updateFundamental() {
  float maxEnergy = 0.0f;
  int32_t maxBin = -1;

  for (int32_t b = 1; b < kNumBins; ++b) {
    if (mTonalMask[b] && mSpectralAverage[b] > maxEnergy) {
      maxEnergy = mSpectralAverage[b];
      maxBin = b;
    }
  }

  if (maxBin > 0) {
    float freq = static_cast<float>(maxBin) * static_cast<float>(mSampleRate) /
                 static_cast<float>(kFftSize);
    if (freq >= 50.0f && freq <= 800.0f) {
      if (mDetectedFundamental < 1.0f) {
        mDetectedFundamental = freq;
      } else {
        mDetectedFundamental = 0.995f * mDetectedFundamental + 0.005f * freq;
      }
    }
  } else {
    mDetectedFundamental *= 0.999f;
    if (mDetectedFundamental < 1.0f)
      mDetectedFundamental = 0.0f;
  }
}

// =============================================================================
// Harmonic Drone — Environment-Reactive Chord Progression (v3)
// =============================================================================
// Two independent timers:
//   1. mListenTimer (~5s): Re-samples the detected fundamental from the
//      environment, octave-folds to a pleasant register, and updates
//      voice target frequencies. The voices glide there via portamento.
//   2. mChordTimer (~20-40s): Changes the chord QUALITY (major, minor,
//      sus4, etc.) via weighted transition table for musical coherency.
//
// The GSC filter upstream subtracts our own speaker output from the mic
// input, preventing acoustic feedback from corrupting the detection.

void MaskingEngine::advanceChordProgression(int32_t numFrames) {
    float dt = static_cast<float>(numFrames) / static_cast<float>(mSampleRate);
    mListenTimer += dt;
    mChordTimer += dt;

    bool rootChanged = false;
    bool chordChanged = false;

    // --- Timer 1: Re-sample environment every ~5 seconds ---
    if (mListenTimer >= kListenInterval) {
        mListenTimer = 0.0f;

        float fund = mDetectedFundamental;

        if (fund > 20.0f) {
            // Octave-fold to pleasant warm register (80-220 Hz)
            while (fund > 220.0f) fund *= 0.5f;
            while (fund < 80.0f) fund *= 2.0f;

            // Only update if root shifted meaningfully (>5 Hz)
            // to avoid constant micro-adjustments
            if (fabsf(fund - mLockedRoot) > 5.0f) {
                mLockedRoot = fund;
                rootChanged = true;
            }
        }
        // If no tonal noise detected, keep the current locked root
        // (fallback was set at init to kFallbackRoot = C3)
    }

    // --- Timer 2: Change chord quality every ~20-40 seconds ---
    if (mChordTimer >= mCurrentChordDuration) {
        mChordTimer = 0.0f;
        chordChanged = true;

        // Weighted transition table for musical coherency
        //   0=I  1=i  2=IV  3=I8  4=5  5=i7  6=Imaj7  7=sus8
        static const uint8_t kTransitions[kNumChords][kNumChords] = {
            {0, 2, 4, 2, 3, 2, 3, 2},  // FROM I
            {3, 0, 2, 1, 2, 3, 1, 2},  // FROM i
            {3, 2, 0, 2, 3, 1, 2, 2},  // FROM IV
            {2, 1, 3, 0, 3, 2, 2, 2},  // FROM I8
            {4, 1, 2, 2, 0, 2, 2, 2},  // FROM 5
            {3, 3, 2, 1, 2, 0, 2, 2},  // FROM i7
            {3, 1, 2, 2, 2, 2, 0, 3},  // FROM Imaj7
            {4, 2, 2, 2, 2, 1, 2, 0},  // FROM sus8
        };

        mPrevChordIndex = mCurrentChordIndex;

        int32_t totalWeight = 0;
        for (int32_t c = 0; c < kNumChords; ++c)
            totalWeight += kTransitions[mPrevChordIndex][c];

        mRngState ^= mRngState << 13;
        mRngState ^= mRngState >> 17;
        mRngState ^= mRngState << 5;
        int32_t roll = static_cast<int32_t>(mRngState % totalWeight);

        int32_t next = 0;
        int32_t cumulative = 0;
        for (int32_t c = 0; c < kNumChords; ++c) {
            cumulative += kTransitions[mPrevChordIndex][c];
            if (roll < cumulative) { next = c; break; }
        }
        mCurrentChordIndex = next;

        // Randomize duration for next chord
        mRngState ^= mRngState << 13;
        mRngState ^= mRngState >> 17;
        mRngState ^= mRngState << 5;
        float r = static_cast<float>(mRngState & 0xFFFF) / 65535.0f;
        mCurrentChordDuration = kChordChangeMin +
                                r * (kChordChangeMax - kChordChangeMin);

        // Randomize vibrato per voice for organic feel
        for (int32_t v = 0; v < kDroneVoices; ++v) {
            mRngState ^= mRngState << 13;
            mRngState ^= mRngState >> 17;
            mRngState ^= mRngState << 5;
            float vr = static_cast<float>(mRngState & 0xFFFF) / 65535.0f;
            mDroneVoices[v].vibratoRate = 0.8f + vr * 2.0f;
            mDroneVoices[v].vibratoDepth = 1.5f + vr * 2.0f;
        }
    }

    // --- Update voice targets if root or chord changed ---
    if (rootChanged || chordChanged) {
        for (int32_t v = 0; v < kDroneVoices; ++v) {
            auto& voice = mDroneVoices[v];
            voice.targetFreq = mLockedRoot * kChordRatios[mCurrentChordIndex][v];
            voice.amplitude = (v == 0) ? 0.30f :
                              (v == 3) ? 0.15f : 0.20f;
        }

        LOGI("Drone → root=%.0fHz, chord=%d (%s), dur=%.0fs %s%s",
             mLockedRoot, mCurrentChordIndex,
             mCurrentChordIndex == 0 ? "I" :
             mCurrentChordIndex == 1 ? "i" :
             mCurrentChordIndex == 2 ? "IV" :
             mCurrentChordIndex == 3 ? "I8" :
             mCurrentChordIndex == 4 ? "5th" :
             mCurrentChordIndex == 5 ? "i7" :
             mCurrentChordIndex == 6 ? "Imaj7" : "sus",
             mCurrentChordDuration,
             rootChanged ? "[env-shift]" : "",
             chordChanged ? "[chord-change]" : "");
    }
}

// =============================================================================
// Harmonic Drone — Portamento Synthesis
// =============================================================================
// Each voice smoothly glides toward its target frequency every sample.
// Pure sine waveform with gentle vibrato. Fundamentally incapable
// of producing clicks or pops because frequency changes are continuous.

void MaskingEngine::synthesizeHarmonic(float *output, int32_t numFrames) {
  advanceChordProgression(numFrames);

  float invSr = 1.0f / static_cast<float>(mSampleRate);

  for (int32_t i = 0; i < numFrames; ++i) {
    float sum = 0.0f;

    for (int32_t v = 0; v < kDroneVoices; ++v) {
      auto &voice = mDroneVoices[v];

      // --- Portamento: smooth exponential glide toward target ---
      // This is the core anti-click mechanism. Frequency changes
      // are spread over thousands of samples, guaranteeing
      // continuous waveform with zero discontinuities.
      voice.currentFreq +=
          (voice.targetFreq - voice.currentFreq) * kPortamentoRate;

      // --- Vibrato: very gentle pitch modulation ---
      voice.vibratoPhase += voice.vibratoRate * invSr;
      if (voice.vibratoPhase >= 1.0f)
        voice.vibratoPhase -= 1.0f;
      float vibrato = sinf(2.0f * M_PI * voice.vibratoPhase);
      float vibratoRatio = powf(2.0f, vibrato * voice.vibratoDepth / 1200.0f);
      float freq = voice.currentFreq * vibratoRatio;

      // --- Phase accumulation ---
      voice.phase += freq * invSr;
      if (voice.phase >= 1.0f)
        voice.phase -= 1.0f;

      // --- Pure sine wave — clean and warm ---
      float wave = sinf(2.0f * M_PI * voice.phase);

      sum += wave * voice.amplitude;
    }

    // Whisper of brown noise for organic texture (3%)
    sum += generateBrownNoise() * 0.03f;

    output[i] = sum;
  }
}

} // namespace less
