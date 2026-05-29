// =============================================================================
// adapter_trainer.cpp — Production SPSA + LiteRT Training Implementation
// =============================================================================
// Phase 4 production upgrade with real LiteRT inference and momentum SPSA.
//
// PERFORMANCE COMPARISON:
//
//   Method                  Forward passes / step    For 4,096 params
//   ──────────────────────────────────────────────────────────────────
//   Finite-difference        2 × P                   8,192
//   SPSA (standard)          2 (constant!)            2
//   SPSA (2-pass averaged)   4 (constant!)            4
//   Speedup vs FD            —                        2,048× – 4,096×
//
// Phase 4 ENHANCEMENTS:
//
//   1. LiteRT Forward Pass:
//      - Loads a SEPARATE TFLite interpreter for training
//      - Runs on WorkManager background thread (NOT the audio thread)
//      - Can use multi-threaded XNNPACK since we're off the RT path
//
//   2. Momentum-Accelerated SPSA:
//      - Gradient EMA: m_k = β·m_{k-1} + (1-β)·ĝ_k
//      - Smooths noisy SPSA gradients → faster convergence
//      - Especially important in early training when perturbations are large
//
//   3. FGS Budget Guard:
//      - Android 15: 6h cumulative foreground service limit per 24h
//      - Training checks elapsed time and gracefully stops at 5.5h
//      - Checkpoints every 100 batches so work is never lost
//
// SPSA UPDATE RULE:
//   1. Δ_i ~ Bernoulli(±1) for all i ∈ [0, 4096)
//   2. θ⁺ = θ + c_k·Δ,  θ⁻ = θ - c_k·Δ
//   3. L⁺ = L(θ⁺; batch),  L⁻ = L(θ⁻; batch)
//   4. ĝ_i = (L⁺ - L⁻) · Δ_i / (2·c_k)
//   5. m_i = β·m_i + (1-β)·ĝ_i            [with momentum]
//   6. θ ← θ - a_k · clip(m)
//
// Memory budget: ~300KB total (adapter + 2 perturbed + grads + momentum + Δ)
// =============================================================================

#include "adapter_trainer.h"

#include <android/log.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <thread>   // Phase 7: std::this_thread::sleep_for for duty-cycle gaps

// LiteRT C API for training interpreter
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_types.h"

#define LOG_TAG "LESS_Trainer"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace less {
namespace training {

// =============================================================================
// Construction / Destruction
// =============================================================================

AdapterTrainer::AdapterTrainer(const TrainingConfig& config)
    : mConfig(config), mRng(42)
{
    std::memset(mGradDown, 0, sizeof(mGradDown));
    std::memset(mGradUp, 0, sizeof(mGradUp));
    std::memset(mMomentumDown, 0, sizeof(mMomentumDown));
    std::memset(mMomentumUp, 0, sizeof(mMomentumUp));
    std::memset(mDeltaDown, 0, sizeof(mDeltaDown));
    std::memset(mDeltaUp, 0, sizeof(mDeltaUp));

    LOGI("AdapterTrainer created [SPSA + Momentum]: epochs=%d, a=%.6f, c=%.6f, "
         "α=%.3f, γ=%.3f, β=%.3f, budget=%.1fh",
         config.epochs, config.spsa_a, config.spsa_c,
         config.spsa_alpha, config.spsa_gamma,
         config.momentum, config.maxTrainingHours);
}

AdapterTrainer::~AdapterTrainer() {
    destroyTrainingModel();
}

// =============================================================================
// Phase 4: LiteRT Training Model — Separate from RT Engine
// =============================================================================
// The training interpreter is NOT shared with the real-time audio engine.
// This avoids thread safety issues and allows different configurations:
//   - Training: multi-threaded XNNPACK for throughput
//   - Real-time: single-threaded with QNN HTP for latency

bool AdapterTrainer::loadTrainingModel(const std::string& modelPath) {
    LOGI("Loading LiteRT model for training: %s", modelPath.c_str());

    mTrainingModel = TfLiteModelCreateFromFile(modelPath.c_str());
    if (!mTrainingModel) {
        LOGW("Failed to load training model — using simulated forward pass");
        return false;
    }

    mTrainingOptions = TfLiteInterpreterOptionsCreate();

    // Multi-threaded for training throughput (this is OFF the RT path)
    // Use 2 threads — enough for a DTLN model without over-subscribing
    TfLiteInterpreterOptionsSetNumThreads(mTrainingOptions, 2);

    // XNNPACK for training (NPU delegation would require synchronous
    // completion, which adds DMA latency that hurts throughput for the
    // many sequential small invocations in the training loop)
    // TfLiteInterpreterOptionsSetEnableXNNPACK(mTrainingOptions, true); // Deprecated in 2.16.1, enabled by default

    mTrainingInterpreter = TfLiteInterpreterCreate(
        mTrainingModel, mTrainingOptions
    );
    if (!mTrainingInterpreter) {
        LOGE("Failed to create training interpreter");
        destroyTrainingModel();
        return false;
    }

    TfLiteStatus status = TfLiteInterpreterAllocateTensors(mTrainingInterpreter);
    if (status != kTfLiteOk) {
        LOGE("Failed to allocate training tensors");
        destroyTrainingModel();
        return false;
    }

    // Cache tensor pointers
    TfLiteTensor* inputTensor =
        TfLiteInterpreterGetInputTensor(mTrainingInterpreter, 0);
    const TfLiteTensor* outputTensor =
        TfLiteInterpreterGetOutputTensor(mTrainingInterpreter, 0);

    if (!inputTensor || !outputTensor) {
        LOGE("Training model must have input and output tensors");
        destroyTrainingModel();
        return false;
    }

    mTrainingInputData = reinterpret_cast<float*>(
        TfLiteTensorData(inputTensor)
    );
    mTrainingOutputData = const_cast<float*>(
        reinterpret_cast<const float*>(TfLiteTensorData(outputTensor))
    );
    mTrainingInputSize = static_cast<int32_t>(
        TfLiteTensorByteSize(inputTensor) / sizeof(float)
    );
    mTrainingOutputSize = static_cast<int32_t>(
        TfLiteTensorByteSize(outputTensor) / sizeof(float)
    );

    mTrainingModelLoaded = true;
    LOGI("✓ Training model ready: inputSize=%d, outputSize=%d (2-thread XNNPACK)",
         mTrainingInputSize, mTrainingOutputSize);

    return true;
}

void AdapterTrainer::destroyTrainingModel() {
    if (mTrainingInterpreter) {
        TfLiteInterpreterDelete(mTrainingInterpreter);
        mTrainingInterpreter = nullptr;
    }
    if (mTrainingOptions) {
        TfLiteInterpreterOptionsDelete(mTrainingOptions);
        mTrainingOptions = nullptr;
    }
    if (mTrainingModel) {
        TfLiteModelDelete(mTrainingModel);
        mTrainingModel = nullptr;
    }
    mTrainingInputData = nullptr;
    mTrainingOutputData = nullptr;
    mTrainingModelLoaded = false;
}

// =============================================================================
// Adapter Persistence
// =============================================================================

bool AdapterTrainer::loadAdapter(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOGI("No existing adapter at %s — starting from initialization", path.c_str());
        mAdapter = AdapterWeights{};
        return true;
    }

    file.read(reinterpret_cast<char*>(mAdapter.down), sizeof(mAdapter.down));
    file.read(reinterpret_cast<char*>(mAdapter.up), sizeof(mAdapter.up));

    if (!file.good()) {
        LOGE("Corrupt adapter file at %s — reinitializing", path.c_str());
        mAdapter = AdapterWeights{};
        return true;
    }

    LOGI("Loaded existing adapter from %s (%d params)", path.c_str(),
         AdapterWeights::kTotalParams);
    return true;
}

bool AdapterTrainer::saveAdapter(const std::string& path) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        LOGE("Failed to save adapter to %s", path.c_str());
        return false;
    }

    file.write(reinterpret_cast<const char*>(mAdapter.down), sizeof(mAdapter.down));
    file.write(reinterpret_cast<const char*>(mAdapter.up), sizeof(mAdapter.up));
    file.flush();

    LOGI("Adapter saved to %s", path.c_str());
    return true;
}

// =============================================================================
// Forward Pass — dispatches to LiteRT or simulated gate
// =============================================================================

void AdapterTrainer::forwardPass(
    const float* noisyInput,
    float* denoisedOutput,
    int32_t numSamples,
    const AdapterWeights& adapter)
{
    if (mTrainingModelLoaded) {
        forwardPassLiteRT(noisyInput, denoisedOutput, numSamples, adapter);
    } else {
        forwardPassSimulated(noisyInput, denoisedOutput, numSamples, adapter);
    }
}

// =============================================================================
// Phase 4: LiteRT-backed Forward Pass
// =============================================================================
// Uses the training interpreter to run the DTLN base model, then applies
// the LoRA adapter residual on top. This gives semantically meaningful
// gradients since SPSA now perturbs through the REAL model output.

void AdapterTrainer::forwardPassLiteRT(
    const float* noisyInput,
    float* denoisedOutput,
    int32_t numSamples,
    const AdapterWeights& adapter)
{
    // Copy input to the model's input tensor
    int32_t copySize = std::min(numSamples, mTrainingInputSize);
    std::memcpy(mTrainingInputData, noisyInput, copySize * sizeof(float));

    // Zero-pad if input is smaller than model expects
    if (copySize < mTrainingInputSize) {
        std::memset(mTrainingInputData + copySize, 0,
                   (mTrainingInputSize - copySize) * sizeof(float));
    }

    // Run base model inference
    TfLiteStatus status = TfLiteInterpreterInvoke(mTrainingInterpreter);
    if (status != kTfLiteOk) {
        // Fall back to simulated pass if inference fails
        LOGW("Training inference failed — falling back to simulated pass");
        forwardPassSimulated(noisyInput, denoisedOutput, numSamples, adapter);
        return;
    }

    // Copy base model output
    int32_t outCopy = std::min(numSamples, mTrainingOutputSize);
    std::memcpy(denoisedOutput, mTrainingOutputData, outCopy * sizeof(float));
    if (outCopy < numSamples) {
        std::memset(denoisedOutput + outCopy, 0,
                   (numSamples - outCopy) * sizeof(float));
    }

    // Apply LoRA adapter residual on top of base model output
    int32_t projSamples = std::min(numSamples, AdapterWeights::kHiddenDim);
    float bottleneck[AdapterWeights::kRank];

    // Down-projection: input → bottleneck
    for (int32_t r = 0; r < AdapterWeights::kRank; ++r) {
        float sum = 0.0f;
        for (int32_t s = 0; s < projSamples; ++s) {
            sum += denoisedOutput[s] * adapter.down[s * AdapterWeights::kRank + r];
        }
        bottleneck[r] = tanhf(sum);
    }

    // Up-projection: bottleneck → output residual
    for (int32_t s = 0; s < projSamples; ++s) {
        float delta = 0.0f;
        for (int32_t r = 0; r < AdapterWeights::kRank; ++r) {
            delta += bottleneck[r] * adapter.up[r * AdapterWeights::kHiddenDim + s];
        }
        denoisedOutput[s] += delta;
    }
}

// =============================================================================
// Simulated Forward Pass (Fallback — no model available)
// =============================================================================

void AdapterTrainer::forwardPassSimulated(
    const float* noisyInput,
    float* denoisedOutput,
    int32_t numSamples,
    const AdapterWeights& adapter)
{
    // Energy-based soft gate (simulates base model)
    float frameEnergy = 0.0f;
    for (int32_t i = 0; i < numSamples; ++i) {
        frameEnergy += noisyInput[i] * noisyInput[i];
    }
    frameEnergy /= static_cast<float>(numSamples);

    float gateGain = 1.0f / (1.0f + expf(-10.0f * (frameEnergy - 0.01f)));
    for (int32_t i = 0; i < numSamples; ++i) {
        denoisedOutput[i] = noisyInput[i] * gateGain;
    }

    // Apply adapter
    int32_t projSamples = std::min(numSamples, AdapterWeights::kHiddenDim);
    float bottleneck[AdapterWeights::kRank];

    for (int32_t r = 0; r < AdapterWeights::kRank; ++r) {
        float sum = 0.0f;
        for (int32_t s = 0; s < projSamples; ++s) {
            sum += denoisedOutput[s] * adapter.down[s * AdapterWeights::kRank + r];
        }
        bottleneck[r] = tanhf(sum);
    }

    for (int32_t s = 0; s < projSamples; ++s) {
        float delta = 0.0f;
        for (int32_t r = 0; r < AdapterWeights::kRank; ++r) {
            delta += bottleneck[r] * adapter.up[r * AdapterWeights::kHiddenDim + s];
        }
        denoisedOutput[s] += delta;
    }
}

// =============================================================================
// Self-Supervised Loss
// =============================================================================

float AdapterTrainer::computeLoss(
    const TrainingFrame& frame,
    const AdapterWeights& adapter)
{
    int32_t numSamples = static_cast<int32_t>(frame.audio.size());
    std::vector<float> denoised(numSamples);

    forwardPass(frame.audio.data(), denoised.data(), numSamples, adapter);

    float loss = 0.0f;
    for (int32_t i = 0; i < numSamples; ++i) {
        float diff = denoised[i] - frame.audio[i];
        loss += diff * diff;
    }

    loss *= frame.weight;
    loss /= static_cast<float>(numSamples);
    return loss;
}

float AdapterTrainer::computeBatchLoss(
    const std::vector<TrainingFrame>& batch,
    const AdapterWeights& adapter)
{
    float totalLoss = 0.0f;
    for (const auto& frame : batch) {
        totalLoss += computeLoss(frame, adapter);
    }
    return totalLoss / static_cast<float>(batch.size());
}

// =============================================================================
// SPSA GRADIENT ESTIMATION — 2 Forward Passes Per Step
// =============================================================================
//
// ALGORITHM (per training step k):
//   1. Δ_i ~ Bernoulli(±1) for all i ∈ [0, 4096)
//   2. c_k = c / (k+1)^γ
//   3. θ⁺ = θ + c_k·Δ,  θ⁻ = θ - c_k·Δ
//   4. L⁺ = L(θ⁺; batch),  L⁻ = L(θ⁻; batch)     ← THE 2 PASSES
//   5. ĝ_i = (L⁺ - L⁻) · Δ_i / (2·c_k)
//
// TOTAL COST: exactly 2 forward passes, regardless of 4,096 parameters.

void AdapterTrainer::computeGradientsSPSA(
    const std::vector<TrainingFrame>& batch)
{
    // --- Step 1: Generate random perturbation vector Δ ---
    generatePerturbation();

    // --- Step 2: Compute gain schedule ---
    float c_k = getPerturbationSize(mIterationCount);

    // --- Step 3: Build perturbed weight sets θ⁺ and θ⁻ ---
    buildPerturbedWeights(c_k);

    // --- Step 4: Evaluate losses (THE ONLY 2 FORWARD PASSES) ---
    float lossPlus  = computeBatchLoss(batch, mPerturbedPlus);
    float lossMinus = computeBatchLoss(batch, mPerturbedMinus);

    // --- Step 5: Estimate gradient for ALL parameters simultaneously ---
    // ĝ_i = (L⁺ - L⁻) · Δ_i / (2·c_k)
    float lossDiff = lossPlus - lossMinus;
    float inv2c = 1.0f / (2.0f * c_k);

    int32_t downSize = AdapterWeights::kHiddenDim * AdapterWeights::kRank;
    for (int32_t i = 0; i < downSize; ++i) {
        mGradDown[i] = lossDiff * mDeltaDown[i] * inv2c;
    }

    int32_t upSize = AdapterWeights::kRank * AdapterWeights::kHiddenDim;
    for (int32_t i = 0; i < upSize; ++i) {
        mGradUp[i] = lossDiff * mDeltaUp[i] * inv2c;
    }
}

// =============================================================================
// Phase 4: Two-Pass Averaged SPSA — Reduced Variance
// =============================================================================
// Uses two independent perturbation vectors and averages the gradient
// estimates. This halves the variance at the cost of 4 forward passes
// per step (still constant w.r.t. parameter count).
//
// ĝ_final = (ĝ_1 + ĝ_2) / 2
//
// Each ĝ_j is computed with an independent Δ_j, so the errors are
// uncorrelated and averaging reduces variance by ~50%.

void AdapterTrainer::computeGradientsSPSA2Pass(
    const std::vector<TrainingFrame>& batch)
{
    float c_k = getPerturbationSize(mIterationCount);
    int32_t downSize = AdapterWeights::kHiddenDim * AdapterWeights::kRank;
    int32_t upSize = AdapterWeights::kRank * AdapterWeights::kHiddenDim;

    // Temporary accumulators
    float accumDown[AdapterWeights::kHiddenDim * AdapterWeights::kRank];
    float accumUp[AdapterWeights::kRank * AdapterWeights::kHiddenDim];
    std::memset(accumDown, 0, sizeof(accumDown));
    std::memset(accumUp, 0, sizeof(accumUp));

    // Two independent SPSA passes
    for (int pass = 0; pass < 2; ++pass) {
        generatePerturbation();
        buildPerturbedWeights(c_k);

        float lossPlus  = computeBatchLoss(batch, mPerturbedPlus);
        float lossMinus = computeBatchLoss(batch, mPerturbedMinus);

        float lossDiff = lossPlus - lossMinus;
        float inv2c = 1.0f / (2.0f * c_k);

        for (int32_t i = 0; i < downSize; ++i) {
            accumDown[i] += lossDiff * mDeltaDown[i] * inv2c;
        }
        for (int32_t i = 0; i < upSize; ++i) {
            accumUp[i] += lossDiff * mDeltaUp[i] * inv2c;
        }
    }

    // Average the two estimates
    for (int32_t i = 0; i < downSize; ++i) {
        mGradDown[i] = accumDown[i] * 0.5f;
    }
    for (int32_t i = 0; i < upSize; ++i) {
        mGradUp[i] = accumUp[i] * 0.5f;
    }
}

// =============================================================================
// Perturbation Generation — Bernoulli(±1)
// =============================================================================

void AdapterTrainer::generatePerturbation() {
    std::bernoulli_distribution coin(0.5);

    int32_t downSize = AdapterWeights::kHiddenDim * AdapterWeights::kRank;
    for (int32_t i = 0; i < downSize; ++i) {
        mDeltaDown[i] = coin(mRng) ? 1.0f : -1.0f;
    }

    int32_t upSize = AdapterWeights::kRank * AdapterWeights::kHiddenDim;
    for (int32_t i = 0; i < upSize; ++i) {
        mDeltaUp[i] = coin(mRng) ? 1.0f : -1.0f;
    }
}

// =============================================================================
// Build Perturbed Weights: θ⁺ = θ + c_k·Δ,  θ⁻ = θ - c_k·Δ
// =============================================================================

void AdapterTrainer::buildPerturbedWeights(float c_k) {
    int32_t downSize = AdapterWeights::kHiddenDim * AdapterWeights::kRank;
    for (int32_t i = 0; i < downSize; ++i) {
        float perturbation = c_k * mDeltaDown[i];
        mPerturbedPlus.down[i]  = mAdapter.down[i] + perturbation;
        mPerturbedMinus.down[i] = mAdapter.down[i] - perturbation;
    }

    int32_t upSize = AdapterWeights::kRank * AdapterWeights::kHiddenDim;
    for (int32_t i = 0; i < upSize; ++i) {
        float perturbation = c_k * mDeltaUp[i];
        mPerturbedPlus.up[i]  = mAdapter.up[i] + perturbation;
        mPerturbedMinus.up[i] = mAdapter.up[i] - perturbation;
    }
}

// =============================================================================
// SPSA Gain Schedules (Spall, 1992)
// =============================================================================

float AdapterTrainer::getStepSize(int32_t k) const {
    return mConfig.spsa_a / powf(
        static_cast<float>(k + 1) + mConfig.spsa_A,
        mConfig.spsa_alpha
    );
}

float AdapterTrainer::getPerturbationSize(int32_t k) const {
    return mConfig.spsa_c / powf(
        static_cast<float>(k + 1),
        mConfig.spsa_gamma
    );
}

// =============================================================================
// Gradient Application — SGD with Momentum + Clipping
// =============================================================================
// Phase 4: Added momentum accumulation for SPSA noise reduction.
//
// Without momentum:    θ ← θ - a_k · clip(ĝ)
// With momentum:       m ← β·m + (1-β)·ĝ
//                      θ ← θ - a_k · clip(m)
//
// Momentum acts as a low-pass filter on the noisy SPSA gradient estimates,
// significantly improving convergence stability.

float AdapterTrainer::computeGradientNorm() const {
    float norm = 0.0f;
    int32_t downSize = AdapterWeights::kHiddenDim * AdapterWeights::kRank;
    for (int32_t i = 0; i < downSize; ++i) {
        norm += mGradDown[i] * mGradDown[i];
    }
    int32_t upSize = AdapterWeights::kRank * AdapterWeights::kHiddenDim;
    for (int32_t i = 0; i < upSize; ++i) {
        norm += mGradUp[i] * mGradUp[i];
    }
    return sqrtf(norm);
}

void AdapterTrainer::applyGradients(float a_k) {
    int32_t downSize = AdapterWeights::kHiddenDim * AdapterWeights::kRank;
    int32_t upSize = AdapterWeights::kRank * AdapterWeights::kHiddenDim;

    // =========================================================================
    // Phase 4: Momentum accumulation
    // =========================================================================
    // m_k = β · m_{k-1} + (1 - β) · ĝ_k
    //
    // This is an exponential moving average that smooths the inherently
    // noisy SPSA gradient estimates. Without momentum, SPSA convergence
    // can be erratic because each ĝ has high variance (estimated from
    // a single random direction). Momentum filters this noise and
    // preserves the consistent gradient signal.

    if (mConfig.useMomentum) {
        float beta = mConfig.momentum;
        float oneMinusBeta = 1.0f - beta;

        for (int32_t i = 0; i < downSize; ++i) {
            mMomentumDown[i] = beta * mMomentumDown[i] +
                               oneMinusBeta * mGradDown[i];
        }
        for (int32_t i = 0; i < upSize; ++i) {
            mMomentumUp[i] = beta * mMomentumUp[i] +
                             oneMinusBeta * mGradUp[i];
        }

        // Compute momentum norm for clipping
        float momentumNorm = 0.0f;
        for (int32_t i = 0; i < downSize; ++i) {
            momentumNorm += mMomentumDown[i] * mMomentumDown[i];
        }
        for (int32_t i = 0; i < upSize; ++i) {
            momentumNorm += mMomentumUp[i] * mMomentumUp[i];
        }
        momentumNorm = sqrtf(momentumNorm);

        float clipScale = 1.0f;
        if (momentumNorm > mConfig.maxGradientNorm) {
            clipScale = mConfig.maxGradientNorm / momentumNorm;
        }

        // SGD update with momentum: θ ← θ - a_k · clip(m)
        for (int32_t i = 0; i < downSize; ++i) {
            mAdapter.down[i] -= a_k * clipScale * mMomentumDown[i];
        }
        for (int32_t i = 0; i < upSize; ++i) {
            mAdapter.up[i] -= a_k * clipScale * mMomentumUp[i];
        }

        mProgress.gradientNorm = momentumNorm;
    } else {
        // Standard SPSA without momentum
        float gradNorm = computeGradientNorm();
        float clipScale = 1.0f;
        if (gradNorm > mConfig.maxGradientNorm) {
            clipScale = mConfig.maxGradientNorm / gradNorm;
        }

        for (int32_t i = 0; i < downSize; ++i) {
            mAdapter.down[i] -= a_k * clipScale * mGradDown[i];
        }
        for (int32_t i = 0; i < upSize; ++i) {
            mAdapter.up[i] -= a_k * clipScale * mGradUp[i];
        }

        mProgress.gradientNorm = gradNorm;
    }
}

// =============================================================================
// Phase 4: FGS Budget Guard
// =============================================================================
// Android 15 enforces a 6-hour cumulative foreground service limit per
// 24-hour window. We stop training at 5.5 hours to leave margin for
// checkpoint saving and cleanup.

bool AdapterTrainer::hasTimeBudget(double elapsedSeconds) const {
    double maxSeconds = mConfig.maxTrainingHours * 3600.0;
    return elapsedSeconds < maxSeconds;
}

// =============================================================================
// Main Training Loop
// =============================================================================

bool AdapterTrainer::train(
    const std::string& corpusDir,
    const std::string& baseModelPath,
    const std::string& adapterPath)
{
    LOGI("=== LESS Training Starting [Phase 4: SPSA + LiteRT + Momentum] ===");
    LOGI("Corpus: %s", corpusDir.c_str());
    LOGI("Base model: %s", baseModelPath.c_str());
    LOGI("Adapter: %s", adapterPath.c_str());
    LOGI("Config: epochs=%d, a=%.6f, c=%.6f, α=%.3f, γ=%.3f, β=%.3f",
         mConfig.epochs, mConfig.spsa_a, mConfig.spsa_c,
         mConfig.spsa_alpha, mConfig.spsa_gamma, mConfig.momentum);
    LOGI("Budget: %.1f hours max", mConfig.maxTrainingHours);
    if (mConfig.enableDutyCycling) {
        LOGI("Duty cycling: %d batches / cycle, %d ms sleep",
             mConfig.microBatchesPerDutyCycle, mConfig.dutyCycleSleepMs);
    } else {
        LOGI("Duty cycling: DISABLED (continuous compute)");
    }

    auto trainingStart = std::chrono::steady_clock::now();

    mCancelled.store(false);
    mIterationCount = 0;

    // Reset momentum buffers for fresh training
    std::memset(mMomentumDown, 0, sizeof(mMomentumDown));
    std::memset(mMomentumUp, 0, sizeof(mMomentumUp));

    // --- Load LiteRT model for training forward pass ---
    if (!baseModelPath.empty()) {
        if (loadTrainingModel(baseModelPath)) {
            mProgress.backendName = "LiteRT (XNNPACK 2-thread)";
        } else {
            mProgress.backendName = "Simulated (no model)";
            LOGW("Training will use simulated forward pass");
        }
    } else {
        mProgress.backendName = "Simulated (no model path)";
        LOGW("No base model provided — using simulated forward pass");
    }

    // --- Load or initialize adapter ---
    if (!loadAdapter(adapterPath)) {
        LOGE("Failed to load adapter — aborting training");
        destroyTrainingModel();
        return false;
    }

    // --- Load and prepare training data ---
    DataPipelineConfig dpConfig;
    dpConfig.frameSize = mConfig.frameSize;
    dpConfig.sampleRate = mConfig.sampleRate;
    DataPipeline pipeline(dpConfig);

    int32_t totalFrames = pipeline.loadCorpus(corpusDir);
    if (totalFrames < 10) {
        LOGE("Insufficient training data: %d frames (need ≥10)", totalFrames);
        destroyTrainingModel();
        return false;
    }

    LOGI("Corpus loaded: %d frames, mean weight=%.3f, duration=%.1fs",
         totalFrames, pipeline.meanWeight(), pipeline.corpusDurationSeconds());

    // --- Training loop ---
    mProgress.totalEpochs = mConfig.epochs;
    mProgress.totalFrames = totalFrames;
    mProgress.isRunning = true;
    mProgress.isComplete = false;

    float bestLoss = std::numeric_limits<float>::max();
    int32_t patienceCounter = 0;

    for (int32_t epoch = 0; epoch < mConfig.epochs; ++epoch) {
        if (mCancelled.load(std::memory_order_relaxed)) {
            LOGW("Training cancelled at epoch %d", epoch);
            break;
        }

        // FGS budget check
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - trainingStart).count();
        mProgress.elapsedSeconds = elapsed;

        if (!hasTimeBudget(elapsed)) {
            LOGW("⚠ FGS budget exhausted at %.1f hours — saving and stopping",
                 elapsed / 3600.0);
            saveAdapter(adapterPath);
            break;
        }

        mProgress.currentEpoch = epoch + 1;
        pipeline.reset();

        float epochLoss = 0.0f;
        int32_t batchCount = 0;
        int32_t framesInEpoch = 0;

        std::vector<TrainingFrame> batch;
        while (pipeline.nextBatch(batch)) {
            if (mCancelled.load(std::memory_order_relaxed)) break;

            // Per-batch FGS budget check
            now = std::chrono::steady_clock::now();
            elapsed = std::chrono::duration<double>(now - trainingStart).count();
            if (!hasTimeBudget(elapsed)) break;

            // =================================================================
            // SPSA gradient estimation — 2 or 4 forward passes per batch
            // =================================================================
            if (mConfig.useTwoPassAveraging) {
                computeGradientsSPSA2Pass(batch);
            } else {
                computeGradientsSPSA(batch);
            }

            // Compute scheduled gains
            float a_k = getStepSize(mIterationCount);
            float c_k = getPerturbationSize(mIterationCount);

            // Apply SGD update with momentum + gradient clipping
            applyGradients(a_k);

            mIterationCount++;

            // Update progress diagnostics
            float batchLoss = computeBatchLoss(batch, mAdapter);
            epochLoss += batchLoss;
            batchCount++;
            framesInEpoch += static_cast<int32_t>(batch.size());

            mProgress.framesProcessed = framesInEpoch;
            mProgress.currentLoss = batchLoss;
            mProgress.iterationCount = mIterationCount;
            mProgress.stepSize = a_k;
            mProgress.perturbationSize = c_k;

            // Checkpoint periodically
            if (batchCount % mConfig.checkpointEveryNBatches == 0) {
                saveAdapter(adapterPath);
                LOGI("  Checkpoint: batch=%d, loss=%.6f, a_k=%.6f, c_k=%.6f, "
                     "‖∇‖=%.6f, elapsed=%.0fs",
                     batchCount, batchLoss, a_k, c_k,
                     mProgress.gradientNorm, elapsed);
            }

            // =================================================================
            // Phase 7: Duty-cycle thermal cooldown
            // =================================================================
            // After every N SPSA steps, pause briefly to let the SoC
            // dissipate heat. This prevents the device from entering
            // thermal throttling during the overnight training window.
            //
            // The sleep is implemented with std::this_thread::sleep_for
            // (NOT in the audio path) and is interruptible via mCancelled.
            if (mConfig.enableDutyCycling &&
                (batchCount % mConfig.microBatchesPerDutyCycle == 0)) {

                mProgress.dutyCyclesSoFar++;
                mProgress.totalSleepMs += mConfig.dutyCycleSleepMs;

                std::this_thread::sleep_for(
                    std::chrono::milliseconds(mConfig.dutyCycleSleepMs));

                // Re-check cancellation after the sleep
                if (mCancelled.load(std::memory_order_relaxed)) break;
            }
        }

        // Epoch summary
        float avgEpochLoss = (batchCount > 0) ? epochLoss / batchCount : 0.0f;
        now = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration<double>(now - trainingStart).count();

        LOGI("Epoch %d/%d: avgLoss=%.6f, frames=%d, batches=%d, "
             "iter=%d, elapsed=%.0fs (%.1fh)",
             epoch + 1, mConfig.epochs, avgEpochLoss, framesInEpoch,
             batchCount, mIterationCount, elapsed, elapsed / 3600.0);

        // Early stopping check
        if (avgEpochLoss < bestLoss - mConfig.minLossImprovement) {
            bestLoss = avgEpochLoss;
            mProgress.bestLoss = bestLoss;
            patienceCounter = 0;
            saveAdapter(adapterPath);
            LOGI("  New best loss: %.6f", bestLoss);
        } else {
            patienceCounter++;
            LOGI("  No improvement (%d/%d patience)", patienceCounter,
                 mConfig.patienceEpochs);
            if (patienceCounter >= mConfig.patienceEpochs) {
                LOGI("Early stopping triggered — no improvement for %d epochs",
                     patienceCounter);
                break;
            }
        }
    }

    // Final save
    saveAdapter(adapterPath);

    // Clean up training model
    destroyTrainingModel();

    auto trainingEnd = std::chrono::steady_clock::now();
    double totalSeconds =
        std::chrono::duration<double>(trainingEnd - trainingStart).count();

    mProgress.isRunning = false;
    mProgress.isComplete = !mCancelled.load();
    mProgress.elapsedSeconds = totalSeconds;

    LOGI("=== LESS Training %s ===",
         mProgress.isComplete ? "Complete" : "Cancelled");
    LOGI("  Backend: %s", mProgress.backendName);
    LOGI("  Iterations: %d (SPSA %s)",
         mIterationCount,
         mConfig.useTwoPassAveraging ? "2-pass averaged" : "standard");
    LOGI("  Duration: %.0fs (%.2fh)", totalSeconds, totalSeconds / 3600.0);
    LOGI("  Best loss: %.6f", mProgress.bestLoss);
    LOGI("  Momentum: %s (β=%.3f)",
         mConfig.useMomentum ? "enabled" : "disabled", mConfig.momentum);

    return mProgress.isComplete;
}

void AdapterTrainer::cancel() {
    LOGI("Training cancellation requested");
    mCancelled.store(true, std::memory_order_release);
}

TrainingProgress AdapterTrainer::getProgress() const {
    return mProgress;
}

} // namespace training
} // namespace less
