// =============================================================================
// adapter_trainer.h — Production SPSA-Optimized LoRA Adapter Training
// =============================================================================
// Phase 4 production upgrade: LiteRT-backed forward pass with momentum-
// accelerated SPSA gradient estimation.
// Phase 7 upgrade: Duty-cycled micro-batch training for thermal management.
//
// Architecture:
//   Frozen base model (~2MB .tflite) + mutable adapter weights (16KB)
//   Only the adapter weights are updated — 0.2% of total model parameters.
//
// Self-supervised loss:
//   L = Σ w_i × (denoised_i - pseudo_target_i)²
//   where w_i is the SNR-derived weight from SNRPredictor.
//
// Phase 4 SPSA Enhancements:
//   1. LiteRT-based forward pass (replaces simulated gate)
//   2. Momentum accumulation (exponential moving average of gradients)
//   3. 2-pass averaging for reduced variance (optional 2nd SPSA sample)
//   4. Iteration-aware batch scheduling for 6-hour FGS budget
//
// SPSA theory (Spall, 1992):
//   ĝ_i = (L(θ + c·Δ) - L(θ - c·Δ)) / (2c·Δ_i)
//   Total cost: 2 forward passes per step (decoupled from param count)
// =============================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <atomic>
#include <functional>
#include <random>
#include <memory>

#include "data_pipeline.h"
#include "../noise_suppressor.h"

// Forward declarations for LiteRT
struct TfLiteModel;
struct TfLiteInterpreterOptions;
struct TfLiteInterpreter;

namespace less {
namespace training {

struct TrainingConfig {
    int32_t epochs = 5;
    float learningRate = 0.001f;
    int32_t sampleRate = 48000;
    int32_t frameSize = 480;

    // SPSA hyperparameters (Spall's recommended schedule)
    float spsa_a = 0.001f;          // step size numerator
    float spsa_c = 0.01f;           // perturbation size
    float spsa_alpha = 0.602f;      // step size decay exponent
    float spsa_gamma = 0.101f;      // perturbation decay exponent
    float spsa_A = 10.0f;           // step size stabilizer

    // Phase 4: Momentum (reduces SPSA variance via gradient EMA)
    float momentum = 0.9f;          // β for gradient EMA: m_k = β·m_{k-1} + (1-β)·ĝ_k
    bool useMomentum = true;

    // Phase 4: Two-pass averaging (optional — reduces variance at 4 fwd passes)
    bool useTwoPassAveraging = false;

    // Checkpointing
    std::string checkpointDir;
    int32_t checkpointEveryNBatches = 100;

    // Gradient clipping
    float maxGradientNorm = 1.0f;

    // Early stopping
    float minLossImprovement = 1e-5f;
    int32_t patienceEpochs = 2;

    // Phase 4: FGS budget awareness (Android 15 6-hour limit)
    float maxTrainingHours = 5.5f;  // Leave margin for cleanup

    // Phase 7: Duty-Cycled Micro-Batch Training
    // ==========================================================================
    // To prevent thermal throttling during overnight training, we split
    // the continuous compute workload into short bursts with cooldown gaps.
    //
    // microBatchesPerDutyCycle:
    //   How many SPSA update steps to execute in a burst before sleeping.
    //   Each step involves 2-4 forward passes (~5-20ms total on XNNPACK).
    //   Running 8 steps gives ~40-160ms of continuous compute.
    //
    // dutyCycleSleepMs:
    //   Milliseconds to sleep between bursts. This allows the SoC to
    //   dissipate heat. The sleep is interruptible (checks mCancelled).
    //   50ms gives a ~70-80% compute duty cycle — sufficient for overnight
    //   training while keeping the device cool.
    //
    // Thermal budget math:
    //   5.5h budget × 3600 s/h = 19,800s total
    //   At 70% duty: ~13,860s of active compute
    //   At ~50ms/step: ~277,200 SPSA steps — far more than needed for
    //   a 4,096-parameter adapter with 5 epochs over a 15-min corpus.
    //
    int32_t microBatchesPerDutyCycle = 8;
    int32_t dutyCycleSleepMs = 50;
    bool enableDutyCycling = true;
};

// Progress information exposed to the Kotlin layer via JNI
struct TrainingProgress {
    int32_t currentEpoch{0};
    int32_t totalEpochs{0};
    float currentLoss{0.0f};
    float bestLoss{std::numeric_limits<float>::max()};
    int32_t framesProcessed{0};
    int32_t totalFrames{0};
    bool isRunning{false};
    bool isComplete{false};

    // Phase 4: Additional diagnostics
    int32_t iterationCount{0};
    float gradientNorm{0.0f};
    float stepSize{0.0f};
    float perturbationSize{0.0f};
    double elapsedSeconds{0.0};
    const char* backendName{"Unknown"};  // Delegate backend used for training

    // Phase 7: Duty-cycle diagnostics
    int32_t dutyCyclesSoFar{0};
    int32_t totalSleepMs{0};
};

class AdapterTrainer {
public:
    explicit AdapterTrainer(const TrainingConfig& config = TrainingConfig{});
    ~AdapterTrainer();

    // =========================================================================
    // Main entry point — runs the full training loop
    // =========================================================================
    bool train(
        const std::string& corpusDir,
        const std::string& baseModelPath,
        const std::string& adapterPath
    );

    void cancel();
    TrainingProgress getProgress() const;

private:
    TrainingConfig mConfig;

    // Adapter weights — the ONLY trainable parameters
    AdapterWeights mAdapter;

    // SPSA gradient estimate
    float mGradDown[AdapterWeights::kHiddenDim * AdapterWeights::kRank];
    float mGradUp[AdapterWeights::kRank * AdapterWeights::kHiddenDim];

    // Phase 4: Momentum buffers — exponential moving average of gradients
    float mMomentumDown[AdapterWeights::kHiddenDim * AdapterWeights::kRank];
    float mMomentumUp[AdapterWeights::kRank * AdapterWeights::kHiddenDim];

    // SPSA perturbation vector Δ — Bernoulli ±1
    float mDeltaDown[AdapterWeights::kHiddenDim * AdapterWeights::kRank];
    float mDeltaUp[AdapterWeights::kRank * AdapterWeights::kHiddenDim];

    // Perturbed adapter copies — reused across iterations
    AdapterWeights mPerturbedPlus;
    AdapterWeights mPerturbedMinus;

    // Training state
    std::atomic<bool> mCancelled{false};
    TrainingProgress mProgress;
    int32_t mIterationCount{0};

    // RNG for Bernoulli perturbation vector
    std::mt19937 mRng;

    // =========================================================================
    // Phase 4: LiteRT-based forward pass (replaces simulated gate)
    // =========================================================================
    // A SEPARATE interpreter instance for training (not shared with the
    // real-time engine). This runs on the WorkManager background thread,
    // NOT the audio thread, so it can allocate and use heavier delegates.
    TfLiteModel* mTrainingModel{nullptr};
    TfLiteInterpreterOptions* mTrainingOptions{nullptr};
    TfLiteInterpreter* mTrainingInterpreter{nullptr};
    float* mTrainingInputData{nullptr};
    float* mTrainingOutputData{nullptr};
    int32_t mTrainingInputSize{0};
    int32_t mTrainingOutputSize{0};
    bool mTrainingModelLoaded{false};

    bool loadTrainingModel(const std::string& modelPath);
    void destroyTrainingModel();

    // =========================================================================
    // Core training methods
    // =========================================================================
    bool loadAdapter(const std::string& path);
    bool saveAdapter(const std::string& path);

    float computeLoss(
        const TrainingFrame& frame,
        const AdapterWeights& adapter
    );

    float computeBatchLoss(
        const std::vector<TrainingFrame>& batch,
        const AdapterWeights& adapter
    );

    void forwardPass(
        const float* noisyInput,
        float* denoisedOutput,
        int32_t numSamples,
        const AdapterWeights& adapter
    );

    // Phase 4: LiteRT-backed forward pass
    void forwardPassLiteRT(
        const float* noisyInput,
        float* denoisedOutput,
        int32_t numSamples,
        const AdapterWeights& adapter
    );

    // Phase 4: Simulated forward pass (fallback if no model)
    void forwardPassSimulated(
        const float* noisyInput,
        float* denoisedOutput,
        int32_t numSamples,
        const AdapterWeights& adapter
    );

    // SPSA gradient estimation — 2 forward passes per step
    void computeGradientsSPSA(const std::vector<TrainingFrame>& batch);

    // Phase 4: Two-pass averaged SPSA (4 forward passes, ~50% less variance)
    void computeGradientsSPSA2Pass(const std::vector<TrainingFrame>& batch);

    void generatePerturbation();
    void buildPerturbedWeights(float c_k);
    void applyGradients(float a_k);
    float getStepSize(int32_t k) const;
    float getPerturbationSize(int32_t k) const;
    float computeGradientNorm() const;

    // Phase 4: FGS budget guard
    bool hasTimeBudget(double elapsedSeconds) const;
};

} // namespace training
} // namespace less
