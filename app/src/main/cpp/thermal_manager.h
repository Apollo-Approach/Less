// =============================================================================
// thermal_manager.h — ADPF Thermal Headroom Monitoring (Android 16 Upgraded)
// =============================================================================
// Phase 7 + 8.1: Integrates the Android Dynamic Performance Framework (ADPF)
// Thermal API with dual-mode operation:
//
//   API 36+ (Android 16): AThermal_HeadroomCallback listener — asynchronous,
//     OS-driven notifications on thermal threshold crossings. No polling needed.
//     The callback fires on a system binder thread and publishes headroom to
//     atomics for lock-free consumption by the audio thread.
//
//   API 31–35 (Android 12–15): Legacy AThermal_getThermalHeadroom polling.
//     Must be driven by a dedicated monitoring thread or coroutine at ≥1s
//     intervals. Provided for backward compatibility.
//
// Key concepts:
//   - Thermal headroom is a float in [0.0, ∞):
//       0.0 = cold (maximum performance available)
//       0.7 = warm (consider reducing workload — ATHERMAL_STATUS_LIGHT)
//       1.0 = throttling imminent → OS will force-throttle
//       >1.0 = already throttling
//
//   - The callback also provides forecastHeadroom (system-chosen lookahead)
//     and AThermalHeadroomThreshold array for interpreting thermal status.
//
// Policy for LESS:
//   Headroom < 0.5   →  Full NPU inference (maximum quality)
//   Headroom 0.5–0.7 →  NPU inference (log warning)
//   Headroom 0.7–0.9 →  Switch to CPU / XNNPACK (cooler than NPU sustained)
//   Headroom ≥ 0.9   →  Emergency fallback to spectral gate (minimal compute)
//
// Thread safety:
//   - In callback mode (API 36+): the callback fires on the system binder
//     thread pool. All state updates use atomic operations — safe for the
//     audio thread to read without locking.
//   - In polling mode (API 31-35): pollHeadroom() is called from a monitoring
//     thread. Same atomic publication pattern.
// =============================================================================

#pragma once

#include <cstdint>
#include <atomic>

// Android Thermal API (NDK)
// AThermalManager: API 31+
// AThermal_HeadroomCallback: API 36+
struct AThermalManager;

// Forward-declare the threshold struct used by the API 36 callback.
// The actual struct contains { AThermalStatus thermalStatus; float headroom; }
// We define a compatible layout here so we don't hard-link against API 36 headers.
struct AThermalHeadroomThreshold {
    int32_t thermalStatus;
    float   headroom;
};

namespace less {

// Thermal state as seen by the real-time engine
enum class ThermalState {
    kNominal,       // < 0.5 headroom — full NPU inference
    kWarm,          // 0.5–0.7 — NPU inference with warning
    kThrottling,    // 0.7–0.9 — switch to CPU (XNNPACK)
    kCritical       // ≥ 0.9 — emergency spectral gate fallback
};

class ThermalManager {
public:
    ThermalManager();
    ~ThermalManager();

    // Non-copyable
    ThermalManager(const ThermalManager&) = delete;
    ThermalManager& operator=(const ThermalManager&) = delete;

    // =========================================================================
    // Initialization — acquires AThermalManager handle + registers callbacks
    // =========================================================================
    // Must be called once during engine startup.
    // Returns false if ADPF Thermal API is not available (pre-API 31).
    //
    // On API 36+, this additionally registers the AThermal_HeadroomCallback
    // listener, eliminating the need for pollHeadroom() calls.
    bool initialize();

    // =========================================================================
    // Shutdown — unregisters callbacks and releases resources
    // =========================================================================
    // Called automatically by destructor, but can be invoked early.
    void shutdown();

    // =========================================================================
    // Thermal headroom query (legacy polling — API 31–35)
    // =========================================================================
    // On API 36+, this is a no-op that returns the callback-provided headroom.
    // On API 31–35, polls AThermal_getThermalHeadroom() and updates state.
    //
    // MUST be called from a non-audio thread. Respects 1-second minimum
    // polling interval.
    //
    // @param forecastSeconds  How many seconds ahead to predict (default 10s)
    // @return The raw thermal headroom value, or -1.0f if unavailable
    float pollHeadroom(int32_t forecastSeconds = 10);

    // =========================================================================
    // State accessors — lock-free, safe for audio thread
    // =========================================================================
    ThermalState getState() const {
        return static_cast<ThermalState>(
            mThermalState.load(std::memory_order_relaxed));
    }

    float getHeadroom() const {
        return mHeadroom.load(std::memory_order_relaxed);
    }

    float getForecastHeadroom() const {
        return mForecastHeadroom.load(std::memory_order_relaxed);
    }

    int32_t getForecastSeconds() const {
        return mForecastSeconds.load(std::memory_order_relaxed);
    }

    // Whether the ADPF API is available on this device (API 31+)
    bool isAvailable() const { return mAvailable; }

    // Whether we're using the async callback path (API 36+)
    bool isCallbackActive() const { return mCallbackRegistered; }

    // Human-readable state name
    static const char* stateName(ThermalState state);

    // =========================================================================
    // Policy thresholds — can be tuned at runtime
    // =========================================================================
    struct Thresholds {
        float warm     = 0.50f;   // Transition to kWarm
        float throttle = 0.70f;   // Transition to kThrottling (switch to CPU)
        float critical = 0.90f;   // Transition to kCritical (spectral gate)
    };

    void setThresholds(const Thresholds& t) { mThresholds = t; }
    const Thresholds& getThresholds() const { return mThresholds; }

    // =========================================================================
    // Updates thresholds from OEM-provided AThermalHeadroomThreshold array
    // =========================================================================
    // Called from the callback when the OS provides device-specific thresholds.
    // Maps ATHERMAL_STATUS_LIGHT → warm, ATHERMAL_STATUS_MODERATE → throttle,
    // ATHERMAL_STATUS_SEVERE → critical.
    void updateThresholdsFromOEM(const AThermalHeadroomThreshold* thresholds,
                                  size_t count);

private:
    bool mAvailable{false};
    bool mCallbackRegistered{false};
    AThermalManager* mThermalMgr{nullptr};

    // Lock-free state shared with the audio thread
    std::atomic<int32_t> mThermalState{
        static_cast<int32_t>(ThermalState::kNominal)};
    std::atomic<float> mHeadroom{0.0f};
    std::atomic<float> mForecastHeadroom{0.0f};
    std::atomic<int32_t> mForecastSeconds{0};

    Thresholds mThresholds;

    // Timestamp of last poll (legacy mode - to enforce minimum interval)
    int64_t mLastPollTimeNs{0};
    static constexpr int64_t kMinPollIntervalNs = 1'000'000'000LL;  // 1 second

    // =========================================================================
    // Dynamically loaded ADPF function pointers
    // =========================================================================
    // API 31+ (core)
    using AcquireFn = AThermalManager* (*)();
    using ReleaseFn = void (*)(AThermalManager*);
    using HeadroomFn = float (*)(AThermalManager*, int);

    AcquireFn mAcquireFn{nullptr};
    ReleaseFn mReleaseFn{nullptr};
    HeadroomFn mHeadroomFn{nullptr};

    // =========================================================================
    // API 36+ (headroom callback)
    // =========================================================================
    // AThermal_HeadroomCallback signature:
    //   void (*)(void* data, float headroom, float forecastHeadroom,
    //            int forecastSeconds,
    //            const AThermalHeadroomThreshold* thresholds, size_t count)
    using HeadroomCallbackFn = void (*)(void*, float, float, int,
                                        const AThermalHeadroomThreshold*, size_t);

    using RegisterHeadroomListenerFn = int (*)(AThermalManager*,
                                                HeadroomCallbackFn, void*);
    using UnregisterHeadroomListenerFn = int (*)(AThermalManager*,
                                                  HeadroomCallbackFn, void*);

    RegisterHeadroomListenerFn mRegisterListenerFn{nullptr};
    UnregisterHeadroomListenerFn mUnregisterListenerFn{nullptr};

    void* mLibHandle{nullptr};

    ThermalState classifyHeadroom(float headroom) const;

    // Publishes headroom + state atomically and logs transitions
    void applyHeadroom(float headroom, float forecastHeadroom,
                       int forecastSeconds);

    // =========================================================================
    // Static callback trampoline for AThermal_HeadroomCallback
    // =========================================================================
    // This is invoked by the OS on a system binder thread pool. It forwards
    // to the ThermalManager instance via the data pointer.
    static void onHeadroomChanged(void* data,
                                   float headroom,
                                   float forecastHeadroom,
                                   int forecastSeconds,
                                   const AThermalHeadroomThreshold* thresholds,
                                   size_t thresholdsCount);
};

} // namespace less
