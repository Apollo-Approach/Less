// =============================================================================
// thermal_manager.cpp — ADPF Thermal Headroom Implementation (Android 16+)
// =============================================================================
// Phase 7 + 8.1: Dual-mode thermal management:
//
//   API 36+ (Android 16): Uses AThermal_registerThermalHeadroomListener to
//     receive asynchronous callbacks when thermal headroom or thresholds change.
//     The callback fires on a system binder thread and publishes state to
//     atomics. pollHeadroom() becomes a no-op in this mode.
//
//   API 31–35 (Android 12–15): Falls back to AThermal_getThermalHeadroom
//     polling with 1-second minimum interval enforcement.
//
// Implementation notes:
//   - AThermalManager is loaded via dlopen("libandroid.so") to avoid
//     hard-linking against API 31+ or 36+ symbols. This allows a single
//     binary to run on all API levels with graceful degradation.
//   - Headroom values are published to atomic<float> for lock-free reading
//     by the real-time audio thread.
//   - State transitions are logged but NOT enacted here — the NoiseSuppressor
//     reads the state and decides which compute path to take.
//   - The callback trampoline uses the data pointer to recover the
//     ThermalManager instance — fully re-entrant and thread-safe.
// =============================================================================

#include "thermal_manager.h"

#include <android/log.h>
#include <dlfcn.h>
#include <ctime>
#include <cerrno>
#include <cmath>

#define LOG_TAG "LESS_Thermal"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// AThermalStatus enums used for interpreting OEM threshold levels.
// Defined in android/thermal.h but we avoid hard-including API 36 headers.
static constexpr int32_t ATHERMAL_STATUS_LIGHT    = 1;  // ATHERMAL_STATUS_LIGHT
static constexpr int32_t ATHERMAL_STATUS_MODERATE = 2;  // ATHERMAL_STATUS_MODERATE
static constexpr int32_t ATHERMAL_STATUS_SEVERE   = 3;  // ATHERMAL_STATUS_SEVERE

namespace less {

// =============================================================================
// State Name Strings
// =============================================================================

const char* ThermalManager::stateName(ThermalState state) {
    switch (state) {
        case ThermalState::kNominal:    return "Nominal";
        case ThermalState::kWarm:       return "Warm";
        case ThermalState::kThrottling: return "Throttling";
        case ThermalState::kCritical:   return "Critical";
        default: return "Unknown";
    }
}

// =============================================================================
// Construction / Destruction
// =============================================================================

ThermalManager::ThermalManager() = default;

ThermalManager::~ThermalManager() {
    shutdown();
}

void ThermalManager::shutdown() {
    // Unregister the headroom callback BEFORE releasing the manager.
    // AThermal_unregisterThermalHeadroomListener guarantees no subsequent
    // invocations after it returns successfully.
    if (mCallbackRegistered && mThermalMgr && mUnregisterListenerFn) {
        int result = mUnregisterListenerFn(
            mThermalMgr,
            &ThermalManager::onHeadroomChanged,
            this
        );
        if (result == 0) {
            LOGI("✓ Unregistered AThermal_HeadroomCallback listener");
        } else {
            LOGW("Failed to unregister headroom listener (errno=%d)", result);
        }
        mCallbackRegistered = false;
    }

    if (mThermalMgr && mReleaseFn) {
        mReleaseFn(mThermalMgr);
        mThermalMgr = nullptr;
    }
    if (mLibHandle) {
        dlclose(mLibHandle);
        mLibHandle = nullptr;
    }
    mAvailable = false;
}

// =============================================================================
// Initialization — Dynamic Loading of ADPF Symbols
// =============================================================================
// Load order:
//   1. Open libandroid.so
//   2. Resolve API 31+ core symbols (acquire, release, headroom)
//   3. Resolve API 36+ callback symbols (register/unregister listener)
//   4. If API 36+ available, register the headroom callback
//   5. If API 36+ unavailable, fall back to polling-only mode

bool ThermalManager::initialize() {
    LOGI("Initializing ADPF Thermal Manager...");

    // Open libandroid.so (always present on Android)
    mLibHandle = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
    if (!mLibHandle) {
        LOGW("Failed to open libandroid.so: %s", dlerror());
        LOGW("Thermal monitoring unavailable — running without ADPF");
        mAvailable = false;
        return false;
    }

    // =========================================================================
    // Step 1: Resolve API 31+ core symbols
    // =========================================================================
    mAcquireFn = reinterpret_cast<AcquireFn>(
        dlsym(mLibHandle, "AThermal_acquireManager"));
    mReleaseFn = reinterpret_cast<ReleaseFn>(
        dlsym(mLibHandle, "AThermal_releaseManager"));
    mHeadroomFn = reinterpret_cast<HeadroomFn>(
        dlsym(mLibHandle, "AThermal_getThermalHeadroom"));

    if (!mAcquireFn || !mReleaseFn || !mHeadroomFn) {
        LOGW("ADPF Thermal API not available (pre-API 31 device)");
        LOGW("Thermal monitoring disabled — using static power config");
        dlclose(mLibHandle);
        mLibHandle = nullptr;
        mAvailable = false;
        return false;
    }

    // Acquire the thermal manager handle
    mThermalMgr = mAcquireFn();
    if (!mThermalMgr) {
        LOGE("AThermal_acquireManager() returned null");
        dlclose(mLibHandle);
        mLibHandle = nullptr;
        mAvailable = false;
        return false;
    }

    mAvailable = true;

    // =========================================================================
    // Step 2: Probe for API 36+ headroom callback symbols
    // =========================================================================
    mRegisterListenerFn = reinterpret_cast<RegisterHeadroomListenerFn>(
        dlsym(mLibHandle, "AThermal_registerThermalHeadroomListener"));
    mUnregisterListenerFn = reinterpret_cast<UnregisterHeadroomListenerFn>(
        dlsym(mLibHandle, "AThermal_unregisterThermalHeadroomListener"));

    if (mRegisterListenerFn && mUnregisterListenerFn) {
        // =====================================================================
        // API 36+ AVAILABLE — Register the async headroom callback
        // =====================================================================
        // This eliminates the need for polling loops. The OS will invoke
        // onHeadroomChanged() on a system binder thread when:
        //   1. Thermal throttling events occur (skin temp crosses thresholds)
        //   2. Skin temperature threshold values change
        //
        // The callback provides both current headroom AND forecast headroom,
        // giving us predictive capability without explicit polling.
        int result = mRegisterListenerFn(
            mThermalMgr,
            &ThermalManager::onHeadroomChanged,
            this  // data pointer — recovered in the static trampoline
        );

        if (result == 0) {
            mCallbackRegistered = true;
            LOGI("✓ Android 16+ AThermal_HeadroomCallback registered");
            LOGI("  Async thermal monitoring active — polling disabled");
        } else {
            LOGW("AThermal_registerThermalHeadroomListener failed (err=%d)", result);
            LOGW("Falling back to legacy polling mode");
            mCallbackRegistered = false;
        }
    } else {
        LOGI("API 36 headroom listener not available — using legacy polling");
        mCallbackRegistered = false;
    }

    // Initial headroom reading to establish baseline state
    float initialHeadroom = mHeadroomFn(mThermalMgr, 10);
    if (!std::isnan(initialHeadroom)) {
        applyHeadroom(initialHeadroom, initialHeadroom, 10);
    }

    LOGI("✓ ADPF Thermal Manager initialized — mode: %s, headroom: %.2f (%s)",
         mCallbackRegistered ? "CALLBACK (API 36+)" : "POLLING (API 31+)",
         mHeadroom.load(std::memory_order_relaxed),
         stateName(getState()));
    LOGI("  Thresholds: warm=%.2f, throttle=%.2f, critical=%.2f",
         mThresholds.warm, mThresholds.throttle, mThresholds.critical);

    return true;
}

// =============================================================================
// Static Callback Trampoline — AThermal_HeadroomCallback (API 36+)
// =============================================================================
// Invoked by the OS on a system binder thread pool when thermal headroom
// or thresholds change significantly. This static function recovers the
// ThermalManager instance from the data pointer and updates atomic state.
//
// Callback parameters:
//   data             — Our ThermalManager instance pointer
//   headroom         — Current thermal headroom (non-negative, normalized)
//   forecastHeadroom — Predicted headroom in forecastSeconds
//   forecastSeconds  — System-chosen lookahead window (device-dependent)
//   thresholds       — AThermalHeadroomThreshold array (may be null)
//   thresholdsCount  — Number of threshold entries

void ThermalManager::onHeadroomChanged(
    void* data,
    float headroom,
    float forecastHeadroom,
    int forecastSeconds,
    const AThermalHeadroomThreshold* thresholds,
    size_t thresholdsCount
) {
    if (!data) return;

    auto* self = static_cast<ThermalManager*>(data);

    // Publish headroom + state + forecast atomically
    self->applyHeadroom(headroom, forecastHeadroom, forecastSeconds);

    // If the OS provides device-specific thresholds, update our policy
    // thresholds to match the hardware. This ensures our classification
    // aligns with what the OS considers LIGHT, MODERATE, SEVERE.
    if (thresholds && thresholdsCount > 0) {
        self->updateThresholdsFromOEM(thresholds, thresholdsCount);
    }

    LOGI("⚡ Thermal callback: headroom=%.2f, forecast=%.2f (%ds), state=%s",
         headroom, forecastHeadroom, forecastSeconds,
         stateName(self->getState()));
}

// =============================================================================
// OEM Threshold Update
// =============================================================================
// Maps the AThermalHeadroomThreshold array from the OS to our internal
// policy thresholds. The OS provides thresholds keyed by AThermalStatus.
//
// Mapping:
//   ATHERMAL_STATUS_LIGHT (1)    → warm threshold
//   ATHERMAL_STATUS_MODERATE (2) → throttle threshold
//   ATHERMAL_STATUS_SEVERE (3)   → critical threshold

void ThermalManager::updateThresholdsFromOEM(
    const AThermalHeadroomThreshold* thresholds,
    size_t count
) {
    bool updated = false;

    for (size_t i = 0; i < count; ++i) {
        float value = thresholds[i].headroom;
        if (value <= 0.0f || std::isnan(value)) continue;

        switch (thresholds[i].thermalStatus) {
            case ATHERMAL_STATUS_LIGHT:
                if (mThresholds.warm != value) {
                    LOGI("  OEM threshold: LIGHT → warm=%.2f (was %.2f)",
                         value, mThresholds.warm);
                    mThresholds.warm = value;
                    updated = true;
                }
                break;

            case ATHERMAL_STATUS_MODERATE:
                if (mThresholds.throttle != value) {
                    LOGI("  OEM threshold: MODERATE → throttle=%.2f (was %.2f)",
                         value, mThresholds.throttle);
                    mThresholds.throttle = value;
                    updated = true;
                }
                break;

            case ATHERMAL_STATUS_SEVERE:
                if (mThresholds.critical != value) {
                    LOGI("  OEM threshold: SEVERE → critical=%.2f (was %.2f)",
                         value, mThresholds.critical);
                    mThresholds.critical = value;
                    updated = true;
                }
                break;

            default:
                break;
        }
    }

    if (updated) {
        // Re-classify with new thresholds
        float currentHeadroom = mHeadroom.load(std::memory_order_relaxed);
        ThermalState newState = classifyHeadroom(currentHeadroom);
        mThermalState.store(static_cast<int32_t>(newState),
                            std::memory_order_release);
        LOGI("  Thresholds updated → re-classified to %s",
             stateName(newState));
    }
}

// =============================================================================
// Thermal Headroom Polling (Legacy Mode — API 31–35)
// =============================================================================
// In callback mode (API 36+), this is a no-op that returns the most recent
// callback-provided headroom. The NoiseSuppressor can still call this
// method without knowing which mode is active.

float ThermalManager::pollHeadroom(int32_t forecastSeconds) {
    if (!mAvailable || !mThermalMgr) {
        return -1.0f;
    }

    // =========================================================================
    // API 36+ (callback mode): No polling needed — return cached value
    // =========================================================================
    // The AThermal_HeadroomCallback already pushed the latest headroom to
    // our atomics. Polling is unnecessary and would waste CPU cycles.
    if (mCallbackRegistered) {
        return mHeadroom.load(std::memory_order_relaxed);
    }

    // =========================================================================
    // API 31–35 (polling mode): Query thermal subsystem with rate limiting
    // =========================================================================
    if (!mHeadroomFn) {
        return -1.0f;
    }

    // Enforce minimum 1-second polling interval (ADPF spec requirement)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    int64_t nowNs = ts.tv_sec * 1'000'000'000LL + ts.tv_nsec;

    if (nowNs - mLastPollTimeNs < kMinPollIntervalNs) {
        // Return cached value — too soon to poll again
        return mHeadroom.load(std::memory_order_relaxed);
    }
    mLastPollTimeNs = nowNs;

    // Query the thermal subsystem
    float headroom = mHeadroomFn(mThermalMgr, forecastSeconds);

    // AThermal_getThermalHeadroom returns NaN on error
    if (std::isnan(headroom)) {
        LOGW("AThermal_getThermalHeadroom returned NaN — sensor error?");
        return -1.0f;
    }

    // In polling mode, forecast = current (no separate forecast available)
    applyHeadroom(headroom, headroom, forecastSeconds);

    return headroom;
}

// =============================================================================
// Shared State Publication — Used by both callback and polling paths
// =============================================================================
// Publishes headroom values atomically and logs state transitions.

void ThermalManager::applyHeadroom(float headroom, float forecastHeadroom,
                                    int forecastSeconds) {
    ThermalState previousState = getState();
    ThermalState newState = classifyHeadroom(headroom);

    // Publish all values atomically for lock-free audio thread consumption
    mHeadroom.store(headroom, std::memory_order_relaxed);
    mForecastHeadroom.store(forecastHeadroom, std::memory_order_relaxed);
    mForecastSeconds.store(forecastSeconds, std::memory_order_relaxed);
    mThermalState.store(static_cast<int32_t>(newState),
                        std::memory_order_release);

    // Log state transitions with diagnostic context
    if (newState != previousState) {
        const char* mode = mCallbackRegistered ? "callback" : "poll";

        LOGI("Thermal state transition [%s]: %s → %s "
             "(headroom=%.2f, forecast=%.2f@%ds)",
             mode, stateName(previousState), stateName(newState),
             headroom, forecastHeadroom, forecastSeconds);

        if (newState == ThermalState::kThrottling) {
            LOGW("⚠ Approaching thermal limit — switching to CPU inference");
        } else if (newState == ThermalState::kCritical) {
            LOGE("🔥 CRITICAL thermal state — emergency spectral gate fallback!");
        } else if (newState == ThermalState::kNominal &&
                   previousState >= ThermalState::kThrottling) {
            LOGI("✓ Device cooled — returning to NPU inference");
        }
    }
}

// =============================================================================
// Headroom → State Classification
// =============================================================================
// In callback mode, the policy thresholds may have been updated by OEM
// values from the AThermalHeadroomThreshold array, making classification
// more accurate than generic hardcoded values.

ThermalState ThermalManager::classifyHeadroom(float headroom) const {
    if (headroom >= mThresholds.critical) return ThermalState::kCritical;
    if (headroom >= mThresholds.throttle) return ThermalState::kThrottling;
    if (headroom >= mThresholds.warm)     return ThermalState::kWarm;
    return ThermalState::kNominal;
}

} // namespace less
