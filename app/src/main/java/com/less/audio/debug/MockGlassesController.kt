package com.less.audio.debug

import android.content.Context
import android.util.Log
// import com.meta.wearable.dat.mockdevice.MockDeviceKit

/**
 * MockGlassesController — Debug utility for simulating Meta Ray-Ban states.
 *
 * Uses the MWDAT MockDeviceKit (mwdat-mockdevice) to simulate physical glasses
 * behavior during development WITHOUT requiring real hardware.
 *
 * Simulated states:
 *   don()    → Glasses on head    → SessionState.RUNNING
 *   doff()   → Glasses removed    → SessionState.PAUSED
 *   fold()   → Glasses folded     → SessionState.STOPPED
 *   unfold() → Glasses unfolded   → (still PAUSED until don())
 *
 * Usage from debug UI:
 *   val mockCtrl = MockGlassesController(context)
 *   mockCtrl.initialize()
 *   // Bind buttons:
 *   button.setOnClickListener { mockCtrl.simulateDon() }
 *
 * THIS CLASS IS DEBUG-ONLY. It should NOT be included in release builds.
 * Guard with BuildConfig.DEBUG checks in the calling code.
 */
class MockGlassesController(private val context: Context) {

    companion object {
        private const val TAG = "LESS_Mock"
    }

    private var mockKit: Any? = null
    private var mockDevice: Any? = null  // MockDevice type from SDK

    // Track whether mock device is initialized
    var isInitialized: Boolean = false
        private set

    /**
     * Initialize the MockDeviceKit and pair a simulated Ray-Ban Meta device.
     *
     * This creates a virtual BLE device that the MWDAT SDK treats as real
     * glasses. Registration and session state flows work identically.
     */
    fun initialize(): Boolean {
        return try {
            val mockKitClass = Class.forName("com.meta.wearable.dat.mockdevice.MockDeviceKit")
            val getInstanceMethod = mockKitClass.getMethod("getInstance", Context::class.java)
            mockKit = getInstanceMethod.invoke(null, context)
            val pairMethod = mockKit!!.javaClass.getMethod("pairRaybanMeta")
            mockDevice = pairMethod.invoke(mockKit)

            if (mockDevice != null) {
                isInitialized = true
                Log.i(TAG, "✓ Mock Ray-Ban Meta paired — ready for state simulation")

                // Power on and unfold to initial state
                simulatePowerOn()
                simulateUnfold()

                true
            } else {
                Log.e(TAG, "Failed to create mock Ray-Ban Meta device")
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "MockDeviceKit initialization failed: ${e.message}")
            false
        }
    }

    /**
     * Simulate: User puts on the glasses → SessionState.RUNNING
     * The audio engine should RESUME processing.
     */
    fun simulateDon() {
        if (!isInitialized) {
            Log.w(TAG, "Mock device not initialized — call initialize() first")
            return
        }
        try {
            // Use reflection to call don() since the exact type varies by SDK version
            val device = mockDevice ?: return
            val donMethod = device.javaClass.getMethod("don")
            donMethod.invoke(device)
            Log.i(TAG, "🕶️ Simulated: DON (glasses on head)")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to simulate don(): ${e.message}")
        }
    }

    /**
     * Simulate: User takes off the glasses → SessionState.PAUSED
     * The audio engine should PAUSE immediately to prevent buffer underruns.
     */
    fun simulateDoff() {
        if (!isInitialized) return
        try {
            val device = mockDevice ?: return
            val doffMethod = device.javaClass.getMethod("doff")
            doffMethod.invoke(device)
            Log.i(TAG, "🕶️ Simulated: DOFF (glasses removed)")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to simulate doff(): ${e.message}")
        }
    }

    /**
     * Simulate: Glasses folded → SessionState.STOPPED
     * The audio engine should STOP to prevent crashes.
     */
    fun simulateFold() {
        if (!isInitialized) return
        try {
            val device = mockDevice ?: return
            val foldMethod = device.javaClass.getMethod("fold")
            foldMethod.invoke(device)
            Log.i(TAG, "🕶️ Simulated: FOLD (glasses folded)")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to simulate fold(): ${e.message}")
        }
    }

    /**
     * Simulate: Glasses unfolded (but not yet on head)
     */
    fun simulateUnfold() {
        if (!isInitialized) return
        try {
            val device = mockDevice ?: return
            val unfoldMethod = device.javaClass.getMethod("unfold")
            unfoldMethod.invoke(device)
            Log.i(TAG, "🕶️ Simulated: UNFOLD")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to simulate unfold(): ${e.message}")
        }
    }

    /**
     * Simulate: Power on the glasses
     */
    private fun simulatePowerOn() {
        try {
            val device = mockDevice ?: return
            val powerOnMethod = device.javaClass.getMethod("powerOn")
            powerOnMethod.invoke(device)
            Log.i(TAG, "🕶️ Simulated: POWER ON")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to simulate powerOn(): ${e.message}")
        }
    }

    /**
     * Reset mock state — call in test teardown.
     */
    fun reset() {
        try {
            if (mockKit != null) {
                val resetMethod = mockKit!!.javaClass.getMethod("reset")
                resetMethod.invoke(mockKit!!)
            }
            Log.i(TAG, "Mock state reset")
        } catch (e: Exception) {
            Log.e(TAG, "Mock reset failed: ${e.message}")
        }
    }

    fun destroy() {
        reset()
        mockDevice = null
        mockKit = null
        isInitialized = false
    }
}
