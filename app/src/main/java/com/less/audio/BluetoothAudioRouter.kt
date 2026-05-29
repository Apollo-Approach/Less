package com.less.audio

import android.Manifest
import android.annotation.SuppressLint
import android.bluetooth.*
import android.bluetooth.le.BluetoothLeScanner
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.os.Build
import android.util.Log
import com.meta.wearable.dat.core.Wearables
import com.meta.wearable.dat.core.session.SessionState
import com.meta.wearable.dat.core.types.DeviceIdentifier
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.collectLatest

/**
 * BluetoothAudioRouter — BLE Audio device management for Meta Ray-Ban Gen 2.
 *
 * Phase 3 Upgrade: Meta Wearables SDK (MWDAT) Integration
 *
 * Responsibilities:
 *   1. Scan for and identify Meta Ray-Ban smart glasses via BLE
 *   2. Establish LE Audio profile connection
 *   3. Route system audio via setCommunicationDevice() — explicit device targeting
 *   4. **NEW**: Observe Wearables SessionState to control native engine lifecycle
 *   5. Monitor connection lifecycle (connect/disconnect)
 *
 * DSP Safety Architecture:
 *   The glasses physically interrupt data streams when taken off or folded.
 *   SessionState.PAUSED (doffed) and SessionState.STOPPED (folded) must
 *   immediately pause the Oboe streams to prevent buffer underruns and crashes.
 *
 *   SessionState flow is device-driven and asynchronous — we cannot predict
 *   transitions, only react. The reaction path is:
 *     Glasses doffed/folded → MWDAT SessionState change →
 *     BluetoothAudioRouter observer → JNI nativePauseEngine() →
 *     C++ LessAudioEngine::pause() → Oboe streams requestPause()
 */
class BluetoothAudioRouter(private val context: Context) {

    companion object {
        private const val TAG = "LESS_BT"

        // Meta Ray-Ban device name patterns
        private val META_DEVICE_NAMES = listOf(
            "Ray-Ban",
            "Meta Ray-Ban",
            "Ray-Ban | Meta",
            "Ray-Ban Stories",
            "RB Meta",           // Actual bonded device name pattern (e.g. "RB Meta 00N6")
            "Oakley Meta"
        )
    }

    // State
    private val audioManager: AudioManager =
        context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private val bluetoothManager: BluetoothManager =
        context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
    private val bluetoothAdapter: BluetoothAdapter? = bluetoothManager.adapter

    private var connectedDevice: BluetoothDevice? = null
    private var communicationDevice: AudioDeviceInfo? = null
    private var isScanning = false

    // Session state observation
    private var sessionObserverJob: Job? = null
    private val coroutineScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    // Current session state — observable by the dashboard
    var currentSessionState: SessionState = SessionState.STOPPED
        private set

    // Registered device ID from MWDAT (set after registration completes)
    var registeredDeviceId: String? = null
        private set

    // Callbacks
    var onDeviceConnected: ((BluetoothDevice) -> Unit)? = null
    var onDeviceDisconnected: (() -> Unit)? = null
    var onScanResult: ((BluetoothDevice) -> Unit)? = null
    var onSessionStateChanged: ((SessionState) -> Unit)? = null

    // =========================================================================
    // BLE Scanning
    // =========================================================================

    @SuppressLint("MissingPermission")
    fun startScan() {
        if (isScanning) {
            Log.w(TAG, "Already scanning — ignoring")
            return
        }

        val scanner: BluetoothLeScanner = bluetoothAdapter?.bluetoothLeScanner ?: run {
            Log.e(TAG, "BLE Scanner not available — is Bluetooth enabled?")
            return
        }

        val settings = ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
            .build()

        scanner.startScan(null, settings, scanCallback)
        isScanning = true
        Log.i(TAG, "BLE scan started — looking for Meta Ray-Ban devices")
    }

    @SuppressLint("MissingPermission")
    fun stopScan() {
        if (!isScanning) return
        val scanner = bluetoothAdapter?.bluetoothLeScanner ?: return
        scanner.stopScan(scanCallback)
        isScanning = false
        Log.i(TAG, "BLE scan stopped")
    }

    private val scanCallback = object : ScanCallback() {
        @SuppressLint("MissingPermission")
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            val device = result.device
            val name = device.name ?: return

            val isMetaRayBan = META_DEVICE_NAMES.any { pattern ->
                name.contains(pattern, ignoreCase = true)
            }

            if (isMetaRayBan) {
                Log.i(TAG, "Found Meta Ray-Ban device: $name (${device.address})")
                onScanResult?.invoke(device)
            }
        }

        override fun onScanFailed(errorCode: Int) {
            Log.e(TAG, "BLE scan failed with error: $errorCode")
            isScanning = false
        }
    }

    // =========================================================================
    // Explicit Audio Routing — setCommunicationDevice()
    // =========================================================================
    // Even though we bypass the SDK's restricted audio APIs, we MUST explicitly
    // tell the Android OS to route audio to the glasses. This ensures the Oboe
    // C++ engine intercepts the correct LE Audio stream.

    @SuppressLint("MissingPermission")
    fun routeAudioToDevice(device: BluetoothDevice): Boolean {
        connectedDevice = device
        Log.i(TAG, "Routing audio to: ${device.name} (${device.address})")

        // Find the corresponding AudioDeviceInfo for this BT device
        val audioDevice = findAudioDevice(device) ?: run {
            Log.e(TAG, "Could not find AudioDeviceInfo for ${device.name}")
            logAvailableAudioDevices()
            return false
        }

        communicationDevice = audioDevice

        // =====================================================================
        // setCommunicationDevice() — THE audio routing call
        // =====================================================================
        // This forces the Android AudioFlinger to route ALL communication audio
        // (input AND output) through this specific device. Combined with Oboe's
        // Exclusive/MMAP mode, this guarantees we're capturing from the glasses'
        // 5-mic beamforming array and outputting through their open-ear speakers.
        //
        // API 31+ (Android 12+) — we require minSdk 33, so this is safe.
        val success = audioManager.setCommunicationDevice(audioDevice)
        if (success) {
            Log.i(TAG, "✓ Audio routed to ${device.name} " +
                    "(type=${deviceTypeName(audioDevice.type)}, id=${audioDevice.id})")
        } else {
            Log.e(TAG, "✗ Failed to set communication device — OS rejected routing")
        }

        return success
    }

    /**
     * Finds the AudioDeviceInfo matching a BluetoothDevice.
     *
     * Priority order:
     *   1. TYPE_BLE_HEADSET  — LE Audio (preferred, lowest latency with LC3)
     *   2. TYPE_BLE_SPEAKER  — LE Audio alternative
     *   3. TYPE_BLUETOOTH_SCO — Legacy Bluetooth (fallback)
     */
    @SuppressLint("MissingPermission")
    private fun findAudioDevice(btDevice: BluetoothDevice): AudioDeviceInfo? {
        val allDevices = audioManager.getDevices(AudioManager.GET_DEVICES_ALL)

        // Try to match by both type AND name for highest confidence
        val candidates = allDevices.filter { audioDevice ->
            val isBluetoothType = audioDevice.type in listOf(
                AudioDeviceInfo.TYPE_BLE_HEADSET,
                AudioDeviceInfo.TYPE_BLE_SPEAKER,
                AudioDeviceInfo.TYPE_BLUETOOTH_SCO
            )
            isBluetoothType
        }

        // Prefer LE Audio device types over legacy SCO
        return candidates.sortedBy { audioDevice ->
            when (audioDevice.type) {
                AudioDeviceInfo.TYPE_BLE_HEADSET -> 0   // Best: LE Audio headset
                AudioDeviceInfo.TYPE_BLE_SPEAKER -> 1   // Good: LE Audio speaker
                AudioDeviceInfo.TYPE_BLUETOOTH_SCO -> 2 // Fallback: legacy SCO
                else -> 3
            }
        }.firstOrNull()
    }

    /**
     * Releases the communication device routing.
     */
    fun releaseAudioRoute() {
        audioManager.clearCommunicationDevice()
        communicationDevice = null
        Log.i(TAG, "Audio routing released — returning to default device")
    }

    // =========================================================================
    // Session Lifecycle Binding — CRITICAL DSP SAFETY
    // =========================================================================
    // The Meta Ray-Ban glasses physically interrupt data streams during:
    //   - Doff (taken off head) → SessionState.PAUSED
    //   - Fold (arms folded)    → SessionState.STOPPED
    //   - Don (put back on)     → SessionState.RUNNING
    //
    // If the Oboe audio callback continues to fire after a doff/fold, the
    // input stream will starve (no BLE data arriving), causing:
    //   1. Buffer underruns → audio glitches
    //   2. Accumulated read timeouts → thread starvation
    //   3. ANR or native crash
    //
    // Solution: observe SessionState and immediately pause/resume the native
    // engine via JNI when the glasses change physical state.

    /**
     * Start observing the Wearables session state for a registered device.
     * Call this AFTER registration completes and a deviceId is available.
     *
     * @param deviceId The MWDAT device ID from registration
     */
    fun startSessionObserver(deviceId: String) {
        registeredDeviceId = deviceId

        // Cancel any existing observer
        sessionObserverJob?.cancel()

        sessionObserverJob = coroutineScope.launch {
            Log.i(TAG, "Starting session state observer for device: $deviceId")

            try {
                Wearables.getDeviceSessionState(DeviceIdentifier(deviceId)).collectLatest { state ->
                    val previousState = currentSessionState
                    currentSessionState = state

                    Log.i(TAG, "Session state: $previousState → $state")

                    when (state) {
                        SessionState.RUNNING -> {
                            // Glasses are on the user's head and active
                            // Resume the native audio engine
                            Log.i(TAG, "✓ Glasses ACTIVE — resuming audio engine")
                            NativeAudioBridge.resumeEngine()
                        }

                        SessionState.PAUSED -> {
                            // Glasses doffed (taken off) — BLE stream interrupted
                            // IMMEDIATELY pause to prevent buffer underruns
                            Log.w(TAG, "⚠ Glasses DOFFED — pausing audio engine")
                            NativeAudioBridge.pauseEngine()
                        }

                        SessionState.STOPPED -> {
                            // Glasses folded or session ended — full stream halt
                            // Stop the engine entirely to prevent crashes
                            Log.w(TAG, "⚠ Glasses FOLDED/STOPPED — stopping audio engine")
                            NativeAudioBridge.pauseEngine()
                        }
                    }

                    onSessionStateChanged?.invoke(state)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Session state observation failed: ${e.message}")
            }
        }
    }

    /**
     * Stop observing session state (cleanup).
     */
    fun stopSessionObserver() {
        sessionObserverJob?.cancel()
        sessionObserverJob = null
        Log.i(TAG, "Session state observer stopped")
    }

    // =========================================================================
    // Connection Lifecycle Monitoring
    // =========================================================================

    private val connectionReceiver = object : BroadcastReceiver() {
        @SuppressLint("MissingPermission")
        override fun onReceive(context: Context, intent: Intent) {
            @Suppress("DEPRECATION")
            val device = intent.getParcelableExtra<BluetoothDevice>(
                BluetoothDevice.EXTRA_DEVICE
            ) ?: return

            when (intent.action) {
                BluetoothDevice.ACTION_ACL_CONNECTED -> {
                    val name = device.name ?: "Unknown"
                    val isMetaRayBan = META_DEVICE_NAMES.any {
                        name.contains(it, ignoreCase = true)
                    }
                    if (isMetaRayBan) {
                        Log.i(TAG, "Meta Ray-Ban connected: $name")
                        connectedDevice = device

                        // Automatically route audio to the newly connected glasses
                        routeAudioToDevice(device)
                        onDeviceConnected?.invoke(device)
                    }
                }
                BluetoothDevice.ACTION_ACL_DISCONNECTED -> {
                    if (device.address == connectedDevice?.address) {
                        Log.i(TAG, "Meta Ray-Ban disconnected")
                        releaseAudioRoute()
                        connectedDevice = null
                        onDeviceDisconnected?.invoke()
                    }
                }
            }
        }
    }

    fun registerConnectionListener() {
        val filter = IntentFilter().apply {
            addAction(BluetoothDevice.ACTION_ACL_CONNECTED)
            addAction(BluetoothDevice.ACTION_ACL_DISCONNECTED)
        }
        context.registerReceiver(connectionReceiver, filter)
        Log.i(TAG, "Connection listener registered")
    }

    fun unregisterConnectionListener() {
        try {
            context.unregisterReceiver(connectionReceiver)
        } catch (e: IllegalArgumentException) {
            // Already unregistered
        }
        Log.i(TAG, "Connection listener unregistered")
    }

    // =========================================================================
    // State queries
    // =========================================================================

    fun isDeviceConnected(): Boolean {
        // First check if we have an explicitly routed device
        if (connectedDevice != null) return true
        // Otherwise, try to auto-detect bonded Meta glasses
        return detectBondedGlasses()
    }
    fun getConnectedDevice(): BluetoothDevice? = connectedDevice
    fun getConnectedDeviceName(): String? =
        @SuppressLint("MissingPermission") connectedDevice?.name

    fun isSessionActive(): Boolean = currentSessionState == SessionState.RUNNING

    /**
     * Scans the phone's bonded BT device list for known Meta Ray-Ban name patterns.
     * If found, sets connectedDevice so the UI can display the connection.
     * Returns true if a bonded Meta device was found.
     */
    @SuppressLint("MissingPermission")
    fun detectBondedGlasses(): Boolean {
        if (connectedDevice != null) return true
        val adapter = bluetoothAdapter ?: return false
        try {
            val bonded = adapter.bondedDevices ?: return false
            for (device in bonded) {
                val name = device.name ?: continue
                val isMetaRayBan = META_DEVICE_NAMES.any { pattern ->
                    name.contains(pattern, ignoreCase = true)
                }
                if (isMetaRayBan) {
                    connectedDevice = device
                    Log.i(TAG, "✓ Auto-detected bonded Meta glasses: $name (${device.address})")
                    onDeviceConnected?.invoke(device)
                    return true
                }
            }
        } catch (e: SecurityException) {
            Log.w(TAG, "Bluetooth permission denied for bonded device check")
        }
        return false
    }

    // =========================================================================
    // Debug Utilities
    // =========================================================================

    private fun logAvailableAudioDevices() {
        Log.i(TAG, "Available audio devices:")
        audioManager.getDevices(AudioManager.GET_DEVICES_ALL).forEach { d ->
            Log.i(TAG, "  ${d.productName} type=${deviceTypeName(d.type)} " +
                    "isSource=${d.isSource} isSink=${d.isSink} id=${d.id}")
        }
    }

    private fun deviceTypeName(type: Int): String = when (type) {
        AudioDeviceInfo.TYPE_BLE_HEADSET -> "BLE_HEADSET"
        AudioDeviceInfo.TYPE_BLE_SPEAKER -> "BLE_SPEAKER"
        AudioDeviceInfo.TYPE_BLUETOOTH_SCO -> "BT_SCO"
        AudioDeviceInfo.TYPE_BLUETOOTH_A2DP -> "BT_A2DP"
        AudioDeviceInfo.TYPE_BUILTIN_SPEAKER -> "BUILTIN_SPEAKER"
        AudioDeviceInfo.TYPE_BUILTIN_MIC -> "BUILTIN_MIC"
        else -> "TYPE_$type"
    }

    /**
     * Clean up all resources.
     */
    fun destroy() {
        stopScan()
        stopSessionObserver()
        releaseAudioRoute()
        unregisterConnectionListener()
        coroutineScope.cancel()
        connectedDevice = null
    }
}
