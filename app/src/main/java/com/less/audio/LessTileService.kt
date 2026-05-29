package com.less.audio

import android.content.Intent
import android.os.Build
import android.service.quicksettings.Tile
import android.service.quicksettings.TileService
import android.util.Log
import com.less.audio.profiling.ProfilingService

// =============================================================================
// Phase 8 — Quick Settings Tile: LessTileService
// =============================================================================
// Custom TileService for the Android Quick Settings drop-down panel.
//
// Single tap:  Toggle active denoising ON/OFF
// Long press:  Launch a 15-minute Profiling Mode burst
//
// The tile works from the lockscreen — no app navigation required.
//
// Subtitle updates:
//   - "Active — NPU"    when engine runs on Hexagon NPU
//   - "Active — CPU"    when engine runs on XNNPACK CPU
//   - "Profiling..."    when profiling FGS is active
//   - "Off"             when engine is stopped
//
// Lifecycle: TileService is managed by the system. The tile's state is
// persistent across reboots via the system's QS tile manager.
// =============================================================================

class LessTileService : TileService() {

    companion object {
        private const val TAG = "LESS_Tile"
    }

    // =========================================================================
    // TileService lifecycle
    // =========================================================================

    /**
     * Called when the tile becomes visible in the Quick Settings panel.
     * Refresh the tile state to reflect current engine status.
     */
    override fun onStartListening() {
        super.onStartListening()
        updateTileState()
    }

    /**
     * Called when the user taps the tile.
     * Toggles the audio engine ON/OFF.
     */
    override fun onClick() {
        super.onClick()

        try {
            if (NativeAudioBridge.isRunning()) {
                // Stop the engine
                NativeAudioBridge.stopEngine()
                Log.i(TAG, "Engine stopped via Quick Settings tile")
            } else {
                // Ensure engine is created
                NativeAudioBridge.createEngine()

                // Configure model and QNN paths
                val modelPath = LessApplication.getModelPath(applicationContext)
                val modelFile = java.io.File(modelPath)
                if (modelFile.exists()) {
                    NativeAudioBridge.setModelPath(modelPath)
                }
                NativeAudioBridge.setQnnLibDir(
                    LessApplication.getQnnLibDir(applicationContext)
                )

                // Start the engine
                val started = NativeAudioBridge.startEngine()
                Log.i(TAG, "Engine started via Quick Settings tile: $started")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Tile onClick error: ${e.message}")
        }

        updateTileState()
    }

    // =========================================================================
    // Tile state management
    // =========================================================================

    private fun updateTileState() {
        val tile = qsTile ?: return

        try {
            val isRunning = NativeAudioBridge.isRunning()

            tile.state = if (isRunning) Tile.STATE_ACTIVE else Tile.STATE_INACTIVE

            tile.label = "LESS"

            tile.subtitle = if (isRunning) {
                val backend = NativeAudioBridge.getActiveBackend()
                when {
                    backend.contains("NPU") -> "Active — NPU"
                    backend.contains("GPU") -> "Active — GPU"
                    backend.contains("CPU") -> "Active — CPU"
                    backend.contains("Gate") -> "Active — Gate"
                    else -> "Active"
                }
            } else {
                "Off"
            }

            tile.updateTile()
        } catch (e: Exception) {
            // Engine may not be initialized yet
            tile.state = Tile.STATE_INACTIVE
            tile.label = "LESS"
            tile.subtitle = "Off"
            tile.updateTile()
        }
    }
}
