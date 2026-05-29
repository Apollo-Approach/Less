package com.less.audio.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * VisionCameraClient
 *
 * Interfaces with the Meta Wearables SDK (MWDAT Camera) to pull live video feed
 * from the Ray-Ban Meta glasses. It resamples, crops, and formats the incoming 
 * camera frames strictly to the [1, 320, 320, 3] INT8 buffer required by YOLO11n.
 *
 * If the glasses are not connected, runs an internal mock loop (or falls back to 
 * the phone's camera) to maintain the Vision -> Music engine.
 */
class VisionCameraClient(private val context: Context) {

    companion object {
        private const val TAG = "LESS_CameraClient"
        const val MODEL_INPUT_SIZE = 320
        const val CHANNELS = 3 // RGB
    }

    private val _frameFlow = MutableSharedFlow<ByteBuffer>(extraBufferCapacity = 1)
    val frameFlow: SharedFlow<ByteBuffer> = _frameFlow

    private val _colorFlow = MutableSharedFlow<Int>(extraBufferCapacity = 1)
    val colorFlow: SharedFlow<Int> = _colorFlow

    private var isStreaming = false
    private val scope = CoroutineScope(Dispatchers.Default + Job())
    private var mockStreamJob: Job? = null

    // For MWDAT we normally have:
    // private var cameraSession: com.meta.wearable.dat.camera.CameraStreamSession? = null

    fun startStream() {
        if (isStreaming) return
        isStreaming = true
        Log.i(TAG, "Starting camera stream to feed Vision/Music synthesizer.")
        
        // TODO: MWDAT integration block.
        // val device = Wearables.getConnectedDevice()
        // cameraSession = CameraStreamSession.create(device)
        // cameraSession?.start(Resolution.R_320x320, fps = 5) { frame -> 
        //     val buffer = convertFrameToYoloInput(frame)
        //     _frameFlow.tryEmit(buffer)
        // }

        // Start fallback fallback stream if hardware isn't linked
        startFallbackStream()
    }

    fun stopStream() {
        if (!isStreaming) return
        isStreaming = false
        Log.i(TAG, "Stopping camera stream.")
        
        // cameraSession?.stop()
        // cameraSession = null

        mockStreamJob?.cancel()
    }

    /**
     * Resizes and formats an Android Bitmap into the [1, 320, 320, 3] INT8 ByteBuffer 
     * expected by the TFLite interpreter.
     */
    fun convertBitmapToYoloInput(bitmap: Bitmap): ByteBuffer {
        // Extract dominant color using Palette API before resizing
        androidx.palette.graphics.Palette.from(bitmap).generate { palette ->
            val dominantColor = palette?.dominantSwatch?.rgb 
                ?: palette?.vibrantSwatch?.rgb 
                ?: Color.DKGRAY
            _colorFlow.tryEmit(dominantColor)
        }

        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true)
        val byteBuffer = ByteBuffer.allocateDirect(1 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * CHANNELS)
        byteBuffer.order(ByteOrder.nativeOrder())
        
        val intValues = IntArray(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.width, 0, 0, scaledBitmap.width, scaledBitmap.height)
        
        var pixel = 0
        for (i in 0 until MODEL_INPUT_SIZE) {
            for (j in 0 until MODEL_INPUT_SIZE) {
                val value = intValues[pixel++]
                // Quantized INT8 representation (using raw byte values)
                byteBuffer.put(Color.red(value).toByte())
                byteBuffer.put(Color.green(value).toByte())
                byteBuffer.put(Color.blue(value).toByte())
            }
        }
        
        return byteBuffer
    }

    /**
     * Simulates the camera providing frames at ~5 FPS if no hardware is present.
     */
    private fun startFallbackStream() {
        mockStreamJob = scope.launch {
            Log.i(TAG, "Hardware camera not found/mocked. Emitting blank dynamic buffers and mock colors.")
            var hueMock = 0f
            while (isActive && isStreaming) {
                // Emit a dummy buffer of the correct size
                val dummyBuffer = ByteBuffer.allocateDirect(1 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * CHANNELS)
                _frameFlow.tryEmit(dummyBuffer)

                // Mock color shifting slowly from red to violet
                hueMock = (hueMock + 2f) % 360f
                val mockColor = Color.HSVToColor(floatArrayOf(hueMock, 0.8f, 0.8f))
                _colorFlow.tryEmit(mockColor)

                delay(200) // 5 FPS
            }
        }
    }
}
