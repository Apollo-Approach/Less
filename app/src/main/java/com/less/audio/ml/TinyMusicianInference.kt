package com.less.audio.ml

import android.content.Context
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

class TinyMusicianInference(private val context: Context) {

    companion object {
        private const val TAG = "LESS_TinyMusician"
        
        // Mode Enums map to MainActivity's tinyMusicianMode
        const val MODE_DISABLED = 0
        const val MODE_TWEAKER = 1
        const val MODE_COMPOSER = 2
    }

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    private var currentMode = MODE_DISABLED

    fun initialize() {
        Log.i(TAG, "Initializing TinyMusician ONNX Runtime...")
        try {
            val modelPath = getBaseModelPath()
            if (!modelPath.exists()) {
                Log.w(TAG, "ONNX model not found: ${modelPath.absolutePath}. Falling back to mock/heuristic engine.")
                return
            }
            ortEnv = OrtEnvironment.getEnvironment()
            // We use NNAPI session options to offload to the NPU/DSP as requested in Option B
            val sessionOptions = OrtSession.SessionOptions().apply {
                addNnapi() 
            }
            ortSession = ortEnv?.createSession(modelPath.absolutePath, sessionOptions)
            Log.i(TAG, "Successfully loaded TinyMusician ONNX model.")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ONNX Runtime: ${e.message}")
        }
    }

    fun setMode(mode: Int) {
        currentMode = mode
    }

    /**
     * Executes the neural model using the 4 scene variables.
     * Returns an array of size 20 (4 tweaker values + 16 composer notes)
     */
    fun process(density: Float, valence: Float, arousal: Float, timbre: Float): FloatArray {
        if (currentMode == MODE_DISABLED) {
            return FloatArray(20) { 0f }
        }

        val session = ortSession ?: return FloatArray(20) { index -> 
            if (index < 4) 0.1f else 60.0f + (index * 2) 
        }
        val env = ortEnv ?: return FloatArray(20) { 0f }

        return try {
            val inputArray = floatArrayOf(density, valence, arousal, timbre)
            // Shape is [1, 4] for batch_size 1
            val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputArray), longArrayOf(1, 4))
            
            val inputs = mapOf("input" to tensor)
            val result = session.run(inputs)
            
            val outputTensor = result.get(0).value as Array<FloatArray>
            val outputData = outputTensor[0] // Get batch 0
            
            // Clean up
            result.close()
            tensor.close()
            
            outputData
        } catch (e: Exception) {
            Log.e(TAG, "ONNX Inference failed: ${e.message}")
            FloatArray(20) { 0f }
        }
    }

    fun destroy() {
        ortSession?.close()
        ortEnv?.close()
    }

    private fun getBaseModelPath(): File {
        return File(context.filesDir, "tinymusician_base.onnx")
    }
}
