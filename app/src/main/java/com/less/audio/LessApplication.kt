package com.less.audio

import android.app.Application
import android.util.Log
import com.meta.wearable.dat.core.Wearables
import com.less.audio.telemetry.TelemetryManager
import java.io.File
import java.io.FileOutputStream

class LessApplication : Application() {

    companion object {
        private const val TAG = "LESS_App"
        private const val MODEL_ASSET_NAME = "base_denoiser.tflite"

        fun getModelPath(context: android.content.Context): String {
            return File(context.filesDir, "models/base_denoiser.tflite").absolutePath
        }

        fun getQnnLibDir(context: android.content.Context): String {
            return context.applicationInfo.nativeLibraryDir
        }
    }

    override fun onCreate() {
        super.onCreate()
        extractModelAsset()
        TelemetryManager.initialize(this)
        
        try {
            Wearables.initialize(this)
            Log.i(TAG, "Meta Wearables SDK initialized")
        } catch (e: Exception) {
            Log.e(TAG, "Meta Wearables SDK initialization failed: \$e")
        }
    }

    private fun extractModelAsset() {
        val assetsToCopy = listOf(
            "models/base_denoiser.tflite" to "base_denoiser.tflite", // destDir = models
            "tinymusician_base.onnx" to "tinymusician_base.onnx", // destDir = filesDir
            "training_model.onnx" to "training_model.onnx",
            "eval_model.onnx" to "eval_model.onnx",
            "optimizer_model.onnx" to "optimizer_model.onnx",
            "checkpoint" to "checkpoint"
        )

        assetsToCopy.forEach { (assetName, fileName) ->
            try {
                // If it's the base denoiser, put it in models/
                val destFile = if (assetName.startsWith("models/")) {
                    val destDir = File(filesDir, "models")
                    destDir.mkdirs()
                    File(destDir, fileName)
                } else {
                    File(filesDir, fileName)
                }
                
                // For subdirectories we must check the actual path in assets, but we know base_denoiser is at root of assets?
                // Wait, previously MODEL_ASSET_NAME was "base_denoiser.tflite".
                val inputAssetName = if (assetName == "models/base_denoiser.tflite") "base_denoiser.tflite" else assetName

                val assetFd = try {
                    assets.openFd(inputAssetName)
                } catch (e: java.io.FileNotFoundException) {
                    return@forEach
                }
                val assetSize = assetFd.length
                assetFd.close()

                if (!destFile.exists() || destFile.length() != assetSize) {
                    assets.open(inputAssetName).use { input ->
                        FileOutputStream(destFile).use { output ->
                            input.copyTo(output, bufferSize = 8192)
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to extract asset: $assetName", e)
            }
        }
    }
}
