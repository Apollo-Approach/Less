package com.less.audio.ml

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import android.net.Uri

class GemmaDownloader(private val context: Context) {
    companion object {
        private const val TAG = "LESS_GemmaDownloader"
        private const val MODEL_FILENAME = "gemma-2b-it-cpu-int4.bin"
        
        // Public Google Cloud Storage bucket containing MediaPipe-compatible LLMs
        private const val MODEL_URL = "https://storage.googleapis.com/mediapipe-models/llm/gemma-2b-it-cpu-int4/float32/latest/gemma-2b-it-cpu-int4.bin"
    }

    private val _downloadProgress = MutableStateFlow<Float?>(null)
    val downloadProgress: StateFlow<Float?> = _downloadProgress

    fun getModelFile(): File {
        return File(context.filesDir, MODEL_FILENAME)
    }

    fun isModelPresent(): Boolean {
        val file = getModelFile()
        return file.exists() && file.length() > 100_000_000L // Ensure it's not a tiny corrupted file
    }

    suspend fun copyModelFromUri(uri: Uri): Boolean = withContext(Dispatchers.IO) {
        try {
            Log.i(TAG, "Copying Gemma model from user-selected URI: $uri")
            val inputStream = context.contentResolver.openInputStream(uri) ?: return@withContext false
            val tempFile = File(context.filesDir, "$MODEL_FILENAME.tmp")
            
            _downloadProgress.value = 0.0f
            
            inputStream.use { input ->
                FileOutputStream(tempFile).use { output ->
                    val data = ByteArray(128 * 1024)
                    var count: Int
                    while (input.read(data).also { count = it } != -1) {
                        output.write(data, 0, count)
                    }
                }
            }
            
            val finalFile = getModelFile()
            if (finalFile.exists()) finalFile.delete()
            tempFile.renameTo(finalFile)
            
            _downloadProgress.value = null
            Log.i(TAG, "Successfully copied Gemma model to local sandbox.")
            return@withContext true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to copy Gemma model: ${e.message}")
            _downloadProgress.value = null
            return@withContext false
        }
    }

    suspend fun downloadModelIfNeeded(): Boolean = withContext(Dispatchers.IO) {
        if (isModelPresent()) {
            return@withContext true
        }

        try {
            Log.i(TAG, "Starting Gemma 2B download from $MODEL_URL")
            val url = URL(MODEL_URL)
            val connection = url.openConnection() as HttpURLConnection
            connection.connectTimeout = 15000
            connection.readTimeout = 60000
            connection.connect()

            if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                Log.e(TAG, "Server returned HTTP ${connection.responseCode} ${connection.responseMessage}")
                return@withContext false
            }

            val fileLength = connection.contentLength
            val tempFile = File(context.filesDir, "$MODEL_FILENAME.tmp")
            
            connection.inputStream.use { input ->
                FileOutputStream(tempFile).use { output ->
                    val data = ByteArray(64 * 1024)
                    var total: Long = 0
                    var count: Int
                    
                    while (input.read(data).also { count = it } != -1) {
                        total += count.toLong()
                        output.write(data, 0, count)
                        
                        if (fileLength > 0) {
                            _downloadProgress.value = total.toFloat() / fileLength.toFloat()
                        }
                    }
                }
            }

            // Rename temp to final
            tempFile.renameTo(getModelFile())
            _downloadProgress.value = null
            Log.i(TAG, "Gemma 2B download complete!")
            return@withContext true

        } catch (e: Exception) {
            Log.e(TAG, "Failed to download Gemma model: ${e.message}")
            _downloadProgress.value = null
            return@withContext false
        }
    }
}
