package com.less.audio.ml

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileWriter

/**
 * Appends user feedback events to a JSON Lines (JSONL) dataset file.
 * This file acts as the training corpus for tailoring generative music to the user's preferences over time.
 */
object FeedbackLogger {
    private const val TAG = "FeedbackLogger"
    private const val FILE_NAME = "vision_music_feedback.jsonl"

    /**
     * @param context Application context
     * @param yoloClasses List of YOLO class indices currently detected
     * @param interpretation Actual mapping selected (0=Ambient, 1=Arp, 2=Pulse)
     * @param feedbackScore The feedback (+1 for 👍, -1 for 👎)
     */
    suspend fun logFeedback(
        context: Context,
        yoloClasses: List<Int>,
        interpretation: Int,
        feedbackScore: Int
    ) {
        withContext(Dispatchers.IO) {
            try {
                val file = File(context.filesDir, FILE_NAME)
                
                val jsonRecord = JSONObject().apply {
                    put("timestamp", System.currentTimeMillis())
                    put("yolo_classes", JSONArray(yoloClasses))
                    put("interpretation", interpretation)
                    put("feedback", feedbackScore)
                }

                FileWriter(file, true).use { writer ->
                    writer.append(jsonRecord.toString())
                    writer.append('\n')
                }
                
                Log.d(TAG, "Logged feedback record: $jsonRecord")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to log feedback", e)
            }
        }
    }
}
