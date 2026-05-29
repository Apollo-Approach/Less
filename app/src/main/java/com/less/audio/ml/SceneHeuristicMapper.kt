package com.less.audio.ml

import android.util.Log
import kotlin.math.abs
import kotlin.math.min

data class Detection(
    val classId: Int,         // COCO class index (0-79)
    val confidence: Float,    // 0.0 - 1.0
    val centerX: Float,       // normalized 0.0 - 1.0
    val centerY: Float,
    val width: Float,
    val height: Float
)

class SceneHeuristicMapper {

    companion object {
        private const val TAG = "LESS_SceneMapper"
        
        // Rough groupings of COCO classes for valence mapping
        private val POSITIVE_CLASSES = setOf(
            0, // person
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, // animals (bird, cat, dog, etc.)
            24, 25, 26, 27, 28, // personal items (backpack, umbrella, etc.)
            32, 33, // sports balls
            46, 47, // banana, apple
            49, 50, 51, 52, 53, 54, 55, // foods
            73, // laptop
            77 // cell phone
        )
        
        private val NEGATIVE_CLASSES = setOf(
            1, 2, 3, 4, 5, 6, 7, 8, // vehicles (car, motorcycle, bus, train, truck, etc.)
            9, 10, 11, // traffic light, fire hydrant, stop sign
            31, // snowboard
            34, 35, 36, // baseball bat, baseball glove, skateboard
            38 // tennis racket
        )
    }

    // State for cross-frame comparisons (movement sensing for Arousal)
    private var lastDetections: List<Detection> = emptyList()

    /**
     * Maps a list of YOLO detections into musical parameters.
     * @return FloatArray of [density, valence, arousal, timbre] all in 0.0-1.0 range
     */
    fun mapToMusicalParameters(detections: List<Detection>): FloatArray {
        if (detections.isEmpty()) {
            // Decay towards neutral if nothing is seen
            lastDetections = emptyList()
            return floatArrayOf(0.1f, 0.5f, 0.1f, 0.5f)
        }

        // 1. Density: Based on number of distinct objects
        // Caps at 10 objects making it 1.0
        val count = detections.size.toFloat()
        val density = min(1.0f, count / 10.0f)

        // 2. Valence: Positivity/Negativity of the scene
        var posScore = 0.0f
        var negScore = 0.0f
        
        for (det in detections) {
            if (det.classId in POSITIVE_CLASSES) posScore += det.confidence
            else if (det.classId in NEGATIVE_CLASSES) negScore += det.confidence
        }
        
        val totalAffect = posScore + negScore
        val valence = if (totalAffect > 0) {
            val ratio = posScore / totalAffect
            // Map 0 -> 0.1, 1 -> 0.9 to avoid extreme bounds
            0.1f + (ratio * 0.8f)
        } else {
            0.5f // Neutral
        }

        // 3. Arousal: Amount of screen space filled + frame-to-frame movement
        var totalArea = 0.0f
        var movementScore = 0.0f
        
        for (det in detections) {
            totalArea += min(1.0f, det.width * det.height)
            
            // Try to find a matching object in the previous frame (naive logic)
            val match = lastDetections.find { it.classId == det.classId }
            if (match != null) {
                val dx = abs(det.centerX - match.centerX)
                val dy = abs(det.centerY - match.centerY)
                movementScore += (dx + dy)
            }
        }
        
        val areaArousal = min(1.0f, totalArea)
        val motionArousal = min(1.0f, movementScore * 2.0f) // amplify motion
        val arousal = (areaArousal * 0.4f) + (motionArousal * 0.6f)

        // 4. Timbre: Complexity and variation in sizes
        // Huge objects = low timbre/dark, lots of tiny objects = high timbre/bright
        var avgSize = 0.0f
        if (count > 0) {
            avgSize = totalArea / count
        }
        // Invert average size so large = 0, small = 1
        val timbre = 1.0f - min(1.0f, avgSize * 2.0f)

        // Save current frame for future comparisons
        lastDetections = ArrayList(detections)

        Log.d(TAG, "Mapped Params -> D:%.2f V:%.2f A:%.2f T:%.2f (objects: %d)".format(
            density, valence, arousal, timbre, detections.size
        ))

        return floatArrayOf(
            density.coerceIn(0f, 1f),
            valence.coerceIn(0f, 1f),
            arousal.coerceIn(0f, 1f),
            timbre.coerceIn(0f, 1f)
        )
    }
}
