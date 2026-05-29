package com.less.audio.ml

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtTrainingSession
import java.nio.file.Paths

class PersonalizationWorker(
    appContext: Context,
    workerParams: WorkerParameters
) : CoroutineWorker(appContext, workerParams) {

    companion object {
        const val TAG = "LESS_Personalization"
    }

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        Log.i(TAG, "Starting PersonalizationWorker for TinyMusician...")
        
        val env = OrtEnvironment.getEnvironment()
        
        try {
            val trainingModel = File(applicationContext.filesDir, "training_model.onnx")
            val evalModel = File(applicationContext.filesDir, "eval_model.onnx")
            val optimModel = File(applicationContext.filesDir, "optimizer_model.onnx")
            val checkpoint = File(applicationContext.filesDir, "checkpoint")
            
            val feedbackFile = File(applicationContext.filesDir, "vision_music_feedback.jsonl")

            if (!trainingModel.exists() || !checkpoint.exists() || !feedbackFile.exists()) {
                Log.w(TAG, "Missing training artifacts or feedback data. Skipping training.")
                return@withContext Result.success()
            }

            Log.i(TAG, "Initializing ONNX Training Session...")
            val sessionOptions = OrtSession.SessionOptions()
            val trainingSession = env.createTrainingSession(
                checkpoint.absolutePath, 
                trainingModel.absolutePath, 
                evalModel.absolutePath, 
                optimModel.absolutePath,
                sessionOptions
            )
            
            Log.i(TAG, "Parsing vision_music_feedback.jsonl...")
            val records = feedbackFile.readLines().mapNotNull {
                try { org.json.JSONObject(it) } catch (e: Exception) { null }
            }

            val mapper = SceneHeuristicMapper()
            val tinyMusician = TinyMusicianInference(applicationContext)
            tinyMusician.initialize()
            tinyMusician.setMode(TinyMusicianInference.MODE_COMPOSER)

            for (record in records) {
                val yoloArray = record.getJSONArray("yolo_classes")
                val yoloClasses = mutableListOf<Int>()
                for (i in 0 until yoloArray.length()) yoloClasses.add(yoloArray.getInt(i))
                val feedback = record.getInt("feedback")

                // Reconstruct fake detections to map to musical parameters since we only have class IDs
                val detections = yoloClasses.map { classId -> 
                    Detection(
                        classId = classId, 
                        confidence = 1.0f, 
                        centerX = 0.5f, centerY = 0.5f, 
                        width = 0.2f, height = 0.2f
                    ) 
                }

                val params = mapper.mapToMusicalParameters(detections)
                val density = params[0]
                val valence = params[1]
                val arousal = params[2]
                val timbre = params[3]
                
                val inputArray = floatArrayOf(density, valence, arousal, timbre)
                
                // Get current prediction to formulate target
                val currentOutput = tinyMusician.process(density, valence, arousal, timbre)
                val targetOutput = FloatArray(20)
                for (i in 0 until 20) {
                    if (feedback > 0) {
                        targetOutput[i] = currentOutput[i] // reinforce
                    } else {
                        // nudge away by adding a small perturbation
                        targetOutput[i] = currentOutput[i] + ((Math.random().toFloat() - 0.5f) * 0.5f)
                    }
                }

                // Run training step
                val inputTensor = ai.onnxruntime.OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(inputArray), longArrayOf(1, 4))
                val targetTensor = ai.onnxruntime.OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(targetOutput), longArrayOf(1, 20))
                
                val inputs = mapOf("input" to inputTensor, "target" to targetTensor)
                
                try {
                    // trainStep, optimizerStep, lazyResetGrad are standard in OrtTrainingSession
                    // But OrtTrainingSession might return the loss or we might need to handle the outputs map
                    val outs = trainingSession.trainStep(inputs)
                    outs?.close()
                    
                    trainingSession.optimizerStep()
                    trainingSession.lazyResetGrad()
                } catch (e: Exception) {
                    Log.e(TAG, "Training step failed: ${e.message}")
                } finally {
                    inputTensor.close()
                    targetTensor.close()
                }
            }

            tinyMusician.destroy()
            
            // Export model back to tinymusician_base.onnx so next inference uses the updated model
            val exportPath = File(applicationContext.filesDir, "tinymusician_base.onnx").absolutePath
            trainingSession.exportModelForInference(Paths.get(exportPath), arrayOf("output"))
            
            // Close session
            trainingSession.close()
            sessionOptions.close()
            
            // Clear feedback file after successful training
            feedbackFile.delete()
            
            Log.i(TAG, "Successfully exported personalized model and cleared feedback history.")
            
            Result.success()
        } catch (e: Exception) {
            Log.e(TAG, "Training failed: ${e.message}", e)
            Result.retry()
        } finally {
            env.close()
        }
    }
}
