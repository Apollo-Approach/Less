package com.less.audio.ml

import android.content.Context
import android.util.Log
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.max
import kotlin.math.min

/**
 * Wraps the MediaPipe GenAI LlmInference API to drive the Vision Synth.
 */
class GemmaConductor(private val context: Context, private val modelFile: File) {

    companion object {
        private const val TAG = "LESS_GemmaConductor"
        
        // System prompt instructing the LLM on how to behave.
        private const val SYSTEM_PROMPT = """You are a Neural Conductor embedded in a sensory app. 
You will be provided with real-time visual scene context (Density, Valence, Arousal, Timbre). 
Your task is to orchestrate a generative synthesizer to match this vibe.

Think for a moment about the musical translation, then provide exactly 4 numerical float values between 0.0 and 1.0 on a new line strictly formatted as:
[ACTION] DENSITY=0.x VALENCE=0.x AROUSAL=0.x TIMBRE=0.x

Rules:
- 0.0 is low/dark/calm/sparse, 1.0 is high/bright/intense/dense.
- If valence is high, output high valence.
- Output ONLY a brief thought followed by the [ACTION] line."""
    }

    private var llmInference: LlmInference? = null
    var isInitialized = false
        private set

    suspend fun initialize() = withContext(Dispatchers.IO) {
        try {
            Log.i(TAG, "Initializing Gemma LLM Inference...")
            val options = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelFile.absolutePath)
                .setMaxTokens(1024)
                .setTemperature(0.7f)
                .setTopK(40)
                .build()

            llmInference = LlmInference.createFromOptions(context, options)
            isInitialized = true
            Log.i(TAG, "Gemma LLM Initialized successfully.")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Gemma: ${e.message}")
            isInitialized = false
        }
    }

    /**
     * Feed the scene values to the LLM and return the parsed new parameters + thought string.
     */
    suspend fun conduct(
        sceneDensity: Float,
        sceneValence: Float,
        sceneArousal: Float,
        sceneTimbre: Float
    ): ConductorResult? = withContext(Dispatchers.IO) {
        if (llmInference == null || !isInitialized) return@withContext null

        val userPrompt = "Scene: Density=${formatF(sceneDensity)}, Valence=${formatF(sceneValence)}, AROUSAL=${formatF(sceneArousal)}, TIMBRE=${formatF(sceneTimbre)}. What should the synthesizer play next?"
        
        // Gemma 2B formatting needs <start_of_turn> <end_of_turn> tags if we were doing true chat,
        // but LlmInference usually handles the internal templating if we feed it pure text, or we might need manual tags.
        // We will pass the system prompt merged into the user prompt as instructions.
        val combinedPrompt = "$SYSTEM_PROMPT\n\n$userPrompt"

        try {
            Log.v(TAG, "Generating response from Gemma...")
            val response = llmInference?.generateResponse(combinedPrompt) ?: return@withContext null
            Log.v(TAG, "Gemma Response: $response")
            return@withContext parseResponse(response)
        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}")
            return@withContext null
        }
    }

    private fun parseResponse(text: String): ConductorResult {
        // Look for the [ACTION] line and extract variables
        val actionLineRegex = Regex("""\[ACTION\].*?DENSITY=([\d.]+).*?VALENCE=([\d.]+).*?AROUSAL=([\d.]+).*?TIMBRE=([\d.]+)""")
        val match = actionLineRegex.find(text)
        
        if (match != null) {
            val d = match.groupValues[1].toFloatOrNull()?.coerceIn(0f, 1f) ?: 0.5f
            val v = match.groupValues[2].toFloatOrNull()?.coerceIn(0f, 1f) ?: 0.5f
            val a = match.groupValues[3].toFloatOrNull()?.coerceIn(0f, 1f) ?: 0.5f
            val t = match.groupValues[4].toFloatOrNull()?.coerceIn(0f, 1f) ?: 0.5f
            
            // Extract the thought (everything before the ACTION tag)
            var thought = text.substringBefore("[ACTION]").trim()
            
            // Clean up generic bot boilerplate
            if (thought.contains("Sure", ignoreCase = true) || thought.contains("Here is", ignoreCase = true)) {
                thought = ""
            }
            
            if (thought.isEmpty()) {
                val formattedD = String.format("%.2f", d)
                val formattedV = String.format("%.2f", v)
                thought = "Translating vibe to synthesis context... [D:$formattedD, V:$formattedV]"
            }
            
            return ConductorResult(thought, d, v, a, t)
        }
        
        // Fallback: If it just outputted thoughts, extract vaguely
        return ConductorResult(text.trim(), -1f, -1f, -1f, -1f)
    }

    fun destroy() {
        Log.i(TAG, "Closing Gemma LLM Inference.")
        llmInference?.close()
        llmInference = null
        isInitialized = false
    }

    private fun formatF(f: Float) = String.format("%.2f", f)

    data class ConductorResult(
        val thought: String,
        val density: Float,
        val valence: Float,
        val arousal: Float,
        val timbre: Float
    ) {
        val isValid get() = density >= 0f
    }
}
