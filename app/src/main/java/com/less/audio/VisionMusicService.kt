package com.less.audio

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlin.math.sin

object VisionSceneContext {
    private val _vibeSummary = MutableStateFlow("Scanning environment...")
    val vibeSummary = _vibeSummary.asStateFlow()

    private val _gemmaThoughts = MutableStateFlow<String?>(null)
    val gemmaThoughts = _gemmaThoughts.asStateFlow()

    private val _dominantColor = MutableStateFlow<Int?>(null)
    val dominantColor = _dominantColor.asStateFlow()

    fun updateVibe(density: Float, valence: Float, arousal: Float, timbre: Float) {
        val mood = if (valence > 0.65f) "Uplifting" else if (valence < 0.35f) "Melancholy" else "Neutral"
        val energy = if (arousal > 0.7f) "High Energy" else if (arousal < 0.4f) "Calm" else "Mid-Tempo"
        val densityDesc = if (density > 0.7f) "Dense orchestration" else "Minimalist arrangement"
        val color = if (timbre > 0.6f) "bright" else "warm"
        _vibeSummary.value = "$mood, $energy scene. $densityDesc with $color tones."
    }

    fun updateGemmaThoughts(thought: String) {
        _gemmaThoughts.value = thought
    }

    fun updateDominantColor(color: Int) {
        _dominantColor.value = color
    }
}

/**
 * VisionMusicService — Experimental Foreground Service for Vision-to-Music (Phase 15).
 * 
 * This service manages the connection to the Meta Ray-Ban camera stream via MWDAT,
 * runs the YOLO11n object detection model, and maps the resulting scene semantic
 * parameters into NativeAudioBridge to drive the C++ VisionSynth.
 *
 * Current state: MOCK INFERENCE
 * Rather than actually booting the camera and running the NPU inference (which
 * requires the TFLite model to be bundled), this generates coherent mock data
 * that simulates a changing visual soundscape. This allows tuning the synthesizer
 * audio mapping safely without thermal throttling or camera battery drain.
 */
class VisionMusicService : Service() {

    companion object {
        private const val TAG = "LESS_VisionMusic"
        private const val CHANNEL_ID = "less_vision_channel"
        private const val NOTIFICATION_ID = 2002

        const val ACTION_START = "com.less.audio.VISION_START"
        const val ACTION_STOP = "com.less.audio.VISION_STOP"
    }

    private var isRunning = false
    private val serviceScope = CoroutineScope(Dispatchers.Default + Job())
    private var mockInferenceJob: Job? = null

    private var mapper = com.less.audio.ml.SceneHeuristicMapper()
    private var tflite: org.tensorflow.lite.Interpreter? = null
    private var tinyMusician: com.less.audio.ml.TinyMusicianInference? = null
    
    // State machine timer for Phrase Transitions (Phase 19)
    private var phraseStartTime = 0L

    // State parameters for mock generation
    private var time = 0.0f
    
    // Gemma Conductor Layer (Phase 18)
    private lateinit var gemmaDownloader: com.less.audio.ml.GemmaDownloader
    private var gemmaConductor: com.less.audio.ml.GemmaConductor? = null
    private var gemmaConductorJob: Job? = null
    private var currentGemmaDensity = 0.5f
    private var currentGemmaValence = 0.5f
    private var currentGemmaArousal = 0.5f
    private var currentGemmaTimbre = 0.5f
    private var useGemmaParams = false

    // Camera Client
    private lateinit var cameraClient: com.less.audio.camera.VisionCameraClient

    override fun onCreate() {
        super.onCreate()
        cameraClient = com.less.audio.camera.VisionCameraClient(this)
        createNotificationChannel()
        Log.i(TAG, "VisionMusicService created.")
        
        tinyMusician = com.less.audio.ml.TinyMusicianInference(this)
        tinyMusician?.initialize()
        
        try {
            val modelFile = assets.open("yolo11n_int8.tflite").readBytes()
            val buffer = java.nio.ByteBuffer.allocateDirect(modelFile.size).apply {
                put(modelFile)
                rewind()
            }
            tflite = org.tensorflow.lite.Interpreter(buffer)
            Log.i(TAG, "Loaded yolo11n_int8.tflite successfully.")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load TFLite model, falling back to mock outputs: ${e.message}")
        }
        
        gemmaDownloader = com.less.audio.ml.GemmaDownloader(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> startVisionProcessing()
            ACTION_STOP -> stopVisionProcessing()
            else -> startVisionProcessing()
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        stopVisionProcessing()
        serviceScope.cancel()
        tflite?.close()
        tinyMusician?.destroy()
        gemmaConductor?.destroy()
        Log.i(TAG, "VisionMusicService destroyed.")
        super.onDestroy()
    }

    private fun startVisionProcessing() {
        if (isRunning) return

        // Build FGS notification
        val notification = buildNotification("Analyzing scene for music generation...")

        // MWDAT Camera normally requires FOREGROUND_SERVICE_TYPE_CAMERA
        startForeground(
            NOTIFICATION_ID,
            notification,
            ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA
        )
        
        isRunning = true
        Log.i(TAG, "Started VisionMusic Foreground Service")
        
        cameraClient.startStream()

        // Start the ML Inference Loop
        mockInferenceJob = serviceScope.launch {
            Log.i(TAG, "Starting YOLO11n scene inference ML loop...")
            
            phraseStartTime = System.currentTimeMillis()

            // State Machine Timer for New Song Transitions
            launch {
                while (isActive) {
                    delay(1000)
                    val elapsed = System.currentTimeMillis() - phraseStartTime
                    // 120,000ms = 2 minutes threshold
                    if (elapsed > 120_000L) {
                        Log.i(TAG, "Triggering Phase 19 Phrase Transition: New Song generated!")
                        NativeAudioBridge.flushGenerativeState() // Tells C++ to change key, rhythm and tracking
                        phraseStartTime = System.currentTimeMillis()
                        VisionSceneContext.updateGemmaThoughts("Forcing a drastic musical transition...")
                    }
                }
            }

            // Color Synesthesia Extraction Loop
            launch {
                cameraClient.colorFlow.collect { color ->
                    VisionSceneContext.updateDominantColor(color)
                    val hsv = FloatArray(3)
                    android.graphics.Color.colorToHSV(color, hsv)
                    NativeAudioBridge.updateSynesthesiaParams(hsv[0], hsv[1], hsv[2])
                }
            }

            // Collect camera frames when hardware is bound or fallback is running
            cameraClient.frameFlow.collect { frameBuffer ->
                if (tflite != null) {
                    processCameraFrame(frameBuffer)
                } else {
                    simulateYoloInferenceFallback()
                }
            }
        }
        
        // Start Gemma Conductor Loop asynchronously
        gemmaConductorJob = serviceScope.launch {
            VisionSceneContext.updateGemmaThoughts("Preparing Gemma Conductor...")
            val success = gemmaDownloader.downloadModelIfNeeded()
            if (success) {
                gemmaConductor = com.less.audio.ml.GemmaConductor(this@VisionMusicService, gemmaDownloader.getModelFile())
                gemmaConductor?.initialize()
                
                if (gemmaConductor?.isInitialized == true) {
                    VisionSceneContext.updateGemmaThoughts("Gemma ready. Observing scene...")
                    runGemmaConductorLoop()
                } else {
                    VisionSceneContext.updateGemmaThoughts("Gemma inference allocation failed.")
                }
            } else {
                VisionSceneContext.updateGemmaThoughts("Failed to fetch or find Gemma model.")
            }
        }
    }

    private suspend fun runGemmaConductorLoop() {
        // Slow polling loop to prevent thermal overload
        while (isRunning) {
            val d = currentGemmaDensity
            val v = currentGemmaValence
            val a = currentGemmaArousal
            val t = currentGemmaTimbre
            
            VisionSceneContext.updateGemmaThoughts("Thinking...")
            val result = gemmaConductor?.conduct(d, v, a, t)
            if (result != null && result.isValid) {
                VisionSceneContext.updateGemmaThoughts(result.thought)
                currentGemmaDensity = result.density
                currentGemmaValence = result.valence
                currentGemmaArousal = result.arousal
                currentGemmaTimbre = result.timbre
                useGemmaParams = true // Lock in JNI bridge to Gemma overrides
                
                Log.i(TAG, "Gemma Action -> D:${result.density} V:${result.valence} A:${result.arousal} T:${result.timbre}")
            }
            // Wait 10 seconds before generating new thoughts
            delay(10000)
        }
    }

    private fun stopVisionProcessing() {
        if (!isRunning) return
        isRunning = false
        cameraClient.stopStream()
        mockInferenceJob?.cancel()
        
        Log.i(TAG, "Stopping VisionMusic Foreground Service")
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun processCameraFrame(inputBuffer: java.nio.ByteBuffer) {
        try {
            // Check thermal state to potentially bypass inference
            val thermalState = NativeAudioBridge.getThermalState()
            if (thermalState >= 2) {
                // If Throttling (2) or Critical (3), skip TinyMusician inference to cool down
                Log.w(TAG, "Thermal state $thermalState - Bypassing TinyMusician inference")
                return
            }

            // Allocate output tensor [1, 84, 2100] for YOLO (assuming standard output)
            // Array approach:
            val outputBuffer = Array(1) { Array(84) { FloatArray(2100) } }

            // Run inference
            tflite?.run(inputBuffer, outputBuffer)
            
            // Generate some mock detections to feed the mapper for now since 
            // decoding standard YOLO11n output requires Non-Maximum Suppression (NMS).
            val mockDetections = buildMockDetections()
            
            // Map to synth parameters
            val params = mapper.mapToMusicalParameters(mockDetections)
            val (density, valence, arousal, timbre) = params

            // Offload YOLO -> ONNX sequence on Dispatchers.Default (which this already runs on)
            val mode = NativeAudioBridge.getTinyMusicianMode()
            if (mode > 0 && tinyMusician != null) {
                tinyMusician!!.setMode(mode)
                val neuralData = tinyMusician!!.process(density, valence, arousal, timbre)
                NativeAudioBridge.pushTinyMusicianData(neuralData)
            }

            // Push base params to JNI bridge
            val finalDensity = if (useGemmaParams) currentGemmaDensity else density
            val finalValence = if (useGemmaParams) currentGemmaValence else valence
            val finalArousal = if (useGemmaParams) currentGemmaArousal else arousal
            val finalTimbre = if (useGemmaParams) currentGemmaTimbre else timbre
            
            NativeAudioBridge.updateMusicalParameters(finalDensity, finalValence, finalArousal, finalTimbre)
            VisionSceneContext.updateVibe(finalDensity, finalValence, finalArousal, finalTimbre)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in ML inference loop: ${e.message}")
        }
    }
    
    private fun buildMockDetections(): List<com.less.audio.ml.Detection> {
        time += 0.05f
        val mockDetections = mutableListOf<com.less.audio.ml.Detection>()
        
        // Number of objects oscillating between 1 and 8
        val objectCount = (4 + 3 * sin(time * 0.4f)).toInt().coerceIn(1, 8)
        
        for (i in 0 until objectCount) {
            // Mix between positive (dogs=16) and negative (cars=2) classes
            val classId = if (sin(time * 0.15f + i) > 0) 16 else 2
            mockDetections.add(
                com.less.audio.ml.Detection(
                    classId = classId,
                    confidence = 0.8f,
                    centerX = 0.5f + 0.2f * sin(time * 0.8f + i).toFloat(),
                    centerY = 0.5f + 0.2f * kotlin.math.cos(time * 0.8f + i).toFloat(),
                    width = 0.2f + 0.1f * sin(time * 0.6f + i).toFloat(),
                    height = 0.2f + 0.1f * kotlin.math.cos(time * 0.6f + i).toFloat()
                )
            )
        }
        return mockDetections
    }

    /**
     * Fallback if TFLite model is not present, directly simulates previous behavior.
     */
    private fun simulateYoloInferenceFallback() {
        time += 0.05f // Move mock time forward

        // Density: maps to note density/richness. Breathes slowly between 0.2 and 0.8.
        val density = 0.5f + 0.3f * sin(time * 0.4f).toFloat()

        // Valence: maps to key/scale (mood). Oscillates very slowly between 0.1 and 0.9.
        val valence = 0.5f + 0.4f * sin(time * 0.15f).toFloat()

        // Arousal: maps to tempo and rhythm. Has sharper spikes simulating sudden movement.
        var arousal = 0.4f + 0.2f * sin(time * 0.8f).toFloat()
        // Simulate a sudden action/event (person walking into frame quickly)
        if (time % 10.0f < 1.0f) {
            arousal += 0.3f
        }
        arousal = arousal.coerceIn(0.0f, 1.0f)

        // Timbre: maps to frequency cutoff/waveform. Medium pace modulation.
        val timbre = 0.5f + 0.35f * sin(time * 0.6f + 1.2f).toFloat()

        // Push to JNI bridge
        val finalDensity = if (useGemmaParams) currentGemmaDensity else density
        val finalValence = if (useGemmaParams) currentGemmaValence else valence
        val finalArousal = if (useGemmaParams) currentGemmaArousal else arousal
        val finalTimbre = if (useGemmaParams) currentGemmaTimbre else timbre

        NativeAudioBridge.updateMusicalParameters(finalDensity, finalValence, finalArousal, finalTimbre)
        VisionSceneContext.updateVibe(finalDensity, finalValence, finalArousal, finalTimbre)
        
        // Log occasionally (prevent log spam, maybe once every 2 seconds = 10 frames)
        if ((time % 1.0f) < 0.05f) {
            Log.d(TAG, "Mock Fallback Update -> D:%.2f  V:%.2f  A:%.2f  T:%.2f".format(density, valence, arousal, timbre))
        }
    }

    private fun buildNotification(contentText: String): Notification {
        val builder = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("LESS Vision-to-Music (Mock)")
            .setContentText(contentText)
            // Note: Use the mipmap or drawable explicitly. E.g. android.R.drawable.ic_media_play 
            // Here just using an empty string as a placeholder until we link the proper drawable
            .setSmallIcon(android.R.drawable.ic_menu_camera) 
            .setOngoing(true)

        return builder.build()
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Vision-to-Music Inference",
            NotificationManager.IMPORTANCE_LOW 
        ).apply {
            description = "Maintains camera connection for real-time generative music"
        }

        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(channel)
    }
}
