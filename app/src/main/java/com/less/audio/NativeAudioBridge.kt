package com.less.audio

import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * NativeAudioBridge — JNI declarations for the LESS native audio engine.
 *
 * Singleton object that loads libless_engine.so and exposes all native
 * functions to the Kotlin layer. The engine handle (jlong) is managed
 * here to prevent lifecycle leaks.
 *
 * Threading: all functions are called from the main/UI thread EXCEPT
 * nativeStartTraining and nativeGetTrainingProgress, which are called
 * from WorkManager's coroutine dispatcher.
 *
 * Phase 3: Added pause/resume for session lifecycle binding.
 * These are called from BluetoothAudioRouter when the glasses are
 * doffed (PAUSED) or folded (STOPPED) to prevent DSP callback starvation.
 *
 * Phase 4: Added NPU configuration (QNN lib dir) and active backend query.
 */
object NativeAudioBridge {

    init {
        System.loadLibrary("less_engine")
    }

    // =========================================================================
    // Engine handle — stored as a Long (jlong on the native side)
    // =========================================================================
    private var engineHandle: Long = 0L

    fun createEngine(): Boolean {
        engineHandle = nativeCreateEngine()
        return engineHandle != 0L
    }

    fun setModelPath(path: String) {
        if (engineHandle == 0L) return
        nativeSetModelPath(engineHandle, path)
    }

    fun startEngine(): Boolean {
        if (engineHandle == 0L) return false
        return nativeStartEngine(engineHandle)
    }

    fun stopEngine() {
        if (engineHandle == 0L) return
        nativeStopEngine(engineHandle)
    }

    fun destroyEngine() {
        if (engineHandle == 0L) return
        nativeDestroyEngine(engineHandle)
        engineHandle = 0L
    }

    fun setSuppressionLevel(level: Float) {
        if (engineHandle == 0L) return
        nativeSetSuppressionLevel(engineHandle, level)
    }

    fun getLatencyMs(): Double {
        if (engineHandle == 0L) return -1.0
        return nativeGetLatencyMs(engineHandle)
    }

    fun isRunning(): Boolean {
        if (engineHandle == 0L) return false
        return nativeIsRunning(engineHandle)
    }

    fun reloadAdapter(adapterPath: String): Boolean {
        if (engineHandle == 0L) return false
        return nativeReloadAdapter(engineHandle, adapterPath)
    }

    // =========================================================================
    // Phase 9: Lock-free Stream Health Polling
    // =========================================================================

    /**
     * Polls the atomic stream error status exposed by the C++ engine.
     * @return 0 if healthy, non-zero oboe::Result if an error occurred.
     */
    fun getStreamErrorStatus(): Int {
        if (engineHandle == 0L) return -1
        return nativeGetStreamErrorStatus(engineHandle)
    }

    /**
     * Observes stream health via lock-free polling with exponential backoff on errors.
     * Emits non-zero error codes for the Kotlin layer to handle (e.g., triggering restarts).
     */
    fun observeStreamHealth(): Flow<Int> = flow {
        var backoffMs = 100L
        val maxBackoffMs = 5000L
        while (true) {
            val status = getStreamErrorStatus()
            if (status != 0) {
                emit(status)
                delay(backoffMs)
                backoffMs = (backoffMs * 2).coerceAtMost(maxBackoffMs)
            } else {
                backoffMs = 100L
                delay(100L) // 10Hz poll rate
            }
        }
    }

    // =========================================================================
    // Phase 3: Session lifecycle — pause/resume for doff/fold safety
    // =========================================================================
    // These are called from BluetoothAudioRouter's SessionState observer.
    // They must execute FAST — the glasses may stop sending BLE audio data
    // within milliseconds of a state change.

    /**
     * Pause the audio engine — stops Oboe streams without destroying them.
     * Called when glasses are doffed (PAUSED) or folded (STOPPED).
     * The engine can be resumed without full re-initialization.
     */
    fun pauseEngine() {
        if (engineHandle == 0L) return
        nativePauseEngine(engineHandle)
    }

    /**
     * Resume the audio engine — restarts Oboe streams from paused state.
     * Called when glasses are donned (RUNNING) after a doff/fold.
     */
    fun resumeEngine() {
        if (engineHandle == 0L) return
        nativeResumeEngine(engineHandle)
    }

    // =========================================================================
    // Phase 4: NPU Configuration
    // =========================================================================

    /**
     * Set the directory containing QNN shared libraries (libQnnHtp.so, etc.).
     * Must be called BEFORE startEngine() for NPU delegation to be active.
     * If not called or path is empty, falls back to XNNPACK CPU.
     */
    fun setQnnLibDir(path: String) {
        if (engineHandle == 0L) return
        nativeSetQnnLibDir(engineHandle, path)
    }

    /**
     * Returns the name of the currently active inference delegate backend.
     * Possible values: "QNN HTP (NPU)", "QNN GPU (Adreno)", "XNNPACK (CPU)",
     * "None (Spectral Gate)", "Not initialized".
     */
    fun getActiveBackend(): String {
        if (engineHandle == 0L) return "Not initialized"
        return nativeGetActiveBackend(engineHandle)
    }

    // =========================================================================
    // Phase 1: Real-time engine
    // =========================================================================
    private external fun nativeCreateEngine(): Long
    private external fun nativeSetModelPath(engineHandle: Long, modelPath: String)
    private external fun nativeStartEngine(engineHandle: Long): Boolean
    private external fun nativeStopEngine(engineHandle: Long)
    private external fun nativeDestroyEngine(engineHandle: Long)
    private external fun nativeSetSuppressionLevel(engineHandle: Long, level: Float)
    private external fun nativeGetLatencyMs(engineHandle: Long): Double
    private external fun nativeIsRunning(engineHandle: Long): Boolean
    private external fun nativeGetStreamErrorStatus(engineHandle: Long): Int
    private external fun nativeReloadAdapter(engineHandle: Long, adapterPath: String): Boolean

    // Phase 3: Session lifecycle
    private external fun nativePauseEngine(engineHandle: Long)
    private external fun nativeResumeEngine(engineHandle: Long)

    // Phase 4: NPU configuration
    private external fun nativeSetQnnLibDir(engineHandle: Long, qnnLibDir: String)
    private external fun nativeGetActiveBackend(engineHandle: Long): String

    // Phase 7: Thermal management (ADPF)
    private external fun nativeInitThermalManager(engineHandle: Long): Boolean
    private external fun nativePollThermalHeadroom(engineHandle: Long, forecastSeconds: Int): Float
    private external fun nativeGetThermalState(engineHandle: Long): Int
    private external fun nativeGetThermalHeadroom(engineHandle: Long): Float

    // =========================================================================
    // Phase 7: Thermal Management — ADPF Integration
    // =========================================================================

    /**
     * Initialize the ADPF Thermal Manager. Call once after startEngine().
     * Returns false if ADPF is unavailable on this device (pre-API 31).
     */
    fun initThermalManager(): Boolean {
        if (engineHandle == 0L) return false
        return nativeInitThermalManager(engineHandle)
    }

    /**
     * Poll the device's thermal headroom using AThermal_getThermalHeadroom().
     * Call periodically (every 2-5s) from a monitoring coroutine, NOT from
     * the audio thread.
     *
     * @param forecastSeconds How many seconds ahead to predict (default 10)
     * @return Thermal headroom (0.0=cold, 1.0=throttling), or -1.0 if unavailable
     */
    fun pollThermalHeadroom(forecastSeconds: Int = 10): Float {
        if (engineHandle == 0L) return -1.0f
        return nativePollThermalHeadroom(engineHandle, forecastSeconds)
    }

    /**
     * Get the current thermal state as an integer:
     *   0 = Nominal (< 0.5 headroom — full NPU)
     *   1 = Warm (0.5–0.7 — NPU with warning)
     *   2 = Throttling (0.7–0.9 — spectral gate fallback)
     *   3 = Critical (≥ 0.9 — emergency fallback)
     *  -1 = Not available
     */
    fun getThermalState(): Int {
        if (engineHandle == 0L) return -1
        return nativeGetThermalState(engineHandle)
    }

    /**
     * Get the raw thermal headroom value (0.0 = cold, 1.0 = imminent throttling).
     * Returns -1.0 if thermal monitoring is unavailable.
     */
    fun getThermalHeadroom(): Float {
        if (engineHandle == 0L) return -1.0f
        return nativeGetThermalHeadroom(engineHandle)
    }

    /**
     * Returns the thermal state as a human-readable string for UI display.
     */
    fun getThermalStateName(): String {
        return when (getThermalState()) {
            0 -> "Nominal"
            1 -> "Warm"
            2 -> "Throttling"
            3 -> "Critical"
            else -> "Unavailable"
        }
    }

    // =========================================================================
    // Phase 2: Offline training (called from WorkManager background thread)
    // =========================================================================
    external fun nativeStartTraining(
        corpusDir: String,
        baseModelPath: String,
        adapterPath: String,
        epochs: Int,
        learningRate: Float
    ): Boolean

    external fun nativeStopTraining()

    /**
     * Returns a float array of 8 values:
     * [epoch, totalEpochs, currentLoss, bestLoss, framesProcessed,
     *  totalFrames, isRunning, isComplete]
     */
    external fun nativeGetTrainingProgress(): FloatArray?

    // =========================================================================
    // Phase 6: Encrypted corpus data feed
    // =========================================================================

    /**
     * Feeds a decrypted corpus buffer directly to the native training pipeline.
     * Called from Kotlin after reading and decrypting an EncryptedFile.
     *
     * The buffer must be a direct ByteBuffer containing float32_le audio samples.
     * This replaces the file-based loadCorpus() path for encrypted corpus files.
     *
     * @param buffer Direct ByteBuffer of decrypted float32 PCM audio
     * @param sampleCount Number of float samples in the buffer
     * @param sourceId Unique identifier for the corpus chunk (e.g., filename hash)
     * @return Number of training frames extracted from this buffer
     */
    external fun nativeFeedDecryptedCorpus(
        buffer: java.nio.ByteBuffer,
        sampleCount: Int,
        sourceId: String
    ): Int

    /**
     * Signals the native pipeline to finalize corpus ingestion.
     * Must be called after all chunks have been fed via nativeFeedDecryptedCorpus().
     * The C++ DataPipeline will compute SNR weights and build the shuffled index.
     */
    external fun nativeFinalizeCorpus(): Int

    // =========================================================================
    // Phase 8: Audio Level + Waveform Data for UI Visualizer
    // =========================================================================

    /**
     * Retrieves a snapshot of audio levels and waveform data from the native
     * lock-free double buffer. Returns a FloatArray of 130 elements:
     *   [0]       = input RMS (0.0–1.0)
     *   [1]       = output RMS (0.0–1.0)
     *   [2..65]   = input waveform (64 samples, -1.0 to 1.0)
     *   [66..129] = output waveform (64 samples, -1.0 to 1.0)
     *
     * Safe to call from any thread at ~30fps. The native side uses a
     * lock-free double buffer so this never blocks the audio callback.
     */
    @JvmStatic
    fun nativeGetAudioLevels(): FloatArray? {
        val handle = engineHandle
        if (handle == 0L) return null
        return nativeGetAudioLevels(handle)
    }

    private external fun nativeGetAudioLevels(engineHandle: Long): FloatArray?

    // =========================================================================
    // Phase 11: Psychoacoustic Masking — Kotlin Interface
    // =========================================================================

    /** Processing mode: 0 = Voice Isolate (DTLN), 1 = Comfort Mask */
    @JvmStatic
    fun setProcessingMode(mode: Int) {
        val handle = engineHandle
        if (handle != 0L) nativeSetProcessingMode(handle, mode)
    }

    @JvmStatic
    fun getProcessingMode(): Int {
        val handle = engineHandle
        if (handle == 0L) return 0
        return nativeGetProcessingMode(handle)
    }

    /** Whether the masking engine is actively generating mask audio */
    @JvmStatic
    fun isMaskActive(): Boolean {
        val handle = engineHandle
        if (handle == 0L) return false
        return nativeIsMaskActive(handle)
    }

    /** Mask texture: 0=Brown, 1=Pink, 2=White, 3=Nature */
    @JvmStatic
    fun setMaskTexture(texture: Int) {
        val handle = engineHandle
        if (handle != 0L) nativeSetMaskTexture(handle, texture)
    }

    @JvmStatic
    fun getMaskTexture(): Int {
        val handle = engineHandle
        if (handle == 0L) return 0
        return nativeGetMaskTexture(handle)
    }

    /** Detected environment: 0=Quiet, 1=Office, 2=Café, 3=Transit, 4=Outdoor */
    @JvmStatic
    fun getDetectedEnvironment(): Int {
        val handle = engineHandle
        if (handle == 0L) return 0
        return nativeGetDetectedEnvironment(handle)
    }

    /** Whether a transient alert (horn, alarm, shout) is currently active */
    @JvmStatic
    fun isAlertActive(): Boolean {
        val handle = engineHandle
        if (handle == 0L) return false
        return nativeIsAlertActive(handle)
    }

    // --- Phase 11 native extern declarations ---
    private external fun nativeSetProcessingMode(engineHandle: Long, mode: Int)
    private external fun nativeGetProcessingMode(engineHandle: Long): Int
    private external fun nativeIsMaskActive(engineHandle: Long): Boolean
    private external fun nativeSetMaskTexture(engineHandle: Long, texture: Int)
    private external fun nativeGetMaskTexture(engineHandle: Long): Int
    private external fun nativeGetDetectedEnvironment(engineHandle: Long): Int
    private external fun nativeIsAlertActive(engineHandle: Long): Boolean

    // =========================================================================
    // Phase 15: Vision-to-Music — Kotlin Interface
    // =========================================================================

    /**
     * Push new scene parameters from YOLO inference to the native synth.
     * The synth smoothly transitions to matching music over ~3 seconds.
     *
     * @param density  Object density [0-1], affects note density and richness
     * @param valence  Emotional valence [0-1], affects key, scale, chord quality
     * @param arousal  Energy level [0-1], affects tempo and rhythmic intensity
     * @param timbre   Color/brightness [0-1], affects waveform and filter cutoff
     */
    @JvmStatic
    fun updateMusicalParameters(density: Float, valence: Float,
                                 arousal: Float, timbre: Float) {
        val handle = engineHandle
        if (handle != 0L) nativeUpdateMusicalParameters(handle, density, valence, arousal, timbre)
    }

    /**
     * Phase 19: Synesthesia Mapping
     * Pushes the dominant extracted HSV values from the camera frame.
     */
    @JvmStatic
    fun updateSynesthesiaParams(hue: Float, saturation: Float, value: Float) {
        val handle = engineHandle
        if (handle != 0L) nativeUpdateSynesthesiaParams(handle, hue, saturation, value)
    }

    /**
     * Phase 19: New Song Transitions
     * Instructs the engine to hard-flush the current motif and transition into a new state.
     */
    @JvmStatic
    fun flushGenerativeState() {
        val handle = engineHandle
        if (handle != 0L) nativeFlushGenerativeState(handle)
    }

    /**
     * Push inference offsets generated by TinyMusician inference context
     * param data 20-float array (4 offsets + 16 sequence notes)
     */
    @JvmStatic
    fun pushTinyMusicianData(data: FloatArray) {
        val handle = engineHandle
        if (handle != 0L) nativePushTinyMusicianData(handle, data)
    }

    /**
     * Set the music interpretation style.
     * 0 = Ambient Drift (warm evolving pads)
     * 1 = Melodic Arpeggio (plucked arpeggiated patterns)
     * 2 = Rhythmic Pulse (pulsing bass + rhythmic gates)
     */
    @JvmStatic
    fun setMusicInterpretation(interpretation: Int) {
        val handle = engineHandle
        if (handle != 0L) nativeSetMusicInterpretation(handle, interpretation)
    }
    
    // Kotlin-level state for tinyMusicianMode
    @Volatile
    private var tinyMusicianMode = 0
    
    @JvmStatic
    fun setTinyMusicianMode(mode: Int) {
        tinyMusicianMode = mode
    }
    
    @JvmStatic
    fun getTinyMusicianMode(): Int {
        return tinyMusicianMode
    }

    @JvmStatic
    fun getMusicInterpretation(): Int {
        val handle = engineHandle
        if (handle == 0L) return 0
        return nativeGetMusicInterpretation(handle)
    }

    /** Current tempo in BPM (mapped from arousal) */
    @JvmStatic
    fun getMusicBpm(): Float {
        val handle = engineHandle
        if (handle == 0L) return 0.0f
        return nativeGetMusicBpm(handle)
    }

    /** Current key (0=C, 1=C#, ... 11=B) */
    @JvmStatic
    fun getMusicKey(): Int {
        val handle = engineHandle
        if (handle == 0L) return 0
        return nativeGetMusicKey(handle)
    }

    /** Interpretation names for UI display */
    @JvmStatic
    fun getInterpretationName(interpretation: Int): String {
        return when (interpretation) {
            0 -> "Ambient Drift"
            1 -> "Melodic Arpeggio"
            2 -> "Rhythmic Pulse"
            else -> "Unknown"
        }
    }

    /** Key name for UI display */
    @JvmStatic
    fun getKeyName(key: Int): String {
        val names = arrayOf("C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B")
        return if (key in 0..11) names[key] else "?"
    }

    /** 
     * Set the synthesizer DSP quality level.
     * 0 = Battery Saver (1-pole lowpass filters, minimum compute)
     * 1 = Beautiful (4-pole resonant Moog-style filters, slightly more compute)
     */
    @JvmStatic
    fun setSynthQuality(quality: Int) {
        val handle = engineHandle
        if (handle != 0L) nativeSetSynthQuality(handle, quality)
    }

    @JvmStatic
    fun getSynthQuality(): Int {
        val handle = engineHandle
        if (handle == 0L) return 0
        return nativeGetSynthQuality(handle)
    }

    // --- Phase 15 native extern declarations ---
    private external fun nativeUpdateMusicalParameters(
        engineHandle: Long, density: Float, valence: Float, arousal: Float, timbre: Float
    )
    private external fun nativeSetMusicInterpretation(engineHandle: Long, interpretation: Int)
    private external fun nativeGetMusicInterpretation(engineHandle: Long): Int
    private external fun nativeGetMusicBpm(engineHandle: Long): Float
    private external fun nativeGetMusicKey(engineHandle: Long): Int
    private external fun nativePushTinyMusicianData(engineHandle: Long, data: FloatArray)
    private external fun nativeSetSynthQuality(engineHandle: Long, quality: Int)
    private external fun nativeGetSynthQuality(engineHandle: Long): Int

    // --- Phase 19 native extern declarations ---
    private external fun nativeUpdateSynesthesiaParams(engineHandle: Long, hue: Float, saturation: Float, value: Float)
    private external fun nativeFlushGenerativeState(engineHandle: Long)
}
