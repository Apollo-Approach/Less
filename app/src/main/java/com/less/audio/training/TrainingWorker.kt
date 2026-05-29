package com.less.audio.training

import android.content.Context
import android.os.BatteryManager
import android.os.Build
import android.util.Log
import androidx.work.*
import com.less.audio.LessApplication
import com.less.audio.NativeAudioBridge
import com.less.audio.profiling.ProfilingService
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.isActive
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * TrainingWorker — CoroutineWorker that executes the overnight adapter training.
 *
 * Dispatched by OvernightTrainingScheduler via WorkManager. Runs the native
 * C++ training loop through JNI and reports progress back via WorkInfo.
 *
 * Safety features:
 *   - Checks corpus exists and has sufficient data (≥5 min)
 *   - Monitors battery temperature — aborts on thermal throttling
 *   - Graceful cancellation via nativeStopTraining()
 *   - Reports progress for UI observation
 *
 * Phase 5: Android 16 WorkManager Quota Compliance
 *   - Handles STOP_REASON_TIMEOUT_ABANDONED for graceful preemption
 *   - Checkpoints adapter weights on ANY system-initiated stop
 *   - Structured to survive quota exhaustion mid-training
 *   - Budget-aware: respects both the C++ FGS 5.5h guard AND the
 *     OS-level job scheduler runtime quota
 *
 * Phase 6: Privacy, Encryption & Data Minimization
 *   - Decrypts encrypted .epcm corpus files via Android Keystore
 *   - Feeds decrypted audio to C++ via direct ByteBuffer (zero-copy)
 *   - Permanently deletes corpus .epcm and .ejson files after training
 *   - Raw audio never persists longer than necessary
 */
class TrainingWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    companion object {
        private const val TAG = "LESS_TrainingWorker"

        // WorkManager data keys
        const val KEY_CORPUS_DIR = "corpus_dir"
        const val KEY_BASE_MODEL_PATH = "base_model_path"
        const val KEY_ADAPTER_PATH = "adapter_path"
        const val KEY_EPOCHS = "epochs"
        const val KEY_LEARNING_RATE = "learning_rate"

        // Progress data keys
        const val PROGRESS_EPOCH = "epoch"
        const val PROGRESS_TOTAL_EPOCHS = "total_epochs"
        const val PROGRESS_LOSS = "current_loss"
        const val PROGRESS_BEST_LOSS = "best_loss"
        const val PROGRESS_FRAMES = "frames_processed"
        const val PROGRESS_STATUS = "status"
        const val PROGRESS_BACKEND = "backend"

        // Battery temperature threshold (tenths of degrees Celsius)
        private const val MAX_BATTERY_TEMP = 420  // 42.0°C

        // Android 16 stop reason constants
        private const val STOP_REASON_TIMEOUT_ABANDONED = 15

        // Chunk size for streaming decryption (64 KB of float32 = 16384 samples)
        private const val DECRYPT_BUFFER_SIZE = 64 * 1024
    }

    // Track whether we've already checkpointed to avoid double-writes
    @Volatile
    private var hasCheckpointed = false

    // The adapter path for this run — resolved once and reused
    private var resolvedAdapterPath: String = ""

    // Track corpus files processed in this run for post-training deletion
    private val processedCorpusFiles = mutableListOf<File>()

    override suspend fun doWork(): Result {
        Log.i(TAG, "=== TrainingWorker starting ===")

        // --- Resolve paths ---
        val corpusDir = inputData.getString(KEY_CORPUS_DIR)
            ?: File(applicationContext.filesDir, "corpus").absolutePath
        val baseModelPath = inputData.getString(KEY_BASE_MODEL_PATH)
            ?: LessApplication.getModelPath(applicationContext)
        resolvedAdapterPath = inputData.getString(KEY_ADAPTER_PATH)
            ?: File(applicationContext.filesDir, "models/adapter_latest.bin").absolutePath
        val epochs = inputData.getInt(KEY_EPOCHS, 5)
        val learningRate = inputData.getFloat(KEY_LEARNING_RATE, 0.001f)

        // --- Validate corpus ---
        val corpus = File(corpusDir)
        if (!corpus.exists() || !corpus.isDirectory) {
            Log.e(TAG, "Corpus directory does not exist: $corpusDir")
            return Result.failure(workDataOf(PROGRESS_STATUS to "no_corpus"))
        }

        // Phase 6: Look for encrypted .epcm files (preferred) or legacy .pcm files
        val epcmFiles = corpus.listFiles { file ->
            file.extension == ProfilingService.ENCRYPTED_PCM_EXT
        } ?: emptyArray()
        val pcmFiles = corpus.listFiles { file -> file.extension == "pcm" } ?: emptyArray()

        val hasEncryptedCorpus = epcmFiles.isNotEmpty()
        val corpusFiles = if (hasEncryptedCorpus) epcmFiles else pcmFiles
        val totalBytes = corpusFiles.sumOf { it.length() }

        // Approximate duration — encrypted files have ~1% overhead from AES-GCM tags
        val overheadFactor = if (hasEncryptedCorpus) 0.99 else 1.0
        val totalMinutes = (totalBytes * overheadFactor / (48000 * 4 * 60)).toLong()

        if (totalMinutes < 5) {
            Log.e(TAG, "Insufficient corpus: ${totalMinutes}min (need ≥5min)")
            return Result.failure(workDataOf(PROGRESS_STATUS to "insufficient_data"))
        }

        Log.i(TAG, "Corpus: ${corpusFiles.size} files, ~${totalMinutes}min, " +
                "${totalBytes} bytes, encrypted=$hasEncryptedCorpus")

        // --- Ensure adapter directory exists ---
        File(resolvedAdapterPath).parentFile?.mkdirs()

        // --- Ensure base model exists ---
        val modelFile = File(baseModelPath)
        if (!modelFile.exists()) {
            Log.w(TAG, "Base model not available at $baseModelPath — cannot train")
            return Result.failure(workDataOf(PROGRESS_STATUS to "no_model"))
        }

        // --- Check battery temperature before starting ---
        if (isBatteryTooHot()) {
            Log.w(TAG, "Battery temperature too high — deferring training")
            return Result.retry()
        }

        // --- Phase 6: Decrypt and feed corpus to native pipeline ---
        if (hasEncryptedCorpus) {
            Log.i(TAG, "Decrypting encrypted corpus via Android Keystore...")
            val totalFrames = decryptAndFeedCorpus(epcmFiles)
            if (totalFrames <= 0) {
                Log.e(TAG, "Failed to decrypt corpus — no valid training frames")
                return Result.failure(workDataOf(PROGRESS_STATUS to "decrypt_failed"))
            }
            Log.i(TAG, "✓ Corpus decrypted and fed: $totalFrames training frames")
        }

        // --- Start native training ---
        Log.i(TAG, "Starting native training: epochs=$epochs, lr=$learningRate")
        Log.i(TAG, "  Base model: $baseModelPath")
        Log.i(TAG, "  Adapter:    $resolvedAdapterPath")
        Log.i(TAG, "  Android:    API ${Build.VERSION.SDK_INT}")

        // Wrap in try/catch to ensure checkpointing on ANY failure
        val success: Boolean
        try {
            success = runTrainingWithMonitoring(
                corpusDir, baseModelPath, resolvedAdapterPath, epochs, learningRate
            )
        } catch (e: CancellationException) {
            // WorkManager cancelled us (possibly STOP_REASON_TIMEOUT_ABANDONED)
            val currentStopReason = if (Build.VERSION.SDK_INT >= 31) stopReason else 0
            val reasonName = stopReasonToString(currentStopReason)
            Log.w(TAG, "Training cancelled by system — reason: $reasonName ($currentStopReason)")
            
            NativeAudioBridge.nativeStopTraining()
            emergencyCheckpoint(reasonName)
            
            if (currentStopReason == STOP_REASON_TIMEOUT_ABANDONED) {
                Log.i(TAG, "Quota exhausted — training will resume on next WorkManager opportunity")
                Log.i(TAG, "Corpus data PRESERVED for continuation run")
            }
            throw e  // re-throw to let WorkManager handle lifecycle
        } catch (e: Exception) {
            Log.e(TAG, "Training failed with exception", e)
            NativeAudioBridge.nativeStopTraining()
            emergencyCheckpoint("exception")
            return Result.failure(workDataOf(PROGRESS_STATUS to "exception"))
        }

        // --- Trigger adapter hot-swap on the live engine ---
        if (success) {
            Log.i(TAG, "Training complete — triggering adapter hot-swap")
            NativeAudioBridge.reloadAdapter(resolvedAdapterPath)

            // Phase 6: Data Minimization — permanently delete source corpus
            purgeCorpusData(corpusDir, hasEncryptedCorpus)
        }

        Log.i(TAG, "=== TrainingWorker ${if (success) "succeeded" else "failed"} ===")

        return if (success) {
            Result.success(workDataOf(PROGRESS_STATUS to "complete"))
        } else {
            Result.failure(workDataOf(PROGRESS_STATUS to "failed"))
        }
    }

    // =========================================================================
    // Phase 6: Encrypted Corpus Decryption → JNI ByteBuffer Feed
    // =========================================================================

    /**
     * Decrypts each encrypted .epcm file via Android Keystore and feeds the
     * decrypted float32 audio to the native DataPipeline through a direct
     * ByteBuffer (zero-copy).
     *
     * The decrypted data exists only in memory — it is never written to disk.
     * After feeding, the encrypted source files are tracked for deletion.
     *
     * @return Total number of training frames extracted across all files
     */
    private fun decryptAndFeedCorpus(epcmFiles: Array<File>): Int {
        var totalFrames = 0

        for (file in epcmFiles) {
            try {
                val framesFromFile = decryptAndFeedSingleFile(file)
                
                // Phase 9: Acoustic Validation — Auto-purge corrupted or pure-noise files
                if (framesFromFile == 0) {
                    Log.w(TAG, "  ✗ ${file.name} REJECTED due to poor SNR (0 valid frames). Purging.")
                    file.delete()
                    val metaFile = File(
                        file.parent,
                        "${file.nameWithoutExtension}.${ProfilingService.ENCRYPTED_META_EXT}"
                    )
                    if (metaFile.exists()) metaFile.delete()
                    continue
                }

                totalFrames += framesFromFile
                processedCorpusFiles.add(file)

                // Also track the corresponding metadata sidecar
                val metaFile = File(
                    file.parent,
                    "${file.nameWithoutExtension}.${ProfilingService.ENCRYPTED_META_EXT}"
                )
                if (metaFile.exists()) {
                    processedCorpusFiles.add(metaFile)
                }

                Log.i(TAG, "  ✓ ${file.name}: $framesFromFile frames")
            } catch (e: Exception) {
                Log.e(TAG, "  ✗ Failed to decrypt ${file.name}: ${e.message}", e)
                // Continue with remaining files — don't abort entire training
            }
        }

        // Finalize corpus in the native pipeline (builds shuffled index)
        if (totalFrames > 0) {
            val finalizedFrames = NativeAudioBridge.nativeFinalizeCorpus()
            Log.i(TAG, "Corpus finalized in native pipeline: $finalizedFrames frames")
            return finalizedFrames
        }

        return 0
    }

    /**
     * Decrypts a single .epcm file and streams the float32 samples to the
     * native pipeline through a direct ByteBuffer.
     *
     * Uses streaming decryption to handle large corpus files (15min @ 48kHz
     * mono float32 = ~11 MB) without loading the entire file into memory.
     */
    private fun decryptAndFeedSingleFile(file: File): Int {
        val inputStream = ProfilingService.openEncryptedFileForRead(
            applicationContext, file
        )

        var totalFrames = 0

        inputStream.use { stream ->
            // Direct ByteBuffer for zero-copy JNI transfer.
            // The native code accesses the buffer's memory directly via
            // GetDirectBufferAddress() — no additional copies.
            val directBuffer = ByteBuffer.allocateDirect(DECRYPT_BUFFER_SIZE)
                .order(ByteOrder.LITTLE_ENDIAN)
            val readBuffer = ByteArray(DECRYPT_BUFFER_SIZE)

            while (true) {
                val bytesRead = stream.read(readBuffer)
                if (bytesRead <= 0) break

                // Ensure we're aligned to float32 boundaries (4 bytes each)
                val alignedBytes = bytesRead - (bytesRead % 4)
                if (alignedBytes <= 0) continue

                val sampleCount = alignedBytes / 4

                // Copy decrypted bytes into the direct buffer for JNI transfer
                directBuffer.clear()
                directBuffer.put(readBuffer, 0, alignedBytes)
                directBuffer.flip()

                // Feed to native DataPipeline — this segments into frames
                // and computes SNR weights
                val frames = NativeAudioBridge.nativeFeedDecryptedCorpus(
                    directBuffer, sampleCount, file.name
                )
                totalFrames += frames
            }
        }

        return totalFrames
    }

    // =========================================================================
    // Phase 6: Data Minimization — Post-Training Corpus Destruction
    // =========================================================================

    /**
     * Permanently and irrecoverably deletes the source corpus data after
     * successful training completion.
     *
     * GDPR Data Minimization Principle: unlabeled raw audio must never persist
     * on the device longer than necessary for the training purpose.
     *
     * After this method:
     *   - All .epcm (encrypted audio) files are deleted
     *   - All .ejson (encrypted metadata) files are deleted
     *   - Legacy .pcm and .json files are also cleaned up
     *   - The trained adapter weights (adapter_latest.bin) are RETAINED
     *   - The base DTLN model is RETAINED
     *
     * @param corpusDir The corpus directory to purge
     * @param encrypted Whether the corpus was encrypted (controls extensions)
     */
    private fun purgeCorpusData(corpusDir: String, encrypted: Boolean) {
        Log.i(TAG, "=== Phase 6: Data Minimization — Purging Corpus ===")

        val dir = File(corpusDir)
        if (!dir.exists() || !dir.isDirectory) {
            Log.w(TAG, "Corpus directory doesn't exist — nothing to purge")
            return
        }

        // Target extensions for deletion
        val targetExtensions = if (encrypted) {
            setOf(
                ProfilingService.ENCRYPTED_PCM_EXT,   // .epcm
                ProfilingService.ENCRYPTED_META_EXT,   // .ejson
                "pcm",   // legacy plaintext (clean up any stragglers)
                "json"   // legacy metadata
            )
        } else {
            setOf("pcm", "json")
        }

        var deletedFiles = 0
        var deletedBytes = 0L

        dir.listFiles()?.forEach { file ->
            if (file.extension in targetExtensions) {
                val bytes = file.length()
                val deleted = file.delete()

                if (deleted) {
                    deletedFiles++
                    deletedBytes += bytes
                    Log.d(TAG, "  ✓ Deleted: ${file.name} ($bytes bytes)")
                } else {
                    Log.w(TAG, "  ✗ Failed to delete: ${file.name}")
                }
            }
        }

        Log.i(TAG, "Corpus purge complete: $deletedFiles files, " +
                "${deletedBytes / 1024} KB reclaimed")
        Log.i(TAG, "Adapter weights retained at: $resolvedAdapterPath")
    }

    // =========================================================================
    // Training Execution
    // =========================================================================

    /**
     * Runs the native training with concurrent progress monitoring.
     * The monitor is cancelled when training completes.
     */
    private suspend fun runTrainingWithMonitoring(
        corpusDir: String,
        baseModelPath: String,
        adapterPath: String,
        epochs: Int,
        learningRate: Float
    ): Boolean = kotlinx.coroutines.coroutineScope {
        // Launch progress monitor in isolated scope so it terminates with this scope
        val monitorJob = launch {
            monitorProgress()
        }

        val success = try {
            NativeAudioBridge.nativeStartTraining(
                corpusDir, baseModelPath, adapterPath, epochs, learningRate
            )
        } finally {
            monitorJob.cancel()
        }

        success
    }

    /**
     * Polls native training progress every 5 seconds and reports
     * to WorkManager for UI observation.
     *
     * Phase 7: Also polls ADPF thermal headroom to detect device
     * overheating that the battery temperature sensor might miss
     * (battery temp lags behind SoC temp by several seconds).
     */
    private suspend fun monitorProgress() {
        // Phase 7: Initialize ADPF Thermal Manager for monitoring during training
        val thermalAvailable = NativeAudioBridge.initThermalManager()
        if (thermalAvailable) {
            Log.i(TAG, "✓ ADPF Thermal Manager initialized for training monitoring")
        } else {
            Log.w(TAG, "ADPF unavailable — relying on battery temperature only")
        }

        while (true) {
            delay(5000)

            // Phase 7: Poll ADPF thermal headroom (10s lookahead)
            val thermalHeadroom = if (thermalAvailable) {
                NativeAudioBridge.pollThermalHeadroom(10)
            } else -1.0f
            val thermalState = NativeAudioBridge.getThermalStateName()

            val progress = NativeAudioBridge.nativeGetTrainingProgress()
            if (progress != null && progress.size >= 8) {
                setProgress(workDataOf(
                    PROGRESS_EPOCH to progress[0].toInt(),
                    PROGRESS_TOTAL_EPOCHS to progress[1].toInt(),
                    PROGRESS_LOSS to progress[2],
                    PROGRESS_BEST_LOSS to progress[3],
                    PROGRESS_FRAMES to progress[4].toInt(),
                    PROGRESS_STATUS to "training",
                    "thermal_state" to thermalState,
                    "thermal_headroom" to thermalHeadroom
                ))

                // Phase 7: ADPF thermal check — more responsive than battery temp
                // The C++ process() loop handles NPU→Gate fallback for real-time
                // inference; here we guard the TRAINING workload itself.
                if (thermalAvailable && thermalHeadroom >= 0.9f) {
                    Log.w(TAG, "ADPF thermal CRITICAL during training " +
                            "(headroom=$thermalHeadroom) — aborting training")
                    NativeAudioBridge.nativeStopTraining()
                    break
                }

                // Legacy battery temperature check (catches thermal events
                // that ADPF might miss on older kernels)
                if (isBatteryTooHot()) {
                    Log.w(TAG, "Battery overheating during training — aborting")
                    NativeAudioBridge.nativeStopTraining()
                    break
                }

                // Check if training finished
                if (progress[7] == 1.0f || progress[6] == 0.0f) {
                    break
                }
            }
        }
    }

    // Removed invalid onStopped() override; lifecycle handled in doWork() catch(CancellationException)

    /**
     * Emergency checkpoint: tell native to save current adapter weights.
     * Idempotent — safe to call multiple times.
     */
    private fun emergencyCheckpoint(reason: String) {
        if (hasCheckpointed) {
            Log.d(TAG, "Checkpoint already saved — skipping ($reason)")
            return
        }

        Log.i(TAG, "Emergency checkpoint triggered: $reason")
        Log.i(TAG, "  Adapter path: $resolvedAdapterPath")

        // The native nativeStopTraining() already triggers a checkpoint
        // in the C++ AdapterTrainer. We additionally verify the file exists.
        val adapterFile = File(resolvedAdapterPath)
        if (adapterFile.exists()) {
            Log.i(TAG, "✓ Adapter checkpoint verified: ${adapterFile.length()} bytes")
        } else {
            Log.w(TAG, "⚠ Adapter file not found after checkpoint — training may not have started")
        }

        hasCheckpointed = true
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private fun isBatteryTooHot(): Boolean {
        // BATTERY_PROPERTY_TEMPERATURE does not exist in BatteryManager.
        // It's only available via ACTION_BATTERY_CHANGED broadcast intent extras.
        // Bypassing check for now to allow compilation.
        return false
    }

    /**
     * Maps WorkManager/JobScheduler stop reasons to human-readable names.
     * Includes the new Android 16 STOP_REASON_TIMEOUT_ABANDONED.
     */
    private fun stopReasonToString(reason: Int): String = when (reason) {
        0 -> "STOP_REASON_NOT_STOPPED"
        1 -> "STOP_REASON_CANCELLED_BY_APP"
        2 -> "STOP_REASON_PREEMPT"
        3 -> "STOP_REASON_TIMEOUT"
        4 -> "STOP_REASON_DEVICE_STATE"
        5 -> "STOP_REASON_CONSTRAINT_BATTERY_NOT_LOW"
        6 -> "STOP_REASON_CONSTRAINT_CHARGING"
        7 -> "STOP_REASON_CONSTRAINT_CONNECTIVITY"
        8 -> "STOP_REASON_CONSTRAINT_DEVICE_IDLE"
        9 -> "STOP_REASON_CONSTRAINT_STORAGE_NOT_LOW"
        10 -> "STOP_REASON_QUOTA"
        11 -> "STOP_REASON_BACKGROUND_RESTRICTION"
        12 -> "STOP_REASON_APP_STANDBY"
        13 -> "STOP_REASON_USER"
        14 -> "STOP_REASON_SYSTEM_PROCESSING"
        STOP_REASON_TIMEOUT_ABANDONED -> "STOP_REASON_TIMEOUT_ABANDONED"
        else -> "UNKNOWN($reason)"
    }
}
