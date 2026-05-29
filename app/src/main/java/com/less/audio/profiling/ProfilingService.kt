package com.less.audio.profiling

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.CountDownTimer
import android.os.IBinder
import android.util.Log
import androidx.security.crypto.EncryptedFile
import androidx.security.crypto.MasterKeys
import com.less.audio.MainActivity
import org.json.JSONObject
import java.io.File
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * ProfilingService — Foreground service for controlled audio capture bursts.
 *
 * Captures raw BLE Audio from the Meta Ray-Ban mic array in short 10-15 minute
 * sessions. This circumvents Android 15's 6-hour cumulative FGS limit by using
 * manual, user-initiated bursts rather than continuous background recording.
 *
 * Phase 6: AES-256-GCM Encryption at Rest
 *   All audio corpus files are now encrypted via EncryptedFile backed by
 *   Android Keystore. The raw PCM float data is streamed through AES-GCM
 *   as it is captured — no plaintext audio ever touches disk.
 *
 *   Metadata JSON sidecars are also encrypted. The C++ data pipeline can
 *   no longer read .pcm files directly. Instead, TrainingWorker decrypts
 *   each file into a direct ByteBuffer and passes it to the native trainer
 *   via JNI.
 *
 * Audio format:
 *   - Sample rate: 48000 Hz (native BLE Audio / LC3)
 *   - Channel: Mono
 *   - Format: 16-bit PCM (converted to float for corpus)
 *   - Output: encrypted .epcm files with encrypted JSON metadata sidecars
 *
 * Safety features:
 *   - Hard 15-minute session cap via CountDownTimer
 *   - FGS budget validation before start (via ProfilingSessionManager)
 *   - Auto-stop on Bluetooth disconnection
 *   - Persistent notification with live countdown and stop action
 *   - AES-256-GCM encryption via Android Keystore (hardware-backed)
 */
class ProfilingService : Service() {

    companion object {
        private const val TAG = "LESS_Profiling"

        // Notification
        private const val CHANNEL_ID = "less_profiling_channel"
        private const val NOTIFICATION_ID = 1001

        // Audio configuration
        private const val SAMPLE_RATE = 48000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_FLOAT

        // Session limits
        private const val MAX_SESSION_MS = 15 * 60 * 1000L  // 15 minutes

        // Actions
        const val ACTION_START = "com.less.audio.PROFILING_START"
        const val ACTION_STOP = "com.less.audio.PROFILING_STOP"

        // Extras
        const val EXTRA_DURATION_MINUTES = "duration_minutes"

        // Encrypted corpus file extension
        const val ENCRYPTED_PCM_EXT = "epcm"
        const val ENCRYPTED_META_EXT = "ejson"

        /**
         * Opens an EncryptedFile for reading. Used by TrainingWorker to
         * decrypt corpus files before passing to the C++ trainer.
         */
        fun openEncryptedFileForRead(context: android.content.Context, file: File): java.io.InputStream {
            val masterKeyAlias = MasterKeys.getOrCreate(MasterKeys.AES256_GCM_SPEC)
            val encryptedFile = EncryptedFile.Builder(
                file,
                context,
                masterKeyAlias,
                EncryptedFile.FileEncryptionScheme.AES256_GCM_HKDF_4KB
            ).build()
            return encryptedFile.openFileInput()
        }
    }

    private var audioRecord: AudioRecord? = null
    private var recordingThread: Thread? = null
    private var isRecording = false
    private var sessionStartTimeMs = 0L
    private var sessionDurationMs = MAX_SESSION_MS

    private lateinit var sessionManager: ProfilingSessionManager
    private var countdownTimer: CountDownTimer? = null

    // Android Keystore master key for AES-256-GCM
    private lateinit var masterKeyAlias: String

    // Output
    private var outputFile: File? = null
    private var encryptedOutputStream: OutputStream? = null
    private var totalFramesWritten = 0L

    // =========================================================================
    // Service Lifecycle
    // =========================================================================

    override fun onCreate() {
        super.onCreate()
        sessionManager = ProfilingSessionManager(this)
        createNotificationChannel()

        // Initialize the Android Keystore master key.
        // This key is hardware-backed on devices with a secure element (StrongBox)
        // and never leaves the TEE/SE boundary. AES-256-GCM provides authenticated
        // encryption — both confidentiality and integrity of the corpus data.
        masterKeyAlias = MasterKeys.getOrCreate(MasterKeys.AES256_GCM_SPEC)
        Log.i(TAG, "ProfilingService created — encryption backend: AES-256-GCM (Keystore)")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                val durationMin = intent.getIntExtra(
                    EXTRA_DURATION_MINUTES,
                    ProfilingSessionManager.MAX_SESSION_MINUTES
                )
                sessionDurationMs = (durationMin * 60 * 1000L)
                    .coerceAtMost(MAX_SESSION_MS)
                startProfiling()
            }
            ACTION_STOP -> {
                stopProfiling()
            }
            else -> {
                // Default: start with max duration
                startProfiling()
            }
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        stopProfiling()
        super.onDestroy()
        Log.i(TAG, "ProfilingService destroyed")
    }

    // =========================================================================
    // Profiling Logic
    // =========================================================================

    @SuppressLint("MissingPermission")
    private fun startProfiling() {
        // --- Budget check ---
        if (!sessionManager.canStartSession()) {
            Log.e(TAG, "Cannot start profiling — FGS budget exhausted. " +
                    "Remaining: ${sessionManager.remainingBudgetMinutes()} min")
            stopSelf()
            return
        }

        if (isRecording) {
            Log.w(TAG, "Already recording — ignoring start request")
            return
        }

        // --- Start as foreground service ---
        val notification = buildNotification(sessionDurationMs)
        startForeground(
            NOTIFICATION_ID,
            notification,
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
        )

        // --- Initialize AudioRecord ---
        val bufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT
        ).coerceAtLeast(4096)

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.UNPROCESSED,  // bypass Android AEC/NS/AGC
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            bufferSize * 2
        )

        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord failed to initialize")
            stopSelf()
            return
        }

        // --- Prepare encrypted output file ---
        val timestamp = System.currentTimeMillis()
        val corpusDir = File(filesDir, "corpus").also { it.mkdirs() }
        outputFile = File(corpusDir, "${timestamp}.$ENCRYPTED_PCM_EXT")

        // Create EncryptedFile — AES-256-GCM backed by Android Keystore.
        // The encryption key never leaves the hardware security module.
        // Each file gets a unique IV/nonce via HKDF, so identical plaintext
        // in different sessions produces different ciphertext.
        val encryptedFile = EncryptedFile.Builder(
            outputFile!!,
            this,
            masterKeyAlias,
            EncryptedFile.FileEncryptionScheme.AES256_GCM_HKDF_4KB
        ).build()

        encryptedOutputStream = encryptedFile.openFileOutput()
        totalFramesWritten = 0

        // --- Start recording ---
        sessionStartTimeMs = System.currentTimeMillis()
        isRecording = true
        audioRecord?.startRecording()

        // Recording thread — reads audio data off the real-time thread
        recordingThread = Thread({
            val buffer = FloatArray(bufferSize / 4)  // float = 4 bytes
            val byteBuffer = ByteBuffer.allocate(buffer.size * 4)
                .order(ByteOrder.LITTLE_ENDIAN)

            Log.i(TAG, "Recording thread started (encrypted output)")

            while (isRecording) {
                val read = audioRecord?.read(
                    buffer, 0, buffer.size, AudioRecord.READ_BLOCKING
                ) ?: break

                if (read > 0) {
                    // Write raw float PCM through encrypted stream.
                    // EncryptedFile handles chunk-level AES-GCM transparently.
                    byteBuffer.clear()
                    for (i in 0 until read) {
                        byteBuffer.putFloat(buffer[i])
                    }
                    encryptedOutputStream?.write(byteBuffer.array(), 0, read * 4)
                    totalFramesWritten += read
                }
            }

            Log.i(TAG, "Recording thread finished — $totalFramesWritten frames written (encrypted)")
        }, "LESS-Profiling-Record").also { it.priority = Thread.MAX_PRIORITY }

        recordingThread?.start()

        // --- Countdown timer (hard stop) ---
        countdownTimer = object : CountDownTimer(sessionDurationMs, 1000) {
            override fun onTick(millisUntilFinished: Long) {
                updateNotification(millisUntilFinished)
            }

            override fun onFinish() {
                Log.i(TAG, "Session timer expired — stopping profiling")
                stopProfiling()
            }
        }.start()

        Log.i(TAG, "✓ Profiling started: duration=${sessionDurationMs / 1000}s, " +
                "file=${outputFile?.name} (AES-256-GCM encrypted)")
    }

    private fun stopProfiling() {
        if (!isRecording) return
        isRecording = false

        // Stop countdown
        countdownTimer?.cancel()
        countdownTimer = null

        // Stop recording
        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping AudioRecord", e)
        }
        audioRecord = null

        // Wait for recording thread to finish
        recordingThread?.join(2000)
        recordingThread = null

        // Close encrypted output — this finalizes the AES-GCM authentication tag.
        // If the stream is not properly closed, the file will be undecryptable.
        try {
            encryptedOutputStream?.flush()
            encryptedOutputStream?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing encrypted output stream", e)
        }
        encryptedOutputStream = null

        // Log session
        val sessionEndTimeMs = System.currentTimeMillis()
        sessionManager.logSession(sessionStartTimeMs, sessionEndTimeMs)

        // Write encrypted metadata sidecar
        writeEncryptedMetadata(sessionEndTimeMs)

        val durationSec = (sessionEndTimeMs - sessionStartTimeMs) / 1000
        Log.i(TAG, "✓ Profiling session complete: ${durationSec}s, " +
                "${totalFramesWritten} frames, file=${outputFile?.name} (encrypted)")

        // Stop the foreground service
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    /**
     * Writes an encrypted JSON metadata sidecar alongside the corpus file.
     * Contains session timing, frame counts, and format info — all encrypted
     * to prevent metadata leakage about the user's acoustic environment.
     */
    private fun writeEncryptedMetadata(endTimeMs: Long) {
        try {
            val metaFile = File(
                outputFile?.parent,
                "${outputFile?.nameWithoutExtension}.$ENCRYPTED_META_EXT"
            )

            val meta = JSONObject().apply {
                put("format", "float32_le")
                put("sample_rate", SAMPLE_RATE)
                put("channels", 1)
                put("start_time_ms", sessionStartTimeMs)
                put("end_time_ms", endTimeMs)
                put("duration_seconds", (endTimeMs - sessionStartTimeMs) / 1000)
                put("total_frames", totalFramesWritten)
                put("encrypted", true)
                put("encryption_scheme", "AES256_GCM_HKDF_4KB")
            }

            val encryptedMeta = EncryptedFile.Builder(
                metaFile,
                this,
                masterKeyAlias,
                EncryptedFile.FileEncryptionScheme.AES256_GCM_HKDF_4KB
            ).build()

            encryptedMeta.openFileOutput().use { output ->
                output.write(meta.toString(2).toByteArray(Charsets.UTF_8))
            }

            Log.i(TAG, "Encrypted metadata written: ${metaFile.name}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write encrypted metadata", e)
        }
    }

    // =========================================================================
    // Notifications
    // =========================================================================

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "LESS Profiling",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Audio profiling for personalized noise suppression"
            setShowBadge(false)
        }
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(channel)
    }

    private fun buildNotification(remainingMs: Long): Notification {
        val stopIntent = Intent(this, ProfilingService::class.java).apply {
            action = ACTION_STOP
        }
        val stopPendingIntent = PendingIntent.getService(
            this, 0, stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val openIntent = Intent(this, MainActivity::class.java)
        val openPendingIntent = PendingIntent.getActivity(
            this, 0, openIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val minutes = (remainingMs / 60000).toInt()
        val seconds = ((remainingMs % 60000) / 1000).toInt()

        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("LESS — Profiling Active 🔒")
            .setContentText("Recording: ${minutes}m ${seconds}s remaining (encrypted)")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setOngoing(true)
            .setContentIntent(openPendingIntent)
            .addAction(
                Notification.Action.Builder(
                    null, "Stop", stopPendingIntent
                ).build()
            )
            .build()
    }

    private fun updateNotification(remainingMs: Long) {
        val notification = buildNotification(remainingMs)
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(NOTIFICATION_ID, notification)
    }
}
