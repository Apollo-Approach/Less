package com.less.audio.training

import android.content.Context
import android.util.Log
import androidx.work.*
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * OvernightTrainingScheduler — Schedules adapter training via WorkManager.
 *
 * Constraints:
 *   - RequiresCharging(true) — only train while plugged in
 *   - RequiresDeviceIdle(true) — only when user is sleeping / inactive
 *   - NetworkType.NOT_REQUIRED — fully offline, no data needed
 *
 * The training job is dispatched as a OneTimeWork request that runs the
 * native C++ training loop through TrainingWorker. It can also be
 * scheduled as periodic work for continuous improvement.
 */
object OvernightTrainingScheduler {

    private const val TAG = "LESS_Scheduler"

    // WorkManager unique work names
    private const val WORK_NAME_ONESHOT = "less_training_oneshot"
    private const val WORK_NAME_PERIODIC = "less_training_periodic"

    /**
     * Schedules a one-time training run for the next time the device
     * is charging and idle.
     */
    fun scheduleTraining(
        context: Context,
        epochs: Int = 5,
        learningRate: Float = 0.001f
    ) {
        val corpusDir = File(context.filesDir, "corpus").absolutePath
        val adapterPath = File(context.filesDir, "models/adapter_latest.bin").absolutePath

        val inputData = workDataOf(
            TrainingWorker.KEY_CORPUS_DIR to corpusDir,
            TrainingWorker.KEY_ADAPTER_PATH to adapterPath,
            TrainingWorker.KEY_EPOCHS to epochs,
            TrainingWorker.KEY_LEARNING_RATE to learningRate
        )

        val constraints = Constraints.Builder()
            .setRequiresCharging(true)
            .setRequiresDeviceIdle(true)
            .setRequiresBatteryNotLow(true)
            .setRequiredNetworkType(NetworkType.NOT_REQUIRED)
            .build()

        val workRequest = OneTimeWorkRequestBuilder<TrainingWorker>()
            .setInputData(inputData)
            .setConstraints(constraints)
            .addTag("less_training")
            .setBackoffCriteria(
                BackoffPolicy.EXPONENTIAL,
                30, TimeUnit.MINUTES
            )
            .build()

        WorkManager.getInstance(context).enqueueUniqueWork(
            WORK_NAME_ONESHOT,
            ExistingWorkPolicy.KEEP,  // don't restart if already queued
            workRequest
        )

        Log.i(TAG, "Training scheduled — will run when device is charging + idle")
    }

    /**
     * Schedules periodic training (every 24 hours) for continuous
     * model improvement as the user records more profiling sessions.
     */
    fun schedulePeriodicTraining(context: Context) {
        val corpusDir = File(context.filesDir, "corpus").absolutePath
        val adapterPath = File(context.filesDir, "models/adapter_latest.bin").absolutePath

        val inputData = workDataOf(
            TrainingWorker.KEY_CORPUS_DIR to corpusDir,
            TrainingWorker.KEY_ADAPTER_PATH to adapterPath,
            TrainingWorker.KEY_EPOCHS to 3,       // fewer epochs for periodic
            TrainingWorker.KEY_LEARNING_RATE to 0.0005f  // lower LR for fine-tuning
        )

        val constraints = Constraints.Builder()
            .setRequiresCharging(true)
            .setRequiresDeviceIdle(true)
            .setRequiresBatteryNotLow(true)
            .setRequiredNetworkType(NetworkType.NOT_REQUIRED)
            .build()

        val workRequest = PeriodicWorkRequestBuilder<TrainingWorker>(
            24, TimeUnit.HOURS,       // repeat interval
            6, TimeUnit.HOURS          // flex interval
        )
            .setInputData(inputData)
            .setConstraints(constraints)
            .addTag("less_training_periodic")
            .build()

        WorkManager.getInstance(context).enqueueUniquePeriodicWork(
            WORK_NAME_PERIODIC,
            ExistingPeriodicWorkPolicy.KEEP,
            workRequest
        )

        Log.i(TAG, "Periodic training scheduled — every 24h when charging + idle")
    }

    /**
     * Cancels all scheduled training work.
     */
    fun cancelAll(context: Context) {
        WorkManager.getInstance(context).cancelAllWorkByTag("less_training")
        WorkManager.getInstance(context).cancelAllWorkByTag("less_training_periodic")
        Log.i(TAG, "All training work cancelled")
    }

    /**
     * Returns the current state of the training work.
     */
    fun getTrainingState(context: Context): LiveDataWorkInfo {
        return LiveDataWorkInfo(
            WorkManager.getInstance(context)
                .getWorkInfosForUniqueWorkLiveData(WORK_NAME_ONESHOT)
        )
    }
}

/**
 * Wrapper for observing training work state in Compose.
 */
class LiveDataWorkInfo(
    val liveData: androidx.lifecycle.LiveData<List<WorkInfo>>
)
