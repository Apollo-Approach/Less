package com.less.audio.profiling

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKeys

/**
 * ProfilingSessionManager — Tracks cumulative foreground service usage
 * to comply with Android 15's 6-hour/24h limit for mediaProcessing FGS.
 *
 * Android 15 enforces a strict 6-hour cumulative limit per 24-hour rolling
 * window for foreground services with types like mediaProcessing and dataSync.
 * Exceeding this limit causes the system to force-stop the service.
 *
 * This manager:
 *   1. Logs the start/end time of each profiling session
 *   2. Calculates remaining budget in a rolling 24-hour window
 *   3. Refuses to start new sessions if the budget is exhausted
 *   4. Persists state in EncryptedSharedPreferences for security
 */
class ProfilingSessionManager(context: Context) {

    companion object {
        private const val TAG = "LESS_SessionMgr"
        private const val PREFS_NAME = "less_profiling_sessions"

        // Android 15 FGS limit
        private const val MAX_FGS_MINUTES_PER_24H = 360  // 6 hours
        private const val ROLLING_WINDOW_MS = 24 * 60 * 60 * 1000L  // 24 hours

        // Safety margin — stop 15 minutes before the hard limit
        private const val SAFETY_MARGIN_MINUTES = 15

        // Maximum single session duration
        const val MAX_SESSION_MINUTES = 15

        // Minimum corpus for useful training
        const val MIN_USEFUL_CORPUS_MINUTES = 5

        // Keys for SharedPreferences
        private const val KEY_SESSION_LOG = "session_log"
        private const val KEY_SESSION_COUNT = "session_count"
    }

    private val prefs: SharedPreferences = try {
        val masterKeyAlias = MasterKeys.getOrCreate(MasterKeys.AES256_GCM_SPEC)
        EncryptedSharedPreferences.create(
            PREFS_NAME,
            masterKeyAlias,
            context,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        )
    } catch (e: Exception) {
        // Fallback to regular SharedPreferences if encryption fails
        Log.w(TAG, "EncryptedSharedPreferences failed — using regular prefs", e)
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    /**
     * A single profiling session record.
     * Stored as "startMs,endMs" pairs in a semicolon-delimited string.
     */
    data class SessionRecord(
        val startTimeMs: Long,
        val endTimeMs: Long
    ) {
        val durationMinutes: Int
            get() = ((endTimeMs - startTimeMs) / 60_000).toInt()
    }

    // =========================================================================
    // Budget Calculation
    // =========================================================================

    /**
     * Returns true if a new profiling session can be started without
     * exceeding the Android 15 FGS time limit.
     */
    fun canStartSession(): Boolean {
        val remaining = remainingBudgetMinutes()
        val canStart = remaining >= MAX_SESSION_MINUTES
        Log.i(TAG, "canStartSession: remaining=${remaining}min, " +
                "needed=${MAX_SESSION_MINUTES}min, result=$canStart")
        return canStart
    }

    /**
     * Returns the remaining FGS budget in minutes for the current
     * rolling 24-hour window.
     */
    fun remainingBudgetMinutes(): Int {
        val sessions = getRecentSessions()
        val usedMinutes = sessions.sumOf { it.durationMinutes }
        val remaining = MAX_FGS_MINUTES_PER_24H - SAFETY_MARGIN_MINUTES - usedMinutes
        return remaining.coerceAtLeast(0)
    }

    /**
     * Returns sessions within the rolling 24-hour window.
     */
    fun getRecentSessions(): List<SessionRecord> {
        val now = System.currentTimeMillis()
        val cutoff = now - ROLLING_WINDOW_MS
        return getAllSessions().filter { it.startTimeMs >= cutoff }
    }

    /**
     * Returns the total recorded corpus duration across all sessions, in minutes.
     */
    fun totalCorpusMinutes(): Int {
        return getAllSessions().sumOf { it.durationMinutes }
    }

    /**
     * Returns true if enough corpus data exists for a training run.
     */
    fun hasMinimumCorpus(): Boolean {
        return totalCorpusMinutes() >= MIN_USEFUL_CORPUS_MINUTES
    }

    // =========================================================================
    // Session Logging
    // =========================================================================

    /**
     * Records the completion of a profiling session.
     * Called by ProfilingService when a capture burst finishes.
     */
    fun logSession(startTimeMs: Long, endTimeMs: Long) {
        val session = SessionRecord(startTimeMs, endTimeMs)
        val sessions = getAllSessionsRaw().toMutableList()
        sessions.add("${startTimeMs},${endTimeMs}")

        // Prune sessions older than 48 hours to prevent unbounded growth
        val cutoff = System.currentTimeMillis() - (48 * 60 * 60 * 1000L)
        val pruned = sessions.filter { entry ->
            val startMs = entry.split(",").firstOrNull()?.toLongOrNull() ?: 0L
            startMs >= cutoff
        }

        val count = prefs.getInt(KEY_SESSION_COUNT, 0)
        prefs.edit()
            .putString(KEY_SESSION_LOG, pruned.joinToString(";"))
            .putInt(KEY_SESSION_COUNT, count + 1)
            .apply()

        Log.i(TAG, "Session logged: ${session.durationMinutes}min " +
                "(total in window: ${getRecentSessions().sumOf { it.durationMinutes }}min)")
    }

    // =========================================================================
    // Internal
    // =========================================================================

    private fun getAllSessions(): List<SessionRecord> {
        return getAllSessionsRaw().mapNotNull { entry ->
            val parts = entry.split(",")
            if (parts.size == 2) {
                val start = parts[0].toLongOrNull() ?: return@mapNotNull null
                val end = parts[1].toLongOrNull() ?: return@mapNotNull null
                SessionRecord(start, end)
            } else null
        }
    }

    private fun getAllSessionsRaw(): List<String> {
        val raw = prefs.getString(KEY_SESSION_LOG, "") ?: ""
        return if (raw.isBlank()) emptyList() else raw.split(";")
    }

    /**
     * Clears all session history. Use for testing only.
     */
    fun clearHistory() {
        prefs.edit()
            .remove(KEY_SESSION_LOG)
            .putInt(KEY_SESSION_COUNT, 0)
            .apply()
        Log.i(TAG, "Session history cleared")
    }
}
