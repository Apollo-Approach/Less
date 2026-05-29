package com.less.audio.telemetry

import android.app.Application
import android.os.Build
import android.os.ProfilingManager
import android.util.Log
import io.opentelemetry.android.OpenTelemetryRum
import io.opentelemetry.api.common.AttributeKey
import io.opentelemetry.api.common.Attributes
import io.opentelemetry.api.trace.Tracer
import io.opentelemetry.exporter.otlp.http.trace.OtlpHttpSpanExporter
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor

object TelemetryManager {
    private const val TAG = "TelemetryManager"
    private var rum: OpenTelemetryRum? = null
    private var tracer: Tracer? = null

    fun initialize(application: Application) {
        try {
            // Configure local OTLP endpoint for testing (10.0.2.2 reaches localhost from Android emulator)
            val otlpExporter = OtlpHttpSpanExporter.builder()
                .setEndpoint("http://10.0.2.2:4318/v1/traces")
                .build()

            rum = OpenTelemetryRum.builder(application)
                .addTracerProviderCustomizer { builder, _ ->
                    builder.addSpanProcessor(BatchSpanProcessor.builder(otlpExporter).build())
                }
                .build()
            
            tracer = rum?.openTelemetry?.getTracer("LESS-Audio-Tracer")
            Log.i(TAG, "OpenTelemetry configured with offline buffering.")
            
            initializeProfilingManager(application)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize OpenTelemetry", e)
        }
    }

    private fun initializeProfilingManager(application: Application) {
        // Android 16 (API 36) introduces ProfilingManager for system-triggered profiling
        if (Build.VERSION.SDK_INT >= 36) {
            try {
                val profilingManager = application.getSystemService(ProfilingManager::class.java)
                if (profilingManager != null) {
                    profilingManager.registerForAllProfilingResults(application.mainExecutor) { result ->
                        Log.i(TAG, "ProfilingManager captured trace: ${result.tag}")
                        recordAnomaly("ProfilingTraceCaptured", "tag" to (result.tag ?: "unknown"))
                        // Trace files can be extracted and sent to the telemetry backend later
                    }
                    Log.i(TAG, "Android 16 ProfilingManager hooked successfully.")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to hook ProfilingManager", e)
            }
        }
    }

    fun recordError(domain: String, message: String, exception: Throwable? = null) {
        Log.e(TAG, "[$domain] $message", exception)
        
        val attributesBuilder = Attributes.builder()
            .put(AttributeKey.stringKey("domain"), domain)
            .put(AttributeKey.stringKey("message"), message)
            
        exception?.let {
            attributesBuilder.put(AttributeKey.stringKey("exception.type"), it.javaClass.simpleName)
            attributesBuilder.put(AttributeKey.stringKey("exception.message"), it.message ?: "Unknown")
        }
        
        tracer?.spanBuilder("mechanical_error")
            ?.setAttribute("error", true)
            ?.setAllAttributes(attributesBuilder.build())
            ?.startSpan()
            ?.end()
    }

    fun recordAnomaly(type: String, vararg metadata: Pair<String, String>) {
        Log.w(TAG, "Anomaly detected: $type")
        val attributesBuilder = Attributes.builder()
            .put(AttributeKey.stringKey("anomaly.type"), type)
            
        metadata.forEach { (k, v) ->
            attributesBuilder.put(AttributeKey.stringKey(k), v)
        }
        
        tracer?.spanBuilder("anomaly_$type")
            ?.setAllAttributes(attributesBuilder.build())
            ?.startSpan()
            ?.end()
    }
}
