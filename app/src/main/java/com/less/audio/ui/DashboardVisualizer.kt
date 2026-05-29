package com.less.audio.ui

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.less.audio.NativeAudioBridge
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive

// =============================================================================
// Phase 8 — Explainability Dashboard: Dual Waveform Visualizer
// =============================================================================
// Real-time Canvas composable showing two audio waveforms:
//   - Top (amber): Raw incoming audio from the mic array
//   - Bottom (teal): Cleaned output after AI denoising
//
// This serves as a "trust engine" — the user can SEE the AI working.
//
// Data source: NativeAudioBridge.nativeGetAudioLevels() returns a FloatArray:
//   [0]    = input RMS level (0.0–1.0)
//   [1]    = output RMS level (0.0–1.0)
//   [2..65]  = input waveform snapshot (64 samples, -1.0 to 1.0)
//   [66..129] = output waveform snapshot (64 samples, -1.0 to 1.0)
//
// Rendering: 30fps polling via LaunchedEffect. Lock-free double-buffer
// in the native layer ensures the audio thread is never blocked.
// =============================================================================

private val WaveAmber = Color(0xFFFFB74D)
private val WaveTeal = Color(0xFF03DAC5)
private val WaveAmberDim = Color(0x33FFB74D)
private val WaveTealDim = Color(0x3303DAC5)
private val LabelDim = Color(0xFF666677)
private val SurfaceDark = Color(0xFF1A1A2E)
private val BackgroundDark = Color(0xFF0D0D1A)

private const val WAVEFORM_SAMPLES = 128
private const val POLL_INTERVAL_MS = 33L  // ~30fps

@Composable
fun DashboardVisualizer(
    isEngineRunning: Boolean,
    modifier: Modifier = Modifier
) {
    // Waveform sample buffers (updated at 30fps)
    var inputRms by remember { mutableFloatStateOf(0f) }
    var outputRms by remember { mutableFloatStateOf(0f) }
    var inputWaveform by remember { mutableStateOf(FloatArray(WAVEFORM_SAMPLES)) }
    var outputWaveform by remember { mutableStateOf(FloatArray(WAVEFORM_SAMPLES)) }

    // Smooth RMS animation for the level indicators
    val animatedInputRms by animateFloatAsState(
        targetValue = inputRms,
        animationSpec = tween(100, easing = EaseOutCubic),
        label = "input_rms"
    )
    val animatedOutputRms by animateFloatAsState(
        targetValue = outputRms,
        animationSpec = tween(100, easing = EaseOutCubic),
        label = "output_rms"
    )

    // Idle wave animation when engine is off
    val infiniteTransition = rememberInfiniteTransition(label = "idle_wave")
    val idlePhase by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * Math.PI.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(4000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "idle_phase"
    )

    // Poll audio levels at ~30fps when engine is running
    LaunchedEffect(isEngineRunning) {
        if (!isEngineRunning) {
            inputRms = 0f
            outputRms = 0f
            inputWaveform = FloatArray(WAVEFORM_SAMPLES)
            outputWaveform = FloatArray(WAVEFORM_SAMPLES)
            return@LaunchedEffect
        }

        while (isActive) {
            val levels = NativeAudioBridge.nativeGetAudioLevels()
            if (levels != null && levels.size >= 2 + WAVEFORM_SAMPLES * 2) {
                inputRms = levels[0].coerceIn(0f, 1f)
                outputRms = levels[1].coerceIn(0f, 1f)
                inputWaveform = levels.sliceArray(2 until 2 + WAVEFORM_SAMPLES)
                outputWaveform = levels.sliceArray(
                    2 + WAVEFORM_SAMPLES until 2 + WAVEFORM_SAMPLES * 2
                )
            }
            delay(POLL_INTERVAL_MS)
        }
    }

    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = SurfaceDark)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            // Header row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    "Audio Pipeline",
                    color = Color(0xFFAABBCC),
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium
                )
                Text(
                    if (isEngineRunning) "● LIVE" else "○ IDLE",
                    color = if (isEngineRunning) WaveTeal else LabelDim,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.SemiBold
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Input waveform
            WaveformRow(
                label = "RAW INPUT",
                rms = animatedInputRms,
                waveColor = WaveAmber,
                glowColor = WaveAmberDim,
                waveform = inputWaveform,
                isActive = isEngineRunning,
                idlePhase = idlePhase,
                modifier = Modifier.height(52.dp)
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Divider with arrow
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .height(1.dp)
                        .background(LabelDim.copy(alpha = 0.2f))
                )
                Text(
                    " ▼ AI DENOISING ▼ ",
                    color = LabelDim,
                    fontSize = 9.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.sp
                )
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .height(1.dp)
                        .background(LabelDim.copy(alpha = 0.2f))
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Output waveform
            WaveformRow(
                label = "CLEANED OUTPUT",
                rms = animatedOutputRms,
                waveColor = WaveTeal,
                glowColor = WaveTealDim,
                waveform = outputWaveform,
                isActive = isEngineRunning,
                idlePhase = idlePhase,
                modifier = Modifier.height(52.dp)
            )

            // Noise reduction indicator
            if (isEngineRunning && inputRms > 0.01f) {
                Spacer(modifier = Modifier.height(8.dp))
                val reduction = if (inputRms > 0f) {
                    ((1f - (outputRms / inputRms).coerceIn(0f, 1f)) * 100).toInt()
                } else 0

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.Center
                ) {
                    Text(
                        "Noise Reduction: ~${reduction}%",
                        color = WaveTeal,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }
        }
    }
}

@Composable
private fun WaveformRow(
    label: String,
    rms: Float,
    waveColor: Color,
    glowColor: Color,
    waveform: FloatArray,
    isActive: Boolean,
    idlePhase: Float,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Label + RMS bar
        Column(
            modifier = Modifier.width(80.dp),
            horizontalAlignment = Alignment.Start
        ) {
            Text(
                label,
                color = LabelDim,
                fontSize = 9.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 0.5.sp
            )
            Spacer(modifier = Modifier.height(4.dp))
            // RMS level bar
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(3.dp)
                    .background(
                        BackgroundDark,
                        RoundedCornerShape(2.dp)
                    )
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth(fraction = rms.coerceIn(0f, 1f))
                        .fillMaxHeight()
                        .background(waveColor, RoundedCornerShape(2.dp))
                )
            }
        }

        Spacer(modifier = Modifier.width(12.dp))

        // Waveform canvas
        Canvas(
            modifier = modifier
                .weight(1f)
                .background(BackgroundDark, RoundedCornerShape(8.dp))
        ) {
            if (isActive) {
                drawWaveform(waveform, waveColor, glowColor)
            } else {
                drawIdleWave(idlePhase, waveColor.copy(alpha = 0.3f))
            }
        }
    }
}

private fun DrawScope.drawWaveform(
    samples: FloatArray,
    color: Color,
    glowColor: Color
) {
    val w = size.width
    val h = size.height
    val centerY = h / 2f
    val amplitude = h * 0.4f

    if (samples.isEmpty()) return

    val path = Path()
    val stepX = w / (samples.size - 1).coerceAtLeast(1).toFloat()

    // Build smooth curve using quadratic interpolation
    path.moveTo(0f, centerY + samples[0] * amplitude)

    for (i in 1 until samples.size) {
        val x = i * stepX
        val prevX = (i - 1) * stepX
        val midX = (prevX + x) / 2f
        val y = centerY + samples[i].coerceIn(-1f, 1f) * amplitude
        val prevY = centerY + samples[i - 1].coerceIn(-1f, 1f) * amplitude

        path.quadraticBezierTo(prevX, prevY, midX, (prevY + y) / 2f)
    }

    // Draw glow (thick, low-alpha)
    drawPath(
        path = path,
        color = glowColor,
        style = Stroke(width = 6f, cap = StrokeCap.Round)
    )

    // Draw main waveform
    drawPath(
        path = path,
        color = color,
        style = Stroke(width = 2f, cap = StrokeCap.Round)
    )

    // Center baseline
    drawLine(
        color = color.copy(alpha = 0.15f),
        start = Offset(0f, centerY),
        end = Offset(w, centerY),
        strokeWidth = 1f
    )
}

private fun DrawScope.drawIdleWave(phase: Float, color: Color) {
    val w = size.width
    val h = size.height
    val centerY = h / 2f

    val path = Path()
    val steps = 60

    path.moveTo(0f, centerY)

    for (i in 0..steps) {
        val x = (i.toFloat() / steps) * w
        val y = centerY + kotlin.math.sin(
            (i.toFloat() / steps) * 4f * Math.PI.toFloat() + phase
        ) * h * 0.05f

        if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
    }

    drawPath(
        path = path,
        color = color,
        style = Stroke(width = 1.5f, cap = StrokeCap.Round)
    )

    drawLine(
        color = color.copy(alpha = 0.1f),
        start = Offset(0f, centerY),
        end = Offset(w, centerY),
        strokeWidth = 1f
    )
}
