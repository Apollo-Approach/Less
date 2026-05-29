package com.less.audio.ui

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// =============================================================================
// Phase 8 — Environmental Acoustic Presets
// =============================================================================
// Quick-select horizontal chip row for common noise environments.
// Each preset maps to a specific suppression level that adjusts how
// aggressively the SPSA-trained adapter weights filter noise.
//
// Presets reset to "Custom" on each app launch because the user's
// acoustic environment changes drastically between sessions.
// =============================================================================

data class AcousticPreset(
    val name: String,
    val emoji: String,
    val suppression: Float,
    val description: String
)

val ACOUSTIC_PRESETS = listOf(
    AcousticPreset("Office", "🏢", 0.50f, "Moderate — keyboards & AC"),
    AcousticPreset("Coffee Shop", "☕", 0.70f, "Balanced — ambient buzz"),
    AcousticPreset("Subway", "🚇", 0.90f, "Aggressive — max gate"),
    AcousticPreset("Focus", "🎧", 0.95f, "Near-silent — deep work"),
    AcousticPreset("Custom", "🌊", -1f, "Manual slider control")
)

@Composable
fun AcousticPresetsRow(
    currentSuppression: Float,
    onPresetSelected: (AcousticPreset) -> Unit,
    modifier: Modifier = Modifier
) {
    // Track which preset is selected (null = Custom / manual slider)
    var selectedPreset by remember { mutableStateOf<String?>(null) }

    Column(modifier = modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                "Environment",
                color = Color(0xFFAABBCC),
                fontSize = 14.sp,
                fontWeight = FontWeight.Medium
            )
            if (selectedPreset != null && selectedPreset != "Custom") {
                Text(
                    "${(ACOUSTIC_PRESETS.find { it.name == selectedPreset }?.suppression
                        ?: 0f) * 100}% suppression",
                    color = Color(0xFF6C63FF),
                    fontSize = 11.sp,
                    fontWeight = FontWeight.SemiBold
                )
            }
        }

        Spacer(modifier = Modifier.height(10.dp))

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            ACOUSTIC_PRESETS.forEach { preset ->
                val isSelected = selectedPreset == preset.name
                val backgroundColor by animateColorAsState(
                    if (isSelected) Color(0xFF03DAC5).copy(alpha = 0.15f)
                    else Color(0xFF1A1A2E),
                    animationSpec = tween(200),
                    label = "preset_bg_${preset.name}"
                )
                val borderColor by animateColorAsState(
                    if (isSelected) Color(0xFF03DAC5).copy(alpha = 0.6f)
                    else Color(0xFF2A2A4A),
                    animationSpec = tween(200),
                    label = "preset_border_${preset.name}"
                )

                Card(
                    onClick = {
                        selectedPreset = preset.name
                        onPresetSelected(preset)
                    },
                    modifier = Modifier
                        .border(1.dp, borderColor, RoundedCornerShape(12.dp)),
                    shape = RoundedCornerShape(12.dp),
                    colors = CardDefaults.cardColors(containerColor = backgroundColor)
                ) {
                    Column(
                        modifier = Modifier
                            .padding(horizontal = 16.dp, vertical = 10.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(preset.emoji, fontSize = 20.sp)
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            preset.name,
                            color = if (isSelected) Color(0xFF03DAC5)
                                    else Color(0xFFE0E0E0),
                            fontSize = 12.sp,
                            fontWeight = if (isSelected) FontWeight.Bold
                                         else FontWeight.Normal
                        )
                        Text(
                            preset.description,
                            color = Color(0xFF666677),
                            fontSize = 9.sp
                        )
                    }
                }
            }
        }
    }
}
