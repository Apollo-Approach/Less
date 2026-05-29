package com.less.audio.ui

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.airbnb.lottie.compose.*
import kotlinx.coroutines.launch

// =============================================================================
// Phase 8 — First Run Onboarding Wizard
// =============================================================================
// Full-screen modal wizard shown on first app launch. Integrates:
//   - Animated welcome with Lottie sound wave
//   - Pipeline explainer (Profile → Train → Suppress)
//   - GDPR privacy consent gate (consolidated from Phase 6 AlertDialog)
//   - Interactive contextual tooltip tour for Profiling Mode
//
// State persisted in SharedPreferences("less_onboarding"):
//   - onboarding_complete: Boolean — skip wizard on subsequent launches
//
// Privacy consent persisted in SharedPreferences("less_privacy"):
//   - profiling_consent_granted: Boolean
//   - profiling_consent_timestamp: Long
// =============================================================================

private val LessIndigo = Color(0xFF6C63FF)
private val LessTeal = Color(0xFF03DAC5)
private val LessDark = Color(0xFF0D0D1A)
private val LessSurface = Color(0xFF1A1A2E)
private val LessSubtext = Color(0xFF888899)
private val LessText = Color(0xFFE0E0E0)
private val LessAmber = Color(0xFFFFB74D)
private val LessRed = Color(0xFFFF6B6B)

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun OnboardingWizard(
    onComplete: () -> Unit
) {
    val context = LocalContext.current
    val pagerState = rememberPagerState(pageCount = { 4 })
    val coroutineScope = rememberCoroutineScope()

    val privacyPrefs = remember {
        context.getSharedPreferences("less_privacy", 0)
    }

    // GDPR consent checkboxes
    var consentDataCollection by remember { mutableStateOf(false) }
    var consentOnDeviceOnly by remember { mutableStateOf(false) }
    var consentAutoDeletion by remember { mutableStateOf(false) }
    val allConsented = consentDataCollection && consentOnDeviceOnly && consentAutoDeletion

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    colors = listOf(LessDark, Color(0xFF12122A), LessDark)
                )
            )
    ) {
        HorizontalPager(
            state = pagerState,
            modifier = Modifier.fillMaxSize(),
            // Prevent swiping past the privacy gate without consent
            userScrollEnabled = pagerState.currentPage != 2
        ) { page ->
            when (page) {
                0 -> WelcomePage()
                1 -> HowItWorksPage()
                2 -> PrivacyGatePage(
                    consentDataCollection = consentDataCollection,
                    consentOnDeviceOnly = consentOnDeviceOnly,
                    consentAutoDeletion = consentAutoDeletion,
                    onToggleDataCollection = { consentDataCollection = it },
                    onToggleOnDeviceOnly = { consentOnDeviceOnly = it },
                    onToggleAutoDeletion = { consentAutoDeletion = it }
                )
                3 -> ProfilingTourPage()
            }
        }

        // Bottom navigation: dot indicators + next/finish button
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 48.dp, start = 24.dp, end = 24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Dot indicators
            Row(
                horizontalArrangement = Arrangement.Center,
                modifier = Modifier.padding(bottom = 24.dp)
            ) {
                repeat(4) { index ->
                    val selected = pagerState.currentPage == index
                    Box(
                        modifier = Modifier
                            .padding(horizontal = 4.dp)
                            .size(if (selected) 10.dp else 8.dp)
                            .clip(CircleShape)
                            .background(
                                if (selected) LessIndigo
                                else LessSubtext.copy(alpha = 0.4f)
                            )
                    )
                }
            }

            // Navigation button
            val isLastPage = pagerState.currentPage == 3
            val isPrivacyPage = pagerState.currentPage == 2
            val canProceed = !isPrivacyPage || allConsented

            Button(
                onClick = {
                    if (isLastPage) {
                        // Persist consent + complete onboarding
                        privacyPrefs.edit()
                            .putBoolean("profiling_consent_granted", true)
                            .putLong("profiling_consent_timestamp",
                                System.currentTimeMillis())
                            .apply()

                        val onboardingPrefs = context.getSharedPreferences(
                            "less_onboarding", 0
                        )
                        onboardingPrefs.edit()
                            .putBoolean("onboarding_complete", true)
                            .apply()

                        onComplete()
                    } else {
                        coroutineScope.launch {
                            pagerState.animateScrollToPage(pagerState.currentPage + 1)
                        }
                    }
                },
                enabled = canProceed,
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isLastPage) LessTeal else LessIndigo,
                    disabledContainerColor = LessSubtext.copy(alpha = 0.3f)
                ),
                shape = RoundedCornerShape(14.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp)
            ) {
                Text(
                    when {
                        isLastPage -> "🚀  Start Using LESS"
                        isPrivacyPage && !allConsented -> "Accept All to Continue"
                        isPrivacyPage -> "I Consent — Continue"
                        else -> "Next"
                    },
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 15.sp
                )
            }

            // Skip on Privacy page — lets user decline gracefully
            if (isPrivacyPage && !allConsented) {
                TextButton(
                    onClick = {
                        // Complete onboarding WITHOUT consent
                        val onboardingPrefs = context.getSharedPreferences(
                            "less_onboarding", 0
                        )
                        onboardingPrefs.edit()
                            .putBoolean("onboarding_complete", true)
                            .apply()
                        onComplete()
                    }
                ) {
                    Text(
                        "Skip — I'll review later",
                        color = LessSubtext,
                        fontSize = 12.sp
                    )
                }
            }
        }
    }
}

// =============================================================================
// Page 1: Welcome
// =============================================================================

@Composable
private fun WelcomePage() {
    val composition by rememberLottieComposition(
        LottieCompositionSpec.Asset("lottie/welcome_wave.json")
    )
    val progress by animateLottieCompositionAsState(
        composition,
        iterations = LottieConstants.IterateForever
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        LottieAnimation(
            composition = composition,
            progress = { progress },
            modifier = Modifier.size(200.dp)
        )

        Spacer(modifier = Modifier.height(32.dp))

        Text(
            "LESS",
            fontSize = 48.sp,
            fontWeight = FontWeight.Bold,
            color = LessIndigo,
            letterSpacing = 12.sp
        )

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            "Low-latency Edge Sound Suppression",
            fontSize = 14.sp,
            color = LessSubtext,
            letterSpacing = 2.sp
        )

        Spacer(modifier = Modifier.height(32.dp))

        Text(
            "Your AI learns the noises in YOUR world\n" +
            "and silences them in real-time.",
            fontSize = 17.sp,
            color = LessText,
            textAlign = TextAlign.Center,
            lineHeight = 26.sp
        )

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            "Personalized • On-Device • Private",
            fontSize = 13.sp,
            color = LessTeal,
            fontWeight = FontWeight.SemiBold,
            letterSpacing = 1.sp
        )

        Spacer(modifier = Modifier.height(80.dp))
    }
}

// =============================================================================
// Page 2: How It Works (3-Stage Pipeline)
// =============================================================================

@Composable
private fun HowItWorksPage() {
    val composition by rememberLottieComposition(
        LottieCompositionSpec.Asset("lottie/pipeline_flow.json")
    )
    val progress by animateLottieCompositionAsState(
        composition,
        iterations = LottieConstants.IterateForever
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            "How LESS Works",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            color = LessText
        )

        Spacer(modifier = Modifier.height(24.dp))

        LottieAnimation(
            composition = composition,
            progress = { progress },
            modifier = Modifier
                .fillMaxWidth()
                .height(120.dp)
        )

        Spacer(modifier = Modifier.height(32.dp))

        PipelineStep(
            number = "1",
            title = "Profile Your World",
            description = "Capture 15-minute audio snapshots of your daily environments " +
                    "(subway, office, coffee shop).",
            color = LessIndigo
        )

        Spacer(modifier = Modifier.height(16.dp))

        PipelineStep(
            number = "2",
            title = "Train Overnight",
            description = "While your phone charges, LESS trains a personal AI model " +
                    "that learns YOUR specific noise patterns.",
            color = LessTeal
        )

        Spacer(modifier = Modifier.height(16.dp))

        PipelineStep(
            number = "3",
            title = "Suppress in Real-Time",
            description = "Audio is processed on the NPU in under 3ms — background " +
                    "noise vanishes while voices stay crystal clear.",
            color = LessIndigo
        )

        Spacer(modifier = Modifier.height(80.dp))
    }
}

@Composable
private fun PipelineStep(
    number: String,
    title: String,
    description: String,
    color: Color
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.Top
    ) {
        // Step number badge
        Box(
            modifier = Modifier
                .size(32.dp)
                .clip(CircleShape)
                .background(color.copy(alpha = 0.2f))
                .border(1.dp, color.copy(alpha = 0.5f), CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Text(
                number,
                color = color,
                fontWeight = FontWeight.Bold,
                fontSize = 14.sp
            )
        }

        Spacer(modifier = Modifier.width(12.dp))

        Column(modifier = Modifier.weight(1f)) {
            Text(
                title,
                color = LessText,
                fontWeight = FontWeight.SemiBold,
                fontSize = 15.sp
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                description,
                color = LessSubtext,
                fontSize = 13.sp,
                lineHeight = 18.sp
            )
        }
    }
}

// =============================================================================
// Page 3: GDPR Privacy Consent Gate
// =============================================================================

@Composable
private fun PrivacyGatePage(
    consentDataCollection: Boolean,
    consentOnDeviceOnly: Boolean,
    consentAutoDeletion: Boolean,
    onToggleDataCollection: (Boolean) -> Unit,
    onToggleOnDeviceOnly: (Boolean) -> Unit,
    onToggleAutoDeletion: (Boolean) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 28.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("🔒", fontSize = 40.sp)

        Spacer(modifier = Modifier.height(12.dp))

        Text(
            "Your Privacy Matters",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            color = LessText
        )

        Text(
            "Please review and consent to each item",
            fontSize = 13.sp,
            color = LessSubtext
        )

        Spacer(modifier = Modifier.height(24.dp))

        ConsentCheckbox(
            checked = consentDataCollection,
            onCheckedChange = onToggleDataCollection,
            emoji = "📱",
            title = "Audio Data Collection",
            description = "Raw environmental audio is captured from your Meta Ray-Ban " +
                    "microphones during manually-initiated 15-minute profiling sessions."
        )

        Spacer(modifier = Modifier.height(12.dp))

        ConsentCheckbox(
            checked = consentOnDeviceOnly,
            onCheckedChange = onToggleOnDeviceOnly,
            emoji = "🔐",
            title = "On-Device Only — AES-256-GCM",
            description = "All audio is encrypted with AES-256-GCM backed by the Android " +
                    "hardware Keystore. Data is NEVER transmitted to any server, cloud, " +
                    "or third party."
        )

        Spacer(modifier = Modifier.height(12.dp))

        ConsentCheckbox(
            checked = consentAutoDeletion,
            onCheckedChange = onToggleAutoDeletion,
            emoji = "🗑️",
            title = "Automatic Deletion After Training",
            description = "Raw audio is used solely to train your personal noise model. " +
                    "After training completes, all raw audio is permanently deleted. " +
                    "You can revoke consent and delete data at any time."
        )

        Spacer(modifier = Modifier.height(16.dp))

        // Rights reminder
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = LessSurface.copy(alpha = 0.7f))
        ) {
            Row(modifier = Modifier.padding(12.dp)) {
                Text("↩️", fontSize = 16.sp)
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    "You can revoke consent at any time from Settings. " +
                    "This will permanently delete all stored audio data.",
                    color = LessSubtext,
                    fontSize = 11.sp,
                    lineHeight = 16.sp
                )
            }
        }

        Spacer(modifier = Modifier.height(80.dp))
    }
}

@Composable
private fun ConsentCheckbox(
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
    emoji: String,
    title: String,
    description: String
) {
    val borderColor by animateColorAsState(
        if (checked) LessTeal.copy(alpha = 0.5f) else LessSurface,
        label = "consent_border"
    )

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, borderColor, RoundedCornerShape(12.dp)),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (checked) Color(0xFF0D1F1A) else LessSurface
        ),
        onClick = { onCheckedChange(!checked) }
    ) {
        Row(
            modifier = Modifier.padding(14.dp),
            verticalAlignment = Alignment.Top
        ) {
            Checkbox(
                checked = checked,
                onCheckedChange = onCheckedChange,
                colors = CheckboxDefaults.colors(
                    checkedColor = LessTeal,
                    uncheckedColor = LessSubtext
                ),
                modifier = Modifier.size(20.dp)
            )
            Spacer(modifier = Modifier.width(10.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    "$emoji  $title",
                    color = if (checked) LessTeal else LessText,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 13.sp
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    description,
                    color = LessSubtext,
                    fontSize = 11.sp,
                    lineHeight = 16.sp
                )
            }
        }
    }
}

// =============================================================================
// Page 4: Profiling Tour (Interactive Contextual Tooltips)
// =============================================================================

@Composable
private fun ProfilingTourPage() {
    // Animated tooltip appearance sequence
    var showTooltip1 by remember { mutableStateOf(false) }
    var showTooltip2 by remember { mutableStateOf(false) }
    var showTooltip3 by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        kotlinx.coroutines.delay(500)
        showTooltip1 = true
        kotlinx.coroutines.delay(800)
        showTooltip2 = true
        kotlinx.coroutines.delay(800)
        showTooltip3 = true
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 28.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            "Your First Profiling Session",
            fontSize = 22.sp,
            fontWeight = FontWeight.Bold,
            color = LessText,
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            "Here's how to capture your first audio profile:",
            fontSize = 14.sp,
            color = LessSubtext,
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(32.dp))

        // Simulated dashboard card with tooltip overlays
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = LessSurface)
        ) {
            Column(modifier = Modifier.padding(20.dp)) {
                // Simulated profiling button
                AnimatedVisibility(
                    visible = showTooltip1,
                    enter = fadeIn() + slideInVertically { -20 }
                ) {
                    TooltipCard(
                        number = "1",
                        text = "Tap this button when you're in a noisy environment " +
                                "you want LESS to learn (subway, coffee shop, etc.)",
                        color = LessIndigo
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                Button(
                    onClick = { },
                    colors = ButtonDefaults.buttonColors(containerColor = LessTeal),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.fillMaxWidth(),
                    enabled = false
                ) {
                    Text("🎙  Start 15-min Capture", fontWeight = FontWeight.SemiBold)
                }

                Spacer(modifier = Modifier.height(16.dp))

                AnimatedVisibility(
                    visible = showTooltip2,
                    enter = fadeIn() + slideInVertically { -20 }
                ) {
                    TooltipCard(
                        number = "2",
                        text = "LESS records for exactly 15 minutes, then stops " +
                                "automatically. All audio is encrypted on-device.",
                        color = LessTeal
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))

                AnimatedVisibility(
                    visible = showTooltip3,
                    enter = fadeIn() + slideInVertically { -20 }
                ) {
                    TooltipCard(
                        number = "3",
                        text = "After 2-3 profiling sessions, schedule overnight " +
                                "training. Your personal AI model will be ready by morning!",
                        color = LessAmber
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(80.dp))
    }
}

@Composable
private fun TooltipCard(
    number: String,
    text: String,
    color: Color
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, color.copy(alpha = 0.3f), RoundedCornerShape(10.dp)),
        shape = RoundedCornerShape(10.dp),
        colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.08f))
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.Top
        ) {
            Box(
                modifier = Modifier
                    .size(22.dp)
                    .clip(CircleShape)
                    .background(color),
                contentAlignment = Alignment.Center
            ) {
                Text(number, color = Color.White, fontSize = 11.sp,
                    fontWeight = FontWeight.Bold)
            }
            Spacer(modifier = Modifier.width(10.dp))
            Text(
                text,
                color = LessText,
                fontSize = 12.sp,
                lineHeight = 17.sp,
                modifier = Modifier.weight(1f)
            )
        }
    }
}
