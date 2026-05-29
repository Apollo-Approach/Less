package com.less.audio

import android.Manifest
import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope
import com.less.audio.debug.MockGlassesController
import com.less.audio.profiling.ProfilingService
import com.less.audio.profiling.ProfilingSessionManager
import com.less.audio.training.OvernightTrainingScheduler
import com.less.audio.ui.AcousticPresetsRow
import com.less.audio.ui.DashboardVisualizer
import com.less.audio.ui.OnboardingWizard
import com.meta.wearable.dat.core.Wearables
import android.net.Uri
import com.meta.wearable.dat.core.session.SessionState
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import androidx.work.Constraints
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import com.less.audio.ml.PersonalizationWorker
import java.util.concurrent.TimeUnit
import androidx.compose.runtime.collectAsState

/**
 * MainActivity — LESS Control Dashboard
 *
 * Jetpack Compose UI providing:
 *   - Meta Wearables SDK registration flow
 *   - Real-time engine status and latency display
 *   - Suppression level slider
 *   - Bluetooth connection indicator
 *   - Session state indicator (don/doff/fold)
 *   - Profiling Mode trigger with live countdown
 *   - Training status card with epoch/loss tracking
 *   - Debug panel for MockDeviceKit simulation
 */
class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "LESS_Main"
    }

    private lateinit var btRouter: BluetoothAudioRouter
    private lateinit var sessionManager: ProfilingSessionManager

    // Debug: Mock glasses controller (only used in debug builds)
    private var mockController: MockGlassesController? = null

    // Registration state tracking
    private val _registrationState = mutableStateOf("NOT_REGISTERED")
    private val _isRegistered = mutableStateOf(false)

    // Permission launcher
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        val allGranted = results.all { it.value }
        if (allGranted) {
            Log.i(TAG, "All permissions granted")
        } else {
            Log.w(TAG, "Some permissions denied: ${results.filter { !it.value }.keys}")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        btRouter = BluetoothAudioRouter(this)
        sessionManager = ProfilingSessionManager(this)

        // Initialize mock controller for debug builds
        if (BuildConfig.DEBUG) {
            mockController = MockGlassesController(this)
        }

        // Request permissions
        permissionLauncher.launch(arrayOf(
            Manifest.permission.BLUETOOTH_CONNECT,
            Manifest.permission.BLUETOOTH_SCAN,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.POST_NOTIFICATIONS,
            Manifest.permission.CAMERA
        ))

        // Register BT lifecycle listener
        btRouter.registerConnectionListener()

        // Initialize native engine
        NativeAudioBridge.createEngine()

        // Phase 5: Configure model path and QNN lib directory from
        // centralized LessApplication helpers. These MUST be set
        // before startEngine() for the delegate chain to work.
        val modelPath = LessApplication.getModelPath(this)
        val modelFile = java.io.File(modelPath)
        if (modelFile.exists()) {
            NativeAudioBridge.setModelPath(modelPath)
        } else {
            Log.w(TAG, "Model not found at $modelPath — using spectral gate")
        }

        // Set QNN lib directory so the C++ dlopen() can find
        // libQnnTFLiteDelegate.so and libQnnHtp.so
        NativeAudioBridge.setQnnLibDir(LessApplication.getQnnLibDir(this))

        // =====================================================================
        // Meta Wearables SDK — Registration State Observer
        // =====================================================================
        // Observe registration state to know when the app is authenticated
        // with the Meta AI companion app. Audio processing should NOT begin
        // until registration is complete (state == REGISTERED).
        lifecycleScope.launch {
            try {
                Wearables.registrationState.collectLatest { state ->
                    val stateName = state.toString()
                    Log.i(TAG, "Registration state: $stateName")
                    _registrationState.value = stateName

                    // Check if registration is complete
                    // The state name contains "REGISTERED" when successful
                    if (stateName.contains("REGISTERED", ignoreCase = true) &&
                        !stateName.contains("NOT_REGISTERED", ignoreCase = true)) {
                        _isRegistered.value = true
                        Log.i(TAG, "✓ App registered with Meta AI companion")

                        // TODO: Extract deviceId from registration result
                        // and start session observer:
                        // btRouter.startSessionObserver(deviceId)
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Registration state observation failed: ${e.message}")
                Log.w(TAG, "App will operate in standalone mode (no SDK registration)")
            }
        }

        // Phase 6: Schedule TinyMusician personalizer
        schedulePersonalizationWorker()

        setContent {
            LessTheme {
                // Phase 8: Check if onboarding has been completed
                val onboardingPrefs = remember {
                    getSharedPreferences("less_onboarding", 0)
                }
                var onboardingComplete by remember {
                    mutableStateOf(
                        onboardingPrefs.getBoolean("onboarding_complete", false)
                    )
                }

                if (!onboardingComplete) {
                    OnboardingWizard(
                        onComplete = { onboardingComplete = true }
                    )
                } else {
                    LessDashboard(
                        btRouter = btRouter,
                        sessionManager = sessionManager,
                        registrationState = _registrationState.value,
                        isRegistered = _isRegistered.value,
                        mockController = mockController,
                        onStartRegistration = {
                            startWearablesRegistration()
                        },
                        onStartEngine = {
                            NativeAudioBridge.startEngine()
                        },
                        onStopEngine = {
                            NativeAudioBridge.stopEngine()
                        },
                        onSuppressionChanged = { level ->
                            NativeAudioBridge.setSuppressionLevel(level)
                        },
                        onStartProfiling = {
                            val intent = Intent(this, ProfilingService::class.java).apply {
                                action = ProfilingService.ACTION_START
                            }
                            startForegroundService(intent)
                        },
                        onStopProfiling = {
                            val intent = Intent(this, ProfilingService::class.java).apply {
                                action = ProfilingService.ACTION_STOP
                            }
                            startService(intent)
                        },
                        onScheduleTraining = {
                            OvernightTrainingScheduler.scheduleTraining(this)
                        },
                        onScanBluetooth = {
                            btRouter.startScan()
                        },
                        onStartVisionMusic = {
                            val intent = Intent(this@MainActivity, VisionMusicService::class.java).apply {
                                action = VisionMusicService.ACTION_START
                            }
                            startForegroundService(intent)
                        },
                        onStopVisionMusic = {
                            val intent = Intent(this@MainActivity, VisionMusicService::class.java).apply {
                                action = VisionMusicService.ACTION_STOP
                            }
                            startService(intent)
                        }
                    )
                }
            }
        }
    }

    /**
     * Initiate the Wearables SDK registration flow.
     * This deeplinks to the Meta AI companion app for user confirmation.
     * The companion app must be in Developer Mode for development builds.
     */
    private fun startWearablesRegistration() {
        try {
            Wearables.startRegistration(this)
            Log.i(TAG, "Registration flow initiated — redirecting to Meta AI app")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start registration: ${e.message}")
        }
    }

    private fun schedulePersonalizationWorker() {
        Log.i(TAG, "Scheduling TinyMusician personalization worker")
        val constraints = Constraints.Builder()
            .setRequiresCharging(true)
            .setRequiresDeviceIdle(true)
            .setRequiredNetworkType(NetworkType.NOT_REQUIRED)
            .build()
            
        val workRequest = PeriodicWorkRequestBuilder<PersonalizationWorker>(
            24, TimeUnit.HOURS,
            6, TimeUnit.HOURS
        )
            .setConstraints(constraints)
            .addTag("tinymusician_personalization")
            .build()
            
        WorkManager.getInstance(this).enqueueUniquePeriodicWork(
            "tinymusician_personalization_periodic",
            ExistingPeriodicWorkPolicy.KEEP,
            workRequest
        )
    }

    override fun onDestroy() {
        NativeAudioBridge.destroyEngine()
        btRouter.destroy()
        mockController?.destroy()
        super.onDestroy()
    }
}

// =============================================================================
// Theme
// =============================================================================

@Composable
fun LessTheme(content: @Composable () -> Unit) {
    val darkColors = darkColorScheme(
        primary = Color(0xFF6C63FF),
        secondary = Color(0xFF03DAC5),
        background = Color(0xFF0D0D1A),
        surface = Color(0xFF1A1A2E),
        onPrimary = Color.White,
        onSecondary = Color.Black,
        onBackground = Color(0xFFE0E0E0),
        onSurface = Color(0xFFE0E0E0),
        error = Color(0xFFFF6B6B)
    )
    MaterialTheme(colorScheme = darkColors, content = content)
}

// =============================================================================
// Dashboard
// =============================================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LessDashboard(
    btRouter: BluetoothAudioRouter,
    sessionManager: ProfilingSessionManager,
    registrationState: String,
    isRegistered: Boolean,
    mockController: MockGlassesController?,
    onStartRegistration: () -> Unit,
    onStartEngine: () -> Unit,
    onStopEngine: () -> Unit,
    onSuppressionChanged: (Float) -> Unit,
    onStartProfiling: () -> Unit,
    onStopProfiling: () -> Unit,
    onScheduleTraining: () -> Unit,
    onScanBluetooth: () -> Unit,
    onStartVisionMusic: () -> Unit,
    onStopVisionMusic: () -> Unit
) {
    val coroutineScope = rememberCoroutineScope()
    var isEngineRunning by remember { mutableStateOf(false) }
    var suppressionLevel by remember { mutableFloatStateOf(0.7f) }
    var latencyMs by remember { mutableDoubleStateOf(0.0) }
    var isConnected by remember { mutableStateOf(false) }
    var isProfiling by remember { mutableStateOf(false) }
    var sessionState by remember { mutableStateOf("STOPPED") }
    var activeBackend by remember { mutableStateOf("Not initialized") }
    val context = LocalContext.current
    val gemmaDownloader = remember { com.less.audio.ml.GemmaDownloader(context) }
    var isCopyingModel by remember { mutableStateOf(false) }

    val gemmaPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) {
            coroutineScope.launch {
                isCopyingModel = true
                val success = gemmaDownloader.copyModelFromUri(uri)
                isCopyingModel = false
                if (success) {
                    VisionSceneContext.updateGemmaThoughts("Model ready! Toggle engine to start conductor.")
                } else {
                    VisionSceneContext.updateGemmaThoughts("Failed to copy model.")
                }
            }
        }
    }

    // Detect already-bonded glasses immediately on load
    LaunchedEffect(Unit) {
        isConnected = btRouter.isDeviceConnected()
    }

    // Poll connection state even when engine isn't running
    LaunchedEffect(Unit) {
        while (isActive) {
            isConnected = btRouter.isDeviceConnected()
            delay(2000)
        }
    }

    // Poll latency/state every 500ms while engine runs
    LaunchedEffect(isEngineRunning) {
        while (isActive && isEngineRunning) {
            latencyMs = NativeAudioBridge.getLatencyMs()
            isConnected = btRouter.isDeviceConnected()
            sessionState = btRouter.currentSessionState.name
            activeBackend = NativeAudioBridge.getActiveBackend()
            delay(500)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    colors = listOf(
                        Color(0xFF0D0D1A),
                        Color(0xFF16162A),
                        Color(0xFF0D0D1A)
                    )
                )
            )
            .padding(24.dp)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.height(16.dp))

        // --- Header ---
        Text(
            text = "LESS",
            fontSize = 36.sp,
            fontWeight = FontWeight.Bold,
            color = Color(0xFF6C63FF),
            letterSpacing = 8.sp
        )
        Text(
            text = "Low-latency Edge Sound Suppression",
            fontSize = 12.sp,
            color = Color(0xFF888899),
            letterSpacing = 2.sp
        )

        Spacer(modifier = Modifier.height(24.dp))

        // --- Meta Wearables Registration ---
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = if (isRegistered)
                    Color(0xFF1A2E1A) else Color(0xFF1A1A2E)
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        "🕶️",
                        fontSize = 20.sp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            "Meta Wearables SDK",
                            color = Color(0xFFAABBCC),
                            fontSize = 12.sp
                        )
                        Text(
                            if (isRegistered) "✓ Registered" else registrationState,
                            color = if (isRegistered) Color(0xFF03DAC5) else Color(0xFFFF6B6B),
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Medium
                        )
                    }

                    if (!isRegistered) {
                        TextButton(onClick = onStartRegistration) {
                            Text("Register", color = Color(0xFF6C63FF), fontSize = 13.sp)
                        }
                    }
                }

                // Session state indicator (when registered)
                if (isRegistered) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            "Session: $sessionState",
                            color = when (sessionState) {
                                "RUNNING" -> Color(0xFF03DAC5)
                                "PAUSED" -> Color(0xFFFFB74D)
                                else -> Color(0xFF666677)
                            },
                            fontSize = 11.sp
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // --- Connection Status ---
        StatusCard(
            title = "Connection",
            isActive = isConnected,
            activeText = btRouter.getConnectedDeviceName() ?: "Ray-Ban Connected",
            inactiveText = "No device connected",
            onAction = onScanBluetooth,
            actionText = "Scan"
        )

        Spacer(modifier = Modifier.height(12.dp))

        // --- Engine Status ---
        StatusCard(
            title = "Audio Engine",
            isActive = isEngineRunning,
            activeText = "Running — ${String.format("%.1f", latencyMs)}ms latency",
            inactiveText = "Stopped",
            onAction = {
                if (isEngineRunning) {
                    onStopEngine()
                    isEngineRunning = false
                    activeBackend = "Not initialized"
                } else {
                    onStartEngine()
                    isEngineRunning = NativeAudioBridge.isRunning()
                    if (isEngineRunning) {
                        activeBackend = NativeAudioBridge.getActiveBackend()
                    }
                }
            },
            actionText = if (isEngineRunning) "Stop" else "Start"
        )

        // Phase 5: Active delegate backend indicator
        if (isEngineRunning) {
            Spacer(modifier = Modifier.height(4.dp))
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(
                    containerColor = when {
                        activeBackend.contains("NPU") -> Color(0xFF1A2E1A)   // green tint
                        activeBackend.contains("GPU") -> Color(0xFF2E2A1A)   // amber tint
                        else -> Color(0xFF1A1A2E)                            // default
                    }
                )
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        "Inference Backend",
                        color = Color(0xFF888899),
                        fontSize = 11.sp
                    )
                    Text(
                        activeBackend,
                        color = when {
                            activeBackend.contains("NPU") -> Color(0xFF03DAC5)
                            activeBackend.contains("GPU") -> Color(0xFFFFB74D)
                            activeBackend.contains("CPU") -> Color(0xFF6C63FF)
                            else -> Color(0xFF888899)
                        },
                        fontSize = 12.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // =================================================================
        // Phase 8: Real-Time Audio Pipeline Visualizer
        // =================================================================
        DashboardVisualizer(
            isEngineRunning = isEngineRunning,
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(16.dp))

        // =================================================================
        // Phase 8: Acoustic Environment Presets
        // =================================================================
        AcousticPresetsRow(
            currentSuppression = suppressionLevel,
            onPresetSelected = { preset ->
                if (preset.suppression >= 0f) {
                    suppressionLevel = preset.suppression
                    onSuppressionChanged(preset.suppression)
                }
                // "Custom" preset (suppression = -1) keeps manual slider
            }
        )

        Spacer(modifier = Modifier.height(16.dp))

        // =================================================================
        // Phase 11: Comfort Masking & Phase 15: Vision Music Controls
        // =================================================================
        var processingMode by remember { mutableIntStateOf(0) }  // 0=VoiceIsolate, 1=ComfortMask, 2=VisionMusic
        var maskTexture by remember { mutableIntStateOf(0) }
        var detectedEnvironment by remember { mutableIntStateOf(0) }
        var isAlertActive by remember { mutableStateOf(false) }
        var isMaskActive by remember { mutableStateOf(false) }

        // Phase 15 states
        var musicInterpretation by remember { mutableIntStateOf(NativeAudioBridge.getMusicInterpretation()) }
        var synthQuality by remember { mutableIntStateOf(NativeAudioBridge.getSynthQuality()) }
        var musicBpm by remember { mutableFloatStateOf(0f) }
        var musicKey by remember { mutableIntStateOf(0) }
        
        // Phase 5 ONNX states
        var tinyMusicianMode by remember { mutableIntStateOf(0) } // 0=Off, 1=Tweaker, 2=Composer
        var nnapiThermalWarning by remember { mutableStateOf(false) }

        // Poll engine state
        LaunchedEffect(isEngineRunning) {
            while (isActive && isEngineRunning) {
                detectedEnvironment = NativeAudioBridge.getDetectedEnvironment()
                isAlertActive = NativeAudioBridge.isAlertActive()
                isMaskActive = NativeAudioBridge.isMaskActive()
                
                if (processingMode == 2) {
                    musicBpm = NativeAudioBridge.getMusicBpm()
                    musicKey = NativeAudioBridge.getMusicKey()
                    
                    val thermalState = NativeAudioBridge.getThermalState()
                    // ThermalState: 0=Nominal, 1=Warm, 2=Throttling, 3=Critical
                    nnapiThermalWarning = thermalState >= 2
                } else {
                    nnapiThermalWarning = false
                }
                delay(300)
            }
        }

        val textureNames = listOf("Brown", "Pink", "White", "Nature", "Harmonic")
        val textureEmojis = listOf("🟫", "🌸", "⬜", "🌿", "🎵")
        val envNames = listOf("Quiet", "Office", "Café", "Transit", "Outdoor")
        val envEmojis = listOf("🤫", "🏢", "☕", "🚇", "🌳")

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = if (processingMode == 1)
                    Color(0xFF1A1E2E) else Color(0xFF1A1A2E)
            )
        ) {
            Column(modifier = Modifier.padding(20.dp)) {

                // Header row with environment badge
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        "Acoustic Mode",
                        color = Color(0xFFAABBCC),
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Medium
                    )

                    // Alert indicator (only visible during transient alerts)
                    if (isAlertActive && processingMode == 1) {
                        Text(
                            "⚠️ Alert Pass-through",
                            color = Color(0xFFFFB74D),
                            fontSize = 11.sp,
                            fontWeight = FontWeight.SemiBold
                        )
                    }
                }

                Spacer(modifier = Modifier.height(12.dp))

                // Mode toggle: Voice Isolate ↔ Comfort Mask ↔ Vision Music
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    // Voice Isolate button
                    Button(
                        onClick = {
                            if (processingMode == 2) onStopVisionMusic()
                            processingMode = 0
                            NativeAudioBridge.setProcessingMode(0)
                        },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (processingMode == 0)
                                Color(0xFF6C63FF) else Color(0xFF2A2A4A)
                        ),
                        contentPadding = PaddingValues(horizontal = 4.dp, vertical = 8.dp)
                    ) {
                        Text(
                            "🎙 Isolate",
                            fontSize = 12.sp,
                            fontWeight = if (processingMode == 0)
                                FontWeight.Bold else FontWeight.Normal
                        )
                    }

                    // Comfort Mask button
                    Button(
                        onClick = {
                            if (processingMode == 2) onStopVisionMusic()
                            processingMode = 1
                            NativeAudioBridge.setProcessingMode(1)
                        },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (processingMode == 1)
                                Color(0xFF03DAC5) else Color(0xFF2A2A4A)
                        ),
                        contentPadding = PaddingValues(horizontal = 4.dp, vertical = 8.dp)
                    ) {
                        Text(
                            "🎶 Mask",
                            fontSize = 12.sp,
                            fontWeight = if (processingMode == 1)
                                FontWeight.Bold else FontWeight.Normal
                        )
                    }
                    
                    // Vision Music button
                    Button(
                        onClick = {
                            processingMode = 2
                            NativeAudioBridge.setProcessingMode(2)
                            onStartVisionMusic()
                        },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (processingMode == 2)
                                Color(0xFFE040FB) else Color(0xFF2A2A4A)
                        ),
                        contentPadding = PaddingValues(horizontal = 4.dp, vertical = 8.dp)
                    ) {
                        Text(
                            "🎵 Vision",
                            fontSize = 12.sp,
                            fontWeight = if (processingMode == 2)
                                FontWeight.Bold else FontWeight.Normal
                        )
                    }
                }

                // Comfort Mask sub-controls (only shown when masking is active)
                if (processingMode == 1) {
                    Spacer(modifier = Modifier.height(16.dp))

                    // --- Mask Texture selector ---
                    Text(
                        "Mask Texture",
                        color = Color(0xFF888899),
                        fontSize = 12.sp
                    )
                    Spacer(modifier = Modifier.height(8.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        textureNames.forEachIndexed { index, name ->
                            val isSelected = maskTexture == index
                            val chipColor = when (index) {
                                4 -> if (isSelected) Color(0xFFE040FB) else Color(0xFF2A2A4A) // Harmonic = purple
                                3 -> if (isSelected) Color(0xFF66BB6A) else Color(0xFF2A2A4A) // Nature = green
                                else -> if (isSelected) Color(0xFF03DAC5) else Color(0xFF2A2A4A)
                            }
                            Button(
                                onClick = {
                                    maskTexture = index
                                    NativeAudioBridge.setMaskTexture(index)
                                },
                                modifier = Modifier.weight(1f),
                                shape = RoundedCornerShape(10.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = chipColor
                                ),
                                contentPadding = PaddingValues(
                                    horizontal = 2.dp, vertical = 6.dp
                                )
                            ) {
                                Column(
                                    horizontalAlignment = Alignment.CenterHorizontally
                                ) {
                                    Text(
                                        textureEmojis[index],
                                        fontSize = 14.sp
                                    )
                                    Text(
                                        name,
                                        fontSize = 9.sp,
                                        fontWeight = if (isSelected)
                                            FontWeight.Bold else FontWeight.Normal,
                                        color = if (isSelected)
                                            Color.White else Color(0xFF888899)
                                    )
                                }
                            }
                        }
                    }

                    // Harmonic description
                    if (maskTexture == 4) {
                        Spacer(modifier = Modifier.height(6.dp))
                        Text(
                            "♪ Generates ambient chords from the room's tone",
                            color = Color(0xFFE040FB).copy(alpha = 0.7f),
                            fontSize = 10.sp
                        )
                    }

                    Spacer(modifier = Modifier.height(14.dp))

                    // --- Environment Detection ---
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            "Detected Environment",
                            color = Color(0xFF888899),
                            fontSize = 12.sp
                        )

                        val envIdx = detectedEnvironment.coerceIn(
                            0, envNames.size - 1
                        )
                        Text(
                            "${envEmojis[envIdx]} ${envNames[envIdx]}",
                            color = Color(0xFF03DAC5),
                            fontSize = 13.sp,
                            fontWeight = FontWeight.SemiBold
                        )
                    }

                    // Mask active indicator
                    if (isMaskActive) {
                        Spacer(modifier = Modifier.height(6.dp))
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(8.dp)
                                    .clip(CircleShape)
                                    .background(Color(0xFF03DAC5))
                            )
                            Text(
                                "Mask generating — tonal noise detected",
                                color = Color(0xFF03DAC5).copy(alpha = 0.7f),
                                fontSize = 10.sp
                            )
                        }
                    }
                }

                // Phase 15: Vision Music sub-controls
                if (processingMode == 2) {
                    Spacer(modifier = Modifier.height(16.dp))

                    Text(
                        "TinyMusician AI Mode",
                        color = Color(0xFF888899),
                        fontSize = 12.sp
                    )
                    Spacer(modifier = Modifier.height(8.dp))

                    val aiModes = listOf("Disabled", "Tweaker (Subtle)", "Composer (Full)")
                    val aiEmojis = listOf("⏸️", "🎛️", "🎼")

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        aiModes.forEachIndexed { index, name ->
                            val isSelected = tinyMusicianMode == index
                            Button(
                                onClick = {
                                    tinyMusicianMode = index
                                    NativeAudioBridge.setTinyMusicianMode(index)
                                },
                                modifier = Modifier.weight(1f),
                                shape = RoundedCornerShape(10.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = if (isSelected) Color(0xFF03DAC5) else Color(0xFF2A2A4A)
                                ),
                                contentPadding = PaddingValues(horizontal = 2.dp, vertical = 6.dp)
                            ) {
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text(aiEmojis[index], fontSize = 14.sp)
                                    Text(
                                        name,
                                        fontSize = 9.sp,
                                        fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                        color = if (isSelected) Color.Black else Color(0xFF888899)
                                    )
                                }
                            }
                        }
                    }

                    if (tinyMusicianMode > 0 && nnapiThermalWarning) {
                        Spacer(modifier = Modifier.height(6.dp))
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(8.dp)
                                    .clip(CircleShape)
                                    .background(Color(0xFFFF6B6B))
                            )
                            Text(
                                "Thermal limit reached. AI generation slowed.",
                                color = Color(0xFFFF6B6B).copy(alpha = 0.9f),
                                fontSize = 10.sp
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    Text(
                        "Interpretation Style",
                        color = Color(0xFF888899),
                        fontSize = 12.sp
                    )
                    Spacer(modifier = Modifier.height(8.dp))

                    val interpretations = listOf("Ambient Drift", "Melodic Arp", "Rhythmic Pulse")
                    val interEmojis = listOf("☁️", "🎹", "🥁")

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        interpretations.forEachIndexed { index, name ->
                            val isSelected = musicInterpretation == index
                            Button(
                                onClick = {
                                    musicInterpretation = index
                                    NativeAudioBridge.setMusicInterpretation(index)
                                },
                                modifier = Modifier.weight(1f),
                                shape = RoundedCornerShape(10.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = if (isSelected) Color(0xFFE040FB) else Color(0xFF2A2A4A)
                                ),
                                contentPadding = PaddingValues(horizontal = 2.dp, vertical = 6.dp)
                            ) {
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text(interEmojis[index], fontSize = 14.sp)
                                    Text(
                                        name,
                                        fontSize = 9.sp,
                                        fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                        color = if (isSelected) Color.White else Color(0xFF888899)
                                    )
                                }
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(14.dp))

                    Text(
                        "Synth DSP Quality",
                        color = Color(0xFF888899),
                        fontSize = 12.sp
                    )
                    Spacer(modifier = Modifier.height(8.dp))

                    val qualities = listOf("Battery Saver", "Beautiful")
                    val qualityEmojis = listOf("🔋", "✨")

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        qualities.forEachIndexed { index, name ->
                            val isSelected = synthQuality == index
                            Button(
                                onClick = {
                                    synthQuality = index
                                    NativeAudioBridge.setSynthQuality(index)
                                },
                                modifier = Modifier.weight(1f),
                                shape = RoundedCornerShape(10.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = if (isSelected) Color(0xFF03DAC5) else Color(0xFF2A2A4A)
                                ),
                                contentPadding = PaddingValues(horizontal = 2.dp, vertical = 6.dp)
                            ) {
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text(qualityEmojis[index], fontSize = 14.sp)
                                    Text(
                                        name,
                                        fontSize = 9.sp,
                                        fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                        color = if (isSelected) Color.Black else Color(0xFF888899)
                                    )
                                }
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(14.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            "Musical Context",
                            color = Color(0xFF888899),
                            fontSize = 12.sp
                        )
                        val keyName = NativeAudioBridge.getKeyName(musicKey)
                        Text(
                            "Key: $keyName | ${musicBpm.toInt()} BPM",
                            color = Color(0xFFE040FB),
                            fontSize = 13.sp,
                            fontWeight = FontWeight.SemiBold
                        )
                    }

                    if (isEngineRunning) {
                        Spacer(modifier = Modifier.height(6.dp))
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(8.dp)
                                    .clip(CircleShape)
                                    .background(Color(0xFFE040FB))
                            )
                            Text(
                                "Camera scene mapped to generative synth",
                                color = Color(0xFFE040FB).copy(alpha = 0.7f),
                                fontSize = 10.sp
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    val vibeSum by VisionSceneContext.vibeSummary.collectAsState()
                    val gemmaThoughts by VisionSceneContext.gemmaThoughts.collectAsState()
                    val dominantColor by VisionSceneContext.dominantColor.collectAsState()

                    // Convert raw int color to compose Color, fallback to Dark Gray
                    val composeColor = dominantColor?.let { Color(it) } ?: Color.DarkGray

                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(12.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = Color(0xFF1E1E2E)
                        )
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp)
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    "🧠",
                                    fontSize = 16.sp
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(
                                    "Neural Conductor Insights",
                                    color = Color(0xFF03DAC5),
                                    fontSize = 12.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                vibeSum,
                                color = Color(0xFFAABBCC),
                                fontSize = 13.sp,
                                lineHeight = 18.sp
                            )
                            
                            Spacer(modifier = Modifier.height(12.dp))
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Box(
                                    modifier = Modifier
                                        .size(16.dp)
                                        .clip(RoundedCornerShape(4.dp))
                                        .background(composeColor)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(
                                    "Synesthesia Mapping Color",
                                    color = Color.Gray,
                                    fontSize = 11.sp
                                )
                            }
                            
                            if (gemmaThoughts != null) {
                                Spacer(modifier = Modifier.height(8.dp))
                                Text(
                                    "Gemma suggests:",
                                    color = Color(0xFF888899),
                                    fontSize = 11.sp,
                                    fontWeight = FontWeight.Bold
                                )
                                Spacer(modifier = Modifier.height(2.dp))
                                Text(
                                    gemmaThoughts!!,
                                    color = Color.White,
                                    fontSize = 13.sp,
                                    fontStyle = androidx.compose.ui.text.font.FontStyle.Italic,
                                    lineHeight = 18.sp
                                )
                                
                                if (!gemmaDownloader.getModelFile().exists()) {
                                    Spacer(modifier = Modifier.height(12.dp))
                                    Button(
                                        onClick = {
                                            gemmaPickerLauncher.launch(arrayOf("*/*"))
                                        },
                                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFE040FB)),
                                        enabled = !isCopyingModel,
                                        modifier = Modifier.fillMaxWidth()
                                    ) {
                                        Text(if (isCopyingModel) "Copying..." else "Pick Gemma Model (.bin)")
                                    }
                                }
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            "Does this match the scene?",
                            color = Color(0xFF888899),
                            fontSize = 12.sp
                        )
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Button(
                                onClick = {
                                    // Feedback acknowledged
                                    coroutineScope.launch(kotlinx.coroutines.Dispatchers.IO) {
                                        com.less.audio.ml.FeedbackLogger.logFeedback(
                                            context = context,
                                            yoloClasses = listOf(), // Empty for now, wired to ML state internally later
                                            interpretation = musicInterpretation,
                                            feedbackScore = 1
                                        )
                                    }
                                },
                                modifier = Modifier.size(36.dp),
                                shape = CircleShape,
                                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2A2A4A)),
                                contentPadding = PaddingValues(0.dp)
                            ) {
                                Text("👍", fontSize = 14.sp)
                            }
                            Button(
                                onClick = {
                                    // Re-roll interpretation on thumbs down
                                    coroutineScope.launch(kotlinx.coroutines.Dispatchers.IO) {
                                        com.less.audio.ml.FeedbackLogger.logFeedback(
                                            context = context,
                                            yoloClasses = listOf(), // Empty for now
                                            interpretation = musicInterpretation,
                                            feedbackScore = -1
                                        )
                                    }
                                    musicInterpretation = (musicInterpretation + 1) % interpretations.size
                                    NativeAudioBridge.setMusicInterpretation(musicInterpretation)
                                },
                                modifier = Modifier.size(36.dp),
                                shape = CircleShape,
                                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2A2A4A)),
                                contentPadding = PaddingValues(0.dp)
                            ) {
                                Text("👎", fontSize = 14.sp)
                            }
                        }
                    }
                }
            }
        }

        // --- Suppression Level ---
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1A1A2E))
        ) {
            Column(modifier = Modifier.padding(20.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("Suppression", color = Color(0xFFAABBCC), fontSize = 14.sp)
                    Text(
                        "${(suppressionLevel * 100).toInt()}%",
                        color = Color(0xFF6C63FF),
                        fontWeight = FontWeight.Bold
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
                Slider(
                    value = suppressionLevel,
                    onValueChange = {
                        suppressionLevel = it
                        onSuppressionChanged(it)
                    },
                    valueRange = 0f..1f,
                    colors = SliderDefaults.colors(
                        thumbColor = Color(0xFF6C63FF),
                        activeTrackColor = Color(0xFF6C63FF),
                        inactiveTrackColor = Color(0xFF2A2A4A)
                    )
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("Passthrough", color = Color(0xFF666677), fontSize = 11.sp)
                    Text("Max Suppression", color = Color(0xFF666677), fontSize = 11.sp)
                }
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        // --- Profiling Mode ---
        val canProfile = sessionManager.canStartSession()
        val remainingBudget = sessionManager.remainingBudgetMinutes()

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1A1A2E))
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text("Profiling Mode", color = Color(0xFFAABBCC), fontSize = 14.sp)
                    Text("🔒 AES-256-GCM", color = Color(0xFF03DAC5), fontSize = 10.sp)
                }
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    "Budget: ${remainingBudget}min remaining (24h window)",
                    color = Color(0xFF666677),
                    fontSize = 11.sp
                )
                Spacer(modifier = Modifier.height(12.dp))

                // Phase 6: GDPR Consent Gate
                var showConsentDialog by remember { mutableStateOf(false) }
                val prefs = context.getSharedPreferences("less_privacy", 0)
                var hasConsented by remember {
                    mutableStateOf(prefs.getBoolean("profiling_consent_granted", false))
                }

                Button(
                    onClick = {
                        if (isProfiling) {
                            onStopProfiling()
                            isProfiling = false
                        } else if (!hasConsented) {
                            // Show GDPR consent dialog before first profiling session
                            showConsentDialog = true
                        } else {
                            onStartProfiling()
                            isProfiling = true
                        }
                    },
                    enabled = canProfile || isProfiling,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (isProfiling)
                            Color(0xFFFF6B6B) else Color(0xFF03DAC5)
                    ),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        if (isProfiling) "⏹  Stop Recording"
                        else if (!hasConsented) "🔐  Review Privacy & Start"
                        else "🎙  Start 15-min Capture",
                        fontWeight = FontWeight.SemiBold
                    )
                }

                if (!hasConsented) {
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        "Privacy consent required before recording",
                        color = Color(0xFFFFB74D),
                        fontSize = 11.sp
                    )
                }

                // --- GDPR Privacy Consent Dialog ---
                if (showConsentDialog) {
                    AlertDialog(
                        onDismissRequest = { showConsentDialog = false },
                        containerColor = Color(0xFF0D0D1A),
                        titleContentColor = Color(0xFFEEEEFF),
                        textContentColor = Color(0xFFAABBCC),
                        title = {
                            Text("🔒  Audio Privacy Consent", fontWeight = FontWeight.Bold)
                        },
                        text = {
                            Column {
                                Text(
                                    "LESS needs to capture environmental audio to personalize " +
                                    "your noise suppression model. Please review the following:",
                                    fontSize = 13.sp,
                                    lineHeight = 18.sp
                                )
                                Spacer(modifier = Modifier.height(12.dp))

                                // Data collection
                                Text("📱  What is captured:", color = Color(0xFF03DAC5),
                                    fontWeight = FontWeight.SemiBold, fontSize = 12.sp)
                                Text(
                                    "Raw environmental audio from your Meta Ray-Ban microphones " +
                                    "during manually-initiated 15-minute profiling sessions.",
                                    fontSize = 12.sp, lineHeight = 16.sp
                                )
                                Spacer(modifier = Modifier.height(8.dp))

                                // Encryption
                                Text("🔐  How it is protected:", color = Color(0xFF03DAC5),
                                    fontWeight = FontWeight.SemiBold, fontSize = 12.sp)
                                Text(
                                    "All audio is encrypted with AES-256-GCM backed by the " +
                                    "Android hardware Keystore before touching disk. " +
                                    "Plaintext audio never exists in storage.",
                                    fontSize = 12.sp, lineHeight = 16.sp
                                )
                                Spacer(modifier = Modifier.height(8.dp))

                                // Storage & transmission
                                Text("📍  Where it is stored:", color = Color(0xFF03DAC5),
                                    fontWeight = FontWeight.SemiBold, fontSize = 12.sp)
                                Text(
                                    "Encrypted data is stored ONLY on this device in the app's " +
                                    "private storage. Audio is NEVER transmitted to any server, " +
                                    "cloud service, or third party.",
                                    fontSize = 12.sp, lineHeight = 16.sp
                                )
                                Spacer(modifier = Modifier.height(8.dp))

                                // Purpose
                                Text("🎯  Sole purpose:", color = Color(0xFF03DAC5),
                                    fontWeight = FontWeight.SemiBold, fontSize = 12.sp)
                                Text(
                                    "Your audio data is used exclusively to train a personalized " +
                                    "on-device noise suppression model. After training completes, " +
                                    "the raw audio is automatically and permanently deleted.",
                                    fontSize = 12.sp, lineHeight = 16.sp
                                )
                                Spacer(modifier = Modifier.height(8.dp))

                                // Revocation
                                Text("↩️  Your rights:", color = Color(0xFF03DAC5),
                                    fontWeight = FontWeight.SemiBold, fontSize = 12.sp)
                                Text(
                                    "You can revoke this consent at any time. Revoking consent " +
                                    "will permanently delete all stored audio corpus data and " +
                                    "disable profiling mode.",
                                    fontSize = 12.sp, lineHeight = 16.sp
                                )
                            }
                        },
                        confirmButton = {
                            Button(
                                onClick = {
                                    // Persist consent + timestamp
                                    prefs.edit()
                                        .putBoolean("profiling_consent_granted", true)
                                        .putLong("profiling_consent_timestamp",
                                            System.currentTimeMillis())
                                        .apply()
                                    hasConsented = true
                                    showConsentDialog = false
                                    // Immediately start profiling after consent
                                    onStartProfiling()
                                    isProfiling = true
                                },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = Color(0xFF03DAC5)
                                ),
                                shape = RoundedCornerShape(8.dp)
                            ) {
                                Text("I Understand & Consent", fontWeight = FontWeight.Bold)
                            }
                        },
                        dismissButton = {
                            OutlinedButton(
                                onClick = { showConsentDialog = false },
                                shape = RoundedCornerShape(8.dp)
                            ) {
                                Text("Cancel", color = Color(0xFF888899))
                            }
                        }
                    )
                }

                if (sessionManager.hasMinimumCorpus()) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        "✓ ${sessionManager.totalCorpusMinutes()}min corpus ready for training",
                        color = Color(0xFF03DAC5),
                        fontSize = 11.sp
                    )
                }

                // Consent revocation option
                if (hasConsented && !isProfiling) {
                    Spacer(modifier = Modifier.height(8.dp))
                    TextButton(
                        onClick = {
                            // Revoke consent and delete all corpus data
                            prefs.edit()
                                .putBoolean("profiling_consent_granted", false)
                                .remove("profiling_consent_timestamp")
                                .apply()
                            hasConsented = false

                            // Permanently delete all corpus data
                            val corpusDir = java.io.File(context.filesDir, "corpus")
                            if (corpusDir.exists()) {
                                corpusDir.listFiles()?.forEach { it.delete() }
                                Log.i("LESS_Privacy",
                                    "Consent revoked — all corpus data deleted")
                            }
                        }
                    ) {
                        Text(
                            "Revoke Consent & Delete All Audio Data",
                            color = Color(0xFFFF6B6B),
                            fontSize = 11.sp
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // --- Training ---
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1A1A2E))
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("Overnight Training", color = Color(0xFFAABBCC), fontSize = 14.sp)
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    "Runs when charging + idle • Updates adapter weights",
                    color = Color(0xFF666677),
                    fontSize = 11.sp
                )
                Spacer(modifier = Modifier.height(12.dp))

                OutlinedButton(
                    onClick = onScheduleTraining,
                    enabled = sessionManager.hasMinimumCorpus(),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = Color(0xFF6C63FF)
                    )
                ) {
                    Text("Schedule Training", fontWeight = FontWeight.SemiBold)
                }

                if (!sessionManager.hasMinimumCorpus()) {
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        "Need ≥5min corpus — record more profiling sessions",
                        color = Color(0xFFFF6B6B),
                        fontSize = 11.sp
                    )
                }
            }
        }

        // =====================================================================
        // Debug Panel — MockDeviceKit Simulation (DEBUG BUILDS ONLY)
        // =====================================================================
        if (mockController != null) {
            Spacer(modifier = Modifier.height(20.dp))
            MockDevicePanel(mockController = mockController)
        }

        Spacer(modifier = Modifier.height(24.dp))

        // --- Footer ---
        Text(
            "LESS v0.4.0 — Phase 11",
            color = Color(0xFF444455),
            fontSize = 10.sp
        )

        Spacer(modifier = Modifier.height(16.dp))
    }
}

// =============================================================================
// Mock Device Panel (Debug)
// =============================================================================

@Composable
fun MockDevicePanel(mockController: MockGlassesController) {
    var isMockInitialized by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF2A1A1A))
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                "🔧 DEBUG: MockDeviceKit",
                color = Color(0xFFFF6B6B),
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold
            )
            Text(
                "Simulate glasses states without hardware",
                color = Color(0xFF666677),
                fontSize = 10.sp
            )

            Spacer(modifier = Modifier.height(12.dp))

            if (!isMockInitialized) {
                Button(
                    onClick = {
                        isMockInitialized = mockController.initialize()
                    },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF6C63FF)
                    ),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Initialize Mock Glasses", fontSize = 12.sp)
                }
            } else {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    OutlinedButton(
                        onClick = { mockController.simulateDon() },
                        shape = RoundedCornerShape(8.dp),
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = Color(0xFF03DAC5)
                        )
                    ) {
                        Text("Don", fontSize = 11.sp)
                    }
                    OutlinedButton(
                        onClick = { mockController.simulateDoff() },
                        shape = RoundedCornerShape(8.dp),
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = Color(0xFFFFB74D)
                        )
                    ) {
                        Text("Doff", fontSize = 11.sp)
                    }
                    OutlinedButton(
                        onClick = { mockController.simulateFold() },
                        shape = RoundedCornerShape(8.dp),
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = Color(0xFFFF6B6B)
                        )
                    ) {
                        Text("Fold", fontSize = 11.sp)
                    }
                }
            }
        }
    }
}

// =============================================================================
// Reusable Components
// =============================================================================

@Composable
fun StatusCard(
    title: String,
    isActive: Boolean,
    activeText: String,
    inactiveText: String,
    onAction: () -> Unit,
    actionText: String
) {
    // Pulsing dot animation
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val alpha by infiniteTransition.animateFloat(
        initialValue = 0.3f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = EaseInOutCubic),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse_alpha"
    )

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF1A1A2E))
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Status dot
            Box(
                modifier = Modifier
                    .size(10.dp)
                    .clip(CircleShape)
                    .background(
                        if (isActive)
                            Color(0xFF03DAC5).copy(alpha = alpha)
                        else
                            Color(0xFF666677)
                    )
            )
            Spacer(modifier = Modifier.width(12.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(title, color = Color(0xFFAABBCC), fontSize = 12.sp)
                Text(
                    if (isActive) activeText else inactiveText,
                    color = if (isActive) Color(0xFF03DAC5) else Color(0xFF666677),
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium
                )
            }

            TextButton(onClick = onAction) {
                Text(actionText, color = Color(0xFF6C63FF), fontSize = 13.sp)
            }
        }
    }
}
