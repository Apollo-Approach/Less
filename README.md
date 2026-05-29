<p align="center">
  <img src="Less%20Logo%20Icon%20192x.png" alt="LESS Logo" width="96" height="96" />
</p>

<h1 align="center">LESS</h1>

<p align="center">
  <strong>Real-time audio intelligence for Meta Ray-Ban smart glasses</strong><br/>
  6ms latency · NPU-accelerated · On-device ML inference
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Android%2013%2B-3ddc84?style=for-the-badge&logo=android&logoColor=white" alt="Android 13+" />
  <img src="https://img.shields.io/badge/NDK-C%2B%2B17-00599C?style=for-the-badge&logo=cplusplus&logoColor=white" alt="C++17" />
  <img src="https://img.shields.io/badge/Hardware-Meta%20Ray--Ban%20Gen%202-0467DF?style=for-the-badge&logo=meta&logoColor=white" alt="Meta Ray-Ban" />
  <img src="https://img.shields.io/badge/NPU-Qualcomm%20QNN%20HTP-EF4444?style=for-the-badge&logo=qualcomm&logoColor=white" alt="Qualcomm QNN" />
  <img src="https://img.shields.io/badge/Version-0.5.0--phase20-8B5CF6?style=for-the-badge" alt="Version" />
</p>

---

## What is LESS?

LESS transforms Meta Ray-Ban Gen 2 smart glasses into an intelligent audio processing platform. It runs entirely on-device with three distinct processing modes:

| Mode | Description | Mic Input | Use Case |
|------|-------------|-----------|----------|
| **Voice Isolate** | DTLN neural network strips background noise, preserves speech | ✅ | Noisy commutes, open offices |
| **Comfort Mask** | Psychoacoustic masking overlays to blunt tonal noise | ✅ | HVAC hum, persistent drone |
| **Vision Music** | Generative music driven by camera scene analysis | ❌ | Ambient soundtrack to your world |

All processing runs within a single Oboe/AAudio callback at ≤6ms round-trip latency through BLE Audio (LC3).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Android / Kotlin                         │
│                                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌────────────┐ │
│  │ MainActivity   │ BluetoothAudio│  │ Training   │  │ Profiling  │ │
│  │ (Compose UI)   │ Router       │  │ Worker     │  │ Service    │ │
│  └──────┬───┘  └──────┬───────┘  └─────┬─────┘  └──────┬─────┘ │
│         │             │                 │               │        │
│         └─────────────┴────────┬────────┴───────────────┘        │
│                                │ JNI                             │
├────────────────────────────────┼──────────────────────────────────┤
│                         C++ / Native                             │
│                                │                                 │
│  ┌─────────────────────────────▼──────────────────────────────┐  │
│  │              LessAudioEngine (audio_engine.cpp)             │  │
│  │                   Oboe MMAP Exclusive Mode                  │  │
│  │                                                             │  │
│  │  Mode 0                Mode 1                Mode 2         │  │
│  │  ┌──────────┐    ┌─────────────────┐    ┌──────────────┐   │  │
│  │  │NoiseSup- │    │ GSC → MaskEngine│    │ VisionSynth  │   │  │
│  │  │pressor   │    │   → AlertDet    │    │ (generative  │   │  │
│  │  │(DTLN/QNN)│    │   → EnvClassify │    │  music)      │   │  │
│  │  └──────────┘    └─────────────────┘    └──────────────┘   │  │
│  │                                                             │  │
│  │  ┌──────────────────────────────────────────────────────┐   │  │
│  │  │           ThermalManager (ADPF)                       │   │  │
│  │  │    NPU → GPU → CPU → SpectralGate fallback chain      │   │  │
│  │  └──────────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Processing Modes

### 🎙️ Mode 0: Voice Isolate

Real-time neural noise suppression using a dual-signal DTLN (Deep TensorFlow Lite Noise) model running at ~0.4ms inference on the Qualcomm HTP (NPU).

- **Model**: DTLN INT8 TFLite with LoRA adapter fine-tuning
- **Pipeline**: BLE Mic → Oboe → DTLN → Oboe → BLE Speaker
- **Backend Cascade**: QNN HTP (NPU) → QNN GPU (Adreno) → XNNPACK (CPU) → Spectral Gate

### 🔊 Mode 1: Comfort Mask

Psychoacoustic masking engine that analyzes ambient noise and generates spectrally-shaped mask audio to reduce perceived noise without full isolation.

- **Textures**: Brown noise, Pink noise, White noise, Nature, Harmonic drone
- **Pipeline**: Mic → GSC echo-cancel → Spectral analysis → Mask generation → Alert detection → Mixing
- **Features**: Automatic environment classification, transient alert pass-through, harmonic chord progressions

### 🎵 Mode 2: Vision Music (Phase 15)

Generative music synthesizer that produces real-time, musically coherent audio driven by visual scene parameters from YOLO11n object detection on the device camera, enhanced by the TinyMusician ONNX neural network for personalized playback mapping.

- **Interpretations**: Ambient Drift (warm pads), Melodic Arpeggio (plucked patterns), Rhythmic Pulse (bass + gates)
- **Neural Personalization**: Employs ONNX Runtime On-Device Training (ODT) via a `PersonalizationWorker` to update a base `tinymusician_base.onnx` model using real-time user feedback.
- **Music Theory**: 8 scale types, circle-of-fifths key modulation, functional chord progressions with voice leading
- **Architecture**: Output-only (no mic) → inherently feedback-free
- **Scene Mapping**: Object density → note density, Valence → key/scale, Arousal → tempo, Timbre → waveform

---

## Project Structure

```
Less/
├── app/
│   ├── build.gradle.kts                  # Build config (NDK, QNN, Meta SDK)
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── cpp/                           # ── Native C++ Layer ──
│       │   ├── CMakeLists.txt             # Oboe, TFLite, QNN linking
│       │   ├── audio_engine.h/cpp         # Core Oboe MMAP duplex engine
│       │   ├── noise_suppressor.h/cpp     # DTLN inference + NPU delegation
│       │   ├── masking_engine.h/cpp       # Psychoacoustic mask generation
│       │   ├── gsc_filter.h/cpp           # Generalized sidelobe canceller
│       │   ├── alert_detector.h/cpp       # Transient alert detection
│       │   ├── environment_classifier.h/cpp # Ambient scene classification
│       │   ├── vision_synth.h/cpp         # Generative music synthesizer
│       │   ├── thermal_manager.h/cpp      # ADPF thermal-aware throttling
│       │   ├── less_jni.cpp               # JNI bridge to Kotlin
│       │   └── training/                  # On-device LoRA training pipeline
│       │       ├── snr_predictor.cpp
│       │       ├── data_pipeline.cpp/h
│       │       ├── adapter_trainer.cpp
│       │       └── training_jni.cpp
│       └── java/com/less/audio/           # ── Kotlin Layer ──
│           ├── MainActivity.kt            # Compose UI (dashboard, controls)
│           ├── NativeAudioBridge.kt        # JNI wrapper singleton
│           ├── BluetoothAudioRouter.kt     # BLE Audio session management
│           ├── LessApplication.kt          # App-level init
│           ├── LessTileService.kt          # Quick Settings tile
│           ├── ui/
│           │   ├── DashboardVisualizer.kt  # Real-time waveform + metrics
│           │   ├── OnboardingWizard.kt     # First-run setup flow
│           │   └── AcousticPresets.kt      # Noise profile presets
│           ├── training/
│           │   ├── TrainingWorker.kt       # WorkManager overnight training
│           │   └── OvernightTrainingScheduler.kt
│           ├── profiling/
│           │   ├── ProfilingService.kt     # Audio capture FGS
│           │   └── ProfilingSessionManager.kt
│           ├── telemetry/
│           │   └── TelemetryManager.kt     # OpenTelemetry offline metrics
│           └── debug/
│               └── MockGlassesController.kt
├── settings.gradle.kts                    # Meta MWDAT Maven repo config
├── build.gradle.kts                       # Root Gradle config
└── gradle/                                # Gradle wrapper
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Audio I/O | [Oboe](https://github.com/google/oboe) 1.9.0 | MMAP Exclusive mode, ≤6ms round-trip |
| ML Inference | TensorFlow Lite (LiteRT) 2.16 | DTLN noise suppression model |
| NPU Delegate | Qualcomm QNN HTP | 0.4ms inference, 60% power savings |
| Thermal API | ADPF (Android Dynamic Performance Framework) | Proactive backend throttling |
| Glasses SDK | Meta Wearables DAT 0.5.0 | BLE Audio, camera, sensor access |
| UI | Jetpack Compose + Material 3 | Dashboard, visualizer, onboarding |
| Telemetry | OpenTelemetry Android Agent | Offline-buffered metrics |
| Build | Gradle 8.x + CMake 3.22 | Native C++17, ARM64 NEON |

---

## Requirements

### Hardware
- Meta Ray-Ban Gen 2 smart glasses (or MWDAT mock device for development)
- Android phone with Snapdragon SoC (NPU acceleration requires QNN-compatible chipset)

### Software
- **Android 13+** (API 33) — required for BLE Audio / LC3 codec
- **Android NDK r25+** — C++17 ARM64 toolchain
- **Qualcomm AI Engine Direct SDK** — for QNN HTP delegate `.so` libraries
- **GitHub PAT** — with `read:packages` scope for Meta Wearables SDK

---

## Build & Run

### 1. Clone
```bash
git clone https://github.com/Apollo-Approach/Less.git
cd Less
```

### 2. Configure credentials

Create or edit `local.properties`:
```properties
sdk.dir=/path/to/android/sdk
github_token=YOUR_GITHUB_PAT_HERE
meta_app_id=YOUR_META_APP_ID_HERE
meta_client_token=YOUR_META_CLIENT_TOKEN_HERE
```

The GitHub token requires `read:packages` scope to pull the Meta Wearables DAT SDK from GitHub Packages.

### 3. QNN library setup (optional — enables NPU acceleration)

Place Qualcomm QNN shared libraries in:
```
app/src/main/jniLibs/arm64-v8a/
├── libQnnTFLiteDelegate.so
├── libQnnHtp.so
├── libQnnHtpPrepare.so
├── libQnnHtpV73Stub.so
├── libQnnSystem.so
└── libQnnGpu.so          # optional GPU fallback
```

Without these, the engine falls back to XNNPACK (CPU) → Spectral Gate.

### 4. Build
```bash
./gradlew assembleDebug
```

### 5. Install & run
```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

---

## Development Phases

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Real-time Oboe/AAudio duplex engine | ✅ Complete |
| 2 | On-device LoRA adapter training pipeline | ✅ Complete |
| 3 | Session lifecycle (doff/fold pause/resume) | ✅ Complete |
| 4 | Qualcomm QNN NPU delegation | ✅ Complete |
| 5 | QNN SDK native bundling + fallback chain | ✅ Complete |
| 6 | Encrypted corpus data feed (EncryptedFile → JNI) | ✅ Complete |
| 7 | ADPF thermal-aware dynamic backend switching | ✅ Complete |
| 8 | UI: Waveform visualizer, Quick Settings tile, Onboarding | ✅ Complete |
| 9 | Lock-free stream health monitoring + self-healing | ✅ Complete |
| 10 | OpenTelemetry offline telemetry | ✅ Complete |
| 11 | Psychoacoustic masking engine (Comfort Mask mode) | ✅ Complete |
| 12 | GSC echo cancellation + alert detection | ✅ Complete |
| 13 | Environment classification + harmonic drone | ✅ Complete |
| 14 | Acoustic presets + profiling session manager | ✅ Complete |
| 15 | Vision-to-Music generative synthesizer | ✅ Complete |
| 16 | TinyMusician Neural Integration (ONNX OdT) | ✅ Complete |
| 17 | DSP Quality Control & Architecture Benchmarks | ✅ Complete |
| 18 | Neural Conductor (Gemma 2B) Prototyping | ✅ Complete |
| 19 | Gemma 2B INT4 MediaPipe Wrapper + Storage Import | ✅ Complete |
| 20 | Generative Audio Orchestration Pipeline | ✅ Complete |

---

## Audio Pipeline Constraints

The Oboe audio callback (`onAudioReady`) runs on a real-time priority thread with strict constraints:

```
❌ No heap allocations (new, malloc, vector resize)
❌ No mutex locks (std::mutex, pthread_mutex)
❌ No JNI calls
❌ No file I/O or logging
❌ No syscalls that might block

Budget: 480 frames @ 48kHz = 10ms deadline
Target: ≤3ms inference + <1ms overhead = <4ms total
```

All cross-thread communication uses `std::atomic` with `memory_order_relaxed`. The lock-free architecture ensures zero audio glitches even under thermal pressure.

---

## License

Private — all rights reserved.
