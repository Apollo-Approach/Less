plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

import java.util.Properties

val localProperties = Properties()
val localPropertiesFile = rootProject.file("local.properties")
if (localPropertiesFile.exists()) {
    localProperties.load(localPropertiesFile.inputStream())
}

val metaAppId = localProperties.getProperty("meta_app_id") ?: ""
val metaClientToken = localProperties.getProperty("meta_client_token") ?: ""


android {
    namespace = "com.less.audio"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.less.audio"
        minSdk = 33          // Android 13 — required for BLE Audio / LC3
        targetSdk = 36       // Android 16 — for STOP_REASON_TIMEOUT_ABANDONED compliance
        versionCode = 2
        versionName = "0.3.0-phase5"
        
        manifestPlaceholders["metaAppId"] = metaAppId
        manifestPlaceholders["metaClientToken"] = metaClientToken

        ndk {
            // Target ARM64 only — all modern Android phones + Qualcomm SoCs
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                cppFlags += listOf(
                    "-std=c++17",
                    "-O2",
                    "-ffast-math",
                    "-DANDROID"
                )
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DANDROID_ARM_NEON=TRUE"
                )
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    // =========================================================================
    // Phase 5: QNN SDK Native Library Bundling
    // =========================================================================
    // Place QNN vendor .so files in:
    //   app/src/main/jniLibs/arm64-v8a/
    //
    // Required libraries (from Qualcomm AI Engine Direct SDK):
    //   - libQnnTFLiteDelegate.so — LiteRT → QNN bridge delegate
    //   - libQnnHtp.so            — HTP (NPU) backend
    //   - libQnnHtpPrepare.so     — HTP graph preparation
    //   - libQnnHtpV73Stub.so     — HTP v73 skeleton (SoC-specific)
    //   - libQnnSystem.so         — QNN system interface
    //
    // Optional (GPU fallback):
    //   - libQnnGpu.so            — Adreno GPU backend
    //
    // These are loaded at runtime via dlopen() in noise_suppressor.cpp.
    // The nativeLibraryDir is passed from Kotlin → JNI → C++ so the
    // dlopen() fallback chain can find them.
    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    lint {
        checkReleaseBuilds = false
        abortOnError = false
    }

    compileOptions {
        isCoreLibraryDesugaringEnabled = true
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
        prefab = true  // Enable prefab for TFLite native linking via CMake
        buildConfig = true
    }
}

dependencies {
    coreLibraryDesugaring("com.android.tools:desugar_jdk_libs:2.0.4")

    // Jetpack Compose
    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")

    // Compose Foundation — HorizontalPager, scroll APIs
    implementation("androidx.compose.foundation:foundation")

    // Lottie — vector animations for onboarding wizard (Phase 8)
    implementation("com.airbnb.android:lottie-compose:6.6.2")

    // Core AndroidX
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")

    // WorkManager — overnight training scheduler
    // 2.11.0+ required for Android 16 stopReason == STOP_REASON_TIMEOUT_ABANDONED
    implementation("androidx.work:work-runtime-ktx:2.11.0")

    // Encrypted SharedPreferences — FGS budget tracking
    implementation("androidx.security:security-crypto:1.1.0-alpha06")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    // TensorFlow Lite (LiteRT) — native C API for ML inference
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-api:2.16.1")

    // ONNX Runtime for Phase 5 On-Device Training
    implementation("com.microsoft.onnxruntime:onnxruntime-training-android:1.17.0")

    // MediaPipe GenAI for Phase 18: Neural Conductor (Gemma On-Device)
    implementation("com.google.mediapipe:tasks-genai:0.10.14")

    // Android Palette API for Phase 19 (Synesthesia color extraction)
    implementation("androidx.palette:palette-ktx:1.0.0")

    // Meta Wearables Device Access Toolkit (MWDAT)
    // Requires GitHub PAT in settings.gradle.kts for repo auth
    implementation("com.meta.wearable:mwdat-core:0.5.0")
    implementation("com.meta.wearable:mwdat-camera:0.5.0")
    debugImplementation("com.meta.wearable:mwdat-mockdevice:0.5.0")

    // Debug tooling
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")

    // OpenTelemetry SDK for offline-buffered telemetry
    implementation("io.opentelemetry.android:android-agent:0.6.0-alpha")
    implementation("io.opentelemetry:opentelemetry-exporter-otlp:1.37.0")
}

// Task to extract TFLite C API headers and JNI libs from the AAR so CMake can include them
tasks.register("extractTFLite") {
    doLast {
        val aarFiles = configurations.getByName("debugCompileClasspath").files.filter { it.name.contains("tensorflow-lite") }
        aarFiles.forEach { aar ->
            copy {
                from(zipTree(aar).matching { include("headers/**", "jni/**") })
                into(layout.buildDirectory.dir("generated/tflite").get())
            }
        }
    }
}

tasks.named("preBuild") {
    dependsOn("extractTFLite")
}
