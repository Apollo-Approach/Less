cmake_minimum_required(VERSION 3.22.1)
project(less_engine LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# Dependencies
# =============================================================================

# Oboe — Google's high-performance audio library for Android
include(FetchContent)
FetchContent_Declare(
    oboe
    GIT_REPOSITORY https://github.com/google/oboe.git
    GIT_TAG        1.9.0
)
FetchContent_MakeAvailable(oboe)

# =============================================================================
# TensorFlow Lite (LiteRT) — linked via prebuilt .so from Gradle AAR
# =============================================================================
# The LiteRT AAR (org.tensorflow:tensorflow-lite) ships:
#   - Headers in:  jni/include/
#   - Shared lib:  jni/<ABI>/libtensorflowlite_jni.so
#
# These are extracted by Gradle and made available via the
# Find the JNI library extracted by Gradle
# We bypass find_library because CMake configure phase runs before Gradle's extraction task.
set(TFLITE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/generated/tflite)
set(TFLITE_LIB ${TFLITE_DIR}/jni/${ANDROID_ABI}/libtensorflowlite_jni.so)

# =============================================================================
# Source files
# =============================================================================

# Phase 1: Real-time audio engine
set(ENGINE_SOURCES
    audio_engine.cpp
    noise_suppressor.cpp
    thermal_manager.cpp
    less_jni.cpp
)

# Phase 11: Psychoacoustic masking pipeline
set(MASKING_SOURCES
    masking_engine.cpp
    gsc_filter.cpp
    alert_detector.cpp
    environment_classifier.cpp
)

# Phase 2: Offline training pipeline
set(TRAINING_SOURCES
    training/snr_predictor.cpp
    training/data_pipeline.cpp
    training/adapter_trainer.cpp
    training/training_jni.cpp
)

# Phase 15: Vision-to-Music generative synthesizer
set(VISION_SOURCES
    vision_synth.cpp
)

# =============================================================================
# Shared library target
# =============================================================================

add_library(less_engine SHARED
    ${ENGINE_SOURCES}
    ${MASKING_SOURCES}
    ${TRAINING_SOURCES}
    ${VISION_SOURCES}
)

target_include_directories(less_engine PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/training
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/generated/tflite/headers
)

target_link_libraries(less_engine PRIVATE
    oboe
    ${TFLITE_LIB}
    android
    log
    dl      # Phase 7: dlopen/dlsym for ADPF thermal API dynamic loading
)

# =============================================================================
# Compiler flags
# =============================================================================

target_compile_options(less_engine PRIVATE
    -O2
    -ffast-math
    -Wall
    -Wextra
    -Wno-unused-parameter
)

# ARM NEON SIMD — enabled by default for arm64-v8a, explicit for v7a fallback
if(ANDROID_ABI STREQUAL "armeabi-v7a")
    target_compile_options(less_engine PRIVATE -mfpu=neon)
endif()
