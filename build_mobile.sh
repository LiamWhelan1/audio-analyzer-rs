#!/usr/bin/env bash
# =============================================================================
# build_mobile.sh — Build audio_analyzer_rs for Android (Kotlin) and iOS (Swift)
#
# USAGE:
#   ./build_mobile.sh            # Build for current platform (Android on Windows/Linux, both on macOS)
#   ./build_mobile.sh android    # Android only
#   ./build_mobile.sh ios        # iOS only (requires macOS)
#   ./build_mobile.sh setup      # Install targets + tools only (no build)
#
# REQUIREMENTS:
#   Android: Android NDK installed, ANDROID_NDK_HOME env var set
#            e.g. export ANDROID_NDK_HOME="$HOME/AppData/Local/Android/Sdk/ndk/27.2.12479018"
#   iOS:     macOS + Xcode command-line tools
# =============================================================================

set -euo pipefail

CRATE_LIB_NAME="audio_analyzer_rs"
BUILD_FLAG="--release"
BUILD_SUBDIR="release"

ANDROID_OUT_DIR="bindings/android/jniLibs"
KOTLIN_BINDINGS_DIR="bindings/kotlin"
SWIFT_BINDINGS_DIR="bindings/swift"
IOS_XCFRAMEWORK_DIR="bindings/ios"

OS="$(uname -s)"
MODE="${1:-}"

# =============================================================================
# Helpers
# =============================================================================
info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }
step()  { echo ""; echo "── $* ──────────────────────────────────────────────────"; }

# =============================================================================
# SETUP: Install Rust targets + cargo-ndk (idempotent)
# =============================================================================
setup() {
    step "Installing Rust cross-compilation targets"

    info "Android targets..."
    rustup target add aarch64-linux-android
    rustup target add armv7-linux-androideabi
    rustup target add i686-linux-android
    rustup target add x86_64-linux-android

    if [ "$OS" = "Darwin" ]; then
        info "iOS targets..."
        rustup target add aarch64-apple-ios
        rustup target add aarch64-apple-ios-sim
        rustup target add x86_64-apple-ios
    else
        warn "iOS targets skipped — requires macOS. Current OS: $OS"
    fi

    step "Installing cargo-ndk"
    cargo install cargo-ndk
}

# =============================================================================
# ANDROID BUILD
# =============================================================================
build_android() {
    step "Verifying Android NDK"

    if [ -z "${ANDROID_NDK_HOME:-}" ]; then
        error "ANDROID_NDK_HOME is not set.
  Windows (Git Bash): export ANDROID_NDK_HOME=\"/c/Users/<you>/AppData/Local/Android/Sdk/ndk/<version>\"
  macOS/Linux:        export ANDROID_NDK_HOME=\"\$HOME/Library/Android/sdk/ndk/<version>\"
  Typical NDK versions: 25.2.9519653 or 27.2.12479018"
    fi

    info "Using NDK: $ANDROID_NDK_HOME"
    mkdir -p "$ANDROID_OUT_DIR" "$KOTLIN_BINDINGS_DIR"

    step "Building Android targets (all ABIs)"
    # API level 26+ required for cpal/AAudio; set via [package.metadata.ndk] in Cargo.toml
    cargo ndk \
        --platform 26 \
        -t aarch64-linux-android \
        -t armv7-linux-androideabi \
        -t i686-linux-android \
        -t x86_64-linux-android \
        -o "$ANDROID_OUT_DIR" \
        build $BUILD_FLAG

    step "Generating Kotlin bindings"
    # Use the arm64 build to generate bindings (all targets produce the same API)
    ANDROID_LIB="target/aarch64-linux-android/${BUILD_SUBDIR}/lib${CRATE_LIB_NAME}.so"

    if [ ! -f "$ANDROID_LIB" ]; then
        error "Expected Android library not found: $ANDROID_LIB"
    fi

    cargo run --features bindgen --bin uniffi-bindgen generate \
        --library "$ANDROID_LIB" \
        --language kotlin \
        --out-dir "$KOTLIN_BINDINGS_DIR"

    echo ""
    info "Android build complete:"
    info "  .so files:       $ANDROID_OUT_DIR/"
    info "    arm64-v8a:     (aarch64)"
    info "    armeabi-v7a:   (armv7)"
    info "    x86:           (i686)"
    info "    x86_64:        (x86_64)"
    info "  Kotlin bindings: $KOTLIN_BINDINGS_DIR/"
}

# =============================================================================
# iOS BUILD (macOS only)
# =============================================================================
build_ios() {
    if [ "$OS" != "Darwin" ]; then
        error "iOS builds require macOS. Current OS: $OS
  Options:
    1. Run this script on a macOS machine or CI runner (e.g. GitHub Actions macos-latest)
    2. Use a cross-compilation service"
    fi

    command -v xcodebuild >/dev/null 2>&1 || error "xcodebuild not found. Install Xcode command-line tools: xcode-select --install"

    mkdir -p "$SWIFT_BINDINGS_DIR" "$IOS_XCFRAMEWORK_DIR"

    step "Building iOS device target (aarch64-apple-ios)"
    cargo build $BUILD_FLAG --target aarch64-apple-ios

    step "Building iOS simulator targets"
    cargo build $BUILD_FLAG --target aarch64-apple-ios-sim
    cargo build $BUILD_FLAG --target x86_64-apple-ios

    step "Creating universal simulator library (arm64 + x86_64)"
    mkdir -p "target/ios-sim-universal/${BUILD_SUBDIR}"
    lipo -create \
        "target/aarch64-apple-ios-sim/${BUILD_SUBDIR}/lib${CRATE_LIB_NAME}.a" \
        "target/x86_64-apple-ios/${BUILD_SUBDIR}/lib${CRATE_LIB_NAME}.a" \
        -output "target/ios-sim-universal/${BUILD_SUBDIR}/lib${CRATE_LIB_NAME}.a"

    step "Generating Swift bindings (needed for XCFramework headers)"
    cargo run --features bindgen --bin uniffi-bindgen generate \
        --library "target/aarch64-apple-ios/${BUILD_SUBDIR}/lib${CRATE_LIB_NAME}.a" \
        --language swift \
        --out-dir "$SWIFT_BINDINGS_DIR"

    # Copy headers from Swift bindings output for XCFramework
    HEADERS_DIR="target/ios-xcframework-headers"
    mkdir -p "$HEADERS_DIR"
    cp "${SWIFT_BINDINGS_DIR}/${CRATE_LIB_NAME}FFI.h"         "$HEADERS_DIR/"
    cp "${SWIFT_BINDINGS_DIR}/${CRATE_LIB_NAME}FFI.modulemap" "$HEADERS_DIR/module.modulemap"

    step "Creating XCFramework"
    XCFRAMEWORK_PATH="${IOS_XCFRAMEWORK_DIR}/${CRATE_LIB_NAME}.xcframework"
    rm -rf "$XCFRAMEWORK_PATH"

    xcodebuild -create-xcframework \
        -library "target/aarch64-apple-ios/${BUILD_SUBDIR}/lib${CRATE_LIB_NAME}.a" \
        -headers "$HEADERS_DIR" \
        -library "target/ios-sim-universal/${BUILD_SUBDIR}/lib${CRATE_LIB_NAME}.a" \
        -headers "$HEADERS_DIR" \
        -output "$XCFRAMEWORK_PATH"

    echo ""
    info "iOS build complete:"
    info "  Swift bindings:  $SWIFT_BINDINGS_DIR/"
    info "    ${CRATE_LIB_NAME}.swift        (Swift wrapper)"
    info "    ${CRATE_LIB_NAME}FFI.h         (C header)"
    info "    ${CRATE_LIB_NAME}FFI.modulemap (module map)"
    info "  XCFramework:     $XCFRAMEWORK_PATH"
}

# =============================================================================
# MAIN
# =============================================================================
case "$MODE" in
    setup)
        setup
        ;;
    android)
        setup
        build_android
        ;;
    ios)
        setup
        build_ios
        ;;
    "")
        setup
        build_android
        if [ "$OS" = "Darwin" ]; then
            build_ios
        else
            warn "iOS build skipped (not on macOS). Run on macOS or use: ./build_mobile.sh ios"
        fi
        ;;
    *)
        error "Unknown mode: $MODE. Use: setup | android | ios | (no arg = all)"
        ;;
esac

echo ""
echo "Done."
