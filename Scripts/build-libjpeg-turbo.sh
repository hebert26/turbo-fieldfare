#!/bin/bash
set -euo pipefail

# Builds only pinned local source. No download, installation or runtime Python.
# Default includes both package platforms. --macos-only is the bounded decoder
# qualification build and produces a universal arm64/x86_64 macOS slice.
TF_JPEG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_JPEG_VENDOR="$TF_JPEG_ROOT/ThirdParty/libjpeg-turbo"
TF_JPEG_CMAKE="${TF_JPEG_CMAKE:-/opt/homebrew/bin/cmake}"
TF_JPEG_WITH_IOS=1
if [[ "${1:-}" == "--macos-only" ]]; then
    TF_JPEG_WITH_IOS=0
elif [[ $# -ne 0 ]]; then
    echo "Usage: Scripts/build-libjpeg-turbo.sh [--macos-only]" >&2
    exit 2
fi
cd "$TF_JPEG_VENDOR"
shasum -a 256 -c source-sha256.txt >/dev/null
mkdir -p "$TF_JPEG_ROOT/.build/libjpeg-turbo" "$TF_JPEG_VENDOR/Artifacts"
TF_JPEG_STAGE="$(mktemp -d "$TF_JPEG_VENDOR/Artifacts/.stage.XXXXXX")"
trap 'rm -rf "$TF_JPEG_STAGE"' EXIT

build_slice() {
    local sdk="$1" arch="$2" system="$3" deployment="$4"
    local build="$TF_JPEG_ROOT/.build/libjpeg-turbo/$sdk-$arch"
    local sdk_path
    sdk_path="$(xcrun --sdk "$sdk" --show-sdk-path)"
    "$TF_JPEG_CMAKE" --fresh -S "$TF_JPEG_VENDOR/3.0.2" -B "$build" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_NAME="$system" -DCMAKE_SYSTEM_PROCESSOR="$arch" \
        -DCMAKE_OSX_SYSROOT="$sdk_path" -DCMAKE_OSX_ARCHITECTURES="$arch" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment" -DBUILD=20240124 \
        -DENABLE_SHARED=OFF -DENABLE_STATIC=ON -DWITH_SIMD=OFF \
        -DWITH_TURBOJPEG=OFF -DWITH_JAVA=OFF -DWITH_JPEG7=OFF -DWITH_JPEG8=OFF
    "$TF_JPEG_CMAKE" --build "$build" --target jpeg-static --parallel 2
    mkdir -p "$TF_JPEG_STAGE/$sdk-headers"
    cp "$TF_JPEG_VENDOR/3.0.2/jpeglib.h" "$TF_JPEG_VENDOR/3.0.2/jmorecfg.h" \
        "$TF_JPEG_VENDOR/3.0.2/jerror.h" "$build/jconfig.h" "$TF_JPEG_STAGE/$sdk-headers/"
    cat > "$TF_JPEG_STAGE/$sdk-headers/TurboFieldfareLibJPEG.h" <<'HEADER'
#include <stddef.h>
#include <stdio.h>
#include "jpeglib.h"
#include "jerror.h"
HEADER
    cat > "$TF_JPEG_STAGE/$sdk-headers/module.modulemap" <<'MODULE'
module TurboFieldfareLibJPEG {
    umbrella header "TurboFieldfareLibJPEG.h"
    export *
}
MODULE
}

build_slice macosx arm64 Darwin 26.0
cp "$TF_JPEG_ROOT/.build/libjpeg-turbo/macosx-arm64/jconfig.h" "$TF_JPEG_STAGE/mac-arm64-jconfig.h"
build_slice macosx x86_64 Darwin 26.0
cmp "$TF_JPEG_STAGE/mac-arm64-jconfig.h" "$TF_JPEG_ROOT/.build/libjpeg-turbo/macosx-x86_64/jconfig.h"
xcrun lipo -create "$TF_JPEG_ROOT/.build/libjpeg-turbo/macosx-arm64/libjpeg.a" \
    "$TF_JPEG_ROOT/.build/libjpeg-turbo/macosx-x86_64/libjpeg.a" -output "$TF_JPEG_STAGE/libjpeg-macos.a"
TF_JPEG_XCF_ARGS=(-library "$TF_JPEG_STAGE/libjpeg-macos.a" -headers "$TF_JPEG_STAGE/macosx-headers")
if [[ "$TF_JPEG_WITH_IOS" == 1 ]]; then
    build_slice iphoneos arm64 iOS 26.0
    TF_JPEG_XCF_ARGS+=(-library "$TF_JPEG_ROOT/.build/libjpeg-turbo/iphoneos-arm64/libjpeg.a" -headers "$TF_JPEG_STAGE/iphoneos-headers")
    build_slice iphonesimulator arm64 iOS 26.0
    cp "$TF_JPEG_ROOT/.build/libjpeg-turbo/iphonesimulator-arm64/jconfig.h" "$TF_JPEG_STAGE/sim-arm64-jconfig.h"
    build_slice iphonesimulator x86_64 iOS 26.0
    cmp "$TF_JPEG_STAGE/sim-arm64-jconfig.h" "$TF_JPEG_ROOT/.build/libjpeg-turbo/iphonesimulator-x86_64/jconfig.h"
    xcrun lipo -create "$TF_JPEG_ROOT/.build/libjpeg-turbo/iphonesimulator-arm64/libjpeg.a" \
        "$TF_JPEG_ROOT/.build/libjpeg-turbo/iphonesimulator-x86_64/libjpeg.a" -output "$TF_JPEG_STAGE/libjpeg-ios-simulator.a"
    TF_JPEG_XCF_ARGS+=(-library "$TF_JPEG_STAGE/libjpeg-ios-simulator.a" -headers "$TF_JPEG_STAGE/iphonesimulator-headers")
fi
xcodebuild -create-xcframework "${TF_JPEG_XCF_ARGS[@]}" -output "$TF_JPEG_STAGE/LibJPEG.xcframework"
if [[ -e "$TF_JPEG_VENDOR/Artifacts/LibJPEG.xcframework" ]]; then
    rm -rf "$TF_JPEG_VENDOR/Artifacts/LibJPEG.xcframework"
fi
mv "$TF_JPEG_STAGE/LibJPEG.xcframework" "$TF_JPEG_VENDOR/Artifacts/LibJPEG.xcframework"
(
    cd "$TF_JPEG_VENDOR/Artifacts"
    find LibJPEG.xcframework -type f -print0 | sort -z | xargs -0 shasum -a 256 > built-files-sha256.txt
)
echo "Prepared $TF_JPEG_VENDOR/Artifacts/LibJPEG.xcframework (WITH_SIMD=OFF)"
