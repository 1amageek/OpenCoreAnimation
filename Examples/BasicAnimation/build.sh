#!/bin/bash

# OpenCoreAnimation BasicAnimation Demo Build Script
# This script builds the WASM module using PackageToJS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== OpenCoreAnimation BasicAnimation Build ==="

# Check for Swift WASM SDK
SWIFT_SDK="${SWIFT_SDK:-DEVELOPMENT-SNAPSHOT-2025-06-16-a_wasm}"
echo "Using Swift SDK: $SWIFT_SDK"

# Build using PackageToJS plugin
echo ""
echo "Building WASM module with PackageToJS..."
swift package --swift-sdk "$SWIFT_SDK" js -o Bundle

echo ""
echo "=== Build Complete ==="
echo ""
echo "To run the demo:"
echo "  1. Start a local HTTP server: python3 -m http.server 8080"
echo "  2. Open in a WebGPU-enabled browser: http://localhost:8080"
echo ""
