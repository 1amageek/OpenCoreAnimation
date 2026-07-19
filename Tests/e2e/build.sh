#!/usr/bin/env bash
# Build OCASmoke.wasm and stage it next to the HTML/JS loader.
#
# Uses Swift 6.3.1 because 6.2.3 deadlocks inside any @MainActor hop on WASM
# (see root workspace memory: feedback_wasm_swift_version_mainactor).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$SCRIPT_DIR/../../Examples/SmokeTest"
SDK="${OCA_SMOKE_SDK:-swift-6.3.1-RELEASE_wasm}"
JAVASCRIPTKIT_VERSION="${OCA_JAVASCRIPTKIT_VERSION:-0.56.1}"

echo "→ Building OCASmoke against SDK=$SDK"
cd "$SMOKE_DIR"
swiftly run swift package resolve --version "$JAVASCRIPTKIT_VERSION" javascriptkit
swiftly run swift build --product OCASmoke --swift-sdk "$SDK" -c release

BUILT_WASM="$SMOKE_DIR/.build/wasm32-unknown-wasip1/release/OCASmoke.wasm"
JAVASCRIPTKIT_RUNTIME="$SMOKE_DIR/.build/checkouts/JavaScriptKit/Plugins/PackageToJS/Templates/runtime.mjs"
if [[ ! -f "$BUILT_WASM" ]]; then
    echo "✗ Build succeeded but $BUILT_WASM is missing" >&2
    exit 1
fi
if [[ ! -f "$JAVASCRIPTKIT_RUNTIME" ]]; then
    echo "✗ JavaScriptKit runtime was not found at $JAVASCRIPTKIT_RUNTIME" >&2
    exit 1
fi

cp "$BUILT_WASM" "$SMOKE_DIR/web/OCASmoke.wasm"
cp "$JAVASCRIPTKIT_RUNTIME" "$SMOKE_DIR/web/runtime.mjs"
echo "✓ Staged $(du -h "$SMOKE_DIR/web/OCASmoke.wasm" | awk '{print $1}') at Examples/SmokeTest/web/OCASmoke.wasm"
