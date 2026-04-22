#!/usr/bin/env bash
# Build OCASmoke.wasm and stage it next to the HTML/JS loader.
#
# Uses Swift 6.3.1 because 6.2.3 deadlocks inside any @MainActor hop on WASM
# (see root workspace memory: feedback_wasm_swift_version_mainactor).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DIR="$SCRIPT_DIR/../../Examples/SmokeTest"
SDK="${OCA_SMOKE_SDK:-swift-6.3.1-RELEASE_wasm}"

echo "→ Building OCASmoke against SDK=$SDK"
cd "$SMOKE_DIR"
swift build --product OCASmoke --swift-sdk "$SDK" -c release

BUILT_WASM="$SMOKE_DIR/.build/wasm32-unknown-wasip1/release/OCASmoke.wasm"
if [[ ! -f "$BUILT_WASM" ]]; then
    echo "✗ Build succeeded but $BUILT_WASM is missing" >&2
    exit 1
fi

cp "$BUILT_WASM" "$SMOKE_DIR/web/OCASmoke.wasm"
echo "✓ Staged $(du -h "$SMOKE_DIR/web/OCASmoke.wasm" | awk '{print $1}') at Examples/SmokeTest/web/OCASmoke.wasm"
