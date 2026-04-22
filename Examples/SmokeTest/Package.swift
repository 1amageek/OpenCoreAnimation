// swift-tools-version: 6.0
//
// OpenCoreAnimation WASM smoke-test executable. Built with the same toolchain
// and runtime layout as OpenCoreGraphics/SmokeTest and megaman
// (`swift-wasmport`) so Playwright can exercise the CALayer →
// CAWebGPURenderer pipeline in headless Chromium.
//
// Builds with:
//   swift build --product OCASmoke --swift-sdk swift-6.3.1-RELEASE_wasm -c release
// then copy .build/wasm32-unknown-wasip1/release/OCASmoke.wasm into
// Examples/SmokeTest/web/ where server.mjs serves it.

import PackageDescription

let package = Package(
    name: "OCASmoke",
    platforms: [
        .macOS(.v15)
    ],
    dependencies: [
        .package(path: "../.."),
        .package(url: "https://github.com/swiftwasm/JavaScriptKit", from: "0.50.2"),
    ],
    targets: [
        .executableTarget(
            name: "OCASmoke",
            dependencies: [
                .product(name: "OpenCoreAnimation", package: "OpenCoreAnimation"),
                .product(name: "JavaScriptKit", package: "JavaScriptKit"),
                .product(name: "JavaScriptEventLoop", package: "JavaScriptKit"),
            ],
            linkerSettings: [
                .unsafeFlags([
                    // JavaScriptKit only binds against WASI-reactor modules.
                    // Encode the flag here so plain `swift build` produces a
                    // JavaScriptKit-compatible artifact (the CLI wrapper is
                    // not in the loop for this target).
                    "-Xclang-linker", "-mexec-model=reactor",
                    "-Xlinker", "--export=setup",
                ])
            ]
        ),
    ]
)
