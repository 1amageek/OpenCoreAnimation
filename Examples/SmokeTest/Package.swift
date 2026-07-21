// swift-tools-version: 6.0
//
// OpenCoreAnimation WASM smoke-test executable. Built with the same toolchain
// and runtime layout as the other Open* smoke tests so Playwright can
// exercise the CALayer → CAWebGPURenderer pipeline in headless Chromium.
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
        .package(path: "../../../OpenCoreImage"),
        .package(url: "https://github.com/1amageek/swift-wasm-testing", branch: "main"),
    ],
    targets: [
        .executableTarget(
            name: "OCASmoke",
            dependencies: [
                .product(name: "OpenCoreAnimation", package: "OpenCoreAnimation"),
                .product(name: "OpenCoreImage", package: "OpenCoreImage"),
                .product(name: "WasmTesting", package: "swift-wasm-testing"),
            ],
            linkerSettings: [
                .unsafeFlags([
                    // JavaScriptKit only binds against WASI-reactor modules.
                    // Encode the flag here so plain `swift build` produces a
                    // JavaScriptKit-compatible artifact.
                    "-Xclang-linker", "-mexec-model=reactor",
                    "-Xlinker", "--export=setup",
                ])
            ]
        ),
    ]
)
