// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "OpenCoreAnimation",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "OpenCoreAnimation",
            targets: ["OpenCoreAnimation"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/1amageek/OpenCoreGraphics.git", branch: "main"),
        .package(url: "https://github.com/1amageek/swift-webgpu.git", branch: "main"),
    ],
    targets: [
        .target(
            name: "OpenCoreAnimation",
            dependencies: [
                .product(name: "OpenCoreGraphics", package: "OpenCoreGraphics"),
                .product(name: "SwiftWebGPU", package: "swift-webgpu", condition: .when(platforms: [.wasi])),
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "OpenCoreAnimationTests",
            dependencies: ["OpenCoreAnimation"]
        ),
    ]
)
