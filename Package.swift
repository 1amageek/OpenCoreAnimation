// swift-tools-version: 6.4
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "OpenCoreAnimation",
    platforms: [
        .macOS(.v15),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "OpenCoreAnimation",
            targets: ["OpenCoreAnimation"]
        ),
    ],
    dependencies: [
        .package(path: "../OpenCoreGraphics"),
        .package(path: "../OpenCoreImage"),
        .package(path: "../swift-webgpu"),
    ],
    targets: [
        .target(
            name: "OpenCoreAnimation",
            dependencies: [
                "OpenCoreAnimationTLS",
                .product(name: "OpenCoreGraphics", package: "OpenCoreGraphics"),
                .product(name: "OpenCoreImage", package: "OpenCoreImage", condition: .when(platforms: [.wasi])),
                .product(name: "SwiftWebGPU", package: "swift-webgpu", condition: .when(platforms: [.wasi])),
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "OpenCoreAnimationTLS",
            path: "Sources/OpenCoreAnimationTLS",
            publicHeadersPath: "include"
        ),
        .testTarget(
            name: "OpenCoreAnimationTests",
            dependencies: ["OpenCoreAnimation"]
        ),
    ]
)
