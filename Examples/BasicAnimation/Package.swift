// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "BasicAnimation",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(path: "../.."),
        .package(url: "https://github.com/swiftwasm/JavaScriptKit", from: "0.37.0"),
    ],
    targets: [
        .executableTarget(
            name: "BasicAnimation",
            dependencies: [
                .product(name: "OpenCoreAnimation", package: "OpenCoreAnimation"),
                .product(name: "JavaScriptKit", package: "JavaScriptKit"),
            ]
        ),
    ]
)
