// The Swift Programming Language
// https://docs.swift.org/swift-book

// Re-export OpenCoreGraphics for convenience
#if canImport(CoreGraphics)
@_exported import CoreGraphics
#else
@_exported import OpenCoreGraphics
#endif

// WASM: Re-export SwiftWebGPU
#if arch(wasm32)
@_exported import SwiftWebGPU
#endif
