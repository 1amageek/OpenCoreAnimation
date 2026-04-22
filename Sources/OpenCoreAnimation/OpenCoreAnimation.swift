//
//  OpenCoreAnimation.swift
//  OpenCoreAnimation
//
//  Umbrella file that re-exports the graphics module so source files inside
//  this module do not need to repeat `import OpenCoreGraphics` at the top of
//  every file.
//
//  OpenCoreGraphics is the canonical graphics module for this module on every
//  platform — on WASM it is the only implementation, and on native builds it
//  provides additional types (e.g. `CGContextStatefulRendererDelegate`,
//  `CGClipPath`) that are not present in Apple's `CoreGraphics`.
//
//  Note: we deliberately do NOT `@_exported import Foundation` here. Doing so
//  would re-export Apple's Foundation through this module on Darwin, which on
//  macOS transitively exposes Apple's `CoreGraphics` to `@testable` importers
//  and collides with OpenCoreGraphics' retroactive conformances (CGPoint ==,
//  CGSize ==, CGRect ==, etc.). Individual source files still
//  `import Foundation` where they need it.
//

@_exported import OpenCoreGraphics
