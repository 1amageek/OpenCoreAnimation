//
//  OpenCoreAnimation.swift
//  OpenCoreAnimation
//
//  Umbrella file that re-exports the graphics module so source files inside
//  this module — and consumers of `import OpenCoreAnimation` — do not need to
//  repeat `import OpenCoreGraphics` (or `import Foundation`) at the top of
//  every file.
//
//  OpenCoreGraphics is the canonical graphics module for this module on every
//  platform — on WASM it is the only implementation, and on native builds it
//  provides additional types (e.g. `CGContextStatefulRendererDelegate`,
//  `CGClipPath`) that are not present in Apple's `CoreGraphics`. We also
//  re-export Foundation so that the geometry types (`CGFloat`, `CGPoint`,
//  `CGSize`, `CGRect`) are visible through this umbrella alone — on Darwin
//  Foundation transitively re-exports CoreGraphics, on swift-corelibs platforms
//  Foundation provides the geometry types directly.
//

@_exported import Foundation
@_exported import OpenCoreGraphics
