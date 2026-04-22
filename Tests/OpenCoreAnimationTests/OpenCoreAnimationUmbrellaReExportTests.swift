//
//  OpenCoreAnimationUmbrellaReExportTests.swift
//  OpenCoreAnimationTests
//
//  Verifies that `import OpenCoreAnimation` alone is sufficient to bring
//  the underlying graphics types (CGFloat / CGRect / CGPoint / CGSize)
//  into scope. The umbrella file does this via `@_exported import
//  OpenCoreGraphics`. If that re-export breaks, this file fails to compile.
//
//  This file deliberately does NOT import CoreGraphics or OpenCoreGraphics
//  directly, so the only way CGFloat/CGRect/CGPoint/CGSize can resolve is
//  through OpenCoreAnimation's re-export.
//

import Testing
import OpenCoreAnimation

@Suite("OpenCoreAnimation umbrella re-exports graphics types")
struct OpenCoreAnimationUmbrellaReExportTests {

    @Test("CGFloat is in scope via OpenCoreAnimation")
    func cgFloatInScope() {
        let value: CGFloat = 1.5
        #expect(value == 1.5)
    }

    @Test("CGPoint is in scope via OpenCoreAnimation")
    func cgPointInScope() {
        let point = CGPoint(x: 3, y: 4)
        #expect(point.x == 3)
        #expect(point.y == 4)
    }

    @Test("CGSize is in scope via OpenCoreAnimation")
    func cgSizeInScope() {
        let size = CGSize(width: 10, height: 20)
        #expect(size.width == 10)
        #expect(size.height == 20)
    }

    @Test("CGRect is in scope via OpenCoreAnimation")
    func cgRectInScope() {
        let rect = CGRect(x: 1, y: 2, width: 30, height: 40)
        #expect(rect.origin.x == 1)
        #expect(rect.origin.y == 2)
        #expect(rect.size.width == 30)
        #expect(rect.size.height == 40)
    }

    @Test("CALayer accepts CGRect via re-exported graphics types")
    func caLayerConsumesReExportedTypes() {
        let layer = CALayer()
        layer.frame = CGRect(x: 0, y: 0, width: 50, height: 50)
        #expect(layer.frame.size.width == 50)
        #expect(layer.frame.size.height == 50)
    }
}
