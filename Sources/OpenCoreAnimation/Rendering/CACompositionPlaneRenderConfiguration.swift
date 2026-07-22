import Foundation

internal struct CACompositionPlaneRenderConfiguration {
    let size: SIMD2<Float>
    let viewportSize: SIMD2<Float>

    init(
        bounds: CGRect,
        viewportSize: CGSize
    ) throws(CACompositionFilterRenderFailure) {
        let originX = Float(bounds.minX)
        let originY = Float(bounds.minY)
        let width = Float(bounds.width)
        let height = Float(bounds.height)
        let viewportWidth = Float(viewportSize.width)
        let viewportHeight = Float(viewportSize.height)
        guard originX.isFinite,
              originY.isFinite,
              width.isFinite,
              height.isFinite,
              width > 0,
              height > 0,
              viewportWidth.isFinite,
              viewportHeight.isFinite,
              viewportWidth > 0,
              viewportHeight > 0 else {
            throw .invalidDisplayGeometry
        }
        size = SIMD2(width, height)
        self.viewportSize = SIMD2(viewportWidth, viewportHeight)
    }

    func validateDisplayTransform(
        columns: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
    ) throws(CACompositionFilterRenderFailure) {
        guard Self.columnsAreFinite(columns) else {
            throw .invalidDisplayTransform
        }
    }

    func standardVertices() -> [CACompositionPlaneVertex] {
        let white = SIMD4<Float>(repeating: 1)
        return [
            CACompositionPlaneVertex(position: SIMD2(0, 0), texCoord: .zero, color: white),
            CACompositionPlaneVertex(position: SIMD2(1, 0), texCoord: .zero, color: white),
            CACompositionPlaneVertex(position: SIMD2(0, 1), texCoord: .zero, color: white),
            CACompositionPlaneVertex(position: SIMD2(1, 0), texCoord: .zero, color: white),
            CACompositionPlaneVertex(position: SIMD2(1, 1), texCoord: .zero, color: white),
            CACompositionPlaneVertex(position: SIMD2(0, 1), texCoord: .zero, color: white),
        ]
    }

    func capturedVertices(
        samplingColumns: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
    ) throws(CACompositionFilterRenderFailure) -> [CACompositionPlaneVertex] {
        guard Self.columnsAreFinite(samplingColumns) else {
            throw .invalidSamplingTransform
        }

        func vertex(at position: SIMD2<Float>) throws(CACompositionFilterRenderFailure)
            -> CACompositionPlaneVertex {
            let local = SIMD4<Float>(position.x * size.x, position.y * size.y, 0, 1)
            let clip = samplingColumns.0 * local.x
                + samplingColumns.1 * local.y
                + samplingColumns.2 * local.z
                + samplingColumns.3 * local.w
            guard clip.x.isFinite,
                  clip.y.isFinite,
                  clip.w.isFinite,
                  abs(clip.w) > 0.000001 else {
                throw .invalidSamplingTransform
            }
            let viewportNumerator = SIMD2<Float>(
                (clip.x + clip.w) * 0.5,
                (clip.w - clip.y) * 0.5
            )
            guard viewportNumerator.x.isFinite,
                  viewportNumerator.y.isFinite else {
                throw .invalidSamplingTransform
            }
            return CACompositionPlaneVertex(
                position: position,
                texCoord: viewportNumerator,
                color: SIMD4<Float>(clip.w, 0, 0, 0)
            )
        }

        let minMin = try vertex(at: SIMD2(0, 0))
        let maxMin = try vertex(at: SIMD2(1, 0))
        let minMax = try vertex(at: SIMD2(0, 1))
        let maxMax = try vertex(at: SIMD2(1, 1))
        return [minMin, maxMin, minMax, maxMin, maxMax, minMax]
    }

    private static func columnsAreFinite(
        _ columns: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
    ) -> Bool {
        columnIsFinite(columns.0)
            && columnIsFinite(columns.1)
            && columnIsFinite(columns.2)
            && columnIsFinite(columns.3)
    }

    private static func columnIsFinite(_ column: SIMD4<Float>) -> Bool {
        column.x.isFinite
            && column.y.isFinite
            && column.z.isFinite
            && column.w.isFinite
    }
}
