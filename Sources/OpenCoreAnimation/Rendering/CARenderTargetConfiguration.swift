import Foundation

/// Validated integer and GPU viewport dimensions for a render target.
internal struct CARenderTargetConfiguration: Equatable {
    let width: Int
    let height: Int
    let viewportSize: SIMD2<Float>

    init(
        width: Double,
        height: Double,
        maximumTextureDimension: Int
    ) throws(CARenderTargetConfigurationError) {
        guard width.isFinite,
              height.isFinite,
              width > 0,
              height > 0,
              width.rounded(.towardZero) == width,
              height.rounded(.towardZero) == height,
              let integerWidth = Int(exactly: width),
              let integerHeight = Int(exactly: height),
              maximumTextureDimension > 0 else {
            throw .invalidDimensions(width: width, height: height)
        }
        guard integerWidth <= maximumTextureDimension,
              integerHeight <= maximumTextureDimension else {
            throw .dimensionLimitExceeded(
                width: integerWidth,
                height: integerHeight,
                maximum: maximumTextureDimension
            )
        }

        let viewportWidth = Float(integerWidth)
        let viewportHeight = Float(integerHeight)
        guard viewportWidth.isFinite, viewportHeight.isFinite else {
            throw .invalidDimensions(width: width, height: height)
        }

        self.width = integerWidth
        self.height = integerHeight
        viewportSize = SIMD2(viewportWidth, viewportHeight)
    }
}
