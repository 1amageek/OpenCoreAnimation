internal enum ShapeFillTessellationError: Error, Equatable {
    case unsupportedFillRule(String)
    case nonFinitePath
}
