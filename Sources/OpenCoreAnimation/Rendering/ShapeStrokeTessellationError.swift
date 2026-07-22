internal enum ShapeStrokeTessellationError: Error, Equatable {
    case invalidGeometry
    case invalidDashPattern
    case unsupportedLineCap(String)
    case unsupportedLineJoin(String)
}
