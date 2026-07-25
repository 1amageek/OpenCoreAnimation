#if arch(wasm32)
internal typealias GeometryCacheKeyMap<Value> =
    OpenAddressingHashMap<GeometryCacheKey, Value>
#endif
