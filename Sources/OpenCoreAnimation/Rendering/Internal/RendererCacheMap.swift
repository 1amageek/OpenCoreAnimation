#if arch(wasm32)
internal enum RendererCacheKey: Hashable {
    case textured(TexturedCacheKey)
    case text(TextTextureCacheKey)
    case emitterRemainder(EmitterBirthRemainderKey)
}

/// Fixed-key cache map that avoids custom-key `Dictionary` specialization on
/// Swift 6.4 WASM while retaining average O(1) lookup and mutation.
internal struct RendererCacheMap<Value> {
    private var storage = OpenAddressingHashMap<RendererCacheKey, Value>()

    internal init() {}

    internal var count: Int { storage.count }
    internal var values: [Value] { storage.values }

    internal var texturedKeys: [TexturedCacheKey] {
        storage.keys.compactMap {
            guard case .textured(let key) = $0 else { return nil }
            return key
        }
    }

    internal func filteringEmitterRemainders(
        _ isIncluded: (EmitterBirthRemainderKey, Value) throws -> Bool
    ) rethrows -> RendererCacheMap<Value> {
        var result = RendererCacheMap<Value>()
        try storage.forEach { storedKey, value in
            if case .emitterRemainder(let key) = storedKey,
               try isIncluded(key, value) {
                result[emitterRemainder: key] = value
            }
        }
        return result
    }

    internal subscript(textured key: TexturedCacheKey) -> Value? {
        get { value(for: .textured(key)) }
        set { setValue(newValue, for: .textured(key)) }
    }

    internal subscript(text key: TextTextureCacheKey) -> Value? {
        get { value(for: .text(key)) }
        set { setValue(newValue, for: .text(key)) }
    }

    internal subscript(emitterRemainder key: EmitterBirthRemainderKey) -> Value? {
        get { value(for: .emitterRemainder(key)) }
        set { setValue(newValue, for: .emitterRemainder(key)) }
    }

    @discardableResult
    internal mutating func updateValue(
        _ value: Value,
        forTextKey key: TextTextureCacheKey
    ) -> Value? {
        updateValue(value, for: .text(key))
    }

    @discardableResult
    internal mutating func removeValue(
        forTexturedKey key: TexturedCacheKey
    ) -> Value? {
        removeValue(for: .textured(key))
    }

    @discardableResult
    internal mutating func removeValue(
        forTextKey key: TextTextureCacheKey
    ) -> Value? {
        removeValue(for: .text(key))
    }

    @discardableResult
    internal mutating func removeValue(
        forEmitterRemainderKey key: EmitterBirthRemainderKey
    ) -> Value? {
        removeValue(for: .emitterRemainder(key))
    }

    internal mutating func removeAll(keepingCapacity: Bool = false) {
        storage.removeAll(keepingCapacity: keepingCapacity)
    }

    private func value(for key: RendererCacheKey) -> Value? {
        storage[key]
    }

    private mutating func setValue(_ value: Value?, for key: RendererCacheKey) {
        if let value {
            _ = updateValue(value, for: key)
        } else {
            _ = removeValue(for: key)
        }
    }

    private mutating func updateValue(
        _ value: Value,
        for key: RendererCacheKey
    ) -> Value? {
        storage.updateValue(value, forKey: key)
    }

    private mutating func removeValue(for key: RendererCacheKey) -> Value? {
        storage.removeValue(forKey: key)
    }
}
#endif
