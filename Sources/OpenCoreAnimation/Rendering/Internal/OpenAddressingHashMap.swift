/// Hash map backed by parallel contiguous storage and primitive buckets.
///
/// Swift 6.4 WASM release reactors can trap while allocating the standard
/// library's custom-key `Dictionary` storage. This implementation retains
/// average constant-time operations without using that specialization.
internal struct OpenAddressingHashMap<Key: Hashable, Value> {
    private static var minimumBucketCount: Int { 16 }

    private var keyStorage: [Key] = []
    private var valueStorage: [Value] = []
    /// Zero is empty; an occupied bucket stores the storage index + 1.
    private var buckets: [Int] = []

    internal init() {}

    internal var count: Int {
        keyStorage.count
    }

    internal var isEmpty: Bool {
        keyStorage.isEmpty
    }

    internal var keys: [Key] {
        keyStorage
    }

    internal var values: [Value] {
        valueStorage
    }

    internal subscript(key: Key) -> Value? {
        get {
            guard let bucket = occupiedBucket(for: key) else { return nil }
            return valueStorage[buckets[bucket] - 1]
        }
        set {
            if let newValue {
                _ = updateValue(newValue, forKey: key)
            } else {
                _ = removeValue(forKey: key)
            }
        }
    }

    internal func forEach(
        _ body: (Key, Value) throws -> Void
    ) rethrows {
        for index in keyStorage.indices {
            try body(keyStorage[index], valueStorage[index])
        }
    }

    internal func compactMap<Result>(
        _ transform: (Key, Value) throws -> Result?
    ) rethrows -> [Result] {
        var result: [Result] = []
        result.reserveCapacity(keyStorage.count)
        for index in keyStorage.indices {
            if let transformed = try transform(keyStorage[index], valueStorage[index]) {
                result.append(transformed)
            }
        }
        return result
    }

    internal func filter(
        _ isIncluded: (Key, Value) throws -> Bool
    ) rethrows -> OpenAddressingHashMap<Key, Value> {
        var result = OpenAddressingHashMap<Key, Value>()
        result.reserveCapacity(keyStorage.count)
        for index in keyStorage.indices
        where try isIncluded(keyStorage[index], valueStorage[index]) {
            result[keyStorage[index]] = valueStorage[index]
        }
        return result
    }

    internal func sorted(
        by areInIncreasingOrder: (
            (key: Key, value: Value),
            (key: Key, value: Value)
        ) throws -> Bool
    ) rethrows -> [(key: Key, value: Value)] {
        var entries: [(key: Key, value: Value)] = []
        entries.reserveCapacity(keyStorage.count)
        for index in keyStorage.indices {
            entries.append((keyStorage[index], valueStorage[index]))
        }
        return try entries.sorted(by: areInIncreasingOrder)
    }

    @discardableResult
    internal mutating func updateValue(
        _ value: Value,
        forKey key: Key
    ) -> Value? {
        ensureCapacity(forAdditionalElementCount: 1)
        let bucket = insertionBucket(for: key)
        let storedIndex = buckets[bucket]
        if storedIndex != 0 {
            let storageIndex = storedIndex - 1
            let previous = valueStorage[storageIndex]
            valueStorage[storageIndex] = value
            return previous
        }
        keyStorage.append(key)
        valueStorage.append(value)
        buckets[bucket] = keyStorage.count
        return nil
    }

    @discardableResult
    internal mutating func removeValue(forKey key: Key) -> Value? {
        guard let bucket = occupiedBucket(for: key) else { return nil }
        let storageIndex = buckets[bucket] - 1
        keyStorage.remove(at: storageIndex)
        let removed = valueStorage.remove(at: storageIndex)
        rebuildBuckets(keepingCapacity: true)
        return removed
    }

    internal mutating func removeAll(keepingCapacity: Bool = false) {
        keyStorage.removeAll(keepingCapacity: keepingCapacity)
        valueStorage.removeAll(keepingCapacity: keepingCapacity)
        if keepingCapacity {
            clearBuckets()
        } else {
            buckets.removeAll(keepingCapacity: false)
        }
    }

    internal mutating func reserveCapacity(_ requestedCapacity: Int) {
        precondition(requestedCapacity >= 0)
        keyStorage.reserveCapacity(requestedCapacity)
        valueStorage.reserveCapacity(requestedCapacity)
        let requiredBucketCount = Self.bucketCount(
            forElementCapacity: requestedCapacity
        )
        if requiredBucketCount > buckets.count {
            buckets = [Int](repeating: 0, count: requiredBucketCount)
            rebuildBuckets(keepingCapacity: true)
        }
    }

    private mutating func ensureCapacity(forAdditionalElementCount additionalCount: Int) {
        let requiredCount = keyStorage.count + additionalCount
        if buckets.isEmpty {
            buckets = [Int](repeating: 0, count: Self.minimumBucketCount)
            let capacity = Self.maximumElementCount(
                forBucketCount: Self.minimumBucketCount
            )
            keyStorage.reserveCapacity(capacity)
            valueStorage.reserveCapacity(capacity)
        } else if requiredCount > Self.maximumElementCount(forBucketCount: buckets.count) {
            buckets = [Int](repeating: 0, count: buckets.count * 2)
            rebuildBuckets(keepingCapacity: true)
        }
    }

    private func occupiedBucket(for key: Key) -> Int? {
        guard !buckets.isEmpty else { return nil }
        let bucket = insertionBucket(for: key)
        return buckets[bucket] == 0 ? nil : bucket
    }

    private func insertionBucket(for key: Key) -> Int {
        precondition(!buckets.isEmpty)
        let mask = buckets.count - 1
        var hasher = Hasher()
        key.hash(into: &hasher)
        var bucket = Int(UInt(bitPattern: hasher.finalize()) & UInt(mask))
        while true {
            let storedIndex = buckets[bucket]
            if storedIndex == 0 || keyStorage[storedIndex - 1] == key {
                return bucket
            }
            bucket = (bucket + 1) & mask
        }
    }

    private mutating func rebuildBuckets(keepingCapacity: Bool) {
        if keyStorage.isEmpty {
            if keepingCapacity {
                clearBuckets()
            } else {
                buckets.removeAll(keepingCapacity: false)
            }
            return
        }

        let requiredBucketCount = Self.bucketCount(
            forElementCapacity: keyStorage.count
        )
        let bucketCount = keepingCapacity
            ? Swift.max(requiredBucketCount, buckets.count)
            : requiredBucketCount
        buckets = [Int](repeating: 0, count: bucketCount)
        for storageIndex in keyStorage.indices {
            let bucket = insertionBucket(for: keyStorage[storageIndex])
            buckets[bucket] = storageIndex + 1
        }
    }

    private mutating func clearBuckets() {
        for index in buckets.indices {
            buckets[index] = 0
        }
    }

    private static func bucketCount(forElementCapacity capacity: Int) -> Int {
        var bucketCount = minimumBucketCount
        while maximumElementCount(forBucketCount: bucketCount) < capacity {
            bucketCount *= 2
        }
        return bucketCount
    }

    private static func maximumElementCount(forBucketCount bucketCount: Int) -> Int {
        bucketCount * 3 / 4
    }
}
