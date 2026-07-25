/// Hash set backed by contiguous element storage and primitive buckets.
///
/// This is the set counterpart of ``OpenAddressingHashMap`` and avoids the
/// failing Swift 6.4 WASM custom-element `Set` storage specialization.
internal struct OpenAddressingHashSet<Element: Hashable> {
    private static var minimumBucketCount: Int { 16 }

    private var elements: [Element] = []
    /// Zero is empty; an occupied bucket stores `elements` index + 1.
    private var buckets: [Int] = []

    internal init() {}

    internal init(_ elements: [Element]) {
        reserveCapacity(elements.count)
        for element in elements {
            _ = insert(element)
        }
    }

    internal var count: Int {
        elements.count
    }

    internal var isEmpty: Bool {
        elements.isEmpty
    }

    internal func contains(_ element: Element) -> Bool {
        guard !buckets.isEmpty else { return false }
        return occupiedBucket(for: element) != nil
    }

    @discardableResult
    internal mutating func insert(
        _ element: Element
    ) -> (inserted: Bool, memberAfterInsert: Element) {
        ensureCapacity(forAdditionalElementCount: 1)
        let bucket = insertionBucket(for: element)
        let storedIndex = buckets[bucket]
        if storedIndex != 0 {
            return (false, elements[storedIndex - 1])
        }
        elements.append(element)
        buckets[bucket] = elements.count
        return (true, element)
    }

    @discardableResult
    internal mutating func remove(_ element: Element) -> Element? {
        guard let bucket = occupiedBucket(for: element) else { return nil }
        let removed = elements.remove(at: buckets[bucket] - 1)
        rebuildBuckets(keepingCapacity: true)
        return removed
    }

    internal mutating func removeAll(keepingCapacity: Bool = false) {
        elements.removeAll(keepingCapacity: keepingCapacity)
        if keepingCapacity {
            clearBuckets()
        } else {
            buckets.removeAll(keepingCapacity: false)
        }
    }

    internal mutating func formIntersection(
        _ other: OpenAddressingHashSet<Element>
    ) {
        var retained: [Element] = []
        retained.reserveCapacity(Swift.min(elements.count, other.elements.count))
        for element in elements where other.contains(element) {
            retained.append(element)
        }
        elements = retained
        rebuildBuckets(keepingCapacity: true)
    }

    internal mutating func formUnion(_ newElements: [Element]) {
        reserveCapacity(elements.count + newElements.count)
        for element in newElements {
            _ = insert(element)
        }
    }

    internal mutating func reserveCapacity(_ requestedCapacity: Int) {
        precondition(requestedCapacity >= 0)
        elements.reserveCapacity(requestedCapacity)
        let requiredBucketCount = Self.bucketCount(
            forElementCapacity: requestedCapacity
        )
        if requiredBucketCount > buckets.count {
            buckets = [Int](repeating: 0, count: requiredBucketCount)
            rebuildBuckets(keepingCapacity: true)
        }
    }

    private mutating func ensureCapacity(forAdditionalElementCount additionalCount: Int) {
        let requiredCount = elements.count + additionalCount
        if buckets.isEmpty {
            buckets = [Int](repeating: 0, count: Self.minimumBucketCount)
            elements.reserveCapacity(Self.maximumElementCount(
                forBucketCount: Self.minimumBucketCount
            ))
        } else if requiredCount > Self.maximumElementCount(forBucketCount: buckets.count) {
            buckets = [Int](repeating: 0, count: buckets.count * 2)
            rebuildBuckets(keepingCapacity: true)
        }
    }

    private func occupiedBucket(for element: Element) -> Int? {
        guard !buckets.isEmpty else { return nil }
        let bucket = insertionBucket(for: element)
        return buckets[bucket] == 0 ? nil : bucket
    }

    private func insertionBucket(for element: Element) -> Int {
        precondition(!buckets.isEmpty)
        let mask = buckets.count - 1
        var hasher = Hasher()
        element.hash(into: &hasher)
        var bucket = Int(UInt(bitPattern: hasher.finalize()) & UInt(mask))
        while true {
            let storedIndex = buckets[bucket]
            if storedIndex == 0 || elements[storedIndex - 1] == element {
                return bucket
            }
            bucket = (bucket + 1) & mask
        }
    }

    private mutating func rebuildBuckets(keepingCapacity: Bool) {
        if elements.isEmpty {
            if keepingCapacity {
                clearBuckets()
            } else {
                buckets.removeAll(keepingCapacity: false)
            }
            return
        }

        let requiredBucketCount = Self.bucketCount(
            forElementCapacity: elements.count
        )
        let bucketCount = keepingCapacity
            ? Swift.max(requiredBucketCount, buckets.count)
            : requiredBucketCount
        buckets = [Int](repeating: 0, count: bucketCount)
        for elementIndex in elements.indices {
            let bucket = insertionBucket(for: elements[elementIndex])
            buckets[bucket] = elementIndex + 1
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
