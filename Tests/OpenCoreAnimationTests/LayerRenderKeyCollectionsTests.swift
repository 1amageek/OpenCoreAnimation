import Testing
@testable import OpenCoreAnimation

@Suite("Layer render key collections")
struct LayerRenderKeyCollectionsTests {
    private final class Owner {}

    @Test("Set preserves structural key identity through growth and removal")
    func setGrowthAndRemoval() {
        let owners = (0..<64).map { _ in Owner() }
        let keys = owners.enumerated().map { index, owner in
            LayerRenderKey(
                layer: ObjectIdentifier(owner),
                replicatorPath: index.isMultiple(of: 2)
                    ? []
                    : [
                        ReplicatorInstancePathComponent(
                            replicator: ObjectIdentifier(owner),
                            instanceIndex: index
                        )
                    ]
            )
        }
        var set = LayerRenderKeySet()
        set.reserveCapacity(keys.count)

        for key in keys {
            #expect(set.insert(key).inserted)
            #expect(!set.insert(key).inserted)
        }
        #expect(set.count == keys.count)
        #expect(keys.allSatisfy(set.contains))

        for key in keys.prefix(32) {
            #expect(set.remove(key) == key)
            #expect(!set.contains(key))
            #expect(set.remove(key) == nil)
        }
        #expect(set.count == 32)
        #expect(keys.suffix(32).allSatisfy(set.contains))
    }

    @Test("Set union and intersection retain equal reconstructed keys")
    func setUnionAndIntersection() {
        let owner = Owner()
        let component = ReplicatorInstancePathComponent(
            replicator: ObjectIdentifier(owner),
            instanceIndex: 3
        )
        let original = LayerRenderKey(
            layer: ObjectIdentifier(owner),
            replicatorPath: [component]
        )
        let reconstructed = LayerRenderKey(
            layer: ObjectIdentifier(owner),
            replicatorPath: [component]
        )
        var lhs = LayerRenderKeySet()
        lhs.formUnion([original])
        var rhs = LayerRenderKeySet()
        _ = rhs.insert(reconstructed)

        lhs.formIntersection(rhs)
        #expect(lhs.count == 1)
        #expect(lhs.contains(reconstructed))
    }

    @Test("Map preserves values through growth, updates, and removals")
    func mapGrowthUpdatesAndRemovals() {
        let owners = (0..<64).map { _ in Owner() }
        let keys = owners.map {
            LayerRenderKey(layer: ObjectIdentifier($0))
        }
        var map = LayerRenderKeyMap<Int>()

        for (index, key) in keys.enumerated() {
            #expect(map.updateValue(index, forKey: key) == nil)
        }
        #expect(map.count == keys.count)
        for (index, key) in keys.enumerated() {
            #expect(map[key] == index)
        }

        #expect(map.updateValue(999, forKey: keys[7]) == 7)
        #expect(map[keys[7]] == 999)
        #expect(map.removeValue(forKey: keys[7]) == 999)
        #expect(map[keys[7]] == nil)
        #expect(map.count == keys.count - 1)
    }
}
