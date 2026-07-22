import Testing
@testable import OpenCoreAnimation

@Suite("CAEmitterCell Tests")
struct CAEmitterCellTests {
    @Test("Content sampling properties match Core Animation defaults")
    func contentSamplingDefaults() {
        let cell = CAEmitterCell()

        #expect(cell.contents == nil)
        #expect(cell.contentsRect == CGRect(x: 0, y: 0, width: 1, height: 1))
        #expect(cell.contentsScale == 1)
        #expect(cell.magnificationFilter == CALayerContentsFilter.linear.rawValue)
        #expect(cell.minificationFilter == CALayerContentsFilter.linear.rawValue)
        #expect(cell.minificationFilterBias == 0)
        #expect(
            CAEmitterCell.defaultValue(forKey: "magnificationFilter") as? String
                == CALayerContentsFilter.linear.rawValue
        )
        #expect(
            CAEmitterCell.defaultValue(forKey: "minificationFilter") as? String
                == CALayerContentsFilter.linear.rawValue
        )
        #expect(CAEmitterCell.defaultValue(forKey: "minificationFilterBias") == nil)
    }

    @Test("Content sampling properties preserve configured values")
    func contentSamplingConfiguration() {
        let cell = CAEmitterCell()
        cell.contentsRect = CGRect(x: 0.25, y: 0.5, width: 0.5, height: 0.25)
        cell.contentsScale = 2
        cell.magnificationFilter = CALayerContentsFilter.nearest.rawValue
        cell.minificationFilter = CALayerContentsFilter.trilinear.rawValue
        cell.minificationFilterBias = 1.5

        #expect(cell.contentsRect == CGRect(x: 0.25, y: 0.5, width: 0.5, height: 0.25))
        #expect(cell.contentsScale == 2)
        #expect(cell.magnificationFilter == CALayerContentsFilter.nearest.rawValue)
        #expect(cell.minificationFilter == CALayerContentsFilter.trilinear.rawValue)
        #expect(cell.minificationFilterBias == 1.5)
    }
}
