#if !canImport(CoreVideo)
/// A platform-neutral representation of a SMPTE time used by `CVTimeStamp`.
public struct CVSMPTETime: Sendable {
    public var subframes: Int16
    public var subframeDivisor: Int16
    public var counter: UInt32
    public var type: UInt32
    public var flags: UInt32
    public var hours: Int16
    public var minutes: Int16
    public var seconds: Int16
    public var frames: Int16

    public init(
        subframes: Int16 = 0,
        subframeDivisor: Int16 = 0,
        counter: UInt32 = 0,
        type: UInt32 = 0,
        flags: UInt32 = 0,
        hours: Int16 = 0,
        minutes: Int16 = 0,
        seconds: Int16 = 0,
        frames: Int16 = 0
    ) {
        self.subframes = subframes
        self.subframeDivisor = subframeDivisor
        self.counter = counter
        self.type = type
        self.flags = flags
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds
        self.frames = frames
    }
}

/// A platform-neutral display timestamp matching Core Video's public layout.
public struct CVTimeStamp: Sendable {
    public var version: UInt32
    public var videoTimeScale: Int32
    public var videoTime: Int64
    public var hostTime: UInt64
    public var rateScalar: Double
    public var videoRefreshPeriod: Int64
    public var smpteTime: CVSMPTETime
    public var flags: UInt64
    public var reserved: UInt64

    public init(
        version: UInt32 = 0,
        videoTimeScale: Int32 = 0,
        videoTime: Int64 = 0,
        hostTime: UInt64 = 0,
        rateScalar: Double = 0,
        videoRefreshPeriod: Int64 = 0,
        smpteTime: CVSMPTETime = CVSMPTETime(),
        flags: UInt64 = 0,
        reserved: UInt64 = 0
    ) {
        self.version = version
        self.videoTimeScale = videoTimeScale
        self.videoTime = videoTime
        self.hostTime = hostTime
        self.rateScalar = rateScalar
        self.videoRefreshPeriod = videoRefreshPeriod
        self.smpteTime = smpteTime
        self.flags = flags
        self.reserved = reserved
    }
}
#endif
