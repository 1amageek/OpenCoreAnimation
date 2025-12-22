// CGImageMetadataTag.swift
// OpenCoreAnimation
//
// Full API compatibility with Apple's CoreAnimation framework


/// An object that allows multiple animations to be grouped and run concurrently.
open class CAAnimationGroup: CAAnimation {

    /// An array of CAAnimation objects to be evaluated concurrently.
    open var animations: [CAAnimation]?
}
