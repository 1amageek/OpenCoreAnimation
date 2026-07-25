import OpenCoreAnimationTLS

/// A typed boundary around the process runtime's real thread-local slot.
///
/// The C boundary owns no Swift object by itself. The caller establishes one
/// retained owner before `setValue(_:)`, and the registered destructor releases
/// that owner exactly once at thread exit. Clearing a slot transfers the retained
/// owner back to the caller, which must release it after `setValue(nil)` succeeds.
internal struct CATransactionThreadLocalSlot: Sendable {
    internal enum StorageError: Error, Equatable, Sendable {
        case initializationFailed(code: Int32)
        case updateFailed(code: Int32)
    }

    internal init(
        destructor: @escaping @convention(c) (UnsafeMutableRawPointer?) -> Void
    ) throws {
        let status = oca_tls_initialize(destructor)
        guard status == 0 else {
            throw StorageError.initializationFailed(code: status)
        }
    }

    internal func value() -> UnsafeMutableRawPointer? {
        oca_tls_get()
    }

    internal func setValue(_ value: UnsafeMutableRawPointer?) throws {
        let status = oca_tls_set(value)
        guard status == 0 else {
            throw StorageError.updateFailed(code: status)
        }
    }
}
