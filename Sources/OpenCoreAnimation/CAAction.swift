
/// An interface that allows instances to respond to actions triggered by a Core Animation layer change.
///
/// When queried with an action identifier (a key path, an external action name, or a predefined action identifier)
/// a layer returns the appropriate action object–which must implement the CAAction protocol–and sends it a
/// `run(forKey:object:arguments:)` message.
public protocol CAAction {
    /// Called to trigger the action specified by the identifier.
    ///
    /// - Parameters:
    ///   - event: The identifier for the action.
    ///   - anObject: The layer on which the action should occur.
    ///   - dict: A dictionary containing parameters associated with this event, or `nil` if there are no parameters.
    func run(forKey event: String, object anObject: Any, arguments dict: [AnyHashable: Any]?)
}
