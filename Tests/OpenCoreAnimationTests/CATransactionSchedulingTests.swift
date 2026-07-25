import Testing
@_spi(RendererDiagnostics) @testable import OpenCoreAnimation

extension CATransactionTestSuites {
@Suite("Browser scheduling")
struct BrowserScheduling {
    @Test("Timer identifiers preserve the JavaScript safe-integer contract")
    func timerIdentifierValidation() {
        let maximum = CATransactionBrowserTimerIdentifierValidator.maximumSafeInteger

        #expect(CATransactionBrowserTimerIdentifierValidator.identifier(1) == .success(1))
        #expect(CATransactionBrowserTimerIdentifierValidator.identifier(maximum)
            == .success(maximum))
        #expect(CATransactionBrowserTimerIdentifierValidator.identifier(nil)
            == .failure(.timerIdentifierUnavailable))
        #expect(CATransactionBrowserTimerIdentifierValidator.identifier(.infinity)
            == .failure(.timerIdentifierNonFinite))
        #expect(CATransactionBrowserTimerIdentifierValidator.identifier(0)
            == .failure(.timerIdentifierNonPositive(0)))
        #expect(CATransactionBrowserTimerIdentifierValidator.identifier(-1)
            == .failure(.timerIdentifierNonPositive(-1)))
        #expect(CATransactionBrowserTimerIdentifierValidator.identifier(1.5)
            == .failure(.timerIdentifierFractional(1.5)))
        #expect(CATransactionBrowserTimerIdentifierValidator.identifier(maximum + 2)
            == .failure(.timerIdentifierUnsafeInteger(maximum + 2)))
    }

    @Test("Native transaction scheduling diagnostics remain empty")
    func nativeSchedulingDiagnostics() {
        CATransaction.flush()

        #expect(CATransaction.implicitCommitSchedulingFailureCount == 0)
        #expect(CATransaction.lastImplicitCommitSchedulingFailure == nil)
    }

    @Test("Committing a blocking explicit transaction reschedules implicit work")
    @MainActor
    func explicitTransactionReschedulesImplicitWork() {
        CATransaction.flush()
        var completionCount = 0

        CATransaction.setCompletionBlock {
            completionCount += 1
        }
        CATransaction.begin()
        CATransaction.deliverScheduledImplicitCommitForTesting()
        #expect(completionCount == 0)

        CATransaction.commit()
        CATransaction.deliverScheduledImplicitCommitForTesting()
        #expect(completionCount == 1)
    }
}
}
