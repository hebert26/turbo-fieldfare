import Foundation
import TurboFieldfareDecodeProtocol

/// One command plus the exact lifecycle decision made when its frame arrived.
/// A refused command stays refused even if a later command reuses its caller IDs.
struct DecodeQueuedCommand: Sendable {
    enum Admission: Sendable {
        case none
        case load(DecodeServiceSession.LoadLease?)
        case unloadBound(DecodeServiceSession.TeardownLease?)
    }

    let command: DecodeServiceCommand
    let admission: Admission

    static func admitting(
        _ command: DecodeServiceCommand,
        using session: DecodeServiceSession
    ) -> Self {
        switch command {
        case .load(let request):
            Self(command: command, admission: .load(session.registerLoad(request)))
        case .unloadBound(let request):
            Self(
                command: command,
                admission: .unloadBound(session.registerBoundTeardown(
                    requestID: request.requestID, loadID: request.loadID)))
        default:
            Self(command: command, admission: .none)
        }
    }
}

final class DecodeCommandQueue: @unchecked Sendable {
    private let condition = NSCondition()
    private var commands: [DecodeQueuedCommand] = []
    private var closed = false

    func append(_ command: DecodeQueuedCommand) {
        condition.lock()
        commands.append(command)
        condition.signal()
        condition.unlock()
    }

    func close() {
        condition.lock()
        closed = true
        condition.broadcast()
        condition.unlock()
    }

    func next() -> DecodeQueuedCommand? {
        condition.lock()
        defer { condition.unlock() }
        while commands.isEmpty && !closed { condition.wait() }
        guard !commands.isEmpty else { return nil }
        return commands.removeFirst()
    }
}
