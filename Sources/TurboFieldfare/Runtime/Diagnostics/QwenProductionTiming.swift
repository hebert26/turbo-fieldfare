import Darwin
import Metal
import Synchronization

/// Numeric production-path observations, enabled only by a request collector.
/// No transaction hook is installed and no command boundary is introduced.
enum QwenProductionStage: UInt64 {
    case embedding = 1, router = 2, head = 3, moe = 4
    case fullQKV = 5, fullNormRoPE = 6, fullGate = 7, fullOutput = 8, fullGateOutput = 9
    case linearProjections = 10, linearConvolution = 11, linearProjectionsConvolution = 12
    case linearRecurrence = 13, linearOutput = 14, linearRecurrenceOutput = 15
    case groupedMoE = 16, expertMap = 17, cpuAttention = 18, sampler = 19, forward = 20
    case linearPreparedStep = 21
    case expertMapWorkerResume = 22
    case unknown = 255

    static func full(_ stage: String) -> Self {
        switch stage {
        case "qkv": .fullQKV
        case "normRoPE": .fullNormRoPE
        case "gate": .fullGate
        case "output": .fullOutput
        case "gate-output": .fullGateOutput
        default: .unknown
        }
    }
    static func linear(_ stage: String) -> Self {
        switch stage {
        case "projections": .linearProjections
        case "convolution": .linearConvolution
        case "projections-convolution": .linearProjectionsConvolution
        case "recurrence": .linearRecurrence
        case "output": .linearOutput
        case "recurrence-output": .linearRecurrenceOutput
        case "projections-convolution-preparation-recurrence-output": .linearPreparedStep
        default: .unknown
        }
    }
}

struct QwenProductionLocation: Sendable {
    let capture: RuntimeMeasurementCapture
    let position: Int
    let tokenCount: Int
    let layer: Int
    func atLayer(_ layer: Int) -> Self {
        Self(capture: capture, position: position, tokenCount: tokenCount, layer: layer)
    }
}

enum QwenProductionTimingMeasurement {
    @TaskLocal static var location: QwenProductionLocation?
    static func command(_ stage: QwenProductionStage, layer: Int? = nil) -> QwenProductionCommand? {
        guard let location else { return nil }
        return QwenProductionCommand(location: layer.map(location.atLayer) ?? location, stage: stage)
    }
    static func span(_ stage: QwenProductionStage, layer: Int? = nil,
                     synchronousCPU: Bool = false) -> QwenProductionSpan? {
        guard let location else { return nil }
        return QwenProductionSpan(location: layer.map(location.atLayer) ?? location,
                                  stage: stage, synchronousCPU: synchronousCPU)
    }
    static func workerHandoff() -> QwenProductionWorkerHandoff? {
        guard let location else { return nil }
        return QwenProductionWorkerHandoff(location: location)
    }
    static func uptime() -> UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }
    static func threadCPU() -> UInt64 {
        var value = timespec()
        guard clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0,
              value.tv_sec >= 0, value.tv_nsec >= 0 else { return 0 }
        return UInt64(value.tv_sec) * 1_000_000_000 + UInt64(value.tv_nsec)
    }
}

/// Only this numeric clock box crosses into the Metal/dispatch callback.
/// A late callback never writes to the request collector or waits for an actor.
final class QwenProductionCompletionClock: Sendable {
    private let clock = Mutex<UInt64>(0)
    func completed() {
        let observed = QwenProductionTimingMeasurement.uptime()
        clock.withLock { if $0 == 0 { $0 = observed } }
    }
    func snapshot() -> UInt64 { clock.withLock { $0 } }
}

/// The worker writes before resuming its existing continuation. This span is
/// worker-result-ready to the first timestamp after that continuation await,
/// not a measurement of the later caller actor's return or pure scheduling.
final class QwenProductionWorkerHandoff: Sendable {
    private let location: QwenProductionLocation
    private let clock = QwenProductionCompletionClock()
    init(location: QwenProductionLocation) { self.location = location }
    func completed() { clock.completed() }
    func resumed() {
        let afterAwait = QwenProductionTimingMeasurement.uptime()
        location.capture.recordQwenProductionSpan(location, stage: .expertMapWorkerResume,
            started: clock.snapshot(), ended: afterAwait, cpuStarted: 0, cpuEnded: 0)
    }
}

/// One sequential owner passes this reference through synchronous lease submit
/// and existing async settlement. The completion handler touches only its clock
/// box. All command metadata fields remain owned by the sequential caller.
final class QwenProductionCommand: @unchecked Sendable {
    let location: QwenProductionLocation
    let stage: QwenProductionStage
    private var submitBefore: UInt64 = 0
    private var submitAfter: UInt64 = 0
    private var waitBefore: UInt64 = 0
    private var recorded = false
    private let completionClock = QwenProductionCompletionClock()
    private var completionInstalled = false
    init(location: QwenProductionLocation, stage: QwenProductionStage) {
        self.location = location; self.stage = stage
    }
    func willCommit(_ command: MTLCommandBuffer) {
        submitBefore = QwenProductionTimingMeasurement.uptime()
        guard !completionInstalled else { return }
        completionInstalled = true
        let completionClock = self.completionClock
        // Register before commit. The existing awaited SDK completion remains
        // untouched, and callback order relative to it is not assumed.
        command.addCompletedHandler { _ in completionClock.completed() }
    }
    func didCommit() { submitAfter = QwenProductionTimingMeasurement.uptime() }
    func willWait() { waitBefore = QwenProductionTimingMeasurement.uptime() }
    func resumed(_ command: MTLCommandBuffer) {
        let resumed = QwenProductionTimingMeasurement.uptime()
        guard !recorded else { return }
        recorded = true
        // Read GPU clocks only after the existing completion wait returned.
        let kernelStart = command.kernelStartTime, kernelEnd = command.kernelEndTime
        let start = command.gpuStartTime, end = command.gpuEndTime
        let valid = command.status == .completed && command.error == nil
            && start.isFinite && end.isFinite && start > 0 && end >= start
        location.capture.recordQwenProductionCommand(location, stage: stage,
            submitBefore: submitBefore, submitAfter: submitAfter, waitBefore: waitBefore,
            resumed: resumed, gpuStartBits: start.bitPattern, gpuEndBits: end.bitPattern,
            flags: (valid ? 1 : 0) | (command.status == .completed ? 2 : 0),
            completionCallback: completionClock.snapshot(),
            kernelStartBits: kernelStart.bitPattern, kernelEndBits: kernelEnd.bitPattern,
            kernelFlags: Self.driverClockFlags(start: kernelStart, end: kernelEnd,
                completedSuccessfully: command.status == .completed && command.error == nil))
    }
    /// Bit 1 means sampled after settlement. Bit 2 means usable duration.
    /// Zero, nonfinite and reversed samples retain their raw bits as invalid.
    static func driverClockFlags(start: Double, end: Double,
                                 completedSuccessfully: Bool) -> UInt64 {
        1 | (completedSuccessfully && start.isFinite && end.isFinite
             && start > 0 && end > 0 && end >= start ? 2 : 0)
    }

    func submittedWithoutSettlement() {
        guard !recorded else { return }
        recorded = true
        location.capture.recordQwenProductionCommand(location, stage: stage,
            submitBefore: submitBefore, submitAfter: submitAfter, waitBefore: 0,
            resumed: 0, gpuStartBits: 0, gpuEndBits: 0, flags: 4,
            completionCallback: completionClock.snapshot())
    }
}

struct QwenProductionSpan {
    let location: QwenProductionLocation
    let stage: QwenProductionStage
    let started: UInt64
    let cpuStarted: UInt64
    init(location: QwenProductionLocation, stage: QwenProductionStage, synchronousCPU: Bool) {
        self.location = location; self.stage = stage
        started = QwenProductionTimingMeasurement.uptime()
        cpuStarted = synchronousCPU ? QwenProductionTimingMeasurement.threadCPU() : 0
    }
    func finish() {
        let ended = QwenProductionTimingMeasurement.uptime()
        let cpuEnded = cpuStarted > 0 ? QwenProductionTimingMeasurement.threadCPU() : 0
        location.capture.recordQwenProductionSpan(location, stage: stage, started: started,
                                                ended: ended, cpuStarted: cpuStarted, cpuEnded: cpuEnded)
    }
}
