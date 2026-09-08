import Darwin
import Foundation
import Metal

/// Serial runner-owned observation of the existing two full-attention encoders.
/// Sample storage is reused only after the runner's existing router-buffer wait.
/// No command buffers, dispatches, barriers, completion handlers or waits are added.
final class FullAttentionStageTiming {
    enum Pass: Int { case partial, combine }

    private static let layerCount = 5
    private static let sampleCount = layerCount * 4
    private static let resolvedBytes = sampleCount * MemoryLayout<MTLCounterResultTimestamp>.stride
    private static let forwardLimit = RuntimeMeasurementCapture.decodePositionLimit

    private struct LayerState {
        var layer: UInt64 = .max
        var encoded: UInt64 = 0
        var completion: UInt64 = 0 // 0 absent, 1 completed, 2 GPU error
        var fallback: UInt64 = 0
        var encodeCommitNanos: UInt64 = .max
        var waitNanos: UInt64 = .max
        var bufferGPUNanos: UInt64 = .max
        var driverNanos: UInt64 = .max
    }

    weak var capture: RuntimeMeasurementCapture?
    private let device: MTLDevice
    private let states: UnsafeMutablePointer<LayerState>
    private var sampleBuffer: (any MTLCounterSampleBuffer)?
    private var partialDescriptor: MTLComputePassDescriptor?
    private var combineDescriptor: MTLComputePassDescriptor?
    /// 0 available, 1 unsupported stage boundary, 2 missing set, 3 missing
    /// timestamp counter, 4 allocation failed, 5 unexpected layer geometry.
    private var unavailableReason: UInt64 = 0
    private var expectedLayers: UInt64 = 0
    private var ordinal = -1
    private var position: UInt64 = 0
    private var forwards = 0
    private var cpuBefore: UInt64 = 0
    private var gpuBefore: UInt64 = 0
    private var calibrationStarted = false

    init(device: MTLDevice, fullLayerMask: [UInt8], capture: RuntimeMeasurementCapture) {
        self.device = device
        self.capture = capture
        states = .allocate(capacity: Self.layerCount)
        states.initialize(repeating: LayerState(), count: Self.layerCount)
        var count = 0
        for (layer, isFull) in fullLayerMask.enumerated() where isFull != 0 {
            if layer < 32 { expectedLayers |= UInt64(1) << layer }
            if count < Self.layerCount { states[count].layer = UInt64(layer) }
            count += 1
        }
        var capabilities: UInt64 = 0
        if count != Self.layerCount || fullLayerMask.count > 32 {
            unavailableReason = 5
        } else {
            capabilities = configureCounterBuffer()
        }
        capture.record(.fullAttentionTimingConfiguration, 1, unavailableReason,
                       capabilities, UInt64(Self.sampleCount), UInt64(Self.forwardLimit))
        // Opaque driver allocation has no public byte-count API. Do not present
        // the resolved 160-byte payload or object allowance as its allocation.
        capture.record(.fullAttentionTimingStorage, UInt64(malloc_size(states)),
                       UInt64(Self.resolvedBytes), 4_096, .max, 17)
    }

    deinit {
        states.deinitialize(count: Self.layerCount)
        states.deallocate()
    }

    private func configureCounterBuffer() -> UInt64 {
        var capabilities: UInt64 = 0
        if device.supportsCounterSampling(.atStageBoundary) { capabilities |= 1 }
        if device.supportsCounterSampling(.atDispatchBoundary) { capabilities |= 2 }
        guard capabilities & 1 != 0 else { unavailableReason = 1; return capabilities }
        guard let set = device.counterSets?.first(where: {
            $0.name == MTLCommonCounterSet.timestamp.rawValue
        }) else { unavailableReason = 2; return capabilities }
        capabilities |= 4
        guard set.counters.contains(where: {
            $0.name == MTLCommonCounter.timestamp.rawValue
        }) else { unavailableReason = 3; return capabilities }
        capabilities |= 8
        let descriptor = MTLCounterSampleBufferDescriptor()
        descriptor.counterSet = set
        descriptor.storageMode = .shared
        descriptor.sampleCount = Self.sampleCount
        descriptor.label = "full-attention-stage-timestamps"
        do {
            sampleBuffer = try device.makeCounterSampleBuffer(descriptor: descriptor)
        } catch {
            unavailableReason = 4
            return capabilities
        }
        capabilities |= 16
        partialDescriptor = MTLComputePassDescriptor()
        combineDescriptor = MTLComputePassDescriptor()
        partialDescriptor?.dispatchType = .serial
        combineDescriptor?.dispatchType = .serial
        return capabilities
    }

    /// Bounds observation work to the first 128 single-token forwards of this
    /// request. This is measurement coverage, never a generation limit.
    func beginForward(position: Int) -> Bool {
        guard forwards < Self.forwardLimit else { return false }
        forwards += 1
        self.position = UInt64(position)
        ordinal = -1
        cpuBefore = 0
        gpuBefore = 0
        calibrationStarted = false
        for index in 0..<Self.layerCount {
            let layer = states[index].layer
            states[index] = LayerState(layer: layer)
        }
        return true
    }

    func beginLayer(_ layer: Int) {
        ordinal = -1
        for index in 0..<Self.layerCount where states[index].layer == UInt64(layer) {
            ordinal = index
            break
        }
    }

    /// If observation setup fails, create the original production encoder once.
    /// A measurement failure must never omit an otherwise valid dispatch.
    func makeEncoder(commandBuffer: MTLCommandBuffer, pass: Pass) -> MTLComputeCommandEncoder? {
        guard unavailableReason == 0, ordinal >= 0, let sampleBuffer else {
            return commandBuffer.makeComputeCommandEncoder()
        }
        let descriptor = pass == .partial ? partialDescriptor : combineDescriptor
        guard let descriptor, let attachment = descriptor.sampleBufferAttachments[0] else {
            states[ordinal].fallback |= UInt64(1) << pass.rawValue
            return commandBuffer.makeComputeCommandEncoder()
        }
        if !calibrationStarted {
            let reference = device.sampleTimestamps()
            cpuBefore = reference.cpu
            gpuBefore = reference.gpu
            calibrationStarted = true
        }
        let start = ordinal * 4 + pass.rawValue * 2
        attachment.sampleBuffer = sampleBuffer
        attachment.startOfEncoderSampleIndex = start
        attachment.endOfEncoderSampleIndex = start + 1
        guard let encoder = commandBuffer.makeComputeCommandEncoder(descriptor: descriptor) else {
            states[ordinal].fallback |= UInt64(1) << pass.rawValue
            return commandBuffer.makeComputeCommandEncoder()
        }
        states[ordinal].encoded |= UInt64(1) << pass.rawValue
        return encoder
    }

    /// Called immediately after the already-existing wait, before any error is
    /// thrown. Only scalar timing values survive; no command buffer is retained.
    func completedLayer(_ layer: Int, buffer: MTLCommandBuffer,
                        encodeCommitNanos: UInt64, waitNanos: UInt64) {
        guard ordinal >= 0, states[ordinal].layer == UInt64(layer) else { return }
        states[ordinal].completion = buffer.status == .completed ? 1 : 2
        states[ordinal].encodeCommitNanos = encodeCommitNanos
        states[ordinal].waitNanos = waitNanos
        guard buffer.status == .completed else { return }
        states[ordinal].bufferGPUNanos = Self.hostDuration(buffer.gpuStartTime, buffer.gpuEndTime)
        states[ordinal].driverNanos = Self.hostDuration(buffer.kernelStartTime, buffer.kernelEndTime)
    }

    private static func hostDuration(_ start: Double, _ end: Double) -> UInt64 {
        guard start.isFinite, end.isFinite, start > 0, end >= start else { return .max }
        return nanoseconds((end - start) * 1_000_000_000)
    }

    private static func nanoseconds(_ value: Double) -> UInt64 {
        guard value.isFinite, value >= 0, value < Double(UInt64.max) else { return .max }
        return UInt64(value)
    }

    /// Every encoded sampled pass reached its existing wait, even when later
    /// I/O, cancellation or another command-buffer error exits the forward.
    func finishForward(capture: RuntimeMeasurementCapture, succeeded: Bool) {
        let started = DispatchTime.now().uptimeNanoseconds
        var cpuAfter: UInt64 = 0, gpuAfter: UInt64 = 0
        if calibrationStarted {
            let reference = device.sampleTimestamps()
            cpuAfter = reference.cpu
            gpuAfter = reference.gpu
        }
        let calibrated = calibrationStarted && cpuBefore > 0 && gpuBefore > 0
            && cpuAfter > cpuBefore && gpuAfter > gpuBefore
        capture.record(.fullAttentionTimingCalibration, position, cpuBefore, gpuBefore,
                       cpuAfter, gpuAfter)
        var completedLayers: UInt64 = 0
        var anyCompletedSamples = false
        for index in 0..<Self.layerCount where states[index].completion == 1 {
            if states[index].layer < 32 { completedLayers |= UInt64(1) << states[index].layer }
            if states[index].encoded != 0 { anyCompletedSamples = true }
        }
        var flags: UInt64 = calibrated ? 1 : 0
        if succeeded { flags |= 1 << 2 }
        if forwards == Self.forwardLimit { flags |= 1 << 3 }
        if unavailableReason != 0 { flags |= 1 << 4 }
        autoreleasepool {
            let data: Data?
            if anyCompletedSamples, let sampleBuffer {
                data = try? sampleBuffer.resolveCounterRange(0..<Self.sampleCount)
            } else {
                data = nil
            }
            let resolved = data?.count == Self.resolvedBytes
            if resolved { flags |= 1 << 1 }
            func duration(_ start: UInt64, _ end: UInt64) -> UInt64 {
                guard calibrated, start != MTLCounterErrorValue, end != MTLCounterErrorValue,
                      start > 0, end >= start, start >= gpuBefore, end <= gpuAfter else { return .max }
                return Self.nanoseconds(Double(end - start) / Double(gpuAfter - gpuBefore)
                                        * Double(cpuAfter - cpuBefore))
            }
            for index in 0..<Self.layerCount {
                let state = states[index]
                guard state.layer != .max else { continue }
                var partialStart = UInt64.max, partialEnd = UInt64.max
                var combineStart = UInt64.max, combineEnd = UInt64.max
                if let data, resolved, state.completion == 1 {
                    data.withUnsafeBytes { bytes in
                        func timestamp(_ sample: Int) -> UInt64 {
                            bytes.loadUnaligned(fromByteOffset: sample * MemoryLayout<MTLCounterResultTimestamp>.stride,
                                                as: MTLCounterResultTimestamp.self).timestamp
                        }
                        if state.encoded & 1 != 0 {
                            partialStart = timestamp(index * 4)
                            partialEnd = timestamp(index * 4 + 1)
                        }
                        if state.encoded & 2 != 0 {
                            combineStart = timestamp(index * 4 + 2)
                            combineEnd = timestamp(index * 4 + 3)
                        }
                    }
                }
                let partial = duration(partialStart, partialEnd)
                let combine = duration(combineStart, combineEnd)
                let gap = partial != .max && combine != .max ? duration(partialEnd, combineStart) : .max
                var validity: UInt64 = partial != .max ? 1 : 0
                if combine != .max { validity |= 2 }
                if gap != .max { validity |= 4 }
                validity |= state.encoded << 8 | state.fallback << 16
                if state.completion == 1 { validity |= 1 << 24 }
                if state.completion == 2 { validity |= 1 << 25 }
                let identity = position | state.layer << 32
                capture.record(.fullAttentionTimingSamples, identity, partialStart, partialEnd,
                               combineStart, combineEnd)
                capture.record(.fullAttentionStageDurations, identity, partial, combine, gap, validity)
                // Router wait includes prior queued work and CPU wakeup. It is
                // not a stage duration or a measurement of pure queue latency.
                capture.record(.fullAttentionHostTiming, identity, state.encodeCommitNanos,
                               state.waitNanos, state.bufferGPUNanos, state.driverNanos)
            }
        }
        capture.record(.fullAttentionTimingForward, position, expectedLayers, completedLayers,
                       flags, DispatchTime.now().uptimeNanoseconds - started)
    }
}
