import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized)
struct QwenRetainedCheckpointTests {
    private func context() throws -> MetalContext {
        try MetalContext()
    }

    private func geometry() throws -> QwenLinearAttentionGeometry {
        try QwenLinearAttentionGeometry(
            convolutionWidth: 4,
            convolutionChannelCount: 6,
            valueHeadCount: 2,
            keyHeadDimension: 2,
            valueHeadDimension: 3)
    }

    private func state() throws -> QwenLinearAttentionState {
        let context = try context()
        return try QwenLinearAttentionState(
            device: context.device,
            linearAttentionLayerMask: [1, 0, 1],
            geometry: try geometry(),
            expectedLinearLayerCount: 2)
    }

    private func commit(
        _ state: QwenLinearAttentionState,
        context: MetalContext,
        historyValue: Float,
        recurrentValue: Float
    ) throws -> QwenLinearAttentionLayerState {
        let update = try state.reserveUpdate(layer: 0)
        writeFloats(
            Array(repeating: historyValue, count: update.convolutionHistory.length / MemoryLayout<Float>.stride),
            to: update.convolutionHistory)
        writeFloats(
            Array(repeating: recurrentValue, count: update.recurrentMatrix.length / MemoryLayout<Float>.stride),
            to: update.recurrentMatrix)
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try state.submit(update, on: commandBuffer)
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        return try state.layerState(0)
    }

    @Test
    func retainedCheckpointRestoresExactCommittedStateAfterASecondGPUCommit() throws {
        let context = try context()
        let state = try state()
        let old = try commit(state, context: context, historyValue: 1.25, recurrentValue: -2.5)
        #expect(old.convolutionHistory.allSatisfy { $0 == 1.25 })
        #expect(old.recurrentMatrix.allSatisfy { $0 == -2.5 })
        let checkpoint = try state.retainCheckpoint()

        let changed = try commit(state, context: context, historyValue: 9.25, recurrentValue: -7.5)
        #expect(changed != old)
        try state.restore(checkpoint)
        #expect(try state.layerState(0) == old)

        // The retained token is reusable for a retry after another committed update.
        let retry = try commit(state, context: context, historyValue: 4.5, recurrentValue: 3.75)
        #expect(retry.convolutionHistory.allSatisfy { $0 == 4.5 })
        #expect(retry.recurrentMatrix.allSatisfy { $0 == 3.75 })
        try state.restore(checkpoint)
        #expect(try state.layerState(0) == old)

        // A committed replacement and a later CPU reset must not invalidate
        // the same retained GPU buffers or change their saved bytes.
        try state.reset()
        try state.restore(checkpoint)
        #expect(try state.layerState(0) == old)
    }

    @Test
    func resetAndCPUCloneRestoreLeaveHeldCheckpointImmutable() throws {
        let context = try context()
        let state = try state()
        let old = try commit(state, context: context, historyValue: 2.25, recurrentValue: -4.75)
        let checkpoint = try state.retainCheckpoint()
        let cpuSnapshot = try state.clone()

        try state.reset()
        let zero = try state.layerState(0)
        #expect(zero.convolutionHistory.allSatisfy { $0 == 0 })
        #expect(zero.recurrentMatrix.allSatisfy { $0 == 0 })

        try state.restore(cpuSnapshot)
        #expect(try state.layerState(0) == old)

        try state.reset()
        try state.restore(checkpoint)
        #expect(try state.layerState(0) == old)
    }

    @Test
    func retainedCheckpointRejectsForeignOwnerWithoutChangingTarget() throws {
        let sourceContext = try context()
        let source = try state()
        let target = try state()
        _ = try commit(source, context: sourceContext, historyValue: 6.5, recurrentValue: -1.25)
        let checkpoint = try source.retainCheckpoint()
        let targetBefore = try target.clone()

        #expect(throws: QwenLinearAttentionStateError.invalidSnapshotOwner) {
            try target.restore(checkpoint)
        }
        #expect(try target.clone() == targetBefore)
    }

    @Test
    func retainedCheckpointRestoreWaitsForInFlightGPUUseAndCancelSettlement() throws {
        let context = try context()
        let state = try state()
        let old = try commit(state, context: context, historyValue: 0.75, recurrentValue: 1.5)
        let checkpoint = try state.retainCheckpoint()
        let update = try state.reserveUpdate(layer: 0)
        writeFloats(
            Array(repeating: 17, count: update.convolutionHistory.length / MemoryLayout<Float>.stride),
            to: update.convolutionHistory)

        let event = try #require(context.device.makeSharedEvent())
        let blocked = try #require(context.queue.makeCommandBuffer())
        blocked.encodeWaitForEvent(event, value: 1)
        try state.submit(update, on: blocked)

        #expect(throws: QwenLinearAttentionStateError.activeGPUUse) {
            try state.restore(checkpoint)
        }
        try state.cancel(update)
        #expect(throws: QwenLinearAttentionStateError.activeGPUUse) {
            try state.restore(checkpoint)
        }
        event.signaledValue = 1
        blocked.waitUntilCompleted()

        try state.restore(checkpoint)
        #expect(try state.layerState(0) == old)
        let reacquired = try state.reserveUpdate(layer: 0)
        try state.abort(reacquired)
    }
}

private func writeFloats(_ values: [Float], to buffer: MTLBuffer) {
    values.withUnsafeBytes { bytes in
        if let address = bytes.baseAddress {
            memcpy(buffer.contents(), address, bytes.count)
        }
    }
}
