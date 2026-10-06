import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct QwenLinearAttentionStateTests {
    private func geometry() throws -> QwenLinearAttentionGeometry {
        try QwenLinearAttentionGeometry(
            convolutionWidth: 4,
            convolutionChannelCount: 6,
            valueHeadCount: 2,
            keyHeadDimension: 2,
            valueHeadDimension: 3)
    }

    private func context() throws -> MetalContext { try MetalContext() }

    private func state(mask: [UInt8] = [1, 0, 1]) throws -> QwenLinearAttentionState {
        let context = try context()
        return try QwenLinearAttentionState(
            device: context.device,
            linearAttentionLayerMask: mask,
            geometry: geometry(),
            expectedLinearLayerCount: mask.filter { $0 == 1 }.count)
    }

    @Test func validatesWidthDimensionsAndLayerMasks() throws {
        #expect(throws: QwenLinearAttentionStateError.invalidGeometry(
            field: "convolutionWidth", value: 3)) {
            try QwenLinearAttentionGeometry(
                convolutionWidth: 3, convolutionChannelCount: 6,
                valueHeadCount: 2, keyHeadDimension: 2, valueHeadDimension: 3)
        }
        #expect(throws: QwenLinearAttentionStateError.invalidGeometry(
            field: "convolutionChannelCount", value: 0)) {
            try QwenLinearAttentionGeometry(
                convolutionWidth: 4, convolutionChannelCount: 0,
                valueHeadCount: 2, keyHeadDimension: 2, valueHeadDimension: 3)
        }
        let context = try context()
        let geometry = try geometry()
        #expect(throws: QwenLinearAttentionStateError.invalidLayerMaskCount(0)) {
            try QwenLinearAttentionState(
                device: context.device, linearAttentionLayerMask: [], geometry: geometry)
        }
        #expect(throws: QwenLinearAttentionStateError.invalidLayerMaskValue(layer: 1, value: 2)) {
            try QwenLinearAttentionState(
                device: context.device, linearAttentionLayerMask: [1, 2, 0], geometry: geometry)
        }
        #expect(throws: QwenLinearAttentionStateError.invalidLinearLayerCount(expected: 2, actual: 1)) {
            try QwenLinearAttentionState(
                device: context.device, linearAttentionLayerMask: [1, 0, 0],
                geometry: geometry, expectedLinearLayerCount: 2)
        }
    }

    @Test func officialGeometrySelectsExactlyThirtyOfFortyLinearLayers() throws {
        let context = try context()
        let mask = (0..<40).map { $0 % 4 == 3 ? UInt8(0) : UInt8(1) }
        let state = try QwenLinearAttentionState.official(
            device: context.device, linearAttentionLayerMask: mask)
        #expect(state.linearLayerIndices.count == 30)
        #expect(state.linearLayerIndices == (0..<40).filter { $0 % 4 != 3 })
        #expect(state.geometry.convolutionWidth == 4)
        #expect(state.geometry.convolutionChannelCount == 8_192)
        #expect(state.geometry.recurrentElementCount == 32 * 128 * 128)
        #expect(throws: QwenLinearAttentionStateError.notLinearAttentionLayer(3)) {
            _ = try state.layerState(3)
        }
    }

    @Test func cloneRestoreAndResetUseAllHistoryAndMatrixElements() throws {
        let state = try state()
        let before = try state.clone()
        let update = try state.reserveUpdate(layer: 0)
        writeFloats(Array(repeating: 1.25, count: update.convolutionHistory.length / 4), to: update.convolutionHistory)
        writeFloats(Array(repeating: -2.5, count: update.recurrentMatrix.length / 4), to: update.recurrentMatrix)
        let context = try context()
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try state.submit(update, on: commandBuffer)
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        let changed = try state.layerState(0)
        #expect(changed.convolutionHistory.allSatisfy { $0 == 1.25 })
        #expect(changed.recurrentMatrix.allSatisfy { $0 == -2.5 })

        try state.restore(before)
        #expect(try state.clone() == before)
        try state.reset()
        let reset = try state.layerState(0)
        #expect(reset.convolutionHistory.allSatisfy { $0 == 0 })
        #expect(reset.recurrentMatrix.allSatisfy { $0 == 0 })
        #expect(try state.layerState(2).convolutionHistory.allSatisfy { $0 == 0 })
    }

    @Test func abortAndPreSubmitCancelReleaseAdmissionWithoutPublishingStaging() throws {
        let state = try state()
        let committed = try state.clone()
        let update = try state.reserveUpdate(layer: 0)
        writeFloats([9, 8, 7, 6], to: update.convolutionHistory)
        try state.abort(update)
        #expect(try state.clone() == committed)
        #expect(throws: QwenLinearAttentionStateError.unknownUpdate) { try state.cancel(update) }

        let cancelled = try state.reserveUpdate(layer: 0)
        writeFloats([4, 3, 2, 1], to: cancelled.recurrentMatrix)
        try state.cancel(cancelled)
        #expect(try state.clone() == committed)
        let reacquired = try state.reserveUpdate(layer: 0)
        try state.abort(reacquired)
    }

    @Test func encodeAndSubmitCatchesEncodingFailureAndAllowsImmediateReacquire() throws {
        enum ExpectedFailure: Error { case encode }
        let state = try state()
        #expect(throws: ExpectedFailure.self) {
            try state.encodeAndSubmit(layer: 0, commandBuffer: try #require(context().queue.makeCommandBuffer())) { update in
                writeFloats(Array(repeating: 17, count: update.convolutionHistory.length / 4), to: update.convolutionHistory)
                throw ExpectedFailure.encode
            }
        }
        let reacquired = try state.reserveUpdate(layer: 0)
        try state.abort(reacquired)
    }

    @Test func submittedCancellationPreservesCommittedHistoryAndMatrixThenAllowsReuse() throws {
        let context = try context()
        let state = try state()
        let initial = try state.clone()
        let update = try state.reserveUpdate(layer: 0)
        writeFloats(Array(repeating: 3.25, count: update.convolutionHistory.length / 4), to: update.convolutionHistory)
        writeFloats(Array(repeating: -6.5, count: update.recurrentMatrix.length / 4), to: update.recurrentMatrix)
        let event = try #require(context.device.makeSharedEvent())
        let blocked = try #require(context.queue.makeCommandBuffer())
        blocked.encodeWaitForEvent(event, value: 1)
        try state.submit(update, on: blocked)
        try state.cancel(update)
        #expect(throws: QwenLinearAttentionStateError.activeGPUUse) { _ = try state.clone() }
        event.signaledValue = 1
        blocked.waitUntilCompleted()
        #expect(blocked.status == .completed)
        #expect(try state.clone() == initial)

        let reacquired = try state.reserveUpdate(layer: 0)
        writeFloats(Array(repeating: 0.75, count: reacquired.convolutionHistory.length / 4), to: reacquired.convolutionHistory)
        writeFloats(Array(repeating: 1.5, count: reacquired.recurrentMatrix.length / 4), to: reacquired.recurrentMatrix)
        let successful = try #require(context.queue.makeCommandBuffer())
        try state.submit(reacquired, on: successful)
        successful.waitUntilCompleted()
        let published = try state.layerState(0)
        #expect(published.convolutionHistory.allSatisfy { $0 == 0.75 })
        #expect(published.recurrentMatrix.allSatisfy { $0 == 1.5 })
    }

    @Test func waitUntilIdleReturnsImmediatelyWhenOwnerIsIdle() async throws {
        let state = try state()
        let finished = StateTestLatch()
        let waiter = Task {
            await state.waitUntilIdle()
            await finished.signal()
        }
        await finished.wait()
        await waiter.value
        #expect(await finished.isSignaled())
    }

    @Test func waitUntilIdleWakesAfterAbortAndPreSubmitCancel() async throws {
        let aborted = try state()
        let update = try aborted.reserveUpdate(layer: 0)
        let abortedFinished = StateTestLatch()
        let abortedWaiter = Task {
            await aborted.waitUntilIdle()
            await abortedFinished.signal()
        }
        await Task.yield()
        #expect(await abortedFinished.isSignaled() == false)
        try aborted.abort(update)
        await abortedWaiter.value
        #expect(await abortedFinished.isSignaled())
        let abortedSnapshot = try aborted.clone()
        let abortedLayer = try #require(abortedSnapshot.layers[0])
        #expect(abortedLayer.convolutionHistory.allSatisfy { $0 == 0 })

        let cancelled = try state()
        let cancelledUpdate = try cancelled.reserveUpdate(layer: 0)
        let cancelledFinished = StateTestLatch()
        let cancelledWaiter = Task {
            await cancelled.waitUntilIdle()
            await cancelledFinished.signal()
        }
        await Task.yield()
        #expect(await cancelledFinished.isSignaled() == false)
        try cancelled.cancel(cancelledUpdate)
        await cancelledWaiter.value
        #expect(await cancelledFinished.isSignaled())
        let cancelledSnapshot = try cancelled.clone()
        let cancelledLayer = try #require(cancelledSnapshot.layers[0])
        #expect(cancelledLayer.recurrentMatrix.allSatisfy { $0 == 0 })
    }

    @Test func waitUntilIdleStaysPendingForSubmittedCancelAndIgnoresWaiterCancellation() async throws {
        let context = try context()
        let state = try state()
        let initial = try state.clone()
        let update = try state.reserveUpdate(layer: 0)
        let event = try #require(context.device.makeSharedEvent())
        let blocked = try #require(context.queue.makeCommandBuffer())
        blocked.encodeWaitForEvent(event, value: 1)
        try state.submit(update, on: blocked)

        let finished = StateTestLatch()
        let waiter = Task {
            await state.waitUntilIdle()
            await finished.signal()
        }
        await Task.yield()
        try state.cancel(update)
        waiter.cancel()
        #expect(await finished.isSignaled() == false)
        #expect(throws: QwenLinearAttentionStateError.activeGPUUse) { _ = try state.clone() }

        event.signaledValue = 1
        _ = await blocked.completed()
        await waiter.value
        #expect(await finished.isSignaled())
        #expect(try state.clone() == initial)
    }

    @Test func waitUntilIdleResumesAfterOwnerCompletionAndAllowsRestore() async throws {
        let context = try context()
        let state = try state()
        let initial = try state.clone()
        let update = try state.reserveUpdate(layer: 0)
        writeFloats(Array(repeating: 0.75, count: update.convolutionHistory.length / 4), to: update.convolutionHistory)
        writeFloats(Array(repeating: 1.5, count: update.recurrentMatrix.length / 4), to: update.recurrentMatrix)
        let event = try #require(context.device.makeSharedEvent())
        let blocked = try #require(context.queue.makeCommandBuffer())
        blocked.encodeWaitForEvent(event, value: 1)
        try state.submit(update, on: blocked)

        let finished = StateTestLatch()
        let waiter = Task {
            await state.waitUntilIdle()
            await finished.signal()
        }
        await Task.yield()
        #expect(await finished.isSignaled() == false)
        event.signaledValue = 1
        _ = await blocked.completed()
        await waiter.value
        #expect(await finished.isSignaled())
        let published = try state.layerState(0)
        #expect(published.convolutionHistory.allSatisfy { $0 == 0.75 })
        #expect(published.recurrentMatrix.allSatisfy { $0 == 1.5 })
        let committed = try state.clone()
        try state.restore(initial)
        #expect(try state.clone() == initial)
        try state.restore(committed)
        #expect(try state.clone() == committed)
    }

    @Test func pendingUpdateBlocksSnapshotRestoreAndResetAndSeparatesLayers() throws {
        let state = try state()
        let snapshot = try state.clone()
        let update = try state.reserveUpdate(layer: 0)
        #expect(throws: QwenLinearAttentionStateError.activeGPUUse) { _ = try state.clone() }
        #expect(throws: QwenLinearAttentionStateError.activeGPUUse) { try state.restore(snapshot) }
        #expect(throws: QwenLinearAttentionStateError.activeGPUUse) { try state.reset() }
        try state.abort(update)

        let layerTwoBefore = try state.layerState(2)
        let layerZeroUpdate = try state.reserveUpdate(layer: 0)
        writeFloats(Array(repeating: 5, count: layerZeroUpdate.convolutionHistory.length / 4), to: layerZeroUpdate.convolutionHistory)
        let context = try context()
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try state.submit(layerZeroUpdate, on: commandBuffer)
        commandBuffer.waitUntilCompleted()
        #expect(try state.layerState(2) == layerTwoBefore)
        #expect(throws: QwenLinearAttentionStateError.notLinearAttentionLayer(1)) {
            _ = try state.reserveUpdate(layer: 1)
        }
    }
}

private actor StateTestLatch {
    private var signaled = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func signal() {
        guard !signaled else { return }
        signaled = true
        let pending = waiters
        waiters.removeAll(keepingCapacity: false)
        for waiter in pending { waiter.resume() }
    }

    func wait() async {
        if signaled { return }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func isSignaled() -> Bool {
        signaled
    }
}

private func writeFloats(_ values: [Float], to buffer: MTLBuffer) {
    values.withUnsafeBytes { bytes in
        if let address = bytes.baseAddress {
            memcpy(buffer.contents(), address, bytes.count)
        }
    }
}
