import Darwin
import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareOfficialQwenSource

@Suite(.serialized) struct QwenBF16WeightsTests {
    @Test func bindsLiteralBF16BitsIntoSharedCompleteRowChunks() throws {
        let literal = QwenBF16LiteralTensor(
            name: "weights", rows: QwenBF16TestFixture.rows,
            columns: QwenBF16TestFixture.columns, bits: QwenBF16TestFixture.matrixBits)
        let source = try QwenBF16SyntheticSource.make(tensors: [literal])
        defer { source.remove() }
        let context = try MetalContext()
        let cap: UInt64 = 22 // 10-byte rows -> two, two, then one complete row.
        let weights = try QwenBF16Weights(
            context: context, source: source.handle,
            specifications: [spec(source, name: "weights", role: .dense,
                                  rows: 5, columns: 5)],
            residencyBudget: 50, maximumChunkBytes: cap,
            allocate: { device, length in
                device.makeBuffer(length: length, options: .storageModeShared)
            }, checkpoint: { _ in })

        let chunks = weights.inspectedChunks
        #expect(chunks.map(\.name) == ["weights", "weights", "weights"])
        #expect(chunks.map(\.firstRow) == [0, 2, 4])
        #expect(chunks.map(\.rowCount) == [2, 2, 1])
        #expect(chunks.map { $0.buffer.length } == [20, 20, 10])
        #expect(chunks.allSatisfy { $0.buffer.device === context.device })
        #expect(chunks.allSatisfy { $0.buffer.storageMode == .shared })
        #expect(chunks.allSatisfy {
            UInt64($0.buffer.length) <= min(UInt64(context.device.maxBufferLength), cap)
        })
        #expect(chunks.flatMap { readBF16Words($0.buffer) }
            == QwenBF16TestFixture.matrixBits)
        try weights.requireShape("weights", role: .dense, rows: 5, columns: 5)
        #expect(throws: QwenBF16WeightError.self) {
            try weights.requireShape("weights", role: .head, rows: 5, columns: 5)
        }
    }

    @Test func modelEntrypointBindsAnAdmittedTinyTensorWithoutFloatExpansion() throws {
        let source = try QwenBF16SyntheticSource.make(tensors: [
            QwenBF16LiteralTensor(
                name: "embedding", rows: 5, columns: 5,
                bits: QwenBF16TestFixture.matrixBits),
        ])
        defer { source.remove() }
        let context = try MetalContext()
        let weights = try QwenTextModel.makeBF16Weights(
            context: context, source: source.handle,
            specifications: [spec(source, name: "embedding", role: .embedding,
                                  rows: 5, columns: 5)],
            residencyBudget: 50)

        let chunks = weights.inspectedChunks
        #expect(chunks.count == 1)
        #expect(chunks[0].rowCount == 5)
        #expect(chunks[0].buffer.length == 5 * 5 * MemoryLayout<UInt16>.stride)
        #expect(chunks[0].buffer.length != 5 * 5 * MemoryLayout<Float>.stride)
        #expect(readBF16Words(chunks[0].buffer) == QwenBF16TestFixture.matrixBits)
    }

    @Test func rejectsShapeBudgetAndSubrowCapBeforeAllocationOrPayloadRead() throws {
        let tensors = [
            QwenBF16LiteralTensor(name: "a", rows: 5, columns: 5,
                                  bits: QwenBF16TestFixture.matrixBits),
            QwenBF16LiteralTensor(name: "b", rows: 5, columns: 5,
                                  bits: QwenBF16TestFixture.matrixBits),
        ]
        let source = try QwenBF16SyntheticSource.make(tensors: tensors)
        defer { source.remove() }
        let context = try MetalContext()

        var allocationCalls = 0
        var beforeAllocationCalls = 0
        var afterReadCalls = 0
        let allocator: (MTLDevice, Int) -> MTLBuffer? = { device, length in
            allocationCalls += 1
            return device.makeBuffer(length: length, options: .storageModeShared)
        }
        let checkpoint: (QwenBF16Weights.StageCheckpoint) throws -> Void = { stage in
            switch stage {
            case .beforeAllocation: beforeAllocationCalls += 1
            case .afterChunkRead: afterReadCalls += 1
            case .beforePublish: break
            }
        }

        expectWeightError({ if case .budgetExceeded = $0 { return true }; return false }) {
            _ = try QwenBF16Weights(
                context: context, source: source.handle,
                specifications: [
                    spec(source, name: "a", role: .dense, rows: 5, columns: 5),
                    spec(source, name: "b", role: .dense, rows: 5, columns: 5),
                ], residencyBudget: 99, maximumChunkBytes: 64,
                allocate: allocator, checkpoint: checkpoint)
        }
        #expect(allocationCalls == 0)
        #expect(beforeAllocationCalls == 0)
        #expect(afterReadCalls == 0)

        expectWeightError({ if case .invalidGeometry = $0 { return true }; return false }) {
            _ = try QwenBF16Weights(
                context: context, source: source.handle,
                specifications: [spec(source, name: "a", role: .dense,
                                      rows: 4, columns: 5)],
                residencyBudget: 100, maximumChunkBytes: 64,
                allocate: allocator, checkpoint: checkpoint)
        }
        #expect(allocationCalls == 0)
        #expect(beforeAllocationCalls == 0)
        #expect(afterReadCalls == 0)

        expectWeightError({ if case .invalidGeometry = $0 { return true }; return false }) {
            _ = try QwenBF16Weights(
                context: context, source: source.handle,
                specifications: [spec(source, name: "a", role: .dense,
                                      rows: 5, columns: 5)],
                residencyBudget: 50, maximumChunkBytes: 9,
                allocate: allocator, checkpoint: checkpoint)
        }
        #expect(allocationCalls == 0)
        #expect(beforeAllocationCalls == 0)
        #expect(afterReadCalls == 0)

        expectWeightError({ if case .invalidGeometry = $0 { return true }; return false }) {
            _ = try QwenBF16Weights(
                context: context, source: source.handle,
                specifications: [spec(source, name: "a", role: .dense,
                                      rows: 0, columns: 5)],
                residencyBudget: 50, maximumChunkBytes: 64,
                allocate: allocator, checkpoint: checkpoint)
        }
        #expect(allocationCalls == 0)
        #expect(beforeAllocationCalls == 0)
        #expect(afterReadCalls == 0)

        expectWeightError({ if case .invalidGeometry = $0 { return true }; return false }) {
            _ = try QwenBF16Weights(
                context: context, source: source.handle,
                specifications: [spec(source, name: "a", role: .dense,
                                      rows: 5, columns: 0)],
                residencyBudget: 50, maximumChunkBytes: 64,
                allocate: allocator, checkpoint: checkpoint)
        }
        #expect(allocationCalls == 0)
        #expect(beforeAllocationCalls == 0)
        #expect(afterReadCalls == 0)

        expectWeightError({ if case .invalidGeometry = $0 { return true }; return false }) {
            _ = try QwenBF16Weights(
                context: context, source: source.handle,
                specifications: [spec(source, name: "a", role: .dense,
                                      rows: Int(UInt32.max) + 1, columns: 5)],
                residencyBudget: 50, maximumChunkBytes: 64,
                allocate: allocator, checkpoint: checkpoint)
        }
        #expect(allocationCalls == 0)
        #expect(beforeAllocationCalls == 0)
        #expect(afterReadCalls == 0)
    }

    @Test func rejectsPrivateAndUndersizedAllocatorBuffersBeforePayloadRead() throws {
        let source = try QwenBF16SyntheticSource.make(tensors: [
            QwenBF16LiteralTensor(name: "weights", rows: 5, columns: 5,
                                  bits: QwenBF16TestFixture.matrixBits),
        ])
        defer { source.remove() }
        let context = try MetalContext()
        var afterReadCalls = 0
        let checkpoint: (QwenBF16Weights.StageCheckpoint) throws -> Void = { stage in
            if case .afterChunkRead = stage { afterReadCalls += 1 }
        }

        expectWeightError({ if case .invalidBuffer = $0 { return true }; return false }) {
            _ = try QwenBF16Weights(
                context: context, source: source.handle,
                specifications: [spec(source, name: "weights", role: .dense,
                                      rows: 5, columns: 5)],
                residencyBudget: 50, maximumChunkBytes: 50,
                allocate: { device, length in
                    device.makeBuffer(length: length, options: .storageModePrivate)
                }, checkpoint: checkpoint)
        }
        #expect(afterReadCalls == 0)

        expectWeightError({ if case .invalidBuffer = $0 { return true }; return false }) {
            _ = try QwenBF16Weights(
                context: context, source: source.handle,
                specifications: [spec(source, name: "weights", role: .dense,
                                      rows: 5, columns: 5)],
                residencyBudget: 50, maximumChunkBytes: 50,
                allocate: { device, length in
                    device.makeBuffer(length: length - 1, options: .storageModeShared)
                }, checkpoint: checkpoint)
        }
        #expect(afterReadCalls == 0)
    }

    @Test func nilSecondAllocationReleasesAllStagedBuffers() throws {
        let source = try QwenBF16SyntheticSource.make(tensors: [
            QwenBF16LiteralTensor(name: "a", rows: 5, columns: 5,
                                  bits: QwenBF16TestFixture.matrixBits),
            QwenBF16LiteralTensor(name: "b", rows: 1, columns: 5,
                                  bits: Array(QwenBF16TestFixture.matrixBits.prefix(5))),
        ])
        defer { source.remove() }
        let context = try MetalContext()
        var allocationCalls = 0
        var afterReadCalls = 0
        weak var firstBuffer: MTLBuffer?

        expectWeightError({
            if case .allocationFailed("a") = $0 { return true }
            return false
        }) {
            _ = try QwenBF16Weights(
                context: context, source: source.handle,
                specifications: [
                    spec(source, name: "a", role: .dense, rows: 5, columns: 5),
                    spec(source, name: "b", role: .dense, rows: 1, columns: 5),
                ], residencyBudget: 60, maximumChunkBytes: 10,
                allocate: { device, length in
                    allocationCalls += 1
                    guard allocationCalls == 1 else { return nil }
                    let buffer = device.makeBuffer(length: length, options: .storageModeShared)
                    firstBuffer = buffer
                    return buffer
                }, checkpoint: { stage in
                    if case .afterChunkRead = stage { afterReadCalls += 1 }
                })
        }
        #expect(allocationCalls == 2)
        #expect(afterReadCalls == 1)
        #expect(firstBuffer == nil)
    }

    @Test func cancellationAfterReadDropsStagedBufferWithoutPublishing() throws {
        let source = try QwenBF16SyntheticSource.make(tensors: [
            QwenBF16LiteralTensor(name: "weights", rows: 5, columns: 5,
                                  bits: QwenBF16TestFixture.matrixBits),
        ])
        defer { source.remove() }
        let context = try MetalContext()
        weak var stagedBuffer: MTLBuffer?
        var readStages = 0

        do {
            _ = try QwenBF16Weights(
                context: context, source: source.handle,
                specifications: [spec(source, name: "weights", role: .dense,
                                      rows: 5, columns: 5)],
                residencyBudget: 50, maximumChunkBytes: 50,
                allocate: { device, length in
                    let buffer = device.makeBuffer(length: length, options: .storageModeShared)
                    stagedBuffer = buffer
                    return buffer
                }, checkpoint: { stage in
                    if case .afterChunkRead = stage {
                        readStages += 1
                        throw CancellationError()
                    }
                })
            Issue.record("Expected cancellation after the direct source read")
        } catch is CancellationError {
            // Cancellation is injected after the destination buffer was filled.
        } catch {
            Issue.record("Expected CancellationError, got \(error)")
        }
        #expect(readStages == 1)
        #expect(stagedBuffer == nil)
    }

    @Test func sourceTruncationDuringLoadFailsAndReleasesPartialBuffer() throws {
        let source = try QwenBF16SyntheticSource.make(tensors: [
            QwenBF16LiteralTensor(name: "weights", rows: 5, columns: 5,
                                  bits: QwenBF16TestFixture.matrixBits),
        ])
        defer { source.remove() }
        let context = try MetalContext()
        weak var stagedBuffer: MTLBuffer?
        var didTruncate = false

        expectSourceError({ if case .replaced = $0 { return true }; return false }) {
            _ = try QwenBF16Weights(
                context: context, source: source.handle,
                specifications: [spec(source, name: "weights", role: .dense,
                                      rows: 5, columns: 5)],
                residencyBudget: 50, maximumChunkBytes: 50,
                allocate: { device, length in
                    let buffer = device.makeBuffer(length: length, options: .storageModeShared)
                    stagedBuffer = buffer
                    return buffer
                }, checkpoint: { stage in
                    if case .beforeAllocation = stage, !didTruncate {
                        didTruncate = true
                        let fd = open(source.shardURL.path, O_WRONLY | O_CLOEXEC)
                        guard fd >= 0 else {
                            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
                        }
                        defer { close(fd) }
                        guard ftruncate(fd, 0) == 0 else {
                            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
                        }
                    }
                })
        }
        #expect(didTruncate)
        #expect(stagedBuffer == nil)
    }

    @Test func mutationAfterChunkReadFailsFinalFreshnessCheckAndReleasesBuffer() throws {
        let source = try QwenBF16SyntheticSource.make(tensors: [
            QwenBF16LiteralTensor(name: "weights", rows: 5, columns: 5,
                                  bits: QwenBF16TestFixture.matrixBits),
        ])
        defer { source.remove() }
        let context = try MetalContext()
        weak var stagedBuffer: MTLBuffer?
        var didMutate = false

        expectSourceError({ if case .replaced = $0 { return true }; return false }) {
            _ = try QwenBF16Weights(
                context: context, source: source.handle,
                specifications: [spec(source, name: "weights", role: .dense,
                                      rows: 5, columns: 5)],
                residencyBudget: 50, maximumChunkBytes: 50,
                allocate: { device, length in
                    let buffer = device.makeBuffer(length: length, options: .storageModeShared)
                    stagedBuffer = buffer
                    return buffer
                }, checkpoint: { stage in
                    if case .afterChunkRead = stage, !didMutate {
                        didMutate = true
                        try mutateFirstPayloadByte(source, tensorName: "weights")
                    }
                })
        }
        #expect(didMutate)
        #expect(stagedBuffer == nil)
    }
}

private func spec(_ source: QwenBF16SyntheticSource, name: String,
                  role: QwenBF16TensorSpec.Role, rows: Int, columns: Int) -> QwenBF16TensorSpec {
    QwenBF16TensorSpec(name: name, shardName: source.shardName,
                       role: role, rows: rows, columns: columns)
}

private func readBF16Words(_ buffer: MTLBuffer) -> [UInt16] {
    let words = buffer.contents().assumingMemoryBound(to: UInt16.self)
    return Array(UnsafeBufferPointer(start: words, count: buffer.length / MemoryLayout<UInt16>.stride))
}

private func mutateFirstPayloadByte(_ source: QwenBF16SyntheticSource,
                                    tensorName: String) throws {
    guard let offset = source.tensorOffsets[tensorName] else {
        throw NSError(domain: "QwenBF16TestFixture", code: 1)
    }
    let fd = open(source.shardURL.path, O_WRONLY | O_CLOEXEC)
    guard fd >= 0 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
    defer { close(fd) }
    var byte: UInt8 = 0xa5
    let written = withUnsafeBytes(of: &byte) { raw -> Int in
        guard let base = raw.baseAddress else { return -1 }
        return Darwin.pwrite(fd, base, 1, off_t(offset))
    }
    guard written == 1 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
    let times = [timeval(tv_sec: 1, tv_usec: 0), timeval(tv_sec: 1, tv_usec: 0)]
    let result = times.withUnsafeBufferPointer { futimes(fd, $0.baseAddress) }
    guard result == 0 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
}

private func expectWeightError(
    _ matches: (QwenBF16WeightError) -> Bool,
    operation: () throws -> Void,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    do {
        try operation()
        Issue.record("Expected a QwenBF16WeightError", sourceLocation: sourceLocation)
    } catch let error as QwenBF16WeightError {
        #expect(matches(error), "Unexpected QwenBF16WeightError: \(error)",
                sourceLocation: sourceLocation)
    } catch {
        Issue.record("Expected QwenBF16WeightError, got \(error)", sourceLocation: sourceLocation)
    }
}

private func expectSourceError(
    _ matches: (OfficialSourceHandleError) -> Bool,
    operation: () throws -> Void,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    do {
        try operation()
        Issue.record("Expected an OfficialSourceHandleError", sourceLocation: sourceLocation)
    } catch let error as OfficialSourceHandleError {
        #expect(matches(error), "Unexpected OfficialSourceHandleError: \(error)",
                sourceLocation: sourceLocation)
    } catch {
        Issue.record("Expected OfficialSourceHandleError, got \(error)", sourceLocation: sourceLocation)
    }
}
