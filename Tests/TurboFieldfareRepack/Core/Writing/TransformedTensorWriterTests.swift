import Foundation
import Testing
@testable import TurboFieldfareRepackCore

private enum WriterTestError: Error {
    case syncFailed
    case checkpointFailed
}

@Suite struct TransformedTensorWriterTests {
    @Test func writesRowLocalInt4ComponentsAtGoldenOffsetsAndScalarProgress() throws {
        let fixture = try TransformWriterFixture.make()
        defer { fixture.remove() }
        let layout = try QwenRepackPlanner.affineComponentSizes(shape: [2, 65], bitWidth: .int4)
        #expect(layout.valuesSize == 66)
        #expect(layout.scalesSize == 8)
        #expect(layout.biasesSize == 8)
        let request = try fixture.request(components: layout)
        let progress = try fixture.initialProgress(request: request)
        var committed: [RemoteTransformProgress] = []

        let result = try TransformedTensorWriter.write(
            request, progress: progress, audit: RepackAudit()) { committed.append($0) }

        let bytes = try Data(contentsOf: URL(fileURLWithPath: fixture.destinationPath))
        #expect(result.completedUnitCount == 4)
        #expect(committed.count == 4)
        #expect(committed.last?.totalCompletedUnitCount == 4)
        #expect(committed.allSatisfy {
            $0.requestID == request.requestID && $0.sourceChainSHA256.count == 64
        })
        #expect(bytes[0] == 0x50)
        #expect(bytes[32] == 0x00)
        #expect(bytes[33] == 0x50)
        #expect(bytes[65] == 0x00)
        #expect(bytes[66..<74].count == 8)
        #expect(bytes[74..<82].count == 8)
        #expect(result.maximumScratchBytes <= WriterCore.tileBytes)
    }

    @Test func shortWriteAndCancellationDoNotCommitAUnit() throws {
        let fixture = try TransformWriterFixture.make()
        defer { fixture.remove() }
        let layout = try QwenRepackPlanner.affineComponentSizes(shape: [2, 65], bitWidth: .int4)
        let request = try fixture.request(components: layout)
        let progress = try fixture.initialProgress(request: request)
        var committed: [RemoteTransformProgress] = []
        var shortWrite = TransformedTensorWriterOperations.production
        shortWrite.write = { _, _, bytes, _ in max(0, bytes.count - 1) }

        #expect(throws: RepackError.self) {
            _ = try TransformedTensorWriter.write(
                request, progress: progress, audit: RepackAudit(), operations: shortWrite) {
                    committed.append($0)
                }
        }
        #expect(committed.isEmpty)

        var cancellationChecks = 0
        var cancelled = TransformedTensorWriterOperations.production
        cancelled.cancellationCheck = {
            cancellationChecks += 1
            if cancellationChecks == 2 { throw CancellationError() }
        }
        #expect(throws: CancellationError.self) {
            _ = try TransformedTensorWriter.write(
                request, progress: progress, audit: RepackAudit(), operations: cancelled) {
                    committed.append($0)
                }
        }
        #expect(committed.isEmpty)
    }

    @Test func shortReadFailsBeforeAnyUnitIsCommitted() throws {
        let fixture = try TransformWriterFixture.make()
        defer { fixture.remove() }
        let layout = try QwenRepackPlanner.affineComponentSizes(shape: [2, 65], bitWidth: .int4)
        let request = try fixture.request(components: layout)
        let progress = try fixture.initialProgress(request: request)
        var operations = TransformedTensorWriterOperations.production
        var reads = 0
        let productionRead = operations.read
        operations.read = { fd, path, offset, count in
            reads += 1
            let data = try productionRead(fd, path, offset, count)
            return reads == 2 ? Data(data.dropLast()) : data
        }
        var committed: [RemoteTransformProgress] = []
        #expect(throws: RepackError.self) {
            _ = try TransformedTensorWriter.write(
                request, progress: progress, audit: RepackAudit(), operations: operations) {
                    committed.append($0)
                }
        }
        #expect(committed.isEmpty)
    }

    @Test func affineDurabilityBatchPreservesOutputAndHashChains() throws {
        let scalarFixture = try TransformWriterFixture.make()
        defer { scalarFixture.remove() }
        let batchedFixture = try TransformWriterFixture.make()
        defer { batchedFixture.remove() }
        let layout = try QwenRepackPlanner.affineComponentSizes(shape: [2, 65], bitWidth: .int4)

        let scalarRequest = try scalarFixture.request(components: layout)
        var scalarOperations = TransformedTensorWriterOperations.production
        let scalarSync = scalarOperations.sync
        var scalarSyncCount = 0
        scalarOperations.sync = { fd, path in
            scalarSyncCount += 1
            try scalarSync(fd, path)
        }
        var scalarCommitted: [RemoteTransformProgress] = []
        let scalarResult = try TransformedTensorWriter.write(
            scalarRequest,
            progress: try scalarFixture.initialProgress(request: scalarRequest),
            audit: RepackAudit(),
            operations: scalarOperations) { scalarCommitted.append($0) }

        let batchedRequest = try batchedFixture.request(components: layout)
        var batchedOperations = TransformedTensorWriterOperations.production
        let batchedSync = batchedOperations.sync
        var batchedSyncCount = 0
        batchedOperations.sync = { fd, path in
            batchedSyncCount += 1
            try batchedSync(fd, path)
        }
        var batchedCommitted: [RemoteTransformProgress] = []
        let batchedResult = try TransformedTensorWriter.write(
            batchedRequest,
            progress: try batchedFixture.initialProgress(request: batchedRequest),
            audit: RepackAudit(),
            affineDurabilityInterval: 2,
            operations: batchedOperations) { batchedCommitted.append($0) }

        let scalarBytes = try Data(contentsOf: URL(fileURLWithPath: scalarFixture.destinationPath))
        let batchedBytes = try Data(contentsOf: URL(fileURLWithPath: batchedFixture.destinationPath))
        #expect(scalarResult.completedUnitCount == 4)
        #expect(batchedResult.completedUnitCount == 4)
        #expect(scalarSyncCount == 4)
        #expect(batchedSyncCount == 2)
        #expect(scalarCommitted.count == 4)
        #expect(batchedCommitted.count == 2)
        #expect(batchedCommitted.map(\.completedUnitCount) == [2, 4])
        #expect(scalarResult.progress.sourceChainSHA256 == batchedResult.progress.sourceChainSHA256)
        #expect(scalarResult.progress.destinationChainSHA256 == batchedResult.progress.destinationChainSHA256)
        #expect(scalarBytes == batchedBytes)
    }

    @Test func affineDurabilityBatchCommitsFinalPartialBatchExactlyOnce() throws {
        let fixture = try TransformWriterFixture.make()
        defer { fixture.remove() }
        let layout = try QwenRepackPlanner.affineComponentSizes(shape: [2, 65], bitWidth: .int4)
        let request = try fixture.request(components: layout)
        var operations = TransformedTensorWriterOperations.production
        let productionSync = operations.sync
        var syncCount = 0
        operations.sync = { fd, path in
            syncCount += 1
            try productionSync(fd, path)
        }
        var committed: [RemoteTransformProgress] = []

        let result = try TransformedTensorWriter.write(
            request,
            progress: try fixture.initialProgress(request: request),
            audit: RepackAudit(),
            affineDurabilityInterval: 3,
            operations: operations) { committed.append($0) }

        #expect(result.completedUnitCount == 4)
        #expect(syncCount == 2)
        #expect(committed.map(\.completedUnitCount) == [3, 4])
        #expect(committed.last?.sourceChainSHA256 == result.progress.sourceChainSHA256)
        #expect(committed.last?.destinationChainSHA256 == result.progress.destinationChainSHA256)
    }

    @Test func cancellationBeforeDurabilityBoundaryResumesToCleanOutput() throws {
        let interruptedFixture = try TransformWriterFixture.make()
        defer { interruptedFixture.remove() }
        let cleanFixture = try TransformWriterFixture.make()
        defer { cleanFixture.remove() }
        let layout = try QwenRepackPlanner.affineComponentSizes(shape: [2, 65], bitWidth: .int4)
        let interruptedRequest = try interruptedFixture.request(components: layout)
        var interruptedOperations = TransformedTensorWriterOperations.production
        let productionWrite = interruptedOperations.write
        var componentWrites = 0
        interruptedOperations.write = { fd, path, bytes, offset in
            componentWrites += 1
            return try productionWrite(fd, path, bytes, offset)
        }
        var cancellationChecks = 0
        interruptedOperations.cancellationCheck = {
            cancellationChecks += 1
            // The request validates its payload hash first. Checks 2...5
            // process and durably commit groups 1 and 2. Check 6 runs before
            // group 3, and check 7 runs after group 3 has been written but
            // before its interval-2 durability boundary.
            if cancellationChecks == 7 { throw CancellationError() }
        }
        var interruptedCommits: [RemoteTransformProgress] = []

        #expect(throws: CancellationError.self) {
            _ = try TransformedTensorWriter.write(
                interruptedRequest,
                progress: try interruptedFixture.initialProgress(request: interruptedRequest),
                audit: RepackAudit(),
                affineDurabilityInterval: 2,
                operations: interruptedOperations) { interruptedCommits.append($0) }
        }
        #expect(interruptedCommits.map(\.completedUnitCount) == [2])
        #expect(componentWrites == 9)
        guard let durableProgress = interruptedCommits.last else {
            throw WriterTestError.checkpointFailed
        }

        let resumed = try TransformedTensorWriter.write(
            interruptedRequest,
            progress: durableProgress,
            resumeUnitCount: durableProgress.completedUnitCount,
            audit: RepackAudit(),
            affineDurabilityInterval: 2) { _ in }

        let cleanRequest = try cleanFixture.request(components: layout)
        let clean = try TransformedTensorWriter.write(
            cleanRequest,
            progress: try cleanFixture.initialProgress(request: cleanRequest),
            audit: RepackAudit()) { _ in }
        let resumedBytes = try Data(contentsOf: URL(fileURLWithPath: interruptedFixture.destinationPath))
        let cleanBytes = try Data(contentsOf: URL(fileURLWithPath: cleanFixture.destinationPath))
        #expect(resumed.progress.sourceChainSHA256 == clean.progress.sourceChainSHA256)
        #expect(resumed.progress.destinationChainSHA256 == clean.progress.destinationChainSHA256)
        #expect(resumedBytes == cleanBytes)
    }

    @Test func syncFailureBeforeCheckpointDoesNotCommitTheBatch() throws {
        let fixture = try TransformWriterFixture.make()
        defer { fixture.remove() }
        let layout = try QwenRepackPlanner.affineComponentSizes(shape: [2, 65], bitWidth: .int4)
        let request = try fixture.request(components: layout)
        var operations = TransformedTensorWriterOperations.production
        var syncCount = 0
        operations.sync = { _, _ in
            syncCount += 1
            throw WriterTestError.syncFailed
        }
        var committed: [RemoteTransformProgress] = []

        #expect(throws: WriterTestError.self) {
            _ = try TransformedTensorWriter.write(
                request,
                progress: try fixture.initialProgress(request: request),
                audit: RepackAudit(),
                affineDurabilityInterval: 2,
                operations: operations) { committed.append($0) }
        }
        #expect(syncCount == 1)
        #expect(committed.isEmpty)
    }

    @Test func checkpointFailureOccursOnlyAfterDurabilitySync() throws {
        let fixture = try TransformWriterFixture.make()
        defer { fixture.remove() }
        let layout = try QwenRepackPlanner.affineComponentSizes(shape: [2, 65], bitWidth: .int4)
        let request = try fixture.request(components: layout)
        var operations = TransformedTensorWriterOperations.production
        var events: [String] = []
        var attemptedCheckpoints: [RemoteTransformProgress] = []
        let productionSync = operations.sync
        operations.sync = { fd, path in
            events.append("sync")
            try productionSync(fd, path)
        }

        #expect(throws: WriterTestError.self) {
            _ = try TransformedTensorWriter.write(
                request,
                progress: try fixture.initialProgress(request: request),
                audit: RepackAudit(),
                affineDurabilityInterval: 2,
                operations: operations) { updated in
                    events.append("commit")
                    attemptedCheckpoints.append(updated)
                    throw WriterTestError.checkpointFailed
                }
        }
        #expect(events == ["sync", "commit"])
        #expect(attemptedCheckpoints.count == 1)
        #expect(attemptedCheckpoints.map(\.completedUnitCount) == [2])
    }

    @Test func zeroAffineDurabilityIntervalIsRejected() throws {
        let fixture = try TransformWriterFixture.make()
        defer { fixture.remove() }
        let layout = try QwenRepackPlanner.affineComponentSizes(shape: [2, 65], bitWidth: .int4)
        let request = try fixture.request(components: layout)

        #expect(throws: RepackError.self) {
            _ = try TransformedTensorWriter.write(
                request,
                progress: try fixture.initialProgress(request: request),
                audit: RepackAudit(),
                affineDurabilityInterval: 0)
        }
    }

    @Test func retainedBF16IgnoresAffineDurabilityIntervalAndKeepsTileBoundary() throws {
        let fixture = try TransformWriterFixture.makeRetained()
        defer { fixture.remove() }
        let request = fixture.retainedRequest()
        var operations = TransformedTensorWriterOperations.production
        let productionSync = operations.sync
        var syncCount = 0
        operations.sync = { fd, path in
            syncCount += 1
            try productionSync(fd, path)
        }
        var committed: [RemoteTransformProgress] = []

        let result = try TransformedTensorWriter.write(
            request,
            progress: try fixture.initialProgress(request: request),
            audit: RepackAudit(),
            affineDurabilityInterval: 0,
            operations: operations) { committed.append($0) }

        let sourceBytes = try Data(contentsOf: URL(fileURLWithPath: fixture.sourcePath))
        let destinationBytes = try Data(contentsOf: URL(fileURLWithPath: fixture.destinationPath))
        #expect(result.completedUnitCount == 1)
        #expect(result.maximumScratchBytes == WriterCore.tileBytes)
        #expect(syncCount == 1)
        #expect(committed.count == 1)
        #expect(sourceBytes == destinationBytes)
    }
}

private struct TransformWriterFixture {
    let root: URL
    let sourcePath: String
    let destinationPath: String
    let source: SourceTensor

    static func make() throws -> Self {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbofieldfare-transform-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let sourceURL = root.appendingPathComponent("source.bf16")
        let destinationURL = root.appendingPathComponent("weights.bin")
        var values = [Float](arrayLiteral: 0, 1, 2, 3)
        values += Array(repeating: 0, count: 61)
        values += values
        var data = Data()
        for value in values {
            var bits = StreamingBF16AffineQuantizer.encodeBF16(value).littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        try data.write(to: sourceURL)
        try Data(repeating: 0, count: 82).write(to: destinationURL)
        return .init(root: root, sourcePath: sourceURL.path, destinationPath: destinationURL.path,
                     source: .init(name: "fixture.tensor", shardPath: sourceURL.path,
                                   dtype: .bf16, shape: [2, 65], absoluteOffset: 0,
                                   sizeBytes: UInt64(data.count)))
    }

    func request(components: QwenAffineComponentLayout) throws -> TransformedTensorWriteRequest {
        .init(requestIndex: 0, requestID: String(repeating: "a", count: 64),
              tensorName: source.name, source: source,
              sourcePayloadSHA256: try TransformedTensorWriter.sourcePayloadSHA256(source: source),
              destinationPath: destinationPath, destinationFileSize: 82,
              fileOffset: 0, storage: .affineInt4, affineComponents: components)
    }

    func retainedRequest() -> TransformedTensorWriteRequest {
        .init(requestIndex: 0, requestID: String(repeating: "a", count: 64),
              tensorName: source.name, source: source,
              sourcePayloadSHA256: nil, destinationPath: destinationPath,
              destinationFileSize: source.sizeBytes, fileOffset: 0,
              storage: .retainedBF16, affineComponents: nil)
    }

    func initialProgress(request: TransformedTensorWriteRequest) throws -> RemoteTransformProgress {
        let binding = RemoteTransformBinding(
            sourcePayloadSHA256: try TransformedTensorWriter.sourcePayloadSHA256(source: source),
            converterVersion: TransformedTensorWriter.converterVersion,
            quantizationPolicySHA256: String(repeating: "b", count: 64),
            planFingerprint: String(repeating: "c", count: 64),
            destinationIdentity: String(repeating: "d", count: 64),
            destinationBytes: request.destinationFileSize)
        return TransformedTensorWriter.initialProgress(binding: binding, firstRequestID: request.requestID)
    }

    static func makeRetained() throws -> Self {
        let fixture = try make()
        try Data(repeating: 0, count: Int(fixture.source.sizeBytes))
            .write(to: URL(fileURLWithPath: fixture.destinationPath))
        return fixture
    }

    func remove() { try? FileManager.default.removeItem(at: root) }
}
