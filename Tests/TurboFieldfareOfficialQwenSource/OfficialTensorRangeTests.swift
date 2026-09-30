import Darwin
import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfareOfficialQwenSource

/// Reads only a tiny, synthetic Safetensors file from the pinned allowlist.
/// The payload bytes and every expected slice are authored independently here.
@Suite(.serialized)
struct OfficialTensorRangeTests {
    @Test func readsExactLiteralBytesAcrossFourMiBBoundaryWithFourMiBRequests() throws {
        let caseFile = try makeFourMiBBoundaryFixture()
        defer { caseFile.fixture.remove() }
        let token = try caseFile.fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        var calls: [(count: Int, offset: off_t)] = []

        let result = try caseFile.fixture.handle.preadTensorRange(
            token, byteOffset: 0, byteCount: UInt64(caseFile.payload.count),
            expectedByteCount: UInt64(caseFile.payload.count),
            allocationBudget: UInt64(caseFile.payload.count),
            readAt: { fd, buffer, count, offset in
                calls.append((count: count, offset: offset))
                return darwinPread(fd, buffer, count, offset)
            }, checkpoint: { _ in }, limits: nil)

        #expect(result == Data(caseFile.payload))
        #expect(calls.map(\.count) == [FourMiBBoundaryFixture.tileBytes,
                                       caseFile.payload.count - FourMiBBoundaryFixture.tileBytes])
        #expect(calls[0].offset == off_t(token.admittedAbsoluteOffset))
        #expect(calls[1].offset == off_t(
            token.admittedAbsoluteOffset + UInt64(FourMiBBoundaryFixture.tileBytes)))
        #expect(calls.allSatisfy { $0.count <= FourMiBBoundaryFixture.tileBytes })
    }

    @Test func boundaryMutationAfterFirstChunkAndReplacementAfterFinalChunkReject() throws {
        let mutated = try makeFourMiBBoundaryFixture()
        defer { mutated.fixture.remove() }
        let mutatedToken = try mutated.fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        var mutatedDestination = [UInt8](repeating: 0xee, count: mutated.payload.count)
        var mutationCalls = 0
        var mutatedAtBoundary = false
        expectReplaced {
            try mutatedDestination.withUnsafeMutableBytes { destination in
                try mutated.fixture.handle.preadTensorRange(
                    mutatedToken, byteOffset: 0, byteCount: UInt64(mutated.payload.count),
                    expectedByteCount: UInt64(mutated.payload.count),
                    into: destination,
                    readAt: { fd, buffer, count, offset in
                        mutationCalls += 1
                        return darwinPread(fd, buffer, count, offset)
                    }, checkpoint: { point in
                        if case .afterPayloadRead = point, !mutatedAtBoundary {
                            mutatedAtBoundary = true
                            try overwriteByteInPlace(
                                at: mutated.fixture.shardURL,
                                offset: off_t(mutatedToken.admittedAbsoluteOffset
                                    + UInt64(FourMiBBoundaryFixture.tileBytes)),
                                with: 0x7a)
                        }
                    }, limits: nil)
            }
        }
        #expect(mutatedAtBoundary)
        #expect(mutationCalls == 1)
        #expect(Array(mutatedDestination.prefix(FourMiBBoundaryFixture.tileBytes))
                == Array(mutated.payload.prefix(FourMiBBoundaryFixture.tileBytes)))
        #expect(mutatedDestination.dropFirst(FourMiBBoundaryFixture.tileBytes)
                .allSatisfy { $0 == 0xee })

        let replaced = try makeFourMiBBoundaryFixture()
        defer { replaced.fixture.remove() }
        let replacedToken = try replaced.fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        var replacedDestination = [UInt8](repeating: 0xee, count: replaced.payload.count)
        var replacementCalls = 0
        var replacedAtFinalChunk = false
        expectReplaced {
            try replacedDestination.withUnsafeMutableBytes { destination in
                try replaced.fixture.handle.preadTensorRange(
                    replacedToken, byteOffset: 0, byteCount: UInt64(replaced.payload.count),
                    expectedByteCount: UInt64(replaced.payload.count),
                    into: destination,
                    readAt: { fd, buffer, count, offset in
                        replacementCalls += 1
                        return darwinPread(fd, buffer, count, offset)
                    }, checkpoint: { point in
                        if case .afterPayloadRead = point,
                           replacementCalls == 2, !replacedAtFinalChunk {
                            replacedAtFinalChunk = true
                            try replaceFile(at: replaced.fixture.shardURL,
                                            with: replaced.fileBytes)
                        }
                    }, limits: nil)
            }
        }
        #expect(replacedAtFinalChunk)
        #expect(replacementCalls == 2)
        #expect(replacedDestination == replaced.payload)
    }

    @Test func cancellationAfterFourMiBChunkReturnsPartialDestinationAndKeepsDescriptor() throws {
        let caseFile = try makeFourMiBBoundaryFixture()
        defer { caseFile.fixture.remove() }
        let token = try caseFile.fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let descriptorsWithToken = descriptorCount()
        var destination = [UInt8](repeating: 0xee, count: caseFile.payload.count)
        var calls = 0
        var observedCancellation = false

        do {
            try destination.withUnsafeMutableBytes { raw in
                try caseFile.fixture.handle.preadTensorRange(
                    token, byteOffset: 0, byteCount: UInt64(caseFile.payload.count),
                    expectedByteCount: UInt64(caseFile.payload.count),
                    into: raw,
                    readAt: { fd, buffer, count, offset in
                        calls += 1
                        return darwinPread(fd, buffer, count, offset)
                    }, checkpoint: { point in
                        if case .afterPayloadRead = point, calls == 1 {
                            throw CancellationError()
                        }
                    }, limits: nil)
            }
            Issue.record("Expected cancellation after the first 4 MiB payload chunk")
        } catch is CancellationError {
            observedCancellation = true
        } catch {
            Issue.record("Expected CancellationError, got \(error)")
        }

        #expect(observedCancellation)
        #expect(calls == 1)
        #expect(Array(destination.prefix(FourMiBBoundaryFixture.tileBytes))
                == Array(caseFile.payload.prefix(FourMiBBoundaryFixture.tileBytes)))
        #expect(destination.dropFirst(FourMiBBoundaryFixture.tileBytes)
                .allSatisfy { $0 == 0xee })
        #expect(descriptorCount() == descriptorsWithToken)
    }

    @Test func readsWholeTensorAndLiteralUnalignedOddLengthSlice() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)

        #expect(try fixture.handle.preadTensorRange(
            token, byteOffset: 0, byteCount: 16,
            expectedByteCount: 16, allocationBudget: 16)
            == Data([0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                     0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff]))
        #expect(try fixture.handle.preadTensorRange(
            token, byteOffset: 3, byteCount: 7,
            expectedByteCount: 7, allocationBudget: 7)
            == Data([0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99]))
    }

    @Test func permitsEmptyReadsAtTensorStartAndEnd() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)

        #expect(try fixture.handle.preadTensorRange(
            token, byteOffset: 0, byteCount: 0,
            expectedByteCount: 0, allocationBudget: 0).isEmpty)
        #expect(try fixture.handle.preadTensorRange(
            token, byteOffset: 16, byteCount: 0,
            expectedByteCount: 0, allocationBudget: 0).isEmpty)
    }

    @Test func rejectsMismatchedCountsBoundsOverflowAndCapacityBeforeAllocating() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let read: (UInt64, UInt64, UInt64, UInt64) throws -> Data = {
            byteOffset, byteCount, expectedByteCount, allocationBudget in
            try fixture.handle.preadTensorRange(
                token, byteOffset: byteOffset, byteCount: byteCount,
                expectedByteCount: expectedByteCount, allocationBudget: allocationBudget)
        }

        expectRangeError { try read(0, 5, 4, 5) }
        expectRangeError { try read(15, 2, 2, 2) }
        expectRangeError { try read(16, 1, 1, 1) }
        expectRangeError { try read(UInt64.max, 1, 1, 1) }
        expectRangeError { try read(UInt64.max - 1, 2, 2, 2) }
        expectRangeError { try read(0, 16, 16, 15) }
        expectRangeError {
            try read(0, UInt64(Int.max) + 1, UInt64(Int.max) + 1, UInt64.max)
        }
        expectRangeError { try read(0, UInt64(Int64.max) + 1, 0, 0) }

        var readCalls = 0
        var checkpointCalls = 0
        let injectedEOF: OfficialSourceHandle.TensorPread = { _, _, _, _ in
            readCalls += 1
            return .bytes(0)
        }
        let countedDarwinRead: OfficialSourceHandle.TensorPread = { fd, buffer, count, offset in
            readCalls += 1
            return darwinPread(fd, buffer, count, offset)
        }
        let countedCheckpoint: (OfficialSourceHandle.TensorReadCheckpoint) throws -> Void = { _ in
            checkpointCalls += 1
        }
        let countLimited = OfficialSourceHandle.TensorReadLimits(
            maximumOffset: UInt64.max, maximumAllocation: 15, maximumSyscall: 4)
        expectRangeError(reasonContains: "count, expected count or allocation budget invalid") {
            try fixture.handle.preadTensorRange(
                token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                allocationBudget: 16, readAt: injectedEOF,
                checkpoint: countedCheckpoint, limits: countLimited)
        }
        #expect(readCalls == 0)
        #expect(checkpointCalls == 0)

        let allowedEnd = UInt64(TinyTensorRangeFixture.validPayloadAbsoluteOffset + 3)
        let offsetLimited = OfficialSourceHandle.TensorReadLimits(
            maximumOffset: allowedEnd, maximumAllocation: 16, maximumSyscall: 16)
        #expect(try fixture.handle.preadTensorRange(
            token, byteOffset: 0, byteCount: 3, expectedByteCount: 3,
            allocationBudget: 3, readAt: countedDarwinRead,
            checkpoint: countedCheckpoint, limits: offsetLimited)
            == Data([0x00, 0x11, 0x22]))
        #expect(readCalls > 0)
        #expect(checkpointCalls > 0)
        let readsBeforeInvalidOffset = readCalls
        let checkpointsBeforeInvalidOffset = checkpointCalls
        expectRangeError {
            try fixture.handle.preadTensorRange(
                token, byteOffset: 0, byteCount: 4, expectedByteCount: 4,
                allocationBudget: 4, readAt: injectedEOF,
                checkpoint: countedCheckpoint, limits: offsetLimited)
        }
        #expect(readCalls == readsBeforeInvalidOffset)
        #expect(checkpointCalls == checkpointsBeforeInvalidOffset)

        let hardCapCount = OfficialSourceHandle.maximumTensorReadBytes + 1
        expectRangeError(reasonContains: "count, expected count or allocation budget invalid") {
            try fixture.handle.preadTensorRange(
                token, byteOffset: 0, byteCount: hardCapCount,
                expectedByteCount: hardCapCount, allocationBudget: UInt64.max,
                readAt: injectedEOF, checkpoint: countedCheckpoint, limits: nil)
        }
        #expect(readCalls == readsBeforeInvalidOffset)
        #expect(checkpointCalls == checkpointsBeforeInvalidOffset)

        let zeroSyscall = OfficialSourceHandle.TensorReadLimits(
            maximumOffset: UInt64.max, maximumAllocation: 16, maximumSyscall: 0)
        expectRangeError(reasonContains: "syscall capacity is zero") {
            try fixture.handle.preadTensorRange(
                token, byteOffset: 0, byteCount: 1, expectedByteCount: 1,
                allocationBudget: 1, readAt: injectedEOF,
                checkpoint: countedCheckpoint, limits: zeroSyscall)
        }
        #expect(readCalls == readsBeforeInvalidOffset)
        #expect(checkpointCalls == checkpointsBeforeInvalidOffset)
    }

    @Test func rejectsNonBF16MalformedAndUnallowlistedAdmission() throws {
        let nonBF16Header = #"{"synthetic.weight":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#
        let nonBF16 = try TinyTensorRangeFixture.make(
            fileBytes: TinyTensorRangeFixture.safetensorsFile(
                header: nonBF16Header, payload: TinyTensorRangeFixture.payload))
        defer { nonBF16.remove() }
        expectInvalidTensor {
            try nonBF16.handle.admitTensor(
                shardName: TinyTensorRangeFixture.shardName,
                tensorName: TinyTensorRangeFixture.tensorName)
        }

        let malformed = try TinyTensorRangeFixture.make(fileBytes: TinyTensorRangeFixture
            .safetensorsFile(declaredHeaderLength: 1, headerBytes: Data("{".utf8),
                             payload: TinyTensorRangeFixture.payload))
        defer { malformed.remove() }
        expectInvalidTensor {
            try malformed.handle.admitTensor(
                shardName: TinyTensorRangeFixture.shardName,
                tensorName: TinyTensorRangeFixture.tensorName)
        }

        let truncated = try TinyTensorRangeFixture.make(fileBytes: TinyTensorRangeFixture
            .safetensorsFile(declaredHeaderLength: 1_000, headerBytes: Data("{}".utf8),
                             payload: TinyTensorRangeFixture.payload))
        defer { truncated.remove() }
        expectInvalidTensor {
            try truncated.handle.admitTensor(
                shardName: TinyTensorRangeFixture.shardName,
                tensorName: TinyTensorRangeFixture.tensorName)
        }

        let invalidOffsetsHeader = #"{"synthetic.weight":{"dtype":"BF16","shape":[8],"data_offsets":[0,17]}}"#
        let invalidOffsets = try TinyTensorRangeFixture.make(fileBytes: TinyTensorRangeFixture
            .safetensorsFile(header: invalidOffsetsHeader, payload: TinyTensorRangeFixture.payload))
        defer { invalidOffsets.remove() }
        expectInvalidTensor {
            try invalidOffsets.handle.admitTensor(
                shardName: TinyTensorRangeFixture.shardName,
                tensorName: TinyTensorRangeFixture.tensorName)
        }

        let valid = try TinyTensorRangeFixture.make()
        defer { valid.remove() }
        expectNotAllowed {
            try valid.handle.admitTensor(
                shardName: "not-allowlisted.safetensors",
                tensorName: TinyTensorRangeFixture.tensorName)
        }
        expectInvalidTensor {
            try valid.handle.admitTensor(
                shardName: TinyTensorRangeFixture.shardName,
                tensorName: "missing.weight")
        }
    }

    @Test func directDestinationPreservesExactBytesForWholeAndUnalignedSlices() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)

        var whole = [UInt8](repeating: 0xee, count: 16)
        var wroteAtCallerDestination = false
        try whole.withUnsafeMutableBytes { destination in
            let base = try #require(destination.baseAddress)
            try fixture.handle.preadTensorRange(
                token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                into: destination,
                readAt: { fd, buffer, count, offset in
                    if buffer == base { wroteAtCallerDestination = true }
                    return darwinPread(fd, buffer, count, offset)
                }, checkpoint: { _ in },
                limits: .init(maximumOffset: UInt64.max,
                              maximumAllocation: 16, maximumSyscall: 3))
        }
        #expect(wroteAtCallerDestination)
        #expect(whole == TinyTensorRangeFixture.payload)

        var guarded = [UInt8](repeating: 0xcc, count: 9)
        guarded[0] = 0xa5
        guarded[8] = 0x5a
        try guarded.withUnsafeMutableBytes { raw in
            let slice = UnsafeMutableRawBufferPointer(
                start: raw.baseAddress!.advanced(by: 1), count: 7)
            try fixture.handle.preadTensorRange(
                token, byteOffset: 3, byteCount: 7, expectedByteCount: 7,
                into: slice)
        }
        #expect(guarded == [0xa5, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0x5a])
    }

    @Test func directDestinationRetriesEintrAndRejectsOverflowBeforeTouchingMemory() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        var oneByte: [UInt8] = [0xa5]
        var invalidReadCalls = 0
        var invalidCheckpointCalls = 0
        expectRangeError {
            try oneByte.withUnsafeMutableBytes { destination in
                try fixture.handle.preadTensorRange(
                    token, byteOffset: UInt64.max, byteCount: 1,
                    expectedByteCount: 1, into: destination,
                    readAt: { _, _, _, _ in
                        invalidReadCalls += 1
                        return .bytes(1)
                    }, checkpoint: { _ in invalidCheckpointCalls += 1 }, limits: nil)
            }
        }
        #expect(oneByte == [0xa5])
        #expect(invalidReadCalls == 0)
        #expect(invalidCheckpointCalls == 0)

        expectRangeError {
            try oneByte.withUnsafeMutableBytes { destination in
                try fixture.handle.preadTensorRange(
                    token, byteOffset: 0, byteCount: 2,
                    expectedByteCount: 2, into: destination,
                    readAt: { _, _, _, _ in
                        invalidReadCalls += 1
                        return .bytes(2)
                    }, checkpoint: { _ in invalidCheckpointCalls += 1 }, limits: nil)
            }
        }
        #expect(oneByte == [0xa5])
        #expect(invalidReadCalls == 0)
        #expect(invalidCheckpointCalls == 0)

        var destinationBytes = [UInt8](repeating: 0xee, count: 16)
        var injectedEintr = true
        var attempts = 0
        try destinationBytes.withUnsafeMutableBytes { destination in
            try fixture.handle.preadTensorRange(
                token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                into: destination,
                readAt: { fd, buffer, count, offset in
                    attempts += 1
                    if injectedEintr {
                        injectedEintr = false
                        return .interrupted
                    }
                    return darwinPread(fd, buffer, count, offset)
                }, checkpoint: { _ in },
                limits: .init(maximumOffset: UInt64.max,
                              maximumAllocation: 16, maximumSyscall: 4))
        }
        #expect(!injectedEintr)
        #expect(attempts > 1)
        #expect(destinationBytes == TinyTensorRangeFixture.payload)
    }

    @Test func directDestinationFailureAfterPrefixAndRealTruncationReturnNoSuccess() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let descriptorsWithToken = descriptorCount()
        var prefix = [UInt8](repeating: 0xee, count: 16)
        var didReadPrefix = false
        expectIOError(errno: EIO) {
            try prefix.withUnsafeMutableBytes { destination in
                try fixture.handle.preadTensorRange(
                    token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                    into: destination,
                    readAt: { fd, buffer, count, offset in
                        if didReadPrefix { return .failure(EIO) }
                        let result = darwinPread(fd, buffer, min(count, 4), offset)
                        if case .bytes(let got) = result, got > 0 { didReadPrefix = true }
                        return result
                    }, checkpoint: { _ in },
                    limits: .init(maximumOffset: UInt64.max,
                                  maximumAllocation: 16, maximumSyscall: 4))
            }
        }
        #expect(didReadPrefix)
        #expect(prefix == [0x00, 0x11, 0x22, 0x33] + Array(repeating: 0xee, count: 12))
        #expect(descriptorCount() == descriptorsWithToken)

        let truncated = try TinyTensorRangeFixture.make()
        defer { truncated.remove() }
        let truncatedToken = try truncated.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let truncatedDescriptors = descriptorCount()
        var shortDestination = [UInt8](repeating: 0xee, count: 16)
        var calls = 0
        var didTruncate = false
        expectShortRead {
            try shortDestination.withUnsafeMutableBytes { destination in
                try truncated.handle.preadTensorRange(
                    truncatedToken, byteOffset: 0, byteCount: 16,
                    expectedByteCount: 16, into: destination,
                    readAt: { fd, buffer, count, offset in
                        calls += 1
                        if calls == 2 {
                            didTruncate = true
                            let writer = open(truncated.shardURL.path, O_WRONLY | O_CLOEXEC)
                            guard writer >= 0 else { return .failure(errno) }
                            let result = ftruncate(writer, 0)
                            let error = errno
                            close(writer)
                            guard result == 0 else { return .failure(error) }
                        }
                        return darwinPread(fd, buffer, min(count, 4), offset)
                    }, checkpoint: { _ in },
                    limits: .init(maximumOffset: UInt64.max,
                                  maximumAllocation: 16, maximumSyscall: 4))
            }
        }
        #expect(didTruncate)
        #expect(shortDestination == [0x00, 0x11, 0x22, 0x33]
                + Array(repeating: 0xee, count: 12))
        #expect(descriptorCount() == truncatedDescriptors)
    }

    @Test func directDestinationMutationAfterPayloadReadReturnsNoSuccess() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let descriptorsWithToken = descriptorCount()
        var destinationBytes = [UInt8](repeating: 0xee, count: 16)
        var evidence: SameInodeMutationEvidence?

        expectReplaced {
            try destinationBytes.withUnsafeMutableBytes { destination in
                try fixture.handle.preadTensorRange(
                    token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                    into: destination, readAt: darwinPread,
                    checkpoint: { point in
                        if case .afterPayloadRead = point {
                            evidence = try overwriteFirstPayloadByteInPlace(fixture, with: 0x5a)
                        }
                    }, limits: nil)
            }
        }
        #expect(evidence?.sameInode == true)
        #expect(evidence?.sameSize == true)
        #expect(evidence?.modificationTimeChanged == true)
        #expect(destinationBytes == TinyTensorRangeFixture.payload)
        #expect(descriptorCount() == descriptorsWithToken)
    }

    @Test func directDestinationCancellationAndReentrantReadUseSameTokenSafely() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let descriptorsWithToken = descriptorCount()
        var canceledBytes = [UInt8](repeating: 0xee, count: 16)
        do {
            try canceledBytes.withUnsafeMutableBytes { destination in
                try fixture.handle.preadTensorRange(
                    token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                    into: destination, readAt: darwinPread,
                    checkpoint: { point in
                        if case .beforePayloadRead = point { throw CancellationError() }
                    }, limits: nil)
            }
            Issue.record("Expected cancellation before direct destination bytes were read")
        } catch is CancellationError {
            // The destination is caller-owned and invalid after failure.
        } catch {
            Issue.record("Expected CancellationError, got \(error)")
        }
        #expect(canceledBytes == Array(repeating: 0xee, count: 16))
        #expect(descriptorCount() == descriptorsWithToken)

        var outerBytes = [UInt8](repeating: 0xee, count: 16)
        var nestedBytes: [UInt8]?
        var didReenter = false
        try outerBytes.withUnsafeMutableBytes { outer in
            try fixture.handle.preadTensorRange(
                token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                into: outer,
                readAt: { fd, buffer, count, offset in
                    if !didReenter {
                        didReenter = true
                        var nested = [UInt8](repeating: 0xee, count: 5)
                        do {
                            try nested.withUnsafeMutableBytes { raw in
                                try fixture.handle.preadTensorRange(
                                    token, byteOffset: 1, byteCount: 5,
                                    expectedByteCount: 5, into: raw)
                            }
                            nestedBytes = nested
                        } catch {
                            nestedBytes = nil
                        }
                    }
                    return darwinPread(fd, buffer, count, offset)
                }, checkpoint: { _ in }, limits: nil)
        }
        #expect(didReenter)
        #expect(outerBytes == TinyTensorRangeFixture.payload)
        #expect(nestedBytes == [0x11, 0x22, 0x33, 0x44, 0x55])
        #expect(descriptorCount() == descriptorsWithToken)
    }

    @Test func successfulShortReadsUseActualDarwinPread() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        var actualBytesRead = 0
        let data = try fixture.handle.preadTensorRange(
            token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
            allocationBudget: 16,
            readAt: { fd, buffer, count, offset in
                let result = darwinPread(fd, buffer, count, offset)
                if case .bytes(let count) = result { actualBytesRead += count }
                return result
            }, checkpoint: { _ in },
            limits: .init(maximumOffset: UInt64.max,
                          maximumAllocation: 16, maximumSyscall: 3))

        #expect(data == Data(TinyTensorRangeFixture.payload))
        #expect(actualBytesRead == 16)
    }

    @Test func retriesInjectedEintrThenReadsRealFileBytes() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        var injectInterruption = true
        let data = try fixture.handle.preadTensorRange(
            token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
            allocationBudget: 16,
            readAt: { fd, buffer, count, offset in
                if injectInterruption {
                    injectInterruption = false
                    return .interrupted
                }
                return darwinPread(fd, buffer, count, offset)
            }, checkpoint: { _ in }, limits: nil)

        #expect(data == Data(TinyTensorRangeFixture.payload))
        #expect(!injectInterruption)
    }

    @Test func injectedEOFAfterAnActualPrefixReturnsNoPartialDataAndLeaksNoDescriptor() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let descriptorsWithToken = descriptorCount()
        var returnEOFAfterPrefix = false

        expectShortRead {
            try fixture.handle.preadTensorRange(
                token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                allocationBudget: 16,
                readAt: { fd, buffer, count, offset in
                    if returnEOFAfterPrefix { return .bytes(0) }
                    let result = darwinPread(fd, buffer, min(count, 4), offset)
                    if case .bytes(let got) = result, got > 0 { returnEOFAfterPrefix = true }
                    return result
                }, checkpoint: { _ in },
                limits: .init(maximumOffset: UInt64.max,
                              maximumAllocation: 16, maximumSyscall: 4))
        }
        #expect(returnEOFAfterPrefix)
        #expect(descriptorCount() == descriptorsWithToken)
    }

    @Test func injectedIoErrorAfterAnActualPrefixReturnsNoPartialData() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let descriptorsWithToken = descriptorCount()
        var failAfterPrefix = false

        expectIOError(errno: EIO) {
            try fixture.handle.preadTensorRange(
                token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                allocationBudget: 16,
                readAt: { fd, buffer, count, offset in
                    if failAfterPrefix { return .failure(EIO) }
                    let result = darwinPread(fd, buffer, min(count, 4), offset)
                    if case .bytes(let got) = result, got > 0 { failAfterPrefix = true }
                    return result
                }, checkpoint: { _ in },
                limits: .init(maximumOffset: UInt64.max,
                              maximumAllocation: 16, maximumSyscall: 4))
        }
        #expect(failAfterPrefix)
        #expect(descriptorCount() == descriptorsWithToken)
    }

    @Test func rejectsImpossibleInjectedPreadCountsWithoutReturningData() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let descriptorsWithToken = descriptorCount()

        expectShortRead {
            try fixture.handle.preadTensorRange(
                token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                allocationBudget: 16,
                readAt: { _, _, count, _ in .bytes(count + 1) },
                checkpoint: { _ in }, limits: nil)
        }
        expectShortRead {
            try fixture.handle.preadTensorRange(
                token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                allocationBudget: 16,
                readAt: { _, _, _, _ in .bytes(-1) },
                checkpoint: { _ in }, limits: nil)
        }
        #expect(descriptorCount() == descriptorsWithToken)
    }

    @Test func rejectsForeignHandleToken() throws {
        let issuer = try TinyTensorRangeFixture.make()
        defer { issuer.remove() }
        let other = try TinyTensorRangeFixture.make()
        defer { other.remove() }
        let token = try issuer.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let descriptorsWithBothHandlesAndToken = descriptorCount()

        expectInvalidTensor {
            try other.handle.preadTensorRange(
                token, byteOffset: 0, byteCount: 1,
                expectedByteCount: 1, allocationBudget: 1)
        }
        #expect(descriptorCount() == descriptorsWithBothHandlesAndToken)
    }

    @Test func rejectsLeafReplacementBeforeRangeReadAndKeepsRetainedDescriptorStable() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let descriptorsWithToken = descriptorCount()
        try replaceFile(at: fixture.shardURL,
                        with: TinyTensorRangeFixture.safetensorsFile(
                            header: TinyTensorRangeFixture.validHeader,
                            payload: TinyTensorRangeFixture.payload))

        expectReplaced {
            _ = try fixture.handle.preadTensorRange(
                token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                allocationBudget: 16)
        }
        #expect(descriptorCount() == descriptorsWithToken)
    }

    @Test func realDarwinPreadObservesTinyFileTruncationDuringRangeRead() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let descriptorsWithToken = descriptorCount()
        var didTruncate = false
        var sawRealEOF = false

        expectMutationOrShortRead {
            try fixture.handle.preadTensorRange(
                token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                allocationBudget: 16,
                readAt: { fd, buffer, count, offset in
                    if !didTruncate {
                        didTruncate = true
                        let writer = open(fixture.shardURL.path, O_WRONLY | O_CLOEXEC)
                        guard writer >= 0 else { return .failure(errno) }
                        let truncated = ftruncate(writer, 0) == 0
                        let truncateError = errno
                        close(writer)
                        guard truncated else { return .failure(truncateError) }
                    }
                    let result = darwinPread(fd, buffer, count, offset)
                    if case .bytes(0) = result { sawRealEOF = true }
                    return result
                }, checkpoint: { _ in }, limits: nil)
        }

        #expect(didTruncate)
        #expect(sawRealEOF)
        #expect(descriptorCount() == descriptorsWithToken)
    }

    @Test func replacementDuringAdmissionClosesOpenedShardDescriptor() throws {
        var shardURL: URL?
        let fixture = try TinyTensorRangeFixture.make(checkpoint: { point in
            guard case .fileOpened = point, let shardURL else { return }
            try replaceFile(at: shardURL,
                            with: TinyTensorRangeFixture.safetensorsFile(
                                header: TinyTensorRangeFixture.validHeader,
                                payload: TinyTensorRangeFixture.payload))
        })
        defer { fixture.remove() }
        shardURL = fixture.shardURL
        let descriptorsBeforeAdmission = descriptorCount()

        expectReplaced {
            _ = try fixture.handle.admitTensor(
                shardName: TinyTensorRangeFixture.shardName,
                tensorName: TinyTensorRangeFixture.tensorName)
        }
        #expect(descriptorCount() == descriptorsBeforeAdmission)
    }

    @Test func rejectsSameInodeSameSizeContentMutationBeforeRead() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let descriptorsWithToken = descriptorCount()
        let evidence = try overwriteFirstPayloadByteInPlace(fixture, with: 0xa5)

        expectReplaced {
            _ = try fixture.handle.preadTensorRange(
                token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                allocationBudget: 16)
        }
        #expect(evidence.sameInode)
        #expect(evidence.sameSize)
        #expect(evidence.modificationTimeChanged)
        #expect(descriptorCount() == descriptorsWithToken)
    }

    @Test func rejectsSameInodeSameSizeMutationAfterFirstPayloadRead() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let descriptorsWithToken = descriptorCount()
        var evidence: SameInodeMutationEvidence?

        expectReplaced {
            _ = try fixture.handle.preadTensorRange(
                token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                allocationBudget: 16, readAt: darwinPread,
                checkpoint: { point in
                    if case .afterPayloadRead = point {
                        evidence = try overwriteFirstPayloadByteInPlace(fixture, with: 0x5a)
                    }
                }, limits: nil)
        }
        #expect(evidence?.sameInode == true)
        #expect(evidence?.sameSize == true)
        #expect(evidence?.modificationTimeChanged == true)
        #expect(descriptorCount() == descriptorsWithToken)
    }

    @Test func emptyReadStillRejectsChangedLeafAndSourceRoot() throws {
        let leafFixture = try TinyTensorRangeFixture.make()
        defer { leafFixture.remove() }
        let leafToken = try leafFixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let leafDescriptors = descriptorCount()
        let leafMutation = try overwriteFirstPayloadByteInPlace(leafFixture, with: 0x31)
        #expect(leafMutation.sameInode)
        #expect(leafMutation.sameSize)
        expectReplaced {
            _ = try leafFixture.handle.preadTensorRange(
                leafToken, byteOffset: 0, byteCount: 0,
                expectedByteCount: 0, allocationBudget: 0)
        }
        #expect(descriptorCount() == leafDescriptors)

        let rootFixture = try TinyTensorRangeFixture.make()
        defer { rootFixture.remove() }
        let rootToken = try rootFixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let rootDescriptors = descriptorCount()
        try replaceSourceRoot(rootFixture)
        expectReplaced {
            _ = try rootFixture.handle.preadTensorRange(
                rootToken, byteOffset: 0, byteCount: 0,
                expectedByteCount: 0, allocationBudget: 0)
        }
        #expect(descriptorCount() == rootDescriptors)
    }

    @Test func tokenClosesItsShardDescriptorWhileOwnerRemainsAlive() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let owner = fixture.handle
        let descriptorsWithOwner = descriptorCount()
        var token: OfficialSourceHandle.TensorRange? = try owner.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        #expect(descriptorCount() == descriptorsWithOwner + 1)
        #expect(token != nil)

        token = nil
        try owner.validateBinding()
        #expect(descriptorCount() == descriptorsWithOwner)
        withExtendedLifetime(owner) {}
    }

    @Test func catchesLeafAndRootReplacementsAfterPayloadRead() throws {
        let leafFixture = try TinyTensorRangeFixture.make()
        defer { leafFixture.remove() }
        let leafToken = try leafFixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let leafDescriptors = descriptorCount()
        expectReplaced {
            _ = try leafFixture.handle.preadTensorRange(
                leafToken, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                allocationBudget: 16, readAt: darwinPread,
                checkpoint: { point in
                    if case .afterPayloadRead = point {
                        try replaceFile(at: leafFixture.shardURL,
                                        with: TinyTensorRangeFixture.safetensorsFile(
                                            header: TinyTensorRangeFixture.validHeader,
                                            payload: TinyTensorRangeFixture.payload))
                    }
                }, limits: nil)
        }
        #expect(descriptorCount() == leafDescriptors)

        let rootFixture = try TinyTensorRangeFixture.make()
        defer { rootFixture.remove() }
        let rootToken = try rootFixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let rootDescriptors = descriptorCount()
        expectReplaced {
            _ = try rootFixture.handle.preadTensorRange(
                rootToken, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                allocationBudget: 16, readAt: darwinPread,
                checkpoint: { point in
                    if case .afterPayloadRead = point { try replaceSourceRoot(rootFixture) }
                }, limits: nil)
        }
        #expect(descriptorCount() == rootDescriptors)
    }

    @Test func cancellationAtPayloadCheckpointReturnsNoDataAndKeepsDescriptorStable() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let descriptorsWithToken = descriptorCount()
        var observedCancellation = false

        do {
            _ = try fixture.handle.preadTensorRange(
                token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                allocationBudget: 16, readAt: darwinPread,
                checkpoint: { point in
                    if case .beforePayloadRead = point { throw CancellationError() }
                }, limits: nil)
            Issue.record("Expected cancellation before any payload was returned")
        } catch is CancellationError {
            observedCancellation = true
        } catch {
            Issue.record("Expected CancellationError, got \(error)")
        }

        #expect(observedCancellation)
        #expect(descriptorCount() == descriptorsWithToken)
    }

    @Test func rangeReadCanReenterOnTheSameRetainedToken() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        var nestedData: Data?
        var didReenter = false
        let outerData = try fixture.handle.preadTensorRange(
            token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
            allocationBudget: 16,
            readAt: { fd, buffer, count, offset in
                if !didReenter {
                    didReenter = true
                    nestedData = try? fixture.handle.preadTensorRange(
                        token, byteOffset: 1, byteCount: 5,
                        expectedByteCount: 5, allocationBudget: 5)
                }
                return darwinPread(fd, buffer, count, offset)
            }, checkpoint: { _ in }, limits: nil)

        #expect(outerData == Data(TinyTensorRangeFixture.payload))
        #expect(nestedData == Data([0x11, 0x22, 0x33, 0x44, 0x55]))
        #expect(didReenter)
    }

    @Test func markerReplacementAtFinalCheckpointReturnsNoDataAndLeaksNoDescriptor() throws {
        let fixture = try TinyTensorRangeFixture.make()
        defer { fixture.remove() }
        let token = try fixture.handle.admitTensor(
            shardName: TinyTensorRangeFixture.shardName,
            tensorName: TinyTensorRangeFixture.tensorName)
        let descriptorsWithToken = descriptorCount()

        expectReplaced {
            _ = try fixture.handle.preadTensorRange(
                token, byteOffset: 0, byteCount: 16, expectedByteCount: 16,
                allocationBudget: 16, readAt: darwinPread,
                checkpoint: { point in
                    if case .beforeReturn = point { try replaceRegistrationMarker(fixture) }
                }, limits: nil)
        }
        #expect(descriptorCount() == descriptorsWithToken)
    }
}

private struct TinyTensorRangeFixture {
    static let shardName = "model-00001-of-00026.safetensors"
    static let tensorName = "synthetic.weight"
    static let payload: [UInt8] = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
    ]
    static let validHeader = #"{"synthetic.weight":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}"#
    static var validPayloadAbsoluteOffset: Int {
        let headerCount = validHeader.utf8.count
        return 8 + ((headerCount + 7) / 8) * 8
    }

    let registration: Task68SyntheticRegistrationFixture
    let shardURL: URL
    let handle: OfficialSourceHandle

    static func make(
        fileBytes: Data? = nil,
        checkpoint: @escaping (OfficialSourceHandle.Checkpoint) throws -> Void = { _ in }
    ) throws -> Self {
        let registration = try Task68SyntheticRegistrationFixture.make()
        let shardURL = registration.sourceRoot.appendingPathComponent(shardName)
        do {
            try (fileBytes ?? safetensorsFile(header: validHeader, payload: payload)).write(to: shardURL)
            let handle = try OfficialSourceHandle(
                registrationURL: registration.modelDirectory,
                checkpoint: checkpoint)
            return Self(registration: registration, shardURL: shardURL, handle: handle)
        } catch {
            registration.remove()
            throw error
        }
    }

    func remove() {
        registration.remove()
    }

    static func safetensorsFile(header: String, payload: [UInt8]) -> Data {
        var headerBytes = Data(header.utf8)
        let paddingCount = (8 - headerBytes.count % 8) % 8
        headerBytes.append(contentsOf: Array(repeating: UInt8(0x20), count: paddingCount))
        var littleEndianHeaderLength = UInt64(headerBytes.count).littleEndian
        let lengthPrefix = withUnsafeBytes(of: &littleEndianHeaderLength) { Data($0) }
        var result = lengthPrefix
        result.append(contentsOf: headerBytes)
        result.append(contentsOf: payload)
        return result
    }

    static func safetensorsFile(declaredHeaderLength: UInt64, headerBytes: Data, payload: [UInt8]) -> Data {
        var littleEndianHeaderLength = declaredHeaderLength.littleEndian
        let lengthPrefix = withUnsafeBytes(of: &littleEndianHeaderLength) { Data($0) }
        var result = lengthPrefix
        result.append(contentsOf: headerBytes)
        result.append(contentsOf: payload)
        return result
    }
}

private enum FourMiBBoundaryFixture {
    static let tileBytes = 4 * 1024 * 1024

    static func payload() -> [UInt8] {
        var bytes = [UInt8](repeating: 0xa5, count: tileBytes + 32)
        bytes.replaceSubrange(0..<16,
                              with: [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                                     0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff])
        bytes.replaceSubrange((tileBytes - 8)..<(tileBytes + 24),
                              with: Array(UInt8(0x40)...UInt8(0x5f)))
        return bytes
    }
}

private func makeFourMiBBoundaryFixture() throws
    -> (fixture: TinyTensorRangeFixture, payload: [UInt8], fileBytes: Data) {
    let payload = FourMiBBoundaryFixture.payload()
    let header = #"{"synthetic.weight":{"dtype":"BF16","shape":[\#(payload.count / 2)],"data_offsets":[0,\#(payload.count)]}}"#
    let fileBytes = TinyTensorRangeFixture.safetensorsFile(header: header, payload: payload)
    let fixture = try TinyTensorRangeFixture.make(fileBytes: fileBytes)
    return (fixture: fixture, payload: payload, fileBytes: fileBytes)
}

private func darwinPread(
    _ fd: Int32,
    _ buffer: UnsafeMutableRawPointer,
    _ count: Int,
    _ offset: off_t
) -> OfficialSourceHandle.TensorPreadResult {
    let result = Darwin.pread(fd, buffer, count, offset)
    if result >= 0 { return .bytes(result) }
    if errno == EINTR { return .interrupted }
    return .failure(errno)
}

private struct SameInodeMutationEvidence {
    let sameInode: Bool
    let sameSize: Bool
    let modificationTimeChanged: Bool
}

private func overwriteFirstPayloadByteInPlace(
    _ fixture: TinyTensorRangeFixture,
    with value: UInt8
) throws -> SameInodeMutationEvidence {
    let fd = open(fixture.shardURL.path, O_WRONLY | O_CLOEXEC)
    guard fd >= 0 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
    defer { close(fd) }

    var before = stat()
    guard fstat(fd, &before) == 0 else {
        throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
    }
    var byte = value
    let payloadOffset = off_t(TinyTensorRangeFixture.validPayloadAbsoluteOffset)
    let written = withUnsafeBytes(of: &byte) { raw -> Int in
        guard let base = raw.baseAddress else { return -1 }
        return Darwin.pwrite(fd, base, raw.count, payloadOffset)
    }
    guard written == 1 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }

    // Force a distinct deterministic mtime without sleeping; pwrite also changes ctime.
    let timestamps = [timeval(tv_sec: 1, tv_usec: 0), timeval(tv_sec: 1, tv_usec: 0)]
    let timestampResult = timestamps.withUnsafeBufferPointer { futimes(fd, $0.baseAddress) }
    guard timestampResult == 0 else {
        throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
    }
    var after = stat()
    guard fstat(fd, &after) == 0 else {
        throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
    }
    return SameInodeMutationEvidence(
        sameInode: before.st_dev == after.st_dev && before.st_ino == after.st_ino,
        sameSize: before.st_size == after.st_size,
        modificationTimeChanged: before.st_mtimespec.tv_sec != after.st_mtimespec.tv_sec
            || before.st_mtimespec.tv_nsec != after.st_mtimespec.tv_nsec)
}

private func overwriteByteInPlace(at url: URL, offset: off_t, with value: UInt8) throws {
    let fd = open(url.path, O_WRONLY | O_CLOEXEC)
    guard fd >= 0 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
    defer { close(fd) }
    var byte = value
    let written = withUnsafeBytes(of: &byte) { raw -> Int in
        guard let base = raw.baseAddress else { return -1 }
        return Darwin.pwrite(fd, base, raw.count, offset)
    }
    guard written == 1 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
    guard fsync(fd) == 0 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
    let timestamps = [timeval(tv_sec: 1, tv_usec: 0), timeval(tv_sec: 1, tv_usec: 0)]
    guard timestamps.withUnsafeBufferPointer({ futimes(fd, $0.baseAddress) }) == 0 else {
        throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
    }
}

private func descriptorCount() -> Int {
    (0..<Int(getdtablesize())).reduce(into: 0) { count, index in
        if fcntl(Int32(index), F_GETFD) != -1 { count += 1 }
    }
}

private func expectSourceHandleError<T>(
    expected: String,
    sourceLocation: SourceLocation = #_sourceLocation,
    matches: (OfficialSourceHandleError) -> Bool,
    operation: () throws -> T
) {
    do {
        _ = try operation()
        Issue.record("Expected \(expected)", sourceLocation: sourceLocation)
    } catch let error as OfficialSourceHandleError {
        if !matches(error) {
            Issue.record("Expected \(expected), got \(error)", sourceLocation: sourceLocation)
        }
    } catch {
        Issue.record("Expected \(expected), got \(error)", sourceLocation: sourceLocation)
    }
}

private func expectRangeError<T>(
    reasonContains: String? = nil,
    sourceLocation: SourceLocation = #_sourceLocation,
    operation: () throws -> T
) {
    expectSourceHandleError(
        expected: "OfficialSourceHandleError.range",
        sourceLocation: sourceLocation,
        matches: { error in
            guard case .range(let reason) = error else { return false }
            return reasonContains.map { reason.contains($0) } ?? true
        },
        operation: operation)
}

private func expectShortRead<T>(
    sourceLocation: SourceLocation = #_sourceLocation,
    operation: () throws -> T
) {
    expectSourceHandleError(
        expected: "OfficialSourceHandleError.shortRead",
        sourceLocation: sourceLocation,
        matches: { if case .shortRead = $0 { return true }; return false },
        operation: operation)
}

private func expectIOError<T>(
    errno expectedErrno: Int32,
    sourceLocation: SourceLocation = #_sourceLocation,
    operation: () throws -> T
) {
    expectSourceHandleError(
        expected: "OfficialSourceHandleError.io(errno: \(expectedErrno))",
        sourceLocation: sourceLocation,
        matches: { error in
            guard case .io(_, let actualErrno) = error else { return false }
            return actualErrno == expectedErrno
        },
        operation: operation)
}

private func expectInvalidTensor<T>(
    sourceLocation: SourceLocation = #_sourceLocation,
    operation: () throws -> T
) {
    expectSourceHandleError(
        expected: "OfficialSourceHandleError.invalidTensor",
        sourceLocation: sourceLocation,
        matches: { if case .invalidTensor = $0 { return true }; return false },
        operation: operation)
}

private func expectNotAllowed<T>(
    sourceLocation: SourceLocation = #_sourceLocation,
    operation: () throws -> T
) {
    expectSourceHandleError(
        expected: "OfficialSourceHandleError.notAllowed",
        sourceLocation: sourceLocation,
        matches: { if case .notAllowed = $0 { return true }; return false },
        operation: operation)
}

private func expectMutationOrShortRead<T>(
    sourceLocation: SourceLocation = #_sourceLocation,
    operation: () throws -> T
) {
    expectSourceHandleError(
        expected: "OfficialSourceHandleError.replaced or shortRead",
        sourceLocation: sourceLocation,
        matches: { error in
            switch error {
            case .replaced, .shortRead: return true
            default: return false
            }
        },
        operation: operation)
}

private func expectReplaced(_ operation: () throws -> Void, sourceLocation: SourceLocation = #_sourceLocation) {
    do {
        try operation()
        Issue.record("Expected a source or registration replacement to be rejected",
                     sourceLocation: sourceLocation)
    } catch let error as OfficialSourceHandleError {
        if case .replaced = error { return }
        Issue.record("Expected OfficialSourceHandleError.replaced, got \(error)",
                     sourceLocation: sourceLocation)
    } catch {
        Issue.record("Expected OfficialSourceHandleError.replaced, got \(error)",
                     sourceLocation: sourceLocation)
    }
}

private func replaceFile(at url: URL, with bytes: Data) throws {
    let displaced = url.deletingLastPathComponent()
        .appendingPathComponent("displaced-\(UUID().uuidString)")
    try FileManager.default.moveItem(at: url, to: displaced)
    try bytes.write(to: url)
}

private func replaceSourceRoot(_ fixture: TinyTensorRangeFixture) throws {
    let displaced = fixture.registration.root.appendingPathComponent(
        "displaced-source-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.moveItem(at: fixture.registration.sourceRoot, to: displaced)
    try FileManager.default.createDirectory(
        at: fixture.registration.sourceRoot, withIntermediateDirectories: false)
}

private func replaceRegistrationMarker(_ fixture: TinyTensorRangeFixture) throws {
    let marker = fixture.registration.modelDirectory.appendingPathComponent(
        OfficialSourceDescriptor.markerFilename)
    try Data("replaced synthetic marker".utf8).write(to: marker, options: .atomic)
}
