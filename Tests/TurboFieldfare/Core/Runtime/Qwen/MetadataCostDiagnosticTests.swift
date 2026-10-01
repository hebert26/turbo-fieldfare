import CryptoKit
import Darwin
import Foundation
import Metal
import Testing
@testable import TurboFieldfareOfficialQwenSource

/// Opt-in cost evidence for the existing accepted-source and tensor-range
/// checks. This deliberately reads only four layer-0 expert slices and repeats
/// them once, so the result is useful for choosing a read-path optimization
/// without becoming another model qualification suite.
@Suite(.serialized)
struct MetadataCostDiagnosticTests {
    private static let enableKey = "TURBO_P23_METADATA_COST_DIAGNOSTIC"
    private static let registrationKey = "TURBO_P23_METADATA_COST_REGISTRATION"
    private static let experts = [0, 1, 127, 255]
    private static let repetitions = 2
    private static let gateTensor =
        "model.language_model.layers.0.mlp.experts.gate_up_proj"
    private static let downTensor =
        "model.language_model.layers.0.mlp.experts.down_proj"
    private static let gateShard = "model-00001-of-00026.safetensors"
    private static let downShard = "model-00002-of-00026.safetensors"
    private static let gateShape: [UInt64] = [256, 1024, 2048]
    private static let downShape: [UInt64] = [256, 2048, 512]

    @Test(.enabled(if: ProcessInfo.processInfo.environment[enableKey] == "1",
                   "set TURBO_P23_METADATA_COST_DIAGNOSTIC=1 to run the bounded diagnostic"))
    func measuresAcceptedExpertPairReadsAndSeparatesGuardOverhead() throws {
        let environment = ProcessInfo.processInfo.environment
        let registrationPath = try #require(environment[Self.registrationKey])
        let registration = URL(fileURLWithPath: registrationPath,
                               isDirectory: true).standardizedFileURL

        let receiptStart = monotonicNanos()
        let receipt = try OfficialSourceTrust.verify(
            at: registration, policy: .sizeCheckTrustedReceipt)
        let receiptVerificationNanos = elapsed(since: receiptStart)

        let handleStart = monotonicNanos()
        let source = try OfficialSourceHandle(registrationURL: registration)
        let handleConstructionNanos = elapsed(since: handleStart)

        let bindingStart = monotonicNanos()
        try source.validateTrustedReceipt(receipt)
        let trustedReceiptValidationNanos = elapsed(since: bindingStart)

        let admissionStart = monotonicNanos()
        let gate = try source.admitTensor(shardName: Self.gateShard,
                                          tensorName: Self.gateTensor)
        let down = try source.admitTensor(shardName: Self.downShard,
                                          tensorName: Self.downTensor)
        let tensorAdmissionNanos = elapsed(since: admissionStart)

        guard gate.shape == Self.gateShape,
              down.shape == Self.downShape else {
            throw MetadataCostDiagnosticError.unexpectedTensorShape(
                gate: gate.shape, down: down.shape)
        }

        let device = try #require(MTLCreateSystemDefaultDevice())
        let gateSliceBytes = UInt64(Self.gateShape[1])
            * UInt64(Self.gateShape[2]) * 2
        let downSliceBytes = UInt64(Self.downShape[1])
            * UInt64(Self.downShape[2]) * 2
        let gateBuffer = try #require(device.makeBuffer(
            length: Int(gateSliceBytes), options: .storageModeShared))
        let downBuffer = try #require(device.makeBuffer(
            length: Int(downSliceBytes), options: .storageModeShared))
        #expect(UInt64(gateBuffer.length) == gateSliceBytes)
        #expect(UInt64(downBuffer.length) == downSliceBytes)

        let zeroByteValidationNanos = try measureZeroByteValidation(
            source: source, gate: gate, down: down)

        var firstDigests: [String: String] = [:]
        var measurements: [[String: Any]] = []
        var totalReadWallNanos: UInt64 = 0
        var totalSummedPreadNanos: UInt64 = 0
        var totalNonPayloadOverheadNanos: UInt64 = 0
        var totalSyscalls = 0
        var totalCheckpoints = 0
        var payloadDigestsStable = true

        for repetition in 0..<Self.repetitions {
            for expert in Self.experts {
                let gateMeasurement = try readSlice(
                    source: source, token: gate, expert: expert,
                    sliceBytes: gateSliceBytes, buffer: gateBuffer)
                let gateKey = "gate_up:\(expert)"
                if let previous = firstDigests[gateKey] {
                    guard previous == gateMeasurement.digest else {
                        payloadDigestsStable = false
                        throw MetadataCostDiagnosticError.payloadDigestChanged(gateKey)
                    }
                } else {
                    firstDigests[gateKey] = gateMeasurement.digest
                }
                measurements.append(gateMeasurement.json(
                    tensor: Self.gateTensor, expert: expert, repetition: repetition))

                let downMeasurement = try readSlice(
                    source: source, token: down, expert: expert,
                    sliceBytes: downSliceBytes, buffer: downBuffer)
                let downKey = "down:\(expert)"
                if let previous = firstDigests[downKey] {
                    guard previous == downMeasurement.digest else {
                        payloadDigestsStable = false
                        throw MetadataCostDiagnosticError.payloadDigestChanged(downKey)
                    }
                } else {
                    firstDigests[downKey] = downMeasurement.digest
                }
                measurements.append(downMeasurement.json(
                    tensor: Self.downTensor, expert: expert, repetition: repetition))

                totalReadWallNanos += gateMeasurement.wallNanos
                    + downMeasurement.wallNanos
                totalSummedPreadNanos += gateMeasurement.summedPreadNanos
                    + downMeasurement.summedPreadNanos
                totalNonPayloadOverheadNanos += gateMeasurement.nonPayloadOverheadNanos
                    + downMeasurement.nonPayloadOverheadNanos
                totalSyscalls += gateMeasurement.syscallCount + downMeasurement.syscallCount
                totalCheckpoints += gateMeasurement.checkpointCount
                    + downMeasurement.checkpointCount
            }
        }

        #expect(measurements.count == Self.experts.count * Self.repetitions * 2)
        #expect(totalReadWallNanos >= totalSummedPreadNanos)
        #expect(totalNonPayloadOverheadNanos
                == totalReadWallNanos - totalSummedPreadNanos)
        #expect(totalSyscalls > 0)
        #expect(totalCheckpoints > 0)

        let report: [String: Any] = [
            "diagnostic": "metadata-cost-diagnostic",
            "registrationPath": registration.path,
            "receiptVerificationNanos": receiptVerificationNanos,
            "handleConstructionNanos": handleConstructionNanos,
            "trustedReceiptValidationNanos": trustedReceiptValidationNanos,
            "tensorAdmissionNanos": tensorAdmissionNanos,
            "zeroByteValidationNanos": zeroByteValidationNanos,
            "gateTensor": Self.gateTensor,
            "downTensor": Self.downTensor,
            "gateSliceBytes": gateSliceBytes,
            "downSliceBytes": downSliceBytes,
            "expertIDs": Self.experts,
            "repetitions": Self.repetitions,
            "dispatchMode": "serial",
            "workerCount": 1,
            "totalReadWallNanos": totalReadWallNanos,
            "totalSummedPreadNanos": totalSummedPreadNanos,
            "totalNonPayloadOverheadNanos": totalNonPayloadOverheadNanos,
            "totalSyscalls": totalSyscalls,
            "totalCheckpoints": totalCheckpoints,
            "payloadDigestsStableAcrossRepetition": payloadDigestsStable,
            "reads": measurements,
        ]
        guard JSONSerialization.isValidJSONObject(report) else {
            throw MetadataCostDiagnosticError.invalidReport
        }
        let encoded = try JSONSerialization.data(withJSONObject: report,
                                                  options: [.sortedKeys])
        print(String(decoding: encoded, as: UTF8.self))
    }

    private func measureZeroByteValidation(
        source: OfficialSourceHandle,
        gate: OfficialSourceHandle.TensorRange,
        down: OfficialSourceHandle.TensorRange
    ) throws -> UInt64 {
        let start = monotonicNanos()
        try validateZeroByteRange(source: source, token: gate)
        try validateZeroByteRange(source: source, token: down)
        return elapsed(since: start)
    }

    private func validateZeroByteRange(
        source: OfficialSourceHandle,
        token: OfficialSourceHandle.TensorRange
    ) throws {
        var checkpointCount = 0
        let destination = UnsafeMutableRawBufferPointer(start: nil, count: 0)
        try source.preadTensorRange(
            token, byteOffset: token.admittedByteCount, byteCount: 0,
            expectedByteCount: 0, into: destination,
            readAt: darwinPread,
            checkpoint: { _ in checkpointCount += 1 }, limits: nil)
        guard checkpointCount >= 1 else {
            throw MetadataCostDiagnosticError.missingProtectedCheckpoint
        }
    }

    private func readSlice(
        source: OfficialSourceHandle,
        token: OfficialSourceHandle.TensorRange,
        expert: Int,
        sliceBytes: UInt64,
        buffer: MTLBuffer
    ) throws -> SliceMeasurement {
        let byteOffset = UInt64(expert) * sliceBytes
        var checkpointCount = 0
        var syscallCount = 0
        var summedPreadNanos: UInt64 = 0
        let wallStart = monotonicNanos()
        let destination = UnsafeMutableRawBufferPointer(
            start: buffer.contents(), count: Int(sliceBytes))
        try source.preadTensorRange(
            token, byteOffset: byteOffset, byteCount: sliceBytes,
            expectedByteCount: sliceBytes, into: destination,
            readAt: { fd, pointer, count, offset in
                let syscallStart = monotonicNanos()
                let result = Darwin.pread(fd, pointer, count, offset)
                let savedErrno = result < 0 ? errno : 0
                let syscallNanos = elapsed(since: syscallStart)
                summedPreadNanos += syscallNanos
                syscallCount += 1
                if result >= 0 { return .bytes(result) }
                if savedErrno == EINTR { return .interrupted }
                return .failure(savedErrno)
            }, checkpoint: { _ in checkpointCount += 1 }, limits: nil)
        let wallNanos = elapsed(since: wallStart)
        guard wallNanos >= summedPreadNanos else {
            throw MetadataCostDiagnosticError.inconsistentTiming(
                wall: wallNanos, summedPread: summedPreadNanos)
        }
        guard syscallCount > 0 else {
            throw MetadataCostDiagnosticError.missingPayloadSyscall
        }
        guard checkpointCount >= 3 else {
            throw MetadataCostDiagnosticError.missingProtectedCheckpoint
        }
        return SliceMeasurement(
            digest: digest(buffer, count: Int(sliceBytes)),
            wallNanos: wallNanos,
            summedPreadNanos: summedPreadNanos,
            nonPayloadOverheadNanos: wallNanos - summedPreadNanos,
            syscallCount: syscallCount,
            checkpointCount: checkpointCount)
    }
}

private struct SliceMeasurement {
    let digest: String
    let wallNanos: UInt64
    let summedPreadNanos: UInt64
    let nonPayloadOverheadNanos: UInt64
    let syscallCount: Int
    let checkpointCount: Int

    func json(tensor: String, expert: Int, repetition: Int) -> [String: Any] {
        [
            "tensor": tensor,
            "expert": expert,
            "repetition": repetition,
            "wallNanos": wallNanos,
            "summedPreadNanos": summedPreadNanos,
            "nonPayloadOverheadNanos": nonPayloadOverheadNanos,
            "syscallCount": syscallCount,
            "checkpointCount": checkpointCount,
            "payloadSHA256": digest,
        ]
    }
}

private enum MetadataCostDiagnosticError: Error {
    case unexpectedTensorShape(gate: [UInt64], down: [UInt64])
    case inconsistentTiming(wall: UInt64, summedPread: UInt64)
    case missingPayloadSyscall
    case missingProtectedCheckpoint
    case payloadDigestChanged(String)
    case invalidReport
}

private func monotonicNanos() -> UInt64 {
    DispatchTime.now().uptimeNanoseconds
}

private func elapsed(since start: UInt64) -> UInt64 {
    monotonicNanos() - start
}

private func digest(_ buffer: MTLBuffer, count: Int) -> String {
    let data = Data(bytesNoCopy: buffer.contents(), count: count, deallocator: .none)
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func darwinPread(
    _ fd: Int32,
    _ buffer: UnsafeMutableRawPointer,
    _ count: Int,
    _ offset: off_t
) -> OfficialSourceHandle.TensorPreadResult {
    let result = Darwin.pread(fd, buffer, count, offset)
    let savedErrno = result < 0 ? errno : 0
    if result >= 0 { return .bytes(result) }
    if savedErrno == EINTR { return .interrupted }
    return .failure(savedErrno)
}
