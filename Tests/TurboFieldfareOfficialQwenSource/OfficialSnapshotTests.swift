import Foundation
import Testing
@testable import TurboFieldfareOfficialQwenSource

@Suite struct OfficialSnapshotTests {
    private struct ReadRequest: Equatable {
        let offset: UInt64
        let count: Int
    }

    private static func fileSize(for header: Data, payloadBytes: UInt64 = 32) -> UInt64 {
        UInt64(8 + header.count) + payloadBytes
    }

    private static func parse(_ json: String, payloadBytes: UInt64 = 32) throws
        -> OfficialSafetensorsSource.Header {
        let bytes = Data(json.utf8)
        return try OfficialSafetensorsSource.parseHeader(
            path: "synthetic.safetensors",
            fileSize: fileSize(for: bytes, payloadBytes: payloadBytes),
            headerBytes: bytes)
    }

    private static func captureError<T>(_ operation: () throws -> T) -> Error? {
        do {
            _ = try operation()
            return nil
        } catch {
            return error
        }
    }

    private static func expectInvalidHeader(
        _ json: String,
        fileSize: UInt64? = nil
    ) {
        let bytes = Data(json.utf8)
        let actualFileSize = fileSize ?? Self.fileSize(for: bytes)
        let captured = captureError {
            try OfficialSafetensorsSource.parseHeader(
                path: "synthetic.safetensors",
                fileSize: actualFileSize,
                headerBytes: bytes)
        }
        guard let error = captured as? OfficialSafetensorsSource.ValidationError else {
            Issue.record("expected invalidHeader, parsing succeeded or threw an unexpected error: \(String(describing: captured))")
            return
        }
        guard case .invalidHeader = error else {
            Issue.record("expected invalidHeader, got \(error)")
            return
        }
    }

    private static func prefixedFile(header: Data, payload: Data) -> Data {
        var bytes = Data()
        var length = UInt64(header.count).littleEndian
        withUnsafeBytes(of: &length) { bytes.append(contentsOf: $0) }
        bytes.append(header)
        bytes.append(payload)
        return bytes
    }

    @Test func validHeaderReturnsExactMetadataForGenericDtypes() throws {
        let rawHeader = #"{"bf16":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"fp16":{"dtype":"F16","shape":[2],"data_offsets":[8,12]},"fp32":{"dtype":"F32","shape":[1],"data_offsets":[12,16]},"u32":{"dtype":"U32","shape":[1],"data_offsets":[16,20]}}"#
        let headerBytes = Data(rawHeader.utf8)
        let header = try Self.parse(rawHeader, payloadBytes: 20)

        #expect(header.path == "synthetic.safetensors")
        #expect(header.payloadBaseOffset == UInt64(8 + headerBytes.count))
        #expect(header.tensors.map(\.name) == ["bf16", "fp16", "fp32", "u32"])
        #expect(header.tensors.map(\.dtype) == ["BF16", "F16", "F32", "U32"])
        #expect(header.tensors.map(\.shape) == [[2, 2], [2], [1], [1]])
        #expect(header.tensors.map(\.sizeBytes) == [8, 4, 4, 4])
        #expect(header.tensors.map(\.absoluteOffset) == [
            header.payloadBaseOffset,
            header.payloadBaseOffset + 8,
            header.payloadBaseOffset + 12,
            header.payloadBaseOffset + 16,
        ])
    }

    @Test func readHeaderRequestsOnlyPrefixAndDeclaredHeader() throws {
        let header = Data(#"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#.utf8)
        let payload = Data(repeating: 0xA5, count: 4)
        let file = Self.prefixedFile(header: header, payload: payload)
        var requests: [ReadRequest] = []

        let parsed = try OfficialSafetensorsSource.readHeader(
            path: "synthetic.safetensors",
            fileSize: UInt64(file.count),
            readAt: { offset, count in
                requests.append(ReadRequest(offset: offset, count: count))
                let start = Int(offset)
                return file.subdata(in: start..<(start + count))
            })

        #expect(parsed.payloadBaseOffset == UInt64(8 + header.count))
        #expect(parsed.tensors.map(\.name) == ["w"])
        #expect(requests == [
            ReadRequest(offset: 0, count: 8),
            ReadRequest(offset: 8, count: header.count),
        ])
        #expect(requests.allSatisfy { $0.offset + UInt64($0.count) <= parsed.payloadBaseOffset })
    }

    @Test func readHeaderRejectsShortPrefixAndOversizeBeforeBodyRead() throws {
        let shortRead = Self.captureError {
            try OfficialSafetensorsSource.readHeader(
                path: "short.safetensors", fileSize: 8,
                readAt: { _, _ in Data(repeating: 0, count: 7) })
        } as? OfficialSafetensorsSource.ValidationError
        #expect(shortRead == .some(.shortRead(expected: 8, got: 7)))

        var shortBodyRequests: [ReadRequest] = []
        var twoBytePrefix = UInt64(2).littleEndian
        let twoBytePrefixData = withUnsafeBytes(of: &twoBytePrefix) { Data($0) }
        let shortBody = Self.captureError {
            try OfficialSafetensorsSource.readHeader(
                path: "short-body.safetensors", fileSize: 10,
                readAt: { offset, count in
                    shortBodyRequests.append(ReadRequest(offset: offset, count: count))
                    return offset == 0 ? twoBytePrefixData : Data([0x7B])
                })
        } as? OfficialSafetensorsSource.ValidationError
        #expect(shortBody == .some(.shortRead(expected: 2, got: 1)))
        #expect(shortBodyRequests == [
            ReadRequest(offset: 0, count: 8), ReadRequest(offset: 8, count: 2),
        ])

        var requests: [ReadRequest] = []
        var oversizedPrefix = (OfficialSafetensorsSource.maximumHeaderBytes + 1).littleEndian
        let oversizedBytes = withUnsafeBytes(of: &oversizedPrefix) { Data($0) }
        let oversized = Self.captureError {
            try OfficialSafetensorsSource.readHeader(
                path: "oversized.safetensors",
                fileSize: OfficialSafetensorsSource.maximumHeaderBytes + 9,
                readAt: { offset, count in
                    requests.append(ReadRequest(offset: offset, count: count))
                    return offset == 0 ? oversizedBytes : Data()
                })
        } as? OfficialSafetensorsSource.ValidationError
        #expect(oversized == .headerTooLarge(size: OfficialSafetensorsSource.maximumHeaderBytes + 1))
        #expect(requests == [ReadRequest(offset: 0, count: 8)])
    }

    @Test func rejectsMalformedAndDuplicateHeaderJSONKeys() {
        let ordinary = #"{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}"#
        let cases = [
            #"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"w":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}"#,
            #"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"\u0077":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}"#,
            #"{"w":{"dtype":"BF16","dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#,
            #"{"w":{"dtype":"BF16","shape":[1],"shape":[1],"data_offsets":[0,2]}}"#,
            #"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,2],"data_offsets":[0,2]}}"#,
            #"{"w":\#(ordinary),}"#,
            #"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}} trailing"#,
            #"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}"#,
        ]
        for json in cases { Self.expectInvalidHeader(json) }
    }

    @Test func rejectsBooleanNegativeFractionalExponentAndOutOfRangeIntegers() {
        let cases = [
            #"{"w":{"dtype":"BF16","shape":[true],"data_offsets":[0,2]}}"#,
            #"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[false,2]}}"#,
            #"{"w":{"dtype":"BF16","shape":[-1],"data_offsets":[0,2]}}"#,
            #"{"w":{"dtype":"BF16","shape":[1.5],"data_offsets":[0,2]}}"#,
            #"{"w":{"dtype":"BF16","shape":[1e30],"data_offsets":[0,2]}}"#,
            #"{"w":{"dtype":"BF16","shape":[18446744073709551616],"data_offsets":[0,2]}}"#,
        ]
        for json in cases { Self.expectInvalidHeader(json) }
    }

    @Test func rejectsUnknownDtypeAndByteCountDisagreement() throws {
        let unknown = Self.captureError {
            try Self.parse(#"{"w":{"dtype":"I8","shape":[2],"data_offsets":[0,2]}}"#)
        } as? OfficialSafetensorsSource.ValidationError
        #expect(unknown == .some(.unknownDtype("I8")))

        let mismatch = Self.captureError {
            try Self.parse(#"{"w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,6]}}"#)
        } as? OfficialSafetensorsSource.ValidationError
        guard case .shapeMismatch(name: "w", detail: "shape product 4*2 != size 6")? = mismatch else {
            Issue.record("unexpected shape mismatch error: \(String(describing: mismatch))")
            return
        }
    }

    @Test func rejectsReversedOutOfBoundsAndOverflowingOffsets() throws {
        Self.expectInvalidHeader(#"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[4,0]}}"#)
        let outOfBoundsHeader = Data(#"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#.utf8)
        let outOfBoundsFileSize = UInt64(8 + outOfBoundsHeader.count + 1)
        let outOfBounds = Self.captureError {
            try OfficialSafetensorsSource.parseHeader(
                path: "out-of-bounds.safetensors",
                fileSize: outOfBoundsFileSize,
                headerBytes: outOfBoundsHeader)
        } as? OfficialSafetensorsSource.ValidationError
        #expect(outOfBounds == .some(.tensorOutOfRange(
            name: "w", end: outOfBoundsFileSize + 1, fileSize: outOfBoundsFileSize)))

        let offsetOverflow = Self.captureError {
            try OfficialSafetensorsSource.parseHeader(
                path: "overflow.safetensors",
                fileSize: UInt64.max,
                headerBytes: Data(#"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[18446744073709551614,18446744073709551615]}}"#.utf8))
        } as? OfficialSafetensorsSource.ValidationError
        guard case .invalidHeader? = offsetOverflow else {
            Issue.record("expected checked absolute-offset overflow, got \(String(describing: offsetOverflow))")
            return
        }

        let absoluteEndOverflow = Self.captureError {
            try OfficialSafetensorsSource.parseHeader(
                path: "end-overflow.safetensors",
                fileSize: UInt64.max,
                headerBytes: Data(#"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[18446744073709541615,18446744073709551615]}}"#.utf8))
        } as? OfficialSafetensorsSource.ValidationError
        guard case .invalidHeader? = absoluteEndOverflow else {
            Issue.record("expected checked absolute-end overflow, got \(String(describing: absoluteEndOverflow))")
            return
        }
    }

    @Test func rejectsShapeProductOverflowAndOverlappingRanges() throws {
        Self.expectInvalidHeader(
            #"{"w":{"dtype":"BF16","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#)
        Self.expectInvalidHeader(
            #"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[2,6]}}"#)
    }

    @Test func rejectsLeadingInternalAndTrailingPayloadGaps() {
        let cases: [(String, UInt64)] = [
            (#"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}"#, 4),
            (#"{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]}}"#, 6),
            (#"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#, 3),
        ]
        for (json, payloadBytes) in cases {
            let error = Self.captureError {
                try Self.parse(json, payloadBytes: payloadBytes)
            } as? OfficialSafetensorsSource.ValidationError
            guard case .invalidHeader = error else {
                Issue.record("expected payload-gap rejection, got \(String(describing: error)) for \(json)")
                continue
            }
        }
    }

    @Test func acceptsZeroLengthTensorAtEmptyPayloadBoundary() throws {
        let header = try Self.parse(
            #"{"empty":{"dtype":"BF16","shape":[0],"data_offsets":[0,0]}}"#,
            payloadBytes: 0)
        let tensor = try #require(header.tensors.first)
        #expect(header.tensors.count == 1)
        #expect(tensor.name == "empty")
        #expect(tensor.sizeBytes == 0)
        #expect(tensor.absoluteOffset == header.payloadBaseOffset)
    }

    @Test func ordersZeroLengthBoundaryBeforeNonemptyRangeDeterministically() throws {
        // Lexical tensor order puts the nonempty range first at the same offset;
        // contiguous validation must place the empty range at the boundary first.
        let header = try Self.parse(
            #"{"a_nonempty":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"z_empty":{"dtype":"BF16","shape":[0],"data_offsets":[0,0]}}"#,
            payloadBytes: 2)
        #expect(header.tensors.map(\.name) == ["a_nonempty", "z_empty"])
        #expect(header.tensors.map(\.sizeBytes) == [2, 0])
        #expect(header.tensors.allSatisfy { $0.absoluteOffset == header.payloadBaseOffset })
    }

    @Test func parsesIndexAndRequiresExactShardMembership() throws {
        let index = try OfficialSafetensorsSource.parseIndex(Data(
            #"{"metadata":{"format":"pt"},"weight_map":{"w":"one.safetensors","b":"two.safetensors"}}"#.utf8))
        #expect(index == ["w": "one.safetensors", "b": "two.safetensors"])

        let header = try Self.parse(
            #"{"w":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#,
            payloadBytes: 4)
        try OfficialSafetensorsSource.validateShard(
            header, weightMap: index, shardName: "one.safetensors")

        let mismatch = Self.captureError {
            try OfficialSafetensorsSource.validateShard(
                header, weightMap: ["ghost": "one.safetensors"], shardName: "one.safetensors")
        } as? OfficialSafetensorsSource.ValidationError
        guard case .invalidIndex = mismatch else {
            Issue.record("expected shard membership rejection, got \(String(describing: mismatch))")
            return
        }
    }

    @Test func rejectsMalformedDuplicateAndUnsafeIndexMetadata() {
        let duplicateIndexes = [
            #"{"weight_map":{"w":"one.safetensors","w":"two.safetensors"}}"#,
            #"{"weight_map":{"w":"one.safetensors"},"weight\u005fmap":{"w":"one.safetensors"}}"#,
        ]
        for json in duplicateIndexes {
            let error = Self.captureError {
                try OfficialSafetensorsSource.parseIndex(Data(json.utf8))
            } as? OfficialSafetensorsSource.ValidationError
            guard case .invalidIndex = error else {
                Issue.record("expected duplicate-key index rejection, got \(String(describing: error))")
                continue
            }
        }

        let unsafeIndexes = [
            #"{"weight_map":{"w":"../escape.safetensors"}}"#,
            #"{"weight_map":{"w":"/tmp/escape.safetensors"}}"#,
            #"{"weight_map":{"w":"..\\escape.safetensors"}}"#,
            #"{"weight_map":{"w":"bad\u0000name.safetensors"}}"#,
            #"{"weight_map":{"w":"weights.bin"}}"#,
            #"{"weight_map":{"w":3}}"#,
            #"{"weight_map":{}}"#,
            #"{"weight_map": "#,
        ]
        for json in unsafeIndexes {
            let error = Self.captureError {
                try OfficialSafetensorsSource.parseIndex(Data(json.utf8))
            } as? OfficialSafetensorsSource.ValidationError
            guard case .invalidIndex = error else {
                Issue.record("expected malformed/unsafe index rejection, got \(String(describing: error))")
                continue
            }
        }
    }
}
