import Darwin
import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfareOfficialQwenSource

/// Production-path coverage for the staged directory scan. These cases keep
/// non-ASCII names and retained-file mutation on the real directory descriptor
/// rather than testing a private scan helper.
@Suite(.serialized)
struct OfficialSourceDirectoryScanTests {
    @Test func handleRejectsUnexpectedNonASCIIRegistrationEntry() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        let unexpected = fixture.modelDirectory.appendingPathComponent(
            "unexpected-\u{00E9}")
        try Data("untrusted".utf8).write(to: unexpected)

        expectDirectoryScanReplacement {
            _ = try OfficialSourceHandle(registrationURL: fixture.modelDirectory)
        }
    }

    @Test func retainedTensorRangeRejectsCaseOnlyShardRename() throws {
        let fixture = try DirectoryScanTensorFixture.make()
        defer { fixture.remove() }

        let token = try fixture.handle.admitTensor(
            shardName: DirectoryScanTensorFixture.shardName,
            tensorName: DirectoryScanTensorFixture.tensorName)
        let temporary = fixture.registration.sourceRoot.appendingPathComponent(
            "temporary-shard.safetensors")
        let renamed = fixture.registration.sourceRoot.appendingPathComponent(
            "MODEL-00001-OF-00026.SAFETENSORS")
        try renamePath(from: fixture.shard, to: temporary)
        try renamePath(from: temporary, to: renamed)

        do {
            _ = try fixture.handle.preadTensorRange(
                token, byteOffset: 0,
                byteCount: UInt64(DirectoryScanTensorFixture.payload.count),
                expectedByteCount: UInt64(DirectoryScanTensorFixture.payload.count),
                allocationBudget: UInt64(DirectoryScanTensorFixture.payload.count))
            Issue.record("expected the retained shard name mutation to be rejected")
        } catch let error as OfficialSourceHandleError {
            guard case let .replaced(reason) = error else {
                Issue.record("expected .replaced, got \(error)")
                return
            }
            #expect(reason == "source shard name changed: \(DirectoryScanTensorFixture.shardName)")
        } catch {
            Issue.record("expected OfficialSourceHandleError.replaced, got \(error)")
        }
    }

    @Test func retainedTensorRangeRemainsReadableWithUnlistedNonASCIISourceEntry() throws {
        let fixture = try DirectoryScanTensorFixture.make()
        defer { fixture.remove() }
        let unexpected = fixture.registration.sourceRoot.appendingPathComponent(
            "source-\u{00E9}")
        try Data("irrelevant".utf8).write(to: unexpected)

        let token = try fixture.handle.admitTensor(
            shardName: DirectoryScanTensorFixture.shardName,
            tensorName: DirectoryScanTensorFixture.tensorName)
        let bytes = try fixture.handle.preadTensorRange(
            token, byteOffset: 0, byteCount: UInt64(DirectoryScanTensorFixture.payload.count),
            expectedByteCount: UInt64(DirectoryScanTensorFixture.payload.count),
            allocationBudget: UInt64(DirectoryScanTensorFixture.payload.count))

        #expect(bytes == Data(DirectoryScanTensorFixture.payload))
    }
}

private struct DirectoryScanTensorFixture {
    static let shardName = "model-00001-of-00026.safetensors"
    static let tensorName = "synthetic.weight"
    static let payload: [UInt8] = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
    ]
    static let header = #"{"synthetic.weight":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}"#

    let registration: Task68SyntheticRegistrationFixture
    let shard: URL
    let handle: OfficialSourceHandle

    static func make() throws -> Self {
        let registration = try Task68SyntheticRegistrationFixture.make()
        let shard = registration.sourceRoot.appendingPathComponent(shardName)
        do {
            try fileBytes().write(to: shard)
            let handle = try OfficialSourceHandle(
                registrationURL: registration.modelDirectory)
            return Self(registration: registration, shard: shard, handle: handle)
        } catch {
            registration.remove()
            throw error
        }
    }

    func remove() {
        registration.remove()
    }

    private static func fileBytes() -> Data {
        var headerBytes = Data(header.utf8)
        let padding = (8 - headerBytes.count % 8) % 8
        headerBytes.append(contentsOf: repeatElement(UInt8(0x20), count: padding))
        var headerLength = UInt64(headerBytes.count).littleEndian
        var result = Data(bytes: &headerLength, count: MemoryLayout<UInt64>.size)
        result.append(headerBytes)
        result.append(contentsOf: payload)
        return result
    }
}

private func renamePath(from source: URL, to destination: URL) throws {
    let result = source.path.withCString { sourcePath in
        destination.path.withCString { destinationPath in
            Darwin.rename(sourcePath, destinationPath)
        }
    }
    guard result == 0 else {
        throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
    }
}

private func expectDirectoryScanReplacement(
    _ operation: () throws -> Void,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    do {
        try operation()
        Issue.record("expected an unexpected registration entry to be rejected",
                     sourceLocation: sourceLocation)
    } catch let error as OfficialSourceHandleError {
        guard case .replaced = error else {
            Issue.record("expected .replaced, got \(error)", sourceLocation: sourceLocation)
            return
        }
    } catch {
        Issue.record("expected OfficialSourceHandleError.replaced, got \(error)",
                     sourceLocation: sourceLocation)
    }
}
