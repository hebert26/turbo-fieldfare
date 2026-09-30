import Darwin
import Foundation

/// Actual filesystem changes for source-identity transaction coverage. These
/// helpers alter the fixture's routed shard, not the injected protected-read
/// hook outcome.
extension QwenBF16TextRunnerFixture.Source {
    func mutateRoutedShardPayloadInPlace() throws {
        let url = sourceRoot.appendingPathComponent(shardNames[1])
        let file = try Data(contentsOf: url)
        let payloadOffset = try Self.payloadOffset(in: file)
        guard payloadOffset < file.count else {
            throw QwenBF16TransactionFixtureFileError.missingPayload
        }
        let descriptor = open(url.path, O_RDWR | O_CLOEXEC)
        guard descriptor >= 0 else {
            throw QwenBF16TransactionFixtureFileError.systemCall("open", errno)
        }
        defer { close(descriptor) }

        var original = file[payloadOffset]
        original ^= 0x01
        let written = withUnsafePointer(to: &original) {
            pwrite(descriptor, $0, 1, off_t(payloadOffset))
        }
        guard written == 1 else {
            throw QwenBF16TransactionFixtureFileError.systemCall("pwrite", errno)
        }
        guard fsync(descriptor) == 0 else {
            throw QwenBF16TransactionFixtureFileError.systemCall("fsync", errno)
        }
    }

    func replaceRoutedShardWithIdenticalBytes() throws {
        let destination = sourceRoot.appendingPathComponent(shardNames[1])
        let bytes = try Data(contentsOf: destination)
        let replacement = root.appendingPathComponent("replacement-\(UUID().uuidString).tmp")
        defer { try? FileManager.default.removeItem(at: replacement) }
        try bytes.write(to: replacement, options: .withoutOverwriting)
        let result = replacement.path.withCString { sourcePath in
            destination.path.withCString { destinationPath in
                Darwin.rename(sourcePath, destinationPath)
            }
        }
        guard result == 0 else {
            throw QwenBF16TransactionFixtureFileError.systemCall("rename", errno)
        }
    }

    private static func payloadOffset(in file: Data) throws -> Int {
        guard file.count >= MemoryLayout<UInt64>.stride else {
            throw QwenBF16TransactionFixtureFileError.invalidHeader
        }
        let headerLength = file.prefix(MemoryLayout<UInt64>.stride).withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
        guard let length = Int(exactly: headerLength),
              length <= file.count - MemoryLayout<UInt64>.stride else {
            throw QwenBF16TransactionFixtureFileError.invalidHeader
        }
        return MemoryLayout<UInt64>.stride + length
    }
}

enum QwenBF16TransactionFixtureFileError: Error {
    case invalidHeader
    case missingPayload
    case systemCall(String, Int32)
}
