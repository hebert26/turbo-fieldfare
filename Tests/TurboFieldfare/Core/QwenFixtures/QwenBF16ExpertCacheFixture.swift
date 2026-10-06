import Darwin
import Foundation
import TurboFieldfareFormat
@testable import TurboFieldfareOfficialQwenSource

/// Literal BF16 payload for a synthetic expert-cache Safetensors tensor. Shapes
/// may be rank two (shared branch) or rank three (routed experts).
struct QwenBF16ExpertCacheLiteralTensor {
    let name: String
    let shape: [Int]
    let words: [UInt16]

    init(name: String, shape: [Int], words: [UInt16]) {
        precondition(!name.isEmpty && !shape.isEmpty && shape.allSatisfy { $0 > 0 })
        precondition(shape.reduce(1, *) == words.count)
        self.name = name
        self.shape = shape
        self.words = words
    }
}

/// Two synthetic, distinct pinned-inventory shard paths with local literal
/// payloads. No original checkpoint bytes are opened or copied.
struct QwenBF16ExpertCacheSourceFixture {
    let root: URL
    let sourceRoot: URL
    let registrationURL: URL
    let handle: OfficialSourceHandle
    let shardNames: [String]
    let tensorPayloadOffsets: [String: UInt64]
    let tensorShardIndices: [String: Int]
    let tensorWordCounts: [String: Int]

    var gateUpShardName: String { shardNames[0] }
    var downShardName: String { shardNames[1] }

    static func make(
        firstShard: [QwenBF16ExpertCacheLiteralTensor],
        secondShard: [QwenBF16ExpertCacheLiteralTensor]
    ) throws -> Self {
        guard !firstShard.isEmpty, !secondShard.isEmpty,
              Set((firstShard + secondShard).map(\.name)).count == firstShard.count + secondShard.count
        else { throw QwenBF16ExpertCacheFixtureError.invalidTensorList }

        let manager = FileManager.default
        var root = manager.temporaryDirectory.appendingPathComponent(
            "qwen-bf16-expert-cache-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: root, withIntermediateDirectories: false)
        var canonical = [CChar](repeating: 0, count: Int(PATH_MAX))
        guard realpath(root.path, &canonical) != nil else {
            try? manager.removeItem(at: root)
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        let canonicalBytes = canonical.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        root = URL(fileURLWithPath: String(decoding: canonicalBytes, as: UTF8.self),
                   isDirectory: true)

        do {
            let sourceRoot = root.appendingPathComponent("source", isDirectory: true)
            try manager.createDirectory(at: sourceRoot, withIntermediateDirectories: false)
            let registrationParent = root.appendingPathComponent("models", isDirectory: true)
            try manager.createDirectory(at: registrationParent, withIntermediateDirectories: false)
            let registrationURL = registrationParent.appendingPathComponent(
                "synthetic.gturbo", isDirectory: true)
            let identity = OfficialQwenSourceIdentity.pinned
            guard identity.shards.count >= 2 else {
                throw QwenBF16ExpertCacheFixtureError.missingPinnedShardPair
            }
            let shardNames = [identity.shards[0].filename, identity.shards[1].filename]
            let first = try safetensorsFile(firstShard)
            let second = try safetensorsFile(secondShard)
            try first.data.write(
                to: sourceRoot.appendingPathComponent(shardNames[0]),
                options: .withoutOverwriting)
            try second.data.write(
                to: sourceRoot.appendingPathComponent(shardNames[1]),
                options: .withoutOverwriting)

            let descriptor = try OfficialSourceDescriptor(
                repository: identity.repository,
                revision: identity.revision,
                storageProfile: identity.storageProfile,
                sidecarSHA256: identity.sidecarSHA256,
                shards: identity.shards.map {
                    OfficialSourceDescriptor.Shard(filename: $0.filename, sha256: $0.sha256)
                },
                sourceRoot: sourceRoot.path)
            let markerData = try JSONEncoder().encode(descriptor)
            _ = try OfficialSourceRegistration.register(markerData: markerData,
                                                       at: registrationURL)
            let handle = try OfficialSourceHandle(registrationURL: registrationURL)
            let firstAbsoluteOffsets = first.offsets.mapValues { offset in
                UInt64(8 + first.headerLength) + offset
            }
            let secondAbsoluteOffsets = second.offsets.mapValues { offset in
                UInt64(8 + second.headerLength) + offset
            }
            let tensorShardIndices = Dictionary(
                uniqueKeysWithValues: firstShard.map { ($0.name, 0) }
                    + secondShard.map { ($0.name, 1) })
            return Self(
                root: root,
                sourceRoot: sourceRoot,
                registrationURL: registrationURL,
                handle: handle,
                shardNames: shardNames,
                tensorPayloadOffsets: firstAbsoluteOffsets.merging(secondAbsoluteOffsets) {
                    _, later in later
                },
                tensorShardIndices: tensorShardIndices,
                tensorWordCounts: Dictionary(uniqueKeysWithValues:
                    (firstShard + secondShard).map { ($0.name, $0.words.count) }))
        } catch {
            try? manager.removeItem(at: root)
            throw error
        }
    }

    func replaceWord(tensorName: String, index: Int, with word: UInt16) throws {
        guard let payloadOffset = tensorPayloadOffsets[tensorName],
              let wordCount = tensorWordCounts[tensorName],
              index >= 0, index < wordCount else {
            throw QwenBF16ExpertCacheFixtureError.unknownTensor(tensorName)
        }
        guard let tensorShard = tensorShardIndices[tensorName] else {
            throw QwenBF16ExpertCacheFixtureError.unknownTensor(tensorName)
        }
        let byteOffset = payloadOffset + UInt64(index * MemoryLayout<UInt16>.stride)
        let file = sourceRoot.appendingPathComponent(shardNames[tensorShard])
        let handle = try FileHandle(forWritingTo: file)
        defer { try? handle.close() }
        try handle.seek(toOffset: byteOffset)
        var littleEndian = word.littleEndian
        let data = withUnsafeBytes(of: &littleEndian) { Data($0) }
        try handle.write(contentsOf: data)
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }

    private static func safetensorsFile(
        _ tensors: [QwenBF16ExpertCacheLiteralTensor]
    ) throws -> (data: Data, offsets: [String: UInt64], headerLength: Int) {
        var payload = Data()
        var offsets: [String: UInt64] = [:]
        var header: [String: [String: Any]] = [:]
        for tensor in tensors {
            let start = payload.count
            offsets[tensor.name] = UInt64(start)
            for word in tensor.words {
                payload.append(UInt8(truncatingIfNeeded: word))
                payload.append(UInt8(truncatingIfNeeded: word >> 8))
            }
            header[tensor.name] = [
                "dtype": "BF16",
                "shape": tensor.shape,
                "data_offsets": [start, payload.count],
            ]
        }
        var headerBytes = try JSONSerialization.data(withJSONObject: header,
                                                     options: [.sortedKeys])
        headerBytes.append(contentsOf: repeatElement(
            UInt8(0x20), count: (8 - headerBytes.count % 8) % 8))
        var headerLength = UInt64(headerBytes.count).littleEndian
        var file = withUnsafeBytes(of: &headerLength) { Data($0) }
        file.append(headerBytes)
        file.append(payload)
        return (file, offsets, headerBytes.count)
    }
}

enum QwenBF16ExpertCacheFixtureError: Error {
    case invalidTensorList
    case missingPinnedShardPair
    case unknownTensor(String)
}
