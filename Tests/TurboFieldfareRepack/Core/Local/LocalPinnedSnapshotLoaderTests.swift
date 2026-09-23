import CryptoKit
import Darwin
import Foundation
import Testing

@testable import TurboFieldfareRepackCore

/// Phase 5 tasks 5.2/5.3 coverage for the bounded, no-follow local snapshot loader.
///
/// Every container here is generated in a temporary directory: a tiny
/// `model.safetensors.index.json`, a tiny official-shaped `config.json` with no
/// quantization slot, and small BF16 safetensors shards. Digests are computed
/// independently with CryptoKit for the injected identity, so no production
/// identity pin is weakened and no public hash bypass is used.
///
/// Error assertions name the exact `RepackError` case the loader contract fixes,
/// except for filesystem failures where the OS supplies the errno; those assert
/// the failure class (missing/unreferenced, or unsafe file type).
@Suite struct LocalPinnedSnapshotLoaderTests {

  // MARK: - Fixture types

  private struct TensorSpec {
    let name: String
    let dtype: String
    let shape: [UInt64]
    let dataOffsets: [UInt64]
  }

  private struct ShardSpec {
    let filename: String
    let tensors: [TensorSpec]
    var payloadLengthOverride: Int? = nil
  }

  private struct ExpectedTensor {
    let name: String
    let shardFilename: String
    let shape: [UInt64]
    let sizeBytes: UInt64
  }

  private struct TinyContainer {
    let parent: String
    let directory: String
    let identity: LocalPinnedSnapshotIdentity
    let indexDigest: String
    let shardFilenames: [String]
    let tensors: [ExpectedTensor]

    func path(_ name: String) -> String {
      (directory as NSString).appendingPathComponent(name)
    }
  }

  // MARK: - Fixture construction

  private static let tinyRepository = "test/local-tiny"
  private static let tinyRevision = "0123456789abcdef0123456789abcdef01234567"
  private static let indexFilename = "model.safetensors.index.json"
  private static let configFilename = "config.json"
  private static let shardOne = "model-00001-of-00002.safetensors"
  private static let shardTwo = "model-00002-of-00002.safetensors"

  /// Official-shaped `config.json` with no `quantization` slot. The official
  /// Qwen 3.6 checkpoint omits that slot, so the local path must not require it.
  private static let officialShapedConfig = Data(
    #"{"architectures":["Qwen3_5MoeForConditionalGeneration"],"model_type":"qwen3_5_moe"}"#.utf8)

  /// The present canonical official snapshot, resolved from this file's path.
  /// Nil when the checkout does not carry it; the canonical proof test is then
  /// reported as skipped rather than silently passing.
  private static let canonicalSnapshotDirectory: String? = {
    var root = URL(fileURLWithPath: #filePath)
    for _ in 0..<5 { root.deleteLastPathComponent() }
    let candidate = root
      .appendingPathComponent("scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0")
      .path
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: candidate, isDirectory: &isDirectory),
          isDirectory.boolValue else { return nil }
    return candidate
  }()

  private static var defaultShards: [ShardSpec] {
    [
      ShardSpec(filename: shardOne, tensors: [
        TensorSpec(name: "a.weight", dtype: "BF16", shape: [2, 2], dataOffsets: [0, 8]),
        TensorSpec(name: "b.weight", dtype: "BF16", shape: [4], dataOffsets: [8, 16]),
      ]),
      ShardSpec(filename: shardTwo, tensors: [
        TensorSpec(name: "c.bias", dtype: "BF16", shape: [1], dataOffsets: [0, 2]),
      ]),
    ]
  }

  private static var defaultWeightMap: [String: String] {
    [
      "a.weight": shardOne,
      "b.weight": shardOne,
      "c.bias": shardTwo,
    ]
  }

  private static func buildTiny(
    tag: String,
    shards: [ShardSpec] = defaultShards,
    config: Data = officialShapedConfig,
    extraFiles: [String: Data] = [:],
    extraSidecarNames: [String] = [],
    indexOverride: Data? = nil,
    skipShardFiles: Set<String> = [],
    indexExcludedNames: Set<String> = [],
    weightMapOverride: [String: String]? = nil
  ) throws -> TinyContainer {
    let parent = (NSTemporaryDirectory() as NSString)
      .appendingPathComponent("turbofieldfare-local-\(tag)-\(UUID().uuidString)")
    let directory = (parent as NSString).appendingPathComponent("snapshot")
    try? FileManager.default.removeItem(atPath: parent)
    try FileManager.default.createDirectory(atPath: directory,
                                            withIntermediateDirectories: true)

    try config.write(to: URL(fileURLWithPath:
      (directory as NSString).appendingPathComponent(configFilename)))
    for (name, data) in extraFiles {
      try data.write(to: URL(fileURLWithPath:
        (directory as NSString).appendingPathComponent(name)))
    }

    var weightMap: [String: String] = [:]
    var expected: [ExpectedTensor] = []
    for shard in shards {
      for tensor in shard.tensors where !indexExcludedNames.contains(tensor.name) {
        weightMap[tensor.name] = shard.filename
        expected.append(ExpectedTensor(
          name: tensor.name,
          shardFilename: shard.filename,
          shape: tensor.shape,
          sizeBytes: tensor.dataOffsets[1] - tensor.dataOffsets[0]))
      }
      if !skipShardFiles.contains(shard.filename) {
        try shardBytes(shard).write(to: URL(fileURLWithPath:
          (directory as NSString).appendingPathComponent(shard.filename)))
      }
    }
    if let weightMapOverride { weightMap = weightMapOverride }

    let indexData: Data
    if let indexOverride {
      indexData = indexOverride
    } else {
      indexData = try indexBytes(weightMap: weightMap)
    }
    try indexData.write(to: URL(fileURLWithPath:
      (directory as NSString).appendingPathComponent(indexFilename)))

    var digests: [String: String] = [
      configFilename: sha256Hex(config),
      indexFilename: sha256Hex(indexData),
    ]
    for name in extraSidecarNames {
      let path = (directory as NSString).appendingPathComponent(name)
      digests[name] = sha256Hex(try Data(contentsOf: URL(fileURLWithPath: path)))
    }

    return TinyContainer(
      parent: parent,
      directory: directory,
      identity: LocalPinnedSnapshotIdentity(
        repository: tinyRepository,
        revision: tinyRevision,
        sidecarSHA256: digests,
        enforcesOfficialTensorMap: false),
      indexDigest: sha256Hex(indexData),
      shardFilenames: Set(weightMap.values).sorted(),
      tensors: expected)
  }

  private static func indexBytes(weightMap: [String: String]) throws -> Data {
    try JSONSerialization.data(
      withJSONObject: ["metadata": ["format": "pt"], "weight_map": weightMap],
      options: [.sortedKeys])
  }

  private static func shardBytes(_ shard: ShardSpec) throws -> Data {
    var header: [String: Any] = [:]
    var maxEnd: UInt64 = 0
    for tensor in shard.tensors {
      header[tensor.name] = [
        "dtype": tensor.dtype,
        "shape": tensor.shape.map { Int($0) },
        "data_offsets": tensor.dataOffsets.map { Int($0) },
      ]
      maxEnd = max(maxEnd, tensor.dataOffsets[1])
    }
    header["__metadata__"] = ["format": "pt"]
    var headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    while headerData.count % 8 != 0 { headerData.append(0x20) }

    var out = Data()
    var length = UInt64(headerData.count).littleEndian
    withUnsafeBytes(of: &length) { raw in out.append(contentsOf: raw) }
    out.append(headerData)
    out.append(Data(repeating: 0xA5, count: shard.payloadLengthOverride ?? Int(maxEnd)))
    return out
  }

  // MARK: - Independent helpers

  /// CryptoKit-backed SHA-256, independent of the production `Sha256Stream`.
  private static func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  private static func rejection(
    _ directory: String,
    _ identity: LocalPinnedSnapshotIdentity
  ) -> RepackError? {
    do {
      _ = try LocalPinnedSnapshotLoader.load(snapshotDirectory: directory,
                                             expectedIdentity: identity)
      return nil
    } catch let error as RepackError {
      return error
    } catch {
      Issue.record("loader threw a non-RepackError: \(error)")
      return nil
    }
  }

  private static func isMissingOrUnreferenced(_ error: RepackError) -> Bool {
    // A missing leaf surfaces as the OS errno from open(2); an extra shard is
    // rejected by the loader itself.
    switch error {
    case .installStateCorrupt, .installStateMissing, .fileOpenFailed, .fileStatFailed:
      return true
    default:
      return false
    }
  }

  private static func isUnsafeFile(_ error: RepackError) -> Bool {
    switch error {
    case .fileOpenFailed, .fileStatFailed, .installPathUnsafe: return true
    default: return false
    }
  }

  private static func isIndexOrMembership(_ error: RepackError) -> Bool {
    switch error {
    case .indexJsonInvalid, .shapeMismatch, .missingTensor, .dtypeMismatch,
         .installStateCorrupt:
      return true
    default:
      return false
    }
  }

  private static func relativeListing(_ root: String) -> [String] {
    guard let enumerator = FileManager.default.enumerator(atPath: root) else { return [] }
    return enumerator.compactMap { $0 as? String }.sorted()
  }

  private static func writeRaw(_ bytes: Data, to path: String) throws {
    try bytes.write(to: URL(fileURLWithPath: path))
  }

  private static func writeHeaderPrefix(_ headerSize: UInt64, payload: Int, to path: String) throws {
    var out = Data()
    var length = headerSize.littleEndian
    withUnsafeBytes(of: &length) { raw in out.append(contentsOf: raw) }
    out.append(Data(repeating: 0, count: payload))
    try out.write(to: URL(fileURLWithPath: path))
  }

  // MARK: - Acceptance

  @Test func validTinySnapshotYieldsOneMetadataObject() throws {
    let container = try Self.buildTiny(tag: "valid")
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let snapshot = try LocalPinnedSnapshotLoader.load(
      snapshotDirectory: container.directory, expectedIdentity: container.identity)

    #expect(snapshot.indexSHA256 == container.indexDigest)
    #expect(snapshot.shardFilenames == container.shardFilenames)
    #expect(snapshot.shardFilenames.count == 2)
    #expect(snapshot.tensors.count == container.tensors.count)

    let byName = Dictionary(
      snapshot.tensors.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
    #expect(Set(byName.keys) == Set(container.tensors.map(\.name)))
    for expected in container.tensors {
      let actual = try #require(byName[expected.name])
      #expect(actual.dtype == .bf16)
      #expect(actual.shape == expected.shape)
      #expect(actual.sizeBytes == expected.sizeBytes)
      #expect(actual.shardPath == container.path(expected.shardFilename))
      #expect(actual.absoluteOffset >= 8)
    }
    // Control on the rejection helper: the unmutated container returns no error,
    // so the negative controls below are not vacuously satisfied.
    if let error = Self.rejection(container.directory, container.identity) {
      Issue.record("a valid container was rejected: \(error)")
    }
  }

  @Test func officialShapedConfigWithoutQuantizationSlotIsAccepted() throws {
    let configText = String(decoding: Self.officialShapedConfig, as: UTF8.self)
    #expect(!configText.contains("quantization"))

    let container = try Self.buildTiny(tag: "no-quant-slot")
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let snapshot = try LocalPinnedSnapshotLoader.load(
      snapshotDirectory: container.directory, expectedIdentity: container.identity)
    // No Gemma/MLX affine fallback: every accepted tensor stays BF16.
    #expect(snapshot.tensors.allSatisfy { $0.dtype == .bf16 })
  }

  @Test func unrelatedCanonicalFilesAreTolerated() throws {
    let unrelated: [String: Data] = [
      "README.md": Data("model card".utf8),
      "LICENSE": Data("license".utf8),
      "vocab.json": Data("{}".utf8),
      "merges.txt": Data("#version".utf8),
      "SHA256SUMS": Data("digests".utf8),
      "PREPARATION.md": Data("prep".utf8),
      "STRUCTURAL_VERIFICATION.json": Data("{}".utf8),
      "download-official.sh": Data("#!/bin/sh".utf8),
      "chat_template.jinja": Data("{{ }}".utf8),
    ]
    let container = try Self.buildTiny(tag: "unrelated", extraFiles: unrelated)
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let snapshot = try LocalPinnedSnapshotLoader.load(
      snapshotDirectory: container.directory, expectedIdentity: container.identity)
    #expect(snapshot.tensors.count == 3)
  }

  @Test func multipleProvidedSidecarDigestsAreAllEnforced() throws {
    let tokenizer = Data(#"{"model":{"type":"BPE"}}"#.utf8)
    let container = try Self.buildTiny(
      tag: "multi-sidecar",
      extraFiles: ["tokenizer.json": tokenizer],
      extraSidecarNames: ["tokenizer.json"])
    defer { try? FileManager.default.removeItem(atPath: container.parent) }
    #expect(container.identity.sidecarSHA256.keys.sorted()
      == ["config.json", "model.safetensors.index.json", "tokenizer.json"])

    _ = try LocalPinnedSnapshotLoader.load(
      snapshotDirectory: container.directory, expectedIdentity: container.identity)

    // Flip one byte of the third sidecar after its digest was captured.
    var tampered = tokenizer
    tampered[0] ^= 0xFF
    try Self.writeRaw(tampered, to: container.path("tokenizer.json"))

    let error = try #require(Self.rejection(container.directory, container.identity))
    guard case .sourceFingerprintRejected(let path, let sha) = error else {
      Issue.record("unexpected error \(error)")
      return
    }
    #expect(path == container.path("tokenizer.json"))
    #expect(sha == Self.sha256Hex(tampered))
  }

  // MARK: - Identity and sidecar digests

  @Test func changedSidecarDigestIsRejectedBeforeAnythingElse() throws {
    let container = try Self.buildTiny(tag: "digest-mismatch")
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let changed = Data(#"{"model_type":"qwen3_5_moe","changed":true}"#.utf8)
    try Self.writeRaw(changed, to: container.path(Self.configFilename))

    let error = try #require(Self.rejection(container.directory, container.identity))
    guard case .sourceFingerprintRejected(let path, let sha) = error else {
      Issue.record("unexpected error \(error)")
      return
    }
    #expect(path == container.path(Self.configFilename))
    #expect(sha == Self.sha256Hex(changed))
  }

  @Test func missingSidecarIsRejected() throws {
    let container = try Self.buildTiny(tag: "missing-sidecar")
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    try FileManager.default.removeItem(atPath: container.path(Self.configFilename))

    let error = try #require(Self.rejection(container.directory, container.identity))
    #expect(Self.isMissingOrUnreferenced(error), "unexpected error \(error)")
  }

  @Test func sidecarDigestEntryWithoutAFileIsRejected() throws {
    let container = try Self.buildTiny(tag: "absent-map-entry")
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let identity = LocalPinnedSnapshotIdentity(
      repository: container.identity.repository,
      revision: container.identity.revision,
      sidecarSHA256: container.identity.sidecarSHA256
        .merging(["generation_config.json": String(repeating: "a", count: 64)]) { _, new in new },
      enforcesOfficialTensorMap: false)

    let error = try #require(Self.rejection(container.directory, identity))
    #expect(Self.isMissingOrUnreferenced(error), "unexpected error \(error)")
  }

  // MARK: - Shard membership

  @Test func unreferencedShardIsRejected() throws {
    let extraShard = try Self.shardBytes(Self.defaultShards[0])
    let container = try Self.buildTiny(
      tag: "extra-shard",
      extraFiles: ["model-00009-of-00009.safetensors": extraShard])
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let error = try #require(Self.rejection(container.directory, container.identity))
    guard case .installStateCorrupt(let path, let detail) = error else {
      Issue.record("unexpected error \(error)")
      return
    }
    #expect(path == container.directory)
    #expect(detail == "unreferenced shard model-00009-of-00009.safetensors")
  }

  @Test func referencedShardMissingFromDiskIsRejected() throws {
    let container = try Self.buildTiny(
      tag: "missing-shard", skipShardFiles: [Self.shardTwo])
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let error = try #require(Self.rejection(container.directory, container.identity))
    #expect(Self.isMissingOrUnreferenced(error), "unexpected error \(error)")
  }

  @Test func traversalShardNameIsRejectedBeforeAnyOutOfDirectoryRead() throws {
    let escape = try Self.shardBytes(Self.defaultShards[0])
    let container = try Self.buildTiny(
      tag: "traversal",
      weightMapOverride: [
        "a.weight": "../escape.safetensors",
        "b.weight": Self.shardOne,
        "c.bias": Self.shardTwo,
      ])
    defer { try? FileManager.default.removeItem(atPath: container.parent) }
    let escapePath = (container.parent as NSString)
      .appendingPathComponent("escape.safetensors")
    try Self.writeRaw(escape, to: escapePath)

    let error = try #require(Self.rejection(container.directory, container.identity))
    #expect(Self.isIndexOrMembership(error), "unexpected error \(error)")
    // The loader must not have adopted the out-of-directory file as a shard.
    #expect(Self.relativeListing(container.parent).contains("escape.safetensors"))
  }

  @Test func absoluteShardPathIsRejected() throws {
    let container = try Self.buildTiny(
      tag: "absolute",
      weightMapOverride: [
        "a.weight": "/etc/hosts",
        "b.weight": Self.shardOne,
        "c.bias": Self.shardTwo,
      ])
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let error = try #require(Self.rejection(container.directory, container.identity))
    #expect(Self.isIndexOrMembership(error), "unexpected error \(error)")
  }

  // MARK: - File types

  @Test func sidecarLeafSymlinkIsRejected() throws {
    let container = try Self.buildTiny(
      tag: "symlink-sidecar",
      extraFiles: ["config.real": Self.officialShapedConfig])
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let link = container.path(Self.configFilename)
    try FileManager.default.removeItem(atPath: link)
    try FileManager.default.createSymbolicLink(
      atPath: link, withDestinationPath: "config.real")

    let error = try #require(Self.rejection(container.directory, container.identity))
    #expect(Self.isUnsafeFile(error), "unexpected error \(error)")
  }

  @Test func shardLeafSymlinkIsRejected() throws {
    let container = try Self.buildTiny(
      tag: "symlink-shard",
      extraFiles: ["real.bin": try Self.shardBytes(Self.defaultShards[1])])
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let link = container.path(Self.shardTwo)
    try FileManager.default.removeItem(atPath: link)
    try FileManager.default.createSymbolicLink(atPath: link, withDestinationPath: "real.bin")

    let error = try #require(Self.rejection(container.directory, container.identity))
    #expect(Self.isUnsafeFile(error), "unexpected error \(error)")
  }

  @Test func fifoShardIsRejectedWithoutBlocking() throws {
    let container = try Self.buildTiny(tag: "fifo-shard", skipShardFiles: [Self.shardTwo])
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let fifo = container.path(Self.shardTwo)
    #expect(mkfifo(fifo, 0o600) == 0)

    let clock = ContinuousClock()
    let started = clock.now
    let error = Self.rejection(container.directory, container.identity)
    let elapsed = clock.now - started
    guard let error else {
      Issue.record("a FIFO leaf must be rejected")
      return
    }
    #expect(Self.isUnsafeFile(error), "unexpected error \(error)")
    // O_NONBLOCK is what keeps open(2) from waiting for a FIFO writer.
    #expect(elapsed < .seconds(5), "FIFO open blocked for \(elapsed)")
  }

  @Test func directoryShardIsRejected() throws {
    let container = try Self.buildTiny(tag: "dir-shard", skipShardFiles: [Self.shardTwo])
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    try FileManager.default.createDirectory(
      atPath: container.path(Self.shardTwo), withIntermediateDirectories: false)

    let error = try #require(Self.rejection(container.directory, container.identity))
    #expect(Self.isUnsafeFile(error), "unexpected error \(error)")
  }

  // MARK: - Header bounds and dtypes

  @Test func shortHeaderIsRejected() throws {
    let container = try Self.buildTiny(tag: "short-header", skipShardFiles: [Self.shardTwo])
    defer { try? FileManager.default.removeItem(atPath: container.parent) }
    try Self.writeRaw(Data([0, 0, 0]), to: container.path(Self.shardTwo))

    let error = try #require(Self.rejection(container.directory, container.identity))
    guard case .safetensorsHeaderInvalid(let path, let detail) = error else {
      Issue.record("unexpected error \(error)")
      return
    }
    #expect(path == container.path(Self.shardTwo))
    #expect(detail == "file too short")
  }

  @Test func oversizedHeaderIsRejected() throws {
    let container = try Self.buildTiny(tag: "big-header", skipShardFiles: [Self.shardTwo])
    defer { try? FileManager.default.removeItem(atPath: container.parent) }
    try Self.writeHeaderPrefix(
      Safetensors.maxHeaderBytes + 1, payload: 64, to: container.path(Self.shardTwo))

    let error = try #require(Self.rejection(container.directory, container.identity))
    guard case .safetensorsHeaderTooLarge(let path, let size) = error else {
      Issue.record("unexpected error \(error)")
      return
    }
    #expect(path == container.path(Self.shardTwo))
    #expect(size == Safetensors.maxHeaderBytes + 1)
  }

  @Test func nonBF16DtypeIsRejected() throws {
    let shards = [ShardSpec(filename: "model-00001-of-00001.safetensors", tensors: [
      TensorSpec(name: "a.weight", dtype: "F16", shape: [4], dataOffsets: [0, 8]),
    ])]
    let container = try Self.buildTiny(tag: "f16", shards: shards)
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let error = try #require(Self.rejection(container.directory, container.identity))
    guard case .dtypeMismatch(let name, let detail) = error else {
      Issue.record("unexpected error \(error)")
      return
    }
    #expect(name == "a.weight")
    #expect(detail == "local Qwen snapshots require BF16")
  }

  @Test func unknownDtypeIsRejected() throws {
    let shards = [ShardSpec(filename: "model-00001-of-00001.safetensors", tensors: [
      TensorSpec(name: "a.weight", dtype: "I8", shape: [4], dataOffsets: [0, 4]),
    ])]
    let container = try Self.buildTiny(tag: "i8", shards: shards)
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let error = try #require(Self.rejection(container.directory, container.identity))
    guard case .safetensorsUnknownDtype(let path, let dtype) = error else {
      Issue.record("unexpected error \(error)")
      return
    }
    #expect(path == container.path("model-00001-of-00001.safetensors"))
    #expect(dtype == "I8")
  }

  // MARK: - Ranges and shapes

  @Test func oneByteRangeDisagreementIsRejected() throws {
    let shards = [ShardSpec(filename: "model-00001-of-00001.safetensors", tensors: [
      TensorSpec(name: "a.weight", dtype: "BF16", shape: [2, 2], dataOffsets: [0, 6]),
    ])]
    let container = try Self.buildTiny(tag: "shape-bytes", shards: shards)
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let error = try #require(Self.rejection(container.directory, container.identity))
    guard case .shapeMismatch(let name, let detail) = error else {
      Issue.record("unexpected error \(error)")
      return
    }
    #expect(name == "a.weight")
    #expect(detail == "shape product 4*2 != size 6")
  }

  @Test func tensorBeyondFileSizeIsRejected() throws {
    let shards = [ShardSpec(
      filename: "model-00001-of-00001.safetensors",
      tensors: [TensorSpec(name: "a.weight", dtype: "BF16", shape: [500], dataOffsets: [0, 1000])],
      payloadLengthOverride: 8)]
    let container = try Self.buildTiny(tag: "out-of-range", shards: shards)
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let error = try #require(Self.rejection(container.directory, container.identity))
    guard case .safetensorsTensorOutOfRange(let path, let name, _, _) = error else {
      Issue.record("unexpected error \(error)")
      return
    }
    #expect(path == container.path("model-00001-of-00001.safetensors"))
    #expect(name == "a.weight")
  }

  @Test func overlappingTensorRangesAreRejected() throws {
    let shards = [ShardSpec(filename: "model-00001-of-00001.safetensors", tensors: [
      TensorSpec(name: "a.weight", dtype: "BF16", shape: [4], dataOffsets: [0, 8]),
      TensorSpec(name: "b.weight", dtype: "BF16", shape: [4], dataOffsets: [4, 12]),
    ])]
    let container = try Self.buildTiny(tag: "overlap", shards: shards)
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let error = try #require(Self.rejection(container.directory, container.identity))
    guard case .safetensorsHeaderInvalid = error else {
      Issue.record("unexpected error \(error)")
      return
    }
  }

  @Test func shapeProductOverflowIsRejected() throws {
    let huge = UInt64(1) << 62
    let shards = [ShardSpec(filename: "model-00001-of-00001.safetensors", tensors: [
      TensorSpec(name: "a.weight", dtype: "BF16", shape: [huge, huge], dataOffsets: [0, 8]),
    ])]
    let container = try Self.buildTiny(tag: "overflow", shards: shards)
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let error = try #require(Self.rejection(container.directory, container.identity))
    guard case .safetensorsHeaderInvalid = error else {
      Issue.record("unexpected error \(error)")
      return
    }
  }

  // MARK: - Index/header agreement

  @Test func headerTensorMissingFromIndexIsRejected() throws {
    let container = try Self.buildTiny(
      tag: "extra-header-name", indexExcludedNames: ["b.weight"])
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let error = try #require(Self.rejection(container.directory, container.identity))
    guard case .indexJsonInvalid(let path, let detail) = error else {
      Issue.record("unexpected error \(error)")
      return
    }
    #expect(path == container.path(Self.indexFilename))
    #expect(detail == "tensor names disagree for \(Self.shardOne)")
  }

  @Test func indexNameMissingFromEveryHeaderIsRejected() throws {
    var weightMap = Self.defaultWeightMap
    weightMap["ghost.weight"] = Self.shardOne
    let container = try Self.buildTiny(tag: "ghost-name", weightMapOverride: weightMap)
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let error = try #require(Self.rejection(container.directory, container.identity))
    #expect(Self.isIndexOrMembership(error), "unexpected error \(error)")
  }

  @Test func wrongShardAssignmentIsRejected() throws {
    let container = try Self.buildTiny(
      tag: "wrong-shard",
      weightMapOverride: [
        "a.weight": Self.shardTwo,
        "b.weight": Self.shardOne,
        "c.bias": Self.shardTwo,
      ])
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let error = try #require(Self.rejection(container.directory, container.identity))
    #expect(Self.isIndexOrMembership(error), "unexpected error \(error)")
  }

  @Test func malformedIndexJsonIsRejected() throws {
    let container = try Self.buildTiny(
      tag: "bad-index", indexOverride: Data(#"{"weight_map": "#.utf8))
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let error = try #require(Self.rejection(container.directory, container.identity))
    guard case .indexJsonInvalid = error else {
      Issue.record("unexpected error \(error)")
      return
    }
  }

  @Test func emptyWeightMapIsRejected() throws {
    let container = try Self.buildTiny(
      tag: "empty-index", indexOverride: try Self.indexBytes(weightMap: [:]))
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    let error = try #require(Self.rejection(container.directory, container.identity))
    guard case .indexJsonInvalid = error else {
      Issue.record("unexpected error \(error)")
      return
    }
  }

  // MARK: - Official identity seam

  @Test func officialPinnedIdentityStillRejectsATinyContainer() throws {
    let container = try Self.buildTiny(tag: "official-identity")
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    // The production catalog entry, unchanged: exact repository/revision and the
    // Phase 1 digest table. A tiny fixture must not satisfy it.
    let official = LocalPinnedSnapshotIdentity(source: ModelSourceCatalog.qwen)
    #expect(official.enforcesOfficialTensorMap)
    #expect(official.sidecarSHA256 == Qwen36OfficialMetadata.sidecarSHA256)
    #expect(official.sidecarSHA256 != container.identity.sidecarSHA256)

    let error = try #require(Self.rejection(container.directory, official))
    guard case .sourceFingerprintRejected = error else {
      Issue.record("unexpected error \(error)")
      return
    }
  }

  @Test func tinyIdentityIsNotAnIdentityPinBypass() throws {
    let container = try Self.buildTiny(tag: "tiny-identity")
    defer { try? FileManager.default.removeItem(atPath: container.parent) }

    #expect(container.identity.enforcesOfficialTensorMap == false)
    #expect(container.identity.repository != ModelSourceCatalog.qwen.repository)
    #expect(container.identity.revision != ModelSourceCatalog.qwen.revision)

    // Enforcement is only relaxed for test identities; a wrong digest still fails.
    let flipped = LocalPinnedSnapshotIdentity(
      repository: container.identity.repository,
      revision: container.identity.revision,
      sidecarSHA256: container.identity.sidecarSHA256
        .merging([Self.configFilename: String(repeating: "0", count: 64)]) { _, new in new },
      enforcesOfficialTensorMap: false)
    let error = try #require(Self.rejection(container.directory, flipped))
    guard case .sourceFingerprintRejected = error else {
      Issue.record("unexpected error \(error)")
      return
    }
  }

  // MARK: - Canonical snapshot proof

  /// Task 5.2/5.3 proof against the real pinned official snapshot. The loader is
  /// bounded to sidecar digests and shard headers; it never reads tensor
  /// payloads, rehashes shards, downloads, or starts a network session.
  @Test(.enabled(
    if: canonicalSnapshotDirectory != nil,
    "canonical official Qwen snapshot is not present in this checkout"))
  func canonicalOfficialSnapshotValidatesWithoutPayloadReads() throws {
    let directory = try #require(Self.canonicalSnapshotDirectory)
    let snapshot = try LocalPinnedSnapshotLoader.load(
      snapshotDirectory: directory,
      expectedIdentity: LocalPinnedSnapshotIdentity(source: ModelSourceCatalog.qwen))

    // 26 index-referenced shards, named as the canonical snapshot names them.
    let expectedShards = (1...26).map {
      String(format: "model-%05d-of-00026.safetensors", $0)
    }
    #expect(snapshot.shardFilenames == expectedShards)

    // The pinned index digest, and the exhaustive 1,045-name BF16 tensor set.
    #expect(snapshot.indexSHA256
      == Qwen36OfficialMetadata.sidecarSHA256["model.safetensors.index.json"])
    #expect(snapshot.tensors.count == 1045)
    #expect(snapshot.tensors.allSatisfy { $0.dtype == .bf16 })
    #expect(Set(snapshot.tensors.map(\.name)).count == 1045)

    // The independent P1 classifier must agree, including exactly 19 MTP tensors.
    let categories = try QwenOfficialTensorMap.classify(
      snapshot.tensors.map {
        QwenOfficialTensorDescriptor(name: $0.name, dataType: $0.dtype)
      })
    #expect(categories.count == 1045)
    #expect(categories.filter { $0 == .mtpOmitted }.count == 19)
  }

  // MARK: - Rejection side effects

  @Test func failingLoadCreatesNoFilesOrDirectories() throws {
    let container = try Self.buildTiny(tag: "no-side-effects")
    defer { try? FileManager.default.removeItem(atPath: container.parent) }
    try Self.writeRaw(Data("tampered".utf8), to: container.path(Self.configFilename))

    let before = Self.relativeListing(container.parent)
    _ = Self.rejection(container.directory, container.identity)
    let after = Self.relativeListing(container.parent)
    #expect(before == after, "a rejected load changed the filesystem")
  }
}
