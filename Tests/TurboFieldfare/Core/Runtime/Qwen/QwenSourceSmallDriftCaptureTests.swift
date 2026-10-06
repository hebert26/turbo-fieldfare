import CryptoKit
import Foundation
import Metal
import Testing
@testable import TurboFieldfare

private enum SmallDriftError: Error { case invalid(String) }
private struct SmallDriftRoute {
    let logits: [Float]
    let ids: [Int]
    let weights: [Float]
}
private final class SmallDriftCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var stages: [String: [Float]] = [:]
    private var routes: [Int: [Int: SmallDriftRoute]] = [:]
    private var duplicate = false
    func activation(position: Int, layer: Int, stage: String, values: [Float]) {
        guard position == 58, (4...5).contains(layer) else { return }
        lock.lock(); defer { lock.unlock() }
        let key = "layer-\(layer).token-58.\(stage)"
        if stages.updateValue(values, forKey: key) != nil { duplicate = true }
    }
    func route(position: Int, layer: Int, logits: [Float], ids: [Int], weights: [Float]) {
        guard position < 59, layer < 6 else { return }
        lock.lock(); defer { lock.unlock() }
        var rows = routes[layer, default: [:]]
        if rows.updateValue(SmallDriftRoute(logits: logits, ids: ids, weights: weights), forKey: position) != nil {
            duplicate = true
        }
        routes[layer] = rows
    }
    func snapshot() -> (stages: [String: [Float]], routes: [Int: [Int: SmallDriftRoute]], duplicate: Bool) {
        lock.lock(); defer { lock.unlock() }
        return (stages, routes, duplicate)
    }
}
private func smallDriftSHA(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
private func smallDriftRead(_ url: URL, sha: String) throws -> Data {
    guard url.standardizedFileURL == url.resolvingSymlinksInPath(),
          let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize,
          size <= 16 * 1024 * 1024 else { throw SmallDriftError.invalid("noncanonical/oversized input") }
    let data = try Data(contentsOf: url)
    guard smallDriftSHA(data) == sha else { throw SmallDriftError.invalid("changed saved input: \(url.lastPathComponent)") }
    return data
}
private func smallDriftFloats(_ data: Data) throws -> [Float] {
    guard data.count % 4 == 0 else { throw SmallDriftError.invalid("FP32 byte count") }
    return data.withUnsafeBytes { bytes in
        stride(from: 0, to: data.count, by: 4).map {
            Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self)))
        }
    }
}
private func smallDriftFloatData(_ values: [Float]) -> Data {
    var result = Data(capacity: values.count * 4)
    for value in values {
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { result.append(contentsOf: $0) }
    }
    return result
}
private func smallDriftIDData(_ values: [Int]) -> Data {
    var result = Data(capacity: values.count * 8)
    for value in values {
        var bits = Int64(value).littleEndian
        withUnsafeBytes(of: &bits) { result.append(contentsOf: $0) }
    }
    return result
}
private func smallDriftObject(_ data: Data) throws -> [String: Any] {
    guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw SmallDriftError.invalid("receipt object")
    }
    return value
}

@Suite(.serialized) struct QwenSourceSmallDriftCaptureTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TURBO_P23_SMALL_DRIFT_CAPTURE"] == "1",
                   "Main-only six-layer source prefix diagnostic"))
    func captureExif6First59TokensThroughSixLayers() async throws {
        let environment = ProcessInfo.processInfo.environment
        let output = URL(fileURLWithPath: try #require(environment["TURBO_P23_SMALL_DRIFT_OUTPUT"]))
        guard output.standardizedFileURL == output.resolvingSymlinksInPath(),
              !FileManager.default.fileExists(atPath: output.path) else { throw SmallDriftError.invalid("output exists/noncanonical") }
        let repo = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let base = repo.appendingPathComponent("scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-22-codex/takeover-20260930")
        let reference = base.appendingPathComponent("phase23-reference/image-text-run-005")
        let candidate = base.appendingPathComponent("image-text-candidate-002")
        let featureRaw = try smallDriftRead(reference.appendingPathComponent("reference.json"),
            sha: "30492a61586b8c895d3f8791e6333bf6f9db79a528de20580c618a0564e53c97")
        let featureReceipt = try smallDriftObject(featureRaw)
        let featureCase = try #require((featureReceipt["results"] as? [[String: Any]])?.first { $0["id"] as? String == "natural-exif-6" })
        let ids = try #require(featureCase["tokenIds"] as? [Int])
        let positions = try #require(featureCase["positions"] as? [[Int]])
        let padRows = try #require(featureCase["padRows"] as? [Int])
        let featureRecord = try #require(featureCase["features"] as? [String: Any])
        guard ids.count == 100, positions.count == 100, padRows.count == 77,
              smallDriftSHA(try JSONSerialization.data(withJSONObject: positions)) == "b7716271562b33194c95284680887210665d7a0bca120a7e87b642c9853b5870" else {
            throw SmallDriftError.invalid("saved prompt/M-RoPE identity")
        }
        let featureName = try #require(featureRecord["file"] as? String)
        let featureBytes = try smallDriftRead(reference.appendingPathComponent(featureName),
            sha: "a0eb847d26fefc0fa503b4e6269674dad1f3726d4b4da4fb05341d2c20240e01")
        let features = try smallDriftFloats(featureBytes)
        guard features.count == 77 * 2048, features.allSatisfy(\.isFinite) else { throw SmallDriftError.invalid("saved features") }
        let expectedMode = environment["TURBO_P23_SMALL_DRIFT_EXPECTED"] ?? "candidate133"
        guard expectedMode == "candidate133" || expectedMode == "official005" else {
            throw SmallDriftError.invalid("unknown expected prefix mode")
        }
        let baselineRaw: Data
        let expectedDirectory: URL
        let expectedRoutes: [[String: Any]]
        if expectedMode == "official005" {
            baselineRaw = try smallDriftRead(reference.appendingPathComponent("image-text-reference.json"),
                sha: "b4a0decd6b338a021ee6e5998c4a0d76b4622977c48558d2f10bb88eb5192a84")
            let baseline = try smallDriftObject(baselineRaw)
            let expectedCase = try #require((baseline["cases"] as? [[String: Any]])?.first { $0["id"] as? String == "natural-exif-6" })
            guard expectedCase["textOutputDirectory"] as? String == "natural-exif-6-text" else {
                throw SmallDriftError.invalid("official text output directory")
            }
            expectedDirectory = reference.appendingPathComponent("natural-exif-6-text")
            let steps = try #require(expectedCase["steps"] as? [[String: Any]])
            let routes = try #require(steps.first?["routes"] as? [[String: Any]])
            expectedRoutes = try routes.map { route in
                var normalized: [String: Any] = ["layer": try #require(route["layer"] as? Int)]
                for (recordKey, fileKey, shaKey, columns) in [
                    ("routerLogits", "routerLogitsFile", "routerLogitsSHA256", 256),
                    ("top8Weights", "top8WeightsFile", "top8WeightsSHA256", 8),
                    ("top8ExpertIds", "top8ExpertIdsFile", "top8ExpertIdsSHA256", 8),
                ] {
                    let record = try #require(route[recordKey] as? [String: Any])
                    let expectedBytes = 100 * columns * (recordKey == "top8ExpertIds" ? 8 : 4)
                    guard record["shape"] as? [Int] == [100, columns],
                          record["byteCount"] as? Int == expectedBytes else {
                        throw SmallDriftError.invalid("official route shape/ID width")
                    }
                    normalized[fileKey] = try #require(record["file"] as? String)
                    normalized[shaKey] = try #require(record["sha256"] as? String)
                }
                return normalized
            }
        } else {
            baselineRaw = try smallDriftRead(candidate.appendingPathComponent("natural-exif-6.candidate.json"),
                sha: "6fedd8b66b0f1805920fa93ff5127b5fdf0027376d0b0c25ff4d6ca272eef873")
            let baseline = try smallDriftObject(baselineRaw)
            let steps = try #require(baseline["steps"] as? [[String: Any]])
            expectedRoutes = try #require(steps.first?["routes"] as? [[String: Any]])
            expectedDirectory = candidate
        }
        let proposedContext = try MetalContext()
        let registration = repo.appendingPathComponent("scratch/qwen3.6-35b-a3b.gturbo")
        let bundle = try ModelFamilyRuntime.loadBundle(directoryURL: registration, device: proposedContext.device,
            streamingMode: .pread(slotCount: 16), expertCachePolicy: .lfu, integrityPolicy: .sizeCheckTrustedReceipt)
        guard case .qwenOfficialSource(let model) = bundle.runtime,
              model.sourceIdentity?.descriptorContentSHA256 == "c1ac463726b716e7db3a9fbe5db4cb690f95be7e5779aea02e49c2020bca7a7a" else {
            throw SmallDriftError.invalid("source descriptor identity")
        }
        let capture = SmallDriftCapture()
        let hooks = QwenOfficialSourceTransactionHooks(
            observeRoute: { position, layer, logits, ids, weights, _ in
                capture.route(position: position, layer: layer, logits: logits, ids: ids, weights: weights)
            }, observeActivation: { position, layer, stage, values in
                capture.activation(position: position, layer: layer, stage: stage, values: values)
            })
        let runner = try QwenSourceSmallDriftPrefixRunner(model: model, maxContext: 1024, expertSlotCount: 16, hooks: hooks)
        let started = ContinuousClock.now
        for position in 0..<59 {
            let featureRow = padRows.firstIndex(of: position).map { Array(features[$0 * 2048..<($0 + 1) * 2048]) }
            let row = positions[position]
            let mrope = try QwenMRoPEPosition(temporal: row[0], height: row[1], width: row[2])
            let hidden = try await runner.produce(token: Int32(ids[position]), position: position, featureRow: featureRow, mropePosition: mrope)
            guard hidden.count == 2048, hidden.allSatisfy(\.isFinite) else { throw SmallDriftError.invalid("prefix hidden") }
            print("small-drift prefix \(position + 1)/59", terminator: "\n")
        }
        let observed = capture.snapshot()
        guard !observed.duplicate, observed.routes.count == 6 else { throw SmallDriftError.invalid("incomplete/duplicate capture") }
        // This exact comparison is a diagnostic lineage gate, not a change to numerical acceptance.
        // candidate133 reproduces the saved candidate execution. official005 independently
        // checks the post-fix six-layer prefix against the original official execution.
        for layer in 0..<6 {
            let rows = try (0..<59).map { try #require(observed.routes[layer]?[$0]) }
            let expected = try #require(expectedRoutes.first { $0["layer"] as? Int == layer })
            for (field, shaField, data) in [
                ("routerLogitsFile", "routerLogitsSHA256", smallDriftFloatData(rows.flatMap(\.logits))),
                ("top8WeightsFile", "top8WeightsSHA256", smallDriftFloatData(rows.flatMap(\.weights))),
                ("top8ExpertIdsFile", "top8ExpertIdsSHA256", smallDriftIDData(rows.flatMap(\.ids))),
            ] {
                let filename = try #require(expected[field] as? String)
                let digest = try #require(expected[shaField] as? String)
                let original = try smallDriftRead(expectedDirectory.appendingPathComponent(filename), sha: digest)
                let fullCount = field == "routerLogitsFile" ? 100 * 256 * 4 : 100 * 8 * (field == "top8ExpertIdsFile" ? 8 : 4)
                guard original.count == fullCount else { throw SmallDriftError.invalid("expected full route byte count") }
                if field == "top8ExpertIdsFile" {
                    // Both saved contracts specify int64 little-endian IDs.
                    // Decode their values explicitly instead of inferring a narrower ID width.
                    let decoded: [Int64] = original.withUnsafeBytes { bytes in
                        (0..<(59 * 8)).map { index in
                            Int64(littleEndian: bytes.loadUnaligned(fromByteOffset: index * 8, as: Int64.self))
                        }
                    }
                    guard decoded.allSatisfy({ (0..<256).contains($0) }),
                          decoded == rows.flatMap(\.ids).map({ Int64($0) }) else {
                        throw SmallDriftError.invalid("shortened prefix ID values differ from \(expectedMode): layer\(layer)")
                    }
                }
                guard original.prefix(data.count) == data else {
                    throw SmallDriftError.invalid("shortened prefix differs from \(expectedMode): layer\(layer) \(field)")
                }
            }
        }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        var arrays: [[String: Any]] = []
        for (stage, values) in observed.stages.sorted(by: { $0.key < $1.key }) {
            let raw = smallDriftFloatData(values)
            let filename = stage + ".fp32-le.bin"
            try raw.write(to: output.appendingPathComponent(filename), options: .withoutOverwriting)
            arrays.append(["stage": stage, "file": filename, "count": values.count, "byteCount": raw.count, "sha256": smallDriftSHA(raw)])
        }
        let elapsed = started.duration(to: .now)
        let receipt: [String: Any] = ["kind": "qwen-source-small-drift-prefix-candidate-v1", "diagnosticOnly": true,
            "scope": "source snapshot first six layers, full59tokenprefix, actual source operators, no head/decode",
            "savedRoutePrefixesBitExact": true, "expectedPrefixMode": expectedMode, "originalTensorShapesPreserved": true,
            "imageFeaturesSHA256": smallDriftSHA(featureBytes), "mropeSHA256": "b7716271562b33194c95284680887210665d7a0bca120a7e87b642c9853b5870",
            "baselineReceiptSHA256": smallDriftSHA(baselineRaw), "seconds": Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18,
            "prefixTokens": 59, "prefixLayers": 6, "arrays": arrays]
        try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("receipt.json"), options: .withoutOverwriting)
    }
}
