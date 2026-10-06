import CryptoKit
import Foundation
import Metal
import Testing
@testable import TurboFieldfare

private enum QwenPrefixStop: Error, Equatable {
    case afterLayer1Route
    case layer1ReadBeforeRoute
}

private enum QwenPrefixDiagnosticError: Error {
    case missingEnvironment(String)
    case wrongRegistration
    case outputMustBeEmpty
    case incompleteCapture(String)
}

private struct QwenPrefixActivationKey: Hashable {
    let layer: Int
    let stage: String
}

private struct QwenPrefixRoute: Sendable {
    let layer: Int
    let logits: [Float]
    let ids: [Int]
    let weights: [Float]
    let margin: Float
}

private final class QwenPrefixCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var activations: [QwenPrefixActivationKey: [Float]] = [:]
    private var routes: [Int: QwenPrefixRoute] = [:]
    private var duplicate = false
    private var protectedReadHook: String?

    func activation(position: Int, layer: Int, stage: String, values: [Float]) {
        guard position == 0, (-1...1).contains(layer) else { return }
        guard layer == 0 || !stage.hasPrefix("linear.") else { return }
        lock.lock()
        defer { lock.unlock() }
        let key = QwenPrefixActivationKey(layer: layer, stage: stage)
        if activations[key] != nil { duplicate = true }
        activations[key] = values
    }

    func route(position: Int, layer: Int, logits: [Float], ids: [Int],
               weights: [Float], margin: Float) {
        guard position == 0, (0...1).contains(layer) else { return }
        lock.lock()
        defer { lock.unlock() }
        if routes[layer] != nil { duplicate = true }
        routes[layer] = QwenPrefixRoute(
            layer: layer, logits: logits, ids: ids, weights: weights, margin: margin)
    }

    func hasRoute(position: Int, layer: Int) -> Bool {
        guard position == 0 else { return false }
        lock.lock()
        defer { lock.unlock() }
        return routes[layer] != nil
    }

    func recordProtectedReadHook(layer: Int, expert: Int,
                                 stream: QwenBF16ExpertReadHooks.Stream) {
        lock.lock()
        defer { lock.unlock() }
        let streamName: String
        switch stream {
        case .gateUp: streamName = "gateUp"
        case .down: streamName = "down"
        }
        protectedReadHook = "layer=\(layer),expert=\(expert),stream=\(streamName)"
    }

    func snapshot() -> (activations: [QwenPrefixActivationKey: [Float]],
                        routes: [Int: QwenPrefixRoute], duplicate: Bool,
                        protectedReadHook: String?) {
        lock.lock()
        defer { lock.unlock() }
        return (activations, routes, duplicate, protectedReadHook)
    }
}

private struct QwenPrefixBlobReceipt: Codable {
    let file: String
    let shape: [Int]
    let count: Int
    let bytes: Int
    let sha256: String
    let dtype: String?
}

private struct QwenPrefixCaptureReceipt: Codable {
    let stepIndex: Int
    let position: Int
    let layer: Int
    let stage: String
    let file: String
    let shape: [Int]
    let count: Int
    let bytes: Int
    let sha256: String
}

private struct QwenPrefixRouteReceipt: Codable {
    let layer: Int
    let position: Int
    let inputTokenId: Int32
    let routerLogits: QwenPrefixBlobReceipt
    let top8Weights: QwenPrefixBlobReceipt
    let top8ExpertIds: QwenPrefixBlobReceipt
    let savedReference: [String: Bool]
    let selectedVsUnselectedLogitMarginFP32: Float
}

private struct QwenPrefixDiagnosticReceipt: Codable {
    let kind: String
    let schemaVersion: Int
    let complete: Bool
    let diagnosticOnly: Bool
    let qualification: String
    let scope: String
    let intentionalStop: String
    let protectedReadHook: String?
    let runnerError: String?
    let inputTokenId: Int32
    let inputTokenIds: [Int32]
    let position: Int
    let positions: [Int32]
    let sequenceLength: Int
    let imageFeatureRowPresent: Bool
    let sourceDescriptorContentSHA256: String
    let sourceMarkerSHA256: String
    let sourceRoot: String
    let sourceShardSetSHA256: String
    let sourceChecksumManifestSHA256: String
    let registrationPath: String
    let referenceReceiptSHA256: String
    let referenceCase: String
    let captures: [QwenPrefixCaptureReceipt]
    let routes: [QwenPrefixRouteReceipt]
}

@Suite(.serialized) struct QwenSourceTextPrefixDiagnosticTests {
    private static let enableKey = "TURBO_P23_TEXT_PREFIX_DIAGNOSTIC"
    private static let registrationKey = "TURBO_P23_TEXT_PREFIX_REGISTRATION"
    private static let outputKey = "TURBO_P23_TEXT_PREFIX_OUTPUT"
    private static let tokenID: Int32 = 248_045
    private static let referenceReceiptSHA256 =
        "013a62fb64eb56cc81b5f8a4dd4655bbbda4ab01e1670bcf93254031dc5495ec"

    @Test(.enabled(if:
        ProcessInfo.processInfo.environment["TURBO_P23_TEXT_PREFIX_DIAGNOSTIC"] == "1"
            && !(ProcessInfo.processInfo.environment["TURBO_P23_TEXT_PREFIX_REGISTRATION"] ?? "").isEmpty
            && !(ProcessInfo.processInfo.environment["TURBO_P23_TEXT_PREFIX_OUTPUT"] ?? "").isEmpty,
        "set explicit Phase 23 text-prefix diagnostic environment after Main authorizes a run"))
    func captureOneTokenThroughLayer1RouterThenStop() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let registrationPath = environment[Self.registrationKey], !registrationPath.isEmpty else {
            throw QwenPrefixDiagnosticError.missingEnvironment(Self.registrationKey)
        }
        guard let outputPath = environment[Self.outputKey], !outputPath.isEmpty else {
            throw QwenPrefixDiagnosticError.missingEnvironment(Self.outputKey)
        }
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let expectedRegistration = root.appendingPathComponent(
            "scratch/qwen3.6-35b-a3b.gturbo", isDirectory: true).standardizedFileURL
        let registration = URL(fileURLWithPath: registrationPath, isDirectory: true)
            .standardizedFileURL
        guard registration.path == expectedRegistration.path,
              registration.path == registration.resolvingSymlinksInPath().path else {
            throw QwenPrefixDiagnosticError.wrongRegistration
        }
        let output = URL(fileURLWithPath: outputPath, isDirectory: true).standardizedFileURL
        guard output.path == output.resolvingSymlinksInPath().path,
              FileManager.default.fileExists(atPath: output.path),
              (try FileManager.default.contentsOfDirectory(atPath: output.path)).isEmpty else {
            throw QwenPrefixDiagnosticError.outputMustBeEmpty
        }

        let context = try MetalContext()
        let bundle = try ModelFamilyRuntime.loadBundle(
            directoryURL: registration, device: context.device,
            streamingMode: .pread(slotCount: 16), expertCachePolicy: .lfu,
            integrityPolicy: .sizeCheckTrustedReceipt)
        guard case .qwenOfficialSource(let model) = bundle.runtime,
              let identity = bundle.sourceIdentity,
              model.architecture.vocabularySize > Int(Self.tokenID),
              model.architecture.layers > 1 else {
            throw QwenPrefixDiagnosticError.wrongRegistration
        }
        let capture = QwenPrefixCapture()
        let hooks = QwenOfficialSourceTransactionHooks(
            beforeProtectedExpertRead: { layer, expert, stream in
                guard layer == 1 else { return }
                capture.recordProtectedReadHook(layer: layer, expert: expert, stream: stream)
                guard capture.hasRoute(position: 0, layer: 1) else {
                    throw QwenPrefixStop.layer1ReadBeforeRoute
                }
                throw QwenPrefixStop.afterLayer1Route
            },
            observeRoute: { position, layer, logits, ids, weights, margin in
                capture.route(position: position, layer: layer, logits: logits,
                              ids: ids, weights: weights, margin: margin)
            },
            observeActivation: { position, layer, stage, values in
                capture.activation(position: position, layer: layer,
                                   stage: stage, values: values)
            })
        let runner = try QwenOfficialSourceRunner(
            model: model, maxContext: 2, expertSlotCount: 16, hooks: hooks)

        var stop: QwenPrefixStop?
        var runnerError: String?
        do {
            _ = try await runner.produce(token: Self.tokenID, position: 0)
        } catch let diagnosticStop as QwenPrefixStop {
            stop = diagnosticStop
            runnerError = String(reflecting: diagnosticStop)
        } catch {
            runnerError = String(reflecting: error)
        }
        let captured = capture.snapshot()
        let captureRows = try qwenPrefixWriteActivations(captured.activations, to: output)
        let routeRows = try qwenPrefixWriteRoutes(captured.routes, to: output)
        let stopsAfterRoute = stop == .afterLayer1Route && captured.protectedReadHook != nil
        let sliceComplete = stopsAfterRoute && !captured.duplicate
            && qwenPrefixHasExpectedSlice(
                captured.activations, routes: captured.routes, architecture: model.architecture)
        let receipt = QwenPrefixDiagnosticReceipt(
            kind: "qwen36-text-prefix-trace-v1", schemaVersion: 1,
            complete: sliceComplete,
            diagnosticOnly: true, qualification: "none",
            scope: "layer0-complete-through-layer1-router; no-image-no-head-no-later-layers",
            intentionalStop: stopsAfterRoute
                ? "layer1 router callback captured; sentinel before protected expert read"
                : "incomplete diagnostic stop",
            protectedReadHook: captured.protectedReadHook,
            runnerError: runnerError,
            inputTokenId: Self.tokenID, inputTokenIds: [Self.tokenID],
            position: 0, positions: [0, 0, 0], sequenceLength: 1,
            imageFeatureRowPresent: false,
            sourceDescriptorContentSHA256: identity.descriptorContentSHA256,
            sourceMarkerSHA256: identity.markerSHA256,
            sourceRoot: identity.sourceRoot,
            sourceShardSetSHA256: identity.shardSetSHA256,
            sourceChecksumManifestSHA256: identity.checksumManifestSHA256,
            registrationPath: registration.path,
            referenceReceiptSHA256: Self.referenceReceiptSHA256,
            referenceCase: "natural-exif-1",
            captures: captureRows, routes: routeRows)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(receipt).write(
            to: output.appendingPathComponent("receipt.json"), options: .atomic)

        guard stopsAfterRoute, !captured.duplicate,
              qwenPrefixHasExpectedSlice(
                captured.activations, routes: captured.routes, architecture: model.architecture) else {
            throw QwenPrefixDiagnosticError.incompleteCapture(
                "stop=\(String(describing: stop)); runnerError=\(runnerError ?? "none"); "
                    + "protectedReadHook=\(captured.protectedReadHook ?? "none"); "
                    + "capturedLayers=\(Set(captured.activations.keys.map(\.layer)).sorted()); "
                    + "routeLayers=\(captured.routes.keys.sorted())")
        }
    }
}

private func qwenPrefixHasExpectedSlice(
    _ activations: [QwenPrefixActivationKey: [Float]], routes: [Int: QwenPrefixRoute],
    architecture: QwenTextArchitecture
) -> Bool {
    let layer0Stages: Set<String> = [
        "input-norm", "mixer", "residual-after-mixer", "post-norm", "residual-after-moe",
        "linear.input", "linear.qkv", "linear.z", "linear.b", "linear.a",
        "linear.convolved", "linear.query-raw", "linear.key-raw", "linear.value",
        "linear.beta", "linear.log-decay", "linear.core", "linear.gated", "linear.output",
    ]
    let layer1Stages: Set<String> = [
        "input-norm", "mixer", "residual-after-mixer", "post-norm",
    ]
    let actualLayer0 = Set(activations.keys.filter { $0.layer == 0 }.map(\.stage))
    let actualLayer1 = Set(activations.keys.filter { $0.layer == 1 }.map(\.stage))
    let hidden = architecture.hiddenSize
    let keyWidth = architecture.linearKeyHeads * architecture.linearKeyDimension
    let valueWidth = architecture.linearValueHeads * architecture.linearValueDimension
    let heads = architecture.linearValueHeads
    let expectedLinearCounts = [
        "linear.input": hidden, "linear.qkv": 2 * keyWidth + valueWidth,
        "linear.z": valueWidth, "linear.b": heads, "linear.a": heads,
        "linear.convolved": 2 * keyWidth + valueWidth,
        "linear.query-raw": architecture.linearValueHeads * architecture.linearKeyDimension,
        "linear.key-raw": architecture.linearValueHeads * architecture.linearKeyDimension,
        "linear.value": valueWidth, "linear.beta": heads, "linear.log-decay": heads,
        "linear.core": valueWidth, "linear.gated": valueWidth, "linear.output": hidden,
    ]
    let dimensionsValid = activations.allSatisfy { key, values in
        values.count == (key.layer == -1 ? hidden
            : key.layer == 0 ? (expectedLinearCounts[key.stage] ?? hidden) : hidden)
            && values.allSatisfy { $0.isFinite }
    }
    return dimensionsValid
        && activations[QwenPrefixActivationKey(layer: -1, stage: "embedding")]?.count == hidden
        && actualLayer0 == layer0Stages
        && actualLayer1 == layer1Stages
        && routes[0]?.logits.count == 256 && routes[1]?.logits.count == 256
        && routes[0]?.ids.count == 8 && routes[1]?.ids.count == 8
        && routes[0]?.weights.count == 8 && routes[1]?.weights.count == 8
        && routes.values.allSatisfy { route in
            route.logits.allSatisfy { $0.isFinite }
                && route.weights.allSatisfy { $0.isFinite }
                && route.margin.isFinite
        }
}

private func qwenPrefixWriteActivations(
    _ activations: [QwenPrefixActivationKey: [Float]], to output: URL
) throws -> [QwenPrefixCaptureReceipt] {
    var receipts: [QwenPrefixCaptureReceipt] = []
    for (key, values) in activations.sorted(by: {
        $0.key.layer == $1.key.layer
            ? $0.key.stage < $1.key.stage : $0.key.layer < $1.key.layer
    }) {
        let data = qwenPrefixFloatData(values)
        let layerName = "layer-\(key.layer)"
        let stageName = key.stage.replacingOccurrences(of: ".", with: "-")
        let file = "step-0.\(layerName).\(stageName).fp32-le.bin"
        try data.write(to: output.appendingPathComponent(file), options: .atomic)
        receipts.append(QwenPrefixCaptureReceipt(
            stepIndex: 0, position: 0, layer: key.layer, stage: key.stage,
            file: file, shape: [1, values.count], count: values.count, bytes: data.count,
            sha256: qwenPrefixSHA256(data)))
    }
    return receipts
}

private func qwenPrefixWriteRoutes(
    _ routes: [Int: QwenPrefixRoute], to output: URL
) throws -> [QwenPrefixRouteReceipt] {
    try routes.values.sorted { $0.layer < $1.layer }.map { route in
        let prefix = "step-0.layer-\(route.layer)"
        let logits = qwenPrefixFloatData(route.logits)
        let weights = qwenPrefixFloatData(route.weights)
        let ids = qwenPrefixIntegerData(route.ids.map(Int64.init))
        let logitsName = "\(prefix).router-fp32-le.bin"
        let weightsName = "\(prefix).weights-fp32-le.bin"
        let idsName = "\(prefix).expert-ids-int64-le.bin"
        try logits.write(to: output.appendingPathComponent(logitsName), options: .atomic)
        try weights.write(to: output.appendingPathComponent(weightsName), options: .atomic)
        try ids.write(to: output.appendingPathComponent(idsName), options: .atomic)
        return QwenPrefixRouteReceipt(
            layer: route.layer, position: 0, inputTokenId: 248_045,
            routerLogits: QwenPrefixBlobReceipt(
                file: logitsName, shape: [1, route.logits.count], count: route.logits.count,
                bytes: logits.count, sha256: qwenPrefixSHA256(logits), dtype: "float32"),
            top8Weights: QwenPrefixBlobReceipt(
                file: weightsName, shape: [1, route.weights.count], count: route.weights.count,
                bytes: weights.count, sha256: qwenPrefixSHA256(weights), dtype: "float32"),
            top8ExpertIds: QwenPrefixBlobReceipt(
                file: idsName, shape: [1, route.ids.count], count: route.ids.count,
                bytes: ids.count, sha256: qwenPrefixSHA256(ids), dtype: "int64"),
            savedReference: [:],
            selectedVsUnselectedLogitMarginFP32: route.margin)
    }
}

private func qwenPrefixFloatData(_ values: [Float]) -> Data {
    qwenPrefixIntegerData(values.map(\.bitPattern))
}

private func qwenPrefixIntegerData<T: FixedWidthInteger>(_ values: [T]) -> Data {
    var result = Data(capacity: values.count * MemoryLayout<T>.size)
    for value in values {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { result.append(contentsOf: $0) }
    }
    return result
}

private func qwenPrefixSHA256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
