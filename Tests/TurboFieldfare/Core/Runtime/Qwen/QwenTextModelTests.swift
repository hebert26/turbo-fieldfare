import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// Shared fixture access for the Qwen model and runner suites. The JSON is the
/// reviewed Stage-A artifact; tests never regenerate, quantize, or substitute
/// its virtual files.
enum QwenTextFixtureSupport {
    static let aggregateSHA256 =
        "d4edeb3c98db5c084b6cbf9cf38b56859c717c6d0c20c522f5e1382ed39db7c6"
    static let fp32AbsoluteTolerance: Float = 1e-5
    static let fp32RelativeTolerance: Float = 1e-5
    static let stateAbsoluteTolerance: Float = 2e-5
    static let stateRelativeTolerance: Float = 2e-5

    static func data() throws -> Data {
        guard let url = Bundle.module.url(
            forResource: "qwen36-tiny-text-model-fixtures", withExtension: "json") else {
            throw FixtureSupportError.missingResource
        }
        return try Data(contentsOf: url)
    }

    static func object() throws -> [String: Any] {
        guard let object = try JSONSerialization.jsonObject(with: data()) as? [String: Any]
        else { throw FixtureSupportError.invalidJSON }
        return object
    }

    static func records() throws -> QwenTextFixtureRecords {
        try QwenTextFixtureRecords.decode(jsonData: data())
    }

    static func model(device: MTLDevice) throws -> QwenTextModel {
        try QwenTextModel.loadFixtureRecords(try records(), device: device)
    }

    static func mutate(_ mutation: (inout [String: Any]) throws -> Void) throws -> Data {
        var root = try object()
        try mutation(&root)
        return try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
    }

    static func value(_ root: Any, at path: [String]) throws -> Any {
        var current = root
        for component in path {
            guard let dictionary = current as? [String: Any],
                  let next = dictionary[component] else {
                throw FixtureSupportError.missingPath(path.joined(separator: "."))
            }
            current = next
        }
        return current
    }

    static func tensorValues(_ root: Any, at path: [String]) throws -> [Float] {
        try tensorValues(try value(root, at: path), label: path.joined(separator: "."))
    }

    static func tensorValues(_ object: Any, label: String = "tensor") throws -> [Float] {
        guard let dictionary = object as? [String: Any],
              let values = dictionary["values"] as? [NSNumber] else {
            throw FixtureSupportError.invalidTensor(label)
        }
        return values.map(\.floatValue)
    }

    static func arrayElement(_ root: Any, at path: [String], index: Int) throws -> Any {
        let object = try value(root, at: path)
        guard let array = object as? [Any], array.indices.contains(index) else {
            throw FixtureSupportError.missingPath(path.joined(separator: ".") + "[\\(index)]")
        }
        return array[index]
    }

    static func arrayTensorValues(
        _ root: Any, at path: [String], index: Int, key: String
    ) throws -> [Float] {
        let element = try arrayElement(root, at: path, index: index)
        return try tensorValues(try value(element, at: [key]), label: key)
    }

    static func nestedDictionary(
        _ root: inout [String: Any], at path: [String]
    ) throws -> [String: Any] {
        guard let first = path.first else { return root }
        guard var child = root[first] as? [String: Any] else {
            throw FixtureSupportError.missingPath(path.joined(separator: "."))
        }
        if path.count > 1 {
            child = try nestedDictionary(&child, at: Array(path.dropFirst()))
        }
        root[first] = child
        return child
    }

    static func withResidentRegions(
        _ root: inout [String: Any],
        mutate: (inout [[String: Any]]) throws -> Void
    ) throws {
        guard var records = root["fixtureRecords"] as? [String: Any],
              var regions = records["residentTensorRegions"] as? [[String: Any]] else {
            throw FixtureSupportError.missingPath("fixtureRecords.residentTensorRegions")
        }
        try mutate(&regions)
        records["residentTensorRegions"] = regions
        root["fixtureRecords"] = records
    }

    static func withVirtualFile(
        _ root: inout [String: Any],
        path: String,
        mutate: (inout [String: Any]) throws -> Void
    ) throws {
        guard var packing = root["packing"] as? [String: Any],
              var files = packing["virtualFiles"] as? [String: Any],
              var file = files[path] as? [String: Any] else {
            throw FixtureSupportError.missingPath("packing.virtualFiles.\(path)")
        }
        try mutate(&file)
        files[path] = file
        packing["virtualFiles"] = files
        root["packing"] = packing
    }

    static func fixtureLayoutConfiguration(
        _ records: QwenTextFixtureRecords
    ) throws -> QwenPackedLayoutConfiguration {
        try QwenPackedLayoutConfiguration(
            layerCount: records.architecture.layers,
            expertCount: records.architecture.experts,
            hiddenSize: records.architecture.hiddenSize,
            intermediateSize: records.architecture.routedIntermediateSize,
            expertStride: records.expertStride)
    }

    static func fixtureLayoutFileSizes(
        _ records: QwenTextFixtureRecords
    ) -> [String: UInt64] {
        records.virtualFiles.reduce(into: [:]) { result, entry in
            if entry.key.hasPrefix("packed_experts/layer_") {
                result[entry.key] = UInt64(entry.value.bytes.count)
            }
        }
    }

    static func mutatedExpertLayout(
        _ mutation: (inout [String: Any]) throws -> Void
    ) throws -> Data {
        guard var root = try JSONSerialization.jsonObject(
            with: try records().expertLayoutData) as? [String: Any],
              var layers = root["layers"] as? [[String: Any]],
              !layers.isEmpty,
              var sources = layers[0]["sources"] as? [[String: Any]],
              !sources.isEmpty else {
            throw FixtureSupportError.invalidJSON
        }
        var source = sources[0]
        guard !source.isEmpty else {
            throw FixtureSupportError.invalidJSON
        }
        try mutation(&source)
        sources[0] = source
        layers[0]["sources"] = sources
        root["layers"] = layers
        return try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
    }

    static func expectLayoutIndexCorrupt(
        _ data: Data, sourceSchema: QwenPackedSourceSchema
    ) throws -> Bool {
        let records = try records()
        let configuration = try fixtureLayoutConfiguration(records)
        do {
            _ = try PackedExpertsLayoutReader.decodeQwenV2(
                data: data,
                configuration: configuration,
                manifestFileSizes: fixtureLayoutFileSizes(records),
                sourceSchema: sourceSchema)
            return false
        } catch let error as ModelError {
            guard case .indexCorrupt = error else { return false }
            return true
        }
    }

    static func expectIndexCorrupt(_ data: Data) -> Bool {
        do {
            let records = try QwenTextFixtureRecords.decode(jsonData: data)
            let context = try MetalContext()
            _ = try QwenTextModel.loadFixtureRecords(records, device: context.device)
            return false
        } catch {
            if case ModelError.indexCorrupt = error { return true }
            return false
        }
    }

    enum FixtureSupportError: Error {
        case missingResource
        case invalidJSON
        case missingPath(String)
        case invalidTensor(String)
    }
}

@Suite(.serialized) struct QwenTextModelTests {
    @Test func reviewedFixtureHasFrozenIdentityAndPackedFiles() throws {
        let records = try QwenTextFixtureSupport.records()
        #expect(records.virtualFileAggregateSHA256 == QwenTextFixtureSupport.aggregateSHA256)
        #expect(records.expertStride == 16_384)
        #expect(records.virtualFiles["model_weights.bin"]?.byteCount == 1_048_576)
        #expect(records.virtualFiles["packed_experts/layout.json"] != nil)
        #expect(records.residentTensorRegions.count == 64)
        #expect(records.expertTensorRegions.count == 40)
    }

    @Test func exactFixtureLayoutBytesUseBoundedSchemaAndSharedMapping() throws {
        let records = try QwenTextFixtureSupport.records()
        let configuration = try QwenTextFixtureSupport.fixtureLayoutConfiguration(records)
        let layout = try PackedExpertsLayoutReader.decodeQwenV2(
            data: records.expertLayoutData,
            configuration: configuration,
            manifestFileSizes: QwenTextFixtureSupport.fixtureLayoutFileSizes(records),
            sourceSchema: .fixtureRedundantEnd)
        #expect(layout.layers.count == records.architecture.layers)
        #expect(layout.layers.allSatisfy { $0.experts.count == records.architecture.experts })
        #expect(try QwenTextFixtureSupport.expectLayoutIndexCorrupt(
            records.expertLayoutData, sourceSchema: .production))

        let context = try MetalContext()
        let model = try QwenTextModel.loadFixtureRecords(records, device: context.device)
        #expect(model.mappingIdentity == QwenTextFixtureSupport.aggregateSHA256)
        #expect(model.tensorNames.count == 64)
    }

    @Test func fixtureLayoutRedundantEndAndUnknownFieldsRemainStrict() throws {
        let wrongTotal = try QwenTextFixtureSupport.mutatedExpertLayout { source in
            source["totalSize"] = NSNumber(value: 0)
        }
        #expect(try QwenTextFixtureSupport.expectLayoutIndexCorrupt(
            wrongTotal, sourceSchema: .fixtureRedundantEnd))

        let missingTotal = try QwenTextFixtureSupport.mutatedExpertLayout { source in
            source.removeValue(forKey: "totalSize")
        }
        #expect(try QwenTextFixtureSupport.expectLayoutIndexCorrupt(
            missingTotal, sourceSchema: .fixtureRedundantEnd))

        let malformedTotal = try QwenTextFixtureSupport.mutatedExpertLayout { source in
            source["totalSize"] = "not-a-size"
        }
        #expect(try QwenTextFixtureSupport.expectLayoutIndexCorrupt(
            malformedTotal, sourceSchema: .fixtureRedundantEnd))

        let overflowTotal = try QwenTextFixtureSupport.mutatedExpertLayout { source in
            source["biasesOffset"] = NSNumber(value: UInt64.max - 1)
            source["biasesSize"] = NSNumber(value: 2)
            source["totalSize"] = NSNumber(value: UInt64.max)
        }
        #expect(try QwenTextFixtureSupport.expectLayoutIndexCorrupt(
            overflowTotal, sourceSchema: .fixtureRedundantEnd))

        let unknownField = try QwenTextFixtureSupport.mutatedExpertLayout { source in
            source["unexpected"] = true
        }
        #expect(try QwenTextFixtureSupport.expectLayoutIndexCorrupt(
            unknownField, sourceSchema: .fixtureRedundantEnd))
        #expect(try QwenTextFixtureSupport.expectLayoutIndexCorrupt(
            unknownField, sourceSchema: .production))
    }

    @Test func mapsUntiedHeadSeparatelyFromEmbeddingAndKeepsNorms() throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let embedding = try model.embedding
        let head = try model.outputHead
        let finalNorm = try model.finalNorm

        #expect(embedding.name == "model.language_model.embed_tokens.weight")
        #expect(head.name == "lm_head.weight")
        #expect(embedding.name != head.name)
        #expect(embedding.shape == [19, 32])
        #expect(head.shape == embedding.shape)
        #expect(embedding.storage == .affineInt8)
        #expect(head.storage == .affineInt8)
        #expect(try embedding.decodedFloat32() != head.decodedFloat32())
        #expect(finalNorm.shape == [32])
        #expect(finalNorm.storage == .bf16)
    }

    @Test func mapsEveryLayerScheduleAndRoutedExpertBytes() throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        #expect(model.tensorNames.count == 64)
        #expect(model.mappingIdentity == QwenTextFixtureSupport.aggregateSHA256)

        for layer in 0..<4 {
            #expect(try model.inputNorm(layer: layer).shape == [32])
            #expect(try model.postAttentionNorm(layer: layer).shape == [32])
            if layer == 3 {
                #expect(try model.attentionTensor(layer: layer, suffix: "q_proj.weight").storage == .affineInt4)
                #expect(try model.attentionTensor(layer: layer, suffix: "o_proj.weight").storage == .affineInt4)
            } else {
                #expect(try model.linearTensor(layer: layer, suffix: "dt_bias").storage == .affineInt8)
                #expect(try model.linearTensor(layer: layer, suffix: "A_log").storage == .affineInt8)
                #expect(try model.linearTensor(layer: layer, suffix: "in_proj_qkv.weight").storage == .affineInt8)
            }
            #expect(try model.router(layer: layer).name.contains("mlp.gate.weight"))
            #expect(try model.sharedExpert(layer: layer, role: "gate").storage == .affineInt4)
        }

        let routed = try model.routedExpert(layer: 0, expert: 0, role: "gate_up")
        #expect(routed.storage == .affineInt4)
        #expect(routed.shape == [14, 32])
        #expect(routed.byteCount == 280)
        #expect(try routed.decodedFloat32().count == 448)
    }

    @Test func decodesSerializedBF16AndAffineValuesWithoutFloatInjection() throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let norm = try model.inputNorm(layer: 0)
        let affine = try model.linearTensor(layer: 0, suffix: "in_proj_qkv.weight")
        let timeBias = try model.linearTensor(layer: 0, suffix: "dt_bias")
        let normValues = try norm.decodedFloat32()
        let affineValues = try affine.decodedFloat32()
        let timeBiasValues = try timeBias.decodedFloat32()

        #expect(normValues.count == 32)
        #expect(affineValues.count == 3 * 2 * 4 * 32)
        #expect(timeBiasValues.count == 2)
        #expect(normValues.allSatisfy { $0.isFinite })
        #expect(affineValues.allSatisfy { $0.isFinite })
        #expect(norm.byteCount == 64)
        #expect(affine.storage == .affineInt8)
        #expect(timeBias.storage == .affineInt8)
    }

    @Test func rejectsMissingResidentRegionBeforeRunnerCreation() throws {
        let data = try QwenTextFixtureSupport.mutate { root in
            try QwenTextFixtureSupport.withResidentRegions(&root) { regions in
                regions.removeFirst()
            }
        }
        #expect(QwenTextFixtureSupport.expectIndexCorrupt(data))
    }

    @Test func rejectsDuplicateResidentRegionBeforeRunnerCreation() throws {
        let data = try QwenTextFixtureSupport.mutate { root in
            try QwenTextFixtureSupport.withResidentRegions(&root) { regions in
                regions.append(try #require(regions.first))
            }
        }
        #expect(QwenTextFixtureSupport.expectIndexCorrupt(data))
    }

    @Test func rejectsWrongShapeStorageAndOverlappingRecords() throws {
        let wrongShape = try QwenTextFixtureSupport.mutate { root in
            try QwenTextFixtureSupport.withResidentRegions(&root) { regions in
                regions[0]["shape"] = [0, 0]
            }
        }
        #expect(QwenTextFixtureSupport.expectIndexCorrupt(wrongShape))

        let wrongStorage = try QwenTextFixtureSupport.mutate { root in
            try QwenTextFixtureSupport.withResidentRegions(&root) { regions in
                regions[0]["storage"] = "bf16"
            }
        }
        #expect(QwenTextFixtureSupport.expectIndexCorrupt(wrongStorage))

        let overlap = try QwenTextFixtureSupport.mutate { root in
            try QwenTextFixtureSupport.withResidentRegions(&root) { regions in
                regions[1]["offset"] = regions[0]["offset"]
            }
        }
        #expect(QwenTextFixtureSupport.expectIndexCorrupt(overlap))
    }

    @Test func rejectsVirtualFileHashAndAggregateMutations() throws {
        let badHash = try QwenTextFixtureSupport.mutate { root in
            try QwenTextFixtureSupport.withVirtualFile(&root, path: "model_weights.bin") { file in
                file["sha256"] = String(repeating: "0", count: 64)
            }
        }
        #expect(QwenTextFixtureSupport.expectIndexCorrupt(badHash))

        let badAggregate = try QwenTextFixtureSupport.mutate { root in
            var packing = try #require(root["packing"] as? [String: Any])
            packing["virtualFileAggregateSHA256"] = String(repeating: "0", count: 64)
            root["packing"] = packing
        }
        #expect(QwenTextFixtureSupport.expectIndexCorrupt(badAggregate))
    }

    @Test func rejectsWrongSyntheticArchitectureBeforeMapping() throws {
        let data = try QwenTextFixtureSupport.mutate { root in
            var architecture = try #require(root["tinyArchitecture"] as? [String: Any])
            architecture["hiddenActivation"] = "gelu"
            root["tinyArchitecture"] = architecture
        }
        #expect(QwenTextFixtureSupport.expectIndexCorrupt(data))
    }
}
