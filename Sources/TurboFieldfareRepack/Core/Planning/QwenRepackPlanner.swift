import Foundation
import TurboFieldfareFormat

struct QwenPlanningProvenance: Sendable, Equatable {
    let repository: String
    let revision: String
    let validatedSidecarSHA256: [String: String]
    let observedIndexSHA256: String
    let observedConfigSHA256: String
    let quantizationPolicySHA256: String
}

struct QwenAffineComponentLayout: Sendable, Equatable {
    let valuesOffset: UInt64
    let valuesSize: UInt64
    let scalesOffset: UInt64
    let scalesSize: UInt64
    let biasesOffset: UInt64
    let biasesSize: UInt64

    var totalSize: UInt64 { biasesOffset + biasesSize }
}

struct QwenPlannedTensor: Sendable, Equatable {
    let source: SourceTensor
    let relativeFile: String
    let executionPosition: Int
    let fileOffset: UInt64
    let regionSize: UInt64
    let storage: BF16AffineTensorStorage
    let quantizationCategory: GTurboQuantizationCategoryV2?
    let affineComponents: QwenAffineComponentLayout?
}

struct QwenPlannedArtifact: Sendable, Equatable {
    let relativePath: String
    let size: UInt64
}

struct QwenRoutedSourcePlan: Sendable, Equatable {
    let role: String
    let source: SourceTensor
    let perExpertShape: [UInt64]
    let components: QwenAffineComponentLayout
}

struct QwenExpertLayerPlan: Sendable, Equatable {
    let layerIndex: Int
    let relativePath: String
    let expertsPerLayer: Int
    let expertStride: UInt64
    let sources: [QwenRoutedSourcePlan]
    let fileSize: UInt64
}

struct QwenVisionCompanionPlan: Sendable, Equatable {
    let artifact: QwenPlannedArtifact
    let tensors: [QwenPlannedTensor]
}

struct QwenScratchBudget: Sendable, Equatable {
    /// Maximum element-payload buffers only. This excludes allocator metadata,
    /// process RSS, recurrent runtime state, output files, and disk reserve.
    let transformTileBytes: Int
    let maximumGroupBytes: Int
    let quantizerPayloadBytes: Int
    let maximumPayloadBufferBytes: Int
}

struct QwenRepackPlan: Sendable {
    let architecture: ArchInfo.QwenPlanningArchitecture
    let provenance: QwenPlanningProvenance
    let quantizationGroups: [GTurboQuantizationGroupV2]
    let runtimeRecurrentStateStorage: GTurboStorageTypeV2
    let textTensors: [QwenPlannedTensor]
    let expertLayers: [QwenExpertLayerPlan]
    /// All 333 names remain assigned here even when no companion is requested.
    let visionTensorNames: [String]
    let visionCompanion: QwenVisionCompanionPlan?
    let omittedMTPNames: [String]
    let artifacts: [QwenPlannedArtifact]
    let requiredArtifactBytes: UInt64
    let scratch: QwenScratchBudget
    let canonicalFingerprint: String
}

/// A no-payload test result. It cannot enter production family dispatch or any
/// writer because it deliberately carries no verified-input token.
struct QwenMetadataFixturePlan: Sendable {
    let textTensors: [QwenPlannedTensor]
    let expertLayers: [QwenExpertLayerPlan]
    let visionTensorNames: [String]
    let visionCompanion: QwenVisionCompanionPlan?
    let omittedMTPNames: [String]
    let artifacts: [QwenPlannedArtifact]
    let requiredArtifactBytes: UInt64
    let scratch: QwenScratchBudget
    let canonicalFingerprint: String
}

enum QwenRepackPlanner {
    static let transformTileBytes = BF16AffineTransformReader.maximumTileBytes

    private struct VerifiedInput: Sendable {
        let snapshot: LocalPinnedSnapshot
        let architecture: ArchInfo.QwenPlanningArchitecture
        let provenance: QwenPlanningProvenance
    }

    private struct LayoutResult: Sendable {
        let textTensors: [QwenPlannedTensor]
        let expertLayers: [QwenExpertLayerPlan]
        let visionTensorNames: [String]
        let visionCompanion: QwenVisionCompanionPlan?
        let omittedMTPNames: [String]
        let artifacts: [QwenPlannedArtifact]
        let requiredArtifactBytes: UInt64
        let scratch: QwenScratchBudget
        let canonicalFingerprint: String
    }

    /// Production entry. All filesystem identity and architecture validation
    /// completes before layout is calculated and before any writer can open.
    static func plan(
        snapshotDirectory: String,
        outputDirectory: String,
        includeVision: Bool = true
    ) throws -> QwenRepackPlan {
        let input = try prepareVerifiedInput(snapshotDirectory: snapshotDirectory)
        let layout = try makeLayout(
            architecture: input.architecture,
            provenance: input.provenance,
            tensors: input.snapshot.tensors,
            outputDirectory: outputDirectory,
            includeVision: includeVision,
            requiresVerifiedProvenance: true)
        return QwenRepackPlan(
            architecture: input.architecture,
            provenance: input.provenance,
            quantizationGroups: BF16AffineQuantizationPolicy.manifestQuantizationGroups,
            runtimeRecurrentStateStorage: .fp32,
            textTensors: layout.textTensors,
            expertLayers: layout.expertLayers,
            visionTensorNames: layout.visionTensorNames,
            visionCompanion: layout.visionCompanion,
            omittedMTPNames: layout.omittedMTPNames,
            artifacts: layout.artifacts,
            requiredArtifactBytes: layout.requiredArtifactBytes,
            scratch: layout.scratch,
            canonicalFingerprint: layout.canonicalFingerprint)
    }

    /// Internal metadata-only seam for official-sized synthetic headers. It
    /// bypasses filesystem authentication, not architecture, policy, tensor,
    /// shape, range, accounting, or overflow validation.
    static func planMetadataFixture(
        architecture: ArchInfo.QwenPlanningArchitecture,
        tensors: [SourceTensor],
        outputDirectory: String,
        includeVision: Bool = true
    ) throws -> QwenMetadataFixturePlan {
        let provenance = QwenPlanningProvenance(
            repository: ModelSourceCatalog.qwen.repository,
            revision: ModelSourceCatalog.qwen.revision,
            validatedSidecarSHA256: ModelSourceCatalog.qwen.sidecarSHA256,
            observedIndexSHA256: GTurboFormatV2.qwenSourceIndexSHA256,
            observedConfigSHA256: architecture.observedConfigSHA256,
            quantizationPolicySHA256: BF16AffineQuantizationPolicy.policySHA256)
        let result = try makeLayout(
            architecture: architecture,
            provenance: provenance,
            tensors: tensors,
            outputDirectory: outputDirectory,
            includeVision: includeVision,
            requiresVerifiedProvenance: false)
        return QwenMetadataFixturePlan(
            textTensors: result.textTensors,
            expertLayers: result.expertLayers,
            visionTensorNames: result.visionTensorNames,
            visionCompanion: result.visionCompanion,
            omittedMTPNames: result.omittedMTPNames,
            artifacts: result.artifacts,
            requiredArtifactBytes: result.requiredArtifactBytes,
            scratch: result.scratch,
            canonicalFingerprint: result.canonicalFingerprint)
    }

    /// Focused internal rejection seam for provenance mutation tests. Passing
    /// it does not create a verified planning input.
    static func validateProvenance(_ provenance: QwenPlanningProvenance) throws {
        guard provenance.repository == ModelSourceCatalog.qwen.repository,
              provenance.revision == ModelSourceCatalog.qwen.revision,
              provenance.validatedSidecarSHA256 == ModelSourceCatalog.qwen.sidecarSHA256,
              provenance.observedIndexSHA256 == GTurboFormatV2.qwenSourceIndexSHA256,
              provenance.observedConfigSHA256
                == GTurboFormatV2.qwenSidecarSHA256["config.json"],
              provenance.quantizationPolicySHA256
                == BF16AffineQuantizationPolicy.policySHA256 else {
            throw RepackError.configurationInvalid(
                detail: "Qwen planning provenance is not the observed pinned source")
        }
    }

    /// Single arithmetic seam shared by resident and expert planning.
    /// Groups restart at every logical row; `[2, 65]` therefore emits group
    /// element counts `[64, 1, 64, 1]`.
    static func affineComponentSizes(
        shape: [UInt64],
        bitWidth: AffineBitWidth
    ) throws -> QwenAffineComponentLayout {
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }),
              let last = shape.last else {
            throw RepackError.configurationInvalid(detail: "affine shape must be nonempty")
        }
        let rows = try product(shape.dropLast())
        let groupSize = UInt64(BF16AffineQuantizationPolicy.affineGroupSize)
        let groupsPerRow = try ceilingDivide(last, by: groupSize)
        let completeGroups = last / groupSize
        let remainder = last % groupSize
        let bytesPerCompleteGroup = try ceilingDivide(
            try multiply(groupSize, UInt64(bitWidth.rawValue)), by: 8)
        var valuesPerRow = try multiply(completeGroups, bytesPerCompleteGroup)
        if remainder > 0 {
            valuesPerRow = try add(valuesPerRow, try ceilingDivide(
                try multiply(remainder, UInt64(bitWidth.rawValue)), by: 8))
        }
        let valuesSize = try multiply(rows, valuesPerRow)
        let groupCount = try multiply(rows, groupsPerRow)
        let scalesSize = try multiply(groupCount, 2)
        let biasesSize = scalesSize
        let scalesOffset = valuesSize
        let biasesOffset = try add(scalesOffset, scalesSize)
        _ = try add(biasesOffset, biasesSize)
        return QwenAffineComponentLayout(
            valuesOffset: 0,
            valuesSize: valuesSize,
            scalesOffset: scalesOffset,
            scalesSize: scalesSize,
            biasesOffset: biasesOffset,
            biasesSize: biasesSize)
    }

    private static func prepareVerifiedInput(
        snapshotDirectory: String
    ) throws -> VerifiedInput {
        let expectedIdentity = LocalPinnedSnapshotIdentity(source: ModelSourceCatalog.qwen)
        let snapshot = try LocalPinnedSnapshotLoader.load(
            snapshotDirectory: snapshotDirectory,
            expectedIdentity: expectedIdentity)
        let configPath = (snapshotDirectory as NSString).appendingPathComponent("config.json")
        let arch = try ArchInfo.load(configPath: configPath)
        guard let qwen = arch.qwenPlanningArchitecture else {
            throw RepackError.configurationInvalid(detail: "official Qwen config decoded as another family")
        }
        let provenance = QwenPlanningProvenance(
            repository: expectedIdentity.repository,
            revision: expectedIdentity.revision,
            validatedSidecarSHA256: expectedIdentity.sidecarSHA256,
            observedIndexSHA256: snapshot.indexSHA256,
            observedConfigSHA256: qwen.observedConfigSHA256,
            quantizationPolicySHA256: BF16AffineQuantizationPolicy.policySHA256)
        try validateProvenance(provenance)
        let basenames = Set(snapshot.tensors.map { ($0.shardPath as NSString).lastPathComponent })
        guard snapshot.shardFilenames.count == 26,
              basenames == Set(snapshot.shardFilenames),
              snapshot.shardFilenames.allSatisfy(isSafeShardBasename) else {
            throw RepackError.configurationInvalid(detail: "Qwen shard provenance is incomplete")
        }
        return VerifiedInput(snapshot: snapshot, architecture: qwen, provenance: provenance)
    }

    private static func makeLayout(
        architecture: ArchInfo.QwenPlanningArchitecture,
        provenance: QwenPlanningProvenance,
        tensors: [SourceTensor],
        outputDirectory: String,
        includeVision: Bool,
        requiresVerifiedProvenance: Bool
    ) throws -> LayoutResult {
        try validateExactArchitecture(
            architecture,
            requiresPinnedConfigDigest: requiresVerifiedProvenance)
        if requiresVerifiedProvenance { try validateProvenance(provenance) }
        guard !outputDirectory.isEmpty else {
            throw RepackError.configurationInvalid(detail: "Qwen output directory is empty")
        }
        let descriptors = tensors.map {
            QwenOfficialTensorDescriptor(name: $0.name, dataType: $0.dtype)
        }
        let categories = try QwenOfficialTensorMap.classify(descriptors)
        let decisions = try BF16AffineQuantizationPolicy.decisions(for: descriptors)
        guard tensors.count == 1_045, categories.count == tensors.count,
              decisions.count == tensors.count else {
            throw RepackError.configurationInvalid(detail: "Qwen tensor policy is incomplete")
        }

        var records: [(SourceTensor, QwenOfficialTensorCategory, BF16AffinePolicyDecision)] = []
        records.reserveCapacity(tensors.count)
        for index in tensors.indices {
            let tensor = tensors[index]
            try validateSourceTensor(tensor, architecture: architecture)
            records.append((tensor, categories[index], decisions[index]))
        }
        let textRecords = records.filter { $0.1 == .textResident }.sorted {
            executionKey($0.0.name) < executionKey($1.0.name)
        }
        let routedRecords = records.filter { $0.1 == .routedExpert }.sorted {
            executionKey($0.0.name) < executionKey($1.0.name)
        }
        let visionRecords = records.filter { $0.1 == .vision }.sorted {
            visionExecutionKey($0.0.name) < visionExecutionKey($1.0.name)
        }
        let omittedRecords = records.filter { $0.1 == .mtpOmitted }.sorted {
            $0.0.name < $1.0.name
        }
        guard textRecords.count == 613, routedRecords.count == 80,
              visionRecords.count == 333, omittedRecords.count == 19 else {
            throw RepackError.configurationInvalid(detail: "Qwen tensor accounting counts disagree")
        }

        var residentCursor: UInt64 = 0
        var textPlans: [QwenPlannedTensor] = []
        textPlans.reserveCapacity(textRecords.count)
        for (position, record) in textRecords.enumerated() {
            residentCursor = try alignUp(residentCursor)
            let plan = try plannedTensor(
                record: record,
                relativeFile: "model_weights.bin",
                executionPosition: position,
                fileOffset: residentCursor)
            textPlans.append(plan)
            residentCursor = try add(residentCursor, plan.regionSize)
        }
        let residentSize = try alignUp(residentCursor)

        let routedByLayer = Dictionary(grouping: routedRecords) {
            layerIndex(in: $0.0.name) ?? -1
        }
        var expertLayers: [QwenExpertLayerPlan] = []
        expertLayers.reserveCapacity(40)
        for layer in 0..<40 {
            guard let layerRecords = routedByLayer[layer], layerRecords.count == 2 else {
                throw RepackError.configurationInvalid(
                    detail: "Qwen layer \(layer) routed source bundle is incomplete")
            }
            let ordered = layerRecords.sorted {
                routedRole($0.0.name) < routedRole($1.0.name)
            }
            var cursor: UInt64 = 0
            var sources: [QwenRoutedSourcePlan] = []
            for record in ordered {
                let source = record.0
                guard source.shape.first == UInt64(architecture.text.numberOfExperts) else {
                    throw RepackError.shapeMismatch(
                        name: source.name, detail: "expected 256 experts")
                }
                let perExpertShape = Array(source.shape.dropFirst())
                let bitWidth = try requiredBitWidth(record.2, name: source.name)
                var components = try affineComponentSizes(
                    shape: perExpertShape, bitWidth: bitWidth)
                components = QwenAffineComponentLayout(
                    valuesOffset: try add(cursor, components.valuesOffset),
                    valuesSize: components.valuesSize,
                    scalesOffset: try add(cursor, components.scalesOffset),
                    scalesSize: components.scalesSize,
                    biasesOffset: try add(cursor, components.biasesOffset),
                    biasesSize: components.biasesSize)
                cursor = components.totalSize
                sources.append(QwenRoutedSourcePlan(
                    role: routedRole(source.name),
                    source: source,
                    perExpertShape: perExpertShape,
                    components: components))
            }
            let stride = try alignUp(cursor)
            let fileSize = try multiply(
                UInt64(architecture.text.numberOfExperts), stride)
            expertLayers.append(QwenExpertLayerPlan(
                layerIndex: layer,
                relativePath: "packed_experts/layer_\(String(format: "%02d", layer)).bin",
                expertsPerLayer: architecture.text.numberOfExperts,
                expertStride: stride,
                sources: sources,
                fileSize: fileSize))
        }

        let visionNames = visionRecords.map { $0.0.name }
        let visionPlan: QwenVisionCompanionPlan?
        if includeVision {
            var cursor: UInt64 = 0
            var plans: [QwenPlannedTensor] = []
            plans.reserveCapacity(visionRecords.count)
            for (position, record) in visionRecords.enumerated() {
                cursor = try alignUp(cursor)
                let plan = try plannedTensor(
                    record: record,
                    relativeFile: "vision_weights.bin",
                    executionPosition: position,
                    fileOffset: cursor)
                plans.append(plan)
                cursor = try add(cursor, plan.regionSize)
            }
            let artifact = QwenPlannedArtifact(
                relativePath: "vision_weights.bin", size: try alignUp(cursor))
            visionPlan = QwenVisionCompanionPlan(artifact: artifact, tensors: plans)
        } else {
            visionPlan = nil
        }

        let layoutBytes = try expertLayoutMetadataSize(expertLayers)
        var artifacts = [
            QwenPlannedArtifact(relativePath: "model_weights.bin", size: residentSize),
            QwenPlannedArtifact(relativePath: "packed_experts/layout.json", size: layoutBytes),
        ]
        artifacts.append(contentsOf: expertLayers.map {
            QwenPlannedArtifact(relativePath: $0.relativePath, size: $0.fileSize)
        })
        if let visionPlan { artifacts.append(visionPlan.artifact) }
        artifacts.sort { $0.relativePath < $1.relativePath }
        var requiredBytes: UInt64 = 0
        for artifact in artifacts { requiredBytes = try add(requiredBytes, artifact.size) }

        let scratchTotal = try checkedIntAdd(
            try checkedIntAdd(transformTileBytes, BF16AffineTransformReader.maximumGroupBytes),
            StreamingBF16AffineQuantizer.maximumScratchPayloadBytes)
        let scratch = QwenScratchBudget(
            transformTileBytes: transformTileBytes,
            maximumGroupBytes: BF16AffineTransformReader.maximumGroupBytes,
            quantizerPayloadBytes: StreamingBF16AffineQuantizer.maximumScratchPayloadBytes,
            maximumPayloadBufferBytes: scratchTotal)
        let omitted = omittedRecords.map { $0.0.name }
        let fingerprint = try fingerprint(
            architecture: architecture,
            provenance: provenance,
            textTensors: textPlans,
            expertLayers: expertLayers,
            visionNames: visionNames,
            visionCompanion: visionPlan,
            omitted: omitted,
            artifacts: artifacts,
            requiredBytes: requiredBytes,
            scratch: scratch,
            outputDirectory: outputDirectory)
        return LayoutResult(
            textTensors: textPlans,
            expertLayers: expertLayers,
            visionTensorNames: visionNames,
            visionCompanion: visionPlan,
            omittedMTPNames: omitted,
            artifacts: artifacts,
            requiredArtifactBytes: requiredBytes,
            scratch: scratch,
            canonicalFingerprint: fingerprint)
    }

    private static func plannedTensor(
        record: (SourceTensor, QwenOfficialTensorCategory, BF16AffinePolicyDecision),
        relativeFile: String,
        executionPosition: Int,
        fileOffset: UInt64
    ) throws -> QwenPlannedTensor {
        let source = record.0
        let decision = record.2
        switch decision.storage {
        case .retainedBF16:
            return QwenPlannedTensor(
                source: source,
                relativeFile: relativeFile,
                executionPosition: executionPosition,
                fileOffset: fileOffset,
                regionSize: source.sizeBytes,
                storage: decision.storage,
                quantizationCategory: decision.manifestCategory,
                affineComponents: nil)
        case .affineInt4, .affineInt8:
            let layout = try affineComponentSizes(
                shape: source.shape,
                bitWidth: try requiredBitWidth(decision, name: source.name))
            return QwenPlannedTensor(
                source: source,
                relativeFile: relativeFile,
                executionPosition: executionPosition,
                fileOffset: fileOffset,
                regionSize: layout.totalSize,
                storage: decision.storage,
                quantizationCategory: decision.manifestCategory,
                affineComponents: layout)
        case .omittedMTP:
            throw RepackError.configurationInvalid(
                detail: "omitted MTP tensor entered an artifact: \(source.name)")
        }
    }

    private static func validateExactArchitecture(
        _ architecture: ArchInfo.QwenPlanningArchitecture,
        requiresPinnedConfigDigest: Bool
    ) throws {
        let expectedLayers = (0..<40).map {
            ($0 + 1).isMultiple(of: 4)
                ? GTurboQwenLayerTypeV2.fullAttention : .linearAttention
        }
        let text = architecture.text
        let vision = architecture.vision
        guard (!requiresPinnedConfigDigest
                || architecture.observedConfigSHA256
                    == GTurboFormatV2.qwenSidecarSHA256["config.json"]),
              architecture.architectureName == "Qwen3_5MoeForConditionalGeneration",
              architecture.modelType == "qwen3_5_moe",
              architecture.textModelType == "qwen3_5_moe_text",
              architecture.sourceDType == "bfloat16",
              architecture.fullAttentionInterval == 4,
              architecture.mtpNumHiddenLayers == 1,
              !architecture.mtpUsesDedicatedEmbeddings,
              architecture.maxPositionEmbeddings == 262_144,
              architecture.rmsNormEpsilon == 0.000001,
              text.hiddenSize == 2_048, text.numLayers == 40,
              text.layerTypes == expectedLayers,
              text.numAttentionHeads == 16, text.numKeyValueHeads == 2,
              text.headDimension == 256, text.attentionOutputGate,
              text.linearConvolutionKernel == 4,
              text.linearKeyHeads == 16, text.linearKeyHeadDimension == 128,
              text.linearValueHeads == 32, text.linearValueHeadDimension == 128,
              text.recurrentStateType == .fp32,
              text.partialRotaryFactor == 0.25, text.ropeTheta == 10_000_000,
              text.mropeInterleaved, text.mropeSections == [11, 11, 10],
              text.numberOfExperts == 256, text.expertsPerToken == 8,
              text.routedExpertIntermediateSize == 512,
              text.sharedExpertIntermediateSize == 512,
              text.vocabularySize == 248_320,
              !text.tiedWordEmbeddings, text.hiddenActivation == "silu",
              vision == ArchInfo.QwenVisionArchitecture(
                depth: 27, hiddenSize: 1_152, intermediateSize: 4_304,
                numHeads: 16, numPositionEmbeddings: 2_304,
                inputChannels: 3, patchSize: 16, temporalPatchSize: 2,
                spatialMergeSize: 2, outputHiddenSize: 2_048,
                hiddenActivation: "gelu_pytorch_tanh") else {
            throw RepackError.configurationInvalid(detail: "Qwen planning architecture is invalid")
        }
    }

    private static func validateSourceTensor(
        _ tensor: SourceTensor,
        architecture: ArchInfo.QwenPlanningArchitecture
    ) throws {
        guard tensor.dtype == .bf16 else {
            throw RepackError.dtypeMismatch(name: tensor.name, detail: "Qwen source must be BF16")
        }
        let expected = try expectedShape(for: tensor.name, architecture: architecture)
        guard tensor.shape == expected else {
            throw RepackError.shapeMismatch(
                name: tensor.name,
                detail: "expected \(expected), got \(tensor.shape)")
        }
        let expectedBytes = try multiply(try product(tensor.shape[...]), 2)
        guard tensor.sizeBytes == expectedBytes else {
            throw RepackError.shapeMismatch(
                name: tensor.name,
                detail: "expected \(expectedBytes) BF16 bytes, got \(tensor.sizeBytes)")
        }
        _ = try add(tensor.absoluteOffset, tensor.sizeBytes)
        guard isSafeShardBasename((tensor.shardPath as NSString).lastPathComponent) else {
            throw RepackError.configurationInvalid(detail: "unsafe Qwen source shard path")
        }
    }

    private static func expectedShape(
        for name: String,
        architecture: ArchInfo.QwenPlanningArchitecture
    ) throws -> [UInt64] {
        let text = architecture.text
        let h = UInt64(text.hiddenSize)
        let expert = UInt64(text.routedExpertIntermediateSize)
        let experts = UInt64(text.numberOfExperts)
        if name == "lm_head.weight" { return [UInt64(text.vocabularySize), h] }
        if name == "model.language_model.embed_tokens.weight" {
            return [UInt64(text.vocabularySize), h]
        }
        if name == "model.language_model.norm.weight" { return [h] }
        if name.hasPrefix("model.visual.") {
            return try expectedVisionShape(name, architecture: architecture)
        }
        if name.hasPrefix("mtp.") {
            let suffix = String(name.dropFirst("mtp.".count))
            if suffix == "fc.weight" { return [h, try multiply(2, h)] }
            if suffix == "norm.weight" || suffix == "pre_fc_norm_embedding.weight"
                || suffix == "pre_fc_norm_hidden.weight" { return [h] }
            if suffix.hasPrefix("layers.0.") {
                return try expectedLayerShape(
                    String(suffix.dropFirst("layers.0.".count)),
                    h: h, expert: expert, experts: experts, text: text)
            }
        }
        guard let layer = layerIndex(in: name), (0..<40).contains(layer),
              let range = name.range(of: ".layers.\(layer).") else {
            throw RepackError.shapeMismatch(name: name, detail: "no Qwen shape rule")
        }
        let suffix = String(name[range.upperBound...])
        let shape = try expectedLayerShape(
            suffix, h: h, expert: expert, experts: experts, text: text)
        let isFullSuffix = suffix.hasPrefix("self_attn.")
        let isLinearSuffix = suffix.hasPrefix("linear_attn.")
        let isFullLayer = text.layerTypes[layer] == .fullAttention
        guard (!isFullSuffix || isFullLayer), (!isLinearSuffix || !isFullLayer) else {
            throw RepackError.shapeMismatch(name: name, detail: "tensor disagrees with layer schedule")
        }
        return shape
    }

    private static func expectedLayerShape(
        _ suffix: String,
        h: UInt64,
        expert: UInt64,
        experts: UInt64,
        text: GTurboQwenArchitectureV2
    ) throws -> [UInt64] {
        switch suffix {
        case "input_layernorm.weight", "post_attention_layernorm.weight": return [h]
        case "mlp.gate.weight": return [experts, h]
        case "mlp.shared_expert_gate.weight": return [1, h]
        case "mlp.shared_expert.down_proj.weight": return [h, expert]
        case "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight":
            return [expert, h]
        case "mlp.experts.down_proj": return [experts, h, expert]
        case "mlp.experts.gate_up_proj": return [experts, try multiply(2, expert), h]
        case "self_attn.q_norm.weight", "self_attn.k_norm.weight":
            return [UInt64(text.headDimension)]
        case "self_attn.q_proj.weight":
            return [try multiply(UInt64(text.numAttentionHeads),
                                 try multiply(UInt64(text.headDimension), 2)), h]
        case "self_attn.k_proj.weight", "self_attn.v_proj.weight":
            return [try multiply(UInt64(text.numKeyValueHeads), UInt64(text.headDimension)), h]
        case "self_attn.o_proj.weight":
            return [h, try multiply(UInt64(text.numAttentionHeads), UInt64(text.headDimension))]
        case "linear_attn.A_log", "linear_attn.dt_bias":
            return [UInt64(text.linearValueHeads)]
        case "linear_attn.conv1d.weight":
            let channels = try add(
                try multiply(2, try multiply(UInt64(text.linearKeyHeads), UInt64(text.linearKeyHeadDimension))),
                try multiply(UInt64(text.linearValueHeads), UInt64(text.linearValueHeadDimension)))
            return [channels, 1, UInt64(text.linearConvolutionKernel)]
        case "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight":
            return [UInt64(text.linearValueHeads), h]
        case "linear_attn.in_proj_qkv.weight":
            let output = try add(
                try multiply(2, try multiply(UInt64(text.linearKeyHeads), UInt64(text.linearKeyHeadDimension))),
                try multiply(UInt64(text.linearValueHeads), UInt64(text.linearValueHeadDimension)))
            return [output, h]
        case "linear_attn.in_proj_z.weight":
            return [try multiply(UInt64(text.linearValueHeads), UInt64(text.linearValueHeadDimension)), h]
        case "linear_attn.norm.weight": return [UInt64(text.linearValueHeadDimension)]
        case "linear_attn.out_proj.weight":
            return [h, try multiply(UInt64(text.linearValueHeads), UInt64(text.linearValueHeadDimension))]
        default:
            throw RepackError.configurationInvalid(detail: "no shape rule for Qwen member \(suffix)")
        }
    }

    private static func expectedVisionShape(
        _ name: String,
        architecture: ArchInfo.QwenPlanningArchitecture
    ) throws -> [UInt64] {
        let v = architecture.vision
        let h = UInt64(v.hiddenSize)
        let intermediate = UInt64(v.intermediateSize)
        if name == "model.visual.patch_embed.proj.weight" {
            return [h, UInt64(v.inputChannels), UInt64(v.temporalPatchSize),
                    UInt64(v.patchSize), UInt64(v.patchSize)]
        }
        if name == "model.visual.patch_embed.proj.bias" { return [h] }
        if name == "model.visual.pos_embed.weight" {
            return [UInt64(v.numPositionEmbeddings), h]
        }
        let merged = try multiply(h, UInt64(v.spatialMergeSize * v.spatialMergeSize))
        switch name {
        case "model.visual.merger.norm.weight", "model.visual.merger.norm.bias": return [h]
        case "model.visual.merger.linear_fc1.weight": return [merged, merged]
        case "model.visual.merger.linear_fc1.bias": return [merged]
        case "model.visual.merger.linear_fc2.weight": return [UInt64(v.outputHiddenSize), merged]
        case "model.visual.merger.linear_fc2.bias": return [UInt64(v.outputHiddenSize)]
        default: break
        }
        guard let block = visionBlockIndex(in: name), (0..<v.depth).contains(block),
              let range = name.range(of: ".blocks.\(block).") else {
            throw RepackError.configurationInvalid(detail: "no shape rule for vision tensor \(name)")
        }
        switch String(name[range.upperBound...]) {
        case "attn.proj.weight": return [h, h]
        case "attn.proj.bias": return [h]
        case "attn.qkv.weight": return [try multiply(3, h), h]
        case "attn.qkv.bias": return [try multiply(3, h)]
        case "mlp.linear_fc1.weight": return [intermediate, h]
        case "mlp.linear_fc1.bias": return [intermediate]
        case "mlp.linear_fc2.weight": return [h, intermediate]
        case "mlp.linear_fc2.bias", "norm1.weight", "norm1.bias",
             "norm2.weight", "norm2.bias": return [h]
        default:
            throw RepackError.configurationInvalid(detail: "no shape rule for vision member \(name)")
        }
    }

    private static func requiredBitWidth(
        _ decision: BF16AffinePolicyDecision,
        name: String
    ) throws -> AffineBitWidth {
        guard let width = decision.storage.affineBitWidth,
              decision.groupingAxis == .lastDimension,
              decision.groupSize == BF16AffineQuantizationPolicy.affineGroupSize,
              decision.manifestCategory != nil else {
            throw RepackError.configurationInvalid(
                detail: "invalid affine policy for \(name)")
        }
        return width
    }

    private static func expertLayoutMetadataSize(
        _ layers: [QwenExpertLayerPlan]
    ) throws -> UInt64 {
        let objects: [[String: Any]] = layers.map { layer in
            [
                "layer": layer.layerIndex,
                "path": layer.relativePath,
                "experts": layer.expertsPerLayer,
                "stride": layer.expertStride,
                "sources": layer.sources.map { source in
                    [
                        "name": source.source.name,
                        "role": source.role,
                        "shape": source.perExpertShape,
                        "valuesOffset": source.components.valuesOffset,
                        "valuesSize": source.components.valuesSize,
                        "scalesOffset": source.components.scalesOffset,
                        "scalesSize": source.components.scalesSize,
                        "biasesOffset": source.components.biasesOffset,
                        "biasesSize": source.components.biasesSize,
                    ] as [String: Any]
                },
            ]
        }
        do {
            let data = try JSONSerialization.data(
                withJSONObject: ["version": 2, "layers": objects],
                options: [.sortedKeys])
            return UInt64(data.count)
        } catch {
            throw RepackError.configurationInvalid(
                detail: "cannot encode Qwen expert layout metadata: \(error)")
        }
    }

    private static func fingerprint(
        architecture: ArchInfo.QwenPlanningArchitecture,
        provenance: QwenPlanningProvenance,
        textTensors: [QwenPlannedTensor],
        expertLayers: [QwenExpertLayerPlan],
        visionNames: [String],
        visionCompanion: QwenVisionCompanionPlan?,
        omitted: [String],
        artifacts: [QwenPlannedArtifact],
        requiredBytes: UInt64,
        scratch: QwenScratchBudget,
        outputDirectory: String
    ) throws -> String {
        // Validate the root but never hash it.
        guard !outputDirectory.contains("\0") else {
            throw RepackError.configurationInvalid(detail: "invalid Qwen output root")
        }
        var writer = QwenFingerprintWriter(domain: "TurboFieldfare.QwenRepackPlan.v1")
        writer.append(provenance.repository)
        writer.append(provenance.revision)
        writer.append(provenance.observedIndexSHA256)
        writer.append(provenance.observedConfigSHA256)
        writer.append(provenance.quantizationPolicySHA256)
        for key in provenance.validatedSidecarSHA256.keys.sorted() {
            writer.append(key)
            writer.append(provenance.validatedSidecarSHA256[key]!)
        }
        append(architecture: architecture, to: &writer)
        writer.append(UInt64(artifacts.count))
        for artifact in artifacts {
            writer.append(artifact.relativePath)
            writer.append(artifact.size)
        }
        writer.append(UInt64(textTensors.count))
        for tensor in textTensors { try append(tensor: tensor, to: &writer) }
        writer.append(UInt64(expertLayers.count))
        for layer in expertLayers {
            writer.append(UInt64(layer.layerIndex))
            writer.append(layer.relativePath)
            writer.append(UInt64(layer.expertsPerLayer))
            writer.append(layer.expertStride)
            writer.append(layer.fileSize)
            for source in layer.sources {
                writer.append(source.role)
                try append(source: source.source, to: &writer)
                for extent in source.perExpertShape { writer.append(extent) }
                append(components: source.components, to: &writer)
            }
        }
        writer.append(visionCompanion == nil ? UInt64(0) : UInt64(1))
        writer.append(UInt64(visionNames.count))
        for name in visionNames { writer.append(name) }
        if let visionCompanion {
            for tensor in visionCompanion.tensors { try append(tensor: tensor, to: &writer) }
        }
        writer.append(UInt64(omitted.count))
        for name in omitted { writer.append(name) }
        writer.append(requiredBytes)
        writer.append(UInt64(scratch.transformTileBytes))
        writer.append(UInt64(scratch.maximumGroupBytes))
        writer.append(UInt64(scratch.quantizerPayloadBytes))
        writer.append(UInt64(scratch.maximumPayloadBufferBytes))
        return writer.finalize()
    }

    private static func append(
        architecture: ArchInfo.QwenPlanningArchitecture,
        to writer: inout QwenFingerprintWriter
    ) {
        let t = architecture.text
        let v = architecture.vision
        writer.append(architecture.architectureName)
        writer.append(architecture.modelType)
        writer.append(architecture.textModelType)
        writer.append(architecture.sourceDType)
        for value in [
            t.hiddenSize, t.numLayers, t.numAttentionHeads, t.numKeyValueHeads,
            t.headDimension, t.linearConvolutionKernel, t.linearKeyHeads,
            t.linearKeyHeadDimension, t.linearValueHeads, t.linearValueHeadDimension,
            t.numberOfExperts, t.expertsPerToken, t.routedExpertIntermediateSize,
            t.sharedExpertIntermediateSize, t.vocabularySize,
            architecture.fullAttentionInterval, architecture.mtpNumHiddenLayers,
            architecture.maxPositionEmbeddings, v.depth, v.hiddenSize,
            v.intermediateSize, v.numHeads, v.numPositionEmbeddings,
            v.inputChannels, v.patchSize, v.temporalPatchSize,
            v.spatialMergeSize, v.outputHiddenSize,
        ] { writer.append(UInt64(value)) }
        for type in t.layerTypes { writer.append(type == .fullAttention ? "full" : "linear") }
        writer.append(t.attentionOutputGate ? UInt64(1) : UInt64(0))
        writer.append(t.mropeInterleaved ? UInt64(1) : UInt64(0))
        for section in t.mropeSections { writer.append(UInt64(section)) }
        writer.append(t.partialRotaryFactor.bitPattern)
        writer.append(t.ropeTheta.bitPattern)
        writer.append(architecture.rmsNormEpsilon.bitPattern)
        writer.append(t.hiddenActivation)
        writer.append(v.hiddenActivation)
        writer.append(architecture.observedConfigSHA256)
    }

    private static func append(
        tensor: QwenPlannedTensor,
        to writer: inout QwenFingerprintWriter
    ) throws {
        try append(source: tensor.source, to: &writer)
        writer.append(tensor.relativeFile)
        writer.append(UInt64(tensor.executionPosition))
        writer.append(tensor.fileOffset)
        writer.append(tensor.regionSize)
        writer.append(tensor.storage.rawValue)
        writer.append(tensor.quantizationCategory?.rawValue ?? "none")
        if let components = tensor.affineComponents {
            writer.append(UInt64(1))
            append(components: components, to: &writer)
        } else {
            writer.append(UInt64(0))
        }
    }

    private static func append(
        source: SourceTensor,
        to writer: inout QwenFingerprintWriter
    ) throws {
        writer.append(source.name)
        let basename = (source.shardPath as NSString).lastPathComponent
        guard isSafeShardBasename(basename) else {
            throw RepackError.configurationInvalid(detail: "unsafe Qwen shard basename")
        }
        writer.append(basename)
        writer.append(source.absoluteOffset)
        writer.append(source.sizeBytes)
        writer.append(UInt64(source.dtype.rawValue))
        writer.append(UInt64(source.shape.count))
        for extent in source.shape { writer.append(extent) }
    }

    private static func append(
        components: QwenAffineComponentLayout,
        to writer: inout QwenFingerprintWriter
    ) {
        writer.append(components.valuesOffset)
        writer.append(components.valuesSize)
        writer.append(components.scalesOffset)
        writer.append(components.scalesSize)
        writer.append(components.biasesOffset)
        writer.append(components.biasesSize)
    }

    private static func executionKey(_ name: String) -> String {
        if name == "model.language_model.embed_tokens.weight" { return "0000" }
        if let layer = layerIndex(in: name) {
            return String(format: "1000/%03d/%@", layer, name)
        }
        if name == "model.language_model.norm.weight" { return "2000" }
        if name == "lm_head.weight" { return "3000" }
        return "9999/\(name)"
    }

    private static func visionExecutionKey(_ name: String) -> String {
        if name.hasPrefix("model.visual.patch_embed.") { return "0000/\(name)" }
        if name == "model.visual.pos_embed.weight" { return "0100/\(name)" }
        if let block = visionBlockIndex(in: name) {
            return String(format: "1000/%03d/%@", block, name)
        }
        if name.hasPrefix("model.visual.merger.") { return "2000/\(name)" }
        return "9999/\(name)"
    }

    private static func layerIndex(in name: String) -> Int? {
        guard let range = name.range(of: ".layers.") else { return nil }
        let suffix = name[range.upperBound...]
        guard let dot = suffix.firstIndex(of: ".") else { return nil }
        return Int(suffix[..<dot])
    }

    private static func visionBlockIndex(in name: String) -> Int? {
        guard let range = name.range(of: ".blocks.") else { return nil }
        let suffix = name[range.upperBound...]
        guard let dot = suffix.firstIndex(of: ".") else { return nil }
        return Int(suffix[..<dot])
    }

    private static func routedRole(_ name: String) -> String {
        name.hasSuffix("gate_up_proj") ? "gate_up" : "down"
    }

    private static func isSafeShardBasename(_ name: String) -> Bool {
        !name.isEmpty && name != "." && name != ".."
            && !name.contains("/") && !name.contains("\\")
            && name.hasSuffix(".safetensors")
    }

    private static func product<S: Sequence>(_ values: S) throws -> UInt64
    where S.Element == UInt64 {
        var result: UInt64 = 1
        for value in values {
            guard value > 0 else {
                throw RepackError.configurationInvalid(detail: "zero Qwen tensor extent")
            }
            result = try multiply(result, value)
        }
        return result
    }

    private static func add(_ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else { throw RepackError.configurationInvalid(detail: "Qwen size addition overflow") }
        return value
    }

    private static func multiply(_ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else { throw RepackError.configurationInvalid(detail: "Qwen size multiplication overflow") }
        return value
    }

    private static func ceilingDivide(_ value: UInt64, by divisor: UInt64) throws -> UInt64 {
        guard divisor > 0 else {
            throw RepackError.configurationInvalid(detail: "Qwen size division by zero")
        }
        return value / divisor + (value % divisor == 0 ? 0 : 1)
    }

    private static func alignUp(_ value: UInt64) throws -> UInt64 {
        let alignment = GTurboFormatV2.alignmentBytes
        return try multiply(try ceilingDivide(value, by: alignment), alignment)
    }

    private static func checkedIntAdd(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else { throw RepackError.configurationInvalid(detail: "Qwen scratch overflow") }
        return value
    }
}

private struct QwenFingerprintWriter {
    private var stream = Sha256Stream()

    init(domain: String) { append(domain) }

    mutating func append(_ value: UInt64) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { stream.update($0) }
    }

    mutating func append(_ value: String) {
        let data = Data(value.utf8)
        append(UInt64(data.count))
        data.withUnsafeBytes { stream.update($0) }
    }

    func finalize() -> String { stream.finalizeHexString() }
}
