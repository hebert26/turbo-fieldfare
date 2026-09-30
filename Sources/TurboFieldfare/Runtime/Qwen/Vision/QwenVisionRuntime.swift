import Foundation
import Metal
import TurboFieldfareFormat

struct QwenVisionRuntimeDiagnostics: Equatable, Sendable {
    var orderedSubmittedGroups: [QwenVisionWeightGroup] = []
    var orderedLayerIntermediates: [[Float]] = []
    var mutableRequestedBytes: Int = 0
    var mutableHighWaterBytes: Int = 0
    var mappedPageAlignedBytes: [UInt64] = []
    var maximumLiveMappedBytes: UInt64 = 0
    var completedCommandBuffers: Int = 0
    var discardedCommandBuffers: Int = 0
}

struct QwenVisionFeatures: Sendable {
    let owner: QwenRetainedFeatureOwner
    let diagnostics: QwenVisionRuntimeDiagnostics

    var tokenCount: Int { owner.rowCount }
    var hiddenSize: Int { owner.hiddenSize }
    var grid: QwenVisionGrid { owner.grid }

    init(
        device: MTLDevice,
        features: [Float],
        positions: [QwenMRoPEPosition],
        imageDigest: String,
        processorDigest: String,
        profile: GTurboQwenVisionProcessorProfileV2,
        grid: QwenVisionGrid,
        diagnostics: QwenVisionRuntimeDiagnostics = .init(),
        hiddenSize: Int = QwenVisionConfig.official.outputHiddenSize
    ) throws {
        owner = try QwenRetainedFeatureOwner(
            device: device, features: features, positions: positions,
            imageDigest: imageDigest, processorDigest: processorDigest,
            profile: profile, grid: grid, hiddenSize: hiddenSize)
        self.diagnostics = diagnostics
    }
}

/// Mutable buffers for one image. The requested-length plan is built before
/// allocation and is the source of truth for the 200 MB safety gate.
/// Immutable buffer topology. All GPU writes are submitted serially by the
/// owning `QwenVisionRuntime` actor and each submission completes before an
/// aliased workspace is reused; callers cannot replace any buffer reference.
final class QwenVisionScratch: @unchecked Sendable {
    let paddedRows: Int
    let mergedRows: Int
    let allocationPlan: QwenVisionAllocationPlan
    let normalizedPatches: MTLBuffer
    let hiddenA: MTLBuffer
    let hiddenB: MTLBuffer
    let q: MTLBuffer
    let k: MTLBuffer
    let v: MTLBuffer
    let attention: MTLBuffer
    let gate: MTLBuffer
    let up: MTLBuffer
    let mergerPacked: MTLBuffer
    let mergerNormalized: MTLBuffer
    let mergerHidden: MTLBuffer
    let outputBF16: MTLBuffer
    let outputFloat32: MTLBuffer
    let sourceFC1Staging: MTLBuffer?

    init(
        device: MTLDevice,
        rows: Int,
        config: QwenVisionConfig,
        retainedRequestedBytes: Int = 0,
        sourceFC1Staging: Bool = false,
        limits: QwenVisionResourceLimits = .provisional
    ) throws {
        guard rows > 0, rows <= config.maximumPatchRows else {
            throw QwenVisionError.patchRowQuotaExceeded(
                requested: rows, maximum: config.maximumPatchRows)
        }
        paddedRows = ((rows + 63) / 64) * 64
        let mergeArea = config.spatialMergeSize * config.spatialMergeSize
        guard rows.isMultiple(of: mergeArea) else {
            throw QwenVisionError.invalidFeatureShape
        }
        mergedRows = rows / mergeArea
        func bytes(_ count: Int, _ width: Int, _ stride: Int, _ name: String) throws -> QwenVisionAllocationPlan.Entry {
            let (elements, firstOverflow) = count.multipliedReportingOverflow(by: width)
            let (length, secondOverflow) = elements.multipliedReportingOverflow(by: stride)
            guard !firstOverflow, !secondOverflow else {
                throw QwenVisionError.arithmeticOverflow(name)
            }
            return .init(name: name, bytes: length)
        }
        // Activations stay Float32 to preserve the P3 1e-5 oracle contract.
        // The entries below are physical allocations; logical stage aliases
        // intentionally reuse them after command-buffer completion.
        let fp32 = MemoryLayout<Float>.stride
        var entries = try [
            bytes(paddedRows, config.hiddenSize, fp32, "hidden A"),
            bytes(paddedRows, config.hiddenSize, fp32, "hidden B"),
            bytes(paddedRows, 3 * config.hiddenSize, fp32, "QKV workspace"),
            bytes(paddedRows, config.hiddenSize, fp32, "attention workspace"),
            bytes(paddedRows, config.intermediateSize, fp32, "MLP workspace"),
            bytes(mergedRows, config.outputHiddenSize, fp32,
                  "retained Float32 output staging"),
            .init(name: "retained lineage", bytes: retainedRequestedBytes),
        ]
        if sourceFC1Staging {
            entries.append(.init(
                name: "source FC1 Float32 weight staging",
                bytes: try QwenSourceVisionLinear.stagingByteCount(
                    inputWidth: config.hiddenSize, outputWidth: config.intermediateSize)))
        }
        allocationPlan = try QwenVisionAllocationPlan(entries: entries, limits: limits)
        try allocationPlan.validate(device: device)
        var allocated: [String: MTLBuffer] = [:]
        for entry in entries where entry.name != "retained lineage" {
            guard let buffer = device.makeBuffer(
                length: entry.bytes, options: .storageModeShared) else {
                throw QwenVisionError.allocationFailed(name: entry.name)
            }
            buffer.label = "qwen.vision.\(entry.name)"
            allocated[entry.name] = buffer
        }
        func required(_ name: String) throws -> MTLBuffer {
            guard let value = allocated[name] else {
                throw QwenVisionError.allocationFailed(name: name)
            }
            return value
        }
        hiddenA = try required("hidden A")
        hiddenB = try required("hidden B")
        let qkv = try required("QKV workspace")
        let attentionWorkspace = try required("attention workspace")
        let mlp = try required("MLP workspace")
        outputFloat32 = try required("retained Float32 output staging")
        self.sourceFC1Staging = allocated["source FC1 Float32 weight staging"]
        // Stage aliases. Runtime offsets Q/K/V within `qkv`.
        normalizedPatches = qkv
        q = qkv
        k = qkv
        v = qkv
        attention = attentionWorkspace
        gate = mlp
        up = mlp
        mergerPacked = qkv
        mergerNormalized = attentionWorkspace
        mergerHidden = mlp
        outputBF16 = outputFloat32
    }
}

private struct QwenVisionKernelParameters {
    var rows: UInt32
    var paddedRows: UInt32
    var inputWidth: UInt32
    var outputWidth: UInt32
    var intermediateWidth: UInt32
    var heads: UInt32
    var gridHeight: UInt32
    var gridWidth: UInt32
    var mergeSize: UInt32
    var positionCount: UInt32
    var epsilon: Float
    var weightScalarBytes: UInt32
    var patchScalarBytes: UInt32
    var reserved: UInt32 = 0
}

struct QwenVisionFixtureResult: Sendable {
    let output: [Float]
    let orderedLayerIntermediates: [[Float]]
    let allocationPlan: QwenVisionAllocationPlan
}

/// Immutable Metal bindings. The runtime actor serializes all encodes and
/// retains each mapped lease through command completion.
private enum QwenVisionTensorSource: @unchecked Sendable {
    case mapped(QwenVisionMappedGroupLease)
    case fixture([String: MTLBuffer])
}

enum QwenVisionExecutionEvent: Equatable, Sendable {
    case scratchAllocated(bytes: Int)
    case commandSubmitted(String)
    case commandCompleted(String, discarded: Bool)
    case groupMapped(QwenVisionWeightGroup, bytes: UInt64)
    case groupReleased(QwenVisionWeightGroup)
    case scratchReleased
}

struct QwenVisionExecutionHooks: Sendable {
    var afterCommandSubmission: @Sendable (String) async -> Void
    var observe: @Sendable (QwenVisionExecutionEvent) -> Void
    /// Opt-in fault isolation, never an acceptance run. It completes one
    /// operation at a time and stops at the first bad output or after block 0.
    var firstBlockDiagnostics: (@Sendable (QwenVisionStageDiagnostic) -> Void)? = nil

    /// Copies only one selected merger row after completed GPU operations.
    var mergerDiagnosticRow: Int = 0
    var mergerDiagnostics: (@Sendable (QwenVisionMergerDiagnostic) -> Void)? = nil
    /// Opt-in completed tower boundaries. No arrays are copied by default.
    var towerDiagnostics: (@Sendable (QwenVisionTowerDiagnostic) -> Void)? = nil
    /// Source-only patch discriminator, intentionally stops before tower blocks.
    var initialStageDiagnostics: (@Sendable (QwenVisionTowerDiagnostic) -> Void)? = nil
    /// Source-only completed block-0 arrays; intentionally stops before block 1.
    var firstBlockStageDiagnostics: (@Sendable (QwenVisionTowerDiagnostic) -> Void)? = nil

    static let none = QwenVisionExecutionHooks(
        afterCommandSubmission: { _ in }, observe: { _ in })
}

struct QwenVisionTowerDiagnostic: Sendable {
    let stage: String
    let rows: Int
    let width: Int
    let values: [Float]
}

struct QwenVisionMergerDiagnostic: Sendable {
    let stage: String
    let row: Int
    let values: [Float]
}

struct QwenVisionStageDiagnostic: Sendable {
    let stage: String
    let count: Int
    let finiteCount: Int
    let nanCount: Int
    let positiveInfinityCount: Int
    let negativeInfinityCount: Int
    let firstNonfiniteIndex: Int?
    let firstNonfiniteIndices: [Int]
    let sampleBits: [UInt32]
    let finiteMinimum: Float?
    let finiteMaximum: Float?
}

enum QwenVisionSourceRotaryArithmetic {
    static func inverseFrequencies(headDimension: Int) throws -> [Float] {
        guard headDimension == 72 else { throw QwenVisionError.invalidConfiguration }
        return (0..<18).map { index in
            let exponent = Float(2 * index) / Float(36)
            return Float(1) / powf(Float(10_000), exponent)
        }
    }
}

actor QwenVisionRuntime {
    private let context: MetalContext
    private let store: QwenVisionWeightStore?
    private let sourceStore: QwenOfficialSourceVisionWeightStore?
    private let config: QwenVisionConfig
    private let limits: QwenVisionResourceLimits
    private let executionHooks: QwenVisionExecutionHooks
    private let patchPipeline: MTLComputePipelineState
    private let normPipeline: MTLComputePipelineState
    private let rotatePipeline: MTLComputePipelineState
    private let sourceInverseFrequencyBuffer: MTLBuffer?
    private let sourceAttentionScale: Float?
    private let sourceMergerGELUPipeline: MTLComputePipelineState?
    private let sourceFC1GELUPipeline: MTLComputePipelineState?
    private let attentionPipeline: MTLComputePipelineState
    private let residualPipeline: MTLComputePipelineState
    private let linearTanhGELUPipeline: MTLComputePipelineState
    private let mergerPackPipeline: MTLComputePipelineState
    private let linearGELUPipeline: MTLComputePipelineState
    private let linearPipeline: MTLComputePipelineState
    private var processing = false

    init(
        context: MetalContext,
        store: QwenVisionWeightStore,
        config: QwenVisionConfig = .official,
        limits: QwenVisionResourceLimits = .provisional,
        executionHooks: QwenVisionExecutionHooks = .none
    ) throws {
        try config.validate()
        guard config.matchesOfficialWeightLayout else {
            throw QwenVisionError.invalidConfiguration
        }
        self.context = context
        self.store = store
        sourceStore = nil
        sourceInverseFrequencyBuffer = nil
        sourceAttentionScale = nil
        sourceMergerGELUPipeline = nil
        sourceFC1GELUPipeline = nil
        self.config = config
        self.limits = limits
        self.executionHooks = executionHooks
        patchPipeline = try Self.safePipeline(context: context, name: "qwen_vision_patch_position")
        normPipeline = try Self.safePipeline(context: context, name: "qwen_vision_layernorm")
        rotatePipeline = try Self.safePipeline(context: context, name: "qwen_vision_rotate_qk")
        attentionPipeline = try Self.safePipeline(context: context, name: "qwen_vision_attention")
        residualPipeline = try Self.safePipeline(context: context, name: "qwen_vision_linear_residual")
        linearTanhGELUPipeline = try Self.safePipeline(context: context, name: "qwen_vision_linear_gelu_tanh")
        mergerPackPipeline = try Self.safePipeline(context: context, name: "qwen_vision_merger_norm_pack")
        linearGELUPipeline = try Self.safePipeline(context: context, name: "qwen_vision_linear_gelu")
        linearPipeline = try Self.safePipeline(context: context, name: "qwen_vision_linear")
    }

    /// Tiny-fixture initializer. It uses the production pipelines and Float32
    /// activation topology without accepting an unverified production pack.
    init(
        context: MetalContext,
        fixtureConfig config: QwenVisionConfig,
        limits: QwenVisionResourceLimits = .provisional,
        executionHooks: QwenVisionExecutionHooks = .none
    ) throws {
        try config.validate()
        self.context = context
        store = nil
        sourceStore = nil
        sourceInverseFrequencyBuffer = nil
        sourceAttentionScale = nil
        sourceMergerGELUPipeline = nil
        sourceFC1GELUPipeline = nil
        self.config = config
        self.limits = limits
        self.executionHooks = executionHooks
        patchPipeline = try Self.safePipeline(context: context, name: "qwen_vision_patch_position")
        normPipeline = try Self.safePipeline(context: context, name: "qwen_vision_layernorm")
        rotatePipeline = try Self.safePipeline(context: context, name: "qwen_vision_rotate_qk")
        attentionPipeline = try Self.safePipeline(context: context, name: "qwen_vision_attention")
        residualPipeline = try Self.safePipeline(context: context, name: "qwen_vision_linear_residual")
        linearTanhGELUPipeline = try Self.safePipeline(context: context, name: "qwen_vision_linear_gelu_tanh")
        mergerPackPipeline = try Self.safePipeline(context: context, name: "qwen_vision_merger_norm_pack")
        linearGELUPipeline = try Self.safePipeline(context: context, name: "qwen_vision_linear_gelu")
        linearPipeline = try Self.safePipeline(context: context, name: "qwen_vision_linear")
    }

    /// The source store reads only the group requested by execute; the
    /// existing BF16 vision pipelines, scratch cap and completion gates apply.
    init(context: MetalContext, sourceStore: QwenOfficialSourceVisionWeightStore,
         limits: QwenVisionResourceLimits = .provisional,
         executionHooks: QwenVisionExecutionHooks = .none) throws {
        try sourceStore.config.validate()
        // mapGroup checks the retained source model's exact Metal device.
        self.context = context
        store = nil
        self.sourceStore = sourceStore
        config = sourceStore.config
        self.limits = limits
        self.executionHooks = executionHooks
        patchPipeline = try Self.safePipeline(context: context, name: "qwen_source_vision_patch_position", preciseFunctions: true)
        normPipeline = try Self.safePipeline(context: context, name: "qwen_source_vision_layernorm", preciseFunctions: true)
        if config.matchesOfficialWeightLayout {
            let frequencies = try QwenVisionSourceRotaryArithmetic.inverseFrequencies(headDimension: config.headDimension)
            guard let buffer = context.device.makeBuffer(
                bytes: frequencies, length: frequencies.count * MemoryLayout<Float>.stride,
                options: .storageModeShared) else {
                throw QwenVisionError.commandBufferUnavailable
            }
            sourceInverseFrequencyBuffer = buffer
            sourceAttentionScale = Float(pow(Double(config.headDimension), -0.5))
            sourceMergerGELUPipeline = try Self.safePipeline(context: context, name: "qwen_source_vision_gelu_in_place", preciseFunctions: true)
            sourceFC1GELUPipeline = try Self.safePipeline(context: context, name: "qwen_source_vision_gelu_tanh_in_place", preciseFunctions: true)
            rotatePipeline = try Self.safePipeline(context: context, name: "qwen_source_vision_rotate_qk", preciseFunctions: true)
        } else {
            // The synthetic source-store seam retains its original small geometry.
            sourceInverseFrequencyBuffer = nil
            sourceAttentionScale = nil
            sourceMergerGELUPipeline = nil
            sourceFC1GELUPipeline = nil
            rotatePipeline = try Self.safePipeline(context: context, name: "qwen_vision_rotate_qk", preciseFunctions: true)
        }
        attentionPipeline = try Self.safePipeline(context: context, name: config.matchesOfficialWeightLayout ? "qwen_source_vision_attention" : "qwen_vision_attention", preciseFunctions: true)
        residualPipeline = try Self.safePipeline(context: context, name: "qwen_vision_linear_residual", preciseFunctions: true)
        linearTanhGELUPipeline = try Self.safePipeline(context: context, name: config.matchesOfficialWeightLayout ? "qwen_source_vision_linear_gelu_tanh" : "qwen_vision_linear_gelu_tanh", preciseFunctions: true)
        mergerPackPipeline = try Self.safePipeline(context: context, name: "qwen_source_vision_merger_norm_pack", preciseFunctions: true)
        linearGELUPipeline = try Self.safePipeline(context: context, name: "qwen_vision_linear_gelu", preciseFunctions: true)
        linearPipeline = try Self.safePipeline(context: context, name: "qwen_vision_linear", preciseFunctions: true)
    }

    private static func safePipeline(
        context: MetalContext, name: String, preciseFunctions: Bool = false
    ) throws -> MTLComputePipelineState {
        let library = try MetalContext.privateLibrary(
            device: context.device, module: "qwen_vision", mathMode: .safe,
            mathFloatingPointFunctions: preciseFunctions ? .precise : nil,
            includeQwenSourceMath: preciseFunctions)
        guard let function = library.makeFunction(name: name) else {
            throw MetalError.missingFunction(name)
        }
        return try context.device.makeComputePipelineState(function: function)
    }

    func process(_ pixels: QwenVisionPixelBuffer) async throws -> QwenVisionFeatures {
        guard !processing else {
            throw QwenVisionError.commandFailed("vision process already active")
        }
        processing = true
        defer { processing = false }
        return try await execute(pixels)
    }

    func process(_ images: [QwenVisionPixelBuffer]) async throws -> [QwenVisionFeatures] {
        guard !processing else {
            throw QwenVisionError.commandFailed("vision process already active")
        }
        processing = true
        defer { processing = false }
        try QwenImagePreprocessor.preflight(images.map(\.geometry), limits: limits)
        var result: [QwenVisionFeatures] = []
        result.reserveCapacity(images.count)
        for image in images { result.append(try await execute(image)) }
        return result
    }

    /// Executes the frozen small P3 tower using Float32 fixture weights. This
    /// is a numerical proof seam, not a production-pack bypass.
    func executeFixture(
        patches: [Float],
        positions: [SIMD2<Int32>],
        gridHeight: Int,
        gridWidth: Int,
        weights: [String: [Float]]
    ) async throws -> QwenVisionFixtureResult {
        guard !processing, store == nil, sourceStore == nil,
              positions.count == gridHeight * gridWidth,
              patches.count == positions.count * config.patchWidth else {
            throw QwenVisionError.invalidFeatureShape
        }
        processing = true
        defer { processing = false }
        let geometry = try QwenImageGeometry(
            sourceWidth: gridWidth * config.patchSize,
            sourceHeight: gridHeight * config.patchSize,
            config: config)
        func floatBuffer(_ values: [Float], label: String) throws -> MTLBuffer {
            let bytes = values.count * MemoryLayout<Float>.stride
            guard bytes > 0, let buffer = context.device.makeBuffer(
                bytes: values, length: bytes, options: .storageModeShared) else {
                throw QwenVisionError.allocationFailed(name: label)
            }
            buffer.label = label
            return buffer
        }
        let patchBuffer = try floatBuffer(patches, label: "qwen.fixture.patches")
        let positionBytes = positions.count * MemoryLayout<SIMD2<Int32>>.stride
        guard let positionBuffer = context.device.makeBuffer(
            bytes: positions, length: positionBytes, options: .storageModeShared) else {
            throw QwenVisionError.allocationFailed(name: "fixture positions")
        }
        var canonical: [String: MTLBuffer] = [:]
        for (name, values) in weights {
            canonical["model.visual.\(name)"] = try floatBuffer(
                values, label: "qwen.fixture.\(name)")
        }
        let source = QwenVisionTensorSource.fixture(canonical)
        let pixels = QwenVisionPixelBuffer(
            patchesBF16: patchBuffer, positionsInt32x2: positionBuffer,
            metadata: VisionImageMetadata(
                encodedBytes: 1, encodedWidth: geometry.processedWidth,
                encodedHeight: geometry.processedHeight,
                orientedWidth: geometry.processedWidth,
                orientedHeight: geometry.processedHeight, orientation: 1,
                bitsPerComponent: 8, colorModel: "RGB", typeIdentifier: "fixture"),
            geometry: geometry, imageDigest: String(repeating: "0", count: 64),
            wallNanoseconds: 0, allocatedBytes: patchBuffer.length + positionBuffer.length)
        let scratch = try QwenVisionScratch(
            device: context.device, rows: geometry.patchRows, config: config,
            limits: limits)
        executionHooks.observe(.scratchAllocated(bytes: scratch.allocationPlan.totalBytes))
        defer { executionHooks.observe(.scratchReleased) }
        var diagnostics = QwenVisionRuntimeDiagnostics(
            mutableRequestedBytes: scratch.allocationPlan.totalBytes,
            mutableHighWaterBytes: scratch.allocationPlan.totalBytes)
        try await submit(stage: "fixture patch", diagnostics: &diagnostics) { command in
            var parameters = self.parameters(
                rows: geometry.patchRows, paddedRows: scratch.paddedRows,
                inputWidth: config.patchWidth, outputWidth: config.hiddenSize,
                geometry: geometry, weightScalarBytes: 4, patchScalarBytes: 4)
            let encoder = try self.encoder(command, pipeline: self.patchPipeline)
            encoder.setBytes(&parameters, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
            encoder.setBuffer(patchBuffer, offset: 0, index: 1)
            encoder.setBuffer(positionBuffer, offset: 0, index: 2)
            try self.bind(source, "model.visual.patch_embed.proj.weight", encoder, 3)
            try self.bind(source, "model.visual.patch_embed.proj.bias", encoder, 4)
            try self.bind(source, "model.visual.pos_embed.weight", encoder, 5)
            encoder.setBuffer(scratch.hiddenA, offset: 0, index: 6)
            self.dispatch(encoder, self.patchPipeline,
                          scratch.paddedRows * config.hiddenSize)
            encoder.endEncoding()
        }
        for layer in 0..<config.depth {
            try await submit(stage: "fixture block \(layer)", diagnostics: &diagnostics) { command in
                try self.encodeBlock(
                    layer: layer, pixels: pixels, scratch: scratch,
                    source: source, command: command, weightScalarBytes: 4)
            }
            diagnostics.orderedLayerIntermediates.append(Self.read(
                scratch.hiddenA, count: geometry.patchRows * config.hiddenSize))
        }
        try await submit(stage: "fixture merger", diagnostics: &diagnostics) { command in
            try self.encodeMerger(
                pixels: pixels, scratch: scratch, source: source,
                command: command, weightScalarBytes: 4)
        }
        return QwenVisionFixtureResult(
            output: Self.read(
                scratch.outputFloat32,
                count: geometry.mergedRows * config.outputHiddenSize),
            orderedLayerIntermediates: diagnostics.orderedLayerIntermediates,
            allocationPlan: scratch.allocationPlan)
    }

    private func execute(_ pixels: QwenVisionPixelBuffer) async throws -> QwenVisionFeatures {
        guard pixels.geometry.patchRows <= config.maximumPatchRows,
              pixels.geometry.mergedRows <= limits.maximumVisibleHistoryRows else {
            throw QwenVisionError.mergedRowQuotaExceeded(
                requested: pixels.geometry.mergedRows,
                maximum: limits.maximumVisibleHistoryRows)
        }
        if (executionHooks.initialStageDiagnostics != nil
            || executionHooks.firstBlockStageDiagnostics != nil), sourceStore == nil {
            throw QwenVisionError.commandFailed("initial stage diagnostic requires the source store")
        }
        if executionHooks.mergerDiagnostics != nil {
            guard executionHooks.mergerDiagnosticRow >= 0,
                  executionHooks.mergerDiagnosticRow < pixels.geometry.mergedRows else {
                throw QwenVisionError.commandFailed("merger diagnostic row is out of bounds")
            }
        }
        let retainedFeatureBytes = pixels.geometry.mergedRows
            * (config.outputHiddenSize * MemoryLayout<Float>.stride
               + 3 * MemoryLayout<Int32>.stride)
        let retainedBytes = retainedFeatureBytes
            + pixels.patchesBF16.length + pixels.positionsInt32x2.length
        let scratch = try QwenVisionScratch(
            device: context.device, rows: pixels.geometry.patchRows,
            config: config, retainedRequestedBytes: retainedBytes,
            sourceFC1Staging: sourceFC1GELUPipeline != nil,
            limits: limits)
        executionHooks.observe(.scratchAllocated(bytes: scratch.allocationPlan.totalBytes))
        defer { executionHooks.observe(.scratchReleased) }
        var diagnostics = QwenVisionRuntimeDiagnostics(
            mutableRequestedBytes: scratch.allocationPlan.totalBytes,
            mutableHighWaterBytes: scratch.allocationPlan.totalBytes)

        guard store != nil || sourceStore != nil else {
            throw QwenVisionError.commandFailed("verified vision store is unavailable")
        }
        var patchLease: QwenVisionMappedGroupLease? = try mapGroup(.patchAndPosition)
        executionHooks.observe(.groupMapped(
            .patchAndPosition, bytes: patchLease!.diagnostic.pageAlignedMappedBytes))
        do {
            try await submit(stage: "patch+position", diagnostics: &diagnostics) { command in
            guard let lease = patchLease else { throw QwenVisionError.commandFailed("patch lease") }
            var parameters = self.parameters(
                rows: pixels.geometry.patchRows, paddedRows: scratch.paddedRows,
                inputWidth: config.patchWidth, outputWidth: config.hiddenSize,
                geometry: pixels.geometry)
            let encoder = try self.encoder(command, pipeline: self.patchPipeline)
            encoder.setBytes(&parameters, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
            encoder.setBuffer(pixels.patchesBF16, offset: 0, index: 1)
            encoder.setBuffer(pixels.positionsInt32x2, offset: 0, index: 2)
            try self.bind(lease, "model.visual.patch_embed.proj.weight", encoder, 3)
            try self.bind(lease, "model.visual.patch_embed.proj.bias", encoder, 4)
            try self.bind(lease, "model.visual.pos_embed.weight", encoder, 5)
            encoder.setBuffer(scratch.hiddenA, offset: 0, index: 6)
            self.dispatch(encoder, self.patchPipeline, scratch.paddedRows * config.hiddenSize)
                encoder.endEncoding()
            }
            if let observer = executionHooks.initialStageDiagnostics {
                for (label, component) in [("patch.projection", UInt32(1)), ("patch.position", UInt32(2))] {
                    try await submit(stage: label, diagnostics: &diagnostics) { command in
                        guard let lease = patchLease else {
                            throw QwenVisionError.commandFailed("patch lease")
                        }
                        var parameters = self.parameters(
                            rows: pixels.geometry.patchRows, paddedRows: scratch.paddedRows,
                            inputWidth: config.patchWidth, outputWidth: config.hiddenSize,
                            geometry: pixels.geometry)
                        parameters.reserved = component
                        let encoder = try self.encoder(command, pipeline: self.patchPipeline)
                        encoder.setBytes(&parameters, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
                        encoder.setBuffer(pixels.patchesBF16, offset: 0, index: 1)
                        encoder.setBuffer(pixels.positionsInt32x2, offset: 0, index: 2)
                        try self.bind(lease, "model.visual.patch_embed.proj.weight", encoder, 3)
                        try self.bind(lease, "model.visual.patch_embed.proj.bias", encoder, 4)
                        try self.bind(lease, "model.visual.pos_embed.weight", encoder, 5)
                        encoder.setBuffer(scratch.hiddenB, offset: 0, index: 6)
                        self.dispatch(encoder, self.patchPipeline, scratch.paddedRows * config.hiddenSize)
                        encoder.endEncoding()
                    }
                    observer(QwenVisionTowerDiagnostic(
                        stage: label, rows: pixels.geometry.patchRows, width: config.hiddenSize,
                        values: Self.read(scratch.hiddenB, count: pixels.geometry.patchRows * config.hiddenSize)))
                }
                observer(QwenVisionTowerDiagnostic(
                    stage: "patch.sum", rows: pixels.geometry.patchRows, width: config.hiddenSize,
                    values: Self.read(scratch.hiddenA, count: pixels.geometry.patchRows * config.hiddenSize)))
                throw QwenVisionError.commandFailed("source vision diagnostic stopped after initial stages")
            }
        } catch {
            patchLease = nil
            executionHooks.observe(.groupReleased(.patchAndPosition))
            throw error
        }
        diagnostics.orderedSubmittedGroups.append(.patchAndPosition)
        diagnostics.mappedPageAlignedBytes.append(patchLease!.diagnostic.pageAlignedMappedBytes)
        diagnostics.maximumLiveMappedBytes = max(
            diagnostics.maximumLiveMappedBytes,
            patchLease!.diagnostic.pageAlignedMappedBytes)
        patchLease = nil
        executionHooks.observe(.groupReleased(.patchAndPosition))

        observeTower(stage: "tower.patch-position", buffer: scratch.hiddenA,
                     rows: pixels.geometry.patchRows, width: config.hiddenSize)
        if executionHooks.firstBlockDiagnostics != nil {
            try observeDiagnosticStage("patch-position", buffer: scratch.hiddenA,
                                       count: pixels.geometry.patchRows * config.hiddenSize)
        }

        for layer in 0..<config.depth {
            var lease: QwenVisionMappedGroupLease? = try mapGroup(.block(layer))
            executionHooks.observe(.groupMapped(
                .block(layer), bytes: lease!.diagnostic.pageAlignedMappedBytes))
            do {
                if (executionHooks.firstBlockDiagnostics != nil
                    || executionHooks.firstBlockStageDiagnostics != nil), layer == 0 {
                    observeFirstBlockStage("block-0.input", buffer: scratch.hiddenA,
                                           rows: pixels.geometry.patchRows, width: config.hiddenSize)
                    let stages: [(String, MTLBuffer, Int, Int)] = [
                        ("norm1", scratch.attention, config.hiddenSize, 0),
                        ("qkv", scratch.q, 3 * config.hiddenSize, 1),
                        ("rotate-qk", scratch.q, 3 * config.hiddenSize, 2),
                        ("attention", scratch.attention, config.hiddenSize, 3),
                        ("attention-residual", scratch.hiddenB, config.hiddenSize, 4),
                        ("norm2", scratch.attention, config.hiddenSize, 5),
                        ("mlp-pre-gelu", scratch.gate, config.intermediateSize, 8),
                        ("mlp-gelu", scratch.gate, config.intermediateSize, 6),
                        ("mlp-residual", scratch.hiddenA, config.hiddenSize, 7),
                    ]
                    for stage in stages {
                        let label = "block-0.\(stage.0)"
                        let stageIndex = stage.3
                        if stageIndex == 3, sourceAttentionScale != nil, scratch.paddedRows > 128 {
                            guard let lease else { throw QwenVisionError.commandFailed("block lease") }
                            try await submitSourceAttentionChunks(
                                layer: layer, pixels: pixels, scratch: scratch,
                                source: .mapped(lease), diagnostics: &diagnostics)
                        } else if (stageIndex == 6 || stageIndex == 8), sourceFC1GELUPipeline != nil {
                            guard let lease else { throw QwenVisionError.commandFailed("block lease") }
                            try await submitSourceFC1(
                                layer: layer, pixels: pixels, scratch: scratch,
                                source: .mapped(lease), applyGELU: stageIndex == 6,
                                diagnostics: &diagnostics)
                        } else {
                            try await submit(stage: label, diagnostics: &diagnostics) { command in
                                guard let lease else { throw QwenVisionError.commandFailed("block lease") }
                                try self.encodeBlock(layer: layer, pixels: pixels, scratch: scratch,
                                                     source: .mapped(lease), command: command,
                                                     onlyStage: stageIndex)
                            }
                        }
                        try observeDiagnosticStage(label, buffer: stage.1,
                                                   count: pixels.geometry.patchRows * stage.2)
                        let arrayLabel: String
                        switch stageIndex {
                        case 3: arrayLabel = "block-0.attention-context"
                        case 8: arrayLabel = "block-0.fc1-pre-gelu"
                        case 6: arrayLabel = "block-0.fc1-gelu"
                        case 7: arrayLabel = "block-0.output"
                        default: arrayLabel = label
                        }
                        observeFirstBlockStage(arrayLabel, buffer: stage.1,
                                               rows: pixels.geometry.patchRows, width: stage.2)
                    }
                    throw QwenVisionError.commandFailed("source vision diagnostic stopped after block-0")
                }
                if sourceFC1GELUPipeline != nil {
                    guard let lease else { throw QwenVisionError.commandFailed("block lease") }
                    try await submitSourceBlock(
                        layer: layer, pixels: pixels, scratch: scratch,
                        source: .mapped(lease), diagnostics: &diagnostics)
                } else {
                    try await submit(stage: "block \(layer)", diagnostics: &diagnostics) { command in
                        guard let lease else { throw QwenVisionError.commandFailed("block lease") }
                        try self.encodeBlock(
                            layer: layer, pixels: pixels, scratch: scratch,
                            source: .mapped(lease), command: command)
                    }
                }
            } catch {
                lease = nil
                executionHooks.observe(.groupReleased(.block(layer)))
                throw error
            }
            observeTower(stage: "tower.block.\(layer)", buffer: scratch.hiddenA,
                         rows: pixels.geometry.patchRows, width: config.hiddenSize)
            diagnostics.orderedSubmittedGroups.append(.block(layer))
            diagnostics.mappedPageAlignedBytes.append(lease!.diagnostic.pageAlignedMappedBytes)
            diagnostics.maximumLiveMappedBytes = max(
                diagnostics.maximumLiveMappedBytes,
                lease!.diagnostic.pageAlignedMappedBytes)
            diagnostics.orderedLayerIntermediates.append(
                Self.orderedLayerSignature(
                    scratch.hiddenA,
                    count: pixels.geometry.patchRows * config.hiddenSize,
                    layer: layer))
            lease = nil
            executionHooks.observe(.groupReleased(.block(layer)))
        }

        var mergerLease: QwenVisionMappedGroupLease? = try mapGroup(.merger)
        executionHooks.observe(.groupMapped(
            .merger, bytes: mergerLease!.diagnostic.pageAlignedMappedBytes))
        do {
            if executionHooks.mergerDiagnostics != nil {
                observeMerger(stage: "merger.input", buffer: scratch.hiddenA,
                              width: config.mergedHiddenSize)
                let stages: [(String, Int)] = [
                    ("merger.norm-packed", 0), ("merger.fc1-pre-gelu", 3),
                    ("merger.fc1-gelu", 1), ("merger.output", 2)]
                for (label, stageIndex) in stages {
                    try await submit(stage: label, diagnostics: &diagnostics) { command in
                        guard let lease = mergerLease else {
                            throw QwenVisionError.commandFailed("merger lease")
                        }
                        try self.encodeMerger(
                            pixels: pixels, scratch: scratch, source: .mapped(lease),
                            command: command, onlyStage: stageIndex)
                    }
                    let buffer = stageIndex == 0 ? scratch.mergerPacked
                        : stageIndex == 2 ? scratch.outputFloat32 : scratch.mergerNormalized
                    observeMerger(stage: label, buffer: buffer,
                                  width: stageIndex == 2 ? config.outputHiddenSize : config.mergedHiddenSize)
                }
            } else {
                try await submit(stage: "merger", diagnostics: &diagnostics) { command in
                    guard let lease = mergerLease else {
                        throw QwenVisionError.commandFailed("merger lease")
                    }
                    try self.encodeMerger(
                        pixels: pixels, scratch: scratch,
                        source: .mapped(lease), command: command)
                }
            }
        } catch {
            mergerLease = nil
            executionHooks.observe(.groupReleased(.merger))
            throw error
        }
        diagnostics.orderedSubmittedGroups.append(.merger)
        diagnostics.mappedPageAlignedBytes.append(mergerLease!.diagnostic.pageAlignedMappedBytes)
        diagnostics.maximumLiveMappedBytes = max(
            diagnostics.maximumLiveMappedBytes,
            mergerLease!.diagnostic.pageAlignedMappedBytes)
        mergerLease = nil
        executionHooks.observe(.groupReleased(.merger))
        try sourceStore?.revalidate()

        observeTower(stage: "merger.output", buffer: scratch.outputFloat32,
                     rows: pixels.geometry.mergedRows, width: config.outputHiddenSize)
        let values = Self.read(
            scratch.outputFloat32,
            count: pixels.geometry.mergedRows * config.outputHiddenSize)
        let grid = try QwenVisionGrid(
            temporal: pixels.geometry.gridT,
            height: pixels.geometry.gridH,
            width: pixels.geometry.gridW)
        let mergedHeight = grid.height / config.spatialMergeSize
        let mergedWidth = grid.width / config.spatialMergeSize
        var positions: [QwenMRoPEPosition] = []
        positions.reserveCapacity(pixels.geometry.mergedRows)
        for temporal in 0..<grid.temporal {
            for height in 0..<mergedHeight {
                for width in 0..<mergedWidth {
                    positions.append(try QwenMRoPEPosition(
                        temporal: temporal, height: height, width: width))
                }
            }
        }
        guard let processorDigest = sourceStore?.descriptor.processorConfigSHA256
                ?? store?.manifest.files[GTurboVisionFormatV2.processorFile]?.sha256,
              let profile = sourceStore?.descriptor.processorProfile
                ?? store?.manifest.processorProfile else {
            throw QwenVisionError.commandFailed("processor digest missing")
        }
        try sourceStore?.revalidate()
        return try QwenVisionFeatures(
            device: context.device, features: values, positions: positions,
            imageDigest: pixels.imageDigest,
            processorDigest: processorDigest,
            profile: profile,
            grid: grid, diagnostics: diagnostics,
            hiddenSize: config.outputHiddenSize)
    }

    private func mapGroup(_ group: QwenVisionWeightGroup) throws -> QwenVisionMappedGroupLease {
        if let sourceStore { return try sourceStore.mapGroup(group, device: context.device) }
        if let store { return try store.mapGroup(group, device: context.device) }
        throw QwenVisionError.commandFailed("verified vision store is unavailable")
    }

    /// A completed command between query ranges bounds the source attention
    /// workload without changing its full-key reduction or allocating buffers.
    private func submitSourceAttentionChunks(
        layer: Int,
        pixels: QwenVisionPixelBuffer,
        scratch: QwenVisionScratch,
        source: QwenVisionTensorSource,
        diagnostics: inout QwenVisionRuntimeDiagnostics
    ) async throws {
        for queryBase in stride(from: 0, to: scratch.paddedRows, by: 128) {
            let queryRows = min(128, scratch.paddedRows - queryBase)
            try await submit(stage: "block \(layer).attention.\(queryBase)", diagnostics: &diagnostics) { command in
                try self.encodeBlock(
                    layer: layer, pixels: pixels, scratch: scratch, source: source,
                    command: command, onlyStage: 3,
                    attentionQueryBase: queryBase, attentionQueryRows: queryRows)
            }
        }
    }

    private func submitSourceBlock(
        layer: Int,
        pixels: QwenVisionPixelBuffer,
        scratch: QwenVisionScratch,
        source: QwenVisionTensorSource,
        diagnostics: inout QwenVisionRuntimeDiagnostics
    ) async throws {
        try await submit(stage: "block \(layer).before-attention", diagnostics: &diagnostics) { command in
            for stage in 0...2 {
                try self.encodeBlock(layer: layer, pixels: pixels, scratch: scratch,
                                     source: source, command: command, onlyStage: stage)
            }
        }
        try await submitSourceAttentionChunks(
            layer: layer, pixels: pixels, scratch: scratch,
            source: source, diagnostics: &diagnostics)
        try await submit(stage: "block \(layer).after-attention", diagnostics: &diagnostics) { command in
            for stage in 4...5 {
                try self.encodeBlock(layer: layer, pixels: pixels, scratch: scratch,
                                     source: source, command: command, onlyStage: stage)
            }
        }
        try await submitSourceFC1(
            layer: layer, pixels: pixels, scratch: scratch, source: source,
            applyGELU: true, diagnostics: &diagnostics)
        try await submit(stage: "block \(layer).after-fc1", diagnostics: &diagnostics) { command in
            try self.encodeBlock(layer: layer, pixels: pixels, scratch: scratch,
                                 source: source, command: command, onlyStage: 7)
        }
    }

    /// The preceding norm2 command has completed before the shared activation
    /// buffers are read by Accelerate. The enclosing block retains its lease.
    private func submitSourceFC1(
        layer: Int, pixels: QwenVisionPixelBuffer, scratch: QwenVisionScratch,
        source: QwenVisionTensorSource, applyGELU: Bool,
        diagnostics: inout QwenVisionRuntimeDiagnostics
    ) async throws {
        guard case .mapped(let lease) = source,
              let staging = scratch.sourceFC1Staging,
              let pipeline = sourceFC1GELUPipeline else {
            throw QwenVisionError.commandFailed("source FC1 staging unavailable")
        }
        let prefix = "model.visual.blocks.\(layer).mlp.linear_fc1."
        try QwenSourceVisionLinear.project(
            rows: pixels.geometry.patchRows, inputWidth: config.hiddenSize,
            outputWidth: config.intermediateSize, input: scratch.attention,
            weight: lease.buffer, weightOffset: try lease.offset(of: prefix + "weight"),
            bias: lease.buffer, biasOffset: try lease.offset(of: prefix + "bias"),
            output: scratch.gate, staging: staging)
        let paddingStart = pixels.geometry.patchRows * config.intermediateSize
        let paddingCount = (scratch.paddedRows - pixels.geometry.patchRows) * config.intermediateSize
        scratch.gate.contents().assumingMemoryBound(to: Float.self)
            .advanced(by: paddingStart).update(repeating: 0, count: paddingCount)
        try Task.checkCancellation()
        guard applyGELU else { return }
        try await submit(stage: "block \(layer).mlp-gelu", diagnostics: &diagnostics) { command in
            var parameters = self.parameters(
                rows: pixels.geometry.patchRows, paddedRows: scratch.paddedRows,
                inputWidth: config.hiddenSize, outputWidth: config.intermediateSize,
                geometry: pixels.geometry)
            let encoder = try self.encoder(command, pipeline: pipeline)
            encoder.setBytes(&parameters, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
            encoder.setBuffer(scratch.gate, offset: 0, index: 1)
            self.dispatch(encoder, pipeline, pixels.geometry.patchRows * config.intermediateSize)
            encoder.endEncoding()
        }
    }

    private func encodeBlock(
        layer: Int,
        pixels: QwenVisionPixelBuffer,
        scratch: QwenVisionScratch,
        source: QwenVisionTensorSource,
        command: MTLCommandBuffer,
        weightScalarBytes: UInt32 = 2,
        onlyStage: Int? = nil,
        attentionQueryBase: Int = 0,
        attentionQueryRows: Int? = nil
    ) throws {
        let prefix = "model.visual.blocks.\(layer)."
        var normalized = parameters(
            rows: pixels.geometry.patchRows, paddedRows: scratch.paddedRows,
            inputWidth: config.hiddenSize, outputWidth: config.hiddenSize,
            geometry: pixels.geometry, weightScalarBytes: weightScalarBytes)
        var encoder: MTLComputeCommandEncoder
        if onlyStage == nil || onlyStage == 0 {
        encoder = try self.encoder(command, pipeline: normPipeline)
        encoder.setBytes(&normalized, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(scratch.hiddenA, offset: 0, index: 1)
        try bind(source, prefix + "norm1.weight", encoder, 2)
        try bind(source, prefix + "norm1.bias", encoder, 3)
        encoder.setBuffer(scratch.attention, offset: 0, index: 4)
        dispatch(encoder, normPipeline, scratch.paddedRows)
        encoder.endEncoding()
        }

        var qkv = parameters(
            rows: pixels.geometry.patchRows, paddedRows: scratch.paddedRows,
            inputWidth: config.hiddenSize, outputWidth: 3 * config.hiddenSize,
            geometry: pixels.geometry, weightScalarBytes: weightScalarBytes)
        if onlyStage == nil || onlyStage == 1 {
        encoder = try self.encoder(command, pipeline: linearPipeline)
        encoder.setBytes(&qkv, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(scratch.attention, offset: 0, index: 1)
        try bind(source, prefix + "attn.qkv.weight", encoder, 2)
        try bind(source, prefix + "attn.qkv.bias", encoder, 3)
        encoder.setBuffer(scratch.q, offset: 0, index: 4)
        dispatch(encoder, linearPipeline,
                 pixels.geometry.patchRows * 3 * config.hiddenSize)
        encoder.endEncoding()
        }

        var rotate = parameters(
            rows: pixels.geometry.patchRows, paddedRows: scratch.paddedRows,
            inputWidth: config.hiddenSize, outputWidth: config.hiddenSize,
            geometry: pixels.geometry, weightScalarBytes: weightScalarBytes)
        if onlyStage == nil || onlyStage == 2 {
        encoder = try self.encoder(command, pipeline: rotatePipeline)
        encoder.setBytes(&rotate, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(pixels.positionsInt32x2, offset: 0, index: 1)
        encoder.setBuffer(scratch.q, offset: 0, index: 2)
        if let sourceInverseFrequencyBuffer {
            encoder.setBuffer(sourceInverseFrequencyBuffer, offset: 0, index: 3)
        }
        dispatch(encoder, rotatePipeline, pixels.geometry.patchRows * config.hiddenSize)
        encoder.endEncoding()
        }

        if onlyStage == nil || onlyStage == 3 {
        var attention = rotate
        attention.reserved = UInt32(attentionQueryBase)
        encoder = try self.encoder(command, pipeline: attentionPipeline)
        encoder.setBytes(&attention, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(scratch.q, offset: 0, index: 1)
        encoder.setBuffer(scratch.attention, offset: 0, index: 2)
        if var sourceAttentionScale {
            encoder.setBytes(&sourceAttentionScale, length: MemoryLayout<Float>.stride, index: 3)
        }
        dispatch(encoder, attentionPipeline, (attentionQueryRows ?? scratch.paddedRows) * config.numHeads)
        encoder.endEncoding()
        }

        if onlyStage == nil || onlyStage == 4 {
        encoder = try self.encoder(command, pipeline: residualPipeline)
        encoder.setBytes(&rotate, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(scratch.attention, offset: 0, index: 1)
        encoder.setBuffer(scratch.hiddenA, offset: 0, index: 2)
        try bind(source, prefix + "attn.proj.weight", encoder, 3)
        try bind(source, prefix + "attn.proj.bias", encoder, 4)
        encoder.setBuffer(scratch.hiddenB, offset: 0, index: 5)
        dispatch(encoder, residualPipeline, scratch.paddedRows * config.hiddenSize)
        encoder.endEncoding()
        }

        if onlyStage == nil || onlyStage == 5 {
        encoder = try self.encoder(command, pipeline: normPipeline)
        encoder.setBytes(&normalized, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(scratch.hiddenB, offset: 0, index: 1)
        try bind(source, prefix + "norm2.weight", encoder, 2)
        try bind(source, prefix + "norm2.bias", encoder, 3)
        encoder.setBuffer(scratch.attention, offset: 0, index: 4)
        dispatch(encoder, normPipeline, scratch.paddedRows)
        encoder.endEncoding()
        }

        var mlp = parameters(
            rows: pixels.geometry.patchRows, paddedRows: scratch.paddedRows,
            inputWidth: config.hiddenSize, outputWidth: config.intermediateSize,
            geometry: pixels.geometry, weightScalarBytes: weightScalarBytes)
        if onlyStage == nil || onlyStage == 6 || onlyStage == 8 {
        let mlpPipeline = onlyStage == 8 ? linearPipeline : linearTanhGELUPipeline
        encoder = try self.encoder(command, pipeline: mlpPipeline)
        encoder.setBytes(&mlp, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(scratch.attention, offset: 0, index: 1)
        try bind(source, prefix + "mlp.linear_fc1.weight", encoder, 2)
        try bind(source, prefix + "mlp.linear_fc1.bias", encoder, 3)
        encoder.setBuffer(scratch.gate, offset: 0, index: 4)
        dispatch(encoder, mlpPipeline,
                 pixels.geometry.patchRows * config.intermediateSize)
        encoder.endEncoding()
        }

        var down = parameters(
            rows: pixels.geometry.patchRows, paddedRows: scratch.paddedRows,
            inputWidth: config.intermediateSize, outputWidth: config.hiddenSize,
            geometry: pixels.geometry, weightScalarBytes: weightScalarBytes)
        if onlyStage == nil || onlyStage == 7 {
        encoder = try self.encoder(command, pipeline: residualPipeline)
        encoder.setBytes(&down, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(scratch.gate, offset: 0, index: 1)
        encoder.setBuffer(scratch.hiddenB, offset: 0, index: 2)
        try bind(source, prefix + "mlp.linear_fc2.weight", encoder, 3)
        try bind(source, prefix + "mlp.linear_fc2.bias", encoder, 4)
        encoder.setBuffer(scratch.hiddenA, offset: 0, index: 5)
        dispatch(encoder, residualPipeline, scratch.paddedRows * config.hiddenSize)
        encoder.endEncoding()
        }
    }

    private func observeDiagnosticStage(_ stage: String, buffer: MTLBuffer, count: Int) throws {
        guard let observe = executionHooks.firstBlockDiagnostics else { return }
        let values = buffer.contents().assumingMemoryBound(to: Float.self)
        var finite = 0, nan = 0, positiveInfinity = 0, negativeInfinity = 0
        var bad: [Int] = []
        var minimum: Float?, maximum: Float?
        for index in 0..<count {
            let value = values[index]
            if value.isFinite {
                finite += 1
                minimum = minimum.map { min($0, value) } ?? value
                maximum = maximum.map { max($0, value) } ?? value
            } else {
                if bad.count < 16 { bad.append(index) }
                if value.isNaN { nan += 1 }
                else if value > 0 { positiveInfinity += 1 }
                else { negativeInfinity += 1 }
            }
        }
        let sampleIndices = Array(0..<min(count, 16))
            + Array(max(0, count - 16)..<count) + bad
        observe(QwenVisionStageDiagnostic(
            stage: stage, count: count, finiteCount: finite, nanCount: nan,
            positiveInfinityCount: positiveInfinity, negativeInfinityCount: negativeInfinity,
            firstNonfiniteIndex: bad.first, firstNonfiniteIndices: bad,
            sampleBits: sampleIndices.map { values[$0].bitPattern },
            finiteMinimum: minimum, finiteMaximum: maximum))
        if finite != count {
            throw QwenVisionError.commandFailed("source vision diagnostic stopped after \(stage)")
        }
    }

    private func observeFirstBlockStage(_ stage: String, buffer: MTLBuffer, rows: Int, width: Int) {
        guard let observer = executionHooks.firstBlockStageDiagnostics else { return }
        observer(QwenVisionTowerDiagnostic(
            stage: stage, rows: rows, width: width,
            values: Self.read(buffer, count: rows * width)))
    }

    private func observeTower(stage: String, buffer: MTLBuffer, rows: Int, width: Int) {
        guard let observer = executionHooks.towerDiagnostics else { return }
        observer(QwenVisionTowerDiagnostic(
            stage: stage, rows: rows, width: width,
            values: Self.read(buffer, count: rows * width)))
    }

    private func observeMerger(stage: String, buffer: MTLBuffer, width: Int) {
        guard let observer = executionHooks.mergerDiagnostics else { return }
        let row = executionHooks.mergerDiagnosticRow
        let values = buffer.contents().assumingMemoryBound(to: Float.self)
            .advanced(by: row * width)
        observer(QwenVisionMergerDiagnostic(
            stage: stage, row: row,
            values: Array(UnsafeBufferPointer(start: values, count: width))))
    }

    private func encodeMerger(
        pixels: QwenVisionPixelBuffer,
        scratch: QwenVisionScratch,
        source: QwenVisionTensorSource,
        command: MTLCommandBuffer,
        weightScalarBytes: UInt32 = 2,
        onlyStage: Int? = nil
    ) throws {
        var pack = parameters(
            rows: pixels.geometry.patchRows, paddedRows: scratch.paddedRows,
            inputWidth: config.hiddenSize, outputWidth: config.mergedHiddenSize,
            geometry: pixels.geometry, weightScalarBytes: weightScalarBytes)
        var encoder: MTLComputeCommandEncoder
        if onlyStage == nil || onlyStage == 0 {
        encoder = try self.encoder(command, pipeline: mergerPackPipeline)
        encoder.setBytes(&pack, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(scratch.hiddenA, offset: 0, index: 1)
        try bind(source, "model.visual.merger.norm.weight", encoder, 2)
        try bind(source, "model.visual.merger.norm.bias", encoder, 3)
        encoder.setBuffer(scratch.mergerPacked, offset: 0, index: 4)
        dispatch(encoder, mergerPackPipeline, pixels.geometry.patchRows)
        encoder.endEncoding()

        }
        var fc1 = parameters(
            rows: pixels.geometry.mergedRows, paddedRows: pixels.geometry.mergedRows,
            inputWidth: config.mergedHiddenSize, outputWidth: config.mergedHiddenSize,
            geometry: pixels.geometry, weightScalarBytes: weightScalarBytes)
        if onlyStage == nil || onlyStage == 1 || onlyStage == 3 {
        let fc1Pipeline = (onlyStage == 3 || sourceMergerGELUPipeline != nil)
            ? linearPipeline : linearGELUPipeline
        encoder = try self.encoder(command, pipeline: fc1Pipeline)
        encoder.setBytes(&fc1, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(scratch.mergerPacked, offset: 0, index: 1)
        try bind(source, "model.visual.merger.linear_fc1.weight", encoder, 2)
        try bind(source, "model.visual.merger.linear_fc1.bias", encoder, 3)
        encoder.setBuffer(scratch.mergerNormalized, offset: 0, index: 4)
        dispatch(encoder, fc1Pipeline,
                 pixels.geometry.mergedRows * config.mergedHiddenSize)
        encoder.endEncoding()
        if onlyStage != 3, let sourceMergerGELUPipeline {
            encoder = try self.encoder(command, pipeline: sourceMergerGELUPipeline)
            encoder.setBytes(&fc1, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
            encoder.setBuffer(scratch.mergerNormalized, offset: 0, index: 1)
            dispatch(encoder, sourceMergerGELUPipeline,
                     pixels.geometry.mergedRows * config.mergedHiddenSize / 4)
            encoder.endEncoding()
        }

        }
        var fc2 = parameters(
            rows: pixels.geometry.mergedRows, paddedRows: pixels.geometry.mergedRows,
            inputWidth: config.mergedHiddenSize, outputWidth: config.outputHiddenSize,
            geometry: pixels.geometry, weightScalarBytes: weightScalarBytes)
        if onlyStage == nil || onlyStage == 2 {
        encoder = try self.encoder(command, pipeline: linearPipeline)
        encoder.setBytes(&fc2, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(scratch.mergerNormalized, offset: 0, index: 1)
        try bind(source, "model.visual.merger.linear_fc2.weight", encoder, 2)
        try bind(source, "model.visual.merger.linear_fc2.bias", encoder, 3)
        encoder.setBuffer(scratch.outputFloat32, offset: 0, index: 4)
        dispatch(encoder, linearPipeline,
                 pixels.geometry.mergedRows * config.outputHiddenSize)
        encoder.endEncoding()
        }
    }

    private func parameters(
        rows: Int, paddedRows: Int, inputWidth: Int, outputWidth: Int,
        geometry: QwenImageGeometry,
        weightScalarBytes: UInt32 = 2,
        patchScalarBytes: UInt32 = 2
    ) -> QwenVisionKernelParameters {
        QwenVisionKernelParameters(
            rows: UInt32(rows), paddedRows: UInt32(paddedRows),
            inputWidth: UInt32(inputWidth), outputWidth: UInt32(outputWidth),
            intermediateWidth: UInt32(config.intermediateSize),
            heads: UInt32(config.numHeads),
            gridHeight: UInt32(geometry.gridH), gridWidth: UInt32(geometry.gridW),
            mergeSize: UInt32(config.spatialMergeSize),
            positionCount: UInt32(config.numPositionEmbeddings), epsilon: 1e-6,
            weightScalarBytes: weightScalarBytes,
            patchScalarBytes: patchScalarBytes)
    }

    private func bind(
        _ lease: QwenVisionMappedGroupLease,
        _ name: String,
        _ encoder: MTLComputeCommandEncoder,
        _ index: Int
    ) throws {
        encoder.setBuffer(
            lease.buffer, offset: try lease.offset(of: name), index: index)
    }

    private func bind(
        _ source: QwenVisionTensorSource,
        _ name: String,
        _ encoder: MTLComputeCommandEncoder,
        _ index: Int
    ) throws {
        switch source {
        case .mapped(let lease):
            try bind(lease, name, encoder, index)
        case .fixture(let values):
            guard let buffer = values[name] else {
                throw QwenVisionError.commandFailed("fixture tensor missing: \(name)")
            }
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
    }

    private func encoder(
        _ command: MTLCommandBuffer,
        pipeline: MTLComputePipelineState
    ) throws -> MTLComputeCommandEncoder {
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw QwenVisionError.commandBufferUnavailable
        }
        encoder.setComputePipelineState(pipeline)
        return encoder
    }

    private func dispatch(
        _ encoder: MTLComputeCommandEncoder,
        _ pipeline: MTLComputePipelineState,
        _ count: Int
    ) {
        let width = max(1, min(pipeline.maxTotalThreadsPerThreadgroup, 256))
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    private func submit(
        stage: String,
        diagnostics: inout QwenVisionRuntimeDiagnostics,
        encode: (MTLCommandBuffer) throws -> Void
    ) async throws {
        guard let command = context.queue.makeCommandBuffer() else {
            throw QwenVisionError.commandBufferUnavailable
        }
        command.label = "qwen.vision.\(stage)"
        try encode(command)
        command.commit()
        executionHooks.observe(.commandSubmitted(stage))
        await executionHooks.afterCommandSubmission(stage)
        await command.completed()
        guard command.status == .completed, command.error == nil else {
            diagnostics.discardedCommandBuffers += 1
            executionHooks.observe(.commandCompleted(stage, discarded: true))
            throw QwenVisionError.commandFailed(
                command.error?.localizedDescription ?? "status \(command.status.rawValue)")
        }
        diagnostics.completedCommandBuffers += 1
        do { try Task.checkCancellation() }
        catch {
            diagnostics.discardedCommandBuffers += 1
            executionHooks.observe(.commandCompleted(stage, discarded: true))
            throw error
        }
        executionHooks.observe(.commandCompleted(stage, discarded: false))
    }

    /// Compact, order-sensitive observation of every production layer. The
    /// P3 fixture seam separately returns its complete small intermediate.
    private static func orderedLayerSignature(
        _ buffer: MTLBuffer, count: Int, layer: Int
    ) -> [Float] {
        guard count > 0 else { return [Float(layer), 0, 0, 0] }
        let values = buffer.contents().assumingMemoryBound(to: Float.self)
        var recurrence = Float(layer + 1) * 0.000_001
        for index in 0..<count {
            recurrence = recurrence * 0.999_91 + values[index] * 0.000_09
        }
        return [Float(layer), values[0], values[count - 1], recurrence]
    }

    private static func read(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    }
}
