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

    init(
        device: MTLDevice,
        rows: Int,
        config: QwenVisionConfig,
        retainedRequestedBytes: Int = 0,
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
        let entries = try [
            bytes(paddedRows, config.hiddenSize, fp32, "hidden A"),
            bytes(paddedRows, config.hiddenSize, fp32, "hidden B"),
            bytes(paddedRows, 3 * config.hiddenSize, fp32, "QKV workspace"),
            bytes(paddedRows, config.hiddenSize, fp32, "attention workspace"),
            bytes(paddedRows, config.intermediateSize, fp32, "MLP workspace"),
            bytes(mergedRows, config.outputHiddenSize, fp32,
                  "retained Float32 output staging"),
            .init(name: "retained lineage", bytes: retainedRequestedBytes),
        ]
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

    static let none = QwenVisionExecutionHooks(
        afterCommandSubmission: { _ in }, observe: { _ in })
}

actor QwenVisionRuntime {
    private let context: MetalContext
    private let store: QwenVisionWeightStore?
    private let config: QwenVisionConfig
    private let limits: QwenVisionResourceLimits
    private let executionHooks: QwenVisionExecutionHooks
    private let patchPipeline: MTLComputePipelineState
    private let normPipeline: MTLComputePipelineState
    private let rotatePipeline: MTLComputePipelineState
    private let attentionPipeline: MTLComputePipelineState
    private let residualPipeline: MTLComputePipelineState
    private let linearTanhGELUPipeline: MTLComputePipelineState
    private let mergerPackPipeline: MTLComputePipelineState
    private let linearGELUPipeline: MTLComputePipelineState
    private let linearPipeline: MTLComputePipelineState

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

    private static func safePipeline(
        context: MetalContext, name: String
    ) throws -> MTLComputePipelineState {
        let library = try MetalContext.privateLibrary(
            device: context.device, module: "qwen_vision", mathMode: .safe)
        guard let function = library.makeFunction(name: name) else {
            throw MetalError.missingFunction(name)
        }
        return try context.device.makeComputePipelineState(function: function)
    }

    func process(_ pixels: QwenVisionPixelBuffer) async throws -> QwenVisionFeatures {
        try await execute(pixels)
    }

    func process(_ images: [QwenVisionPixelBuffer]) async throws -> [QwenVisionFeatures] {
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
        guard store == nil, positions.count == gridHeight * gridWidth,
              patches.count == positions.count * config.patchWidth else {
            throw QwenVisionError.invalidFeatureShape
        }
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
        let retainedFeatureBytes = pixels.geometry.mergedRows
            * (config.outputHiddenSize * MemoryLayout<Float>.stride
               + 3 * MemoryLayout<Int32>.stride)
        let retainedBytes = retainedFeatureBytes
            + pixels.patchesBF16.length + pixels.positionsInt32x2.length
        let scratch = try QwenVisionScratch(
            device: context.device, rows: pixels.geometry.patchRows,
            config: config, retainedRequestedBytes: retainedBytes,
            limits: limits)
        executionHooks.observe(.scratchAllocated(bytes: scratch.allocationPlan.totalBytes))
        defer { executionHooks.observe(.scratchReleased) }
        var diagnostics = QwenVisionRuntimeDiagnostics(
            mutableRequestedBytes: scratch.allocationPlan.totalBytes,
            mutableHighWaterBytes: scratch.allocationPlan.totalBytes)

        guard let store else {
            throw QwenVisionError.commandFailed("verified vision store is unavailable")
        }
        var patchLease: QwenVisionMappedGroupLease? = try store.mapGroup(
            .patchAndPosition, device: context.device)
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

        for layer in 0..<config.depth {
            var lease: QwenVisionMappedGroupLease? = try store.mapGroup(
                .block(layer), device: context.device)
            executionHooks.observe(.groupMapped(
                .block(layer), bytes: lease!.diagnostic.pageAlignedMappedBytes))
            do {
                try await submit(stage: "block \(layer)", diagnostics: &diagnostics) { command in
                guard let lease else { throw QwenVisionError.commandFailed("block lease") }
                try self.encodeBlock(
                    layer: layer, pixels: pixels, scratch: scratch,
                        source: .mapped(lease), command: command)
                }
            } catch {
                lease = nil
                executionHooks.observe(.groupReleased(.block(layer)))
                throw error
            }
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

        var mergerLease: QwenVisionMappedGroupLease? = try store.mapGroup(
            .merger, device: context.device)
        executionHooks.observe(.groupMapped(
            .merger, bytes: mergerLease!.diagnostic.pageAlignedMappedBytes))
        do {
            try await submit(stage: "merger", diagnostics: &diagnostics) { command in
            guard let lease = mergerLease else { throw QwenVisionError.commandFailed("merger lease") }
            try self.encodeMerger(
                pixels: pixels, scratch: scratch,
                    source: .mapped(lease), command: command)
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
        guard let processor = store.manifest.files[GTurboVisionFormatV2.processorFile] else {
            throw QwenVisionError.commandFailed("processor digest missing")
        }
        return try QwenVisionFeatures(
            device: context.device, features: values, positions: positions,
            imageDigest: pixels.imageDigest,
            processorDigest: processor.sha256,
            profile: store.manifest.processorProfile,
            grid: grid, diagnostics: diagnostics,
            hiddenSize: config.outputHiddenSize)
    }

    private func encodeBlock(
        layer: Int,
        pixels: QwenVisionPixelBuffer,
        scratch: QwenVisionScratch,
        source: QwenVisionTensorSource,
        command: MTLCommandBuffer,
        weightScalarBytes: UInt32 = 2
    ) throws {
        let prefix = "model.visual.blocks.\(layer)."
        var normalized = parameters(
            rows: pixels.geometry.patchRows, paddedRows: scratch.paddedRows,
            inputWidth: config.hiddenSize, outputWidth: config.hiddenSize,
            geometry: pixels.geometry, weightScalarBytes: weightScalarBytes)
        var encoder = try self.encoder(command, pipeline: normPipeline)
        encoder.setBytes(&normalized, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(scratch.hiddenA, offset: 0, index: 1)
        try bind(source, prefix + "norm1.weight", encoder, 2)
        try bind(source, prefix + "norm1.bias", encoder, 3)
        encoder.setBuffer(scratch.attention, offset: 0, index: 4)
        dispatch(encoder, normPipeline, scratch.paddedRows)
        encoder.endEncoding()

        var qkv = parameters(
            rows: pixels.geometry.patchRows, paddedRows: scratch.paddedRows,
            inputWidth: config.hiddenSize, outputWidth: 3 * config.hiddenSize,
            geometry: pixels.geometry, weightScalarBytes: weightScalarBytes)
        encoder = try self.encoder(command, pipeline: linearPipeline)
        encoder.setBytes(&qkv, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(scratch.attention, offset: 0, index: 1)
        try bind(source, prefix + "attn.qkv.weight", encoder, 2)
        try bind(source, prefix + "attn.qkv.bias", encoder, 3)
        encoder.setBuffer(scratch.q, offset: 0, index: 4)
        dispatch(encoder, linearPipeline,
                 pixels.geometry.patchRows * 3 * config.hiddenSize)
        encoder.endEncoding()

        var rotate = parameters(
            rows: pixels.geometry.patchRows, paddedRows: scratch.paddedRows,
            inputWidth: config.hiddenSize, outputWidth: config.hiddenSize,
            geometry: pixels.geometry, weightScalarBytes: weightScalarBytes)
        encoder = try self.encoder(command, pipeline: rotatePipeline)
        encoder.setBytes(&rotate, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(pixels.positionsInt32x2, offset: 0, index: 1)
        encoder.setBuffer(scratch.q, offset: 0, index: 2)
        dispatch(encoder, rotatePipeline, pixels.geometry.patchRows * config.hiddenSize)
        encoder.endEncoding()

        encoder = try self.encoder(command, pipeline: attentionPipeline)
        encoder.setBytes(&rotate, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(scratch.q, offset: 0, index: 1)
        encoder.setBuffer(scratch.attention, offset: 0, index: 2)
        dispatch(encoder, attentionPipeline, scratch.paddedRows * config.numHeads)
        encoder.endEncoding()

        encoder = try self.encoder(command, pipeline: residualPipeline)
        encoder.setBytes(&rotate, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(scratch.attention, offset: 0, index: 1)
        encoder.setBuffer(scratch.hiddenA, offset: 0, index: 2)
        try bind(source, prefix + "attn.proj.weight", encoder, 3)
        try bind(source, prefix + "attn.proj.bias", encoder, 4)
        encoder.setBuffer(scratch.hiddenB, offset: 0, index: 5)
        dispatch(encoder, residualPipeline, scratch.paddedRows * config.hiddenSize)
        encoder.endEncoding()

        encoder = try self.encoder(command, pipeline: normPipeline)
        encoder.setBytes(&normalized, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(scratch.hiddenB, offset: 0, index: 1)
        try bind(source, prefix + "norm2.weight", encoder, 2)
        try bind(source, prefix + "norm2.bias", encoder, 3)
        encoder.setBuffer(scratch.attention, offset: 0, index: 4)
        dispatch(encoder, normPipeline, scratch.paddedRows)
        encoder.endEncoding()

        var mlp = parameters(
            rows: pixels.geometry.patchRows, paddedRows: scratch.paddedRows,
            inputWidth: config.hiddenSize, outputWidth: config.intermediateSize,
            geometry: pixels.geometry, weightScalarBytes: weightScalarBytes)
        encoder = try self.encoder(command, pipeline: linearTanhGELUPipeline)
        encoder.setBytes(&mlp, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(scratch.attention, offset: 0, index: 1)
        try bind(source, prefix + "mlp.linear_fc1.weight", encoder, 2)
        try bind(source, prefix + "mlp.linear_fc1.bias", encoder, 3)
        encoder.setBuffer(scratch.gate, offset: 0, index: 4)
        dispatch(encoder, linearTanhGELUPipeline,
                 pixels.geometry.patchRows * config.intermediateSize)
        encoder.endEncoding()

        var down = parameters(
            rows: pixels.geometry.patchRows, paddedRows: scratch.paddedRows,
            inputWidth: config.intermediateSize, outputWidth: config.hiddenSize,
            geometry: pixels.geometry, weightScalarBytes: weightScalarBytes)
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

    private func encodeMerger(
        pixels: QwenVisionPixelBuffer,
        scratch: QwenVisionScratch,
        source: QwenVisionTensorSource,
        command: MTLCommandBuffer,
        weightScalarBytes: UInt32 = 2
    ) throws {
        var pack = parameters(
            rows: pixels.geometry.patchRows, paddedRows: scratch.paddedRows,
            inputWidth: config.hiddenSize, outputWidth: config.mergedHiddenSize,
            geometry: pixels.geometry, weightScalarBytes: weightScalarBytes)
        var encoder = try self.encoder(command, pipeline: mergerPackPipeline)
        encoder.setBytes(&pack, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(scratch.hiddenA, offset: 0, index: 1)
        try bind(source, "model.visual.merger.norm.weight", encoder, 2)
        try bind(source, "model.visual.merger.norm.bias", encoder, 3)
        encoder.setBuffer(scratch.mergerPacked, offset: 0, index: 4)
        dispatch(encoder, mergerPackPipeline, pixels.geometry.patchRows)
        encoder.endEncoding()

        var fc1 = parameters(
            rows: pixels.geometry.mergedRows, paddedRows: pixels.geometry.mergedRows,
            inputWidth: config.mergedHiddenSize, outputWidth: config.mergedHiddenSize,
            geometry: pixels.geometry, weightScalarBytes: weightScalarBytes)
        encoder = try self.encoder(command, pipeline: linearGELUPipeline)
        encoder.setBytes(&fc1, length: MemoryLayout<QwenVisionKernelParameters>.stride, index: 0)
        encoder.setBuffer(scratch.mergerPacked, offset: 0, index: 1)
        try bind(source, "model.visual.merger.linear_fc1.weight", encoder, 2)
        try bind(source, "model.visual.merger.linear_fc1.bias", encoder, 3)
        encoder.setBuffer(scratch.mergerNormalized, offset: 0, index: 4)
        dispatch(encoder, linearGELUPipeline,
                 pixels.geometry.mergedRows * config.mergedHiddenSize)
        encoder.endEncoding()

        var fc2 = parameters(
            rows: pixels.geometry.mergedRows, paddedRows: pixels.geometry.mergedRows,
            inputWidth: config.mergedHiddenSize, outputWidth: config.outputHiddenSize,
            geometry: pixels.geometry, weightScalarBytes: weightScalarBytes)
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
            lease.resident.buffer, offset: try lease.offset(of: name), index: index)
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
