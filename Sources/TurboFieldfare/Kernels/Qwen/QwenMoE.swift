import Foundation
import Metal

enum QwenMoEError: Error, Equatable, Sendable {
    case invalidConfiguration(field: String, value: Int)
    case invalidCount(field: String, expected: Int, actual: Int)
    case nonfiniteRouterValue(token: Int, expert: Int)
    case sourceTopKSelectionFailed
    case duplicateExpertWithinToken(token: Int, expert: Int)
    case missingExpert(Int)
    case invalidExpertShape(expert: Int)
    case arithmeticOverflow(operation: String)
    case bufferTooSmall(name: String, required: Int, actual: Int)
    case invalidAffineBinding(String)
    case commandBufferAlreadySubmitted
    case commandEncoderUnavailable
    case invalidPipelineLimit
    case routingKernelRejectedInput(token: Int)
}

/// Packed routing remains the default. Only the original-BF16 source runner
/// opts into the pinned CPU operation order.
enum QwenMoERoutingArithmetic: Equatable, Sendable {
    case packed
    case officialSourceCPU
}

struct QwenMoEConfiguration: Equatable, Sendable {
    let hiddenSize: Int
    let expertCount: Int
    let topK: Int
    let routedIntermediateSize: Int
    let sharedIntermediateSize: Int

    init(hiddenSize: Int, expertCount: Int, topK: Int = 8,
         routedIntermediateSize: Int, sharedIntermediateSize: Int) throws {
        for (field, value) in [
            ("hiddenSize", hiddenSize), ("expertCount", expertCount),
            ("routedIntermediateSize", routedIntermediateSize),
            ("sharedIntermediateSize", sharedIntermediateSize),
        ] where value <= 0 || UInt32(exactly: value) == nil {
            throw QwenMoEError.invalidConfiguration(field: field, value: value)
        }
        guard topK == 8, expertCount >= topK else {
            throw QwenMoEError.invalidConfiguration(field: "topK", value: topK)
        }
        self.hiddenSize = hiddenSize
        self.expertCount = expertCount
        self.topK = topK
        self.routedIntermediateSize = routedIntermediateSize
        self.sharedIntermediateSize = sharedIntermediateSize
    }

    static func official() throws -> Self {
        try Self(
            hiddenSize: 2_048, expertCount: 256, topK: 8,
            routedIntermediateSize: 512, sharedIntermediateSize: 512)
    }
}

struct QwenMoERoutingDiagnostics: Sendable, Equatable {
    /// Token-major `[token][rank]`.
    let selectedExpertIDs: [[Int]]
    let normalizedWeights: [[Float]]
    /// Full FP32 softmax, token-major `[token][expert]`.
    let probabilities: [[Float]]
}

struct QwenMoEDenseExpert: Sendable {
    /// Row-major `[2 * routedIntermediate, hidden]`, gate rows then up rows.
    let gateUp: [Float]
    /// Row-major `[hidden, routedIntermediate]`.
    let down: [Float]
}

struct QwenMoEDenseSharedExpert: Sendable {
    let gate: [Float]
    let up: [Float]
    let down: [Float]
    /// One FP32 row `[hidden]` used by `sigmoid(shared_expert_gate(x))`.
    let outputGate: [Float]
}

struct QwenMoEReferenceResult: Sendable, Equatable {
    let routing: QwenMoERoutingDiagnostics
    let routedOutput: [Float]
    let sharedGate: [Float]
    let sharedOutput: [Float]
    let output: [Float]
}

struct QwenMoEAffineBinding: @unchecked Sendable {
    let values: MTLBuffer
    let valuesOffset: Int
    let scales: MTLBuffer
    let scalesOffset: Int
    let biases: MTLBuffer
    let biasesOffset: Int
    let layout: QwenMetalAffineLayout

    init(values: MTLBuffer, valuesOffset: Int = 0,
         scales: MTLBuffer, scalesOffset: Int = 0,
         biases: MTLBuffer, biasesOffset: Int = 0,
         layout: QwenMetalAffineLayout) {
        self.values = values
        self.valuesOffset = valuesOffset
        self.scales = scales
        self.scalesOffset = scalesOffset
        self.biases = biases
        self.biasesOffset = biasesOffset
        self.layout = layout
    }
}

struct QwenMoESharedAffineBindings: @unchecked Sendable {
    let gate: QwenMoEAffineBinding
    let up: QwenMoEAffineBinding
    let down: QwenMoEAffineBinding
    let outputGate: QwenMoEAffineBinding

    init(gate: QwenMoEAffineBinding, up: QwenMoEAffineBinding,
         down: QwenMoEAffineBinding, outputGate: QwenMoEAffineBinding) {
        self.gate = gate
        self.up = up
        self.down = down
        self.outputGate = outputGate
    }
}

/// Four independently admitted BF16 shared-expert projections. Names do not
/// grant trust: QwenBF16Weights validates the actual header shape and role.
struct QwenBF16SharedNames: Sendable {
    let gate: String
    let up: String
    let down: String
    let outputGate: String
}

/// One original-rank contribution to a token's routed output row.
struct QwenBF16GroupedExpertWork: Sendable {
    let expertID: Int
    let tokenIndex: Int
    let routeRank: Int
}

final class QwenMoEScratch: @unchecked Sendable {
    let routedActivation: MTLBuffer
    let sharedGate: MTLBuffer
    let sharedUp: MTLBuffer
    let sharedActivation: MTLBuffer
    let sharedOutput: MTLBuffer
    let sharedOutputGate: MTLBuffer

    fileprivate init(routedActivation: MTLBuffer, sharedGate: MTLBuffer,
                     sharedUp: MTLBuffer, sharedActivation: MTLBuffer,
                     sharedOutput: MTLBuffer, sharedOutputGate: MTLBuffer) {
        self.routedActivation = routedActivation
        self.sharedGate = sharedGate
        self.sharedUp = sharedUp
        self.sharedActivation = sharedActivation
        self.sharedOutput = sharedOutput
        self.sharedOutputGate = sharedOutputGate
    }
}

/// Qwen MoE reference ordering and concrete correctness Metal operations.
/// The production session-family integration is deliberately a Phase-12 concern.
final class QwenMoE {
    static let maximumGroupedContributionsPerCommand = 128

    private struct RoutingParameters {
        var tokenCount: UInt32
        var expertCount: UInt32
        var topK: UInt32
        var reserved: UInt32 = 0
    }

    private struct RoutedParameters {
        var hiddenSize: UInt32
        var intermediateSize: UInt32
        var valuesOffset: UInt32
        var scalesOffset: UInt32
        var biasesOffset: UInt32
        var valuesRowStride: UInt32
        var groupsPerRow: UInt32
        var groupSize: UInt32
        var scratchOffset: UInt32
        var reserved0: UInt32 = 0
        var reserved1: UInt32 = 0
        var reserved2: UInt32 = 0
    }

    private struct ElementParameters {
        var elementCount: UInt32
        var reserved0: UInt32 = 0
        var reserved1: UInt32 = 0
        var reserved2: UInt32 = 0
    }

    private struct BF16RoutedParameters {
        var hiddenSize: UInt32
        var intermediateSize: UInt32
        var scratchOffset: UInt32
        var reserved: UInt32 = 0
    }

    private struct RoutedWork {
        let mapped: QwenMappedExpert
        var gateUp: RoutedParameters
        var down: RoutedParameters
    }

    /// Transfers only completion waiting and resource retention to a background
    /// thread after submit has committed the command. It never encodes work or
    /// accesses buffer contents. The caller waits before reusing these buffers.
    private final class GroupedCommandCompletion: @unchecked Sendable {
        private let command: MTLCommandBuffer
        private let lease: QwenBF16ExpertLease
        private let buffers: [MTLBuffer]

        init(command: MTLCommandBuffer, lease: QwenBF16ExpertLease, buffers: [MTLBuffer]) {
            self.command = command
            self.lease = lease
            self.buffers = buffers
        }

        func wait() {
            command.waitUntilCompleted()
            withExtendedLifetime(buffers) {}
            withExtendedLifetime(lease) {}
        }
    }

    let configuration: QwenMoEConfiguration
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let routingPipeline: MTLComputePipelineState
    private let clearPipeline: MTLComputePipelineState
    private let routedGateUpPipeline: MTLComputePipelineState
    private let routedDownPipeline: MTLComputePipelineState
    private let affinePipeline: MTLComputePipelineState
    private let activationPipeline: MTLComputePipelineState
    private let sharedEpiloguePipeline: MTLComputePipelineState
    private let bf16PipelineLock = NSLock()
    private var bf16Pipelines: [String: MTLComputePipelineState] = [:]

    init(context: MetalContext, configuration: QwenMoEConfiguration) throws {
        self.configuration = configuration
        device = context.device
        queue = context.queue
        routingPipeline = try context.pipeline("qwen_moe_route_top8_fp32")
        clearPipeline = try context.pipeline("qwen_moe_clear_fp32")
        routedGateUpPipeline = try context.pipeline("qwen_moe_routed_gate_up_int4")
        routedDownPipeline = try context.pipeline("qwen_moe_routed_down_add_int4")
        affinePipeline = try context.pipeline("qwen_moe_affine_project")
        activationPipeline = try context.pipeline("qwen_moe_silu_multiply")
        sharedEpiloguePipeline = try context.pipeline("qwen_moe_shared_epilogue")
    }

    /// BF16 MoE arithmetic is compiled separately so its rounding and IEEE
    /// behavior do not change the packed shared shader library.
    private func bf16Pipeline(_ name: String) throws -> MTLComputePipelineState {
        bf16PipelineLock.lock()
        defer { bf16PipelineLock.unlock() }
        if let pipeline = bf16Pipelines[name] { return pipeline }
        let library = try MetalContext.privateLibrary(
            device: device, module: "qwen_moe", mathMode: .safe,
            mathFloatingPointFunctions: .precise, includeQwenSourceMath: true)
        guard let function = library.makeFunction(name: name) else {
            throw MetalError.missingFunction(name)
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        bf16Pipelines[name] = pipeline
        return pipeline
    }

    private func bf16RoutedPipeline(serialName: String, cooperativeName: String,
                                    useCooperative: Bool) throws
        -> (pipeline: MTLComputePipelineState, cooperative: Bool) {
        if useCooperative {
            let pipeline = try bf16Pipeline(cooperativeName)
            if pipeline.threadExecutionWidth == 32,
               pipeline.maxTotalThreadsPerThreadgroup >= 32 {
                return (pipeline, true)
            }
        }
        return (try bf16Pipeline(serialName), false)
    }

    static func route(
        logits: [Float], configuration: QwenMoEConfiguration,
        arithmetic: QwenMoERoutingArithmetic = .packed
    ) throws -> QwenMoERoutingDiagnostics {
        guard logits.count.isMultiple(of: configuration.expertCount) else {
            throw QwenMoEError.invalidCount(
                field: "routerLogits", expected: configuration.expertCount,
                actual: logits.count)
        }
        let tokenCount = logits.count / configuration.expertCount
        var allIDs: [[Int]] = []
        var allWeights: [[Float]] = []
        var allProbabilities: [[Float]] = []
        allIDs.reserveCapacity(tokenCount)
        allWeights.reserveCapacity(tokenCount)
        allProbabilities.reserveCapacity(tokenCount)

        for token in 0..<tokenCount {
            let start = token * configuration.expertCount
            let values = Array(logits[start..<(start + configuration.expertCount)])
            for (expert, value) in values.enumerated() where !value.isFinite {
                throw QwenMoEError.nonfiniteRouterValue(token: token, expert: expert)
            }
            guard let maximum = values.max() else {
                throw QwenMoEError.invalidCount(field: "routerLogits", expected: 1, actual: 0)
            }
            let exponentials = values.map {
                arithmetic == .officialSourceCPU
                    ? QwenOfficialSourceRouterArithmetic.exponential($0 - maximum)
                    : Foundation.exp($0 - maximum)
            }
            let denominator = arithmetic == .officialSourceCPU
                ? QwenOfficialSourceRouterArithmetic.softmaxSum(exponentials)
                : exponentials.reduce(Float(0), +)
            guard denominator.isFinite, denominator > 0 else {
                throw QwenMoEError.nonfiniteRouterValue(token: token, expert: 0)
            }
            let probabilities: [Float]
            if arithmetic == .officialSourceCPU {
                let reciprocal: Float = 1 / denominator
                probabilities = exponentials.map { $0 * reciprocal }
            } else {
                probabilities = exponentials.map { $0 / denominator }
            }
            let ids: [Int]
            if arithmetic == .officialSourceCPU, configuration.expertCount == 256 {
                ids = try QwenOfficialSourceRouterArithmetic.top8Indices(probabilities)
            } else {
                // Packed and small-fixture routing retain their defined tie rule.
                ids = Array(probabilities.indices.sorted { lhs, rhs in
                    if probabilities[lhs] != probabilities[rhs] {
                        return probabilities[lhs] > probabilities[rhs]
                    }
                    return lhs < rhs
                }.prefix(configuration.topK))
            }
            let selectedSum = arithmetic == .officialSourceCPU
                ? QwenOfficialSourceRouterArithmetic.top8Sum(ids.map { probabilities[$0] })
                : ids.reduce(Float(0)) { $0 + probabilities[$1] }
            guard selectedSum.isFinite, selectedSum > 0 else {
                throw QwenMoEError.nonfiniteRouterValue(token: token, expert: ids[0])
            }
            allIDs.append(ids)
            allWeights.append(ids.map { probabilities[$0] / selectedSum })
            allProbabilities.append(probabilities)
        }
        return QwenMoERoutingDiagnostics(
            selectedExpertIDs: allIDs,
            normalizedWeights: allWeights,
            probabilities: allProbabilities)
    }

    static func evaluate(
        input: [Float],
        routerLogits: [Float],
        configuration: QwenMoEConfiguration,
        routedExperts: [Int: QwenMoEDenseExpert],
        sharedExpert: QwenMoEDenseSharedExpert
    ) throws -> QwenMoEReferenceResult {
        guard input.count.isMultiple(of: configuration.hiddenSize) else {
            throw QwenMoEError.invalidCount(
                field: "input", expected: configuration.hiddenSize, actual: input.count)
        }
        let tokenCount = input.count / configuration.hiddenSize
        guard routerLogits.count == tokenCount * configuration.expertCount else {
            throw QwenMoEError.invalidCount(
                field: "routerLogits",
                expected: tokenCount * configuration.expertCount,
                actual: routerLogits.count)
        }
        let routing = try route(logits: routerLogits, configuration: configuration)
        try validate(sharedExpert: sharedExpert, configuration: configuration)
        var routedOutput = [Float](repeating: 0, count: input.count)
        var sharedOutput = [Float](repeating: 0, count: input.count)
        var sharedGates = [Float](repeating: 0, count: tokenCount)

        for token in 0..<tokenCount {
            let hidden = Array(input[
                token * configuration.hiddenSize..<(token + 1) * configuration.hiddenSize])
            for rank in 0..<configuration.topK {
                let expertID = routing.selectedExpertIDs[token][rank]
                guard let expert = routedExperts[expertID] else {
                    throw QwenMoEError.missingExpert(expertID)
                }
                try validate(expert: expert, expertID: expertID, configuration: configuration)
                let projected = project(
                    hidden, matrix: expert.gateUp,
                    rows: 2 * configuration.routedIntermediateSize,
                    columns: configuration.hiddenSize)
                var activated = [Float](repeating: 0, count: configuration.routedIntermediateSize)
                for index in activated.indices {
                    activated[index] = silu(projected[index])
                        * projected[configuration.routedIntermediateSize + index]
                }
                let down = project(
                    activated, matrix: expert.down,
                    rows: configuration.hiddenSize,
                    columns: configuration.routedIntermediateSize)
                let weight = routing.normalizedWeights[token][rank]
                for index in 0..<configuration.hiddenSize {
                    routedOutput[token * configuration.hiddenSize + index] += down[index] * weight
                }
            }

            let sharedGateProjection = project(
                hidden, matrix: sharedExpert.gate,
                rows: configuration.sharedIntermediateSize,
                columns: configuration.hiddenSize)
            let sharedUpProjection = project(
                hidden, matrix: sharedExpert.up,
                rows: configuration.sharedIntermediateSize,
                columns: configuration.hiddenSize)
            var sharedActivation = [Float](repeating: 0, count: configuration.sharedIntermediateSize)
            for index in sharedActivation.indices {
                sharedActivation[index] = silu(sharedGateProjection[index]) * sharedUpProjection[index]
            }
            let down = project(
                sharedActivation, matrix: sharedExpert.down,
                rows: configuration.hiddenSize,
                columns: configuration.sharedIntermediateSize)
            var rawGate: Float = 0
            for index in 0..<configuration.hiddenSize {
                rawGate += hidden[index] * sharedExpert.outputGate[index]
            }
            let gate = sigmoid(rawGate)
            sharedGates[token] = gate
            for index in 0..<configuration.hiddenSize {
                sharedOutput[token * configuration.hiddenSize + index] = down[index] * gate
            }
        }
        let output = zip(routedOutput, sharedOutput).map(+)
        return QwenMoEReferenceResult(
            routing: routing, routedOutput: routedOutput,
            sharedGate: sharedGates, sharedOutput: sharedOutput, output: output)
    }

    func makeScratch() throws -> QwenMoEScratch {
        func buffer(elements: Int, label: String) throws -> MTLBuffer {
            let bytes = try Self.checkedMultiply(elements, MemoryLayout<Float>.stride, operation: label)
            guard UInt32(exactly: elements) != nil,
                  bytes <= device.maxBufferLength,
                  let result = device.makeBuffer(length: bytes, options: .storageModePrivate) else {
                throw QwenMoEError.bufferTooSmall(name: label, required: bytes, actual: 0)
            }
            result.label = label
            return result
        }
        return try QwenMoEScratch(
            routedActivation: buffer(
                elements: try Self.checkedMultiply(configuration.topK,
                    configuration.routedIntermediateSize,
                    operation: "qwen.moe.routedActivation elements"),
                label: "qwen.moe.routedActivation"),
            sharedGate: buffer(elements: configuration.sharedIntermediateSize,
                               label: "qwen.moe.sharedGate"),
            sharedUp: buffer(elements: configuration.sharedIntermediateSize,
                             label: "qwen.moe.sharedUp"),
            sharedActivation: buffer(elements: configuration.sharedIntermediateSize,
                                     label: "qwen.moe.sharedActivation"),
            sharedOutput: buffer(elements: configuration.hiddenSize,
                                 label: "qwen.moe.sharedOutput"),
            sharedOutputGate: buffer(elements: 1,
                                     label: "qwen.moe.sharedOutputGate"))
    }

    func encodeRouting(commandBuffer: MTLCommandBuffer,
                       logits: MTLBuffer,
                       tokenCount: Int,
                       selectedExpertIDs: MTLBuffer,
                       normalizedWeights: MTLBuffer,
                       status: MTLBuffer) throws {
        guard tokenCount > 0 else {
            throw QwenMoEError.invalidConfiguration(field: "tokenCount", value: tokenCount)
        }
        let logitsCount = try Self.checkedMultiply(
            tokenCount, configuration.expertCount, operation: "router logits count")
        try Self.requireBuffer(logits, named: "routerLogits", elements: logitsCount, as: Float.self)
        let selectedCount = try Self.checkedMultiply(
            tokenCount, configuration.topK, operation: "selected count")
        try Self.requireBuffer(
            selectedExpertIDs, named: "selectedExpertIDs", elements: selectedCount, as: UInt32.self)
        try Self.requireBuffer(
            normalizedWeights, named: "normalizedWeights", elements: selectedCount, as: Float.self)
        try Self.requireBuffer(status, named: "routingStatus", elements: tokenCount, as: UInt32.self)
        try requireNotSubmitted(commandBuffer)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QwenMoEError.commandEncoderUnavailable
        }
        var parameters = RoutingParameters(
            tokenCount: try Self.uint32(tokenCount, field: "tokenCount"),
            expertCount: UInt32(configuration.expertCount),
            topK: UInt32(configuration.topK))
        encoder.setComputePipelineState(routingPipeline)
        encoder.setBytes(&parameters, length: MemoryLayout<RoutingParameters>.stride,
                         index: QwenMetalBufferIndex.parameters.rawValue)
        encoder.setBuffer(logits, offset: 0, index: QwenMetalBufferIndex.input.rawValue)
        encoder.setBuffer(selectedExpertIDs, offset: 0,
                          index: QwenMetalBufferIndex.output.rawValue)
        encoder.setBuffer(normalizedWeights, offset: 0,
                          index: QwenMetalBufferIndex.scratch.rawValue)
        encoder.setBuffer(status, offset: 0, index: QwenMetalBufferIndex.state.rawValue)
        try dispatch(encoder, pipeline: routingPipeline, count: tokenCount)
        encoder.endEncoding()
    }

    /// Encodes and submits one token's complete routed and shared expert work.
    /// `routingWeights` contains exactly Top-8 FP32 weights in the same order
    /// as `lease.experts`.
    ///
    /// The command buffer is created and committed inside this operation. No
    /// externally committable, encoded-but-unowned interval exists: before the
    /// mapped reads are encoded the lease transfers to completion ownership;
    /// every path after that transfer commits the buffer, and cancellation only
    /// marks its result discarded until actual completion releases the slots.
    @discardableResult
    func submitExperts(hidden: MTLBuffer,
                       lease: QwenMappedExpertLease,
                       routingWeights: MTLBuffer,
                       sharedBindings: QwenMoESharedAffineBindings,
                       scratch: QwenMoEScratch,
                       output: MTLBuffer) throws -> MTLCommandBuffer {
        let work = try prepareSubmission(
            hidden: hidden, lease: lease, routingWeights: routingWeights,
            sharedBindings: sharedBindings, output: output)

        // This is the sole mapped-buffer submission boundary. Validation above
        // leaves the lease unsubmitted on failure. The lease owns command-buffer
        // creation, transition, failure cleanup, commit, and completion release.
        return try lease.submit(on: queue) { commandBuffer in
            try encodeRoutedCommands(
                commandBuffer: commandBuffer, hidden: hidden, work: work,
                routingWeights: routingWeights, scratch: scratch, output: output)
            try encodeSharedCommands(
                commandBuffer: commandBuffer, hidden: hidden,
                bindings: sharedBindings, scratch: scratch, output: output)
        }
    }

    /// Adds a resident shared expert to an externally managed command buffer.
    /// This operation never accesses streamed mappings.
    func encodeShared(commandBuffer: MTLCommandBuffer,
                      hidden: MTLBuffer,
                      bindings: QwenMoESharedAffineBindings,
                      scratch: QwenMoEScratch,
                      output: MTLBuffer) throws {
        try validateShared(hidden: hidden, bindings: bindings, output: output)
        try requireNotSubmitted(commandBuffer)
        try validateSharedDispatches()
        try encodeSharedCommands(
            commandBuffer: commandBuffer, hidden: hidden,
            bindings: bindings, scratch: scratch, output: output)
    }

    private static func requireDistinctSharedWeights(_ weights: QwenBF16Weights,
                                                     from buffers: [MTLBuffer]) throws {
        let callerIDs = Set(buffers.map { ObjectIdentifier($0) })
        guard weights.inspectedChunks.allSatisfy({
            !callerIDs.contains(ObjectIdentifier($0.buffer))
        }) else {
            throw QwenMoEError.invalidAffineBinding("BF16 shared weight aliases caller buffer")
        }
    }

    /// A separate BF16 shared path. Routed packed experts and the existing
    /// packed shared encoder are unchanged. One token, FP32 intermediates.
    /// The caller owns submission; discard the command buffer on any error.
    func encodeSharedBF16(commandBuffer: MTLCommandBuffer,
                          hidden: MTLBuffer,
                          weights: QwenBF16Weights,
                          names: QwenBF16SharedNames,
                          scratch: QwenMoEScratch,
                          output: MTLBuffer) throws {
        try requireNotSubmitted(commandBuffer)
        try Self.requireBuffer(hidden, named: "hidden", elements: configuration.hiddenSize,
                               as: Float.self)
        try Self.requireBuffer(output, named: "output", elements: configuration.hiddenSize,
                               as: Float.self)
        try Self.requireBuffer(scratch.sharedGate, named: "sharedGate",
                               elements: configuration.sharedIntermediateSize, as: Float.self)
        try Self.requireBuffer(scratch.sharedUp, named: "sharedUp",
                               elements: configuration.sharedIntermediateSize, as: Float.self)
        try Self.requireBuffer(scratch.sharedActivation, named: "sharedActivation",
                               elements: configuration.sharedIntermediateSize, as: Float.self)
        try Self.requireBuffer(scratch.sharedOutput, named: "sharedOutput",
                               elements: configuration.hiddenSize, as: Float.self)
        try Self.requireBuffer(scratch.sharedOutputGate, named: "sharedOutputGate",
                               elements: 1, as: Float.self)
        let buffers: [MTLBuffer] = [hidden, output, scratch.sharedGate, scratch.sharedUp,
                                    scratch.sharedActivation, scratch.sharedOutput,
                                    scratch.sharedOutputGate]
        guard buffers.allSatisfy({ $0.device === device }),
              Set(buffers.map { ObjectIdentifier($0) }).count == buffers.count else {
            throw QwenMoEError.invalidAffineBinding("BF16 shared buffers alias or use another device")
        }
        try Self.requireDistinctSharedWeights(weights, from: buffers)
        try weights.requireShape(names.gate, role: .sharedGate,
                                 rows: configuration.sharedIntermediateSize,
                                 columns: configuration.hiddenSize)
        try weights.requireShape(names.up, role: .sharedUp,
                                 rows: configuration.sharedIntermediateSize,
                                 columns: configuration.hiddenSize)
        try weights.requireShape(names.down, role: .sharedDown,
                                 rows: configuration.hiddenSize,
                                 columns: configuration.sharedIntermediateSize)
        try weights.requireShape(names.outputGate, role: .sharedOutputGate,
                                 rows: 1, columns: configuration.hiddenSize)
        let activationPipeline = try bf16Pipeline("qwen_moe_silu_multiply_bf16")
        let sharedEpiloguePipeline = try bf16Pipeline("qwen_moe_shared_epilogue_bf16")
        try requireDispatchable(activationPipeline, count: configuration.sharedIntermediateSize)
        try requireDispatchable(sharedEpiloguePipeline, count: configuration.hiddenSize)
        try weights.encodeProjection(commandBuffer: commandBuffer, tensorName: names.gate,
                                     input: hidden, tokenCount: 1, output: scratch.sharedGate)
        try weights.encodeProjection(commandBuffer: commandBuffer, tensorName: names.up,
                                     input: hidden, tokenCount: 1, output: scratch.sharedUp)
        guard let activationEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QwenMoEError.commandEncoderUnavailable
        }
        var activationParameters = ElementParameters(
            elementCount: UInt32(configuration.sharedIntermediateSize))
        activationEncoder.setComputePipelineState(activationPipeline)
        activationEncoder.setBytes(&activationParameters,
                                   length: MemoryLayout<ElementParameters>.stride,
                                   index: QwenMetalBufferIndex.parameters.rawValue)
        activationEncoder.setBuffer(scratch.sharedGate, offset: 0,
                                    index: QwenMetalBufferIndex.input.rawValue)
        activationEncoder.setBuffer(scratch.sharedUp, offset: 0,
                                    index: QwenMetalBufferIndex.weights.rawValue)
        activationEncoder.setBuffer(scratch.sharedActivation, offset: 0,
                                    index: QwenMetalBufferIndex.output.rawValue)
        try dispatch(activationEncoder, pipeline: activationPipeline,
                     count: configuration.sharedIntermediateSize)
        activationEncoder.endEncoding()
        try weights.encodeProjection(commandBuffer: commandBuffer, tensorName: names.down,
                                     input: scratch.sharedActivation, tokenCount: 1,
                                     output: scratch.sharedOutput)
        try weights.encodeProjection(commandBuffer: commandBuffer,
                                     tensorName: names.outputGate, input: hidden,
                                     tokenCount: 1, output: scratch.sharedOutputGate)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QwenMoEError.commandEncoderUnavailable
        }
        var parameters = ElementParameters(elementCount: UInt32(configuration.hiddenSize))
        encoder.setComputePipelineState(sharedEpiloguePipeline)
        encoder.setBytes(&parameters, length: MemoryLayout<ElementParameters>.stride,
                         index: QwenMetalBufferIndex.parameters.rawValue)
        encoder.setBuffer(scratch.sharedOutputGate, offset: 0,
                          index: QwenMetalBufferIndex.weights.rawValue)
        encoder.setBuffer(output, offset: 0, index: QwenMetalBufferIndex.output.rawValue)
        encoder.setBuffer(scratch.sharedOutput, offset: 0,
                          index: QwenMetalBufferIndex.scratch.rawValue)
        try dispatch(encoder, pipeline: sharedEpiloguePipeline,
                     count: configuration.hiddenSize)
        encoder.endEncoding()
    }

    /// Actual resident BF16 routed experts, followed by the existing BF16
    /// shared branch. No mapped slot may escape this completion-owned submit.
    /// Input routing weights are the finite, normalized Top-8 result of the
    /// existing FP32 router, in exactly `lease.experts` rank order.
    @discardableResult
    func submitExpertsBF16(hidden: MTLBuffer,
                           lease: QwenBF16ExpertLease,
                           routingWeights: MTLBuffer,
                           sharedWeights: QwenBF16Weights,
                           sharedNames: QwenBF16SharedNames,
                           scratch: QwenMoEScratch,
                           output: MTLBuffer) throws -> MTLCommandBuffer {
        try lease.requireUsable()
        guard lease.experts.count == configuration.topK else {
            throw QwenMoEError.invalidCount(
                field: "BF16 routed experts", expected: configuration.topK,
                actual: lease.experts.count)
        }
        guard lease.experts.map(\.expertID) == lease.diagnostics.requestedExpertIDs else {
            throw QwenMoEError.invalidAffineBinding(
                "BF16 expert lease order differs from requested routing IDs")
        }
        let twiceIntermediate = try Self.checkedMultiply(
            2, configuration.routedIntermediateSize, operation: "BF16 gate/up rows")
        let gateCount = try Self.checkedMultiply(
            twiceIntermediate, configuration.hiddenSize, operation: "BF16 gate/up elements")
        let downCount = try Self.checkedMultiply(
            configuration.hiddenSize, configuration.routedIntermediateSize,
            operation: "BF16 down elements")
        let activationCount = try Self.checkedMultiply(
            configuration.topK, configuration.routedIntermediateSize,
            operation: "BF16 activation elements")
        guard UInt32(exactly: gateCount) != nil,
              UInt32(exactly: downCount) != nil,
              UInt32(exactly: activationCount) != nil else {
            throw QwenMoEError.arithmeticOverflow(operation: "BF16 Metal scalar indexing")
        }
        let gateBytes = try Self.checkedMultiply(
            gateCount, MemoryLayout<UInt16>.stride, operation: "BF16 gate/up bytes")
        let downBytes = try Self.checkedMultiply(
            downCount, MemoryLayout<UInt16>.stride, operation: "BF16 down bytes")
        guard gateBytes <= device.maxBufferLength,
              downBytes <= device.maxBufferLength else {
            throw QwenMoEError.invalidExpertShape(expert: -1)
        }
        try Self.requireBuffer(hidden, named: "BF16 hidden",
                               elements: configuration.hiddenSize, as: Float.self)
        try Self.requireBuffer(output, named: "BF16 output",
                               elements: configuration.hiddenSize, as: Float.self)
        try Self.requireBuffer(routingWeights, named: "BF16 routingWeights",
                               elements: configuration.topK, as: Float.self)
        try Self.requireBuffer(scratch.routedActivation, named: "BF16 routedActivation",
                               elements: activationCount, as: Float.self)
        try Self.requireBuffer(scratch.sharedGate, named: "BF16 sharedGate",
                               elements: configuration.sharedIntermediateSize, as: Float.self)
        try Self.requireBuffer(scratch.sharedUp, named: "BF16 sharedUp",
                               elements: configuration.sharedIntermediateSize, as: Float.self)
        try Self.requireBuffer(scratch.sharedActivation, named: "BF16 sharedActivation",
                               elements: configuration.sharedIntermediateSize, as: Float.self)
        try Self.requireBuffer(scratch.sharedOutput, named: "BF16 sharedOutput",
                               elements: configuration.hiddenSize, as: Float.self)
        try Self.requireBuffer(scratch.sharedOutputGate, named: "BF16 sharedOutputGate",
                               elements: 1, as: Float.self)
        let allBuffers: [MTLBuffer] = [hidden, routingWeights, output,
            scratch.routedActivation, scratch.sharedGate, scratch.sharedUp,
            scratch.sharedActivation, scratch.sharedOutput, scratch.sharedOutputGate]
        guard allBuffers.allSatisfy({ $0.device === device }),
              Set(allBuffers.map { ObjectIdentifier($0) }).count == allBuffers.count,
              routingWeights.storageMode == .shared,
              sharedWeights.inspectedChunks.allSatisfy({ $0.buffer.device === device }) else {
            throw QwenMoEError.invalidAffineBinding("BF16 device, alias or routing storage")
        }
        // Reject resident weights in every caller/mutable position before the
        // lease can submit its initial output clear or any routed projection.
        try Self.requireDistinctSharedWeights(sharedWeights, from: allBuffers)
        var seen = Set<Int>()
        for mapped in lease.experts {
            guard mapped.expertID >= 0, mapped.expertID < configuration.expertCount,
                  seen.insert(mapped.expertID).inserted,
                  mapped.gateUp !== mapped.down,
                  mapped.gateUp.device === device, mapped.down.device === device,
                  mapped.gateUp.storageMode == .shared,
                  mapped.down.storageMode == .shared,
                  mapped.gateUpLength == gateBytes,
                  mapped.downLength == downBytes,
                  mapped.gateUp.length >= gateBytes,
                  mapped.down.length >= downBytes,
                  !allBuffers.contains(where: { $0 === mapped.gateUp || $0 === mapped.down }) else {
                throw QwenMoEError.invalidExpertShape(expert: mapped.expertID)
            }
        }
        let route = routingWeights.contents().assumingMemoryBound(to: Float.self)
        var total: Float = 0
        for rank in 0..<configuration.topK {
            let weight = route[rank]
            guard weight.isFinite, weight >= 0 else {
                throw QwenMoEError.invalidAffineBinding("nonfinite/negative BF16 routing weight")
            }
            total += weight
        }
        guard total.isFinite, abs(total - 1) <= 0.001 else {
            throw QwenMoEError.invalidAffineBinding("BF16 Top-8 weights not normalized")
        }
        // Shared shapes are checked before ownership transfers; its encoder
        // repeats binding checks after transfer, with failure-safe commitment.
        try sharedWeights.requireShape(sharedNames.gate, role: .sharedGate,
                                       rows: configuration.sharedIntermediateSize,
                                       columns: configuration.hiddenSize)
        try sharedWeights.requireShape(sharedNames.up, role: .sharedUp,
                                       rows: configuration.sharedIntermediateSize,
                                       columns: configuration.hiddenSize)
        try sharedWeights.requireShape(sharedNames.down, role: .sharedDown,
                                       rows: configuration.hiddenSize,
                                       columns: configuration.sharedIntermediateSize)
        try sharedWeights.requireShape(sharedNames.outputGate, role: .sharedOutputGate,
                                       rows: 1, columns: configuration.hiddenSize)
        // Match the established source large-dot family exactly. Each
        // cooperative row uses one full 32-lane group; other hardware and
        // small/incomplete shapes retain the original per-row kernel.
        let gateSelection = try bf16RoutedPipeline(
            serialName: "qwen_moe_routed_gate_up_bf16",
            cooperativeName: "qwen_moe_routed_gate_up_bf16_cooperative",
            useCooperative: configuration.routedIntermediateSize >= 256
                && configuration.hiddenSize >= 512
                && configuration.hiddenSize.isMultiple(of: 64))
        let downSelection = try bf16RoutedPipeline(
            serialName: "qwen_moe_routed_down_add_bf16",
            cooperativeName: "qwen_moe_routed_down_add_bf16_cooperative",
            useCooperative: configuration.hiddenSize >= 512
                && configuration.routedIntermediateSize >= 512
                && configuration.routedIntermediateSize.isMultiple(of: 64))
        let gatePipeline = gateSelection.pipeline
        let downPipeline = downSelection.pipeline
        try requireDispatchable(gatePipeline, count: configuration.routedIntermediateSize)
        try requireDispatchable(downPipeline, count: configuration.hiddenSize)
        try requireDispatchable(clearPipeline, count: configuration.hiddenSize)
        try requireDispatchable(try bf16Pipeline("qwen_moe_silu_multiply_bf16"),
                                count: configuration.sharedIntermediateSize)
        try requireDispatchable(try bf16Pipeline("qwen_moe_shared_epilogue_bf16"),
                                count: configuration.hiddenSize)
        return try lease.submit(on: queue) { command in
            try encodeClear(commandBuffer: command, buffer: output,
                            count: configuration.hiddenSize)
            // The pinned eager expert loop visits ascending expert ID. Retain
            // the original rank for the matching route weight and scratch row.
            let orderedExperts = lease.experts.enumerated().sorted {
                $0.element.expertID < $1.element.expertID
            }
            for (rank, mapped) in orderedExperts {
                let offset = try Self.checkedMultiply(
                    rank, configuration.routedIntermediateSize,
                    operation: "BF16 routed scratch offset")
                guard UInt32(exactly: offset) != nil else {
                    throw QwenMoEError.arithmeticOverflow(operation: "BF16 routed scratch offset")
                }
                var parameters = BF16RoutedParameters(
                    hiddenSize: UInt32(configuration.hiddenSize),
                    intermediateSize: UInt32(configuration.routedIntermediateSize),
                    scratchOffset: UInt32(offset))
                guard let gateEncoder = command.makeComputeCommandEncoder() else {
                    throw QwenMoEError.commandEncoderUnavailable
                }
                gateEncoder.setComputePipelineState(gatePipeline)
                gateEncoder.setBytes(&parameters, length: MemoryLayout<BF16RoutedParameters>.stride,
                                     index: QwenMetalBufferIndex.parameters.rawValue)
                gateEncoder.setBuffer(hidden, offset: 0,
                                      index: QwenMetalBufferIndex.input.rawValue)
                gateEncoder.setBuffer(mapped.gateUp, offset: 0,
                                      index: QwenMetalBufferIndex.weights.rawValue)
                gateEncoder.setBuffer(scratch.routedActivation, offset: 0,
                                      index: QwenMetalBufferIndex.scratch.rawValue)
                gateEncoder.useResource(mapped.gateUp, usage: .read)
                try dispatchBF16Routed(gateEncoder, pipeline: gatePipeline,
                                       rows: configuration.routedIntermediateSize,
                                       cooperative: gateSelection.cooperative)
                gateEncoder.endEncoding()

                guard let downEncoder = command.makeComputeCommandEncoder() else {
                    throw QwenMoEError.commandEncoderUnavailable
                }
                downEncoder.setComputePipelineState(downPipeline)
                downEncoder.setBytes(&parameters, length: MemoryLayout<BF16RoutedParameters>.stride,
                                     index: QwenMetalBufferIndex.parameters.rawValue)
                downEncoder.setBuffer(scratch.routedActivation, offset: 0,
                                      index: QwenMetalBufferIndex.input.rawValue)
                downEncoder.setBuffer(mapped.down, offset: 0,
                                      index: QwenMetalBufferIndex.weights.rawValue)
                downEncoder.setBuffer(output, offset: 0,
                                      index: QwenMetalBufferIndex.output.rawValue)
                downEncoder.setBuffer(routingWeights,
                                      offset: rank * MemoryLayout<Float>.stride,
                                      index: QwenMetalBufferIndex.state.rawValue)
                downEncoder.useResource(mapped.down, usage: .read)
                try dispatchBF16Routed(downEncoder, pipeline: downPipeline,
                                       rows: configuration.hiddenSize,
                                       cooperative: downSelection.cooperative)
                downEncoder.endEncoding()
            }
            try encodeSharedBF16(commandBuffer: command, hidden: hidden,
                                 weights: sharedWeights, names: sharedNames,
                                 scratch: scratch, output: output)
        }
    }

    /// The caller supplies consecutive chunks of globally ascending expert IDs.
    /// Each chunk binds only its current lease, retaining the original token rank
    /// for weights. Shared experts run after all routed chunks have settled.
    func submitGroupedExpertsBF16(
        hiddenRows: MTLBuffer,
        routingExpertIDs: [Int],
        routingWeights: MTLBuffer,
        outputRows: MTLBuffer,
        tokenCount: Int,
        lease: QwenBF16ExpertLease,
        work: [QwenBF16GroupedExpertWork],
        sharedWeights: QwenBF16Weights,
        scratch: QwenMoEScratch,
        initializeOutput: Bool,
        isolation: isolated (any Actor)? = #isolation
    ) async throws {
        // Capture inside submit: its encoding failure path still commits the
        // partially encoded command and holds both buffers until completion.
        var submittedCommand: MTLCommandBuffer?
        let retainedBuffers = [hiddenRows, routingWeights, outputRows,
            scratch.routedActivation, scratch.sharedGate, scratch.sharedUp,
            scratch.sharedActivation, scratch.sharedOutput, scratch.sharedOutputGate]
        do {
            try Task.checkCancellation()
            try lease.requireUsable()
            guard tokenCount > 0, UInt32(exactly: tokenCount) != nil else {
                throw QwenMoEError.invalidConfiguration(field: "grouped tokenCount", value: tokenCount)
            }
            guard !work.isEmpty, work.count <= Self.maximumGroupedContributionsPerCommand else {
                throw QwenMoEError.invalidCount(
                    field: "grouped contributions", expected: Self.maximumGroupedContributionsPerCommand,
                    actual: work.count)
            }
            guard !lease.experts.isEmpty, lease.experts.count <= configuration.expertCount,
                  lease.experts.map(\.expertID) == lease.diagnostics.requestedExpertIDs else {
                throw QwenMoEError.invalidAffineBinding("grouped lease IDs differ from requested IDs")
            }
            let rowElements = try Self.checkedMultiply(
                tokenCount, configuration.hiddenSize, operation: "grouped hidden elements")
            let routeCount = try Self.checkedMultiply(
                tokenCount, configuration.topK, operation: "grouped route elements")
            let activationCount = try Self.checkedMultiply(
                configuration.topK, configuration.routedIntermediateSize,
                operation: "grouped activation elements")
            let gateRows = try Self.checkedMultiply(
                2, configuration.routedIntermediateSize, operation: "grouped gate/up rows")
            let gateCount = try Self.checkedMultiply(
                gateRows, configuration.hiddenSize, operation: "grouped gate/up elements")
            let downCount = try Self.checkedMultiply(
                configuration.hiddenSize, configuration.routedIntermediateSize,
                operation: "grouped down elements")
            guard [rowElements, routeCount, activationCount, gateCount, downCount]
                    .allSatisfy({ UInt32(exactly: $0) != nil }) else {
                throw QwenMoEError.arithmeticOverflow(operation: "grouped Metal scalar indexing")
            }
            let gateBytes = try Self.checkedMultiply(
                gateCount, MemoryLayout<UInt16>.stride, operation: "grouped gate/up bytes")
            let downBytes = try Self.checkedMultiply(
                downCount, MemoryLayout<UInt16>.stride, operation: "grouped down bytes")
            guard gateBytes <= device.maxBufferLength, downBytes <= device.maxBufferLength else {
                throw QwenMoEError.invalidExpertShape(expert: -1)
            }
            guard routingExpertIDs.count == routeCount else {
                throw QwenMoEError.invalidCount(
                    field: "grouped route IDs", expected: routeCount, actual: routingExpertIDs.count)
            }
            try Self.requireBuffer(hiddenRows, named: "grouped hidden", elements: rowElements, as: Float.self)
            try Self.requireBuffer(outputRows, named: "grouped output", elements: rowElements, as: Float.self)
            try Self.requireBuffer(routingWeights, named: "grouped weights", elements: routeCount, as: Float.self)
            try Self.requireBuffer(scratch.routedActivation, named: "grouped routedActivation",
                                   elements: activationCount, as: Float.self)
            try Self.requireBuffer(scratch.sharedGate, named: "grouped sharedGate",
                                   elements: configuration.sharedIntermediateSize, as: Float.self)
            try Self.requireBuffer(scratch.sharedUp, named: "grouped sharedUp",
                                   elements: configuration.sharedIntermediateSize, as: Float.self)
            try Self.requireBuffer(scratch.sharedActivation, named: "grouped sharedActivation",
                                   elements: configuration.sharedIntermediateSize, as: Float.self)
            try Self.requireBuffer(scratch.sharedOutput, named: "grouped sharedOutput",
                                   elements: configuration.hiddenSize, as: Float.self)
            try Self.requireBuffer(scratch.sharedOutputGate, named: "grouped sharedOutputGate",
                                   elements: 1, as: Float.self)
            guard retainedBuffers.allSatisfy({ $0.device === device }),
                  Set(retainedBuffers.map { ObjectIdentifier($0) }).count == retainedBuffers.count,
                  routingWeights.storageMode == .shared,
                  sharedWeights.inspectedChunks.allSatisfy({ $0.buffer.device === device }) else {
                throw QwenMoEError.invalidAffineBinding("grouped device, alias or routing storage")
            }
            try Self.requireDistinctSharedWeights(sharedWeights, from: retainedBuffers)
            var mappedByID: [Int: QwenBF16MappedExpert] = [:]
            var mappedBufferIDs = Set<ObjectIdentifier>()
            for mapped in lease.experts {
                guard mapped.expertID >= 0, mapped.expertID < configuration.expertCount,
                      mappedByID[mapped.expertID] == nil,
                      mapped.gateUp !== mapped.down,
                      mapped.gateUp.device === device, mapped.down.device === device,
                      mapped.gateUp.storageMode == .shared, mapped.down.storageMode == .shared,
                      mapped.gateUpLength == gateBytes, mapped.downLength == downBytes,
                      mapped.gateUp.length >= gateBytes, mapped.down.length >= downBytes,
                      mappedBufferIDs.insert(ObjectIdentifier(mapped.gateUp)).inserted,
                      mappedBufferIDs.insert(ObjectIdentifier(mapped.down)).inserted,
                      !retainedBuffers.contains(where: { $0 === mapped.gateUp || $0 === mapped.down }) else {
                    throw QwenMoEError.invalidExpertShape(expert: mapped.expertID)
                }
                try Self.requireDistinctSharedWeights(sharedWeights, from: [mapped.gateUp, mapped.down])
                mappedByID[mapped.expertID] = mapped
            }
            let weights = routingWeights.contents().assumingMemoryBound(to: Float.self)
            // The runner checks all rows when routes are created. Recheck each
            // complete row used by this chunk, without rescanning unused tokens.
            var referencedTokens = Set<Int>()
            for item in work {
                guard item.tokenIndex >= 0, item.tokenIndex < tokenCount,
                      item.routeRank >= 0, item.routeRank < configuration.topK else {
                    throw QwenMoEError.invalidAffineBinding("grouped token or route rank out of bounds")
                }
                referencedTokens.insert(item.tokenIndex)
            }
            for token in referencedTokens.sorted() {
                try Task.checkCancellation()
                var selected = Set<Int>()
                var sum: Float = 0
                for rank in 0..<configuration.topK {
                    let index = token * configuration.topK + rank
                    let expertID = routingExpertIDs[index]
                    guard expertID >= 0, expertID < configuration.expertCount else {
                        throw QwenMoEError.missingExpert(expertID)
                    }
                    guard selected.insert(expertID).inserted else {
                        throw QwenMoEError.duplicateExpertWithinToken(token: token, expert: expertID)
                    }
                    let weight = weights[index]
                    guard weight.isFinite, weight >= 0 else {
                        throw QwenMoEError.invalidAffineBinding("nonfinite/negative grouped routing weight")
                    }
                    sum += weight
                }
                guard sum.isFinite, abs(sum - 1) <= 0.001 else {
                    throw QwenMoEError.invalidAffineBinding("grouped Top-8 weights not normalized")
                }
            }
            // Preflight every byte offset and association before transferring
            // the lease. Strict expert/token order also rejects duplicate work.
            var offsets: [(row: Int, weight: Int)] = []
            offsets.reserveCapacity(work.count)
            var previous: QwenBF16GroupedExpertWork?
            for item in work {
                if let previous,
                   !(previous.expertID < item.expertID
                     || (previous.expertID == item.expertID && previous.tokenIndex < item.tokenIndex)) {
                    throw QwenMoEError.invalidAffineBinding("grouped work is not strictly expert/token ordered")
                }
                guard mappedByID[item.expertID] != nil else {
                    throw QwenMoEError.missingExpert(item.expertID)
                }
                let routeIndex = item.tokenIndex * configuration.topK + item.routeRank
                guard routingExpertIDs[routeIndex] == item.expertID else {
                    throw QwenMoEError.invalidAffineBinding("grouped expert differs from original route rank")
                }
                let rowOffset = try Self.checkedMultiply(
                    item.tokenIndex * configuration.hiddenSize, MemoryLayout<Float>.stride,
                    operation: "grouped row byte offset")
                let weightOffset = try Self.checkedMultiply(
                    routeIndex, MemoryLayout<Float>.stride, operation: "grouped weight byte offset")
                try Self.requireBuffer(hiddenRows, named: "grouped hidden row", offset: rowOffset,
                                       elements: configuration.hiddenSize, as: Float.self)
                try Self.requireBuffer(outputRows, named: "grouped output row", offset: rowOffset,
                                       elements: configuration.hiddenSize, as: Float.self)
                try Self.requireBuffer(routingWeights, named: "grouped rank weight", offset: weightOffset,
                                       elements: 1, as: Float.self)
                offsets.append((row: rowOffset, weight: weightOffset))
                previous = item
            }
            let gateSelection = try bf16RoutedPipeline(
                serialName: "qwen_moe_routed_gate_up_bf16",
                cooperativeName: "qwen_moe_routed_gate_up_bf16_cooperative",
                useCooperative: configuration.routedIntermediateSize >= 256
                    && configuration.hiddenSize >= 512 && configuration.hiddenSize.isMultiple(of: 64))
            let downSelection = try bf16RoutedPipeline(
                serialName: "qwen_moe_routed_down_add_bf16",
                cooperativeName: "qwen_moe_routed_down_add_bf16_cooperative",
                useCooperative: configuration.hiddenSize >= 512
                    && configuration.routedIntermediateSize >= 512
                    && configuration.routedIntermediateSize.isMultiple(of: 64))
            try requireDispatchable(gateSelection.pipeline, count: configuration.routedIntermediateSize)
            try requireDispatchable(downSelection.pipeline, count: configuration.hiddenSize)
            if initializeOutput { try requireDispatchable(clearPipeline, count: rowElements) }
            try Task.checkCancellation()
            try lease.submit(on: queue) { command in
                submittedCommand = command
                command.label = "qwen.moe.grouped-bf16"
                if initializeOutput {
                    try encodeClear(commandBuffer: command, buffer: outputRows, count: rowElements)
                }
                for (index, item) in work.enumerated() {
                    try Task.checkCancellation()
                    // No mapping from an earlier chunk is reused here.
                    guard let mapped = mappedByID[item.expertID] else {
                        throw QwenMoEError.missingExpert(item.expertID)
                    }
                    var parameters = BF16RoutedParameters(
                        hiddenSize: UInt32(configuration.hiddenSize),
                        intermediateSize: UInt32(configuration.routedIntermediateSize), scratchOffset: 0)
                    guard let gateEncoder = command.makeComputeCommandEncoder() else {
                        throw QwenMoEError.commandEncoderUnavailable
                    }
                    do {
                        defer { gateEncoder.endEncoding() }
                        gateEncoder.setComputePipelineState(gateSelection.pipeline)
                        gateEncoder.setBytes(&parameters, length: MemoryLayout<BF16RoutedParameters>.stride,
                                             index: QwenMetalBufferIndex.parameters.rawValue)
                        gateEncoder.setBuffer(hiddenRows, offset: offsets[index].row,
                                              index: QwenMetalBufferIndex.input.rawValue)
                        gateEncoder.setBuffer(mapped.gateUp, offset: 0,
                                              index: QwenMetalBufferIndex.weights.rawValue)
                        gateEncoder.setBuffer(scratch.routedActivation, offset: 0,
                                              index: QwenMetalBufferIndex.scratch.rawValue)
                        gateEncoder.useResource(mapped.gateUp, usage: .read)
                        try dispatchBF16Routed(gateEncoder, pipeline: gateSelection.pipeline,
                                               rows: configuration.routedIntermediateSize,
                                               cooperative: gateSelection.cooperative)
                    }
                    guard let downEncoder = command.makeComputeCommandEncoder() else {
                        throw QwenMoEError.commandEncoderUnavailable
                    }
                    do {
                        defer { downEncoder.endEncoding() }
                        downEncoder.setComputePipelineState(downSelection.pipeline)
                        downEncoder.setBytes(&parameters, length: MemoryLayout<BF16RoutedParameters>.stride,
                                             index: QwenMetalBufferIndex.parameters.rawValue)
                        downEncoder.setBuffer(scratch.routedActivation, offset: 0,
                                              index: QwenMetalBufferIndex.input.rawValue)
                        downEncoder.setBuffer(mapped.down, offset: 0,
                                              index: QwenMetalBufferIndex.weights.rawValue)
                        downEncoder.setBuffer(outputRows, offset: offsets[index].row,
                                              index: QwenMetalBufferIndex.output.rawValue)
                        downEncoder.setBuffer(routingWeights, offset: offsets[index].weight,
                                              index: QwenMetalBufferIndex.state.rawValue)
                        downEncoder.useResource(mapped.down, usage: .read)
                        try dispatchBF16Routed(downEncoder, pipeline: downSelection.pipeline,
                                               rows: configuration.hiddenSize,
                                               cooperative: downSelection.cooperative)
                    }
                }
                try Task.checkCancellation()
            }
        } catch {
            if let command = submittedCommand {
                // Await cleanup but preserve the original encoding error. If
                // settlement invariants fail, this throw still rejects output
                // and the coordinator keeps any unresolved slot pins reserved.
                try? await Self.settleGroupedCommand(command, lease: lease, buffers: retainedBuffers)
            } else {
                try? lease.cancel()
            }
            throw error
        }
        guard let command = submittedCommand else {
            throw MetalError.commandBufferFailed("grouped lease submitted without a captured command")
        }
        try await Self.settleGroupedCommand(command, lease: lease, buffers: retainedBuffers)
        try checkCommandBufferError(command)
        try Task.checkCancellation()
        guard lease.snapshot().succeeded == true else {
            throw QwenExpertMappingError.canceled
        }
    }

    private static func settleGroupedCommand(
        _ command: MTLCommandBuffer, lease: QwenBF16ExpertLease, buffers: [MTLBuffer],
        isolation: isolated (any Actor)? = #isolation
    ) async throws {
        // Unlike terminal status, waitUntilCompleted includes all completion
        // handlers. A background thread waits so no cooperative executor blocks.
        let completion = GroupedCommandCompletion(command: command, lease: lease, buffers: buffers)
        await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                DispatchQueue.global(qos: .userInitiated).async {
                    completion.wait()
                    continuation.resume()
                }
            }
        } onCancel: {
            try? lease.cancel()
        }
        let snapshot = lease.snapshot()
        guard snapshot.submitted, snapshot.completed else {
            throw MetalError.commandBufferFailed("grouped command finished before lease settlement")
        }
    }

    private func prepareSubmission(
        hidden: MTLBuffer,
        lease: QwenMappedExpertLease,
        routingWeights: MTLBuffer,
        sharedBindings: QwenMoESharedAffineBindings,
        output: MTLBuffer
    ) throws -> [RoutedWork] {
        try lease.requireUsable()
        guard lease.experts.count == configuration.topK else {
            throw QwenMoEError.invalidCount(
                field: "mappedExperts", expected: configuration.topK,
                actual: lease.experts.count)
        }
        try Self.requireBuffer(hidden, named: "hidden", elements: configuration.hiddenSize, as: Float.self)
        try Self.requireBuffer(
            routingWeights, named: "routingWeights", elements: configuration.topK, as: Float.self)
        try validateShared(hidden: hidden, bindings: sharedBindings, output: output)

        var work: [RoutedWork] = []
        work.reserveCapacity(lease.experts.count)
        for (rank, mapped) in lease.experts.enumerated() {
            try validate(mapped: mapped)
            work.append(RoutedWork(
                mapped: mapped,
                gateUp: try routedParameters(
                    descriptor: mapped.gateUp,
                    scratchOffset: rank * configuration.routedIntermediateSize,
                    intermediateSize: configuration.routedIntermediateSize),
                down: try routedParameters(
                    descriptor: mapped.down,
                    scratchOffset: rank * configuration.routedIntermediateSize,
                    intermediateSize: configuration.routedIntermediateSize)))
        }
        try requireDispatchable(clearPipeline, count: configuration.hiddenSize)
        try requireDispatchable(
            routedGateUpPipeline, count: configuration.routedIntermediateSize)
        try requireDispatchable(routedDownPipeline, count: configuration.hiddenSize)
        try validateSharedDispatches()
        return work
    }

    private func encodeRoutedCommands(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer,
        work: [RoutedWork],
        routingWeights: MTLBuffer,
        scratch: QwenMoEScratch,
        output: MTLBuffer
    ) throws {
        try encodeClear(commandBuffer: commandBuffer, buffer: output, count: configuration.hiddenSize)
        for (rank, item) in work.enumerated() {
            guard let gateUpEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw QwenMoEError.commandEncoderUnavailable
            }
            var gateUp = item.gateUp
            gateUpEncoder.setComputePipelineState(routedGateUpPipeline)
            gateUpEncoder.setBytes(&gateUp, length: MemoryLayout<RoutedParameters>.stride,
                                   index: QwenMetalBufferIndex.parameters.rawValue)
            gateUpEncoder.setBuffer(hidden, offset: 0, index: QwenMetalBufferIndex.input.rawValue)
            gateUpEncoder.setBuffer(item.mapped.buffer, offset: 0,
                                    index: QwenMetalBufferIndex.weights.rawValue)
            gateUpEncoder.setBuffer(scratch.routedActivation, offset: 0,
                                    index: QwenMetalBufferIndex.scratch.rawValue)
            gateUpEncoder.useResource(item.mapped.buffer, usage: .read)
            try dispatch(gateUpEncoder, pipeline: routedGateUpPipeline,
                         count: configuration.routedIntermediateSize)
            gateUpEncoder.endEncoding()

            guard let downEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw QwenMoEError.commandEncoderUnavailable
            }
            var down = item.down
            downEncoder.setComputePipelineState(routedDownPipeline)
            downEncoder.setBytes(&down, length: MemoryLayout<RoutedParameters>.stride,
                                 index: QwenMetalBufferIndex.parameters.rawValue)
            downEncoder.setBuffer(scratch.routedActivation, offset: 0,
                                  index: QwenMetalBufferIndex.input.rawValue)
            downEncoder.setBuffer(item.mapped.buffer, offset: 0,
                                  index: QwenMetalBufferIndex.weights.rawValue)
            downEncoder.setBuffer(output, offset: 0,
                                  index: QwenMetalBufferIndex.output.rawValue)
            downEncoder.setBuffer(
                routingWeights, offset: rank * MemoryLayout<Float>.stride,
                index: QwenMetalBufferIndex.state.rawValue)
            downEncoder.useResource(item.mapped.buffer, usage: .read)
            try dispatch(downEncoder, pipeline: routedDownPipeline,
                         count: configuration.hiddenSize)
            downEncoder.endEncoding()
        }
    }

    private func encodeSharedCommands(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer,
        bindings: QwenMoESharedAffineBindings,
        scratch: QwenMoEScratch,
        output: MTLBuffer
    ) throws {
        try encodeAffine(
            commandBuffer: commandBuffer, input: hidden, binding: bindings.gate,
            output: scratch.sharedGate)
        try encodeAffine(
            commandBuffer: commandBuffer, input: hidden, binding: bindings.up,
            output: scratch.sharedUp)
        try encodeActivation(
            commandBuffer: commandBuffer, gate: scratch.sharedGate,
            up: scratch.sharedUp, output: scratch.sharedActivation,
            count: configuration.sharedIntermediateSize)
        try encodeAffine(
            commandBuffer: commandBuffer, input: scratch.sharedActivation,
            binding: bindings.down, output: scratch.sharedOutput)
        try encodeAffine(
            commandBuffer: commandBuffer, input: hidden,
            binding: bindings.outputGate, output: scratch.sharedOutputGate)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QwenMoEError.commandEncoderUnavailable
        }
        var parameters = ElementParameters(elementCount: UInt32(configuration.hiddenSize))
        encoder.setComputePipelineState(sharedEpiloguePipeline)
        encoder.setBytes(&parameters, length: MemoryLayout<ElementParameters>.stride,
                         index: QwenMetalBufferIndex.parameters.rawValue)
        encoder.setBuffer(scratch.sharedOutputGate, offset: 0,
                          index: QwenMetalBufferIndex.weights.rawValue)
        encoder.setBuffer(output, offset: 0, index: QwenMetalBufferIndex.output.rawValue)
        encoder.setBuffer(scratch.sharedOutput, offset: 0,
                          index: QwenMetalBufferIndex.scratch.rawValue)
        try dispatch(encoder, pipeline: sharedEpiloguePipeline,
                     count: configuration.hiddenSize)
        encoder.endEncoding()
    }

    static func readRoutingDiagnostics(
        selectedExpertIDs: MTLBuffer,
        normalizedWeights: MTLBuffer,
        status: MTLBuffer,
        tokenCount: Int,
        configuration: QwenMoEConfiguration
    ) throws -> QwenMoERoutingDiagnostics {
        let selectedCount = try checkedMultiply(
            tokenCount, configuration.topK, operation: "routing diagnostics count")
        try requireBuffer(
            selectedExpertIDs, named: "selectedExpertIDs", elements: selectedCount,
            as: UInt32.self)
        try requireBuffer(
            normalizedWeights, named: "normalizedWeights", elements: selectedCount,
            as: Float.self)
        try requireBuffer(status, named: "routingStatus", elements: tokenCount, as: UInt32.self)
        let statuses = status.contents().bindMemory(to: UInt32.self, capacity: tokenCount)
        let ids = selectedExpertIDs.contents().bindMemory(to: UInt32.self, capacity: selectedCount)
        let weights = normalizedWeights.contents().bindMemory(to: Float.self, capacity: selectedCount)
        var allIDs: [[Int]] = []
        var allWeights: [[Float]] = []
        for token in 0..<tokenCount {
            guard statuses[token] == 0 else {
                throw QwenMoEError.routingKernelRejectedInput(token: token)
            }
            let range = token * configuration.topK..<(token + 1) * configuration.topK
            allIDs.append(range.map { Int(ids[$0]) })
            allWeights.append(range.map { weights[$0] })
        }
        return QwenMoERoutingDiagnostics(
            selectedExpertIDs: allIDs, normalizedWeights: allWeights,
            probabilities: [])
    }

    private func encodeClear(commandBuffer: MTLCommandBuffer,
                             buffer: MTLBuffer, count: Int) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QwenMoEError.commandEncoderUnavailable
        }
        var parameters = ElementParameters(elementCount: UInt32(count))
        encoder.setComputePipelineState(clearPipeline)
        encoder.setBytes(&parameters, length: MemoryLayout<ElementParameters>.stride,
                         index: QwenMetalBufferIndex.parameters.rawValue)
        encoder.setBuffer(buffer, offset: 0, index: QwenMetalBufferIndex.output.rawValue)
        try dispatch(encoder, pipeline: clearPipeline, count: count)
        encoder.endEncoding()
    }

    private func encodeAffine(commandBuffer: MTLCommandBuffer,
                              input: MTLBuffer,
                              binding: QwenMoEAffineBinding,
                              output: MTLBuffer) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QwenMoEError.commandEncoderUnavailable
        }
        var layout = binding.layout
        encoder.setComputePipelineState(affinePipeline)
        encoder.setBytes(&layout, length: MemoryLayout<QwenMetalAffineLayout>.stride,
                         index: QwenMetalBufferIndex.parameters.rawValue)
        encoder.setBuffer(input, offset: 0, index: QwenMetalBufferIndex.input.rawValue)
        encoder.setBuffer(binding.values, offset: binding.valuesOffset,
                          index: QwenMetalBufferIndex.weights.rawValue)
        encoder.setBuffer(binding.scales, offset: binding.scalesOffset,
                          index: QwenMetalBufferIndex.scales.rawValue)
        encoder.setBuffer(binding.biases, offset: binding.biasesOffset,
                          index: QwenMetalBufferIndex.biases.rawValue)
        encoder.setBuffer(output, offset: 0, index: QwenMetalBufferIndex.output.rawValue)
        try dispatch(encoder, pipeline: affinePipeline, count: Int(layout.rowCount))
        encoder.endEncoding()
    }

    private func encodeActivation(commandBuffer: MTLCommandBuffer,
                                  gate: MTLBuffer, up: MTLBuffer,
                                  output: MTLBuffer, count: Int) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QwenMoEError.commandEncoderUnavailable
        }
        var parameters = ElementParameters(elementCount: UInt32(count))
        encoder.setComputePipelineState(activationPipeline)
        encoder.setBytes(&parameters, length: MemoryLayout<ElementParameters>.stride,
                         index: QwenMetalBufferIndex.parameters.rawValue)
        encoder.setBuffer(gate, offset: 0, index: QwenMetalBufferIndex.input.rawValue)
        encoder.setBuffer(up, offset: 0, index: QwenMetalBufferIndex.weights.rawValue)
        encoder.setBuffer(output, offset: 0, index: QwenMetalBufferIndex.output.rawValue)
        try dispatch(encoder, pipeline: activationPipeline, count: count)
        encoder.endEncoding()
    }

    private func routedParameters(
        descriptor: PackedAffineDescriptor,
        scratchOffset: Int,
        intermediateSize: Int
    ) throws -> RoutedParameters {
        let columns = Int(descriptor.shape.last ?? 0)
        let rowStride = (columns + 1) / 2
        let groups = (columns + Quantization.groupSize - 1) / Quantization.groupSize
        return RoutedParameters(
            hiddenSize: UInt32(configuration.hiddenSize),
            intermediateSize: UInt32(intermediateSize),
            valuesOffset: try Self.uint32(descriptor.valuesOffset, field: "valuesOffset"),
            scalesOffset: try Self.uint32(descriptor.scalesOffset, field: "scalesOffset"),
            biasesOffset: try Self.uint32(descriptor.biasesOffset, field: "biasesOffset"),
            valuesRowStride: try Self.uint32(rowStride, field: "valuesRowStride"),
            groupsPerRow: try Self.uint32(groups, field: "groupsPerRow"),
            groupSize: UInt32(Quantization.groupSize),
            scratchOffset: try Self.uint32(scratchOffset, field: "scratchOffset"))
    }

    private func validate(mapped: QwenMappedExpert) throws {
        guard mapped.length >= expertStrideEnd(mapped.gateUp),
              mapped.length >= expertStrideEnd(mapped.down),
              mapped.gateUp.shape == [UInt32(configuration.routedIntermediateSize * 2),
                                      UInt32(configuration.hiddenSize)],
              mapped.down.shape == [UInt32(configuration.hiddenSize),
                                    UInt32(configuration.routedIntermediateSize)],
              mapped.gateUp.bitWidth == 4, mapped.down.bitWidth == 4 else {
            throw QwenMoEError.invalidExpertShape(expert: mapped.expertID)
        }
    }

    private func expertStrideEnd(_ descriptor: PackedAffineDescriptor) -> UInt64 {
        max(descriptor.valuesOffset + descriptor.valuesSize,
            descriptor.scalesOffset + descriptor.scalesSize,
            descriptor.biasesOffset + descriptor.biasesSize)
    }

    private func validateShared(hidden: MTLBuffer,
                                bindings: QwenMoESharedAffineBindings,
                                output: MTLBuffer) throws {
        try Self.requireBuffer(
            hidden, named: "hidden", elements: configuration.hiddenSize, as: Float.self)
        try Self.requireBuffer(
            output, named: "output", elements: configuration.hiddenSize, as: Float.self)
        try validate(binding: bindings.gate, rows: configuration.sharedIntermediateSize,
                     columns: configuration.hiddenSize, name: "sharedGate")
        try validate(binding: bindings.up, rows: configuration.sharedIntermediateSize,
                     columns: configuration.hiddenSize, name: "sharedUp")
        try validate(binding: bindings.down, rows: configuration.hiddenSize,
                     columns: configuration.sharedIntermediateSize, name: "sharedDown")
        try validate(binding: bindings.outputGate, rows: 1,
                     columns: configuration.hiddenSize, name: "sharedOutputGate")
    }

    private func validateSharedDispatches() throws {
        try requireDispatchable(affinePipeline, count: configuration.sharedIntermediateSize)
        try requireDispatchable(affinePipeline, count: configuration.hiddenSize)
        try requireDispatchable(affinePipeline, count: 1)
        try requireDispatchable(activationPipeline, count: configuration.sharedIntermediateSize)
        try requireDispatchable(sharedEpiloguePipeline, count: configuration.hiddenSize)
    }

    private func validate(binding: QwenMoEAffineBinding,
                          rows: Int, columns: Int, name: String) throws {
        guard binding.layout.rowCount == UInt32(rows),
              binding.layout.columnCount == UInt32(columns),
              binding.layout.groupSize == UInt32(QwenMetalABI.affineGroupSize),
              binding.layout.bitWidth == 4 || binding.layout.bitWidth == 8,
              binding.valuesOffset >= 0, binding.scalesOffset >= 0,
              binding.biasesOffset >= 0 else {
            throw QwenMoEError.invalidAffineBinding(name)
        }
        try Self.requireBuffer(
            binding.values, named: "\(name).values", offset: binding.valuesOffset,
            bytes: Int(binding.layout.rowCount) * Int(binding.layout.valuesRowStrideBytes))
        try Self.requireBuffer(
            binding.scales, named: "\(name).scales", offset: binding.scalesOffset,
            bytes: Int(binding.layout.rowCount) * Int(binding.layout.metadataRowStrideBytes))
        try Self.requireBuffer(
            binding.biases, named: "\(name).biases", offset: binding.biasesOffset,
            bytes: Int(binding.layout.rowCount) * Int(binding.layout.metadataRowStrideBytes))
    }

    private func requireNotSubmitted(_ commandBuffer: MTLCommandBuffer) throws {
        guard commandBuffer.status == .notEnqueued else {
            throw QwenMoEError.commandBufferAlreadySubmitted
        }
    }

    private func requireDispatchable(_ pipeline: MTLComputePipelineState,
                                     count: Int) throws {
        guard count > 0, pipeline.maxTotalThreadsPerThreadgroup > 0,
              pipeline.threadExecutionWidth > 0 else {
            throw QwenMoEError.invalidPipelineLimit
        }
    }

    private func dispatchBF16Routed(_ encoder: MTLComputeCommandEncoder,
                                    pipeline: MTLComputePipelineState,
                                    rows: Int, cooperative: Bool) throws {
        if cooperative {
            guard rows > 0, pipeline.threadExecutionWidth == 32,
                  pipeline.maxTotalThreadsPerThreadgroup >= 32 else {
                throw QwenMoEError.invalidPipelineLimit
            }
            encoder.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        } else {
            try dispatch(encoder, pipeline: pipeline, count: rows)
        }
    }

    private func dispatch(_ encoder: MTLComputeCommandEncoder,
                          pipeline: MTLComputePipelineState, count: Int) throws {
        try requireDispatchable(pipeline, count: count)
        let width = min(pipeline.maxTotalThreadsPerThreadgroup,
                        max(1, pipeline.threadExecutionWidth))
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(width, count), height: 1, depth: 1))
    }

    private static func validate(expert: QwenMoEDenseExpert, expertID: Int,
                                 configuration: QwenMoEConfiguration) throws {
        let gateUpCount = try checkedMultiply(
            2 * configuration.routedIntermediateSize, configuration.hiddenSize,
            operation: "routed gate_up count")
        let downCount = try checkedMultiply(
            configuration.hiddenSize, configuration.routedIntermediateSize,
            operation: "routed down count")
        guard expert.gateUp.count == gateUpCount, expert.down.count == downCount else {
            throw QwenMoEError.invalidExpertShape(expert: expertID)
        }
    }

    private static func validate(sharedExpert: QwenMoEDenseSharedExpert,
                                 configuration: QwenMoEConfiguration) throws {
        let firstCount = try checkedMultiply(
            configuration.sharedIntermediateSize, configuration.hiddenSize,
            operation: "shared gate/up count")
        let downCount = try checkedMultiply(
            configuration.hiddenSize, configuration.sharedIntermediateSize,
            operation: "shared down count")
        guard sharedExpert.gate.count == firstCount,
              sharedExpert.up.count == firstCount,
              sharedExpert.down.count == downCount,
              sharedExpert.outputGate.count == configuration.hiddenSize else {
            throw QwenMoEError.invalidCount(
                field: "sharedExpert", expected: firstCount,
                actual: sharedExpert.gate.count)
        }
    }

    private static func project(_ input: [Float], matrix: [Float],
                                rows: Int, columns: Int) -> [Float] {
        var output = [Float](repeating: 0, count: rows)
        for row in 0..<rows {
            var sum: Float = 0
            let base = row * columns
            for column in 0..<columns {
                sum += matrix[base + column] * input[column]
            }
            output[row] = sum
        }
        return output
    }

    private static func silu(_ value: Float) -> Float {
        value / (1 + Foundation.exp(-value))
    }

    private static func sigmoid(_ value: Float) -> Float {
        1 / (1 + Foundation.exp(-value))
    }

    private static func uint32(_ value: Int, field: String) throws -> UInt32 {
        guard let result = UInt32(exactly: value) else {
            throw QwenMoEError.invalidConfiguration(field: field, value: value)
        }
        return result
    }

    private static func uint32(_ value: UInt64, field: String) throws -> UInt32 {
        guard let result = UInt32(exactly: value) else {
            throw QwenMoEError.arithmeticOverflow(operation: field)
        }
        return result
    }

    private static func checkedMultiply(_ lhs: Int, _ rhs: Int,
                                        operation: String) throws -> Int {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else { throw QwenMoEError.arithmeticOverflow(operation: operation) }
        return result
    }

    private static func requireBuffer<T>(
        _ buffer: MTLBuffer, named name: String, offset: Int = 0,
        elements: Int, as _: T.Type
    ) throws {
        let bytes = try checkedMultiply(elements, MemoryLayout<T>.stride, operation: "\(name) bytes")
        try requireBuffer(buffer, named: name, offset: offset, bytes: bytes)
    }

    private static func requireBuffer(
        _ buffer: MTLBuffer, named name: String, offset: Int = 0, bytes: Int
    ) throws {
        guard offset >= 0, bytes >= 0, offset <= buffer.length,
              bytes <= buffer.length - offset else {
            throw QwenMoEError.bufferTooSmall(
                name: name, required: max(0, offset) + max(0, bytes), actual: buffer.length)
        }
    }
}
