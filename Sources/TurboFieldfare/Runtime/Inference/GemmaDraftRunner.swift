import Foundation
import Metal

/// Proposes tokens. The main model must verify every proposal before use.
final class GemmaDraftRunner {
    struct AttentionState {
        let keys: MTLBuffer
        let values: MTLBuffer
        let count: UInt32
        let ringCapacity: UInt32
    }

    struct Context {
        let embedding: TensorView
        let normalizedHidden: MTLBuffer
        let hiddenOffset: Int
        let sliding: AttentionState
        let full: AttentionState
        let position: UInt32
    }

    private let context: MetalContext
    private let weights: GemmaDraftWeights
    var weightBuffer: MTLBuffer { weights.buffer }
    private let int4: DequantInt4GEMV
    private let rms: RMSNorm
    private let rope: RoPE
    private let attention: Attention
    private let shared: SharedExpertInt4
    private let head: LMHeadChainInt4
    private let embeddingPSO: MTLComputePipelineState
    private let addPSO: MTLComputePipelineState
    private let concatenated: MTLBuffer
    private let projected: MTLBuffer
    private let hidden: MTLBuffer
    private let normed: MTLBuffer
    private let query: MTLBuffer
    private let attended: MTLBuffer
    private let branch: MTLBuffer
    private let gateScratch: MTLBuffer
    private let upScratch: MTLBuffer
    private let activationScratch: MTLBuffer
    private let seed: MTLBuffer
    private let proposals: [MTLBuffer]
    private let layerScales: [Float]
    private(set) var lastGPUSeconds: Double = 0

    init(context: MetalContext, weights: GemmaDraftWeights) throws {
        self.context = context
        self.weights = weights
        int4 = try DequantInt4GEMV(context: context)
        rms = try RMSNorm(context: context)
        rope = try RoPE(context: context)
        attention = try Attention(context: context)
        shared = try SharedExpertInt4(context: context)
        head = try LMHeadChainInt4(context: context, maxD: 1024)
        embeddingPSO = try context.pipeline("embed_lookup_int4")
        addPSO = try context.pipeline("gemma_draft_add_scaled")
        func buffer(_ bytes: Int) throws -> MTLBuffer {
            try Self.makeBuffer(device: context.device, bytes: bytes)
        }
        concatenated = try buffer(5632 * 2)
        projected = try buffer(2816 * 2)
        hidden = try buffer(1024 * 2)
        normed = try buffer(1024 * 2)
        query = try buffer(8192 * 2)
        attended = try buffer(8192 * 2)
        branch = try buffer(1024 * 2)
        gateScratch = try buffer(8192 * 2)
        upScratch = try buffer(8192 * 2)
        activationScratch = try buffer(8192 * 2)
        seed = try buffer(4)
        proposals = try (0..<4).map { _ in try buffer(4) }
        layerScales = (0..<4).map { layer in
            let tensor = weights.tensor("model.layers.\(layer).layer_scalar")
            let bits = weights.buffer.contents().advanced(by: tensor.offset).load(as: UInt16.self)
            return Quantization.bf16ToFloat(bits)
        }
    }

    func draft(after token: Int32, count: Int, using state: Context) throws -> [Int32] {
        guard (1...4).contains(count), token >= 0, token < 262144,
              state.sliding.count > 0, state.full.count > 0,
              state.hiddenOffset >= 0,
              state.hiddenOffset <= state.normalizedHidden.length,
              2816 * 2 <= state.normalizedHidden.length - state.hiddenOffset else {
            throw ModelError.indexCorrupt(detail: "Invalid draft context")
        }
        try Task.checkCancellation()
        guard let command = context.queue.makeCommandBuffer() else { throw MetalError.noQueue }
        command.label = "Gemma draft \(count) tokens"
        seed.contents().storeBytes(of: UInt32(token), as: UInt32.self)
        for step in 0..<count {
            try encodeInput(command: command, token: step == 0 ? seed : proposals[step - 1],
                            hiddenState: step == 0 ? state.normalizedHidden : projected,
                            hiddenOffset: step == 0 ? state.hiddenOffset : 0,
                            embedding: state.embedding)
            project(command: command, name: "pre_projection.weight", input: concatenated, output: hidden)
            for layer in 0..<4 {
                try encodeLayer(command: command, layer: layer, state: state)
            }
            if step + 1 < count {
                normalize(command: command, name: "model.norm.weight", input: hidden, output: normed)
                project(command: command, name: "post_projection.weight", input: normed, output: projected)
            }
            let output = weights.tensor("model.embed_tokens.weight")
            let norm = weights.tensor("model.norm.weight")
            head.encodeGreedyDecode(commandBuffer: command, hidden: hidden,
                normWeight: weights.buffer, normOffset: norm.offset,
                weights: weights.buffer, weightsOffset: output.offset,
                scales: weights.buffer, scalesOffset: output.scaleOffset,
                biases: weights.buffer, biasesOffset: output.biasOffset,
                outToken: proposals[step], d: 1024, vocab: 262144)
        }
        command.commit()
        command.waitUntilCompleted()
        try checkCommandBufferError(command)
        try Task.checkCancellation()
        lastGPUSeconds = max(0, command.gpuEndTime - command.gpuStartTime)
        let tokens = proposals.prefix(count).map { $0.contents().load(as: UInt32.self) }
        guard tokens.allSatisfy({ $0 < 262144 }) else {
            throw ModelError.indexCorrupt(detail: "Invalid draft token")
        }
        return tokens.map { Int32($0) }
    }

    private func encodeInput(command: MTLCommandBuffer, token: MTLBuffer,
                             hiddenState: MTLBuffer, hiddenOffset: Int,
                             embedding: TensorView) throws {
        guard let encoder = command.makeComputeCommandEncoder() else { throw MetalError.noQueue }
        encoder.setComputePipelineState(embeddingPSO)
        encoder.setBuffer(embedding.buffer, offset: Int(embedding.offset), index: 0)
        encoder.setBuffer(embedding.buffer, offset: Int(embedding.scaleOffset), index: 1)
        encoder.setBuffer(embedding.buffer, offset: Int(embedding.biasOffset), index: 2)
        encoder.setBuffer(concatenated, offset: 0, index: 3)
        encoder.setBuffer(token, offset: 0, index: 4)
        var dimension: UInt32 = 2816
        var scale = sqrt(Float(dimension))
        encoder.setBytes(&dimension, length: 4, index: 5)
        encoder.setBytes(&scale, length: 4, index: 6)
        encoder.dispatchThreads(MTLSize(width: 2816, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        guard let copy = command.makeBlitCommandEncoder() else { throw MetalError.noQueue }
        copy.copy(from: hiddenState, sourceOffset: hiddenOffset, to: concatenated,
                  destinationOffset: 2816 * 2, size: 2816 * 2)
        copy.endEncoding()
    }

    private func encodeLayer(command: MTLCommandBuffer, layer: Int, state: Context) throws {
        let prefix = "model.layers.\(layer)."
        let full = layer == 3
        let headDimension: UInt32 = full ? 512 : 256
        normalize(command: command, name: prefix + "input_layernorm.weight", input: hidden, output: normed)
        project(command: command, name: prefix + "self_attn.q_proj.weight", input: normed, output: query)
        let qNorm = weights.tensor(prefix + "self_attn.q_norm.weight")
        rms.encodeBF16WPerHead(commandBuffer: command, x: query,
                              weight: weights.buffer, weightOffset: qNorm.offset, out: query,
                              headDim: headDimension, numHeads: 16, eps: 1e-6)
        // The official drafter keeps this position fixed across its proposals.
        if full {
            rope.encodeProportionalNeox(commandBuffer: command, data: query,
                position: state.position, headDim: 512, numHeads: 16, rotatedPairs: 64)
            attention.encodeFull(commandBuffer: command, q: query,
                k: state.full.keys, v: state.full.values, out: attended,
                headDim: 512, numQHeads: 16, numKVHeads: 2, seqLen: state.full.count, scale: 1)
        } else {
            rope.encodeDefaultNeox(commandBuffer: command, data: query,
                position: state.position, headDim: 256, numHeads: 16)
            attention.encodeSWA(commandBuffer: command, q: query,
                k: state.sliding.keys, v: state.sliding.values, out: attended,
                headDim: 256, numQHeads: 16, numKVHeads: 8, seqLen: state.sliding.count,
                window: 1024, scale: 1, ringCapacity: state.sliding.ringCapacity)
        }
        project(command: command, name: prefix + "self_attn.o_proj.weight", input: attended, output: branch)
        normalize(command: command, name: prefix + "post_attention_layernorm.weight", input: branch, output: branch)
        try add(command: command, scale: 1)
        normalize(command: command, name: prefix + "pre_feedforward_layernorm.weight", input: hidden, output: normed)
        try shared.encode(commandBuffer: command, x: normed,
            gate: projection(prefix + "mlp.gate_proj.weight"),
            up: projection(prefix + "mlp.up_proj.weight"),
            down: projection(prefix + "mlp.down_proj.weight"), y: branch,
            scratchGate: gateScratch, scratchUp: upScratch, scratchAct: activationScratch)
        normalize(command: command, name: prefix + "post_feedforward_layernorm.weight", input: branch, output: branch)
        try add(command: command, scale: layerScales[layer])
    }

    private func normalize(command: MTLCommandBuffer, name: String,
                           input: MTLBuffer, output: MTLBuffer) {
        let tensor = weights.tensor(name)
        rms.encodeBF16W(commandBuffer: command, x: input, weight: weights.buffer,
                       weightOffset: tensor.offset, out: output, d: 1024, eps: 1e-6)
    }

    private func project(command: MTLCommandBuffer, name: String,
                         input: MTLBuffer, output: MTLBuffer) {
        let tensor = weights.tensor(name)
        int4.encode(commandBuffer: command, weights: weights.buffer, weightsOffset: tensor.offset,
                    scales: weights.buffer, scalesOffset: tensor.scaleOffset,
                    biases: weights.buffer, biasesOffset: tensor.biasOffset,
                    x: input, y: output, m: UInt32(tensor.shape[0]), n: UInt32(tensor.shape[1]))
    }

    private func projection(_ name: String) -> SharedExpertProjection {
        let tensor = weights.tensor(name)
        return SharedExpertProjection(weights: weights.buffer, scales: weights.buffer, biases: weights.buffer,
            weightsOffset: tensor.offset, scalesOffset: tensor.scaleOffset, biasesOffset: tensor.biasOffset,
            rows: UInt32(tensor.shape[0]), cols: UInt32(tensor.shape[1]))
    }

    private func add(command: MTLCommandBuffer, scale: Float) throws {
        guard let encoder = command.makeComputeCommandEncoder() else { throw MetalError.noQueue }
        encoder.setComputePipelineState(addPSO)
        encoder.setBuffer(hidden, offset: 0, index: 0)
        encoder.setBuffer(branch, offset: 0, index: 1)
        var scale = scale
        encoder.setBytes(&scale, length: 4, index: 2)
        encoder.dispatchThreads(MTLSize(width: 1024, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private static func makeBuffer(device: MTLDevice, bytes: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        return buffer
    }
}
