import Foundation
import Metal


struct AttentionSplitGeometry: Sendable, Equatable {
    let effectiveLength: Int
    let numChunks: Int
    let chunkLength: Int
    let partialThreadgroups: Int
    let useSWAGroupedPartial: Bool
}


/// Swift wrapper for sliding-window and full-causal decode attention.
///
/// The kernels assume a single decoded query token (`M_q = 1`) and a
/// contiguous KV cache of length `seqLen`. The MPP-tensor-core prefill path
/// (`M_q > 1`) is separate.
///
/// Buffer contracts (FP16 throughout):
///   - `q`   : `[numQHeads, headDim]`
///   - `k`   : `[seqLen, numKVHeads, headDim]`
///   - `v`   : same shape as `k`. Full-layer K and V must remain distinct after
///             their separate per-head normalization and RoPE paths.
///   - `out` : `[numQHeads, headDim]`
final class Attention {
    private let ctx: MetalContext
    private let retainSWAQuery: Bool
    private let useQueryPair: Bool
    var useTokenPairReuse = false
    private let psoPartial: MTLComputePipelineState
    private let psoGQAPartial: MTLComputePipelineState
    private let psoCombine: MTLComputePipelineState
    private let psoPartialSWA: MTLComputePipelineState
    private let psoPartialFull: MTLComputePipelineState
    private let psoGQAPartialSWA: MTLComputePipelineState
    private let psoGQAPartialSWAChunks16: MTLComputePipelineState
    private let psoPartialFullChunks16: MTLComputePipelineState
    private let psoPartialFullQueryRegisters: MTLComputePipelineState
    private let psoPartialFullChunks16QueryRegisters: MTLComputePipelineState
    private let psoCombineSWA: MTLComputePipelineState
    private let psoCombineFull: MTLComputePipelineState
    private let psoCombineSWAChunks16: MTLComputePipelineState
    private let psoCombineFullChunks16: MTLComputePipelineState

    /// Matches the 256-thread group used by the Metal scratch arrays.
    static let threadsPerGroup: Int = 256

    /// Project ceilings for the split-KV partial scratch. `kAttnMaxHeadDim` in
    /// attention.metal is 512; the model has 16 Q heads; `maxChunks` bounds the
    /// split factor (and therefore the scratch size: 16·64·512 FP32 ≈ 2 MB).
    static let maxQHeads = 16
    static let maxHeadDim = 512
    static let maxChunks = 64
    /// Full attention uses 16 base chunks by default.
    private static let defaultFullChunks = 16
    private static let defaultGQASWAChunks = 8

    // Partial state written by pass 1, read by pass 2. One shared allocation:
    // attention runs once per layer, serially, and pass 2 hazard-tracks pass 1
    // within the same command buffer — no race (mirrors MoE.routerLogits).
    private let mPartial: MTLBuffer
    private let dPartial: MTLBuffer
    private let oPartial: MTLBuffer

    init(context: MetalContext, retainSWAQuery: Bool = true, useQueryPair: Bool = true) throws {
        self.ctx = context
        self.retainSWAQuery = retainSWAQuery
        self.useQueryPair = useQueryPair
        self.psoPartial = try context.pipeline("attention_decode_partial")
        self.psoGQAPartial = try context.pipeline("attention_decode_gqa_swa_partial")
        self.psoCombine = try context.pipeline("attention_decode_combine")
        self.psoPartialSWA = try Self.specializedPipeline(context,
                                                          "attention_decode_partial",
                                                          headDim: 256,
                                                          numQHeads: 16,
                                                          numKVHeads: 8)
        self.psoPartialFull = try Self.specializedPipeline(context,
                                                           "attention_decode_partial",
                                                           headDim: 512,
                                                           numQHeads: 16,
                                                           numKVHeads: 2)
        self.psoGQAPartialSWA = try Self.specializedPipeline(context,
                                                             "attention_decode_gqa_swa_partial",
                                                             headDim: 256,
                                                             numQHeads: 16,
                                                             numKVHeads: 8)
        self.psoGQAPartialSWAChunks16 = try Self.specializedPipeline(context,
                                                                     "attention_decode_gqa_swa_partial",
                                                                     headDim: 256,
                                                                     numQHeads: 16,
                                                                     numKVHeads: 8,
                                                                     numChunks: 16)
        self.psoPartialFullChunks16 = try Self.specializedPipeline(context,
                                                                   "attention_decode_partial",
                                                                   headDim: 512,
                                                                   numQHeads: 16,
                                                                   numKVHeads: 2,
                                                                   numChunks: 16)
        self.psoPartialFullQueryRegisters = try Self.specializedPipeline(context,
            "attention_decode_partial", headDim: 512, numQHeads: 16, numKVHeads: 2,
            retainQuery: true, prefetchValue: true)
        self.psoPartialFullChunks16QueryRegisters = try Self.specializedPipeline(context,
            "attention_decode_partial", headDim: 512, numQHeads: 16, numKVHeads: 2,
            numChunks: 16, retainQuery: true, prefetchValue: true)
        self.psoCombineSWA = try Self.specializedPipeline(context,
                                                          "attention_decode_combine",
                                                          headDim: 256,
                                                          numQHeads: 16,
                                                          numKVHeads: 8)
        self.psoCombineFull = try Self.specializedPipeline(context,
                                                           "attention_decode_combine",
                                                           headDim: 512,
                                                           numQHeads: 16,
                                                           numKVHeads: 2)
        self.psoCombineSWAChunks16 = try Self.specializedPipeline(context,
                                                                  "attention_decode_combine",
                                                                  headDim: 256,
                                                                  numQHeads: 16,
                                                                  numKVHeads: 8,
                                                                  numChunks: 16)
        self.psoCombineFullChunks16 = try Self.specializedPipeline(context,
                                                                   "attention_decode_combine",
                                                                   headDim: 512,
                                                                   numQHeads: 16,
                                                                   numKVHeads: 2,
                                                                   numChunks: 16)
        let md = Self.maxQHeads * Self.maxChunks
        guard let m = context.device.makeBuffer(length: md * MemoryLayout<Float>.size,
                                                options: .storageModeShared),
	              let d = context.device.makeBuffer(length: md * MemoryLayout<Float>.size,
	                                                options: .storageModeShared),
	              let o = context.device.makeBuffer(length: md * Self.maxHeadDim * MemoryLayout<Float>.size,
	                                                options: .storageModeShared) else {
            throw MetalError.missingFunction("attention split-KV scratch")
        }
        self.mPartial = m; self.dPartial = d; self.oPartial = o
    }

    /// Number of K/V chunks for a range of `effLen` positions — the split
    /// factor used by the production split path.
    static func chunkCount(effLen: Int, preferGQASWA: Bool = false) -> Int {
        let eff = max(1, effLen)
        let defaultChunks = preferGQASWA ? defaultGQASWAChunks : defaultFullChunks
        return max(1, min(defaultChunks, min(maxChunks, eff)))
    }

    static func splitGeometry(numQHeads: UInt32,
                                     numKVHeads: UInt32,
                                     seqLen: UInt32,
                                     kvStart: UInt32,
                                     preferGQASWA: Bool) -> AttentionSplitGeometry {
        let qPerKV = Int(numQHeads / numKVHeads)
        let useSWAGQAPartial = preferGQASWA && qPerKV <= 2
        let effectiveLength = Int(seqLen) - Int(kvStart)
        let baseChunks = Self.chunkCount(effLen: effectiveLength,
                                         preferGQASWA: useSWAGQAPartial)
        let numChunks = useSWAGQAPartial
            ? max(baseChunks, min(Self.maxChunks, baseChunks * qPerKV))
            : baseChunks
        let chunkLength = (max(1, effectiveLength) + numChunks - 1) / numChunks
        let partialHeadGroups = useSWAGQAPartial ? Int(numKVHeads) : Int(numQHeads)
        return AttentionSplitGeometry(effectiveLength: effectiveLength,
                                      numChunks: numChunks,
                                      chunkLength: chunkLength,
                                      partialThreadgroups: partialHeadGroups * numChunks,
                                      useSWAGroupedPartial: useSWAGQAPartial)
    }


    /// Sliding-window attention. `window` caps the K/V positions to the most
    /// recent `window` entries (`[max(0, seqLen-window), seqLen)`).
    /// `scale` defaults to `rsqrt(head_dim)` for generic callers;
    /// Gemma 4 callers pass 1.0 because the model's configured attention scale
    /// is 1.0.
    func encodeSWA(commandBuffer: MTLCommandBuffer,
                          q: MTLBuffer, qOffset: Int = 0,
                          k: MTLBuffer, kOffset: Int = 0,
                          v: MTLBuffer, vOffset: Int = 0,
                          out: MTLBuffer, outOffset: Int = 0,
                          headDim: UInt32,
                          numQHeads: UInt32,
                          numKVHeads: UInt32,
                          seqLen: UInt32,
                          window: UInt32,
                          scale: Float? = nil,
                          ringCapacity: UInt32 = 0,
                          conditional: DecodeDispatch? = nil) {
        precondition(numQHeads % numKVHeads == 0,
                     "numQHeads must be a multiple of numKVHeads for GQA")
        precondition(headDim <= 512,
                     "head_dim must be <= 512 (kernel scratch is sized for the full-attn case)")
        let sc = scale ?? Self.defaultScale(headDim: headDim)
        let kvStart = seqLen > window ? seqLen - window : 0

        encodeSplit(commandBuffer: commandBuffer,
                    q: q, qOffset: qOffset, k: k, kOffset: kOffset,
                    v: v, vOffset: vOffset, out: out, outOffset: outOffset,
                    headDim: headDim, numQHeads: numQHeads, numKVHeads: numKVHeads,
                    seqLen: seqLen, kvStart: kvStart, scale: sc,
                    preferGQASWA: true,
                    ringCapacity: ringCapacity, conditional: conditional)
    }

    /// Full attention. Gemma 4 reuses the raw K projection as raw V input, but
    /// separate normalization and RoPE make the cache streams distinct here.
    /// `scale` mirrors `encodeSWA`.
    func encodeFull(commandBuffer: MTLCommandBuffer,
                           q: MTLBuffer, qOffset: Int = 0,
                           k: MTLBuffer, kOffset: Int = 0,
                           v: MTLBuffer, vOffset: Int = 0,
                           out: MTLBuffer, outOffset: Int = 0,
                           headDim: UInt32,
                           numQHeads: UInt32,
                           numKVHeads: UInt32,
                           seqLen: UInt32,
                           scale: Float? = nil,
                           stageTiming: FullAttentionStageTiming? = nil,
                           conditional: DecodeDispatch? = nil) {
        precondition(numQHeads % numKVHeads == 0,
                     "numQHeads must be a multiple of numKVHeads for GQA")
        precondition(headDim <= 512,
                     "head_dim must be <= 512 (kernel scratch is sized for the full-attn case)")
        precondition(seqLen > 0, "full attention requires at least one KV position")
        let sc = scale ?? Self.defaultScale(headDim: headDim)


        encodeSplit(commandBuffer: commandBuffer,
                    q: q, qOffset: qOffset, k: k, kOffset: kOffset,
                    v: v, vOffset: vOffset, out: out, outOffset: outOffset,
                    headDim: headDim, numQHeads: numQHeads, numKVHeads: numKVHeads,
                    seqLen: seqLen, kvStart: 0, scale: sc,
                    preferGQASWA: false,
                    stageTiming: stageTiming, conditional: conditional)
    }


    /// Two-pass split-KV (Flash-Decoding) dispatch shared by SWA and full
    /// attention — they differ only by `kvStart`. Pass 1 fans the head's
    /// `[kvStart, seqLen)` range across `chunkCount` threadgroups per head;
    /// pass 2 merges the partials. Both encoders go on the same command buffer
    /// so pass 2 hazard-tracks the partial scratch written by pass 1.
    private func encodeSplit(commandBuffer: MTLCommandBuffer,
                             q: MTLBuffer, qOffset: Int,
                             k: MTLBuffer, kOffset: Int,
                             v: MTLBuffer, vOffset: Int,
                             out: MTLBuffer, outOffset: Int,
                             headDim: UInt32, numQHeads: UInt32, numKVHeads: UInt32,
                             seqLen: UInt32, kvStart: UInt32, scale: Float,
                             preferGQASWA: Bool,
                             ringCapacity: UInt32 = 0,
                             stageTiming: FullAttentionStageTiming? = nil,
                             conditional: DecodeDispatch? = nil) {
        precondition(Int(numQHeads) <= Self.maxQHeads,
                     "numQHeads \(numQHeads) exceeds split-KV scratch (max \(Self.maxQHeads))")
        precondition(Int(headDim) <= Self.maxHeadDim,
                     "head_dim \(headDim) exceeds split-KV scratch (max \(Self.maxHeadDim))")
        precondition(ringCapacity == 0 || preferGQASWA,
                     "FP16 KV ring is only valid for SWA attention")
        let geometry = Self.splitGeometry(numQHeads: numQHeads,
                                          numKVHeads: numKVHeads,
                                          seqLen: seqLen,
                                          kvStart: kvStart,
                                          preferGQASWA: preferGQASWA)
        let useSWAGQAPartial = geometry.useSWAGroupedPartial
        let nChunks = geometry.numChunks
        let chunkLen = geometry.chunkLength
        var partialPSO = partialPipeline(headDim: headDim,
                                         numQHeads: numQHeads,
                                         numKVHeads: numKVHeads,
                                         numChunks: nChunks,
                                         useGQAPartial: useSWAGQAPartial,
                                         ringCapacity: ringCapacity)
        var tgWidth = min(Self.threadsPerGroup, Int(partialPSO.maxTotalThreadsPerThreadgroup))
        var partialGroups = geometry.partialThreadgroups
        // Keep the baseline's exact width. A register variant is eligible only
        // for full attention when its compiled limit also supports 256 threads.
        if !useQueryPair, !preferGQASWA, headDim == 512, numQHeads == 16, numKVHeads == 2,
           tgWidth == Self.threadsPerGroup {
            let candidate = nChunks == 16
                ? psoPartialFullChunks16QueryRegisters : psoPartialFullQueryRegisters
            if candidate.maxTotalThreadsPerThreadgroup >= tgWidth {
                partialPSO = candidate
            }
        }
        if !useQueryPair, retainSWAQuery, useSWAGQAPartial, headDim == 256,
           numQHeads == 16, numKVHeads == 8, tgWidth == Self.threadsPerGroup {
            let candidate = try? Self.specializedPipeline(ctx,
                "attention_decode_gqa_swa_partial", headDim: headDim,
                numQHeads: numQHeads, numKVHeads: numKVHeads,
                numChunks: nChunks == 16 ? 16 : nil,
                ringCapacity: ringCapacity > 0 ? ringCapacity : nil,
                retainQuery: true, prefetchValue: true)
            if let candidate, candidate.maxTotalThreadsPerThreadgroup >= tgWidth {
                partialPSO = candidate
            }
        }

        let queryPairShape = (useSWAGQAPartial && headDim == 256 && numKVHeads == 8)
            || (!preferGQASWA && headDim == 512 && numKVHeads == 2)
        if useQueryPair, numQHeads == 16, queryPairShape, tgWidth == Self.threadsPerGroup {
            let candidate = try? Self.specializedPipeline(ctx, "attention_decode_query_pair_partial",
                headDim: headDim, numQHeads: numQHeads, numKVHeads: numKVHeads,
                numChunks: nChunks == 16 ? 16 : nil,
                ringCapacity: ringCapacity > 0 ? ringCapacity : nil)
            let width = Int(headDim) / 2
            if let candidate, candidate.maxTotalThreadsPerThreadgroup >= width {
                partialPSO = candidate
                tgWidth = width
                partialGroups = Int(numQHeads) / 2 * nChunks
            }
        }

        let partialEncoder: MTLComputeCommandEncoder?
        if let stageTiming {
            partialEncoder = stageTiming.makeEncoder(commandBuffer: commandBuffer, pass: .partial)
        } else {
            partialEncoder = commandBuffer.makeComputeCommandEncoder()
        }
        guard let p1 = partialEncoder else { return }
        p1.setComputePipelineState(partialPSO)
        p1.setBuffer(q, offset: qOffset, index: 0)
        p1.setBuffer(k, offset: kOffset, index: 1)
        p1.setBuffer(v, offset: vOffset, index: 2)
        p1.setBuffer(mPartial, offset: 0, index: 3)
        p1.setBuffer(dPartial, offset: 0, index: 4)
        p1.setBuffer(oPartial, offset: 0, index: 5)
        var hd = headDim, nq = numQHeads, nkv = numKVHeads, sl = seqLen, ks = kvStart
        var cl = UInt32(chunkLen), nc = UInt32(nChunks), sc = scale
        p1.setBytes(&hd,  length: MemoryLayout<UInt32>.size, index: 6)
        p1.setBytes(&nq,  length: MemoryLayout<UInt32>.size, index: 7)
        p1.setBytes(&nkv, length: MemoryLayout<UInt32>.size, index: 8)
        p1.setBytes(&sl,  length: MemoryLayout<UInt32>.size, index: 9)
        p1.setBytes(&ks,  length: MemoryLayout<UInt32>.size, index: 10)
        p1.setBytes(&cl,  length: MemoryLayout<UInt32>.size, index: 11)
        p1.setBytes(&nc,  length: MemoryLayout<UInt32>.size, index: 12)
        p1.setBytes(&sc,  length: MemoryLayout<Float>.size,  index: 13)
        p1.dispatchDecode(MTLSize(width: partialGroups, height: 1, depth: 1),
                          threads: MTLSize(width: tgWidth, height: 1, depth: 1),
                          conditional: conditional)
        p1.endEncoding()

        let combineEncoder: MTLComputeCommandEncoder?
        if let stageTiming {
            combineEncoder = stageTiming.makeEncoder(commandBuffer: commandBuffer, pass: .combine)
        } else {
            combineEncoder = commandBuffer.makeComputeCommandEncoder()
        }
        guard let p2 = combineEncoder else { return }
        let combinePSO = combinePipeline(headDim: headDim,
                                         numQHeads: numQHeads,
                                         numKVHeads: numKVHeads,
                                         numChunks: nChunks)
        p2.setComputePipelineState(combinePSO)
        p2.setBuffer(mPartial, offset: 0, index: 0)
        p2.setBuffer(dPartial, offset: 0, index: 1)
        p2.setBuffer(oPartial, offset: 0, index: 2)
        p2.setBuffer(out, offset: outOffset, index: 3)
        var hd2 = headDim, nc2 = UInt32(nChunks)
        p2.setBytes(&hd2, length: MemoryLayout<UInt32>.size, index: 4)
        p2.setBytes(&nc2, length: MemoryLayout<UInt32>.size, index: 5)
        let combineTGWidth = min(Self.threadsPerGroup,
                                 Int(combinePSO.maxTotalThreadsPerThreadgroup))
        p2.dispatchDecode(MTLSize(width: Int(numQHeads), height: 1, depth: 1),
                          threads: MTLSize(width: combineTGWidth, height: 1, depth: 1),
                          conditional: conditional)
        p2.endEncoding()
    }

    /// Runs two consecutive query tokens with the same sum order as decode.
    func encodePair(commandBuffer: MTLCommandBuffer,
                    queries: [MTLBuffer], keys: MTLBuffer, values: MTLBuffer,
                    outputs: [MTLBuffer], full: Bool, firstLength: UInt32,
                    window: UInt32, ringCapacity: UInt32,
                    conditional: DecodeDispatch? = nil) throws -> Bool {
        guard queries.count == 2, outputs.count == 2 else { return false }
        let hd: UInt32 = full ? 512 : 256
        let nk: UInt32 = full ? 2 : 8
        let lengths = [firstLength, firstLength + 1]
        let starts = lengths.map { !full && $0 > window ? $0 - window : 0 }
        let shapes = zip(lengths, starts).map {
            Self.splitGeometry(numQHeads: 16, numKVHeads: nk,
                seqLen: $0.0, kvStart: $0.1, preferGQASWA: !full)
        }
        guard shapes.allSatisfy({ $0.numChunks == 16 }) else { return false }
        let reuseTokens = useTokenPairReuse && useQueryPair
            && starts[0] == starts[1] && shapes[0].chunkLength == shapes[1].chunkLength
        let partial = try Self.specializedPipeline(ctx,
            reuseTokens ? "attention_decode_token_pair_partial" : useQueryPair ? "attention_decode_query_pair_partial"
                : (full ? "attention_decode_partial" : "attention_decode_gqa_swa_partial"),
            headDim: hd, numQHeads: 16, numKVHeads: nk, numChunks: 16,
            ringCapacity: ringCapacity > 0 ? ringCapacity : nil,
            retainQuery: full || retainSWAQuery,
            prefetchValue: full || retainSWAQuery, paired: true)
        let combine = try Self.specializedPipeline(ctx, "attention_decode_combine",
            headDim: hd, numQHeads: 16, numKVHeads: nk, numChunks: 16, paired: true)
        let partialWidth = useQueryPair ? Int(hd) / 2 : Self.threadsPerGroup
        guard partial.maxTotalThreadsPerThreadgroup >= partialWidth,
              combine.maxTotalThreadsPerThreadgroup >= Self.threadsPerGroup else { return false }
        guard let first = commandBuffer.makeComputeCommandEncoder() else { throw MetalError.noQueue }
        first.setComputePipelineState(partial)
        for (index, buffer) in [queries[0], keys, values, mPartial, dPartial, oPartial].enumerated() {
            first.setBuffer(buffer, offset: 0, index: index)
        }
        first.setBuffer(queries[1], offset: 0, index: 14)
        for (index, value) in [hd, 16, nk].enumerated() {
            var value = value
            first.setBytes(&value, length: 4, index: 6 + index)
        }
        let chunks = shapes.map { UInt32($0.chunkLength) }
        for (index, values) in [lengths, starts, chunks].enumerated() {
            values.withUnsafeBytes { first.setBytes($0.baseAddress!, length: $0.count, index: 9 + index) }
        }
        var count: UInt32 = 16
        var scale: Float = 1
        first.setBytes(&count, length: 4, index: 12)
        first.setBytes(&scale, length: 4, index: 13)
        let threads = MTLSize(width: Self.threadsPerGroup, height: 1, depth: 1)
        first.dispatchDecode(MTLSize(width: useQueryPair ? 128 : shapes[0].partialThreadgroups,
            height: reuseTokens ? 1 : 2, depth: 1), threads: MTLSize(width: partialWidth, height: 1, depth: 1),
            conditional: conditional)
        first.endEncoding()
        guard let second = commandBuffer.makeComputeCommandEncoder() else { throw MetalError.noQueue }
        second.setComputePipelineState(combine)
        for (index, buffer) in [mPartial, dPartial, oPartial, outputs[0]].enumerated() {
            second.setBuffer(buffer, offset: 0, index: index)
        }
        second.setBuffer(outputs[1], offset: 0, index: 6)
        var dimension = hd
        second.setBytes(&dimension, length: 4, index: 4)
        second.setBytes(&count, length: 4, index: 5)
        second.dispatchDecode(MTLSize(width: 16, height: 2, depth: 1),
            threads: threads, conditional: conditional)
        second.endEncoding()
        return true
    }

    /// `1 / sqrt(head_dim)` — the classic transformer scaling. Used as the
    /// default for non-Gemma callers (and for the existing tests that pre-date
    /// the runtime scale arg).
    static func defaultScale(headDim: UInt32) -> Float {
        Float(1.0) / Float(headDim).squareRoot()
    }

    private static func specializedPipeline(_ context: MetalContext,
                                            _ name: String,
                                            headDim: UInt32,
                                            numQHeads: UInt32,
                                            numKVHeads: UInt32,
                                            numChunks: UInt32? = nil,
                                            ringCapacity: UInt32? = nil,
                                            retainQuery: Bool = false,
                                            prefetchValue: Bool = false,
                                            paired: Bool = false) throws -> MTLComputePipelineState {
        var constants = [
            MetalFunctionConstant(index: 60, value: .uint32(headDim)),
            MetalFunctionConstant(index: 61, value: .uint32(numQHeads)),
            MetalFunctionConstant(index: 62, value: .uint32(numKVHeads)),
            MetalFunctionConstant(index: 63, value: .bool(true)),
        ]
        if let numChunks {
            constants.append(MetalFunctionConstant(index: 65, value: .uint32(numChunks)))
        }
        if retainQuery {
            constants.append(MetalFunctionConstant(index: 66, value: .bool(true)))
        }
        if prefetchValue {
            constants.append(MetalFunctionConstant(index: 67, value: .bool(true)))
        }
        if paired {
            constants.append(MetalFunctionConstant(index: 68, value: .bool(true)))
        }
        if let ringCapacity {
            constants.append(MetalFunctionConstant(index: 69, value: .uint32(ringCapacity)))
        }
        return try context.pipeline(name, constants: constants)
    }

    private func partialPipeline(headDim: UInt32,
                                 numQHeads: UInt32,
                                 numKVHeads: UInt32,
                                 numChunks: Int,
                                 useGQAPartial: Bool,
                                 ringCapacity: UInt32 = 0) -> MTLComputePipelineState {
        if ringCapacity > 0 {
            let name = useGQAPartial ? "attention_decode_gqa_swa_partial" : "attention_decode_partial"
            let specializedChunks = numChunks == 16 ? Optional(UInt32(numChunks)) : nil
            do {
                return try Self.specializedPipeline(ctx,
                                                    name,
                                                    headDim: headDim,
                                                    numQHeads: numQHeads,
                                                    numKVHeads: numKVHeads,
                                                    numChunks: specializedChunks,
                                                    ringCapacity: ringCapacity)
            } catch {
                preconditionFailure("failed to build FP16 KV ring attention pipeline: \(error)")
            }
        }
        if useGQAPartial && headDim == 256 && numQHeads == 16 && numKVHeads == 8 {
            if numChunks == 16 {
                return psoGQAPartialSWAChunks16
            }
            return psoGQAPartialSWA
        }
        if !useGQAPartial && headDim == 256 && numQHeads == 16 && numKVHeads == 8 {
            return psoPartialSWA
        }
        if !useGQAPartial && headDim == 512 && numQHeads == 16 && numKVHeads == 2 {
            if numChunks == 16 {
                return psoPartialFullChunks16
            }
            return psoPartialFull
        }
        return useGQAPartial ? psoGQAPartial : psoPartial
    }

    private func combinePipeline(headDim: UInt32,
                                 numQHeads: UInt32,
                                 numKVHeads: UInt32,
                                 numChunks: Int) -> MTLComputePipelineState {
        if headDim == 256 && numQHeads == 16 && numKVHeads == 8 {
            return numChunks == 16 ? psoCombineSWAChunks16 : psoCombineSWA
        }
        if headDim == 512 && numQHeads == 16 && numKVHeads == 2 {
            return numChunks == 16 ? psoCombineFullChunks16 : psoCombineFull
        }
        return psoCombine
    }
}
