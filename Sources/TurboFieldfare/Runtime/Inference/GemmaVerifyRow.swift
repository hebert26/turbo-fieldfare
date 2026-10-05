import Metal

/// Scratch for one target token in a small verification batch.
final class GemmaVerifyRow {
    let hidden, normed, qScratch, attnOut, oOut: MTLBuffer
    let denseX, routedX, routerInput, h1Buf, h2Buf: MTLBuffer
    let denseScratchGate, denseScratchUp, denseScratchAct, moeActs: MTLBuffer
    let outIndices, outWeights, logits: MTLBuffer

    init(device: MTLDevice, config: ArchConfig) throws {
        func buffer(_ count: Int, stride: Int = 2) throws -> MTLBuffer {
            guard let result = device.makeBuffer(length: count * stride, options: .storageModeShared) else {
                throw MetalError.noDevice
            }
            return result
        }
        let d = config.hiddenSize
        let q = config.numHeads * max(config.headDim, config.fullHeadDim)
        hidden = try buffer(d)
        normed = try buffer(d)
        qScratch = try buffer(q)
        attnOut = try buffer(q)
        oOut = try buffer(d)
        denseX = try buffer(d)
        routedX = try buffer(d)
        routerInput = try buffer(d)
        h1Buf = try buffer(d)
        h2Buf = try buffer(d)
        denseScratchGate = try buffer(config.intermediateSize)
        denseScratchUp = try buffer(config.intermediateSize)
        denseScratchAct = try buffer(config.intermediateSize)
        moeActs = try buffer(config.topKExperts * config.moeIntermediateSize)
        outIndices = try buffer(config.topKExperts, stride: 4)
        outWeights = try buffer(config.topKExperts)
        logits = try buffer(config.vocabSize)
    }
}

struct GemmaVerificationGPUTime {
    var cached = 0.0
    var prefix = 0.0
    var dense = 0.0
    var routed = 0.0
    var head = 0.0
}
