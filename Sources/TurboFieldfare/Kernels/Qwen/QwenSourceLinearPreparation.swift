import Foundation
import Metal

/// Opt-in source preparation encoder. The caller owns command submission,
/// completion, speculative state and publication; this helper only encodes.
final class QwenSourceLinearPreparation {
    enum Failure: Error {
        case invalidBinding(String)
    }
    private struct Parameters {
        var tokenCount: UInt32
        var keyHeads: UInt32
        var valueHeads: UInt32
        var keyDimension: UInt32
        var valueDimension: UInt32
        var channels: UInt32
    }
    private let configuration: QwenGatedDeltaNetConfiguration
    private let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    let mathProbePipeline: MTLComputePipelineState

    init(context: MetalContext, configuration: QwenGatedDeltaNetConfiguration) throws {
        self.configuration = configuration
        device = context.device
        // The Metal directory is already copied as a package resource. Keeping
        // this separate avoids changes to the default shader module registry.
        var source = ""
        for name in ["qwen_source_math", "qwen_source_scalar_math", "qwen_linear_preparation"] {
            guard let url = Bundle.module.url(forResource: name, withExtension: "metal",
                                              subdirectory: "Metal/Qwen") else {
                throw MetalError.missingShaderResource(name)
            }
            source += try String(contentsOf: url, encoding: .utf8) + "\n"
        }
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        let library = try device.makeLibrary(source: source, options: options)
        guard let function = library.makeFunction(name: "qwen_source_linear_prepare_fp32") else {
            throw MetalError.missingFunction("qwen_source_linear_prepare_fp32")
        }
        pipeline = try device.makeComputePipelineState(function: function)
        guard let probe = library.makeFunction(name: "qwen_source_linear_preparation_math_probe") else {
            throw MetalError.missingFunction("qwen_source_linear_preparation_math_probe")
        }
        mathProbePipeline = try device.makeComputePipelineState(function: probe)
    }

    /// Caller owns exclusive buffers through completion. Status must start at
    /// zero and be read only after successful completion: bit1=input invalid,
    /// bit2=prepared output invalid. Flags forbid publication. Downstream GPU
    /// work may consume outputs only into speculative scratch/reserved state;
    /// the caller rejects flags after settlement and before result readback.
    func encode(commandBuffer: MTLCommandBuffer, tokenCount: Int,
                convolved: MTLBuffer, rawBeta: MTLBuffer, rawA: MTLBuffer,
                aLog: MTLBuffer, timeStepBias: MTLBuffer,
                query: MTLBuffer, key: MTLBuffer, value: MTLBuffer,
                beta: MTLBuffer, logDecay: MTLBuffer, status: MTLBuffer) throws {
        guard commandBuffer.status == .notEnqueued,
              commandBuffer.commandQueue.device === device,
              (1...256).contains(tokenCount),
              pipeline.threadExecutionWidth > 0,
              pipeline.maxTotalThreadsPerThreadgroup > 0 else {
            throw Failure.invalidBinding("command, count or pipeline")
        }
        func product(_ a: Int, _ b: Int) throws -> Int {
            let (result, overflow) = a.multipliedReportingOverflow(by: b)
            guard !overflow, result > 0, UInt32(exactly: result) != nil else {
                throw Failure.invalidBinding("element count overflow")
            }
            return result
        }
        let channels = configuration.convolutionChannelCount
        let qkvCount = try product(tokenCount, channels)
        let queryCount = try product(tokenCount,
            product(configuration.valueHeadCount, configuration.keyHeadDimension))
        let valueCount = try product(tokenCount, configuration.valueDimension)
        let scalarCount = try product(tokenCount, configuration.valueHeadCount)
        let buffers = [convolved, rawBeta, rawA, aLog, timeStepBias,
                       query, key, value, beta, logDecay, status]
        let counts = [qkvCount, scalarCount, scalarCount, configuration.valueHeadCount,
                      configuration.valueHeadCount, queryCount, queryCount,
                      valueCount, scalarCount, scalarCount, 1]
        guard Set(buffers.map { ObjectIdentifier($0) }).count == buffers.count,
              buffers.allSatisfy({ $0.device === device }), status.storageMode == .shared else {
            throw Failure.invalidBinding("device, alias or status storage")
        }
        for (buffer, count) in zip(buffers, counts) {
            let bytes = try product(count, MemoryLayout<Float>.stride)
            guard bytes <= device.maxBufferLength, buffer.length >= bytes else {
                throw Failure.invalidBinding("buffer length")
            }
        }
        guard status.contents().load(as: UInt32.self) == 0 else {
            throw Failure.invalidBinding("status was not reset after the preceding settled command")
        }
        var parameters = Parameters(tokenCount: UInt32(tokenCount),
            keyHeads: UInt32(configuration.keyHeadCount), valueHeads: UInt32(configuration.valueHeadCount),
            keyDimension: UInt32(configuration.keyHeadDimension), valueDimension: UInt32(configuration.valueHeadDimension),
            channels: UInt32(channels))
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw Failure.invalidBinding("command encoder unavailable")
        }
        defer { encoder.endEncoding() }
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 0)
        for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index + 1) }
        let count = max(qkvCount, max(queryCount, max(valueCount, scalarCount)))
        let width = min(count, min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup))
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
}
