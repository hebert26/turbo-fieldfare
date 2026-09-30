import Foundation
import Metal
import TurboFieldfareOfficialQwenSource

/// One exact named BF16 tensor from a protected registered source header.
/// Shape is checked against the admitted header, never supplied as an offset.
struct QwenBF16TensorSpec: Sendable {
    enum Role: Sendable, Equatable {
        case embedding, head, router, dense
        case sharedGate, sharedUp, sharedDown, sharedOutputGate
    }

    let name: String
    let shardName: String
    let role: Role
    let rows: Int
    let columns: Int

    init(name: String, shardName: String, role: Role, rows: Int, columns: Int) {
        self.name = name
        self.shardName = shardName
        self.role = role
        self.rows = rows
        self.columns = columns
    }
}

enum QwenBF16WeightError: Error, Equatable {
    case invalidGeometry(String)
    case budgetExceeded
    case allocationFailed(String)
    case missingTensor(String)
    case invalidBuffer(String)
    case commandBufferUnavailable
}

/// A bounded, additive BF16 GPU weight set. It does not admit a source to
/// inference, authenticate weights, or replace the existing packed mapping.
/// Published chunks remain immutable and shared; each encoder retains their
/// buffers through command completion. Failed initialization publishes none.
final class QwenBF16Weights: @unchecked Sendable {
    enum StageCheckpoint { case beforeAllocation, afterChunkRead, beforePublish }

    /// Internal, read-only inspection of the actual resident buffers, not a
    /// way to construct weights or supply arbitrary GPU bindings.
    struct ChunkInspection {
        let name: String
        let firstRow: Int
        let rowCount: Int
        let buffer: MTLBuffer
    }

    var inspectedChunks: [ChunkInspection] {
        var result: [ChunkInspection] = []
        for tensor in tensors.values {
            let name = tensor.spec.name
            for chunk in tensor.chunks {
                result.append(ChunkInspection(name: name, firstRow: chunk.firstRow,
                                              rowCount: chunk.rowCount, buffer: chunk.buffer))
            }
        }
        result.sort { left, right in
            if left.name == right.name { return left.firstRow < right.firstRow }
            return left.name < right.name
        }
        return result
    }

    private struct Parameters {
        var rows: UInt32
        var columns: UInt32
        var firstRow: UInt32
        var rowsInChunk: UInt32
        var tokenCount: UInt32
    }

    private struct Chunk {
        let firstRow: Int
        let rowCount: Int
        let buffer: MTLBuffer
    }

    private struct Tensor {
        let spec: QwenBF16TensorSpec
        let chunks: [Chunk]
    }

    /// Metal calls the completion handler after it has finished using these
    /// resources. Its callback may run on a different thread.
    private final class BufferLease: @unchecked Sendable {
        let buffers: [MTLBuffer]
        init(_ buffers: [MTLBuffer]) { self.buffers = buffers }
    }

    private let device: MTLDevice
    private let embeddingPipeline: MTLComputePipelineState
    private let projectionPipeline: MTLComputePipelineState
    private let tensors: [String: Tensor]

    convenience init(context: MetalContext, source: OfficialSourceHandle,
                     specifications: [QwenBF16TensorSpec],
                     residencyBudget: UInt64) throws {
        try self.init(context: context, source: source, specifications: specifications,
                      residencyBudget: residencyBudget,
                      maximumChunkBytes: 8 * 1024 * 1024, checkpoint: { _ in })
    }

    /// Internal, shrinking cap and deterministic interruption seam. It cannot
    /// replace descriptor admission, source reads, or the production Metal device.
    convenience init(context: MetalContext, source: OfficialSourceHandle,
                     specifications: [QwenBF16TensorSpec], residencyBudget: UInt64,
                     maximumChunkBytes: UInt64,
                     checkpoint: (StageCheckpoint) throws -> Void) throws {
        try self.init(context: context, source: source, specifications: specifications,
                      residencyBudget: residencyBudget, maximumChunkBytes: maximumChunkBytes,
                      allocate: { device, length in
                          device.makeBuffer(length: length, options: .storageModeShared)
                      }, checkpoint: checkpoint)
    }

    /// Internal allocation fault seam: nil must fail before publication. Even
    /// injected buffers must come from this device, be shared and fit the tile.
    init(context: MetalContext, source: OfficialSourceHandle,
         specifications: [QwenBF16TensorSpec], residencyBudget: UInt64,
         maximumChunkBytes: UInt64,
         allocate: (MTLDevice, Int) -> MTLBuffer?,
         checkpoint: (StageCheckpoint) throws -> Void) throws {
        let device = context.device
        // Isolate IEEE nonfinite handling from the packed shared library's
        // default math settings; no global shader compile option changes.
        let library = try MetalContext.privateLibrary(device: device, module: "qwen_bf16",
                                                      mathMode: .safe)
        guard let embeddingFunction = library.makeFunction(name: "qwen_bf16_embedding"),
              let projectionFunction = library.makeFunction(name: "qwen_bf16_project_fp32") else {
            throw MetalError.missingFunction("qwen_bf16")
        }
        let embeddingPipeline = try device.makeComputePipelineState(function: embeddingFunction)
        let projectionPipeline = try device.makeComputePipelineState(function: projectionFunction)
        guard !specifications.isEmpty, specifications.count <= 128,
              maximumChunkBytes > 0, device.maxBufferLength > 0 else {
            throw QwenBF16WeightError.invalidGeometry("empty/oversized tensor set or chunk cap")
        }
        struct Plan {
            let spec: QwenBF16TensorSpec
            let token: OfficialSourceHandle.TensorRange
            let rowBytes: UInt64
            let chunkBytes: UInt64
        }
        var plans: [Plan] = []
        plans.reserveCapacity(specifications.count)
        var names = Set<String>()
        var totalBytes: UInt64 = 0
        // Validate ALL header geometry, budget, and chunk capacity before the
        // first GPU allocation or tensor payload read.
        for spec in specifications {
            try Task.checkCancellation()
            guard !spec.name.isEmpty, names.insert(spec.name).inserted,
                  spec.rows > 0, spec.columns > 0,
                  spec.rows <= Int(UInt32.max), spec.columns <= Int(UInt32.max) else {
                throw QwenBF16WeightError.invalidGeometry("invalid or duplicate tensor: \(spec.name)")
            }
            let token = try source.admitTensor(shardName: spec.shardName,
                                               tensorName: spec.name)
            guard token.shape == [UInt64(spec.rows), UInt64(spec.columns)] else {
                throw QwenBF16WeightError.invalidGeometry("header shape disagrees: \(spec.name)")
            }
            let (rowBytes, rowOverflow) = UInt64(spec.columns).multipliedReportingOverflow(by: 2)
            let (tensorBytes, tensorOverflow) = rowBytes.multipliedReportingOverflow(by: UInt64(spec.rows))
            let (next, totalOverflow) = totalBytes.addingReportingOverflow(tensorBytes)
            let chunkLimit = min(UInt64(device.maxBufferLength), maximumChunkBytes)
            guard !rowOverflow, !tensorOverflow, !totalOverflow,
                  rowBytes > 0, rowBytes <= chunkLimit,
                  tensorBytes <= UInt64(Int.max) else {
                throw QwenBF16WeightError.invalidGeometry("geometry or chunk cap: \(spec.name)")
            }
            guard next <= residencyBudget else { throw QwenBF16WeightError.budgetExceeded }
            totalBytes = next
            plans.append(Plan(spec: spec, token: token, rowBytes: rowBytes,
                              chunkBytes: (chunkLimit / rowBytes) * rowBytes))
        }
        var staged: [String: Tensor] = [:]
        staged.reserveCapacity(plans.count)
        var allocatedBufferIDs = Set<ObjectIdentifier>()
        for plan in plans {
            var chunks: [Chunk] = []
            var firstRow = 0
            while firstRow < plan.spec.rows {
                try Task.checkCancellation()
                try checkpoint(.beforeAllocation)
                let rowCount = min(plan.spec.rows - firstRow,
                                   Int(plan.chunkBytes / plan.rowBytes))
                let byteCount = Int(UInt64(rowCount) * plan.rowBytes)
                // Own the ObjC allocation/read autoreleases per chunk. A
                // throwing load drains this pool before returning the error;
                // a successful chunk is retained only by local staging until
                // the entire set passes final checks and publishes atomically.
                let chunk: Chunk = try autoreleasepool {
                    guard rowCount > 0, byteCount > 0,
                          byteCount <= device.maxBufferLength,
                          let buffer = allocate(device, byteCount) else {
                        throw QwenBF16WeightError.allocationFailed(plan.spec.name)
                    }
                    guard buffer.device === device, buffer.storageMode == .shared,
                          buffer.length >= byteCount,
                          allocatedBufferIDs.insert(ObjectIdentifier(buffer)).inserted else {
                        throw QwenBF16WeightError.invalidBuffer("invalid allocated chunk: \(plan.spec.name)")
                    }
                    buffer.label = "qwen.bf16.\(plan.spec.name).\(firstRow)"
                    // Every read points directly into the final shared Metal
                    // buffer. Subtiles may cross rows; chunks themselves never do.
                    var copied = 0
                    let tensorOffset = UInt64(firstRow) * plan.rowBytes
                    while copied < byteCount {
                        try Task.checkCancellation()
                        let length = min(byteCount - copied,
                                         Int(OfficialSourceHandle.maximumTensorReadBytes))
                        let destination = UnsafeMutableRawBufferPointer(
                            start: buffer.contents().advanced(by: copied), count: length)
                        try source.preadTensorRange(
                            plan.token, byteOffset: tensorOffset + UInt64(copied),
                            byteCount: UInt64(length), expectedByteCount: UInt64(length),
                            into: destination)
                        copied += length
                    }
                    try checkpoint(.afterChunkRead)
                    return Chunk(firstRow: firstRow, rowCount: rowCount, buffer: buffer)
                }
                chunks.append(chunk)
                firstRow += rowCount
            }
            staged[plan.spec.name] = Tensor(spec: plan.spec, chunks: chunks)
        }
        try Task.checkCancellation()
        try checkpoint(.beforePublish)
        // Zero-length reads perform the final retained FD/root/marker check
        // without reading any tensor payload before atomic publication.
        for plan in plans {
            try source.preadTensorRange(plan.token, byteOffset: 0, byteCount: 0,
                                        expectedByteCount: 0,
                                        into: UnsafeMutableRawBufferPointer(start: nil, count: 0))
        }
        try Task.checkCancellation()
        self.device = device
        self.embeddingPipeline = embeddingPipeline
        self.projectionPipeline = projectionPipeline
        self.tensors = staged
    }

    func requireShape(_ name: String, role: QwenBF16TensorSpec.Role,
                      rows: Int, columns: Int) throws {
        guard let spec = tensors[name]?.spec,
              spec.role == role, spec.rows == rows, spec.columns == columns else {
            throw QwenBF16WeightError.invalidGeometry("missing or wrong BF16 role/shape: \(name)")
        }
    }

    func encodeEmbedding(commandBuffer: MTLCommandBuffer, tensorName: String,
                         tokenIDs: MTLBuffer, tokenCount: Int,
                         output: MTLBuffer) throws {
        let tensor = try requireTensor(tensorName, role: .embedding)
        try requireReady(commandBuffer)
        try requireTokenCount(tokenCount)
        try requireElements(tokenIDs, count: tokenCount, stride: MemoryLayout<UInt32>.stride,
                            label: "tokenIDs")
        try requireElements(output, count: try product(tokenCount, tensor.spec.columns),
                            stride: MemoryLayout<Float>.stride, label: "embedding output")
        try requireSeparateBuffers(input: tokenIDs, output: output, tensor: tensor)
        // Validate token IDs before encoding. A shared ID buffer is required;
        // a private buffer cannot be inspected without a GPU status/readback.
        guard tokenIDs.storageMode == .shared else {
            throw QwenBF16WeightError.invalidBuffer("tokenIDs must be shared")
        }
        let ids = tokenIDs.contents().bindMemory(to: UInt32.self, capacity: tokenCount)
        for index in 0..<tokenCount where ids[index] >= UInt32(tensor.spec.rows) {
            throw QwenBF16WeightError.invalidGeometry("embedding ID out of bounds")
        }
        try encode(commandBuffer: commandBuffer, tensor: tensor, pipeline: embeddingPipeline,
                   input: tokenIDs, tokenCount: tokenCount, output: output)
    }

    func encodeProjection(commandBuffer: MTLCommandBuffer, tensorName: String,
                          input: MTLBuffer, tokenCount: Int,
                          output: MTLBuffer) throws {
        guard let tensor = tensors[tensorName], tensor.spec.role != .embedding else {
            throw QwenBF16WeightError.missingTensor(tensorName)
        }
        try requireReady(commandBuffer)
        try requireTokenCount(tokenCount)
        try requireElements(input, count: try product(tokenCount, tensor.spec.columns),
                            stride: MemoryLayout<Float>.stride, label: "projection input")
        try requireElements(output, count: try product(tokenCount, tensor.spec.rows),
                            stride: MemoryLayout<Float>.stride, label: "projection output")
        try requireSeparateBuffers(input: input, output: output, tensor: tensor)
        try encode(commandBuffer: commandBuffer, tensor: tensor, pipeline: projectionPipeline,
                   input: input, tokenCount: tokenCount, output: output)
    }

    private func requireTensor(_ name: String, role: QwenBF16TensorSpec.Role) throws -> Tensor {
        guard let tensor = tensors[name], tensor.spec.role == role else {
            throw QwenBF16WeightError.missingTensor(name)
        }
        return tensor
    }

    private func requireReady(_ commandBuffer: MTLCommandBuffer) throws {
        guard commandBuffer.status == .notEnqueued,
              commandBuffer.device === device else {
            throw QwenBF16WeightError.commandBufferUnavailable
        }
    }

    private func requireTokenCount(_ count: Int) throws {
        guard count > 0, count <= Int(UInt32.max) else {
            throw QwenBF16WeightError.invalidGeometry("token count outside GPU bounds")
        }
    }

    private func product(_ a: Int, _ b: Int) throws -> Int {
        guard a > 0, b > 0 else { throw QwenBF16WeightError.invalidGeometry("empty dispatch") }
        let result = a.multipliedReportingOverflow(by: b)
        guard !result.overflow else { throw QwenBF16WeightError.invalidGeometry("dispatch overflow") }
        return result.partialValue
    }

    private func requireElements(_ buffer: MTLBuffer, count: Int,
                                 stride: Int, label: String) throws {
        let needed = count.multipliedReportingOverflow(by: stride)
        guard !needed.overflow, needed.partialValue <= buffer.length,
              buffer.device === device else {
            throw QwenBF16WeightError.invalidBuffer(label)
        }
    }

    private func requireSeparateBuffers(input: MTLBuffer, output: MTLBuffer,
                                        tensor: Tensor) throws {
        guard input !== output,
              !tensor.chunks.contains(where: { $0.buffer === input || $0.buffer === output }) else {
            throw QwenBF16WeightError.invalidBuffer("input, output and weights must not alias")
        }
    }

    private func encode(commandBuffer: MTLCommandBuffer, tensor: Tensor,
                        pipeline: MTLComputePipelineState,
                        input: MTLBuffer, tokenCount: Int,
                        output: MTLBuffer) throws {
        guard tokenCount > 0, tokenCount <= Int(UInt32.max),
              pipeline.maxTotalThreadsPerThreadgroup > 0,
              pipeline.threadExecutionWidth > 0 else {
            throw QwenBF16WeightError.invalidGeometry("GPU dispatch limit")
        }
        // Retain even if a later encoder cannot be created: a caller must
        // discard an errored command buffer rather than submit partial work.
        let lease = BufferLease(tensor.chunks.map(\.buffer))
        commandBuffer.addCompletedHandler { _ in _ = lease.buffers.count }
        // Callers own submission and keep input/output immutable until completion.
        // No waitUntilCompleted or early buffer reuse.
        for chunk in tensor.chunks {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw QwenBF16WeightError.commandBufferUnavailable
            }
            var params = Parameters(rows: UInt32(tensor.spec.rows),
                                    columns: UInt32(tensor.spec.columns),
                                    firstRow: UInt32(chunk.firstRow),
                                    rowsInChunk: UInt32(chunk.rowCount),
                                    tokenCount: UInt32(tokenCount))
            encoder.setComputePipelineState(pipeline)
            encoder.setBytes(&params, length: MemoryLayout<Parameters>.stride,
                             index: QwenMetalBufferIndex.parameters.rawValue)
            encoder.setBuffer(input, offset: 0, index: QwenMetalBufferIndex.input.rawValue)
            encoder.setBuffer(chunk.buffer, offset: 0,
                              index: QwenMetalBufferIndex.weights.rawValue)
            encoder.setBuffer(output, offset: 0, index: QwenMetalBufferIndex.output.rawValue)
            encoder.useResource(chunk.buffer, usage: .read)
            let width = tensor.spec.role == .embedding ? tensor.spec.columns : chunk.rowCount
            let groupWidth = min(width, pipeline.threadExecutionWidth,
                                 pipeline.maxTotalThreadsPerThreadgroup)
            encoder.dispatchThreads(MTLSize(width: width, height: tokenCount, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: groupWidth, height: 1,
                                                                   depth: 1))
            encoder.endEncoding()
        }
    }
}
