import Metal

/// The GPU can set these group counts to zero after a cache miss.
final class DecodeDispatch {
    static let capacity = 32
    static let argumentStride = 3 * MemoryLayout<UInt32>.stride
    let resolved: MTLBuffer
    let byteOffset: Int
    private(set) var count = 0

    init(buffer: MTLBuffer, byteOffset: Int) {
        precondition(byteOffset >= 0 && byteOffset.isMultiple(of: Self.argumentStride))
        precondition(byteOffset + Self.capacity * Self.argumentStride <= buffer.length)
        precondition(buffer.storageMode == .shared)
        let resolved = buffer
        self.resolved = resolved
        self.byteOffset = byteOffset
    }

    func reset() {
        count = 0
        resolved.contents().advanced(by: byteOffset)
            .initializeMemory(as: UInt8.self, repeating: 0,
                              count: Self.capacity * Self.argumentStride)
    }

    func encode(_ encoder: MTLComputeCommandEncoder,
                groups: MTLSize, threads: MTLSize) {
        precondition(count < Self.capacity)
        let offset = byteOffset + count * Self.argumentStride
        let values = resolved.contents().advanced(by: offset)
            .assumingMemoryBound(to: UInt32.self)
        values[0] = UInt32(groups.width)
        values[1] = UInt32(groups.height)
        values[2] = UInt32(groups.depth)
        encoder.dispatchThreadgroups(indirectBuffer: resolved,
                                     indirectBufferOffset: offset,
                                     threadsPerThreadgroup: threads)
        count += 1
    }
}

extension MTLComputeCommandEncoder {
    func dispatchDecode(_ groups: MTLSize, threads: MTLSize,
                        conditional: DecodeDispatch?) {
        if let conditional {
            conditional.encode(self, groups: groups, threads: threads)
        } else {
            dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        }
    }
}
