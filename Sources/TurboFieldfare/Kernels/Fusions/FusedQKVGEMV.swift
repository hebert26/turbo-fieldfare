import Foundation
import Metal

final class FusedQKVGEMV {
    private struct Shape: Hashable {
        var qRows: UInt32
        var kvRows: UInt32
        var n: UInt32
        var kEqualsV: Bool
    }

    private let pso: MTLComputePipelineState
    private let kEqualsVPSO: MTLComputePipelineState
    private let specializedPSOs: [Shape: MTLComputePipelineState]
    private static let threadsPerThreadgroup = 64

    private static let realDecodeShapes: [Shape] = [
        Shape(qRows: 4096, kvRows: 2048, n: 2816, kEqualsV: false),
        Shape(qRows: 8192, kvRows: 1024, n: 2816, kEqualsV: true),
    ]

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("dequant_int4_qkv_gemv_simd",
                                        constants: [],
                                        maxTotalThreadsPerThreadgroup: Self.threadsPerThreadgroup)
        self.kEqualsVPSO = try context.pipeline("dequant_int4_qkv_gemv_simd",
            constants: [MetalFunctionConstant(index: 27, value: .bool(true))],
            maxTotalThreadsPerThreadgroup: Self.threadsPerThreadgroup)
        var variants: [Shape: MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            variants[shape] = try context.pipeline(
                "dequant_int4_qkv_gemv_simd",
                constants: [
                    MetalFunctionConstant(index: 23, value: .uint32(shape.qRows)),
                    MetalFunctionConstant(index: 24, value: .uint32(shape.kvRows)),
                    MetalFunctionConstant(index: 25, value: .uint32(shape.n)),
                    MetalFunctionConstant(index: 26, value: .bool(true)),
                    MetalFunctionConstant(index: 27, value: .bool(shape.kEqualsV)),
                ],
                maxTotalThreadsPerThreadgroup: Self.threadsPerThreadgroup)
        }
        self.specializedPSOs = variants
    }

    func encode(commandBuffer: MTLCommandBuffer,
                       qWeights: MTLBuffer, qWeightsOffset: Int = 0,
                       qScales: MTLBuffer, qScalesOffset: Int = 0,
                       qBiases: MTLBuffer, qBiasesOffset: Int = 0,
                       kWeights: MTLBuffer, kWeightsOffset: Int = 0,
                       kScales: MTLBuffer, kScalesOffset: Int = 0,
                       kBiases: MTLBuffer, kBiasesOffset: Int = 0,
                       vWeights: MTLBuffer, vWeightsOffset: Int = 0,
                       vScales: MTLBuffer, vScalesOffset: Int = 0,
                       vBiases: MTLBuffer, vBiasesOffset: Int = 0,
                       x: MTLBuffer,
                       qOut: MTLBuffer, qOutOffset: Int = 0,
                       kOut: MTLBuffer, kOutOffset: Int = 0,
                       vOut: MTLBuffer, vOutOffset: Int = 0,
                       qRows: UInt32,
                       kvRows: UInt32,
                       n: UInt32,
                       kEqualsV: Bool = false,
                       conditional: DecodeDispatch? = nil) {
        precondition(n % UInt32(Quantization.groupSize) == 0,
                     "N must be a multiple of \(Quantization.groupSize)")
        precondition(qWeightsOffset % 2 == 0 &&
                     kWeightsOffset % 2 == 0 &&
                     vWeightsOffset % 2 == 0,
                     "FusedQKVGEMV needs 2-aligned weights offsets")
        precondition(!kEqualsV || (kWeights === vWeights && kScales === vScales &&
            kBiases === vBiases && kWeightsOffset == vWeightsOffset &&
            kScalesOffset == vScalesOffset && kBiasesOffset == vBiasesOffset),
            "K and V must use the same projection")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        let shape = Shape(qRows: qRows, kvRows: kvRows, n: n, kEqualsV: kEqualsV)
        enc.setComputePipelineState(specializedPSOs[shape] ?? (kEqualsV ? kEqualsVPSO : pso))
        enc.setBuffer(qWeights, offset: qWeightsOffset, index: 0)
        enc.setBuffer(qScales, offset: qScalesOffset, index: 1)
        enc.setBuffer(qBiases, offset: qBiasesOffset, index: 2)
        enc.setBuffer(kWeights, offset: kWeightsOffset, index: 3)
        enc.setBuffer(kScales, offset: kScalesOffset, index: 4)
        enc.setBuffer(kBiases, offset: kBiasesOffset, index: 5)
        enc.setBuffer(vWeights, offset: vWeightsOffset, index: 6)
        enc.setBuffer(vScales, offset: vScalesOffset, index: 7)
        enc.setBuffer(vBiases, offset: vBiasesOffset, index: 8)
        enc.setBuffer(x, offset: 0, index: 9)
        enc.setBuffer(qOut, offset: qOutOffset, index: 10)
        enc.setBuffer(kOut, offset: kOutOffset, index: 11)
        enc.setBuffer(vOut, offset: vOutOffset, index: 12)
        var qVar = qRows
        var kvVar = kvRows
        var nVar = n
        enc.setBytes(&qVar, length: MemoryLayout<UInt32>.size, index: 13)
        enc.setBytes(&kvVar, length: MemoryLayout<UInt32>.size, index: 14)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 15)
        // Each projection keeps its own final group of up to four rows.
        let kvCopies = kEqualsV ? 1 : 2
        let rowGroups = (Int(qRows) + 3) / 4 + kvCopies * ((Int(kvRows) + 3) / 4)
        enc.dispatchDecode(MTLSize(width: (rowGroups + 1) / 2,
                                         height: 1,
                                         depth: 1),
                                 threads: MTLSize(width: Self.threadsPerThreadgroup,
                                                                 height: 1,
                                                                 depth: 1),
                                 conditional: conditional)
        enc.endEncoding()
    }
}
