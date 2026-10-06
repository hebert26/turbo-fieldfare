import Foundation
import Testing
@testable import TurboFieldfare

@Suite struct QwenMetalContractTests {
    @Test func abiTypesBindingsAndLayoutMatchLiteralContract() throws {
        #expect(QwenMetalABI.packedValueByteWidth == 1)
        #expect(QwenMetalABI.affineMetadataByteWidth == 2)
        #expect(QwenMetalABI.dimensionByteWidth == 4)
        #expect(QwenMetalABI.affineGroupSize == 64)
        #expect(QwenMetalABI.maximumAddressableBytes == UInt64(UInt32.max))

        #expect(QwenMetalBufferIndex.allCases.map(\.rawValue) == Array(0 ... 7))
        #expect(QwenMetalBufferIndex.allCases == [
            .parameters, .input, .weights, .scales, .biases, .output, .scratch, .state,
        ])

        #expect(MemoryLayout<QwenMetalAffineLayout>.size == 32)
        #expect(MemoryLayout<QwenMetalAffineLayout>.stride == 32)
        #expect(MemoryLayout<QwenMetalAffineLayout>.alignment == 4)
        #expect(MemoryLayout<QwenMetalAffineLayout>.offset(of: \.rowCount) == 0)
        #expect(MemoryLayout<QwenMetalAffineLayout>.offset(of: \.columnCount) == 4)
        #expect(MemoryLayout<QwenMetalAffineLayout>.offset(of: \.valuesRowStrideBytes) == 8)
        #expect(MemoryLayout<QwenMetalAffineLayout>.offset(of: \.metadataRowStrideBytes) == 12)
        #expect(MemoryLayout<QwenMetalAffineLayout>.offset(of: \.groupsPerRow) == 16)
        #expect(MemoryLayout<QwenMetalAffineLayout>.offset(of: \.bitWidth) == 20)
        #expect(MemoryLayout<QwenMetalAffineLayout>.offset(of: \.groupSize) == 24)
        #expect(MemoryLayout<QwenMetalAffineLayout>.offset(of: \.reserved) == 28)
    }

    @Test func affineLayoutUsesRowLocalGroup64AndPermitsRemainders() throws {
        let layout = try QwenMetalAffineLayout(rows: 2, columns: 65, bitWidth: 4)

        #expect(layout.rowCount == 2)
        #expect(layout.columnCount == 65)
        #expect(layout.valuesRowStrideBytes == 33)
        #expect(layout.metadataRowStrideBytes == 4)
        #expect(layout.groupsPerRow == 2)
        #expect(layout.bitWidth == 4)
        #expect(layout.groupSize == 64)
        #expect(layout.reserved == 0)

        let int8Layout = try QwenMetalAffineLayout(rows: 1, columns: 64, bitWidth: 8)
        #expect(int8Layout.valuesRowStrideBytes == 64)
        #expect(int8Layout.groupsPerRow == 1)
        #expect(int8Layout.metadataRowStrideBytes == 2)
    }

    @Test func affineLayoutRejectsInvalidUnsupportedAndUnaddressableInputs() {
        #expect(throws: QwenMetalContractError.invalidDimension(name: "rows", value: 0)) {
            try QwenMetalAffineLayout(rows: 0, columns: 1, bitWidth: 4)
        }
        #expect(throws: QwenMetalContractError.invalidDimension(name: "columns", value: -1)) {
            try QwenMetalAffineLayout(rows: 1, columns: -1, bitWidth: 4)
        }
        #expect(throws: QwenMetalContractError.dimensionOutOfRange(
            name: "rows", value: Int(UInt32.max) + 1
        )) {
            try QwenMetalAffineLayout(rows: Int(UInt32.max) + 1, columns: 1, bitWidth: 4)
        }
        #expect(throws: QwenMetalContractError.unsupportedBitWidth(3)) {
            try QwenMetalAffineLayout(rows: 1, columns: 1, bitWidth: 3)
        }
        #expect(throws: QwenMetalContractError.addressRangeOverflow(
            component: "values", requiredBytes: 4_295_032_832
        )) {
            try QwenMetalAffineLayout(rows: 65_537, columns: 65_536, bitWidth: 8)
        }
        #expect(throws: QwenMetalContractError.bufferLengthOverflow(
            component: "values", requiredBytes: 18_446_744_065_119_617_025
        )) {
            try QwenMetalAffineLayout(
                rows: Int(UInt32.max), columns: Int(UInt32.max), bitWidth: 8)
        }
    }

    @Test func featureRequirementsFailForTheFirstMissingFeature() throws {
        let required: QwenMetalFeatureSet = [.bfloat16, .nonuniformThreadgroups]
        try QwenMetalAffineLayout.require(required, available: required)

        #expect(throws: QwenMetalContractError.unsupportedFeature(.bfloat16)) {
            try QwenMetalAffineLayout.require(required, available: [.nonuniformThreadgroups])
        }
        #expect(throws: QwenMetalContractError.unsupportedFeature(.simdgroupMatrix)) {
            try QwenMetalAffineLayout.require([.simdgroupMatrix], available: [])
        }
    }

    @Test func qwenCommonIsDiscoveredOnceAfterExistingProductionModules() throws {
        let existingModules = [
            "dequant_int4", "gemma_verify_int4", "dequant_int8", "rmsnorm", "rope", "attention", "moe",
            "gemma_verify_moe",
            "logit", "utility", "fused", "gemma_verify_fused", "prefill", "vision",
        ]
        #expect(MetalContext.shaderModules == existingModules + [
            "qwen_common", "qwen_bf16", "qwen_full_attention", "qwen_linear_attention", "qwen_moe",
            "qwen_vision",
        ])
        #expect(MetalContext.shaderModules.filter { $0 == "qwen_common" }.count == 1)
        #expect(MetalContext.shaderModules.filter { $0 == "qwen_vision" }.count == 1)
        #expect(MetalContext.shaderModules.filter { $0 == "qwen_moe" }.count == 1)
        #expect(MetalContext.shaderModules.filter { $0 == "qwen_full_attention" }.count == 1)
        #expect(MetalContext.shaderModules.filter { $0 == "qwen_linear_attention" }.count == 1)
        #expect(Set(MetalContext.shaderModules).count == MetalContext.shaderModules.count)

        let sourceURLs = try MetalContext.shaderSourceURLs()
        #expect(sourceURLs.count == MetalContext.shaderModules.count)
        let URLsByModule = Dictionary(uniqueKeysWithValues: zip(MetalContext.shaderModules, sourceURLs))
        let qwenCommonURL = try #require(URLsByModule["qwen_common"])
        let qwenFullAttentionURL = try #require(URLsByModule["qwen_full_attention"])
        let qwenLinearAttentionURL = try #require(URLsByModule["qwen_linear_attention"])
        let qwenMoEURL = try #require(URLsByModule["qwen_moe"])
        let qwenVisionURL = try #require(URLsByModule["qwen_vision"])
        #expect(qwenCommonURL.lastPathComponent == "qwen_common.metal")
        #expect(qwenFullAttentionURL.lastPathComponent == "qwen_full_attention.metal")
        #expect(qwenLinearAttentionURL.lastPathComponent == "qwen_linear_attention.metal")
        #expect(qwenMoEURL.lastPathComponent == "qwen_moe.metal")
        #expect(qwenVisionURL.lastPathComponent == "qwen_vision.metal")
        #expect(qwenCommonURL.path.contains("Metal/Qwen/"))
        #expect(qwenFullAttentionURL.path.contains("Metal/Qwen/"))
        #expect(qwenLinearAttentionURL.path.contains("Metal/Qwen/"))
        #expect(qwenMoEURL.path.contains("Metal/Qwen/"))
        #expect(qwenVisionURL.path.hasSuffix("Metal/Qwen/qwen_vision.metal"))
    }

    @Test func qwenMoERegistryContainsAllConcretePipelines() throws {
        let URLsByModule = Dictionary(uniqueKeysWithValues: zip(MetalContext.shaderModules, try MetalContext.shaderSourceURLs()))
        let qwenURL = try #require(URLsByModule["qwen_moe"])
        let source = try String(contentsOf: qwenURL, encoding: .utf8)
        for kernel in [
            "qwen_moe_route_top8_fp32", "qwen_moe_clear_fp32",
            "qwen_moe_routed_gate_up_int4", "qwen_moe_routed_down_add_int4",
            "qwen_moe_affine_project", "qwen_moe_silu_multiply",
            "qwen_moe_shared_epilogue",
        ] {
            #expect(source.contains("kernel void \(kernel)"), "Missing Qwen MoE pipeline: \(kernel)")
        }
    }

    @Test func injectedMissingQwenResourceUsesTheProductionMissingResourceError() throws {
        let realURLs = try MetalContext.shaderSourceURLs()
        let URLsByModule = Dictionary(uniqueKeysWithValues: zip(MetalContext.shaderModules, realURLs))

        do {
            _ = try MetalContext.shaderSourceURLs { module, _, _ in
                module == "qwen_common" ? nil : URLsByModule[module]
            }
            Issue.record("Expected the injected missing Qwen resource to throw")
        } catch MetalError.missingShaderResource(let module) {
            #expect(module == "qwen_common")
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test func commonMSLDeclaresTheHostContractAndNoKernel() throws {
        let URLsByModule = Dictionary(uniqueKeysWithValues: zip(MetalContext.shaderModules, try MetalContext.shaderSourceURLs()))
        let qwenURL = try #require(URLsByModule["qwen_common"])
        let source = try String(contentsOf: qwenURL, encoding: .utf8)

        for declaration in [
            "typedef uchar QwenMetalPackedValue;",
            "typedef bfloat QwenMetalAffineMetadata;",
            "typedef uint QwenMetalDimension;",
            "QwenMetalBufferIndexParameters = 0",
            "QwenMetalBufferIndexInput = 1",
            "QwenMetalBufferIndexWeights = 2",
            "QwenMetalBufferIndexScales = 3",
            "QwenMetalBufferIndexBiases = 4",
            "QwenMetalBufferIndexOutput = 5",
            "QwenMetalBufferIndexScratch = 6",
            "QwenMetalBufferIndexState = 7",
            "QwenMetalPackedValueByteWidth = 1",
            "QwenMetalAffineMetadataByteWidth = 2",
            "QwenMetalDimensionByteWidth = 4",
            "QwenMetalAffineGroupSize = 64",
            "qwenMetalValueByteOffset",
            "qwenMetalMetadataIndex",
        ] {
            #expect(source.contains(declaration), "Missing MSL declaration: \(declaration)")
        }
        let expectedLayoutDeclaration = """
        struct QwenMetalAffineLayout {
            QwenMetalDimension rowCount;
            QwenMetalDimension columnCount;
            QwenMetalDimension valuesRowStrideBytes;
            QwenMetalDimension metadataRowStrideBytes;
            QwenMetalDimension groupsPerRow;
            QwenMetalDimension bitWidth;
            QwenMetalDimension groupSize;
            QwenMetalDimension reserved;
        };
        """
        #expect(source.contains(expectedLayoutDeclaration))
        #expect(!source.contains("kernel void"))
        #expect(!source.contains("kernel "))
    }
}
