import CryptoKit
import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareOfficialQwenSource

/// Explicitly gated primitive captures. Neither branch loads a decoder model.
@Suite(.serialized) struct OfficialBF16ProjectionDiagnosticTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TURBO_P22_SYNTHETIC_PROJECTION"] == "1"))
    func captureFullWidthSyntheticProjection() throws {
        let columns = 2048
        let rows = 32
        let input = (0..<columns).map { index -> Float in
            let bits = UInt16(0x3c00 + index * 37 % 1024)
                | (index % 3 == 0 ? UInt16(0x8000) : 0)
            return Float(bitPattern: UInt32(bits) << 16)
        }
        var bits: [UInt16] = []
        bits.reserveCapacity(rows * columns)
        for row in 0..<rows {
            for index in 0..<columns {
                let magnitude = UInt16(0x3b00 + (index * 17 + row * 29) % 1536)
                let sign: UInt16 = (index + row) % 5 < 2 ? 0x8000 : 0
                bits.append(magnitude | sign)
            }
        }
        let source = try QwenBF16SyntheticSource.make(tensors: [
            QwenBF16LiteralTensor(name: "projection", rows: rows,
                                  columns: columns, bits: bits),
        ])
        defer { source.remove() }
        let context = try MetalContext()
        let weights = try QwenBF16Weights(
            context: context, source: source.handle,
            specifications: [QwenBF16TensorSpec(
                name: "projection", shardName: source.shardName,
                role: .dense, rows: rows, columns: columns)],
            residencyBudget: UInt64(bits.count * 2))
        let actual = try project(input, weights: weights, name: "projection",
                                 rows: rows, context: context)
        try capture(input: input, actual: actual, tensor: "synthetic-32x2048",
                    source: "test-owned-integer-pattern")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TURBO_P22_ORIGINAL_PROJECTION"] == "1"))
    func captureOriginalLayerZeroQKVProjection() throws {
        let environment = ProcessInfo.processInfo.environment
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let registration = root.appendingPathComponent("scratch/qwen3.6-35b-a3b.gturbo")
        guard environment["TURBO_P22_REGISTRATION"] == registration.path,
              let inputPath = environment["TURBO_P22_PROJECTION_INPUT"] else {
            throw ProjectionDiagnosticError.missingConfiguration
        }
        let inputData = try Data(contentsOf: URL(fileURLWithPath: inputPath))
        let input = try floats(inputData, expectedCount: 2048)
        let receipt = try OfficialSourceTrust.verify(
            at: registration, policy: .sizeCheckTrustedReceipt)
        let source = try OfficialSourceHandle(registrationURL: registration)
        try source.validateTrustedReceipt(receipt)
        let directory = try GTurboModelDirectory(protectedOfficialSource: source)
        let indexData = try directory.readMetadata(
            "model.safetensors.index.json", maxBytes: 8 * 1024 * 1024)
        let index = try JSONDecoder().decode(ProjectionIndex.self, from: indexData)
        let name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        let shard = try #require(index.weight_map[name])
        let context = try MetalContext()
        let weights = try QwenBF16Weights(
            context: context, source: source,
            specifications: [QwenBF16TensorSpec(
                name: name, shardName: shard, role: .dense,
                rows: 8192, columns: 2048)],
            residencyBudget: 8192 * 2048 * 2)
        let actual = try project(input, weights: weights, name: name,
                                 rows: 8192, context: context)
        try source.validateTrustedReceipt(receipt)
        #expect(try Data(contentsOf: URL(fileURLWithPath: inputPath)) == inputData)
        try capture(input: input, actual: actual, tensor: name,
                    source: receipt.descriptorContentSHA256)
    }

    private func project(_ input: [Float], weights: QwenBF16Weights,
                         name: String, rows: Int, context: MetalContext) throws -> [Float] {
        let inputBuffer = try #require(input.withUnsafeBytes { bytes in
            context.device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count,
                                      options: .storageModeShared)
        })
        let output = try #require(context.device.makeBuffer(
            length: rows * MemoryLayout<Float>.stride, options: .storageModeShared))
        let command = try #require(context.queue.makeCommandBuffer())
        try weights.encodeProjection(commandBuffer: command, tensorName: name,
                                     input: inputBuffer, tokenCount: 1, output: output)
        command.commit()
        command.waitUntilCompleted()
        #expect(command.status == .completed)
        guard command.status == .completed else {
            throw ProjectionDiagnosticError.gpuFailure
        }
        return Array(UnsafeBufferPointer(
            start: output.contents().bindMemory(to: Float.self, capacity: rows), count: rows))
    }

    private func capture(input: [Float], actual: [Float], tensor: String,
                         source: String) throws {
        let outputPath = try #require(ProcessInfo.processInfo.environment[
            "TURBO_P22_PROJECTION_OUTPUT"])
        let output = URL(fileURLWithPath: outputPath, isDirectory: true)
        #expect(try FileManager.default.contentsOfDirectory(atPath: outputPath).isEmpty)
        guard try FileManager.default.contentsOfDirectory(atPath: outputPath).isEmpty,
              input.allSatisfy(\.isFinite), actual.allSatisfy(\.isFinite) else {
            throw ProjectionDiagnosticError.invalidCapture
        }
        let inputData = data(input)
        let actualData = data(actual)
        try inputData.write(to: output.appendingPathComponent("input-normalized.fp32-le.bin"))
        try actualData.write(to: output.appendingPathComponent("in-proj-qkv.fp32-le.bin"))
        let receipt: [String: Any] = [
            "tensor": tensor, "sourceDescriptorSHA256": source,
            "inputCount": input.count, "outputCount": actual.count,
            "inputSHA256": digest(inputData), "outputSHA256": digest(actualData),
            "actualProductionEncoder": "QwenBF16Weights.encodeProjection",
            "decoderModelLoaded": false,
        ]
        try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("receipt.json"))
    }

    private func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func data(_ values: [Float]) -> Data {
        var result = Data(capacity: values.count * 4)
        for value in values {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { result.append(contentsOf: $0) }
        }
        return result
    }

    private func floats(_ data: Data, expectedCount: Int) throws -> [Float] {
        guard data.count == expectedCount * 4 else {
            throw ProjectionDiagnosticError.invalidCapture
        }
        return data.withUnsafeBytes { bytes in
            (0..<expectedCount).map {
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(
                    fromByteOffset: $0 * 4, as: UInt32.self)))
            }
        }
    }
}

private struct ProjectionIndex: Decodable {
    let weight_map: [String: String]
}

private enum ProjectionDiagnosticError: Error {
    case missingConfiguration, invalidCapture, gpuFailure
}
