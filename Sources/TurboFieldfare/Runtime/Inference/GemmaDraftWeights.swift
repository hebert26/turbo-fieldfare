import Darwin
import Foundation
import Metal

/// Owns the small draft model in a separate weight file.
final class GemmaDraftWeights {
    struct Tensor: Decodable {
        let shape: [Int]
        let bits: Int
        let offset: Int
        let scaleOffset: Int
        let biasOffset: Int
        let length: Int
    }

    struct Manifest: Decodable {
        let version: Int
        let kind: String
        let groupSize: Int
        let sourceModel: String
        let sourceRevision: String
        let sourceSHA256: String
        let weightBytes: Int
        let weightsSHA256: String
        let tensors: [String: Tensor]
    }

    static let sourceRevision = "6e5aaaf4c42b98394530b8fda2e95cadd65c151c"
    static let sourceSHA256 = "006c8713b29b046e296c0f0b7c524aa9264ae07ae8b0c3888b9bc062242297ff"
    static let packedSHA256 = "d20f504cd31d26df56bc806668e984ef32939e71a5bdc693c65926831c694b5d"
    let manifest: Manifest
    private let resident: ResidentBuffer
    var buffer: MTLBuffer { resident.buffer }

    init(directory: URL, device: MTLDevice) throws {
        let manifestURL = directory.appendingPathComponent("manifest.json")
        let size = try manifestURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        guard size > 0, size <= 65_536 else {
            throw ModelError.indexCorrupt(detail: "Invalid draft manifest size")
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        try Self.validate(manifest)
        let weightsURL = directory.appendingPathComponent("weights.bin")
        let fd = open(weightsURL.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)
        guard fd >= 0 else {
            throw ModelError.posixFailed(call: "open draft weights", errno: errno)
        }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG,
              info.st_size == manifest.weightBytes else {
            throw ModelError.indexCorrupt(detail: "Incomplete draft weights")
        }
        try Sha256Verifier.verifyFile(fileDescriptor: fd, named: "draft weights",
                                      expectedHex: manifest.weightsSHA256)
        resident = try ResidentBuffer(fileURL: weightsURL, fileOffset: 0,
                                      residentSize: UInt64(manifest.weightBytes),
                                      device: device, fileDescriptor: fd)
        self.manifest = manifest
    }

    func tensor(_ name: String) -> Tensor {
        // The complete tensor set is checked before any GPU buffer is made.
        precondition(Self.expectedShapes[name] != nil)
        return manifest.tensors[name]!
    }

    static func validate(_ manifest: Manifest) throws {
        func require(_ condition: Bool, _ detail: String) throws {
            guard condition else { throw ModelError.indexCorrupt(detail: detail) }
        }
        try require(manifest.version == 1 && manifest.kind == "gemma4-draft-affine-int4"
                    && manifest.groupSize == 64, "Unsupported draft format")
        try require(manifest.sourceModel == "google/gemma-4-26B-A4B-it-assistant"
                    && manifest.sourceRevision == sourceRevision
                    && manifest.sourceSHA256 == sourceSHA256, "Unexpected draft source")
        try require(manifest.weightBytes > 0 && manifest.weightBytes <= 320 * 1024 * 1024
                    && manifest.weightBytes.isMultiple(of: 16_384), "Invalid draft weight size")
        try require(manifest.weightsSHA256 == packedSHA256, "Unexpected draft weight hash")
        try require(Set(manifest.tensors.keys) == Set(expectedShapes.keys), "Invalid draft tensor set")
        var end = 0
        for (name, tensor) in manifest.tensors.sorted(by: { $0.value.offset < $1.value.offset }) {
            try require(tensor.shape == expectedShapes[name], "Invalid draft shape: \(name)")
            try require(tensor.offset >= end && tensor.offset.isMultiple(of: 64)
                        && tensor.offset <= manifest.weightBytes && tensor.length > 0
                        && tensor.length <= manifest.weightBytes - tensor.offset,
                        "Invalid draft span: \(name)")
            let count = tensor.shape.reduce(1, *)
            if tensor.shape.count == 2 {
                let weightBytes = count / 2
                let scaleBytes = count / 64 * 2
                try require(tensor.bits == 4 && tensor.length == weightBytes + scaleBytes * 2
                            && tensor.scaleOffset == tensor.offset + weightBytes
                            && tensor.biasOffset == tensor.scaleOffset + scaleBytes,
                            "Invalid draft matrix: \(name)")
            } else {
                try require(tensor.bits == 16 && tensor.length == count * 2
                            && tensor.scaleOffset == 0 && tensor.biasOffset == 0,
                            "Invalid draft vector: \(name)")
            }
            end = tensor.offset + tensor.length
        }
    }

    static let expectedShapes: [String: [Int]] = {
        var shapes = [
            "model.embed_tokens.weight": [262144, 1024],
            "model.norm.weight": [1024],
            "pre_projection.weight": [1024, 5632],
            "post_projection.weight": [2816, 1024],
        ]
        for layer in 0..<4 {
            let prefix = "model.layers.\(layer)."
            let head = layer == 3 ? 512 : 256
            for name in ["input_layernorm", "post_attention_layernorm",
                         "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
                shapes[prefix + name + ".weight"] = [1024]
            }
            shapes[prefix + "layer_scalar"] = [1]
            shapes[prefix + "self_attn.q_norm.weight"] = [head]
            shapes[prefix + "self_attn.q_proj.weight"] = [16 * head, 1024]
            shapes[prefix + "self_attn.o_proj.weight"] = [1024, 16 * head]
            shapes[prefix + "mlp.gate_proj.weight"] = [8192, 1024]
            shapes[prefix + "mlp.up_proj.weight"] = [8192, 1024]
            shapes[prefix + "mlp.down_proj.weight"] = [1024, 8192]
        }
        return shapes
    }()
}
