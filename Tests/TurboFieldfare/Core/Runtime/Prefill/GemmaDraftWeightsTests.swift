import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite struct GemmaDraftWeightsTests {
    private func validManifest() -> [String: Any] {
        var cursor = 0
        var tensors: [String: Any] = [:]
        for (name, shape) in GemmaDraftWeights.expectedShapes.sorted(by: { $0.key < $1.key }) {
            cursor = (cursor + 63) / 64 * 64
            let count = shape.reduce(1, *)
            let matrix = shape.count == 2
            let length = matrix ? count / 2 + count / 64 * 4 : count * 2
            tensors[name] = ["shape": shape, "bits": matrix ? 4 : 16, "offset": cursor,
                             "length": length, "scaleOffset": matrix ? cursor + count / 2 : 0,
                             "biasOffset": matrix ? cursor + count / 2 + count / 64 * 2 : 0]
            cursor += length
        }
        return ["version": 1, "kind": "gemma4-draft-affine-int4", "groupSize": 64,
                "sourceModel": "google/gemma-4-26B-A4B-it-assistant",
                "sourceRevision": GemmaDraftWeights.sourceRevision,
                "sourceSHA256": GemmaDraftWeights.sourceSHA256,
                "weightBytes": (cursor + 16383) / 16384 * 16384,
                "weightsSHA256": GemmaDraftWeights.packedSHA256, "tensors": tensors]
    }

    private func validate(_ object: [String: Any]) throws {
        let bytes = try JSONSerialization.data(withJSONObject: object)
        let manifest = try JSONDecoder().decode(GemmaDraftWeights.Manifest.self, from: bytes)
        try GemmaDraftWeights.validate(manifest)
    }

    @Test func acceptsCompleteLayout() throws {
        try validate(validManifest())
    }

    @Test(arguments: ["shape", "overlap", "outside", "scale", "missing", "source", "hash"])
    func rejectsInvalidLayout(change: String) throws {
        var object = validManifest()
        if change == "source" {
            object["sourceRevision"] = String(repeating: "0", count: 40)
        } else if change == "hash" {
            object["weightsSHA256"] = String(repeating: "0", count: 64)
        } else {
            var tensors = try #require(object["tensors"] as? [String: Any])
            let name = "pre_projection.weight"
            var tensor = try #require(tensors[name] as? [String: Any])
            switch change {
            case "shape": tensor["shape"] = [Int.max, Int.max]
            case "overlap": tensor["offset"] = 0
            case "outside": tensor["length"] = Int.max
            case "scale": tensor["scaleOffset"] = -2
            default: break
            }
            tensors[name] = change == "missing" ? nil : tensor
            object["tensors"] = tensors
        }
        #expect(throws: (any Error).self) { try validate(object) }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TURBOFIELDFARE_DRAFT_PACK"] != nil))
    func loadsRequestedDraftPack() throws {
        let path = try #require(ProcessInfo.processInfo.environment["TURBOFIELDFARE_DRAFT_PACK"])
        let context = try MetalContext()
        let weights = try GemmaDraftWeights(directory: URL(fileURLWithPath: path), device: context.device)
        #expect(weights.buffer.length == 236126208)
        #expect(weights.tensor("model.embed_tokens.weight").shape == [262144, 1024])
    }
}
