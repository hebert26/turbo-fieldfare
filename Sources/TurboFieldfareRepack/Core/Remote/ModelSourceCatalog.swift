import Foundation

/// The closed set of model sources the repacker understands.
enum ModelSourceCatalog {
    struct GemmaSource: Sendable, Equatable {
        let displayName: String
        let repository: String
        let revision: String
        let sourceIndexSHA256: String
        let approximateDownloadBytes: UInt64
        let installedBytes: UInt64
        let reserveBytes: UInt64
    }

    /// A source accepted only from a pre-existing local snapshot. It deliberately
    /// contains no remote URL, authentication, or retry policy.
    struct LocalSnapshotSource: Sendable, Equatable {
        let repository: String
        let revision: String
        let sidecarSHA256: [String: String]
        let enforcesOfficialTensorMap: Bool
    }

    static let gemma = GemmaSource(
        displayName: "Gemma 4 26B-A4B IT 4-bit",
        repository: "mlx-community/gemma-4-26b-a4b-it-4bit",
        revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        sourceIndexSHA256: "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        approximateDownloadBytes: 14_620_479_420,
        installedBytes: 14_291_921_884,
        reserveBytes: 1_073_741_824)

    static let qwen = LocalSnapshotSource(
        repository: "Qwen/Qwen3.6-35B-A3B",
        revision: "995ad96eacd98c81ed38be0c5b274b04031597b0",
        sidecarSHA256: [
            "config.json": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
            "configuration.json": "c1b09db419119513247e9b8b912c4b9897106c9b20c6cada7e107d993c5435eb",
            "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
            "model.safetensors.index.json": "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
            "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
            "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            "tokenizer_config.json": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
        ],
        enforcesOfficialTensorMap: true)

    static func localSnapshotSource(repository: String, revision: String) -> LocalSnapshotSource? {
        repository == qwen.repository && revision == qwen.revision ? qwen : nil
    }
}
