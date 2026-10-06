import TurboFieldfareFormat

/// Bind the format-only descriptor to the single Phase 1 pinned inventory.
/// No source files are opened or hashed by this metadata comparison.
public enum OfficialSourceDescriptorValidation {
    public static func validate(_ descriptor: OfficialSourceDescriptor) throws {
        let identity = OfficialQwenSourceIdentity(
            repository: descriptor.repository,
            revision: descriptor.revision,
            storageProfile: descriptor.storageProfile,
            sidecarSHA256: descriptor.sidecarSHA256,
            shards: descriptor.shards.map {
                OfficialQwenShard(filename: $0.filename, sha256: $0.sha256)
            }.sorted { $0.filename < $1.filename })
        try OfficialQwenIdentity.validate(identity)
    }
}
