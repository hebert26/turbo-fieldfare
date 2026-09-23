import Foundation

/// Compatibility aliases for the existing remote Gemma installer source.
/// Additional catalog entries must not change this remote-install contract.
public enum SupportedModelSource {
    private static let source = ModelSourceCatalog.gemma

    public static let displayName = source.displayName
    public static let repoID = source.repository
    public static let revision = source.revision
    public static let sourceIndexSHA256 = source.sourceIndexSHA256
    public static let approximateDownloadBytes = source.approximateDownloadBytes
    public static let installedBytes = source.installedBytes
    public static let reserveBytes = source.reserveBytes

    public static func installOptions(outputDirectory: URL,
                                      overwrite: Bool,
                                      token: String?,
                                      resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision,
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: true,
            minFreeReserveBytes: reserveBytes,
            overwrite: overwrite,
            resume: resume)
    }
}
