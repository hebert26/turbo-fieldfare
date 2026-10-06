import Foundation
import Testing

@testable import TurboFieldfareRepackCore

/// Phase 5 task 5.1 coverage for the closed source catalog.
///
/// Two independent references are used:
/// 1. The Gemma literals are frozen here as plain values, transcribed from the
///    pre-change `SupportedModelSource` contract and the public callers
///    (`SourceFingerprint`, the repack CLI, the app installer client).
/// 2. The Qwen pin comes from `Qwen36OfficialMetadata`, the Phase 1 fixture
///    transcribed from `SHA256SUMS`, which is not maintained by the catalog.
@Suite struct ModelSourceCatalogTests {

  // MARK: - Gemma compatibility

  @Test func gemmaPublicAliasesKeepTheirExactValues() {
    // The catalog keeps these as the Gemma entry. Any changed name, type or value
    // would silently change the remote Gemma install and break callers outside P5.
    #expect(SupportedModelSource.displayName == "Gemma 4 26B-A4B IT 4-bit")
    #expect(SupportedModelSource.repoID == "mlx-community/gemma-4-26b-a4b-it-4bit")
    #expect(SupportedModelSource.revision
      == "0d77464eeb233a2da68ebf9d7dc4edaac7db956d")
    #expect(SupportedModelSource.sourceIndexSHA256
      == "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13")
    #expect(SupportedModelSource.approximateDownloadBytes == 14_620_479_420)
    #expect(SupportedModelSource.installedBytes == 14_291_921_884)
    #expect(SupportedModelSource.reserveBytes == 1_073_741_824)
  }

  @Test func gemmaInstallOptionsKeepRemoteInstallContract() {
    let output = URL(fileURLWithPath: "/tmp/mlx-gemma-source-alias-test")
    let options = SupportedModelSource.installOptions(
      outputDirectory: output, overwrite: true, token: "secret", resume: true)

    #expect(options.repoID == SupportedModelSource.repoID)
    #expect(options.revision == SupportedModelSource.revision)
    #expect(options.outputDir == output.path)
    #expect(options.token == "secret")
    #expect(options.overwrite == true)
    #expect(options.resume == true)
    // The Gemma path must keep requiring the pinned index fingerprint and its
    // disk reserve; the Qwen local entry must not have relaxed either.
    #expect(options.requireKnownSource == true)
    #expect(options.minFreeReserveBytes == SupportedModelSource.reserveBytes)
  }

  @Test func gemmaInstallOptionsDefaultResumeIsUnchanged() {
    let options = SupportedModelSource.installOptions(
      outputDirectory: URL(fileURLWithPath: "/tmp/mlx-gemma-default-test"),
      overwrite: false, token: nil)
    #expect(options.resume == false)
    #expect(options.overwrite == false)
    #expect(options.token == nil)
  }

  @Test func gemmaFingerprintTableStillKnowsOnlyGemma() {
    // `SourceFingerprint` gates `requireKnownSource` for the remote Gemma path.
    // P5 must not add or move an entry here, because a Qwen local source has no
    // remote download and must not be admitted by the remote fingerprint gate.
    #expect(SourceFingerprint.knownFingerprints
      == [SupportedModelSource.repoID: SupportedModelSource.sourceIndexSHA256])
    #expect(SourceFingerprint.modelID(
      forIndexSha256: SupportedModelSource.sourceIndexSHA256)
      == SupportedModelSource.repoID)
  }

  // MARK: - Qwen local source entry

  @Test func onlyTheExactOfficialQwenIdentityResolves() throws {
    let entry = try #require(ModelSourceCatalog.localSnapshotSource(
      repository: Qwen36OfficialMetadata.repository,
      revision: Qwen36OfficialMetadata.revision))

    #expect(entry.repository == Qwen36OfficialMetadata.repository)
    #expect(entry.revision == Qwen36OfficialMetadata.revision)
    // The production Qwen value is the one that must additionally enforce the
    // Phase 1 official tensor map; the flag is the seam tests flip, not a
    // production escape hatch.
    #expect(entry.enforcesOfficialTensorMap == true)
    #expect(entry.sidecarSHA256 == Qwen36OfficialMetadata.sidecarSHA256)
  }

  @Test func qwenEntryCarriesNoRemoteDownloadFields() throws {
    let entry = try #require(ModelSourceCatalog.localSnapshotSource(
      repository: Qwen36OfficialMetadata.repository,
      revision: Qwen36OfficialMetadata.revision))

    // Constructing the memberwise value with exactly these four labels proves the
    // stored property set. A remote URL, token, retry count or auth mode would
    // have to appear as a fifth member here.
    let expected = ModelSourceCatalog.LocalSnapshotSource(
      repository: Qwen36OfficialMetadata.repository,
      revision: Qwen36OfficialMetadata.revision,
      sidecarSHA256: Qwen36OfficialMetadata.sidecarSHA256,
      enforcesOfficialTensorMap: true)
    #expect(entry == expected)
  }

  @Test func gemmaDoesNotResolveThroughTheLocalSnapshotCatalog() {
    // The Gemma source is remote-only. Returning a local snapshot entry for it
    // would let a caller validate an arbitrary local directory as Gemma.
    #expect(ModelSourceCatalog.localSnapshotSource(
      repository: SupportedModelSource.repoID,
      revision: SupportedModelSource.revision) == nil)
  }

  @Test(arguments: [
    (repository: "Qwen/Qwen3.5-35B-A3B",
     revision: "995ad96eacd98c81ed38be0c5b274b04031597b0"),
    (repository: "Qwen/Qwen3.6-35B-A3B ",
     revision: "995ad96eacd98c81ed38be0c5b274b04031597b0"),
    (repository: " Qwen/Qwen3.6-35B-A3B",
     revision: "995ad96eacd98c81ed38be0c5b274b04031597b0"),
    (repository: "Qwen/Qwen3.6-35B-A3B\n",
     revision: "995ad96eacd98c81ed38be0c5b274b04031597b0"),
    (repository: "qwen/Qwen3.6-35B-A3B",
     revision: "995ad96eacd98c81ed38be0c5b274b04031597b0"),
    (repository: "Qwen/Qwen3.6-35b-a3b",
     revision: "995ad96eacd98c81ed38be0c5b274b04031597b0"),
    // A community quantization of the same model is not a release source.
    (repository: "mlx-community/Qwen3.6-35B-A3B-4bit",
     revision: "995ad96eacd98c81ed38be0c5b274b04031597b0"),
    (repository: "Qwen/Qwen3.6-35B-A3B",
     revision: "995ad96eacd98c81ed38be0c5b274b04031597b"),
    (repository: "Qwen/Qwen3.6-35B-A3B",
     revision: "995ad96eacd98c81ed38be0c5b274b04031597b00"),
    (repository: "Qwen/Qwen3.6-35B-A3B",
     revision: "995ad96eacd98c81ed38be0c5b274b04031597b1"),
    (repository: "Qwen/Qwen3.6-35B-A3B",
     revision: "995ad96eacd98c81ed38be0c5b274b04031597b0".uppercased()),
    (repository: "Qwen/Qwen3.6-35B-A3B", revision: "main"),
    (repository: "Qwen/Qwen3.6-35B-A3B", revision: "HEAD"),
    (repository: "Qwen/Qwen3.6-35B-A3B", revision: ""),
    (repository: "", revision: ""),
    (repository: SupportedModelSource.repoID,
     revision: SupportedModelSource.revision),
  ])
  func neighbouringIdentitiesDoNotResolve(candidate: (repository: String, revision: String)) {
    #expect(ModelSourceCatalog.localSnapshotSource(
      repository: candidate.repository,
      revision: candidate.revision) == nil,
      "resolved \(candidate.repository)@\(candidate.revision)")
  }
}
