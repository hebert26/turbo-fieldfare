import Foundation
import Testing
import TurboFieldfare
import TurboFieldfareOfficialQwenSource

@Suite struct OfficialSourceIntegrityPolicyTests {
    @Test func fullPolicyAdapterRejectsMissingLogicalDirectory() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("official-source-full-\(UUID().uuidString)", isDirectory: true)
        #expect(!FileManager.default.fileExists(atPath: url.path))

        #expect(throws: (any Error).self) {
            _ = try ModelIntegrityPolicy.fullSha256.verifyOfficialSource(at: url)
        }
    }

    @Test func trustedPolicyAdapterRejectsMissingLogicalDirectory() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("official-source-trusted-\(UUID().uuidString)", isDirectory: true)
        #expect(!FileManager.default.fileExists(atPath: url.path))

        #expect(throws: (any Error).self) {
            _ = try ModelIntegrityPolicy.sizeCheckTrustedReceipt.verifyOfficialSource(at: url)
        }
    }

    @Test func sourcePolicyAdaptersDispatchToPinnedVerifierAndFailClosedWhenReceiptIsMissing() throws {
        let fixture = try TinyOfficialSourceIntegrityFixture.make(includeReceipt: false)
        defer { fixture.remove() }

        #expect(ModelIntegrityPolicy.fullSha256.officialSourcePolicy == .fullSha256)
        #expect(ModelIntegrityPolicy.sizeCheckTrustedReceipt.officialSourcePolicy
                == .sizeCheckTrustedReceipt)
        let checksumData = try Data(contentsOf: fixture.checksumManifestURL)
        #expect(checksumData.count == 3_571)
        #expect(try Sha256Verifier.hashFile(at: fixture.checksumManifestURL)
                == TinyOfficialSourceIntegrityFixture.checksumManifestSHA256)
        #expect(try Data(contentsOf: fixture.firstShardURL) == Data("abc".utf8))
        #expect(try Sha256Verifier.hashFile(at: fixture.firstShardURL)
                == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

        do {
            _ = try ModelIntegrityPolicy.fullSha256.verifyOfficialSource(
                at: fixture.modelDirectory)
            Issue.record("Expected the real pinned verifier to reject the tiny first shard")
        } catch let error as OfficialQwenPayloadVerificationError {
            guard case let .sourceFingerprintRejected(path, digest) = error else {
                Issue.record("Expected first-shard sourceFingerprintRejected, got \(error)")
                return
            }
            #expect(path == fixture.firstShardURL.path)
            #expect(digest
                    == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        } catch {
            Issue.record("Expected official payload fingerprint rejection, got \(error)")
        }

        do {
            _ = try ModelIntegrityPolicy.sizeCheckTrustedReceipt.verifyOfficialSource(
                at: fixture.modelDirectory)
            Issue.record("Expected trusted reopen to reject the missing receipt")
        } catch let error as OfficialSourceTrust.TrustError {
            if case .missing = error {
                // Expected: trusted policy fails closed without a verified receipt.
            } else {
                Issue.record("Expected TrustError.missing, got \(error)")
            }
        } catch {
            Issue.record("Expected TrustError.missing, got \(error)")
        }
    }
}
