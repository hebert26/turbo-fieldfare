import Darwin
import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppModelLocationTests {
    @Test func explicitURLWins() {
        let result = AppModelLocation.resolve(
            explicitURL: URL(fileURLWithPath: "/models/explicit.gturbo"),
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { _ in false })
        #expect(result.path == "/models/explicit.gturbo")
    }

    @Test func executableAncestorFindsPackageRootOutsideCWD() {
        let files: Set<String> = ["/repo/Package.swift", "/repo/Sources/TurboFieldfareApp/Mac"]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/repo/.build/debug/TurboFieldfareMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/gemma4.gturbo")
    }

    @Test func currentDirectoryCanBePackageRoot() {
        let files: Set<String> = ["/repo/Package.swift", "/repo/Sources/TurboFieldfareApp/Mac"]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/gemma4.gturbo")
    }

    @Test func checkoutSymlinkUsesReceiptBoundCanonicalModelLocation() throws {
        let root = try canonicalTemporaryRootURL(prefix: "model-location")
        defer { try? FileManager.default.removeItem(at: root) }
        let checkout = root.appendingPathComponent("repo", isDirectory: true)
        let scratch = checkout.appendingPathComponent("scratch", isDirectory: true)
        let appSources = checkout.appendingPathComponent(
            "Sources/TurboFieldfareApp/Mac", isDirectory: true)
        let canonical = root.appendingPathComponent(
            "Application Support/TurboFieldfare/gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: scratch, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: appSources, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: canonical, withIntermediateDirectories: true)
        _ = FileManager.default.createFile(
            atPath: checkout.appendingPathComponent("Package.swift").path,
            contents: Data())
        try FileManager.default.createSymbolicLink(
            at: scratch.appendingPathComponent("gemma4.gturbo"),
            withDestinationURL: canonical)

        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: checkout.appendingPathComponent(".build/release/TurboFieldfareMac"),
            currentDirectoryURL: root,
            applicationSupportURL: root.appendingPathComponent("Application Support"),
            fileExists: FileManager.default.fileExists(atPath:))

        #expect(result == canonical.standardizedFileURL)
    }

    @Test func standaloneAppFallsBackToApplicationSupport() {
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/Applications/TurboFieldfareMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { _ in false })
        #expect(result.path == "/support/TurboFieldfare/gemma4.gturbo")
    }

    @Test func qwenRegistrationUsesCanonicalParentForSymlinkedCheckoutScratch() throws {
        let root = try canonicalTemporaryRootURL(prefix: "qwen-location-symlink")
        defer { try? FileManager.default.removeItem(at: root) }
        let checkout = root.appendingPathComponent("checkout", isDirectory: true)
        let canonicalScratch = root.appendingPathComponent("canonical-scratch", isDirectory: true)
        try FileManager.default.createDirectory(at: checkout, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: canonicalScratch, withIntermediateDirectories: false)
        let packageFile = checkout.appendingPathComponent("Package.swift")
        _ = FileManager.default.createFile(atPath: packageFile.path, contents: Data())
        let appSources = checkout.appendingPathComponent("Sources/TurboFieldfareApp/Mac", isDirectory: true)
        try FileManager.default.createDirectory(at: appSources, withIntermediateDirectories: true)
        let scratch = checkout.appendingPathComponent("scratch", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: scratch, withDestinationURL: canonicalScratch)
        let packageFiles: Set<String> = [packageFile.path, appSources.path]

        let destination = AppModelLocation.qwenRegistrationDestination(
            explicitURL: nil,
            executableURL: checkout.appendingPathComponent(".build/release/TurboFieldfareMac"),
            currentDirectoryURL: root,
            applicationSupportURL: root.appendingPathComponent("Application Support", isDirectory: true),
            fileExists: packageFiles.contains)

        #expect(destination == canonicalScratch.appendingPathComponent(
            "qwen3.6-35b-a3b.gturbo", isDirectory: true).standardizedFileURL)
        #expect(try FileManager.default.destinationOfSymbolicLink(atPath: scratch.path)
            == canonicalScratch.path)
    }

    @Test func qwenRegistrationCanResolveUnderFreshApplicationSupportParent() throws {
        let root = try canonicalTemporaryRootURL(prefix: "qwen-location-fresh-support")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        let support = root.appendingPathComponent("Application Support", isDirectory: true)
        let destination = AppModelLocation.qwenRegistrationDestination(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/Applications/TurboFieldfareMac"),
            currentDirectoryURL: root,
            applicationSupportURL: support,
            fileExists: { _ in false })

        #expect(destination == support.appendingPathComponent(
            "TurboFieldfare/qwen3.6-35b-a3b.gturbo", isDirectory: true).standardizedFileURL)
        #expect(!FileManager.default.fileExists(atPath: support.path))
    }
}

private func canonicalTemporaryRootURL(prefix: String) throws -> URL {
    let temporaryPath = FileManager.default.temporaryDirectory.path
    guard let canonicalBuffer = temporaryPath.withCString({ realpath($0, nil) }) else {
        throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
    }
    defer { free(canonicalBuffer) }
    return URL(fileURLWithPath: String(cString: canonicalBuffer), isDirectory: true)
        .appendingPathComponent("\(prefix)-\(UUID().uuidString)", isDirectory: true)
}
