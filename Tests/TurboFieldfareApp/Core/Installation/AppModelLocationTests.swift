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
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("model-location-\(UUID().uuidString)", isDirectory: true)
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
}
