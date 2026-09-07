import Foundation

enum VisionCaptureSkillName: String, Sendable, CaseIterable {
    case main = "vision-capture"
    case efficientExploration = "vision-capture-efficient-exploration"
}

enum VisionCaptureSkillReaderError: Error, CustomStringConvertible {
    case checkoutNotFound
    case unknownSkill
    case unsafePath
    case unreadable
    case tooLarge

    var description: String {
        switch self {
        case .checkoutNotFound:
            "The TurboFieldfare checkout containing the VisionCapture skills was not found."
        case .unknownSkill:
            "That VisionCapture instruction file is not allowlisted."
        case .unsafePath:
            "The allowlisted instruction path did not resolve to its local regular file."
        case .unreadable:
            "The allowlisted VisionCapture instruction file could not be read."
        case .tooLarge:
            "The VisionCapture instruction file exceeds the proof-of-concept read limit."
        }
    }
}

struct VisionCaptureSkillReader: Sendable {
    private static let maximumBytes = 96 * 1_024
    private static let buildCheckoutRoot: URL = {
        var candidate = URL(fileURLWithPath: #filePath, isDirectory: false)
        for _ in 0..<5 {
            candidate.deleteLastPathComponent()
        }
        return candidate.standardizedFileURL
    }()
    private let checkoutRoot: URL

    init(modelDirectory: URL,
         executableURL: URL? = Bundle.main.executableURL,
         currentDirectoryURL: URL = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath,
            isDirectory: true),
         fileManager: FileManager = .default) throws {
        let starts = [
            Self.buildCheckoutRoot,
            executableURL?.deletingLastPathComponent(),
            currentDirectoryURL,
            modelDirectory.deletingLastPathComponent(),
        ].compactMap { $0 }
        guard let root = starts.lazy.compactMap({ start in
            Self.findCheckoutRoot(startingAt: start, fileManager: fileManager)
        }).first else {
            throw VisionCaptureSkillReaderError.checkoutNotFound
        }
        checkoutRoot = root
    }

    func read(_ rawName: String,
              fileManager: FileManager = .default) throws -> String {
        guard let name = VisionCaptureSkillName(rawValue: rawName) else {
            throw VisionCaptureSkillReaderError.unknownSkill
        }

        let relativePath: String
        switch name {
        case .main:
            relativePath = ".codex/skills/vision-capture/SKILL.md"
        case .efficientExploration:
            relativePath = ".codex/skills/vision-capture-efficient-exploration/SKILL.md"
        }

        return try readAllowlisted(relativePath, fileManager: fileManager)
    }

    private func readAllowlisted(
        _ relativePath: String,
        fileManager: FileManager
    ) throws -> String {
        let expected = checkoutRoot
            .appendingPathComponent(relativePath, isDirectory: false)
            .standardizedFileURL
        let resolved = expected.resolvingSymlinksInPath().standardizedFileURL
        let resolvedRoot = checkoutRoot.resolvingSymlinksInPath().standardizedFileURL
        guard expected.path == resolved.path,
              resolved.path.hasPrefix(resolvedRoot.path + "/") else {
            throw VisionCaptureSkillReaderError.unsafePath
        }
        let values = try? resolved.resourceValues(forKeys: [
            .isRegularFileKey,
            .fileSizeKey,
        ])
        guard values?.isRegularFile == true else {
            throw VisionCaptureSkillReaderError.unsafePath
        }
        guard let fileSize = values?.fileSize, fileSize <= Self.maximumBytes else {
            throw VisionCaptureSkillReaderError.tooLarge
        }
        guard let data = fileManager.contents(atPath: resolved.path),
              data.count <= Self.maximumBytes,
              let text = String(data: data, encoding: .utf8) else {
            throw VisionCaptureSkillReaderError.unreadable
        }
        return text
    }

    private static func findCheckoutRoot(startingAt start: URL,
                                         fileManager: FileManager) -> URL? {
        var candidate = start.standardizedFileURL
        while true {
            let package = candidate.appendingPathComponent("Package.swift").path
            let mainSkill = candidate.appendingPathComponent(
                ".codex/skills/vision-capture/SKILL.md").path
            let companion = candidate.appendingPathComponent(
                ".codex/skills/vision-capture-efficient-exploration/SKILL.md").path
            if fileManager.fileExists(atPath: package),
               fileManager.fileExists(atPath: mainSkill),
               fileManager.fileExists(atPath: companion) {
                return candidate
            }
            let parent = candidate.deletingLastPathComponent()
            if parent.path == candidate.path { return nil }
            candidate = parent
        }
    }
}
