import Foundation
import Darwin
import TurboFieldfare

@main struct ExactBlockCommand {
    @MainActor static func main() async {
        guard CommandLine.arguments.count == 4 else {
            FileHandle.standardError.write(Data("usage: TurboFieldfareBaselineMemoryProbe registration.json request.json report.json\n".utf8))
            exit(2)
        }
        let output = URL(fileURLWithPath: CommandLine.arguments[3])
        do {
            guard !FileManager.default.fileExists(atPath: output.path) else {
                throw CocoaError(.fileWriteFileExists)
            }
            let data = try await QwenExactBlockProbe.run(
                registrationURL: URL(fileURLWithPath: CommandLine.arguments[1]),
                requestURL: URL(fileURLWithPath: CommandLine.arguments[2]))
            try data.write(to: output, options: .atomic)
            print(String(decoding: data, as: UTF8.self))
        } catch {
            let failure = ["passed": false, "error": String(describing: error)] as [String: Any]
            do {
                let data = try JSONSerialization.data(withJSONObject: failure, options: [.sortedKeys])
                if !FileManager.default.fileExists(atPath: output.path) { try data.write(to: output, options: .atomic) }
            } catch { FileHandle.standardError.write(Data("failure receipt write: \(error)\n".utf8)) }
            FileHandle.standardError.write(Data("exact block probe failed: \(error)\n".utf8))
            exit(1)
        }
    }
}
