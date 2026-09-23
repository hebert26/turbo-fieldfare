import Darwin
import Foundation
import TurboFieldfare
import TurboFieldfareServerCore

// Every request line goes to stderr, which is unbuffered, while the ready line
// below goes to stdout, which is fully buffered when it is not a terminal. A
// server started with its output redirected therefore showed an empty log for
// its whole life and printed "ready" only as it exited - exactly inverted from
// what an operator needs. Line buffering puts the line where it is useful.
setvbuf(stdout, nil, _IOLBF, 0)

let arguments: ServerArguments
let runtimeConfiguration: RuntimeConfiguration
do {
    arguments = try ServerArguments.parse(Array(CommandLine.arguments.dropFirst()))
    // Resolved here so an unusable flag combination exits with usage instead of
    // failing after the model has started loading.
    runtimeConfiguration = try arguments.resolvedRuntimeConfiguration()
} catch ServerArgumentError.help {
    print(ServerArguments.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n\n\(ServerArguments.usage)\n".utf8))
    exit(2)
}

do {
    let signals = ServerTerminationSignals()
    let modelURL = URL(fileURLWithPath: arguments.model).standardizedFileURL
    let loaded = try await ServerModelLoader.load(
        modelDirectory: modelURL,
        assertedModelID: arguments.modelIDAssertion,
        maxContext: arguments.maxContext,
        visionPackURL: arguments.visionPack.map {
            URL(fileURLWithPath: $0).standardizedFileURL
        },
        visionResidencyPolicy: arguments.visionResidency,
        promptCacheMode: arguments.promptCacheMode,
        runtimeConfiguration: runtimeConfiguration)
    let server = TurboFieldfareHTTPServer(
        modelID: loaded.identity.apiModelID,
        queueLimit: arguments.queueLimit,
        backend: loaded.backend,
        visionCapability: loaded.visionCapability,
        modelFamily: loaded.identity.family,
        modelRevision: loaded.identity.sourceRevision)
    _ = try await server.start(port: arguments.port)
    let cache = loaded.identity.family == .qwen3_6
        ? "off(cached_tokens=0)" : arguments.promptCacheMode.rawValue
    print("TurboFieldfareServer ready at http://127.0.0.1:\(arguments.port) model=\(loaded.identity.apiModelID) family=\(loaded.identity.family.rawValue) context=\(arguments.maxContext) prompt_cache=\(cache) vision=\(loaded.visionCapability) vision_residency=\(arguments.visionResidency.rawValue)")

    _ = await signals.wait()
    try await server.shutdown()
    await signals.cancel()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
