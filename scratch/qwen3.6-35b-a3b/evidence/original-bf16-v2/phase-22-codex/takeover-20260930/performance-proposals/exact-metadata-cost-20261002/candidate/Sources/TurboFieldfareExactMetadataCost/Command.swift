import Foundation
import Darwin
import TurboFieldfareOfficialQwenSource

private struct MetadataDiagnosticError: Error { let detail: String }
private func require(_ condition: Bool, _ detail: String) throws {
    if !condition { throw MetadataDiagnosticError(detail: detail) }
}
private func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
private func cpu() throws -> (user: Int64, system: Int64) {
    var value = rusage()
    guard getrusage(RUSAGE_SELF, &value) == 0 else {
        throw MetadataDiagnosticError(detail: "getrusage failed")
    }
    return (Int64(value.ru_utime.tv_sec) * 1_000_000_000 + Int64(value.ru_utime.tv_usec) * 1000,
            Int64(value.ru_stime.tv_sec) * 1_000_000_000 + Int64(value.ru_stime.tv_usec) * 1000)
}

@main struct ExactMetadataCostCommand {
    static func main() {
        guard CommandLine.arguments.count == 3 else {
            FileHandle.standardError.write(Data("usage: TurboFieldfareExactMetadataCost registration.gturbo report.json\n".utf8))
            exit(2)
        }
        let output = URL(fileURLWithPath: CommandLine.arguments[2])
        do {
            try require(!FileManager.default.fileExists(atPath: output.path), "report already exists")
            let registration = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
            let initCPU = try cpu()
            let verifyStart = now()
            // This fixed policy has no payload hashing or full-verifier fallback.
            let receipt = try OfficialSourceTrust.verify(at: registration, policy: .sizeCheckTrustedReceipt)
            let verifyEnd = now()
            try require(!receipt.files.isEmpty && receipt.files.count <= 64, "unbounded receipt inventory")
            let handleStart = now()
            let handle = try OfficialSourceHandle(registrationURL: registration)
            let handleEnd = now()
            let afterInitCPU = try cpu()
            let files = UInt64(receipt.files.count)
            // One retained handle. Each call still performs every fresh observation.
            let warmOff = OfficialSourceMetadataCost(clockEnabled: false)
            let warmOn = OfficialSourceMetadataCost(clockEnabled: true)
            try handle.validateTrustedReceipt(receipt)
            try handle.validateTrustedReceipt(receipt, metadataCost: warmOff)
            try handle.validateTrustedReceipt(receipt, metadataCost: warmOn)
            var batches: [[String: Any]] = []
            let iterations = 32
            for mode in ["baseline", "clockOff", "clockOn", "clockOn", "clockOff", "baseline"] {
                let measurement: OfficialSourceMetadataCost? = mode == "baseline" ? nil
                    : OfficialSourceMetadataCost(clockEnabled: mode == "clockOn")
                let beforeCPU = try cpu()
                let start = now()
                for _ in 0..<iterations {
                    try Task.checkCancellation()
                    try handle.validateTrustedReceipt(receipt, metadataCost: measurement)
                }
                let end = now()
                let afterCPU = try cpu()
                var record: [String: Any] = ["mode": mode, "successfulCalls": iterations,
                    "wallNanoseconds": end - start,
                    "userCPUNanoseconds": afterCPU.user - beforeCPU.user,
                    "systemCPUNanoseconds": afterCPU.system - beforeCPU.system]
                if let measurement {
                    let counts = measurement.snapshot()
                    let calls = UInt64(iterations)
                    try require(counts.total.count == calls && counts.binding.count == 3 * calls
                        && counts.receiptNames.count == calls && counts.root.count == 4 * calls
                        && counts.listing.count == 2 * calls && counts.fileOpen.count == 2 * files * calls
                        && counts.heldStat.count == 2 * files * calls && counts.namedStat.count == 2 * files * calls
                        && counts.fileCheckAndRemember.count == 2 * files * calls
                        && counts.fileClose.count == 2 * files * calls, "actual check counts differ")
                    record["actualIntervals"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(counts))
                } else {
                    record["actualIntervalsMeasured"] = false
                }
                batches.append(record)
            }
            let result: [String: Any] = ["passed": true, "metadataOnly": true,
                "payloadBytesReadByAuditedPath": 0, "modelConstructed": false,
                "payloadReadEvidence": "Static call-path audit. Fixed sizeCheckTrustedReceipt policy and handle validation. No model or tensor calls. Not system IO tracing.",
                "sourceDescriptorSHA256": receipt.descriptorContentSHA256,
                "receiptFileCount": receipt.files.count, "steadyValidationCalls": iterations * 6,
                "warmValidationCalls": 3, "trustedReceiptMetadataNanoseconds": verifyEnd - verifyStart,
                "handleInitializationNanoseconds": handleEnd - handleStart,
                "initializationUserCPUNanoseconds": afterInitCPU.user - initCPU.user,
                "initializationSystemCPUNanoseconds": afterInitCPU.system - initCPU.system,
                "batches": batches,
                "zeroMeaning": "Clock-off duration is unmeasured. Baseline primitive counts are unmeasured.",
                "scope": "Current exact metadata validation only. No new validator or app throughput qualification."]
            let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: output, options: .atomic)
            print(String(decoding: data, as: UTF8.self))
        } catch {
            FileHandle.standardError.write(Data("metadata diagnostic failed: \(error)\n".utf8))
            if !FileManager.default.fileExists(atPath: output.path) {
                do {
                    let data = try JSONSerialization.data(withJSONObject: ["passed": false, "error": String(describing: error)], options: [.sortedKeys])
                    try data.write(to: output, options: .atomic)
                } catch { FileHandle.standardError.write(Data("failure receipt write failed: \(error)\n".utf8)) }
            }
            exit(1)
        }
    }
}
