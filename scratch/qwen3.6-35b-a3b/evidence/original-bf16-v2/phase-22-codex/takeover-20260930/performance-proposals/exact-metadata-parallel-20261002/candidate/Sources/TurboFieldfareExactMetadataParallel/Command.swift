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

@main struct ExactMetadataParallelCommand {
    static func main() {
        guard CommandLine.arguments.count == 3 else {
            FileHandle.standardError.write(Data("usage: TurboFieldfareExactMetadataParallel registration.gturbo report.json\n".utf8))
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
            try require(receipt.files.count == 39, "expected the pinned 39-file receipt")
            let files = UInt64(receipt.files.count)
            // Factory success binds warm eligibility to this exact handle and receipt.
            let diagnostic = try handle.metadataParallelDiagnostic(receipt)
            for workers in [1, 2, 4] {
                for clock in [false, true] {
                    _ = try diagnostic.validate(workers: workers,
                        metadataCost: OfficialSourceMetadataCost(clockEnabled: clock))
                }
            }
            var batches: [[String: Any]] = []
            var sums = [[UInt64]](repeating: [UInt64](repeating: 0, count: 5), count: 2)
            let experimentStart = now()
            let iterations = 128
            func batch(workers: Int, clock: Bool, count: Int, order: Int?, cycle: Int) throws {
                try require(now() - experimentStart < 50_000_000_000, "bounded experiment deadline")
                let measurement = OfficialSourceMetadataCost(clockEnabled: clock)
                var peakLanes = 0
                var peakFiles = 0
                let beforeCPU = try cpu()
                let start = now()
                for _ in 0..<count {
                    try Task.checkCancellation()
                    let resources = try diagnostic.validate(workers: workers, metadataCost: measurement)
                    try require(resources.finalActiveLanes == 0 && resources.finalActiveFileDescriptors == 0,
                                "worker or descriptor remained active")
                    try require(resources.maxActiveLanes <= workers && resources.maxActiveFileDescriptors <= workers,
                                "worker resource bound exceeded")
                    peakLanes = max(peakLanes, resources.maxActiveLanes)
                    peakFiles = max(peakFiles, resources.maxActiveFileDescriptors)
                }
                let end = now()
                let afterCPU = try cpu()
                let counts = measurement.snapshot()
                let calls = UInt64(count)
                try require(counts.total.count == calls && counts.binding.count == 3 * calls
                    && counts.receiptNames.count == calls && counts.root.count == 4 * calls
                    && counts.listing.count == 2 * calls && counts.fileOpen.count == 2 * files * calls
                    && counts.heldStat.count == 2 * files * calls && counts.namedStat.count == 2 * files * calls
                    && counts.fileCheckAndRemember.count == 2 * files * calls
                    && counts.fileClose.count == 2 * files * calls, "actual check counts differ")
                var record: [String: Any] = ["workers": workers, "clockEnabled": clock,
                    "cycle": cycle, "successfulCalls": count, "wallNanoseconds": end - start,
                    "userCPUNanoseconds": afterCPU.user - beforeCPU.user,
                    "systemCPUNanoseconds": afterCPU.system - beforeCPU.system,
                    "peakTrackedActiveLanes": peakLanes, "peakTrackedOwnedFileDescriptors": peakFiles,
                    "actualIntervals": try JSONSerialization.jsonObject(with: JSONEncoder().encode(counts))]
                if let order { record["order"] = order == 0 ? "1,2,4" : "4,2,1" }
                else { record["timingControlOnly"] = true }
                batches.append(record)
                if let order { sums[order][workers] += end - start }
            }
            // Six balanced cycles. All dispatch, result allocation and joins are timed.
            for cycle in 0..<6 {
                for order in 0..<2 {
                    for workers in order == 0 ? [1, 2, 4] : [4, 2, 1] {
                        try batch(workers: workers, clock: false, count: iterations, order: order, cycle: cycle)
                    }
                }
            }
            for (cycle, workers) in [1, 2, 4, 4, 2, 1].enumerated() {
                try batch(workers: workers, clock: true, count: 32, order: nil, cycle: cycle)
            }
            var gates: [[String: Any]] = []
            for workers in [2, 4] {
                let forward = Double(sums[0][workers]) / Double(sums[0][1])
                let reverse = Double(sums[1][workers]) / Double(sums[1][1])
                gates.append(["workers": workers, "forwardTotalValidationRatio": forward,
                    "reverseTotalValidationRatio": reverse,
                    "gainAtLeast15PercentBothOrders": forward <= 0.85 && reverse <= 0.85])
            }
            let result: [String: Any] = ["passed": true, "metadataOnly": true,
                "payloadBytesReadByAuditedPath": 0, "modelConstructed": false,
                "payloadReadEvidence": "Static call-path audit. Fixed sizeCheckTrustedReceipt policy and handle validation. No model or tensor calls. Not system IO tracing.",
                "sourceDescriptorSHA256": receipt.descriptorContentSHA256,
                "receiptFileCount": receipt.files.count, "headlineSuccessfulCalls": 4608,
                "clockOnControlCalls": 192, "warmValidationCalls": 7,
                "trustedReceiptMetadataNanoseconds": verifyEnd - verifyStart,
                "handleInitializationNanoseconds": handleEnd - handleStart,
                "initializationUserCPUNanoseconds": afterInitCPU.user - initCPU.user,
                "initializationSystemCPUNanoseconds": afterInitCPU.system - initCPU.system,
                "batches": batches, "feasibilityGates": gates,
                "zeroMeaning": "Clock-off primitive duration is unmeasured. Counts are actual.",
                "overlaps": "Worker durations are sums, not critical-path wall. heldStat/namedStat nest in fileCheckAndRemember. All phases nest in total. Batch wall includes dispatch, allocations, joins and counter merges.",
                "productionBlockers": ["Inter-entry observation schedules differ under concurrent mutation.",
                    "Dispatch workers do not inherit caller task cancellation.",
                    "Up to four file descriptors may be open instead of one. Resource-limit failure behavior differs."],
                "scope": "Standalone feasibility only. Production validation stays serial. Passing cost gate permits review, not production or model-speed qualification."]
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
