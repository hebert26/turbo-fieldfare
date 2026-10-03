import Foundation
import Darwin
import Metal

enum QwenBaselineMemoryPhase: String, Sendable {
    case beforeModelLoad, afterModelLoad, runnerReady, prefillEmbeddingDone
    case prefillLayerDone, residencyRequestBefore, residencyRequestAfter
    case prefillDone, decodeDone, diagnosticsStart, diagnosticsEnd
}

// Immutable diagnostic owner. No callbacks, tensor reads, or shared recording lock.
struct QwenBaselineMemoryPhaseProbe: Sendable {
    let requestID: String

    func record(_ phase: QwenBaselineMemoryPhase, device: MTLDevice,
                layer: Int? = nil, cachedLayerCount: Int? = nil,
                allocatedCacheBytes: UInt64? = nil,
                residency: QwenBF16CacheResidencySnapshot? = nil) {
        func value<T>(_ optional: T?) -> Any {
            if let optional { return optional }
            return NSNull()
        }
        let record: [String: Any] = [
            "type": "memoryPhase", "schema": 1, "requestID": requestID,
            "pid": getpid(), "uptimeNanoseconds": DispatchTime.now().uptimeNanoseconds,
            "phase": phase.rawValue, "layer": value(layer),
            "currentAllocatedSizeBytes": device.currentAllocatedSize,
            "cachedLayerCount": value(cachedLayerCount),
            "allocatedCacheBytes": value(allocatedCacheBytes),
            "residencyRegisteredLayers": value(residency?.registeredLayerCount),
            "residencyAllocationCount": value(residency?.allocationCount),
            "residencyAllocatedSizeBytes": value(residency?.allocatedSize),
            "residencyRequested": value(residency?.requested)
        ]
        // Each bounded line reaches stderr without stdio buffering. Missing lines stay unknown.
        guard var data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]),
              data.count < 4095 else { return }
        data.append(10)
        data.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { return }
            var offset = 0
            while offset < bytes.count {
                let count = Darwin.write(STDERR_FILENO, base.advanced(by: offset), bytes.count - offset)
                if count < 0 && errno == EINTR { continue }
                guard count > 0 else { return }
                offset += count
            }
        }
    }
}
