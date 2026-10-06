import Foundation
import Metal

// Candidate IDs only. No caller can use this result as a model route or admission.
enum QwenCPUExpertHint {
    static func candidates(hidden: [Float], postNorm: [Float], weights: QwenBF16Weights,
                           routerName: String, experts: Int, columns: Int) throws -> [Int] {
        guard experts == 256, columns == 2048, hidden.count == columns else {
            throw QwenBF16WeightError.invalidGeometry("CPU hint geometry")
        }
        try weights.requireShape(routerName, role: .router, rows: experts, columns: columns)
        let chunks = weights.inspectedChunks.filter { $0.name == routerName }
        guard !chunks.isEmpty else { throw QwenBF16WeightError.missingTensor(routerName) }
        var nextRow = 0
        for chunk in chunks {
            let (elements, elementsOverflow) = chunk.rowCount.multipliedReportingOverflow(by: columns)
            let (bytes, bytesOverflow) = elements.multipliedReportingOverflow(by: MemoryLayout<UInt16>.stride)
            guard chunk.firstRow == nextRow, chunk.rowCount > 0,
                  chunk.rowCount <= experts - nextRow,
                  !elementsOverflow, !bytesOverflow,
                  chunk.buffer.storageMode == .shared, chunk.buffer.length >= bytes,
                  Int(bitPattern: chunk.buffer.contents()) % MemoryLayout<UInt16>.alignment == 0 else {
                throw QwenBF16WeightError.invalidGeometry("CPU hint router chunk range")
            }
            nextRow += chunk.rowCount
        }
        guard nextRow == experts else {
            throw QwenBF16WeightError.invalidGeometry("CPU hint router row coverage")
        }
        let normalized = try qwenOfficialSourceRMSNorm(hidden, postNorm)
        // Scores are temporary hints. No FP32 router copy or model logits are made.
        var scores = [Float](repeating: 0, count: experts)
        try withExtendedLifetime((weights, chunks)) {
            try normalized.withUnsafeBufferPointer { input in
                for chunk in chunks {
                    let words = UnsafeRawPointer(chunk.buffer.contents()).assumingMemoryBound(to: UInt16.self)
                    for row in 0..<chunk.rowCount {
                        var accumulator = SIMD8<Float>(repeating: 0)
                        for column in stride(from: 0, to: columns, by: 8) {
                            let base = row * columns + column
                            let router = SIMD8<Float>(
                                Float(bitPattern: UInt32(words[base]) << 16),
                                Float(bitPattern: UInt32(words[base + 1]) << 16),
                                Float(bitPattern: UInt32(words[base + 2]) << 16),
                                Float(bitPattern: UInt32(words[base + 3]) << 16),
                                Float(bitPattern: UInt32(words[base + 4]) << 16),
                                Float(bitPattern: UInt32(words[base + 5]) << 16),
                                Float(bitPattern: UInt32(words[base + 6]) << 16),
                                Float(bitPattern: UInt32(words[base + 7]) << 16))
                            let vector = SIMD8<Float>(input[column], input[column + 1], input[column + 2],
                                input[column + 3], input[column + 4], input[column + 5],
                                input[column + 6], input[column + 7])
                            accumulator += router * vector
                        }
                        var score: Float = 0
                        for lane in 0..<8 { score += accumulator[lane] }
                        guard score.isFinite else {
                            throw QwenBF16WeightError.invalidGeometry("nonfinite CPU hint score")
                        }
                        scores[chunk.firstRow + row] = score
                    }
                }
            }
        }
        // Fixed Top8 with lower expert ID winning a tie. No per-case tuning.
        return Array(scores.indices.sorted {
            scores[$0] == scores[$1] ? $0 < $1 : scores[$0] > scores[$1]
        }.prefix(8))
    }
}
