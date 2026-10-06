import Foundation

/// Independent, CPU-only reference for the small multimodal decoder seam.
///
/// This deliberately does not call the production splice, projection, RMSNorm,
/// or rotary helpers.  It uses row-major matrices and direct Float32 loops so
/// the oracle can observe the prepared rows, normalized/rotated Q and K, and a
/// later decode coordinate independently of the Metal implementation.
struct QwenMultimodalDecoderReference {
    enum Error: Swift.Error, Equatable, Sendable {
        case invalidConfiguration
        case invalidCount(String, expected: Int, actual: Int)
        case invalidPosition(String)
        case duplicatePadRow(Int)
        case padRowOutOfRange(Int)
        case invalidDelta
    }

    struct Configuration: Equatable, Sendable {
        let hiddenSize: Int
        let queryHeadCount: Int
        let keyValueHeadCount: Int
        let headDimension: Int
        let rotaryDimension: Int
        let ropeTheta: Float
        let epsilon: Float
        /// Counts rotary pairs assigned to temporal, height, and width.
        let mropeSections: [Int]

        init(
            hiddenSize: Int,
            queryHeadCount: Int,
            keyValueHeadCount: Int,
            headDimension: Int,
            rotaryDimension: Int,
            ropeTheta: Float,
            epsilon: Float,
            mropeSections: [Int]
        ) throws {
            guard hiddenSize > 0,
                  queryHeadCount > 0,
                  keyValueHeadCount > 0,
                  queryHeadCount.isMultiple(of: keyValueHeadCount),
                  headDimension > 0,
                  rotaryDimension > 0,
                  rotaryDimension <= headDimension,
                  rotaryDimension.isMultiple(of: 2),
                  ropeTheta.isFinite,
                  ropeTheta > 0,
                  epsilon.isFinite,
                  epsilon > 0,
                  mropeSections.count == 3,
                  mropeSections.allSatisfy({ $0 > 0 }),
                  mropeSections.reduce(0, +) == rotaryDimension / 2 else {
                throw Error.invalidConfiguration
            }
            self.hiddenSize = hiddenSize
            self.queryHeadCount = queryHeadCount
            self.keyValueHeadCount = keyValueHeadCount
            self.headDimension = headDimension
            self.rotaryDimension = rotaryDimension
            self.ropeTheta = ropeTheta
            self.epsilon = epsilon
            self.mropeSections = mropeSections
        }

        /// The frozen P12 tiny full-attention block configuration.
        static func p12Tiny() throws -> Self {
            try Self(
                hiddenSize: 32,
                queryHeadCount: 2,
                keyValueHeadCount: 1,
                headDimension: 16,
                rotaryDimension: 12,
                ropeTheta: 10_000_000,
                epsilon: 1e-6,
                mropeSections: [2, 1, 3])
        }
    }

    struct ProjectionWeights: Sendable {
        /// Row-major `[queryHeadCount * headDimension, hiddenSize]` Q matrix.
        let query: [Float]
        /// Row-major `[keyValueHeadCount * headDimension, hiddenSize]` K matrix.
        let key: [Float]
        /// Stored residual RMSNorm weights for one head.
        let queryNorm: [Float]
        let keyNorm: [Float]
    }

    struct Input: Sendable {
        /// Row-major token embeddings, `[token, hidden]`.
        let tokenRows: [Float]
        /// Distinctive rows to replace the pad rows, `[row, hidden]`.
        var featureRows: [[Float]]
        /// Token offsets receiving `featureRows`, in prompt order.
        var padRows: [Int]
        /// Token-major `[token][temporal, height, width]` positions.
        var positions: [[Int]]
        /// The coordinate used by a later text decode before applying delta.
        /// It is normally the contiguous cache length after prefill.
        let decodeBasePosition: Int
        /// Qwen's stored text/M-RoPE delta.  The P12 case is -3.
        var decodeDelta: Int
        /// A single later token embedding, used to exercise decode after append.
        let decodeRow: [Float]
    }

    struct Output: Sendable {
        let preparedRows: [Float]
        let queryProjection: [Float]
        let keyProjection: [Float]
        let normalizedQuery: [Float]
        let normalizedKey: [Float]
        let rotatedQuery: [Float]
        let rotatedKey: [Float]
        let decodeCoordinate: Int
        let decodeQuery: [Float]
        let decodeKey: [Float]
        let cacheLengthAfterPrefill: Int
    }

    struct AttentionOutput: Sendable {
        let queryProjection: [Float]
        let keyProjection: [Float]
        let normalizedQuery: [Float]
        let normalizedKey: [Float]
        let rotatedQuery: [Float]
        let rotatedKey: [Float]
    }

    /// Evaluates only the local full-attention seam. `rows` must be the
    /// actual pre-projection input observed at that layer; no preceding-layer
    /// algorithm is replicated here.
    static func evaluateAttentionInput(
        configuration: Configuration,
        weights: ProjectionWeights,
        rows: [Float],
        positions: [[Int]]
    ) throws -> AttentionOutput {
        let hidden = configuration.hiddenSize
        guard rows.count.isMultiple(of: hidden) else {
            throw Error.invalidCount(
                "attention rows", expected: rows.count / hidden * hidden,
                actual: rows.count)
        }
        let tokenCount = rows.count / hidden
        guard positions.count == tokenCount,
              positions.allSatisfy({ $0.count == 3 }) else {
            throw Error.invalidCount(
                "attention positions", expected: tokenCount,
                actual: positions.count)
        }
        try validateWeights(weights, configuration: configuration)
        let queryProjection = project(
            rows: rows,
            matrix: weights.query,
            outputWidth: configuration.queryHeadCount * configuration.headDimension,
            hiddenSize: hidden)
        let keyProjection = project(
            rows: rows,
            matrix: weights.key,
            outputWidth: configuration.keyValueHeadCount * configuration.headDimension,
            hiddenSize: hidden)
        let normalizedQuery = normalize(
            queryProjection,
            weights: weights.queryNorm,
            tokenCount: tokenCount,
            headCount: configuration.queryHeadCount,
            configuration: configuration)
        let normalizedKey = normalize(
            keyProjection,
            weights: weights.keyNorm,
            tokenCount: tokenCount,
            headCount: configuration.keyValueHeadCount,
            configuration: configuration)
        return AttentionOutput(
            queryProjection: queryProjection,
            keyProjection: keyProjection,
            normalizedQuery: normalizedQuery,
            normalizedKey: normalizedKey,
            rotatedQuery: rotate(
                normalizedQuery,
                positions: positions,
                headCount: configuration.queryHeadCount,
                configuration: configuration),
            rotatedKey: rotate(
                normalizedKey,
                positions: positions,
                headCount: configuration.keyValueHeadCount,
                configuration: configuration))
    }

    static func evaluate(
        configuration: Configuration,
        weights: ProjectionWeights,
        input: Input
    ) throws -> Output {
        let hidden = configuration.hiddenSize
        guard input.tokenRows.count.isMultiple(of: hidden) else {
            throw Error.invalidCount(
                "tokenRows", expected: input.tokenRows.count / hidden * hidden,
                actual: input.tokenRows.count)
        }
        let tokenCount = input.tokenRows.count / hidden
        guard input.positions.count == tokenCount else {
            throw Error.invalidCount(
                "positions", expected: tokenCount, actual: input.positions.count)
        }
        guard input.featureRows.count == input.padRows.count else {
            throw Error.invalidCount(
                "featureRows/padRows", expected: input.padRows.count,
                actual: input.featureRows.count)
        }
        guard input.decodeBasePosition >= 0 else {
            throw Error.invalidPosition("decodeBasePosition")
        }
        guard input.decodeRow.count == hidden else {
            throw Error.invalidCount(
                "decodeRow", expected: hidden, actual: input.decodeRow.count)
        }
        for position in input.positions where position.count != 3 {
            throw Error.invalidCount("M-RoPE coordinate", expected: 3, actual: position.count)
        }
        for (row, pad) in zip(input.featureRows, input.padRows) {
            guard row.count == hidden else {
                throw Error.invalidCount(
                    "featureRow", expected: hidden, actual: row.count)
            }
            guard (0..<tokenCount).contains(pad) else {
                throw Error.padRowOutOfRange(pad)
            }
        }
        var seenPads = Set<Int>()
        for pad in input.padRows where !seenPads.insert(pad).inserted {
            throw Error.duplicatePadRow(pad)
        }
        try validateWeights(weights, configuration: configuration)

        let preparedRows = splice(
            tokenRows: input.tokenRows,
            featureRows: input.featureRows,
            padRows: input.padRows,
            hiddenSize: hidden)
        let queryProjection = project(
            rows: preparedRows,
            matrix: weights.query,
            outputWidth: configuration.queryHeadCount * configuration.headDimension,
            hiddenSize: hidden)
        let keyProjection = project(
            rows: preparedRows,
            matrix: weights.key,
            outputWidth: configuration.keyValueHeadCount * configuration.headDimension,
            hiddenSize: hidden)
        let normalizedQuery = normalize(
            queryProjection,
            weights: weights.queryNorm,
            tokenCount: tokenCount,
            headCount: configuration.queryHeadCount,
            configuration: configuration)
        let normalizedKey = normalize(
            keyProjection,
            weights: weights.keyNorm,
            tokenCount: tokenCount,
            headCount: configuration.keyValueHeadCount,
            configuration: configuration)
        let rotatedQuery = rotate(
            normalizedQuery,
            positions: input.positions,
            headCount: configuration.queryHeadCount,
            configuration: configuration)
        let rotatedKey = rotate(
            normalizedKey,
            positions: input.positions,
            headCount: configuration.keyValueHeadCount,
            configuration: configuration)

        let decodeCoordinate = input.decodeBasePosition + input.decodeDelta
        guard decodeCoordinate >= 0 else { throw Error.invalidDelta }
        let decodeQueryProjection = project(
            rows: input.decodeRow,
            matrix: weights.query,
            outputWidth: configuration.queryHeadCount * configuration.headDimension,
            hiddenSize: hidden)
        let decodeKeyProjection = project(
            rows: input.decodeRow,
            matrix: weights.key,
            outputWidth: configuration.keyValueHeadCount * configuration.headDimension,
            hiddenSize: hidden)
        let decodeQuery = rotate(
            normalize(
                decodeQueryProjection,
                weights: weights.queryNorm,
                tokenCount: 1,
                headCount: configuration.queryHeadCount,
                configuration: configuration),
            positions: [[decodeCoordinate, decodeCoordinate, decodeCoordinate]],
            headCount: configuration.queryHeadCount,
            configuration: configuration)
        let decodeKey = rotate(
            normalize(
                decodeKeyProjection,
                weights: weights.keyNorm,
                tokenCount: 1,
                headCount: configuration.keyValueHeadCount,
                configuration: configuration),
            positions: [[decodeCoordinate, decodeCoordinate, decodeCoordinate]],
            headCount: configuration.keyValueHeadCount,
            configuration: configuration)
        return Output(
            preparedRows: preparedRows,
            queryProjection: queryProjection,
            keyProjection: keyProjection,
            normalizedQuery: normalizedQuery,
            normalizedKey: normalizedKey,
            rotatedQuery: rotatedQuery,
            rotatedKey: rotatedKey,
            decodeCoordinate: decodeCoordinate,
            decodeQuery: decodeQuery,
            decodeKey: decodeKey,
            cacheLengthAfterPrefill: tokenCount)
    }

    private static func validateWeights(
        _ weights: ProjectionWeights,
        configuration: Configuration
    ) throws {
        let queryWidth = configuration.queryHeadCount * configuration.headDimension
        let keyWidth = configuration.keyValueHeadCount * configuration.headDimension
        guard weights.query.count == queryWidth * configuration.hiddenSize else {
            throw Error.invalidCount(
                "query weights", expected: queryWidth * configuration.hiddenSize,
                actual: weights.query.count)
        }
        guard weights.key.count == keyWidth * configuration.hiddenSize else {
            throw Error.invalidCount(
                "key weights", expected: keyWidth * configuration.hiddenSize,
                actual: weights.key.count)
        }
        guard weights.queryNorm.count == configuration.headDimension else {
            throw Error.invalidCount(
                "query norm", expected: configuration.headDimension,
                actual: weights.queryNorm.count)
        }
        guard weights.keyNorm.count == configuration.headDimension else {
            throw Error.invalidCount(
                "key norm", expected: configuration.headDimension,
                actual: weights.keyNorm.count)
        }
    }

    /// Replaces rows by copying values into a fresh result.  Production
    /// ownership and buffer semantics are intentionally not used here.
    private static func splice(
        tokenRows: [Float],
        featureRows: [[Float]],
        padRows: [Int],
        hiddenSize: Int
    ) -> [Float] {
        var result = tokenRows
        for (feature, row) in zip(featureRows, padRows) {
            let base = row * hiddenSize
            for column in 0..<hiddenSize {
                result[base + column] = feature[column]
            }
        }
        return result
    }

    private static func project(
        rows: [Float],
        matrix: [Float],
        outputWidth: Int,
        hiddenSize: Int
    ) -> [Float] {
        let tokenCount = rows.count / hiddenSize
        var result = [Float](repeating: 0, count: tokenCount * outputWidth)
        for token in 0..<tokenCount {
            let inputBase = token * hiddenSize
            let outputBase = token * outputWidth
            for output in 0..<outputWidth {
                let matrixBase = output * hiddenSize
                var sum: Float = 0
                for column in 0..<hiddenSize {
                    sum += matrix[matrixBase + column] * rows[inputBase + column]
                }
                result[outputBase + output] = sum
            }
        }
        return result
    }

    private static func normalize(
        _ input: [Float],
        weights: [Float],
        tokenCount: Int,
        headCount: Int,
        configuration: Configuration
    ) -> [Float] {
        let dimension = configuration.headDimension
        var result = [Float](repeating: 0, count: input.count)
        for item in 0..<(tokenCount * headCount) {
            let base = item * dimension
            var squareSum: Float = 0
            for index in 0..<dimension {
                squareSum += input[base + index] * input[base + index]
            }
            let inverseRMS = 1 / (squareSum / Float(dimension)
                + configuration.epsilon).squareRoot()
            for index in 0..<dimension {
                // Qwen's tiny fixture stores residual RMSNorm weights: the
                // effective multiplier is one plus the stored value.
                result[base + index] = input[base + index]
                    * inverseRMS * (1 + weights[index])
            }
        }
        return result
    }

    private static func rotate(
        _ input: [Float],
        positions: [[Int]],
        headCount: Int,
        configuration: Configuration
    ) -> [Float] {
        let dimension = configuration.headDimension
        let half = configuration.rotaryDimension / 2
        let tokenCount = positions.count
        var result = input
        for token in 0..<tokenCount {
            let coordinate = positions[token]
            for head in 0..<headCount {
                let base = (token * headCount + head) * dimension
                var pair = 0
                for axis in 0..<3 {
                    for _ in 0..<configuration.mropeSections[axis] {
                        let exponent = Float(2 * pair)
                            / Float(configuration.rotaryDimension)
                        let inverseFrequency = Float(
                            Foundation.pow(
                                Double(configuration.ropeTheta), Double(-exponent)))
                        let angle = Float(coordinate[axis]) * inverseFrequency
                        let cosine = Float(Foundation.cos(Double(angle)))
                        let sine = Float(Foundation.sin(Double(angle)))
                        let first = input[base + pair]
                        let second = input[base + pair + half]
                        result[base + pair] = first * cosine - second * sine
                        result[base + pair + half] = second * cosine + first * sine
                        pair += 1
                    }
                }
            }
        }
        return result
    }
}
