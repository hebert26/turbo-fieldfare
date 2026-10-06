import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// Differential source-math coverage for the grouped KV-head packing path.
///
/// Each grouped result is compared bit-for-bit with an independently
/// dispatched one-query-head source attention call.  The two KV heads are
/// assembled from different pinned Torch rows, while both groups contain two
/// query heads.  The common 76-token committed prefix plus one candidate row
/// keeps the grouped call valid and selects source SGEMM (77 * 256 elements).
@Suite(.serialized) struct QwenBF16GroupedKeyPackingReuseTests {
    @Test func officialSourceAttentionGroupedHeadsMatchPinnedLongContextTorchHeads() throws {
        let fixture = try loadGroupedFixture()
        let first = try #require(fixture.cases.first)
        let second = try #require(fixture.cases.dropFirst().first)
        let dimension = fixture.geometry.headDimension
        let totalKeyCount = first.keyCount
        let committedCount = totalKeyCount - 1
        let committedElements = try #require(committedCount > 0
            ? committedCount * dimension : nil)
        #expect(fixture.kind == "qwen-source-attention-long-context-torch-subset-v1")
        #expect(fixture.geometry.queryHeads == 16)
        #expect(fixture.geometry.keyValueHeads == 2)
        #expect(first.keyCount == 77)
        #expect(second.keyCount >= totalKeyCount)
        #expect(totalKeyCount * dimension >= 400,
                "the grouped call must take the source SGEMM path")
        try #require(first.cachedKeyBits.count >= committedElements,
                     "first pinned cache must contain all committed rows")
        try #require(second.cachedKeyBits.count >= committedElements,
                     "second pinned cache must contain all committed rows")
        try #require(first.cachedValueBits.count >= committedElements,
                     "first pinned values must contain all committed rows")
        try #require(second.cachedValueBits.count >= committedElements,
                     "second pinned values must contain all committed rows")
        try #require(first.currentKeyBits.count == dimension,
                     "first pinned candidate key width")
        try #require(second.currentKeyBits.count == dimension,
                     "second pinned candidate key width")
        try #require(first.currentValueBits.count == dimension,
                     "first pinned candidate value width")
        try #require(second.currentValueBits.count == dimension,
                     "second pinned candidate value width")

        let firstKeys = Array(first.cachedKeyBits.prefix(committedElements))
        let secondKeys = Array(second.cachedKeyBits.prefix(committedElements))
        let firstValues = Array(first.cachedValueBits.prefix(committedElements))
        let secondValues = Array(second.cachedValueBits.prefix(committedElements))
        #expect(firstKeys != secondKeys,
                "each KV head must have distinct committed keys")
        #expect(firstValues != secondValues,
                "each KV head must have distinct committed values")
        #expect(first.queryBits != second.queryBits,
                "each KV group must have distinct pinned query data")
        #expect(first.currentKeyBits != second.currentKeyBits,
                "each KV head must have a distinct candidate key")

        // Interleave the two pinned per-head fixtures in the production's
        // token-major KV layout.  The query list gives each KV group two
        // queries and makes the group transition observable.
        let queryBits = [first.queryBits, second.queryBits, second.queryBits, first.queryBits]
        let groupedQuery = queryBits.flatMap { $0.map { Float(bitPattern: $0) } }
        var groupedKeys: [Float] = []
        var groupedValues: [Float] = []
        groupedKeys.reserveCapacity(committedCount * 2 * dimension)
        groupedValues.reserveCapacity(committedCount * 2 * dimension)
        for token in 0..<committedCount {
            let row = token * dimension
            groupedKeys.append(contentsOf: firstKeys[row..<(row + dimension)].map { Float(bitPattern: $0) })
            groupedKeys.append(contentsOf: secondKeys[row..<(row + dimension)].map { Float(bitPattern: $0) })
            groupedValues.append(contentsOf: firstValues[row..<(row + dimension)].map { Float(bitPattern: $0) })
            groupedValues.append(contentsOf: secondValues[row..<(row + dimension)].map { Float(bitPattern: $0) })
        }
        let groupedCandidateKeys = [first.currentKeyBits, second.currentKeyBits]
            .flatMap { $0.map { Float(bitPattern: $0) } }
        let groupedCandidateValues = [first.currentValueBits, second.currentValueBits]
            .flatMap { $0.map { Float(bitPattern: $0) } }

        let groupedConfiguration = try QwenFullAttentionConfiguration(
            queryHeadCount: 4,
            keyValueHeadCount: 2,
            headDimension: dimension,
            rotaryDimension: fixture.geometry.rotaryDimension,
            theta: Float(fixture.geometry.theta))
        let context = try MetalContext()
        let groupedAttention = try QwenFullAttention(
            context: context, configuration: groupedConfiguration,
            useOfficialSourceMath: true)
        let groupedCache = QwenFullAttentionKVView(
            key: try makeGroupedFloatBuffer(groupedKeys, device: context.device),
            value: try makeGroupedFloatBuffer(groupedValues, device: context.device),
            keyOffset: 0,
            valueOffset: 0,
            strideBytes: 2 * dimension * MemoryLayout<Float>.stride,
            validTokenCount: committedCount)
        let groupedOutput = try groupedAttention.attentionStep(
            rotatedQuery: groupedQuery,
            rotatedKey: groupedCandidateKeys,
            value: groupedCandidateValues,
            cache: groupedCache)

        let oneHeadConfiguration = try QwenFullAttentionConfiguration(
            queryHeadCount: 1,
            keyValueHeadCount: 1,
            headDimension: dimension,
            rotaryDimension: fixture.geometry.rotaryDimension,
            theta: Float(fixture.geometry.theta))
        let expectedByHead: [[UInt32]] = try (0..<4).map { head in
            let kvHead = head / 2
            let keys = kvHead == 0 ? firstKeys : secondKeys
            let values = kvHead == 0 ? firstValues : secondValues
            let query = queryBits[head].map { Float(bitPattern: $0) }
            let candidateKeyBits = kvHead == 0 ? first.currentKeyBits : second.currentKeyBits
            let candidateValueBits = kvHead == 0 ? first.currentValueBits : second.currentValueBits
            let singleAttention = try QwenFullAttention(
                context: context, configuration: oneHeadConfiguration,
                useOfficialSourceMath: true)
            let singleCache = QwenFullAttentionKVView(
                key: try makeGroupedFloatBuffer(keys.map { Float(bitPattern: $0) }, device: context.device),
                value: try makeGroupedFloatBuffer(values.map { Float(bitPattern: $0) }, device: context.device),
                keyOffset: 0,
                valueOffset: 0,
                strideBytes: dimension * MemoryLayout<Float>.stride,
                validTokenCount: committedCount)
            let output = try singleAttention.attentionStep(
                rotatedQuery: query,
                rotatedKey: candidateKeyBits.map { Float(bitPattern: $0) },
                value: candidateValueBits.map { Float(bitPattern: $0) },
                cache: singleCache)
            return output.map(\.bitPattern)
        }

        for head in 0..<4 {
            let actual = Array(groupedOutput[(head * dimension)..<((head + 1) * dimension)])
                .map(\.bitPattern)
            #expect(actual == expectedByHead[head],
                    "grouped source output head \(head) must equal its no-reuse source reference")
        }

        // Head zero is a direct pinned Torch case.  This anchors the
        // differential reference to the existing official fixture as well as
        // checking the assembled grouped call.
        #expect(
            Array(groupedOutput[0..<dimension]).map(\.bitPattern) == first.expectedCoreBits,
            "grouped head 0 must retain the pinned Torch output")
    }
}

private struct GroupedLongContextFixture: Decodable {
    let kind: String
    let geometry: Geometry
    let cases: [Case]

    struct Geometry: Decodable {
        let headDimension: Int
        let rotaryDimension: Int
        let theta: Double
        let queryHeads: Int
        let keyValueHeads: Int
    }

    struct Case: Decodable {
        let keyCount: Int
        let queryBits: [UInt32]
        let cachedKeyBits: [UInt32]
        let cachedValueBits: [UInt32]
        let currentKeyBits: [UInt32]
        let currentValueBits: [UInt32]
        let expectedCoreBits: [UInt32]
    }
}

private func loadGroupedFixture() throws -> GroupedLongContextFixture {
    let url = try #require(Bundle.module.url(
        forResource: "long-context-attention-subset",
        withExtension: "json",
        subdirectory: "full-attention-source"))
    return try JSONDecoder().decode(
        GroupedLongContextFixture.self,
        from: Data(contentsOf: url))
}

private func makeGroupedFloatBuffer(_ values: [Float], device: MTLDevice) throws -> MTLBuffer {
    try #require(device.makeBuffer(
        bytes: values,
        length: values.count * MemoryLayout<Float>.stride,
        options: .storageModeShared))
}
