import Testing
@testable import TurboFieldfare

@Suite struct QwenMultimodalPositionTests {
    @Test func tinyFixtureUsesBlockMajorImagePositionsAndNonzeroDelta() throws {
        let grid = try QwenVisionGrid(temporal: 1, height: 4, width: 6)
        #expect(try grid.mergedRows(merge: 2) == 6)
        let plan = try QwenMultimodalPositions.make(
            tokenCount: 10,
            imageRanges: [2..<8],
            grids: [grid],
            merge: 2,
            maximumRows: 6)

        #expect(plan.imageRanges == [2..<8])
        #expect(plan.grids == [grid])
        #expect(plan.positions.count == 10)
        #expect(plan.positions[0].values == [0, 0, 0])
        #expect(plan.positions[1].values == [1, 1, 1])
        #expect(plan.positions[2].values == [2, 2, 2])
        #expect(plan.positions[3].values == [2, 2, 3])
        #expect(plan.positions[4].values == [2, 2, 4])
        #expect(plan.positions[5].values == [2, 3, 2])
        #expect(plan.positions[6].values == [2, 3, 3])
        #expect(plan.positions[7].values == [2, 3, 4])
        #expect(plan.positions[8].values == [5, 5, 5])
        #expect(plan.positions[9].values == [6, 6, 6])
        #expect(plan.textRoPEDelta == -3)
    }

    @Test func axisSectionsSelectDistinctTinyAndOfficialOrders() throws {
        let tiny = [2, 1, 3]
        #expect(try QwenMultimodalPositions.axis(frequencyIndex: 0, sections: tiny) == 0)
        #expect(try QwenMultimodalPositions.axis(frequencyIndex: 1, sections: tiny) == 0)
        #expect(try QwenMultimodalPositions.axis(frequencyIndex: 2, sections: tiny) == 1)
        #expect(try QwenMultimodalPositions.axis(frequencyIndex: 3, sections: tiny) == 2)
        #expect(try QwenMultimodalPositions.axis(frequencyIndex: 5, sections: tiny) == 2)

        let official = [11, 11, 10]
        #expect(try QwenMultimodalPositions.axis(frequencyIndex: 0, sections: official) == 0)
        #expect(try QwenMultimodalPositions.axis(frequencyIndex: 10, sections: official) == 0)
        #expect(try QwenMultimodalPositions.axis(frequencyIndex: 11, sections: official) == 1)
        #expect(try QwenMultimodalPositions.axis(frequencyIndex: 21, sections: official) == 1)
        #expect(try QwenMultimodalPositions.axis(frequencyIndex: 22, sections: official) == 2)
        #expect(try QwenMultimodalPositions.axis(frequencyIndex: 31, sections: official) == 2)
    }

    @Test func multipleImagesRemainInPromptOrderWithIndependentGridSpans() throws {
        let first = try QwenVisionGrid(temporal: 1, height: 2, width: 4)
        let second = try QwenVisionGrid(temporal: 1, height: 4, width: 4)
        let plan = try QwenMultimodalPositions.make(
            tokenCount: 9,
            imageRanges: [1..<3, 5..<9],
            grids: [first, second],
            merge: 2,
            maximumRows: 6)

        #expect(plan.positions[1].values == [1, 1, 1])
        #expect(plan.positions[2].values == [1, 1, 2])
        #expect(plan.positions[5].values == [5, 5, 5])
        #expect(plan.positions[6].values == [5, 5, 6])
        #expect(plan.positions[7].values == [5, 6, 5])
        #expect(plan.positions[8].values == [5, 6, 6])
        #expect(plan.imageRanges == [1..<3, 5..<9])
    }

    @Test func rejectsOverlappingOrMismatchedImageSpansAndRowQuota() throws {
        let grid = try QwenVisionGrid(temporal: 1, height: 4, width: 6)
        #expect(throws: QwenVisionError.invalidPositions) {
            try QwenMultimodalPositions.make(
                tokenCount: 8,
                imageRanges: [1..<8],
                grids: [try QwenVisionGrid(temporal: 1, height: 2, width: 2)],
                merge: 2)
        }
        #expect(throws: QwenVisionError.invalidPositions) {
            try QwenMultimodalPositions.make(
                tokenCount: 10,
                imageRanges: [1..<7, 6..<9],
                grids: [grid, grid],
                merge: 2)
        }
        #expect(throws: QwenVisionError.mergedRowQuotaExceeded(
            requested: 7, maximum: 6)) {
            try QwenMultimodalPositions.make(
                tokenCount: 11,
                imageRanges: [1..<7, 8..<9],
                grids: [grid, try QwenVisionGrid(temporal: 1, height: 2, width: 2)],
                merge: 2,
                maximumRows: 6)
        }
    }

    @Test func rejectsInvalidAxesAndCoordinates() throws {
        #expect(throws: QwenVisionError.invalidPositions) {
            try QwenMultimodalPositions.axis(frequencyIndex: -1, sections: [2, 1, 3])
        }
        #expect(throws: QwenVisionError.invalidPositions) {
            try QwenMultimodalPositions.axis(frequencyIndex: 6, sections: [2, 1, 3])
        }
        #expect(throws: QwenVisionError.invalidPositions) {
            try QwenMultimodalPositions.axis(frequencyIndex: 0, sections: [2, 0, 3])
        }
        #expect(throws: QwenVisionError.invalidPositions) {
            try QwenMRoPEPosition(temporal: -1, height: 0, width: 0)
        }
        #expect(throws: QwenVisionError.invalidPositions) {
            try QwenMRoPEPosition(
                temporal: Int(Int32.max) + 1, height: 0, width: 0)
        }
        #expect(throws: QwenVisionError.invalidPositions) {
            try QwenVisionGrid(temporal: 1, height: 0, width: 2)
        }
    }
}
