import CoreGraphics
import Foundation
import ImageIO
import Metal
import Testing
import UniformTypeIdentifiers
@testable import TurboFieldfare

@Suite struct QwenImagePreprocessorTests {
    @Test func officialContractUsesQwenTemporalPatchWidthAndRawTensorCount() throws {
        let config = QwenVisionConfig.official
        try config.validate()
        #expect(config.patchWidth == 3 * 2 * 16 * 16)
        #expect(config.patchWidth == 1_536)
        #expect(config.mergedHiddenSize == 4_608)
        #expect(config.headDimension == 72)
        #expect(config.tensorContract.count == 333)
        #expect(config.rawTensorElementCount == 446_571_248)
    }

    @Test func geometryAtMaximumRowsUsesTemporalOneAndQwenMerge() throws {
        let geometry = try QwenImageGeometry(
            sourceWidth: 960, sourceHeight: 672)
        #expect(geometry.processedWidth == 960)
        #expect(geometry.processedHeight == 672)
        #expect(geometry.gridT == 1)
        #expect(geometry.gridH == 42)
        #expect(geometry.gridW == 60)
        #expect(geometry.patchRows == 2_520)
        #expect(geometry.mergedRows == 630)
    }

    @Test func minimumProcessedPixelsAreRaisedToAValidQwenGrid() throws {
        let geometry = try QwenImageGeometry(
            sourceWidth: 256, sourceHeight: 256)
        #expect(geometry.processedWidth == 256)
        #expect(geometry.processedHeight == 256)
        #expect(geometry.gridT == 1)
        #expect(geometry.gridH == 16)
        #expect(geometry.gridW == 16)
        #expect(geometry.patchRows == 256)
        #expect(geometry.mergedRows == 64)
        #expect(geometry.processedWidth * geometry.processedHeight == 65_536)
    }

    @Test func aggregatePreflightAccepts630AndRejects631BeforeVisionAllocation() throws {
        let exact = try QwenImageGeometry(sourceWidth: 672, sourceHeight: 960)
        try QwenImagePreprocessor.preflight([exact])

        let first = try QwenImageGeometry(sourceWidth: 672, sourceHeight: 864)
        let second = try QwenImageGeometry(sourceWidth: 256, sourceHeight: 256)
        #expect(first.mergedRows == 567)
        #expect(second.mergedRows == 64)
        #expect(throws: QwenVisionError.mergedRowQuotaExceeded(
            requested: 631, maximum: 630)) {
            try QwenImagePreprocessor.preflight([first, second])
        }
    }

    @Test func rejectsVideoAndInvalidDimensionsBeforePlanning() throws {
        #expect(throws: QwenVisionError.unsupportedVideo) {
            try QwenImageGeometry(
                sourceWidth: 672, sourceHeight: 960, media: .video)
        }
        #expect(throws: QwenVisionError.invalidDimensions(width: 0, height: 10)) {
            try QwenImageGeometry(sourceWidth: 0, sourceHeight: 10)
        }
    }

    @Test func stillPreprocessDuplicatesEachPatchAcrossTemporalAxisAndKeepsBlockOrder() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let url = try makeSolidImage(width: 256, height: 256)
        defer {
            try? FileManager.default.removeItem(at: url.deletingLastPathComponent())
        }
        let preprocessor = QwenImagePreprocessor(device: device)
        let plan = try preprocessor.plan(fileURL: url)
        let result = try preprocessor.preprocess(plan)
        let config = QwenVisionConfig.official
        let geometry = result.geometry
        #expect(geometry.patchRows == 256)
        #expect(result.patchesBF16.length
            == geometry.patchRows * config.patchWidth * MemoryLayout<UInt16>.stride)
        #expect(result.positionsInt32x2.length
            == geometry.patchRows * 2 * MemoryLayout<Int32>.stride)
        #expect(result.allocatedBytes >= result.patchesBF16.length
            + result.positionsInt32x2.length)

        let positions = result.positionsInt32x2.contents().bindMemory(
            to: Int32.self, capacity: geometry.patchRows * 2)
        var row = 0
        for blockY in 0..<(geometry.gridH / config.spatialMergeSize) {
            for blockX in 0..<(geometry.gridW / config.spatialMergeSize) {
                for innerY in 0..<config.spatialMergeSize {
                    for innerX in 0..<config.spatialMergeSize {
                        // Qwen vision IDs are ordered [height, width].
                        #expect(positions[row * 2]
                            == Int32(blockY * config.spatialMergeSize + innerY))
                        #expect(positions[row * 2 + 1]
                            == Int32(blockX * config.spatialMergeSize + innerX))
                        row += 1
                    }
                }
            }
        }
        #expect(row == geometry.patchRows)

        let patches = result.patchesBF16.contents().bindMemory(
            to: UInt16.self,
            capacity: geometry.patchRows * config.patchWidth)
        let plane = config.patchSize * config.patchSize
        // C,T,H,W has one complete spatial plane for each temporal index. A
        // still image supplies the same normalized BF16 sample in both planes.
        for patch in 0..<geometry.patchRows {
            for channel in 0..<config.inputChannels {
                let base = patch * config.patchWidth
                    + channel * config.temporalPatchSize * plane
                for element in 0..<plane {
                    #expect(patches[base + element]
                        == patches[base + plane + element])
                }
            }
        }
    }

    private func makeSolidImage(width: Int, height: Int) throws -> URL {
        let phase16 = URL(fileURLWithPath: FileManager.default.currentDirectoryPath,
                          isDirectory: true)
            .appendingPathComponent(
                "scratch/qwen3.6-35b-a3b/evidence/phase-16", isDirectory: true)
        let root = phase16.appendingPathComponent(UUID().uuidString, isDirectory: true)
        var keepRoot = false
        defer {
            if !keepRoot { try? FileManager.default.removeItem(at: root) }
        }
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        let url = root.appendingPathComponent("input.png")
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let destination = CGImageDestinationCreateWithURL(
                url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw QwenVisionError.allocationFailed(name: "test image")
        }
        context.setFillColor(CGColor(red: 0.25, green: 0.5, blue: 0.75, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let filled = context.makeImage() else {
            throw QwenVisionError.allocationFailed(name: "test image")
        }
        CGImageDestinationAddImage(destination, filled, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw QwenVisionError.allocationFailed(name: "test image")
        }
        keepRoot = true
        return url
    }
}
