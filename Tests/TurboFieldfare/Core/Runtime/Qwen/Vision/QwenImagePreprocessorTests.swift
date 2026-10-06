import CoreGraphics
import CryptoKit
import Foundation
import ImageIO
import Metal
import Testing
import UniformTypeIdentifiers
@testable import TurboFieldfare

@Suite struct QwenImagePreprocessorTests {
    private struct JPEGDecodeManifest: Decodable {
        struct Case: Decodable {
            let id: String
            let imageFile: String
            let imageSHA256: String
            let encodedBytes: Int
            let encodedWidth: Int
            let encodedHeight: Int
            let orientation: Int
            let orientedWidth: Int
            let orientedHeight: Int
            let rgbFile: String
            let rgbByteCount: Int
            let rgbSHA256: String
            let patchSHA256: String
        }

        let cases: [Case]
    }

    private struct TinyJPEGDecodeManifest: Decodable {
        struct Case: Decodable {
            let id: String
            let file: String
            let inputBytes: Int
            let inputSHA256: String
            let encodedWidth: Int
            let encodedHeight: Int
            let decodedMode: String
            let orientation: Int
            let orientedWidth: Int
            let orientedHeight: Int
            let rgbFile: String
            let rgbByteCount: Int
            let rgbSHA256: String
        }

        struct RejectedCase: Decodable {
            let id: String
            let file: String
            let inputBytes: Int
            let inputSHA256: String
            let expectedAdmission: String
        }

        struct Generator: Decodable {
            let pillow: String
        }

        let kind: String
        let oracle: String
        let generator: Generator
        let cases: [Case]
        let rejectedCases: [RejectedCase]
    }

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

    @Test func pinnedNaturalJPEGsMatchPillowRGBAndFinalQwenPatches() throws {
        let images = try #require(Bundle.module.url(forResource: "images", withExtension: nil))
        let references = try #require(Bundle.module.url(
            forResource: "vision-jpeg-decode", withExtension: nil))
        let manifest = try JSONDecoder().decode(
            JPEGDecodeManifest.self,
            from: Data(contentsOf: references.appendingPathComponent("manifest.json")))
        #expect(manifest.cases.map(\.id) == ["natural-exif-1", "natural-exif-6"])

        let device = try #require(MTLCreateSystemDefaultDevice())
        let preprocessor = QwenImagePreprocessor(device: device)
        for fixture in manifest.cases {
            let imageURL = images.appendingPathComponent(fixture.imageFile)
            let encoded = try Data(contentsOf: imageURL)
            #expect(encoded.count == fixture.encodedBytes)
            #expect(qwenImageSHA256(encoded) == fixture.imageSHA256)

            let source = try VisionImageSource(fileURL: imageURL)
            let opened = try source.open(maximumEncodedBytes: VisionImageLimits().maximumEncodedBytes)
            let metadata = try ImageMetadataReader().read(opened: opened)
            #expect(metadata.encodedWidth == fixture.encodedWidth)
            #expect(metadata.encodedHeight == fixture.encodedHeight)
            #expect(metadata.orientation == fixture.orientation)
            #expect(metadata.orientedWidth == fixture.orientedWidth)
            #expect(metadata.orientedHeight == fixture.orientedHeight)

            let decoded = try QwenJPEGDecoder.decode(encoded: encoded, metadata: metadata)
            #expect(decoded.width == fixture.orientedWidth)
            #expect(decoded.height == fixture.orientedHeight)
            let expectedRGB = try Data(contentsOf: references.appendingPathComponent(fixture.rgbFile))
            #expect(expectedRGB.count == fixture.rgbByteCount)
            #expect(qwenImageSHA256(expectedRGB) == fixture.rgbSHA256)
            let pixelCount = fixture.orientedWidth * fixture.orientedHeight
            #expect(expectedRGB.count == pixelCount * 3)
            #expect(decoded.rgba.count == pixelCount * 4)
            guard expectedRGB.count == pixelCount * 3,
                  decoded.rgba.count == pixelCount * 4 else { continue }

            var actualRGB = Data(count: expectedRGB.count)
            var alphaMatches = true
            actualRGB.withUnsafeMutableBytes { actualBytes in
                decoded.rgba.withUnsafeBytes { rgbaBytes in
                    let actual = actualBytes.bindMemory(to: UInt8.self)
                    let rgba = rgbaBytes.bindMemory(to: UInt8.self)
                    for pixel in 0..<pixelCount {
                        let rgbOffset = pixel * 3
                        let rgbaOffset = pixel * 4
                        actual[rgbOffset] = rgba[rgbaOffset]
                        actual[rgbOffset + 1] = rgba[rgbaOffset + 1]
                        actual[rgbOffset + 2] = rgba[rgbaOffset + 2]
                        alphaMatches = alphaMatches && rgba[rgbaOffset + 3] == 255
                    }
                }
            }
            #expect(actualRGB == expectedRGB, "\(fixture.id) decoded RGB differs from pinned Pillow")
            #expect(alphaMatches, "\(fixture.id) decoder output alpha must be opaque")

            // This crosses the real no-weights preprocessing path so decoder,
            // resize, and Qwen patch order are pinned together.
            let plan = try preprocessor.plan(fileURL: imageURL)
            let patches = try preprocessor.preprocess(plan)
            #expect(patches.geometry.processedWidth == 352)
            #expect(patches.geometry.processedHeight == 224)
            let patchBytes = patches.geometry.patchRows * QwenVisionConfig.official.patchWidth * 2
            #expect(patches.patchesBF16.length == patchBytes,
                    "\(fixture.id) patch buffer must contain the complete Qwen tensor")
            guard patches.patchesBF16.length == patchBytes else { continue }
            let patchData = Data(bytes: patches.patchesBF16.contents(), count: patchBytes)
            #expect(qwenImageSHA256(patchData) == fixture.patchSHA256,
                    "\(fixture.id) final BF16 patches differ from pinned official preprocessing")
        }
    }

    @Test func jpegDecoderRejectsMalformedTruncatedAndOversizedInputs() throws {
        let images = try #require(Bundle.module.url(forResource: "images", withExtension: nil))
        let imageURL = images.appendingPathComponent("natural-exif-1.jpg")
        let encoded = try Data(contentsOf: imageURL)
        let opened = try VisionImageSource(fileURL: imageURL)
            .open(maximumEncodedBytes: VisionImageLimits().maximumEncodedBytes)
        let metadata = try ImageMetadataReader().read(opened: opened)

        let malformed = Data(repeating: 0, count: encoded.count)
        #expect(throws: VisionImageError.self) {
            try QwenJPEGDecoder.decode(encoded: malformed, metadata: metadata)
        }

        let truncated = Data(encoded.dropLast(2))
        let truncatedMetadata = VisionImageMetadata(
            encodedBytes: truncated.count,
            encodedWidth: metadata.encodedWidth,
            encodedHeight: metadata.encodedHeight,
            orientedWidth: metadata.orientedWidth,
            orientedHeight: metadata.orientedHeight,
            orientation: metadata.orientation,
            bitsPerComponent: metadata.bitsPerComponent,
            colorModel: metadata.colorModel,
            typeIdentifier: metadata.typeIdentifier)
        #expect(throws: VisionImageError.self) {
            try QwenJPEGDecoder.decode(encoded: truncated, metadata: truncatedMetadata)
        }

        #expect(throws: VisionImageError.self) {
            try QwenJPEGDecoder.decode(
                encoded: encoded, metadata: metadata,
                limits: VisionImageLimits(maximumEncodedBytes: encoded.count - 1))
        }
        #expect(throws: VisionImageError.self) {
            try QwenJPEGDecoder.decode(
                encoded: encoded, metadata: metadata,
                limits: VisionImageLimits(maximumSourceDimension: 320))
        }
        #expect(throws: VisionImageError.self) {
            try QwenJPEGDecoder.decode(
                encoded: encoded, metadata: metadata,
                limits: VisionImageLimits(maximumDecodedBytes: 100))
        }
    }

    @Test func pinnedTinyJPEGFixturesCoverAllExifAndColorModes() throws {
        let references = try #require(Bundle.module.url(
            forResource: "vision-jpeg-decode", withExtension: nil))
        let fixtureRoot = references.appendingPathComponent("tiny-fixtures", isDirectory: true)
        let manifest = try JSONDecoder().decode(
            TinyJPEGDecodeManifest.self,
            from: Data(contentsOf: fixtureRoot.appendingPathComponent("manifest.json")))
        #expect(manifest.kind == "phase23-independent-jpeg-decode-fixtures-v1")
        #expect(manifest.oracle == "Pillow 10.3.0 decode + ImageOps.exif_transpose + RGB conversion")
        #expect(manifest.generator.pillow == "10.3.0")
        #expect(manifest.cases.count == 11)
        #expect(manifest.rejectedCases.count == 2)
        #expect(manifest.cases.filter { $0.id.hasPrefix("rgb-exif-") }.map(\.orientation)
            == Array(1...8))
        #expect(manifest.cases.contains { $0.decodedMode == "L" })
        #expect(manifest.cases.contains { $0.id == "cmyk-adobe" })
        #expect(manifest.cases.contains { $0.id == "cmyk-non-adobe" })

        for fixture in manifest.cases {
            let imageURL = fixtureRoot.appendingPathComponent(fixture.file)
            let encoded = try Data(contentsOf: imageURL)
            #expect(encoded.count == fixture.inputBytes, "\(fixture.id) encoded byte count")
            #expect(qwenImageSHA256(encoded) == fixture.inputSHA256, "\(fixture.id) input digest")
            let opened = try VisionImageSource(fileURL: imageURL)
                .open(maximumEncodedBytes: VisionImageLimits().maximumEncodedBytes)
            let metadata = try ImageMetadataReader().read(opened: opened)
            #expect(metadata.encodedWidth == fixture.encodedWidth, "\(fixture.id) encoded width")
            #expect(metadata.encodedHeight == fixture.encodedHeight, "\(fixture.id) encoded height")
            #expect(metadata.orientation == fixture.orientation, "\(fixture.id) EXIF orientation")
            #expect(metadata.orientedWidth == fixture.orientedWidth, "\(fixture.id) oriented width")
            #expect(metadata.orientedHeight == fixture.orientedHeight, "\(fixture.id) oriented height")

            let decoded = try QwenJPEGDecoder.decode(encoded: encoded, metadata: metadata)
            #expect(decoded.width == fixture.orientedWidth, "\(fixture.id) decoded width")
            #expect(decoded.height == fixture.orientedHeight, "\(fixture.id) decoded height")
            let expectedRGB = try Data(contentsOf: fixtureRoot.appendingPathComponent(fixture.rgbFile))
            #expect(expectedRGB.count == fixture.rgbByteCount, "\(fixture.id) RGB byte count")
            #expect(qwenImageSHA256(expectedRGB) == fixture.rgbSHA256, "\(fixture.id) RGB digest")
            let (actualRGB, alphaIsOpaque) = qwenRGBAndAlpha(decoded.rgba)
            #expect(actualRGB == expectedRGB, "\(fixture.id) oriented RGB differs from pinned Pillow")
            #expect(alphaIsOpaque, "\(fixture.id) decoder alpha must be 255")
        }

        for fixture in manifest.rejectedCases {
            #expect(fixture.expectedAdmission == "reject")
            let encoded = try Data(contentsOf: fixtureRoot.appendingPathComponent(fixture.file))
            #expect(encoded.count == fixture.inputBytes)
            #expect(qwenImageSHA256(encoded) == fixture.inputSHA256)
            let source = try VisionImageSource(fileURL: fixtureRoot.appendingPathComponent(fixture.file))
            #expect(throws: VisionImageError.self) {
                let opened = try source.open(maximumEncodedBytes: VisionImageLimits().maximumEncodedBytes)
                _ = try ImageMetadataReader().read(opened: opened)
            }
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

private func qwenImageSHA256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func qwenRGBAndAlpha(_ rgba: Data) -> (rgb: Data, alphaIsOpaque: Bool) {
    guard rgba.count.isMultiple(of: 4) else { return (Data(), false) }
    let pixels = rgba.count / 4
    var rgb = Data(count: pixels * 3)
    var alphaIsOpaque = true
    rgb.withUnsafeMutableBytes { rgbBuffer in
        rgba.withUnsafeBytes { rgbaBuffer in
            let output = rgbBuffer.bindMemory(to: UInt8.self)
            let input = rgbaBuffer.bindMemory(to: UInt8.self)
            for pixel in 0..<pixels {
                let source = pixel * 4
                let destination = pixel * 3
                output[destination] = input[source]
                output[destination + 1] = input[source + 1]
                output[destination + 2] = input[source + 2]
                alphaIsOpaque = alphaIsOpaque && input[source + 3] == 255
            }
        }
    }
    return (rgb, alphaIsOpaque)
}
