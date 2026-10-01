import CryptoKit
import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// Opt-in Phase23 consumer for two real image-conditioned chat requests. It
/// runs the production codec, image path, source conversation session, decoder,
/// and sampler. The independent feature and full-text references are read only.
@Suite(.serialized) struct QwenOfficialSourceImageTextCandidateTests {
    private static let environmentKeys = [
        "TURBO_FIELDFARE_P23_REGISTRATION",
        "TURBO_FIELDFARE_P23_VISION",
        "TURBO_FIELDFARE_P23_REQUEST",
        "TURBO_FIELDFARE_P23_REQUEST_SHA256",
        "TURBO_FIELDFARE_P23_TEXT_REFERENCE",
        "TURBO_FIELDFARE_P23_TEXT_REFERENCE_SHA256",
        "TURBO_FIELDFARE_P23_TEXT_OUTPUT",
    ]
    private static let savedArtifactEnvironmentKeys = [
        "TURBO_FIELDFARE_P23_SAVED_CANDIDATE",
        "TURBO_FIELDFARE_P23_SAVED_CANDIDATE_SHA256",
        "TURBO_FIELDFARE_P23_SAVED_TEXT_REFERENCE",
        "TURBO_FIELDFARE_P23_SAVED_TEXT_REFERENCE_SHA256",
    ]
    private static let activationCaptureEnvironmentKey =
        "TURBO_FIELDFARE_P23_CAPTURE_ACTIVATIONS"

    @Test func groupedRecorderTracksConsumedTokensBeforeRawLogits() {
        let capture = P23ImageTextCapture()
        capture.begin(promptTokenCount: 3)
        capture.consumedInput(position: 0, token: 10, featureRow: nil, positionValues: nil)
        capture.consumedInput(position: 1, token: 11, featureRow: nil, positionValues: nil)
        capture.consumedInput(position: 2, token: 12, featureRow: nil, positionValues: nil)
        capture.raw(position: 2, token: 12, logits: [0])
        let first = capture.snapshot()
        #expect(first.tokenIDs == [0: 10, 1: 11, 2: 12])
        #expect(!first.duplicate)

        capture.raw(position: 2, token: 12, logits: [0])
        #expect(capture.snapshot().duplicate)
    }

    @Test func candidateRouteSelectionUsesCutoffMembershipWithoutTieOrder() {
        var scores = [Float](repeating: 0, count: 256)
        for (index, value) in [10, 9, 8, 7, 6, 5, 4, 3, 3, 2].enumerated() {
            scores[index] = Float(value)
        }
        let weights = [Float](repeating: 0.125, count: 8)
        let margin = candidateRouteCutoffMargin(scores)

        #expect(candidateRouteSelectionPasses(
            scores: scores, ids: Array(0...6) + [7], weights: weights,
            reportedMargin: margin))
        #expect(candidateRouteSelectionPasses(
            scores: scores, ids: Array(0...6) + [8], weights: weights,
            reportedMargin: margin))
        #expect(!candidateRouteSelectionPasses(
            scores: scores, ids: Array(0...6) + [9], weights: weights,
            reportedMargin: margin))
        #expect(!candidateRouteSelectionPasses(
            scores: scores, ids: Array(0...6), weights: weights,
            reportedMargin: margin))
        #expect(!candidateRouteSelectionPasses(
            scores: scores, ids: Array(0...6) + [7, 8], weights: weights,
            reportedMargin: margin))
        #expect(!candidateRouteSelectionPasses(
            scores: scores, ids: Array(0...6) + [-1], weights: weights,
            reportedMargin: margin))
        #expect(!candidateRouteSelectionPasses(
            scores: scores, ids: Array(0...5) + [6, 6], weights: weights,
            reportedMargin: margin))
        #expect(!candidateRouteSelectionPasses(
            scores: scores, ids: Array(0...6) + [256], weights: weights,
            reportedMargin: margin))

        var nonFiniteScores = scores
        nonFiniteScores[1] = .nan
        #expect(!candidateRouteSelectionPasses(
            scores: nonFiniteScores, ids: Array(0...6) + [7], weights: weights,
            reportedMargin: margin))
        var nonFiniteWeights = weights
        nonFiniteWeights[0] = .infinity
        #expect(!candidateRouteSelectionPasses(
            scores: scores, ids: Array(0...6) + [7], weights: nonFiniteWeights,
            reportedMargin: margin))
        #expect(!candidateRouteSelectionPasses(
            scores: scores, ids: Array(0...6) + [7], weights: weights,
            reportedMargin: .nan))
    }

    @Test(.enabled(if: savedArtifactEnvironmentKeys.allSatisfy {
        !(ProcessInfo.processInfo.environment[$0] ?? "").isEmpty
    }, "set explicit saved Phase23 image-text artifacts to replay route validation"))
    func savedImageTextArtifactRevalidatesRouteFilesWithCutoffMembership() throws {
        let environment = ProcessInfo.processInfo.environment
        func required(_ key: String) throws -> String {
            guard let value = environment[key], !value.isEmpty else {
                throw P23ImageTextError.invalid("missing saved-artifact environment value")
            }
            return value
        }

        let candidateFile = try canonical(URL(fileURLWithPath: required(Self.savedArtifactEnvironmentKeys[0])))
        let candidateData = try read(candidateFile, maximum: 64 * 1024 * 1024)
        let candidateSHA = try required(Self.savedArtifactEnvironmentKeys[1])
        guard sha256(candidateData) == candidateSHA else {
            throw P23ImageTextError.invalid("saved candidate receipt digest changed")
        }
        let referenceFile = try canonical(URL(fileURLWithPath: required(Self.savedArtifactEnvironmentKeys[2])))
        let referenceData = try read(referenceFile, maximum: 64 * 1024 * 1024)
        let referenceSHA = try required(Self.savedArtifactEnvironmentKeys[3])
        guard sha256(referenceData) == referenceSHA else {
            throw P23ImageTextError.invalid("saved reference digest changed")
        }
        try validateOfficialEagerExpertReceipt(referenceData)
        let reference = try JSONDecoder().decode(P23ImageTextReference.self, from: referenceData)
        let candidateRoot = candidateFile.deletingLastPathComponent()
        let referenceRoot = referenceFile.deletingLastPathComponent()
        guard let candidateObject = try JSONSerialization.jsonObject(with: candidateData) as? [String: Any],
              let candidateCases = candidateObject["cases"] as? [[String: Any]] else {
            throw P23ImageTextError.invalid("saved candidate receipt has no cases")
        }
        let candidateIDs = candidateCases.compactMap { $0["id"] as? String }
        guard candidateIDs.count == candidateCases.count,
              candidateIDs == reference.cases.map(\.id) else {
            throw P23ImageTextError.invalid("saved candidate and reference case sets differ")
        }

        var sawEXIF6Layer28TieRegression = false
        for candidateCase in candidateCases {
            guard let caseID = candidateCase["id"] as? String,
                  let referenceCase = reference.cases.first(where: { $0.id == caseID }),
                  let candidateSteps = candidateCase["steps"] as? [[String: Any]],
                  candidateSteps.count == referenceCase.steps.count else {
                throw P23ImageTextError.invalid("saved candidate step geometry differs for case")
            }
            for candidateStep in candidateSteps {
                guard let stepIndex = candidateStep["stepIndex"] as? Int,
                      let referenceStep = referenceCase.steps.first(where: { $0.stepIndex == stepIndex }),
                      let candidateRoutes = candidateStep["routes"] as? [[String: Any]],
                      candidateRoutes.count == referenceStep.routes.count else {
                    throw P23ImageTextError.invalid("saved candidate route geometry differs")
                }
                for candidateRoute in candidateRoutes {
                    guard let layer = candidateRoute["layer"] as? Int,
                          let referenceRoute = referenceStep.routes.first(where: { $0.layer == layer }),
                          let sequenceLength = candidateRoute["sequenceLength"] as? Int,
                          let referenceRowCount = referenceRoute.routerLogits.shape.first,
                          sequenceLength == referenceRowCount,
                          let margins = (candidateRoute["selectedVsUnselectedLogitMarginsFP32"] as? [NSNumber])?.map({ $0.floatValue }),
                          margins.count == sequenceLength,
                          let logitsFile = candidateRoute["routerLogitsFile"] as? String,
                          let logitsSHA = candidateRoute["routerLogitsSHA256"] as? String,
                          let weightsFile = candidateRoute["top8WeightsFile"] as? String,
                          let weightsSHA = candidateRoute["top8WeightsSHA256"] as? String,
                          let idsFile = candidateRoute["top8ExpertIdsFile"] as? String,
                          let idsSHA = candidateRoute["top8ExpertIdsSHA256"] as? String else {
                        throw P23ImageTextError.invalid("saved candidate route metadata is incomplete")
                    }
                    guard referenceRoute.routerLogits.shape == [sequenceLength, 256],
                          referenceRoute.top8Weights.shape == [sequenceLength, 8],
                          referenceRoute.top8ExpertIds.shape == [sequenceLength, 8] else {
                        throw P23ImageTextError.invalid("saved reference route geometry differs")
                    }
                    let logitsData = try readVectorData(
                        candidateRoot, file: logitsFile, sha256: logitsSHA,
                        expectedBytes: sequenceLength * 256 * 4)
                    let weightsData = try readVectorData(
                        candidateRoot, file: weightsFile, sha256: weightsSHA,
                        expectedBytes: sequenceLength * 8 * 4)
                    let idsData = try readVectorData(
                        candidateRoot, file: idsFile, sha256: idsSHA,
                        expectedBytes: sequenceLength * 8 * 8)
                    let logits = decodeFloat32(logitsData)
                    let weights = decodeFloat32(weightsData)
                    let ids64 = stride(from: 0, to: idsData.count, by: 8).map { offset in
                        Int64(littleEndian: idsData.withUnsafeBytes {
                            $0.loadUnaligned(fromByteOffset: offset, as: Int64.self)
                        })
                    }
                    let ids = ids64.compactMap(Int.init)
                    guard ids.count == ids64.count else {
                        throw P23ImageTextError.invalid("saved candidate expert ID cannot fit Int")
                    }
                    if caseID == "natural-exif-6", stepIndex == 0, layer == 28 {
                        let referenceDirectory = try safeChild(referenceRoot, referenceCase.textOutputDirectory)
                        let referenceIDs64 = try readInt64Tensor(
                            referenceDirectory, record: referenceRoute.top8ExpertIds)
                        let referenceIDs = referenceIDs64.compactMap(Int.init)
                        guard referenceIDs.count == referenceIDs64.count,
                              ids.count >= 45 * 8,
                              logits.count >= 45 * 256,
                              weights.count >= 45 * 8,
                              margins.count > 44 else {
                            throw P23ImageTextError.invalid("saved tie regression geometry is incomplete")
                        }
                        let row = 44
                        let scoreStart = row * 256
                        let weightStart = row * 8
                        let actualIDs = Array(ids[weightStart..<(weightStart + 8)])
                        let actualScores = Array(logits[scoreStart..<(scoreStart + 256)])
                        let actualWeights = Array(weights[weightStart..<(weightStart + 8)])
                        guard actualIDs == Array(referenceIDs[weightStart..<(weightStart + 8)]),
                              actualIDs.contains(226), !actualIDs.contains(128),
                              candidateRouteSelectionPasses(
                                  scores: actualScores, ids: actualIDs,
                                  weights: actualWeights, reportedMargin: margins[row]) else {
                            throw P23ImageTextError.invalid("saved EXIF6 layer28 row44 baseline changed")
                        }

                        guard let boundaryIndex = actualIDs.firstIndex(of: 226) else {
                            throw P23ImageTextError.invalid("saved EXIF6 layer28 row44 tie boundary missing")
                        }
                        var tiedIDs = actualIDs
                        tiedIDs[boundaryIndex] = 128
                        guard tiedIDs != Array(referenceIDs[weightStart..<(weightStart + 8)]),
                              candidateRouteSelectionPasses(
                                  scores: actualScores, ids: tiedIDs,
                                  weights: actualWeights, reportedMargin: margins[row]) else {
                            throw P23ImageTextError.invalid("saved tied boundary member was rejected")
                        }

                        let cutoff = actualScores.sorted(by: >)[7]
                        guard let belowCutoff = actualScores.indices.first(where: {
                            actualScores[$0] < cutoff && !actualIDs.contains($0)
                        }) else {
                            throw P23ImageTextError.invalid("saved tie regression has no below-cutoff expert")
                        }
                        var belowCutoffIDs = tiedIDs
                        belowCutoffIDs[boundaryIndex] = belowCutoff
                        guard !candidateRouteSelectionPasses(
                            scores: actualScores, ids: belowCutoffIDs,
                            weights: actualWeights, reportedMargin: margins[row]) else {
                            throw P23ImageTextError.invalid("saved below-cutoff expert was accepted")
                        }

                        var duplicateIDs = actualIDs
                        duplicateIDs[boundaryIndex] = actualIDs[0]
                        guard !candidateRouteSelectionPasses(
                            scores: actualScores, ids: duplicateIDs,
                            weights: actualWeights, reportedMargin: margins[row]) else {
                            throw P23ImageTextError.invalid("saved duplicate expert was accepted")
                        }

                        var nonFiniteScores = actualScores
                        nonFiniteScores[0] = .nan
                        guard !candidateRouteSelectionPasses(
                            scores: nonFiniteScores, ids: actualIDs,
                            weights: actualWeights, reportedMargin: margins[row]) else {
                            throw P23ImageTextError.invalid("saved nonfinite score was accepted")
                        }
                        var nonFiniteWeights = actualWeights
                        nonFiniteWeights[0] = .infinity
                        guard !candidateRouteSelectionPasses(
                            scores: actualScores, ids: actualIDs,
                            weights: nonFiniteWeights, reportedMargin: margins[row]) else {
                            throw P23ImageTextError.invalid("saved nonfinite weight was accepted")
                        }
                        sawEXIF6Layer28TieRegression = true
                    }
                    for row in 0..<sequenceLength {
                        let scoreStart = row * 256
                        let weightStart = row * 8
                        guard candidateRouteSelectionPasses(
                            scores: Array(logits[scoreStart..<(scoreStart + 256)]),
                            ids: Array(ids[weightStart..<(weightStart + 8)]),
                            weights: Array(weights[weightStart..<(weightStart + 8)]),
                            reportedMargin: margins[row]) else {
                        throw P23ImageTextError.invalid(
                                "saved candidate route cutoff validation failed")
                        }
                    }
                }
            }
        }
        guard sawEXIF6Layer28TieRegression else {
            throw P23ImageTextError.invalid("saved EXIF6 layer28 row44 tie regression was not exercised")
        }
    }

    @Test(.enabled(if: environmentKeys.allSatisfy {
        !(ProcessInfo.processInfo.environment[$0] ?? "").isEmpty
    }, "set all explicit P23 image-text inputs after Phase22 acceptance"))
    func productionImageTextRunnerMatchesIndependentReference() async throws {
        let environment = ProcessInfo.processInfo.environment
        let captureActivations = environment[Self.activationCaptureEnvironmentKey] != "0"
        let captureMode = captureActivations ? "full-activations" : "routes-raw-sample-only"
        let prefillMode = captureActivations ? "token-major" : "grouped"
        let omittedDiagnosticArtifacts: [String] = captureActivations
            ? [] : ["final-hidden", "final-norm"]
        func required(_ key: String) throws -> String {
            guard let value = environment[key], !value.isEmpty else {
                throw P23ImageTextError.invalid("missing \(key)")
            }
            return value
        }

        let registration = try canonical(URL(fileURLWithPath: required(Self.environmentKeys[0])))
        let companion = try canonical(URL(fileURLWithPath: required(Self.environmentKeys[1])))
        let requestURL = try canonical(URL(fileURLWithPath: required(Self.environmentKeys[2])))
        let requestSHA = try required(Self.environmentKeys[3])
        let textReferenceURL = try canonical(URL(fileURLWithPath: required(Self.environmentKeys[4])))
        let textReferenceSHA = try required(Self.environmentKeys[5])
        let output = URL(fileURLWithPath: try required(Self.environmentKeys[6])).standardizedFileURL
        guard output.path == output.resolvingSymlinksInPath().path,
              !FileManager.default.fileExists(atPath: output.path) else {
            throw P23ImageTextError.invalid("candidate output directory must be new")
        }
        let requestData = try read(requestURL, maximum: 1 * 1024 * 1024)
        guard sha256(requestData) == requestSHA,
              requestSHA == "7b670c73e0c9da82c8ba5734f925d4847c0e26f4db173a10674c23cedb198f17" else {
            throw P23ImageTextError.invalid("pinned image-text request digest changed")
        }
        let request = try JSONDecoder().decode(P23ImageTextRequest.self, from: requestData)
        guard request.kind == "qwen36-original-bf16-vision-request-v1",
              request.cases.map(\.id) == ["natural-exif-1", "natural-exif-6"] else {
            throw P23ImageTextError.invalid("expected the two pinned natural-image requests")
        }

        let textReferenceData = try read(textReferenceURL, maximum: 32 * 1024 * 1024)
        guard sha256(textReferenceData) == textReferenceSHA else {
            throw P23ImageTextError.invalid("image-text reference digest changed")
        }
        try validateOfficialEagerExpertReceipt(textReferenceData)
        let textReference = try JSONDecoder().decode(
            P23ImageTextReference.self, from: textReferenceData)
        guard textReference.kind == "qwen36-original-bf16-image-text-reference-v1",
              textReference.complete,
              textReference.textExecutionMode == "layer-major-one-token-cached",
              textReference.requestSHA256 == requestSHA,
              textReference.cases.map(\.id) == request.cases.map(\.id),
              textReference.maxNewTokens == 2,
              textReference.maximumPromptTokens <= 1024,
              textReference.referenceAppliesPassFail == false else {
            throw P23ImageTextError.invalid("independent image-text reference contract changed")
        }
        let referenceRoot = textReferenceURL.deletingLastPathComponent()
        let visionReferenceURL = try safeChild(referenceRoot, textReference.visionReferenceFile)
        let visionReferenceData = try read(visionReferenceURL, maximum: 32 * 1024 * 1024)
        guard sha256(visionReferenceData) == textReference.visionReferenceSHA256 else {
            throw P23ImageTextError.invalid("feature-only reference digest changed")
        }
        let visionReference = try JSONDecoder().decode(
            P23ImageTextVisionReference.self, from: visionReferenceData)
        guard visionReference.requestSHA256 == requestSHA,
              visionReference.results.map(\.id) == request.cases.map(\.id) else {
            throw P23ImageTextError.invalid("image-text reference and feature reference differ")
        }

        for (requestCase, visionCase) in zip(request.cases, visionReference.results) {
            let imageURL = try canonical(URL(fileURLWithPath: requestCase.imagePath))
            let imageData = try read(imageURL, maximum: 20 * 1024 * 1024)
            guard requestCase.id == visionCase.id,
                  requestCase.promptText == visionCase.promptText,
                  requestCase.imagePath == visionCase.imagePath,
                  imageData.count == requestCase.imageBytes,
                  sha256(imageData) == requestCase.imageSHA256,
                  visionCase.imageBytes == requestCase.imageBytes,
                  visionCase.imageSHA256 == requestCase.imageSHA256,
                  !visionCase.tokenIds.isEmpty,
                  visionCase.positions.count == visionCase.tokenIds.count,
                  !visionCase.padRows.isEmpty,
                  visionCase.tokenIds.count <= textReference.maximumPromptTokens else {
                throw P23ImageTextError.invalid("pinned request image or prompt changed for \(requestCase.id)")
            }
        }

        let proposedContext = try MetalContext()
        try VisionRuntime.requireSupportedDevice(proposedContext.device)
        let bundle = try ModelFamilyRuntime.loadBundle(
            directoryURL: registration, device: proposedContext.device,
            streamingMode: .pread(slotCount: 16), expertCachePolicy: .lfu,
            integrityPolicy: .sizeCheckTrustedReceipt)
        guard case .qwenOfficialSource(let model) = bundle.runtime,
              let identity = bundle.sourceIdentity,
              let codec = bundle.qwenCodec,
              identity.descriptorContentSHA256 == textReference.sourceDescriptorContentSHA256,
              identity.markerSHA256 == textReference.sourceMarkerSHA256,
              identity.shardSetSHA256 == textReference.sourceShardSetSHA256,
              identity.checksumManifestSHA256 == textReference.sourceChecksumManifestSHA256,
              identity.sourceRoot == textReference.sourceRoot,
              registration.path == textReference.registrationPath else {
            throw P23ImageTextError.invalid("loaded source identity differs from the independent reference")
        }
        let context = model.context
        try OfficialQwenSourceVisionMetadataProbe.verify(
            textModelURL: registration, visionURL: companion)

        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        let capture = P23ImageTextCapture()
        let groupedCapture: QwenSourceGroupedPrefillCapture?
        if captureActivations {
            groupedCapture = nil
        } else {
            groupedCapture = QwenSourceGroupedPrefillCapture(
                observeConsumedInput: { position, token, featureRow, mropePosition in
                    capture.consumedInput(position: position, token: token,
                                          featureRow: featureRow,
                                          positionValues: mropePosition?.values)
                },
                observeFinalRawLogits: { position, token, logits in
                    capture.raw(position: position, token: token, logits: logits)
                })
        }
        let activationObserver: (@Sendable (Int, Int, String, [Float]) -> Void)?
        if captureActivations {
            activationObserver = { @Sendable (position: Int, layer: Int, stage: String, values: [Float]) in
                capture.activation(position: position, layer: layer, stage: stage, values: values)
            }
        } else {
            activationObserver = nil
        }
        let hooks = QwenOfficialSourceTransactionHooks(
            observeRoute: { position, layer, logits, ids, weights, margin in
                capture.route(position: position, layer: layer, logits: logits,
                              ids: ids, weights: weights, margin: margin)
            },
            observeRawLogits: { position, token, logits in
                capture.raw(position: position, token: token, logits: logits)
            },
            observePublicLogitsAndSample: { samplePosition, bits, token in
                capture.sample(position: samplePosition, bits: bits, token: token)
            },
            observeActivation: activationObserver,
            observeConsumedInput: { position, token, featureRow, mropePosition in
                capture.consumedInput(position: position, token: token,
                                      featureRow: featureRow,
                                      positionValues: mropePosition?.values)
            })
        let session = try await QwenOfficialSourceConversationGenerationSession(
            model: model, codec: codec, sourceIdentity: identity,
            context: context, maxContext: 1024, expertSlotCount: 16,
            modelDirectoryURL: registration, visionPackURL: companion,
            hooks: hooks, groupedCapture: groupedCapture)
        #expect(hooks.requiresSeparateGPUStages == captureActivations)
        if !captureActivations {
            #expect(!hooks.requiresSeparateGPUStages)
        }

        var caseReceipts: [[String: Any]] = []
        let totalStarted = ContinuousClock.now
        for (requestCase, visionCase) in zip(request.cases, visionReference.results) {
            if !caseReceipts.isEmpty { try await session.reset() }
            let officialCase = try #require(textReference.cases.first { $0.id == requestCase.id })
            guard officialCase.promptTokenIds == visionCase.tokenIds,
                  officialCase.steps.count == 2,
                  officialCase.steps.map(\.stepIndex) == [0, 1],
                  officialCase.sampledTokenIds.count == 2,
                  officialCase.steps[0].inputTokenIds == visionCase.tokenIds,
                  officialCase.steps[0].sequenceLength == visionCase.tokenIds.count,
                  officialCase.steps[0].positions == visionCase.positions,
                  officialCase.steps[1].inputTokenIds == [officialCase.sampledTokenIds[0]],
                  officialCase.steps[1].sequenceLength == visionCase.tokenIds.count + 1,
                  officialCase.steps[1].positions == [Array(repeating: Int32(visionCase.tokenIds.count + visionCase.textRoPEDelta), count: 3)],
                  officialCase.textOutputDirectory == requestCase.id + "-text" else {
                throw P23ImageTextError.invalid("official full-text prompt lineage changed for \(requestCase.id)")
            }
            capture.begin(promptTokenCount: visionCase.tokenIds.count)
            let imageURL = URL(fileURLWithPath: requestCase.imagePath)
            let message = ModelChatMessage(role: .user, content: .parts([
                .image(.init(id: requestCase.id)), .text(requestCase.promptText),
            ]))
            let generationRequest = QwenConversationGenerationRequest(
                turn: .user(message),
                imagesByID: [requestCase.id: imageURL],
                thinking: .disabled, visionResidency: .onDemand,
                config: .qwenRaw(maxNewTokens: 2, temperature: 0, stopTokenIDs: []))
            let caseStarted = ContinuousClock.now
            let result = try await session.generate(generationRequest)
            let elapsed = caseStarted.duration(to: .now)
            let observed = capture.snapshot()
            if !captureActivations {
                #expect(observed.activations.isEmpty)
                let diagnostics = try #require(await session.groupedPrefillDiagnostics())
                #expect(diagnostics.mode == .grouped)
                #expect(diagnostics.tokenCount == visionCase.tokenIds.count)
            }
            let consumedContinuationCount = max(0, result.acceptedGeneratedTokenIDs.count - 1)
            guard result.promptTokens == visionCase.tokenIds.count,
                  !result.acceptedGeneratedTokenIDs.isEmpty,
                  result.acceptedGeneratedTokenIDs.count <= 2,
                  observed.promptTokenCount == visionCase.tokenIds.count,
                  observed.tokenIDs.count == visionCase.tokenIds.count + consumedContinuationCount,
                  observed.consumedInputs.count == visionCase.tokenIds.count + consumedContinuationCount,
                  observed.samples.count == result.acceptedGeneratedTokenIDs.count,
                  !observed.duplicate else {
                throw P23ImageTextError.invalid("candidate text run had incomplete prompt/sample capture for \(requestCase.id)")
            }
            let consumedPositions = Array(0..<(visionCase.tokenIds.count + consumedContinuationCount))
            var promptStructureFailures = 0
            for position in consumedPositions {
                guard let consumed = observed.consumedInputs[position],
                      let token = observed.tokenIDs[position], token == consumed.token else {
                    promptStructureFailures += 1
                    continue
                }
                let expectedToken = position < visionCase.tokenIds.count
                    ? visionCase.tokenIds[position]
                    : result.acceptedGeneratedTokenIDs[position - visionCase.tokenIds.count]
                if consumed.token != expectedToken { promptStructureFailures += 1 }
                let expectedPosition = position < visionCase.positions.count
                    ? visionCase.positions[position]
                    : Array(repeating: Int32(visionCase.tokenIds.count + visionCase.textRoPEDelta), count: 3)
                if consumed.position != expectedPosition { promptStructureFailures += 1 }
            }
            let expectedFeatureValues = try readFloat32Tensor(referenceRoot, record: visionCase.features)
            guard visionCase.features.shape == [visionCase.padRows.count, 2048],
                  expectedFeatureValues.count == visionCase.padRows.count * 2048 else {
                throw P23ImageTextError.invalid("official image feature geometry changed")
            }
            var capturedFeatureValues: [Float] = []
            var featureRowIndex = 0
            for position in visionCase.padRows {
                guard let row = observed.consumedInputs[position]?.featureRow,
                      row.count == 2048, row.allSatisfy(\.isFinite) else {
                    promptStructureFailures += 1
                    continue
                }
                capturedFeatureValues.append(contentsOf: row)
                featureRowIndex += 1
            }
            for position in consumedPositions where !visionCase.padRows.contains(position) {
                if observed.consumedInputs[position]?.featureRow != nil { promptStructureFailures += 1 }
            }
            let featureComparison = compareFloats(capturedFeatureValues, expectedFeatureValues,
                                                  absolute: 1e-5, relative: 1e-5)
            promptStructureFailures += featureRowIndex == visionCase.padRows.count ? 0 : 1
            let featureData = float32LE(capturedFeatureValues)
            let featureFile = requestCase.id + ".candidate-image-features-fp32-le.bin"
            try featureData.write(to: output.appendingPathComponent(featureFile), options: .atomic)
            let consumedInputReceipt: [[String: Any]] = consumedPositions.map { position in
                let item = observed.consumedInputs[position]
                return [
                    "position": position,
                    "tokenId": item?.token as Any? ?? NSNull(),
                    "mropePosition": item?.position as Any? ?? NSNull(),
                    "hasImageFeatureRow": item?.featureRow != nil,
                ]
            }
            let inputReceiptData = try JSONSerialization.data(
                withJSONObject: consumedInputReceipt, options: [.prettyPrinted, .sortedKeys])
            try inputReceiptData.write(
                to: output.appendingPathComponent(requestCase.id + ".candidate-consumed-inputs.json"),
                options: .atomic)
            for index in visionCase.tokenIds.indices {
                guard observed.tokenIDs[index] == visionCase.tokenIds[index] else {
                    throw P23ImageTextError.invalid("candidate prompt token differs at \(requestCase.id):\(index)")
                }
            }
            if consumedContinuationCount == 1 {
                guard observed.tokenIDs[visionCase.tokenIds.count]
                        == result.acceptedGeneratedTokenIDs[0] else {
                    throw P23ImageTextError.invalid("candidate continuation input is not its sampled token")
                }
            }

            var stepReceipts: [[String: Any]] = []
            var casePassed = result.acceptedGeneratedTokenIDs == officialCase.sampledTokenIds
                && promptStructureFailures == 0 && featureComparison.failureCount == 0
            for stepIndex in 0..<2 {
                let referenceStep = officialCase.steps[stepIndex]
                guard stepIndex < result.acceptedGeneratedTokenIDs.count else {
                    stepReceipts.append([
                        "stepIndex": stepIndex,
                        "comparisonStatus": "not-produced-after-early-stop",
                        "candidateStoppedAfterStep": result.acceptedGeneratedTokenIDs.count,
                    ])
                    casePassed = false
                    continue
                }
                let rawPosition = visionCase.tokenIds.count - 1 + stepIndex
                let stepComparable = stepIndex == 0
                    || result.acceptedGeneratedTokenIDs[0] == officialCase.sampledTokenIds[0]
                let expectedInputToken = stepIndex == 0
                    ? visionCase.tokenIds[visionCase.tokenIds.count - 1]
                    : result.acceptedGeneratedTokenIDs[0]
                guard let raw = observed.raw[rawPosition],
                      raw.token == expectedInputToken,
                      let sampled = observed.samples[stepIndex],
                      sampled.token == result.acceptedGeneratedTokenIDs[stepIndex] else {
                    throw P23ImageTextError.invalid("candidate logit input lineage differs at step \(stepIndex)")
                }
                let referenceDirectory = try safeChild(referenceRoot, officialCase.textOutputDirectory)
                let expectedRaw = try readFloat32Vector(
                    referenceDirectory, file: referenceStep.rawLogitsFile,
                    sha256: referenceStep.rawLogitsSHA256,
                    count: referenceStep.rawLogitsCount)
                let expectedPublicData = try readVectorData(
                    referenceDirectory, file: referenceStep.publicFloat16LogitsFile,
                    sha256: referenceStep.publicFloat16LogitsSHA256,
                    expectedBytes: referenceStep.publicFloat16LogitsCount * 2)
                let expectedPublicBits = decodeUInt16(expectedPublicData)
                guard expectedRaw.count == raw.logits.count,
                      expectedRaw.count == 248_320,
                      expectedPublicBits.count == sampled.bits.count,
                      sampled.bits.count == 248_320 else {
                    throw P23ImageTextError.invalid("candidate/reference logit geometry differs")
                }
                let rawComparison = stepComparable ? compareFloats(raw.logits, expectedRaw) : nil
                let rawData = float32LE(raw.logits)
                let publicData = uint16LE(sampled.bits)
                let publicComparisonFailureCount = zip(sampled.bits, expectedPublicBits).reduce(into: 0) {
                    failures, pair in
                    if Float(Float16(bitPattern: pair.0)) != Float(Float16(bitPattern: pair.1)) {
                        failures += 1
                    }
                }
                let publicMatchesReference = stepComparable && publicComparisonFailureCount == 0
                let independentlyRounded = raw.logits.map { Float16($0).bitPattern }
                let publicMatchesRawRounding = independentlyRounded == sampled.bits
                let rawArgmax = argmax(raw.logits)
                let publicArgmax = argmax(sampled.bits.map { Float(Float16(bitPattern: $0)) })
                let publicArgmaxMargin = argmaxPublicMargin(sampled.bits)
                let referenceRawMargin = Float(referenceStep.argmaxMarginFP32)
                let referencePublicMargin = referenceStep.publicArgmaxMarginFP16
                let rawMarginPassed = stepComparable && closeFP32(
                    rawArgmax.margin, referenceRawMargin)
                let publicMarginPassed = stepComparable
                    && closeDouble(publicArgmaxMargin.margin, referencePublicMargin)
                let headPassed = stepComparable && rawComparison?.failureCount == 0
                    && publicMatchesReference && publicMatchesRawRounding
                    && rawArgmax.token == referenceStep.rawArgmaxTokenId
                    && publicArgmax.token == referenceStep.publicArgmaxTokenId
                    && sampled.token == referenceStep.sampledTokenId
                    && sampled.token == publicArgmax.token
                    && rawMarginPassed && publicMarginPassed
                casePassed = casePassed && headPassed

                var hiddenFile: String?
                var normFile: String?
                var hiddenSHA256: String?
                var normSHA256: String?
                var hiddenFailureCount: Int?
                var normFailureCount: Int?
                if captureActivations {
                    guard let finalHidden = observed.activations[rawPosition]?["final-hidden"],
                          let finalNorm = observed.activations[rawPosition]?["final-norm"] else {
                        throw P23ImageTextError.invalid(
                            "candidate final activation capture missing at step \(stepIndex)")
                    }
                    let expectedHidden = try readFloat32Tensor(
                        referenceDirectory, record: referenceStep.finalHidden)
                    let expectedNorm = try readFloat32Tensor(
                        referenceDirectory, record: referenceStep.finalNorm)
                    let hiddenComparison = stepComparable
                        ? compareFloats(finalHidden, expectedHidden) : nil
                    let normComparison = stepComparable
                        ? compareFloats(finalNorm, expectedNorm) : nil
                    hiddenFailureCount = hiddenComparison?.failureCount
                    normFailureCount = normComparison?.failureCount
                    let hiddenData = float32LE(finalHidden)
                    let normData = float32LE(finalNorm)
                    hiddenFile = "\(requestCase.id).step-\(stepIndex).candidate.final-hidden-fp32-le.bin"
                    normFile = "\(requestCase.id).step-\(stepIndex).candidate.final-norm-fp32-le.bin"
                    hiddenSHA256 = sha256(hiddenData)
                    normSHA256 = sha256(normData)
                    try hiddenData.write(
                        to: output.appendingPathComponent(hiddenFile!), options: .atomic)
                    try normData.write(
                        to: output.appendingPathComponent(normFile!), options: .atomic)
                }
                let rawFile = "\(requestCase.id).step-\(stepIndex).candidate.raw-fp32-le.bin"
                let publicFile = "\(requestCase.id).step-\(stepIndex).candidate.public-fp16-le.bin"
                try rawData.write(to: output.appendingPathComponent(rawFile), options: .atomic)
                try publicData.write(to: output.appendingPathComponent(publicFile), options: .atomic)

                let expectedRoutes = referenceStep.routes.sorted { $0.layer < $1.layer }
                guard expectedRoutes.count == 40 else {
                    throw P23ImageTextError.invalid("reference route count differs")
                }
                let positions = stepIndex == 0
                    ? Array(0..<visionCase.tokenIds.count)
                    : [visionCase.tokenIds.count]
                var routeReceipts: [[String: Any]] = []
                for expectedRoute in expectedRoutes {
                    let candidateRows = try positions.map { position -> P23ImageTextRouteRow in
                        guard let row = observed.routes[position]?[expectedRoute.layer] else {
                            throw P23ImageTextError.invalid("missing route at position \(position), layer \(expectedRoute.layer)")
                        }
                        return row
                    }
                    let routeLogits = candidateRows.flatMap(\.logits)
                    let routeWeights = candidateRows.flatMap(\.weights)
                    let routeIDs = candidateRows.flatMap(\.ids)
                    let expectedRouteLogits = try readFloat32Tensor(referenceDirectory, record: expectedRoute.routerLogits)
                    let expectedRouteWeights = try readFloat32Tensor(referenceDirectory, record: expectedRoute.top8Weights)
                    let expectedRouteIDs = try readInt64Tensor(referenceDirectory, record: expectedRoute.top8ExpertIds)
                    guard routeLogits.count == expectedRouteRouteCount(expectedRoute.routerLogits),
                          routeWeights.count == expectedRouteRouteCount(expectedRoute.top8Weights),
                          routeIDs.count == expectedRouteRouteCount(expectedRoute.top8ExpertIds),
                          expectedRoute.routerLogits.shape == [positions.count, 256],
                          expectedRoute.top8Weights.shape == [positions.count, 8],
                          expectedRoute.top8ExpertIds.shape == [positions.count, 8],
                          routeLogits.count == expectedRouteLogits.count,
                          routeWeights.count == expectedRouteWeights.count,
                          routeIDs.count == expectedRouteIDs.count else {
                        throw P23ImageTextError.invalid("candidate/reference route geometry differs for layer \(expectedRoute.layer)")
                    }
                    let routeLogitComparison = stepComparable
                        ? compareFloats(routeLogits, expectedRouteLogits) : nil
                    var alignedExpectedWeights: [Float] = []
                    var routeIDFailureCount = 0
                    var candidateRouteValidationFailureCount = 0
                    var expectedCutoffMargins: [Float] = []
                    var candidateCutoffMargins: [Float] = []
                    for rowIndex in positions.indices {
                        let scoreRange = (rowIndex * 256)..<(rowIndex * 256 + 256)
                        let idRange = (rowIndex * 8)..<(rowIndex * 8 + 8)
                        let scores = Array(expectedRouteLogits[scoreRange])
                        let referenceIDs = Array(expectedRouteIDs[idRange]).map(Int.init)
                        let referenceWeights = Array(expectedRouteWeights[idRange])
                        let actualIDs = candidateRows[rowIndex].ids
                        let candidateScores = candidateRows[rowIndex].logits
                        let ranked = (0..<256).sorted {
                            scores[$0] == scores[$1] ? $0 < $1 : scores[$0] > scores[$1]
                        }
                        expectedCutoffMargins.append(
                            scores[ranked[7]] - scores[ranked[8]])
                        candidateCutoffMargins.append(candidateRouteCutoffMargin(candidateScores))
                        let cutoff = scores[ranked[7]]
                        let mandatory = Set(scores.indices.filter { scores[$0] > cutoff })
                        let tied = Set(scores.indices.filter { scores[$0] == cutoff })
                        let actualSet = Set(actualIDs)
                        if stepComparable {
                            if actualIDs.count != 8 || actualSet.count != 8
                                || !mandatory.isSubset(of: actualSet)
                                || !actualSet.isSubset(of: mandatory.union(tied)) {
                                routeIDFailureCount += 1
                            }
                            if !candidateRouteSelectionPasses(
                                scores: candidateScores,
                                ids: actualIDs,
                                weights: candidateRows[rowIndex].weights,
                                reportedMargin: candidateRows[rowIndex].margin) {
                                candidateRouteValidationFailureCount += 1
                            }
                            if scores[ranked[7]] > scores[ranked[8]], actualIDs != referenceIDs {
                                routeIDFailureCount += 1
                            }
                        }
                        let weightsByID = Dictionary(uniqueKeysWithValues: zip(referenceIDs, referenceWeights))
                        let tiedWeight = referenceIDs.indices.first(where: { tied.contains(referenceIDs[$0]) })
                            .map { referenceWeights[$0] }
                        for expert in actualIDs {
                            if let weight = weightsByID[expert] {
                                alignedExpectedWeights.append(weight)
                            } else if tied.contains(expert), let tiedWeight {
                                alignedExpectedWeights.append(tiedWeight)
                            } else {
                                alignedExpectedWeights.append(.nan)
                            }
                        }
                    }
                    let routeWeightComparison = stepComparable
                        ? compareFloats(routeWeights, alignedExpectedWeights) : nil
                    let actualCutoffMargins = candidateRows.map(\.margin)
                    let routeMarginComparison = stepComparable
                        ? compareFloats(actualCutoffMargins, expectedCutoffMargins) : nil
                    if stepComparable {
                        casePassed = casePassed && routeLogitComparison?.failureCount == 0
                            && routeWeightComparison?.failureCount == 0
                            && routeIDFailureCount == 0
                            && candidateRouteValidationFailureCount == 0
                            && routeMarginComparison?.failureCount == 0
                    }
                    let prefix = "\(requestCase.id).step-\(stepIndex).layer-\(expectedRoute.layer).candidate"
                    let logitsFile = prefix + ".router-fp32-le.bin"
                    let weightsFile = prefix + ".weights-fp32-le.bin"
                    let idsFile = prefix + ".expert-ids-int64-le.bin"
                    let logitsData = float32LE(routeLogits)
                    let weightsData = float32LE(routeWeights)
                    let idsData = int64LE(routeIDs.map(Int64.init))
                    try logitsData.write(to: output.appendingPathComponent(logitsFile), options: .atomic)
                    try weightsData.write(to: output.appendingPathComponent(weightsFile), options: .atomic)
                    try idsData.write(to: output.appendingPathComponent(idsFile), options: .atomic)
                    routeReceipts.append([
                        "layer": expectedRoute.layer,
                        "routerLogitsFile": logitsFile,
                        "routerLogitsSHA256": sha256(logitsData),
                        "top8WeightsFile": weightsFile,
                        "top8WeightsSHA256": sha256(weightsData),
                        "top8ExpertIdsFile": idsFile,
                        "top8ExpertIdsSHA256": sha256(idsData),
                        "sequenceLength": positions.count,
                        "comparisonStatus": stepComparable ? "compared" : "incomparable-input-token-divergence",
                        "routerLogitFailureCount": routeLogitComparison.map { $0.failureCount as Any } ?? NSNull(),
                        "top8WeightFailureCount": routeWeightComparison.map { $0.failureCount as Any } ?? NSNull(),
                        "top8IDFailureCount": stepComparable ? routeIDFailureCount as Any : NSNull(),
                        "candidateRouteValidationFailureCount": stepComparable ? candidateRouteValidationFailureCount as Any : NSNull(),
                        "routeCutoffMarginFailureCount": routeMarginComparison.map { $0.failureCount as Any } ?? NSNull(),
                        "selectedVsUnselectedLogitMarginsFP32": actualCutoffMargins,
                    ])
                }
                let candidateInputTokenIds = stepIndex == 0
                    ? visionCase.tokenIds
                    : [result.acceptedGeneratedTokenIDs[0]]
                stepReceipts.append([
                    "stepIndex": stepIndex,
                    "inputTokenIds": candidateInputTokenIds,
                    "rawLogitsFile": rawFile,
                    "rawLogitsSHA256": sha256(rawData),
                    "rawLogitsCount": raw.logits.count,
                    "comparisonStatus": stepComparable ? "compared" : "incomparable-input-token-divergence",
                    "rawLogitFailureCount": rawComparison.map { $0.failureCount as Any } ?? NSNull(),
                    "rawLogitMaximumAbsoluteError": rawComparison.map { $0.maximumAbsoluteError as Any } ?? NSNull(),
                    "publicFloat16LogitsFile": publicFile,
                    "publicFloat16LogitsSHA256": sha256(publicData),
                    "publicFloat16LogitsCount": sampled.bits.count,
                    "publicMatchesReferenceExactly": publicMatchesReference,
                    "publicLogitFailureCount": stepComparable ? publicComparisonFailureCount as Any : NSNull(),
                    "publicMatchesRawRoundingExactly": publicMatchesRawRounding,
                    "rawArgmaxMarginFP32": rawArgmax.margin,
                    "rawArgmaxMarginMatchesReference": rawMarginPassed,
                    "publicArgmaxMarginFP16": publicArgmaxMargin.margin,
                    "publicArgmaxMarginMatchesReference": publicMarginPassed,
                    "publicArgmaxMarginAbsoluteError": abs(publicArgmaxMargin.margin - referencePublicMargin),
                    "sampledTokenId": sampled.token,
                    "rawArgmaxTokenId": rawArgmax.token,
                    "publicArgmaxTokenId": publicArgmax.token,
                    "finalHiddenFile": hiddenFile.map { $0 as Any } ?? NSNull(),
                    "finalHiddenSHA256": hiddenSHA256.map { $0 as Any } ?? NSNull(),
                    "finalHiddenDiagnosticFailureCount": hiddenFailureCount.map { $0 as Any } ?? NSNull(),
                    "finalNormFile": normFile.map { $0 as Any } ?? NSNull(),
                    "finalNormSHA256": normSHA256.map { $0 as Any } ?? NSNull(),
                    "finalNormDiagnosticFailureCount": normFailureCount.map { $0 as Any } ?? NSNull(),
                    "captureMode": captureMode,
                    "prefillMode": prefillMode,
                    "omittedDiagnosticArtifacts": omittedDiagnosticArtifacts,
                    "routes": routeReceipts,
                ])
            }
            let passed = casePassed
            let caseReceipt: [String: Any] = [
                "id": requestCase.id,
                "promptTokenIds": visionCase.tokenIds,
                "promptTokenCount": visionCase.tokenIds.count,
                "sampledTokenIds": result.acceptedGeneratedTokenIDs,
                "consumedInputTokenIds": consumedPositions.compactMap { observed.consumedInputs[$0]?.token },
                "consumedMRoPEPositions": consumedPositions.map { observed.consumedInputs[$0]?.position as Any? ?? NSNull() },
                "promptStructureFailureCount": promptStructureFailures,
                "imageFeatureFailureCount": featureComparison.failureCount,
                "imageFeatureMaximumAbsoluteError": featureComparison.maximumAbsoluteError,
                "imageFeaturesFile": featureFile,
                "imageFeaturesSHA256": sha256(featureData),
                "imageFeaturesCount": capturedFeatureValues.count,
                "consumedInputsFile": requestCase.id + ".candidate-consumed-inputs.json",
                "consumedInputsSHA256": sha256(inputReceiptData),
                "sourceDescriptorContentSHA256": identity.descriptorContentSHA256,
                "sourceMarkerSHA256": identity.markerSHA256,
                "sourceShardSetSHA256": identity.shardSetSHA256,
                "sourceChecksumManifestSHA256": identity.checksumManifestSHA256,
                "prefillSeconds": result.prefillSeconds,
                "decodeSeconds": result.decodeSeconds,
                "elapsedSeconds": seconds(elapsed),
                "captureMode": captureMode,
                "prefillMode": prefillMode,
                "omittedDiagnosticArtifacts": omittedDiagnosticArtifacts,
                "steps": stepReceipts,
                "passed": passed,
            ]
            let encoded = try JSONSerialization.data(withJSONObject: jsonSafe(caseReceipt),
                                                    options: [.prettyPrinted, .sortedKeys])
            try encoded.write(to: output.appendingPathComponent(requestCase.id + ".candidate.json"),
                              options: .withoutOverwriting)
            caseReceipts.append(caseReceipt)
            try model.revalidateSource()
        }
        let receipt: [String: Any] = [
            "kind": "qwen36-original-bf16-image-text-candidate-v1",
            "complete": true,
            "requestSHA256": requestSHA,
            "textReferenceSHA256": textReferenceSHA,
            "visionReferenceSHA256": textReference.visionReferenceSHA256,
            "sourceDescriptorContentSHA256": identity.descriptorContentSHA256,
            "sourceMarkerSHA256": identity.markerSHA256,
            "sourceShardSetSHA256": identity.shardSetSHA256,
            "sourceChecksumManifestSHA256": identity.checksumManifestSHA256,
            "sourceRoot": identity.sourceRoot,
            "registrationPath": registration.path,
            "integrityPolicy": "sizeCheckTrustedReceipt",
            "expertCacheSlots": 16,
            "expertCachePolicy": "lfu",
            "temperature": 0,
            "maxNewTokens": 2,
            "captureMode": captureMode,
            "prefillMode": prefillMode,
            "omittedDiagnosticArtifacts": omittedDiagnosticArtifacts,
            "tolerance": ["absolute": 1e-7, "relative": 1e-6, "extraULP": 0],
            "elapsedSeconds": seconds(totalStarted.duration(to: .now)),
            "cases": caseReceipts,
            "passed": caseReceipts.count == 2 && caseReceipts.allSatisfy { $0["passed"] as? Bool == true },
        ]
        let receiptData = try JSONSerialization.data(withJSONObject: jsonSafe(receipt),
                                                     options: [.prettyPrinted, .sortedKeys])
        try receiptData.write(to: output.appendingPathComponent("image-text-candidate.json"),
                              options: .withoutOverwriting)
        #expect(caseReceipts.count == 2)
        #expect(caseReceipts.allSatisfy { $0["passed"] as? Bool == true })
    }
}

private final class P23ImageTextCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var promptTokenCount = 0
    private var tokenIDs: [Int: Int32] = [:]
    private var rawTokens: [Int: Int32] = [:]
    private var consumedInputs: [Int: P23ImageTextConsumedInput] = [:]
    private var rawLogits: [Int: P23ImageTextRaw] = [:]
    private var activations: [Int: [String: [Float]]] = [:]
    private var samples: [Int: P23ImageTextSample] = [:]
    private var routes: [Int: [Int: P23ImageTextRouteRow]] = [:]
    private var duplicate = false

    func begin(promptTokenCount: Int) {
        lock.lock(); defer { lock.unlock() }
        self.promptTokenCount = promptTokenCount
        tokenIDs.removeAll(keepingCapacity: true)
        rawTokens.removeAll(keepingCapacity: true)
        consumedInputs.removeAll(keepingCapacity: true)
        rawLogits.removeAll(keepingCapacity: true)
        activations.removeAll(keepingCapacity: true)
        samples.removeAll(keepingCapacity: true)
        routes.removeAll(keepingCapacity: true)
        duplicate = false
    }

    func activation(position: Int, layer: Int, stage: String, values: [Float]) {
        guard layer == -1, (stage == "final-hidden" || stage == "final-norm") else { return }
        lock.lock(); defer { lock.unlock() }
        var atPosition = activations[position, default: [:]]
        if atPosition.updateValue(values, forKey: stage) != nil { duplicate = true }
        activations[position] = atPosition
    }

    func raw(position: Int, token: Int32, logits: [Float]) {
        lock.lock(); defer { lock.unlock() }
        if let existing = tokenIDs[position] {
            if existing != token { duplicate = true }
        } else {
            tokenIDs[position] = token
        }
        if rawTokens.updateValue(token, forKey: position) != nil { duplicate = true }
        if position == promptTokenCount - 1 || position == promptTokenCount {
            if rawLogits.updateValue(P23ImageTextRaw(token: token, logits: logits),
                                     forKey: position) != nil { duplicate = true }
        }
    }

    func consumedInput(position: Int, token: Int32, featureRow: [Float]?,
                       positionValues: [Int32]?) {
        lock.lock(); defer { lock.unlock() }
        guard position >= 0, position <= promptTokenCount else { return }
        if let existing = tokenIDs[position] {
            if existing != token { duplicate = true }
        } else {
            tokenIDs[position] = token
        }
        if consumedInputs.updateValue(P23ImageTextConsumedInput(
            token: token, featureRow: featureRow, position: positionValues),
            forKey: position) != nil { duplicate = true }
    }

    func sample(position: Int, bits: [UInt16], token: Int32) {
        lock.lock(); defer { lock.unlock() }
        if samples.updateValue(P23ImageTextSample(bits: bits, token: token),
                               forKey: position) != nil { duplicate = true }
    }

    func route(position: Int, layer: Int, logits: [Float], ids: [Int],
               weights: [Float], margin: Float) {
        lock.lock(); defer { lock.unlock() }
        guard position <= promptTokenCount else { return }
        var atPosition = routes[position, default: [:]]
        if atPosition.updateValue(P23ImageTextRouteRow(
            logits: logits, ids: ids, weights: weights, margin: margin),
            forKey: layer) != nil { duplicate = true }
        routes[position] = atPosition
    }

    func snapshot() -> P23ImageTextCaptureSnapshot {
        lock.lock(); defer { lock.unlock() }
        return P23ImageTextCaptureSnapshot(promptTokenCount: promptTokenCount,
            tokenIDs: tokenIDs, consumedInputs: consumedInputs,
            raw: rawLogits, samples: samples,
            routes: routes, activations: activations, duplicate: duplicate)
    }
}

private struct P23ImageTextCaptureSnapshot {
    let promptTokenCount: Int
    let tokenIDs: [Int: Int32]
    let consumedInputs: [Int: P23ImageTextConsumedInput]
    let raw: [Int: P23ImageTextRaw]
    let samples: [Int: P23ImageTextSample]
    let routes: [Int: [Int: P23ImageTextRouteRow]]
    let activations: [Int: [String: [Float]]]
    let duplicate: Bool
}

private struct P23ImageTextConsumedInput {
    let token: Int32
    let featureRow: [Float]?
    let position: [Int32]?
}

private struct P23ImageTextRaw { let token: Int32; let logits: [Float] }
private struct P23ImageTextSample { let bits: [UInt16]; let token: Int32 }
private struct P23ImageTextRouteRow {
    let logits: [Float]
    let ids: [Int]
    let weights: [Float]
    let margin: Float
}

private struct P23ImageTextRequest: Decodable {
    let kind: String
    let cases: [P23ImageTextRequestCase]
}
private struct P23ImageTextRequestCase: Decodable {
    let id: String
    let imagePath: String
    let imageSHA256: String
    let imageBytes: Int
    let promptText: String
}
private struct P23ImageTextVisionReference: Decodable {
    let requestSHA256: String
    let results: [P23ImageTextVisionCase]
}
private struct P23ImageTextVisionCase: Decodable {
    let id: String
    let imagePath: String
    let imageSHA256: String
    let imageBytes: Int
    let promptText: String
    let tokenIds: [Int32]
    let padRows: [Int]
    let positions: [[Int32]]
    let textRoPEDelta: Int
    let features: P23ImageTextTensorFile
}
private struct P23ImageTextReference: Decodable {
    let kind: String
    let complete: Bool
    let textExecutionMode: String
    let referenceAppliesPassFail: Bool
    let requestSHA256: String
    let sourceDescriptorContentSHA256: String
    let sourceMarkerSHA256: String
    let sourceShardSetSHA256: String
    let sourceChecksumManifestSHA256: String
    let sourceRoot: String
    let registrationPath: String
    let visionReferenceFile: String
    let visionReferenceSHA256: String
    let maximumPromptTokens: Int
    let maxNewTokens: Int
    let cases: [P23ImageTextReferenceCase]
}
private struct P23ImageTextReferenceCase: Decodable {
    let id: String
    let promptTokenIds: [Int32]
    let textOutputDirectory: String
    let sampledTokenIds: [Int32]
    let steps: [P23ImageTextReferenceStep]
}
private struct P23ImageTextReferenceStep: Decodable {
    let stepIndex: Int
    let inputTokenIds: [Int32]
    let positions: [[Int32]]
    let sequenceLength: Int
    let routes: [P23ImageTextRouteTensorFiles]
    let finalHidden: P23ImageTextTensorFile
    let finalNorm: P23ImageTextTensorFile
    let rawLogitsFile: String
    let rawLogitsSHA256: String
    let rawLogitsCount: Int
    let publicFloat16LogitsFile: String
    let publicFloat16LogitsSHA256: String
    let publicFloat16LogitsCount: Int
    let rawArgmaxTokenId: Int
    let publicArgmaxTokenId: Int
    let sampledTokenId: Int32
    let argmaxMarginFP32: Double
    let publicArgmaxMarginFP16: Double
}
private struct P23ImageTextRouteTensorFiles: Decodable {
    let layer: Int
    let routerLogits: P23ImageTextTensorFile
    let top8Weights: P23ImageTextTensorFile
    let top8ExpertIds: P23ImageTextTensorFile
}
private struct P23ImageTextTensorFile: Decodable {
    let file: String
    let shape: [Int]
    let byteCount: Int
    let sha256: String
}
private struct P23ImageTextFloatComparison {
    let failureCount: Int
    let maximumAbsoluteError: Double
}

private enum P23ImageTextError: Error { case invalid(String) }

/// The earlier run 004 did not prove which expert implementation the official
/// reference actually dispatched. Require the complete run 005 receipt before
/// loading Metal or the model, so an older reference cannot qualify a run.
private func validateOfficialEagerExpertReceipt(_ data: Data) throws {
    func invalid(_ detail: String) -> P23ImageTextError {
        .invalid("reference is not an official eager-experts qualification receipt: \(detail)")
    }
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw invalid("root is not an object")
    }
    guard let environment = root["environment"] as? [String: Any],
          let metadataDispatch = root["metadataExpertDispatch"] as? [String: Any],
          let configurationBefore = root["expertConfigurationBeforeConstructor"] as? [String: Any],
          let configurationAfter = root["expertConfigurationAfterConstructor"] as? [String: Any],
          root["expertsScope"] as? String == "official-eager-experts",
          environment["experts"] as? String == "eager" else {
        throw invalid("required run 005 expert fields are absent (legacy run 004 is not qualifying)")
    }

    let expectedCommit = "bd15bc95a89e728bbc1224084eb3b5829428c353"
    let expectedTree = "80eb369e589827bc7ed45b3a1f0ead5457097535"
    let expectedModelingSource = "971b08ed3eb7452f3f5f1b0f8ab8e602fc4c4e2cc10d0a4e99ec1e1830776074"
    let selectedForward = "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeExperts.forward"
    guard root["transformersCommit"] as? String == expectedCommit,
          environment["transformersCommit"] as? String == expectedCommit,
          environment["transformersTree"] as? String == expectedTree,
          environment["modelingSourceSHA256"] as? String == expectedModelingSource,
          environment["python"] as? String == "3.12.3",
          environment["torch"] as? String == "2.10.0",
          environment["numpy"] as? String == "2.4.3",
          environment["pillow"] as? String == "10.3.0",
          environment["transformers"] as? String == "5.18.0.dev0",
          environment["device"] as? String == "cpu",
          environment["dtype"] as? String == "float32",
          environment["attention"] as? String == "eager",
          environment["threads"] as? Int == 1,
          environment["deterministic"] as? Bool == true,
          (root["sourceRoot"] as? String)?.hasSuffix("/official-995ad96eacd98c81ed38be0c5b274b04031597b0") == true else {
        throw invalid("official Transformers, model, or CPU environment pin changed")
    }
    guard metadataDispatch["implementation"] as? String == "eager",
          metadataDispatch["interfaceSelectedOriginalForward"] as? Bool == true,
          metadataDispatch["selectedForward"] as? String == selectedForward else {
        throw invalid("metadata dispatch did not select the original eager experts forward")
    }

    func configuration(_ value: Any, label: String) throws -> [String: [String: Any]] {
        guard let components = value as? [String: Any],
              Set(components.keys) == Set(["parent", "text", "vision"]) else {
            throw invalid("\(label) configuration snapshot is incomplete")
        }
        var result: [String: [String: Any]] = [:]
        for name in ["parent", "text", "vision"] {
            guard let component = components[name] as? [String: Any],
                  component["publicConfig"] is [String: Any],
                  component.keys.contains("expertsImplementation"),
                  component.keys.contains("attentionImplementation"),
                  component["expertsImplementation"] is NSNull
                    || component["expertsImplementation"] is String,
                  component["attentionImplementation"] is NSNull
                    || component["attentionImplementation"] is String else {
                throw invalid("\(label) \(name) configuration snapshot is incomplete")
            }
            result[name] = component
        }
        return result
    }
    let selection = try configuration(root["expertConfigurationBeforeSelection"] as Any,
                                      label: "before-selection")
    let before = try configuration(configurationBefore, label: "before-constructor")
    let after = try configuration(configurationAfter, label: "after-constructor")
    for name in ["parent", "text", "vision"] {
        guard let old = before[name], let new = after[name],
              NSDictionary(dictionary: old["publicConfig"] as! [String: Any])
                .isEqual(to: new["publicConfig"] as! [String: Any]),
              new["expertsImplementation"] as? String == "eager" else {
            throw invalid("\(name) config changed or experts were not eager after construction")
        }
        guard let selected = selection[name],
              NSDictionary(dictionary: selected["publicConfig"] as! [String: Any])
                .isEqual(to: old["publicConfig"] as! [String: Any]) else {
            throw invalid("\(name) public config changed during backend selection")
        }
    }
    guard after["text"]?["attentionImplementation"] as? String == "eager" else {
        throw invalid("text attention was not eager after construction")
    }

    guard let cases = root["cases"] as? [[String: Any]], !cases.isEmpty else {
        throw invalid("no image-text cases")
    }
    let expectedPairs = Set((0..<2).flatMap { step in (0..<40).map { "\(step):\($0)" } })
    for item in cases {
        guard let caseID = item["id"] as? String,
              let inventory = item["layerExpertInventory"] as? [[String: Any]],
              inventory.count == 80 else {
            throw invalid("case inventory must contain exactly 80 layer records")
        }
        var actualPairs = Set<String>()
        for entry in inventory {
            guard let step = entry["stepIndex"] as? Int,
                  let layer = entry["layer"] as? Int,
                  let loadedExpertCount = entry["loadedExpertCount"] as? Int,
                  loadedExpertCount == 256,
                  entry["expertsImplementation"] as? String == "eager",
                  let beforeDispatch = entry["expertDispatchBeforeTokenLoop"] as? [String: Any],
                  let afterDispatch = entry["expertDispatchAfterTokenLoop"] as? [String: Any],
                  isExpectedEagerDispatch(beforeDispatch, selectedForward: selectedForward),
                  isExpectedEagerDispatch(afterDispatch, selectedForward: selectedForward),
                  NSDictionary(dictionary: beforeDispatch).isEqual(to: afterDispatch) else {
                throw invalid("case \(caseID) has incomplete inventory or dispatch proof")
            }
            actualPairs.insert("\(step):\(layer)")
        }
        guard actualPairs.count == 80, actualPairs == expectedPairs,
              let steps = item["steps"] as? [[String: Any]],
              Set(steps.compactMap { $0["stepIndex"] as? Int }) == Set([0, 1]),
              steps.count == 2 else {
            throw invalid("case \(caseID) inventory does not cover both steps and all 40 layers")
        }
        for step in steps {
            guard let routes = step["routes"] as? [[String: Any]] else {
                throw invalid("case \(caseID) is missing route records")
            }
            let routeLayers = routes.compactMap { $0["layer"] as? Int }
            guard routes.count == 40, routeLayers.count == 40,
                  Set(routeLayers) == Set(0..<40) else {
                throw invalid("case \(caseID) route layer IDs are not exactly 0 through 39")
            }
        }
    }
}

private func isExpectedEagerDispatch(_ value: [String: Any], selectedForward: String) -> Bool {
    value["implementation"] as? String == "eager"
        && value["interfaceSelectedOriginalForward"] as? Bool == true
        && value["selectedForward"] as? String == selectedForward
}

private func compareFloats(_ actual: [Float], _ expected: [Float],
                           absolute: Double = 1e-7, relative: Double = 1e-6)
    -> P23ImageTextFloatComparison {
    guard actual.count == expected.count else {
        return P23ImageTextFloatComparison(failureCount: max(actual.count, expected.count),
                                           maximumAbsoluteError: .infinity)
    }
    var failures = 0
    var maximum = 0.0
    for (a, e) in zip(actual, expected) {
        let error = abs(Double(a) - Double(e))
        maximum = max(maximum, error)
        if !a.isFinite || !e.isFinite || error > absolute + relative * abs(Double(e)) {
            failures += 1
        }
    }
    return P23ImageTextFloatComparison(failureCount: failures, maximumAbsoluteError: maximum)
}

private func candidateRouteCutoffMargin(_ scores: [Float]) -> Float {
    guard scores.count >= 9, scores.allSatisfy(\.isFinite) else { return .nan }
    let sorted = scores.sorted(by: >)
    return sorted[7] - sorted[8]
}

private func candidateRouteSelectionPasses(
    scores: [Float], ids: [Int], weights: [Float], reportedMargin: Float
) -> Bool {
    guard scores.count == 256,
          scores.allSatisfy(\.isFinite),
          ids.count == 8,
          Set(ids).count == 8,
          ids.allSatisfy({ scores.indices.contains($0) }),
          weights.count == 8,
          weights.allSatisfy(\.isFinite),
          reportedMargin.isFinite else {
        return false
    }
    let sorted = scores.sorted(by: >)
    let cutoff = sorted[7]
    let mandatory = Set(scores.indices.filter { scores[$0] > cutoff })
    let tied = Set(scores.indices.filter { scores[$0] == cutoff })
    let actualSet = Set(ids)
    return mandatory.isSubset(of: actualSet)
        && actualSet.isSubset(of: mandatory.union(tied))
        && closeFP32(reportedMargin, sorted[7] - sorted[8])
        && closeDouble(weights.reduce(0.0) { $0 + Double($1) }, 1.0)
}

private func expectedRouteRouteCount(_ tensor: P23ImageTextTensorFile) -> Int {
    tensor.shape.reduce(1, *)
}

private func argmax(_ values: [Float]) -> (token: Int, margin: Float) {
    var best = -1
    var second = -1
    for index in values.indices {
        if best < 0 || values[index] > values[best] {
            second = best
            best = index
        } else if second < 0 || values[index] > values[second] {
            second = index
        }
    }
    guard best >= 0, second >= 0 else { return (-1, .nan) }
    return (best, values[best] - values[second])
}

private func argmax(_ bits: [UInt16]) -> (token: Int, margin: Float) {
    argmax(bits.map { Float(Float16(bitPattern: $0)) })
}

private func argmaxPublicMargin(_ bits: [UInt16]) -> (token: Int, margin: Double) {
    var best = -1
    var second = -1
    for index in bits.indices {
        let value = Float16(bitPattern: bits[index])
        if best < 0 || value > Float16(bitPattern: bits[best]) {
            second = best
            best = index
        } else if second < 0 || value > Float16(bitPattern: bits[second]) {
            second = index
        }
    }
    guard best >= 0, second >= 0 else { return (-1, .nan) }
    return (best, Double(Float16(bitPattern: bits[best]))
        - Double(Float16(bitPattern: bits[second])))
}

private func closeFP32(_ actual: Float, _ expected: Float) -> Bool {
    abs(Double(actual) - Double(expected)) <= 1e-7 + 1e-6 * abs(Double(expected))
}

private func closeDouble(_ actual: Double, _ expected: Double) -> Bool {
    abs(actual - expected) <= 1e-7 + 1e-6 * abs(expected)
}

private func readFloat32Vector(_ root: URL, file: String, sha256 expectedSHA: String,
                               count: Int) throws -> [Float] {
    let data = try readVectorData(root, file: file, sha256: expectedSHA,
                                  expectedBytes: count * 4)
    return decodeFloat32(data)
}

private func readFloat32Tensor(_ root: URL, record: P23ImageTextTensorFile) throws -> [Float] {
    guard record.byteCount == expectedRouteRouteCount(record) * 4 else {
        throw P23ImageTextError.invalid("reference FP32 tensor byte count differs")
    }
    let data = try readVectorData(root, file: record.file, sha256: record.sha256,
                                  expectedBytes: record.byteCount)
    return decodeFloat32(data)
}

private func readInt64Tensor(_ root: URL, record: P23ImageTextTensorFile) throws -> [Int64] {
    guard record.byteCount == expectedRouteRouteCount(record) * 8 else {
        throw P23ImageTextError.invalid("reference Int64 tensor byte count differs")
    }
    let data = try readVectorData(root, file: record.file, sha256: record.sha256,
                                  expectedBytes: record.byteCount)
    return stride(from: 0, to: data.count, by: 8).map { offset in
        Int64(littleEndian: data.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: offset, as: Int64.self)
        })
    }
}

private func readVectorData(_ root: URL, file: String, sha256 expectedSHA: String,
                            expectedBytes: Int) throws -> Data {
    let url = try safeChild(root, file)
    let data = try read(url, maximum: max(expectedBytes, 1))
    guard data.count == expectedBytes, sha256(data) == expectedSHA else {
        throw P23ImageTextError.invalid("reference tensor size or digest changed: \(file)")
    }
    return data
}

private func decodeFloat32(_ data: Data) -> [Float] {
    stride(from: 0, to: data.count, by: 4).map { offset in
        Float(bitPattern: UInt32(littleEndian: data.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
        }))
    }
}

private func decodeUInt16(_ data: Data) -> [UInt16] {
    stride(from: 0, to: data.count, by: 2).map { offset in
        UInt16(littleEndian: data.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: offset, as: UInt16.self)
        })
    }
}

private func float32LE(_ values: [Float]) -> Data {
    var data = Data(capacity: values.count * 4)
    for value in values {
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }
    return data
}

private func uint16LE(_ values: [UInt16]) -> Data {
    var data = Data(capacity: values.count * 2)
    for value in values {
        var bits = value.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }
    return data
}

private func int64LE(_ values: [Int64]) -> Data {
    var data = Data(capacity: values.count * 8)
    for value in values {
        var bits = value.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }
    return data
}

private func safeChild(_ root: URL, _ name: String) throws -> URL {
    guard !name.isEmpty, URL(fileURLWithPath: name).lastPathComponent == name else {
        throw P23ImageTextError.invalid("unsafe evidence filename: \(name)")
    }
    return try canonical(root.appendingPathComponent(name))
}

private func canonical(_ url: URL) throws -> URL {
    let value = url.standardizedFileURL
    guard value.path == value.resolvingSymlinksInPath().path,
          FileManager.default.fileExists(atPath: value.path) else {
        throw P23ImageTextError.invalid("noncanonical or missing path: \(url.path)")
    }
    return value
}

private func read(_ url: URL, maximum: Int) throws -> Data {
    let path = try canonical(url)
    let attributes = try FileManager.default.attributesOfItem(atPath: path.path)
    guard attributes[.type] as? FileAttributeType == .typeRegular,
          let size = attributes[.size] as? NSNumber, size.intValue > 0,
          size.intValue <= maximum else {
        throw P23ImageTextError.invalid("nonregular or oversized evidence: \(path.path)")
    }
    let bytes = try Data(contentsOf: path)
    guard bytes.count == size.intValue else {
        throw P23ImageTextError.invalid("evidence changed during read: \(path.path)")
    }
    return bytes
}

private func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func jsonSafe(_ value: Any) -> Any {
    if let values = value as? [Any] {
        return values.map(jsonSafe)
    }
    if let dictionary = value as? [String: Any] {
        return dictionary.mapValues(jsonSafe)
    }
    if let number = value as? NSNumber, !number.doubleValue.isFinite {
        return NSNull()
    }
    return value
}

private func seconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
}
