import Testing
import TurboFieldfareDecodeProtocol
@testable import TurboFieldfareAppCore
@testable import TurboFieldfareDecodeService

@Suite
struct DecodeServiceRuntimeOptionsTests {
    @Test
    func thinkingDifferenceNeedsToolAwareAdmission() {
        let loaded = DecodeRuntimeOptions(toolThinkingEnabled: false)
        let requested = DecodeRuntimeOptions(toolThinkingEnabled: true)
        #expect(!DecodeServiceGenerationOptions.compatible(requested, with: loaded))
        #expect(DecodeServiceGenerationOptions.compatible(requested, with: loaded, toolAware: true))
    }

    @Test
    func representativeNonThinkingDifferencesRemainStrict() {
        let loaded = DecodeRuntimeOptions(toolThinkingEnabled: false)
        #expect(!DecodeServiceGenerationOptions.compatible(loaded, with: nil, toolAware: true))
        let changedOptions = [
            DecodeRuntimeOptions(expertCacheSlots: 32, toolThinkingEnabled: true),
            DecodeRuntimeOptions(expertCachePolicy: "lru", toolThinkingEnabled: true),
            DecodeRuntimeOptions(rdadvisePolicy: "bounded", toolThinkingEnabled: true),
            DecodeRuntimeOptions(modelVerification: "trusted-install", toolThinkingEnabled: true),
            DecodeRuntimeOptions(visionResidencyPolicy: "resident", toolThinkingEnabled: true),
        ]
        for changed in changedOptions {
            #expect(!DecodeServiceGenerationOptions.compatible(changed, with: loaded, toolAware: true),
                "Only tool thinking may differ for tool-aware generation: \(changed)")
        }
    }

    @Test
    func existingPrefillExceptionsRemainPerRequest() {
        let loaded = DecodeRuntimeOptions(toolThinkingEnabled: false)
        let prefillEnabledOnly = DecodeRuntimeOptions(prefillEnabled: false, toolThinkingEnabled: false)
        let chunkSizeOnly = DecodeRuntimeOptions(prefillChunkTokens: 256, toolThinkingEnabled: false)
        let prefillAndThinking = DecodeRuntimeOptions(prefillEnabled: false, prefillChunkTokens: 256,
            toolThinkingEnabled: true)
        #expect(DecodeServiceGenerationOptions.compatible(prefillEnabledOnly, with: loaded))
        #expect(DecodeServiceGenerationOptions.compatible(chunkSizeOnly, with: loaded))
        #expect(DecodeServiceGenerationOptions.compatible(prefillEnabledOnly, with: loaded, toolAware: true))
        #expect(DecodeServiceGenerationOptions.compatible(chunkSizeOnly, with: loaded, toolAware: true))
        #expect(!DecodeServiceGenerationOptions.compatible(prefillAndThinking, with: loaded))
        #expect(DecodeServiceGenerationOptions.compatible(prefillAndThinking, with: loaded, toolAware: true))
    }

    @Test
    func forwardedThinkingValueSurvivesServiceOptionConversion() throws {
        let options = try TurboFieldfareDecodeServiceMain.appRuntimeOptions(
            DecodeRuntimeOptions(toolThinkingEnabled: true))
        #expect(options.toolThinkingEnabled)
    }

    @Test
    func emptyToolAwareBoundaryIsRetainedOnlyForEmptyRetryableStops() {
        #expect(DecodeServiceGenerationOptions.retainsEmptyToolBoundary(
            toolAware: true, hasVisibleText: false, hasToolCalls: false, stopReason: .maxTokens))
        #expect(DecodeServiceGenerationOptions.retainsEmptyToolBoundary(
            toolAware: true, hasVisibleText: false, hasToolCalls: false, stopReason: .endOfTurn))
        #expect(DecodeServiceGenerationOptions.retainsEmptyToolBoundary(
            toolAware: true, hasVisibleText: false, hasToolCalls: false, stopReason: .eos))
        #expect(DecodeServiceGenerationOptions.retainsEmptyToolBoundary(
            toolAware: true, hasVisibleText: false, hasToolCalls: false, stopReason: .stopString))
        #expect(!DecodeServiceGenerationOptions.retainsEmptyToolBoundary(
            toolAware: true, hasVisibleText: true, hasToolCalls: false, stopReason: .maxTokens))
        #expect(!DecodeServiceGenerationOptions.retainsEmptyToolBoundary(
            toolAware: true, hasVisibleText: false, hasToolCalls: true, stopReason: .maxTokens))
        #expect(!DecodeServiceGenerationOptions.retainsEmptyToolBoundary(
            toolAware: true, hasVisibleText: false, hasToolCalls: false, stopReason: .cancelled))
        #expect(!DecodeServiceGenerationOptions.retainsEmptyToolBoundary(
            toolAware: false, hasVisibleText: false, hasToolCalls: false, stopReason: .maxTokens))
    }
}
