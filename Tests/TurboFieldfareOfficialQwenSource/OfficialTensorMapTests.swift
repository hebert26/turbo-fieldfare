import Testing
import TurboFieldfareOfficialQwenSource

@Suite struct OfficialBF16TensorMapTests {
    @Test func completePinnedIndexMatchesIndependentCategorySplit() throws {
        let descriptors = OfficialTensorMapFixture.descriptors
        let expected = OfficialTensorMapFixture.expectedCategoryByName
        #expect(descriptors.count == OfficialTensorMapFixture.expectedTotal)
        #expect(Set(descriptors.map(\.name)).count == OfficialTensorMapFixture.expectedTotal)
        #expect(descriptors.allSatisfy { $0.dataType == "BF16" })
        #expect(expected.count == OfficialTensorMapFixture.expectedTotal)
        #expect(Set(descriptors.map(\.name)) == Set(expected.keys))
        #expect(expected.values.filter { $0 == .textResident }.count
            == OfficialTensorMapFixture.expectedTextResidentCount)
        #expect(expected.values.filter { $0 == .routedExpert }.count
            == OfficialTensorMapFixture.expectedRoutedExpertCount)
        #expect(expected.values.filter { $0 == .vision }.count
            == OfficialTensorMapFixture.expectedVisionCount)
        #expect(expected.values.filter { $0 == .mtpPresentUnsupported }.count
            == OfficialTensorMapFixture.expectedMTPCount)

        let categories = try OfficialQwenTensorMap.classify(descriptors)
        #expect(categories.count == descriptors.count)
        for (descriptor, actual) in zip(descriptors, categories) {
            let independentlyExpected = try #require(expected[descriptor.name])
            #expect(actual == independentlyExpected, "misclassified \(descriptor.name)")
        }

        #expect(categories.filter { $0 == .textResident }.count
            == OfficialTensorMapFixture.expectedTextResidentCount)
        #expect(categories.filter { $0 == .routedExpert }.count
            == OfficialTensorMapFixture.expectedRoutedExpertCount)
        #expect(categories.filter { $0 == .vision }.count
            == OfficialTensorMapFixture.expectedVisionCount)
        #expect(categories.filter { $0 == .mtpPresentUnsupported }.count
            == OfficialTensorMapFixture.expectedMTPCount)

        var actualByName: [String: OfficialQwenTensorCategory] = [:]
        for (descriptor, category) in zip(descriptors, categories) {
            actualByName[descriptor.name] = category
        }
        #expect(actualByName["lm_head.weight"] == .some(.textResident))
        #expect(actualByName["model.language_model.layers.0.mlp.experts.gate_up_proj"] == .some(.routedExpert))
        #expect(actualByName["model.visual.blocks.0.norm1.weight"] == .some(.vision))
        #expect(actualByName["mtp.fc.weight"] == .some(.mtpPresentUnsupported))
    }

    @Test func everyPinnedMTPNameIsPresentAndUnsupported() throws {
        let descriptors = OfficialTensorMapFixture.descriptors
        let categories = try OfficialQwenTensorMap.classify(descriptors)
        #expect(OfficialTensorMapFixture.mtpNames.count == OfficialTensorMapFixture.expectedMTPCount)

        for name in OfficialTensorMapFixture.mtpNames {
            let index = try #require(descriptors.firstIndex { $0.name == name })
            #expect(categories[index] == .mtpPresentUnsupported, "wrong MTP category for \(name)")
        }
    }

    @Test func removingEachMTPDescriptorIndividuallyIsRejected() {
        let completeIndex = OfficialTensorMapFixture.descriptors
        for name in OfficialTensorMapFixture.mtpNames {
            let withoutOne = completeIndex.filter { $0.name != name }
            #expect(withoutOne.count == 1_044)
            #expect(rejection(for: withoutOne) == .invalidTensorSet,
                    "accepted index with missing MTP entry \(name)")
        }
    }

    @Test func duplicateMTPDescriptorIsRejected() {
        var descriptors = OfficialTensorMapFixture.descriptors
        guard let mtp = descriptors.first(where: { $0.name == "mtp.fc.weight" }) else {
            Issue.record("independent fixture is missing mtp.fc.weight")
            return
        }
        descriptors.append(mtp)
        #expect(descriptors.count == 1_046)
        #expect(rejection(for: descriptors) == .invalidTensorSet)
    }

    @Test func sameCountReplacementWithDuplicateMTPNameIsRejected() {
        let descriptors = replacing(
            "model.language_model.norm.weight",
            with: "mtp.fc.weight",
            in: OfficialTensorMapFixture.descriptors)
        #expect(descriptors.count == 1_045)
        #expect(Set(descriptors.map(\.name)).count == 1_044)
        #expect(rejection(for: descriptors) == .invalidTensorSet)
    }

    @Test func additionalUnknownMTPEntryIsRejected() {
        var descriptors = OfficialTensorMapFixture.descriptors
        descriptors.append(OfficialQwenTensorDescriptor(
            name: "mtp.layers.1.input_layernorm.weight", dataType: "BF16"))
        #expect(descriptors.count == 1_046)
        #expect(rejection(for: descriptors) == .invalidTensorSet)
    }

    @Test func sameCountUnknownMTPNameIsRejected() {
        let descriptors = replacing(
            "mtp.fc.weight",
            with: "mtp.layers.1.fc.weight",
            in: OfficialTensorMapFixture.descriptors)
        #expect(descriptors.count == 1_045)
        #expect(Set(descriptors.map(\.name)).count == 1_045)
        #expect(rejection(for: descriptors) == .invalidTensor)
    }

    @Test func languageLayerPastPinnedConfigIsRejected() {
        let descriptors = replacing(
            "model.language_model.layers.0.input_layernorm.weight",
            with: "model.language_model.layers.40.input_layernorm.weight",
            in: OfficialTensorMapFixture.descriptors)
        #expect(rejection(for: descriptors) == .invalidTensor)
    }

    @Test func visionBlockPastPinnedConfigIsRejected() {
        let descriptors = replacing(
            "model.visual.blocks.0.norm1.weight",
            with: "model.visual.blocks.27.norm1.weight",
            in: OfficialTensorMapFixture.descriptors)
        #expect(rejection(for: descriptors) == .invalidTensor)
    }

    @Test func mtpHiddenLayerPastPinnedConfigIsRejected() {
        let descriptors = replacing(
            "mtp.layers.0.input_layernorm.weight",
            with: "mtp.layers.1.input_layernorm.weight",
            in: OfficialTensorMapFixture.descriptors)
        #expect(rejection(for: descriptors) == .invalidTensor)
    }

    @Test func nonBF16TypeIsRejectedForTextAndSingleMTPDescriptors() throws {
        var textMutation = OfficialTensorMapFixture.descriptors
        let textIndex = try #require(textMutation.firstIndex { $0.name == "lm_head.weight" })
        textMutation[textIndex] = OfficialQwenTensorDescriptor(
            name: textMutation[textIndex].name, dataType: "F32")
        #expect(rejection(for: textMutation) == .invalidTensorSet)

        var mtpMutation = OfficialTensorMapFixture.descriptors
        let mtpIndex = try #require(mtpMutation.firstIndex { $0.name == "mtp.fc.weight" })
        mtpMutation[mtpIndex] = OfficialQwenTensorDescriptor(
            name: mtpMutation[mtpIndex].name, dataType: "F32")
        #expect(rejection(for: mtpMutation) == .invalidTensorSet)
    }

    @Test func reorderedValidInputKeepsOutputAlignedWithNames() throws {
        let descriptors = Array(OfficialTensorMapFixture.descriptors.reversed())
        let expected = OfficialTensorMapFixture.expectedCategoryByName
        let categories = try OfficialQwenTensorMap.classify(descriptors)

        #expect(categories.count == descriptors.count)
        for (descriptor, actual) in zip(descriptors, categories) {
            let independentlyExpected = try #require(expected[descriptor.name])
            #expect(actual == independentlyExpected, "output lost input alignment for \(descriptor.name)")
        }
    }

    private func rejection(
        for descriptors: [OfficialQwenTensorDescriptor]
    ) -> OfficialQwenTensorMapError? {
        do {
            _ = try OfficialQwenTensorMap.classify(descriptors)
            return nil
        } catch let error as OfficialQwenTensorMapError {
            return error
        } catch {
            Issue.record("classify threw an unexpected error: \(error)")
            return nil
        }
    }

    private func replacing(
        _ victim: String,
        with replacement: String,
        in descriptors: [OfficialQwenTensorDescriptor]
    ) -> [OfficialQwenTensorDescriptor] {
        var changed = descriptors
        guard let index = changed.firstIndex(where: { $0.name == victim }) else {
            Issue.record("fixture is missing replacement victim \(victim)")
            return changed
        }
        let original = changed[index]
        changed[index] = OfficialQwenTensorDescriptor(name: replacement, dataType: original.dataType)
        return changed
    }
}
