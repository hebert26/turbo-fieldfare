import Testing
@testable import TurboFieldfareAppCore

@Suite struct AgentUserRestrictionsTests {
    @Test func explicitButtonRestrictionMatchesItsCurrentChoice() {
        var restrictions = AgentUserRestrictions()
        restrictions.apply(
            "Do not test the speak button. Continue testing the other reachable areas.")

        #expect(restrictions.displayTargets == ["speak"])
        #expect(restrictions.prohibits(label: "Speak", selector: "Speak"))
        #expect(!restrictions.prohibits(label: "Bookmarks", selector: "Bookmarks"))
    }

    @Test func aLaterPermissionRemovesTheExactRestriction() {
        var restrictions = AgentUserRestrictions()
        restrictions.apply("Never press the Speak button.")
        restrictions.apply("You may test the Speak button now.")

        #expect(restrictions.displayTargets.isEmpty)
        #expect(!restrictions.prohibits(label: "Speak", selector: "Speak"))
    }

    @Test func exactMatchingDoesNotBlockSimilarControls() {
        var restrictions = AgentUserRestrictions()
        restrictions.apply("Avoid the Speak button.")

        #expect(!restrictions.prohibits(
            label: "Speaker settings",
            selector: "Speaker settings"))
    }

    @Test func alternativeProhibitedVerbsResolveToTheControlName() {
        var restrictions = AgentUserRestrictions()
        restrictions.apply("Do not test or use the Speak button.")

        #expect(restrictions.displayTargets == ["speak"])
        #expect(restrictions.prohibits(label: "Speak", selector: "Speak"))
    }

    @Test func coordinatedProhibitionResolvesToTheSharedWorkflow() {
        var restrictions = AgentUserRestrictions()
        restrictions.apply("Do not open, manage, or create profiles again.")

        #expect(restrictions.displayTargets == ["profiles"])
        #expect(restrictions.prohibits(
            label: "Test Profile, Profile, Bookmarks & Data",
            selector: "ui.profile.card"))
        #expect(!restrictions.prohibits(label: "Done", selector: "Done"))
        #expect(!restrictions.prohibits(label: "Bookmarks", selector: "Bookmarks"))
    }

    @Test func laterPermissionRemovesAWorkflowRestriction() {
        var restrictions = AgentUserRestrictions()
        restrictions.apply("Do not open profiles again.")
        restrictions.apply("You may open profiles now.")

        #expect(restrictions.displayTargets.isEmpty)
        #expect(!restrictions.prohibits(
            label: "Test Profile, Profile, Bookmarks & Data",
            selector: "ui.profile.card"))
    }
}
