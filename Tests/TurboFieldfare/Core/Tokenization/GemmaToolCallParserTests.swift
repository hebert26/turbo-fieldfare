import Foundation
import Testing
@testable import TurboFieldfare

@Suite("Gemma tool-call parser")
struct GemmaToolCallParserTests {
    private let parser = GemmaToolCallParser()
    private let tools: Set<String> = ["visioncapture_navigate"]

    @Test func bareWordValueParsesAsString() throws {
        let call = try parser.parse(
            "call:visioncapture_navigate{action:screenshot}",
            allowedTools: tools, id: "c1")
        #expect(call.name == "visioncapture_navigate")
        #expect(call.arguments == .object(["action": .string("screenshot")]))
    }

    @Test func bareWordMixesWithDelimitedStrings() throws {
        let call = try parser.parse(
            "call:visioncapture_navigate{action:s,target:<|\"|>c3<|\"|>}",
            allowedTools: tools, id: "c2")
        #expect(call.arguments == .object([
            "action": .string("s"),
            "target": .string("c3"),
        ]))
    }

    @Test func strayWordBeforeDelimitedStringIsDropped() throws {
        let call = try parser.parse(
            "call:visioncapture_navigate{action:s<|\"|>tap<|\"|>,target:<|\"|>c3<|\"|>}",
            allowedTools: tools, id: "c5")
        #expect(call.arguments == .object([
            "action": .string("tap"),
            "target": .string("c3"),
        ]))
        #expect(!parser.hasInvalidOpeningPrefix("call:visioncapture_navigate{action:s<|\"|>tap<|\"|>,target:"))
        #expect(!parser.hasInvalidOpeningPrefix("call:visioncapture_navigate{action:s<|\""))
    }

    @Test func bareWordWithOnlyClosingDelimiterKeepsTheWord() throws {
        let call = try parser.parse(
            "call:visioncapture_navigate{action:type<|\"|>,target:<|\"|>o21<|\"|>,text:<|\"|>gemma<|\"|>}",
            allowedTools: tools, id: "c6")
        #expect(call.arguments == .object([
            "action": .string("type"),
            "target": .string("o21"),
            "text": .string("gemma"),
        ]))
        #expect(!parser.hasInvalidOpeningPrefix("call:visioncapture_navigate{action:type<|\"|>,target:<|\"|>o21<|\"|>,text:<|\"|>g"))
        #expect(!parser.hasInvalidOpeningPrefix("call:visioncapture_navigate{action:type<|\"|>"))
    }

    @Test func growingBareWordIsNotAnInvalidPrefix() {
        #expect(!parser.hasInvalidOpeningPrefix("call:visioncapture_navigate{action:scr"))
        #expect(!parser.hasInvalidOpeningPrefix("call:visioncapture_navigate{action:s,target:<|\"|>c3"))
    }

    @Test func emptyValueIsStillMalformed() {
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse("call:visioncapture_navigate{action:}", allowedTools: tools, id: "c3")
        }
    }

    @Test func literalsKeepTheirTypes() throws {
        let call = try parser.parse(
            "call:visioncapture_navigate{a:true,b:null,c:12,d:<|\"|>x<|\"|>}",
            allowedTools: tools, id: "c4")
        #expect(call.arguments == .object([
            "a": .bool(true), "b": .null, "c": .integer(12), "d": .string("x"),
        ]))
    }
}
