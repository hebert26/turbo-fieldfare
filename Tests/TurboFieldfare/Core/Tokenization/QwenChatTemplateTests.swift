import Foundation
import Testing
@testable import TurboFieldfare

/// Independent text and token-ID literals captured from the pinned Qwen fast
/// chat-template oracle. No expected value is produced by the SUT.
private enum QwenTemplateOracleVectors {
    static let ordinaryThinkingOnText = "<|im_start|>user\nExplain NFC briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n"
    static let ordinaryThinkingOnIDs: [Int32] = [
        248045, 846, 198, 814, 20139, 45629, 25899, 13, 248046,
        198, 248045, 74455, 198, 248068, 198,
    ]
    static let ordinaryThinkingOffText = "<|im_start|>user\nExplain NFC briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    static let ordinaryThinkingOffIDs: [Int32] = [
        248045, 846, 198, 814, 20139, 45629, 25899, 13, 248046,
        198, 248045, 74455, 198, 248068, 271, 248069, 271,
    ]

    static let adjacentRolesIDs: [Int32] = [
        248045, 8678, 198, 2244, 18252, 13, 248046, 198,
        248045, 846, 198, 3765, 248046, 198,
        248045, 846, 198, 5394, 248046, 198,
        248045, 74455, 198, 8944, 799, 248046, 198,
        248045, 74455, 198, 8944, 1330, 248046, 198,
        248045, 846, 198, 30686, 248046, 198,
        248045, 74455, 198, 248068, 271, 248069, 271,
    ]
    static let adjacentRolesText = "<|im_start|>system\nSystem guidance.<|im_end|>\n<|im_start|>user\nfirst<|im_end|>\n<|im_start|>user\nsecond<|im_end|>\n<|im_start|>assistant\nanswer one<|im_end|>\n<|im_start|>assistant\nanswer two<|im_end|>\n<|im_start|>user\nthird<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    static let unicodeWhitespaceText = "<|im_start|>user\npadded 雪<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    static let unicodeWhitespaceIDs: [Int32] = [
        248045, 846, 198, 79, 16336, 220, 97055, 248046,
        198, 248045, 74455, 198, 248068, 271, 248069, 271,
    ]

    static let imageText = "<|im_start|>user\nSee <|vision_start|><|image_pad|><|vision_end|> now.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nchecking<|im_end|>\n<|im_start|>user\n<tool_response>\nresult <|vision_start|><|image_pad|><|vision_end|>\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    static let imageIDs: [Int32] = [
        248045, 846, 198, 9538, 220, 248053, 248056, 248054, 1381, 13,
        248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
        55914, 248046, 198, 248045, 846, 198, 248066, 198, 1334, 220,
        248053, 248056, 248054, 198, 248067, 248046, 198, 248045, 74455,
        198, 248068, 271, 248069, 271,
    ]

    static let reasoningFieldText = "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\n<think>\nprivate reason\n</think>\n\nvisible answer<|im_end|>\n<|im_start|>assistant\n<think>\n"
    static let reasoningFieldIDs: [Int32] = [
        248045, 846, 198, 7593, 248046, 198, 248045, 74455, 198,
        248068, 198, 1929, 2781, 198, 248069, 271, 12239, 4087,
        248046, 198, 248045, 74455, 198, 248068, 198,
    ]

    static let embeddedThinkText = "<|im_start|>user\nfirst<|im_end|>\n<|im_start|>assistant\nvisible old<|im_end|>\n<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    static let embeddedThinkIDs: [Int32] = [
        248045, 846, 198, 3765, 248046, 198, 248045, 74455, 198,
        12239, 2235, 248046, 198, 248045, 846, 198, 3480, 248046,
        198, 248045, 74455, 198, 248068, 271, 248069, 271,
    ]

    static let preserveThinkingText = "<|im_start|>user\nfirst<|im_end|>\n<|im_start|>assistant\n<think>\nkept reason\n</think>\n\nkept visible<|im_end|>\n<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    static let preserveThinkingIDs: [Int32] = [
        248045, 846, 198, 3765, 248046, 198, 248045, 74455, 198,
        248068, 198, 94542, 2781, 198, 248069, 271, 94542, 9155,
        248046, 198, 248045, 846, 198, 3480, 248046, 198,
        248045, 74455, 198, 248068, 271, 248069, 271,
    ]

    static let adjacentToolText = "<|im_start|>user\nquery<|im_end|>\n<|im_start|>user\n<tool_response>\norphan accepted\n</tool_response>\n<tool_response>\nsecond orphan accepted\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    static let adjacentToolIDs: [Int32] = [
        248045, 846, 198, 1574, 248046, 198, 248045, 846, 198,
        248066, 198, 269, 9649, 11330, 198, 248067, 198, 248066,
        198, 5394, 12381, 11330, 198, 248067, 248046, 198, 248045,
        74455, 198, 248068, 271, 248069, 271,
    ]

    static let historyWithoutGenerationText = "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nanswer<|im_end|>\n"
    static let historyWithoutGenerationIDs: [Int32] = [
        248045, 846, 198, 7593, 248046, 198, 248045, 74455, 198,
        248068, 271, 248069, 271, 8944, 248046, 198,
    ]

    static let toolCaseRenderedSHA256 = "e72bcb69ed94fcc26d270f6fa56dfc349627e0704d806e3ec0b469be80e11462"
    static let toolCaseIDsSHA256 = "c356f0829c65e7d2c9cbd5d34370af07c7e8e5eeddeedace308ea475518ca1de"
    static let toolCaseTokenCount = 493
    static let toolCaseRenderedTextExact = """
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{\"type\": \"function\", \"function\": {\"name\": \"lookup_雪<&>'\", \"description\": \"Line one\\nLine two: 雪 <>&'\", \"parameters\": {\"type\": \"object\", \"properties\": {\"raw\": {\"type\": \"string\", \"description\": \"雪 <>&'\"}, \"flag\": {\"type\": \"boolean\"}, \"count\": {\"type\": \"number\"}, \"payload\": {\"type\": \"object\"}}, \"required\": [\"raw\", \"flag\", \"count\", \"payload\"]}}}
{\"type\": \"function\", \"function\": {\"name\": \"second\", \"description\": \"Second tool\", \"parameters\": {\"type\": \"object\", \"properties\": {}}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>

Use exact tools.<|im_end|>
<|im_start|>user
Call both.<|im_end|>
<|im_start|>assistant
<think>

</think>

<tool_call>
<function=lookup_雪<&>'>
<parameter=raw>
true<&>'雪
next
</parameter>
<parameter=flag>
true
</parameter>
<parameter=count>
1.5
</parameter>
<parameter=payload>
{\"z\": 2, \"a\": \"雪<&>'\"}
</parameter>
</function>
</tool_call>
<tool_call>
<function=second>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
first result
</tool_response>
<tool_response>
second result
</tool_response><|im_end|>
<|im_start|>assistant
<think>

</think>


"""

    static let toolCaseIDs: [Int32] = [
        248045, 8678, 198, 2, 13455, 271, 2523, 599, 2528, 310, 279, 2614,
        5568, 25, 271, 27, 15449, 29, 198, 4754, 1267, 763, 328, 1628,
        487, 328, 1628, 763, 5046, 591, 763, 328, 20377, 62, 97055, 50490,
        5417, 487, 328, 4532, 763, 328, 2380, 799, 1639, 2380, 1330, 25,
        220, 97055, 361, 5608, 22063, 328, 13390, 763, 5046, 1267, 763,
        328, 1640, 487, 328, 12811, 763, 5046, 1006, 763, 5046, 1267, 763,
        328, 889, 487, 328, 4532, 763, 328, 97055, 361, 5608, 6, 13933,
        328, 9610, 763, 5046, 1267, 763, 328, 5925, 13933, 328, 1767, 763,
        5046, 1267, 763, 328, 3946, 13933, 328, 18837, 763, 5046, 1267,
        763, 328, 1640, 8934, 2069, 328, 6081, 763, 4241, 1006, 487, 328,
        9610, 487, 328, 1767, 487, 328, 18837, 1293, 72964, 198, 4754,
        1267, 763, 328, 1628, 487, 328, 1628, 763, 5046, 591, 763, 328,
        5394, 487, 328, 4532, 763, 328, 15207, 5224, 487, 328, 13390, 763,
        5046, 1267, 763, 328, 1640, 487, 328, 12811, 763, 313, 3307, 3307,
        198, 510, 15449, 29, 271, 2592, 488, 4992, 310, 1562, 264, 709,
        25835, 9559, 303, 279, 2614, 3443, 440, 5486, 19900, 25, 271,
        248058, 198, 27, 1628, 28, 8422, 8901, 1224, 29, 198, 27, 15704,
        28, 8422, 24109, 62, 16, 29, 198, 927, 62, 16, 198, 510, 15704,
        29, 198, 27, 15704, 28, 8422, 24109, 62, 17, 29, 198, 1919, 369,
        279, 869, 364, 279, 2018, 5555, 198, 8761, 628, 9111, 198, 34493,
        4965, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059, 271, 27,
        95328, 29, 198, 92065, 25, 198, 12, 5534, 6526, 26834, 1732, 279,
        5024, 3443, 25, 449, 8906, 361, 1628, 28, 1076, 1419, 1628, 29,
        2424, 1902, 381, 23283, 2785, 220, 248058, 248059, 11535, 9212,
        198, 12, 12296, 4868, 26834, 381, 5024, 198, 12, 1394, 1189, 3300,
        9801, 31626, 364, 678, 709, 1562, 303, 5629, 3992, 54588, 279, 709,
        1562, 11, 694, 4045, 1238, 198, 12, 1368, 1017, 369, 874, 709, 1562,
        2420, 11, 4087, 279, 3296, 1040, 4472, 440, 678, 1428, 6337, 321,
        635, 524, 3184, 279, 1156, 883, 709, 6526, 198, 510, 95328, 29, 271,
        9947, 4581, 7141, 13, 248046, 198, 248045, 846, 198, 6994, 2107, 13,
        248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271, 248058,
        198, 27, 1628, 28, 20377, 62, 97055, 50490, 5417, 29, 198, 27,
        15704, 28, 1006, 29, 198, 1802, 50490, 5417, 97055, 198, 3480,
        198, 510, 15704, 29, 198, 27, 15704, 28, 9610, 29, 198, 1802,
        198, 510, 15704, 29, 198, 27, 15704, 71404, 29, 198, 16, 13, 20,
        198, 510, 15704, 29, 198, 27, 15704, 16874, 6771, 29, 198, 4754,
        89, 763, 220, 17, 11, 328, 64, 763, 328, 97055, 50490, 5417, 8934,
        198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059, 198, 248058,
        198, 27, 1628, 28, 5394, 29, 198, 510, 1628, 29, 198, 248059,
        248046, 198, 248045, 846, 198, 248066, 198, 3765, 1067, 198, 248067,
        198, 248066, 198, 5394, 1067, 198, 248067, 248046, 198, 248045, 74455,
        198, 248068, 271, 248069, 271,
    ]

    static let emptyHistoryError = "Cannot apply chat template to an empty conversation. Provide at least one message."
    static let laterSystemError = "System message must be at the beginning."
    static let noUserQueryError = "No user query found in messages."
    static let unexpectedRoleError = "Unexpected message role."
    static let systemImageError = "System message cannot contain images."
    static let malformedContentError = "Unexpected item type in content."
    static let unsupportedAudioError = "Structured audio content is unsupported in Phase 14."
    static let unsupportedVideoError = "Structured video content is unsupported in Phase 14."
    static let nonObjectArgumentsError = "Can only get item pairs from a mapping."

    // These branch vectors are transcribed from both byte-identical runs of
    // trim-chat-reference-run-{1,2}.json. They are not produced by Swift.
    static let scalarSystemText = "<|im_start|>system\nSystem<|im_end|>\n<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    static let scalarSystemIDs: [Int32] = [
        248045, 8678, 198, 2244, 248046, 198, 248045, 846, 198,
        14162, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
    ]
    static let scalarUserText = "<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    static let scalarUserIDs: [Int32] = [
        248045, 846, 198, 14162, 248046, 198, 248045, 74455, 198,
        248068, 271, 248069, 271,
    ]
    static let scalarReasoningText = "<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\n<think>\nReason\n</think>\n\nAnswer<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    static let scalarReasoningIDs: [Int32] = [
        248045, 846, 198, 14162, 248046, 198, 248045, 74455, 198,
        248068, 198, 24342, 198, 248069, 271, 15666, 248046, 198,
        248045, 74455, 198, 248068, 271, 248069, 271,
    ]
    static let scalarEmbeddedText = "<|im_start|>user\nFirst<|im_end|>\n<|im_start|>assistant\n\u{001f}Visible<|im_end|>\n<|im_start|>user\nNext<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    static let scalarEmbeddedIDs: [Int32] = [
        248045, 846, 198, 5170, 248046, 198, 248045, 74455, 198,
        219, 5537, 248046, 198, 248045, 846, 198, 5666, 248046, 198,
        248045, 74455, 198, 248068, 271, 248069, 271,
    ]
    static let scalarToolTrimText = "<|im_start|>user\nCall<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=lookup>\n<parameter=q>\nx\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    static let scalarToolTrimIDs: [Int32] = [
        248045, 846, 198, 6994, 248046, 198, 248045, 74455, 198,
        248068, 271, 248069, 271, 248058, 198, 27, 1628, 28, 20377, 29,
        198, 27, 15704, 60922, 29, 198, 87, 198, 510, 15704, 29, 198,
        510, 1628, 29, 198, 248059, 248046, 198, 248045, 74455, 198,
        248068, 271, 248069, 271,
    ]
    static let scalarToolInteriorText = "<|im_start|>user\nCall<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nnote\u{001c}inside\n\n<tool_call>\n<function=lookup>\n<parameter=q>\nx\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    static let scalarToolInteriorIDs: [Int32] = [
        248045, 846, 198, 6994, 248046, 198, 248045, 74455, 198,
        248068, 271, 248069, 271, 9679, 216, 39983, 271, 248058, 198,
        27, 1628, 28, 20377, 29, 198, 27, 15704, 60922, 29, 198, 87,
        198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059, 248046,
        198, 248045, 74455, 198, 248068, 271, 248069, 271,
    ]
    static let scalarReverseText = "<|im_start|>user\nReal query<|im_end|>\n<|im_start|>assistant\n<think>\nKept reason\n</think>\n\nAnswer<|im_end|>\n<|im_start|>user\n<tool_response>x</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    static let scalarReverseIDs: [Int32] = [
        248045, 846, 198, 12402, 3134, 248046, 198, 248045, 74455, 198,
        248068, 198, 6399, 409, 2781, 198, 248069, 271, 15666, 248046,
        198, 248045, 846, 198, 248066, 87, 248067, 248046, 198, 248045,
        74455, 198, 248068, 271, 248069, 271,
    ]
}

@Suite("Qwen chat template")
struct QwenChatTemplateTests {
    let tokenizer: QwenTokenizer
    let codec: QwenChatCodec

    init() throws {
        let source = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0", isDirectory: true)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-chat-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        do {
            for name in ["tokenizer.json", "tokenizer_config.json"] {
                let bytes = try Data(contentsOf: source.appendingPathComponent(name), options: [.mappedIfSafe])
                try bytes.write(to: directory.appendingPathComponent(name), options: [.atomic])
            }
            let tokenizer = try QwenTokenizer.loadOfficialSidecar(from: directory)
            try FileManager.default.removeItem(at: directory)
            self.tokenizer = tokenizer
            self.codec = QwenChatCodec(tokenizer: tokenizer)
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    @Test("Thinking-on ordinary prompt matches rendered text and full token vector")
    func ordinaryThinkingOn() throws {
        let messages = [ModelChatMessage(role: .user, content: "Explain NFC briefly.")]
        let options = ModelChatRenderOptions(addGenerationPrompt: true, enableThinking: true)
        #expect(try codec.renderPrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.ordinaryThinkingOnText)
        #expect(try codec.encodePrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.ordinaryThinkingOnIDs)
    }

    @Test("Thinking-off ordinary prompt emits the pinned empty thought channel")
    func ordinaryThinkingOff() throws {
        let messages = [ModelChatMessage(role: .user, content: "Explain NFC briefly.")]
        let options = ModelChatRenderOptions(addGenerationPrompt: true, enableThinking: false)
        #expect(try codec.renderPrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.ordinaryThinkingOffText)
        #expect(try codec.encodePrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.ordinaryThinkingOffIDs)
    }

    @Test("Adjacent same-role messages and developer role follow template boundaries")
    func adjacentRoles() throws {
        let messages = [
            ModelChatMessage(role: .system, content: "System guidance."),
            ModelChatMessage(role: .user, content: "first"),
            ModelChatMessage(role: .user, content: "second"),
            ModelChatMessage(role: .assistant, content: "answer one"),
            ModelChatMessage(role: .assistant, content: "answer two"),
            ModelChatMessage(role: .user, content: "third"),
        ]
        let options = ModelChatRenderOptions(addGenerationPrompt: true, enableThinking: false)
        #expect(try codec.renderPrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.adjacentRolesText)
        #expect(try codec.encodePrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.adjacentRolesIDs)
    }

    @Test("Unicode and whitespace are preserved exactly")
    func unicodeWhitespace() throws {
        let messages = [ModelChatMessage(role: .user, content: "\u{2003}\u{00a0} padded 雪 \u{00a0}\u{2003}")]
        let options = ModelChatRenderOptions(addGenerationPrompt: true, enableThinking: false)
        #expect(try codec.renderPrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.unicodeWhitespaceText)
        #expect(try codec.encodePrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.unicodeWhitespaceIDs)
    }

    @Test("Images in user and tool content use vision markers")
    func images() throws {
        let messages = [
            ModelChatMessage(role: .user, content: .parts([
                .text("See "), .image(), .text(" now."),
            ])),
            ModelChatMessage(role: .assistant, content: "checking"),
            ModelChatMessage(role: .tool, content: .parts([
                .text("result "), .image(),
            ])),
        ]
        let options = ModelChatRenderOptions(addGenerationPrompt: true, enableThinking: false)
        #expect(try codec.renderPrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.imageText)
        #expect(try codec.encodePrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.imageIDs)
    }

    @Test("Assistant reasoning content is preserved only before visible answer")
    func reasoningField() throws {
        let messages = [
            ModelChatMessage(role: .user, content: "question"),
            ModelChatMessage(role: .assistant, content: "visible answer", reasoningContent: "private reason"),
        ]
        let options = ModelChatRenderOptions(addGenerationPrompt: true, enableThinking: true)
        #expect(try codec.renderPrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.reasoningFieldText)
        #expect(try codec.encodePrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.reasoningFieldIDs)
    }

    @Test("Existing assistant text is not treated as a new thought without preserveThinking")
    func embeddedThinking() throws {
        let messages = [
            ModelChatMessage(role: .user, content: "first"),
            ModelChatMessage(role: .assistant, content: "<think>\nold reason\n</think>\nvisible old"),
            ModelChatMessage(role: .user, content: "next"),
        ]
        let options = ModelChatRenderOptions(addGenerationPrompt: true, enableThinking: false)
        #expect(try codec.renderPrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.embeddedThinkText)
        #expect(try codec.encodePrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.embeddedThinkIDs)
    }

    @Test("Preserve-thinking option keeps prior reasoning content")
    func preserveThinking() throws {
        let messages = [
            ModelChatMessage(role: .user, content: "first"),
            ModelChatMessage(role: .assistant, content: "kept visible", reasoningContent: "kept reason"),
            ModelChatMessage(role: .user, content: "next"),
        ]
        let options = ModelChatRenderOptions(addGenerationPrompt: true, enableThinking: false, preserveThinking: true)
        #expect(try codec.renderPrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.preserveThinkingText)
        #expect(try codec.encodePrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.preserveThinkingIDs)
    }

    @Test("Adjacent tool responses do not require linkage metadata")
    func adjacentToolMessages() throws {
        let messages = [
            ModelChatMessage(role: .user, content: "query"),
            ModelChatMessage(role: .tool, content: "orphan accepted"),
            ModelChatMessage(role: .tool, content: "second orphan accepted"),
        ]
        let options = ModelChatRenderOptions(addGenerationPrompt: true, enableThinking: false)
        #expect(try codec.renderPrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.adjacentToolText)
        #expect(try codec.encodePrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.adjacentToolIDs)
    }

    @Test("Generation prompt can be omitted while retaining history")
    func historyWithoutGenerationPrompt() throws {
        let messages = [
            ModelChatMessage(role: .user, content: "question"),
            ModelChatMessage(role: .assistant, content: "answer"),
        ]
        let options = ModelChatRenderOptions(addGenerationPrompt: false, enableThinking: false)
        #expect(try codec.renderPrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.historyWithoutGenerationText)
        #expect(try codec.encodePrompt(messages: messages, tools: [], options: options) == QwenTemplateOracleVectors.historyWithoutGenerationIDs)
    }

    @Test("Python scalar trim branches match both pinned offline reference runs")
    func scalarTrimReferenceBranches() throws {
        let thinkingOff = ModelChatRenderOptions(addGenerationPrompt: true, enableThinking: false)
        let scalarSystem = [
            ModelChatMessage(role: .system, content: "\u{001c}System\u{001f}"),
            ModelChatMessage(role: .user, content: "Question"),
        ]
        #expect(try codec.renderPrompt(messages: scalarSystem, tools: [], options: thinkingOff)
                == QwenTemplateOracleVectors.scalarSystemText)
        #expect(try codec.encodePrompt(messages: scalarSystem, tools: [], options: thinkingOff)
                == QwenTemplateOracleVectors.scalarSystemIDs)

        let scalarUser = [ModelChatMessage(role: .user, content: "\u{001c}Question\u{001f}")]
        #expect(try codec.renderPrompt(messages: scalarUser, tools: [], options: thinkingOff)
                == QwenTemplateOracleVectors.scalarUserText)
        #expect(try codec.encodePrompt(messages: scalarUser, tools: [], options: thinkingOff)
                == QwenTemplateOracleVectors.scalarUserIDs)

        let scalarReasoning = [
            ModelChatMessage(role: .user, content: "Question"),
            ModelChatMessage(role: .assistant, content: "\u{001f}Answer\u{001c}", reasoningContent: "\u{001d}Reason\u{001e}"),
        ]
        #expect(try codec.renderPrompt(messages: scalarReasoning, tools: [], options: thinkingOff)
                == QwenTemplateOracleVectors.scalarReasoningText)
        #expect(try codec.encodePrompt(messages: scalarReasoning, tools: [], options: thinkingOff)
                == QwenTemplateOracleVectors.scalarReasoningIDs)

        let scalarEmbedded = [
            ModelChatMessage(role: .user, content: "First"),
            ModelChatMessage(role: .assistant, content: "\u{001c}<think>\n\u{001d}Reason\u{001e}\n</think>\n\u{001f}Visible\u{001c}"),
            ModelChatMessage(role: .user, content: "Next"),
        ]
        #expect(try codec.renderPrompt(messages: scalarEmbedded, tools: [], options: thinkingOff)
                == QwenTemplateOracleVectors.scalarEmbeddedText)
        #expect(try codec.encodePrompt(messages: scalarEmbedded, tools: [], options: thinkingOff)
                == QwenTemplateOracleVectors.scalarEmbeddedIDs)

        let scalarTool = [
            ModelChatMessage(role: .user, content: "Call"),
            ModelChatMessage(role: .assistant, content: "\u{001c}\u{001d}\u{001e}\u{001f}", toolCalls: [
                ModelChatToolCall(name: "lookup", arguments: .object([
                    ModelChatJSONMember("q", .string("x")),
                ])),
            ]),
        ]
        #expect(try codec.renderPrompt(messages: scalarTool, tools: [], options: thinkingOff)
                == QwenTemplateOracleVectors.scalarToolTrimText)
        #expect(try codec.encodePrompt(messages: scalarTool, tools: [], options: thinkingOff)
                == QwenTemplateOracleVectors.scalarToolTrimIDs)

        let scalarToolInterior = [
            ModelChatMessage(role: .user, content: "Call"),
            ModelChatMessage(role: .assistant, content: "note\u{001c}inside", toolCalls: [
                ModelChatToolCall(name: "lookup", arguments: .object([
                    ModelChatJSONMember("q", .string("x")),
                ])),
            ]),
        ]
        #expect(try codec.renderPrompt(messages: scalarToolInterior, tools: [], options: thinkingOff)
                == QwenTemplateOracleVectors.scalarToolInteriorText)
        #expect(try codec.encodePrompt(messages: scalarToolInterior, tools: [], options: thinkingOff)
                == QwenTemplateOracleVectors.scalarToolInteriorIDs)

        let scalarReverse = [
            ModelChatMessage(role: .user, content: "Real query"),
            ModelChatMessage(role: .assistant, content: "Answer", reasoningContent: "\u{001d}Kept reason\u{001e}"),
            ModelChatMessage(role: .user, content: "\u{001c}<tool_response>x</tool_response>\u{001f}"),
        ]
        #expect(try codec.renderPrompt(messages: scalarReverse, tools: [], options: thinkingOff)
                == QwenTemplateOracleVectors.scalarReverseText)
        #expect(try codec.encodePrompt(messages: scalarReverse, tools: [], options: thinkingOff)
                == QwenTemplateOracleVectors.scalarReverseIDs)

        let scalarOnlyToolResponse = [
            ModelChatMessage(role: .user, content: "\u{001c}<tool_response>x</tool_response>\u{001f}"),
        ]
        do {
            _ = try codec.renderPrompt(messages: scalarOnlyToolResponse, tools: [], options: thinkingOff)
            Issue.record("scalar-wrapped tool response was accepted as a real user query")
        } catch let error as QwenChatCodecError {
            #expect(error == .noUserQuery)
        } catch {
            Issue.record("unexpected scalar no-user-query error: \(error)")
        }
    }

    @Test("Python strip positives and non-whitespace controls are exact at edges")
    func scalarTrimMembership() throws {
        let positives: [Unicode.Scalar] = [
            "\u{0009}", "\u{000a}", "\u{000b}", "\u{000c}", "\u{000d}",
            "\u{001c}", "\u{001d}", "\u{001e}", "\u{001f}", "\u{0020}",
            "\u{0085}", "\u{00a0}", "\u{1680}", "\u{2000}", "\u{2001}",
            "\u{2002}", "\u{2003}", "\u{2004}", "\u{2005}", "\u{2006}",
            "\u{2007}", "\u{2008}", "\u{2009}", "\u{200a}", "\u{2028}",
            "\u{2029}", "\u{202f}", "\u{205f}", "\u{3000}",
        ]
        for scalar in positives {
            let messages = [ModelChatMessage(role: .user, content: "\(scalar)Question\(scalar)")]
            let rendered = try codec.renderPrompt(
                messages: messages, tools: [], options: .init(addGenerationPrompt: false))
            #expect(rendered == "<|im_start|>user\nQuestion<|im_end|>\n")
        }

        let negatives: [Unicode.Scalar] = [
            "\u{0000}", "\u{000e}", "\u{001b}", "\u{007f}",
            "\u{180e}", "\u{200b}", "\u{feff}",
        ]
        for scalar in negatives {
            let content = "\(scalar)Question\(scalar)"
            let messages = [ModelChatMessage(role: .user, content: content)]
            let rendered = try codec.renderPrompt(
                messages: messages, tools: [], options: .init(addGenerationPrompt: false))
            #expect(rendered == "<|im_start|>user\n\(content)<|im_end|>\n")
        }
    }

    @Test("Newline-only helpers preserve interior non-LF controls")
    func newlineOnlyHelpersRemainNarrow() throws {
        let messages = [
            ModelChatMessage(role: .user, content: "first"),
            ModelChatMessage(
                role: .assistant,
                content: "\n<think>\nReason\u{000e}middle\n</think>\nVisible\u{001b}middle\n"),
            ModelChatMessage(role: .user, content: "next"),
        ]
        let rendered = try codec.renderPrompt(
            messages: messages, tools: [], options: .init(addGenerationPrompt: true, enableThinking: false))
        #expect(rendered == "<|im_start|>user\nfirst<|im_end|>\n"
                + "<|im_start|>assistant\nVisible\u{001b}middle<|im_end|>\n"
                + "<|im_start|>user\nnext<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    @Test("Finite JSON numbers match every pinned Python boundary spelling")
    func finiteJSONNumberBoundaries() throws {
        let records: [(String, UInt64, String)] = [
            ("b0000000000000000", 0x0000000000000000, "0.0"),
            ("b8000000000000000", 0x8000000000000000, "-0.0"),
            ("b0000000000000001", 0x0000000000000001, "5e-324"),
            ("b000fffffffffffff", 0x000FFFFFFFFFFFFF, "2.225073858507201e-308"),
            ("b0010000000000000", 0x0010000000000000, "2.2250738585072014e-308"),
            ("b0010000000000001", 0x0010000000000001, "2.225073858507202e-308"),
            ("b7fefffffffffffff", 0x7FEFFFFFFFFFFFFF, "1.7976931348623157e+308"),
            ("b8000000000000001", 0x8000000000000001, "-5e-324"),
            ("b800fffffffffffff", 0x800FFFFFFFFFFFFF, "-2.225073858507201e-308"),
            ("b8010000000000000", 0x8010000000000000, "-2.2250738585072014e-308"),
            ("bffeffffffffffff", 0xFFEFFFFFFFFFFFFF, "-1.7976931348623157e+308"),
            ("b3ee4f8b588e368f0", 0x3EE4F8B588E368F0, "9.999999999999999e-06"),
            ("b3ee4f8b588e368f1", 0x3EE4F8B588E368F1, "1e-05"),
            ("b3ee4f8b588e368f2", 0x3EE4F8B588E368F2, "1.0000000000000003e-05"),
            ("bbee4f8b588e368f0", 0xBEE4F8B588E368F0, "-9.999999999999999e-06"),
            ("bbee4f8b588e368f1", 0xBEE4F8B588E368F1, "-1e-05"),
            ("bbee4f8b588e368f2", 0xBEE4F8B588E368F2, "-1.0000000000000003e-05"),
            ("b3f1a36e2eb1c432c", 0x3F1A36E2EB1C432C, "9.999999999999999e-05"),
            ("b3f1a36e2eb1c432d", 0x3F1A36E2EB1C432D, "0.0001"),
            ("b3f1a36e2eb1c432e", 0x3F1A36E2EB1C432E, "0.00010000000000000002"),
            ("bbf1a36e2eb1c432c", 0xBF1A36E2EB1C432C, "-9.999999999999999e-05"),
            ("bbf1a36e2eb1c432d", 0xBF1A36E2EB1C432D, "-0.0001"),
            ("bbf1a36e2eb1c432e", 0xBF1A36E2EB1C432E, "-0.00010000000000000002"),
            ("b430c6bf52633ffff", 0x430C6BF52633FFFF, "999999999999999.9"),
            ("b430c6bf526340000", 0x430C6BF526340000, "1000000000000000.0"),
            ("b430c6bf526340001", 0x430C6BF526340001, "1000000000000000.1"),
            ("bc30c6bf52633ffff", 0xC30C6BF52633FFFF, "-999999999999999.9"),
            ("bc30c6bf526340000", 0xC30C6BF526340000, "-1000000000000000.0"),
            ("bc30c6bf526340001", 0xC30C6BF526340001, "-1000000000000000.1"),
            ("b4341c37937e07fff", 0x4341C37937E07FFF, "9999999999999998.0"),
            ("b4341c37937e08000", 0x4341C37937E08000, "1e+16"),
            ("b4341c37937e08001", 0x4341C37937E08001, "1.0000000000000002e+16"),
            ("bc341c37937e07fff", 0xC341C37937E07FFF, "-9999999999999998.0"),
            ("bc341c37937e08000", 0xC341C37937E08000, "-1e+16"),
            ("bc341c37937e08001", 0xC341C37937E08001, "-1.0000000000000002e+16"),
            ("b433fffffffffffff", 0x433FFFFFFFFFFFFF, "9007199254740991.0"),
            ("b4340000000000000", 0x4340000000000000, "9007199254740992.0"),
            ("b4340000000000001", 0x4340000000000001, "9007199254740994.0"),
            ("bc33fffffffffffff", 0xC33FFFFFFFFFFFFF, "-9007199254740991.0"),
            ("bc34000000000000", 0xC340000000000000, "-9007199254740992.0"),
            ("bc34000000000001", 0xC340000000000001, "-9007199254740994.0"),
            ("b4415af1d78b58c3f", 0x4415AF1D78B58C3F, "9.999999999999998e+19"),
            ("b4415af1d78b58c40", 0x4415AF1D78B58C40, "1e+20"),
            ("b4415af1d78b58c41", 0x4415AF1D78B58C41, "1.0000000000000002e+20"),
            ("bc415af1d78b58c3f", 0xC415AF1D78B58C3F, "-9.999999999999998e+19"),
            ("bc415af1d78b58c40", 0xC415AF1D78B58C40, "-1e+20"),
            ("bc415af1d78b58c41", 0xC415AF1D78B58C41, "-1.0000000000000002e+20"),
        ]
        #expect(records.count == 47)
        let arguments = records.map { name, bits, _ in
            ModelChatJSONMember(name, .number(Double(bitPattern: bits)))
        }
        let messages = [
            ModelChatMessage(role: .user, content: "numbers"),
            ModelChatMessage(role: .assistant, content: "", toolCalls: [
                ModelChatToolCall(name: "numbers", arguments: .object(arguments)),
            ]),
        ]
        let rendered = try codec.renderPrompt(messages: messages, tools: [], options: .init())
        for (name, _, expected) in records {
            #expect(rendered.contains("<parameter=\(name)>\n\(expected)\n</parameter>"))
        }
    }

    @Test("Nonfinite JSON numbers are rejected without rendering")
    func nonfiniteJSONNumbers() {
        for value in [Double.nan, Double.infinity, -Double.infinity] {
            let messages = [
                ModelChatMessage(role: .user, content: "numbers"),
                ModelChatMessage(role: .assistant, content: "", toolCalls: [
                    ModelChatToolCall(name: "numbers", arguments: .object([
                        ModelChatJSONMember("value", .number(value)),
                    ])),
                ]),
            ]
            do {
                _ = try codec.renderPrompt(messages: messages, tools: [], options: .init())
                Issue.record("nonfinite JSON number was rendered")
            } catch let error as QwenChatCodecError {
                #expect(error == .nonFiniteJSONNumber)
            } catch {
                Issue.record("unexpected error for nonfinite JSON number: \(error)")
            }
        }
    }

    @Test("Tool schema JSON preserves member order, Unicode, raw strings, and non-string values")
    func toolJSONRendering() throws {
        let parameters = ModelChatJSONValue.object([
            ModelChatJSONMember("type", .string("object")),
            ModelChatJSONMember("properties", .object([
                ModelChatJSONMember("raw", .object([
                    ModelChatJSONMember("type", .string("string")),
                    ModelChatJSONMember("description", .string("雪 <>&'")),
                ])),
                ModelChatJSONMember("flag", .object([ModelChatJSONMember("type", .string("boolean"))])),
                ModelChatJSONMember("count", .object([ModelChatJSONMember("type", .string("number"))])),
                ModelChatJSONMember("payload", .object([ModelChatJSONMember("type", .string("object"))])),
            ])),
            ModelChatJSONMember("required", .array([
                .string("raw"), .string("flag"), .string("count"), .string("payload"),
            ])),
        ])
        let tools = [
            ModelChatToolDefinition(function: ModelChatFunctionDefinition(
                name: "lookup_雪<&>'", description: "Line one\nLine two: 雪 <>&'", parameters: parameters)),
            ModelChatToolDefinition(function: ModelChatFunctionDefinition(
                name: "second", description: "Second tool",
                parameters: .object([
                    ModelChatJSONMember("type", .string("object")),
                    ModelChatJSONMember("properties", .object([])),
                ]))),
        ]
        let calls = [
            ModelChatToolCall(id: "host-call-ignored-1", name: "lookup_雪<&>'", arguments: .object([
                ModelChatJSONMember("raw", .string("true<&>'雪\nnext")),
                ModelChatJSONMember("flag", .bool(true)),
                ModelChatJSONMember("count", .number(1.5)),
                ModelChatJSONMember("payload", .object([
                    ModelChatJSONMember("z", .integer(2)),
                    ModelChatJSONMember("a", .string("雪<&>'")),
                ])),
            ])),
            ModelChatToolCall(id: "host-call-ignored-2", name: "second", arguments: .object([])),
        ]
        let messages = [
            ModelChatMessage(role: .system, content: "Use exact tools."),
            ModelChatMessage(role: .user, content: "Call both."),
            ModelChatMessage(role: .assistant, content: "", toolCalls: calls),
            ModelChatMessage(role: .tool, content: "first result", toolCallID: "mismatch-is-ignored", name: "also-ignored"),
            ModelChatMessage(role: .tool, content: "second result", toolCallID: "another-ignored", name: "also-ignored"),
        ]
        let options = ModelChatRenderOptions(addGenerationPrompt: true, enableThinking: false)
        let rendered = try codec.renderPrompt(messages: messages, tools: tools, options: options)
        #expect(rendered == QwenTemplateOracleVectors.toolCaseRenderedTextExact)
        #expect(try codec.encodePrompt(messages: messages, tools: tools, options: options) == QwenTemplateOracleVectors.toolCaseIDs)
    }

    private func errorMessage(_ operation: () throws -> Void) -> String {
        do {
            try operation()
            return "<no error>"
        } catch {
            return String(describing: error)
        }
    }

    @Test("Pinned refusal and precedence messages remain observable")
    func refusalMessages() {
        let cases: [(messages: [ModelChatMessage], expected: String)] = [
            ([], QwenTemplateOracleVectors.emptyHistoryError),
            ([ModelChatMessage(role: .user, content: "q"), ModelChatMessage(role: .system, content: "late")], QwenTemplateOracleVectors.laterSystemError),
            ([ModelChatMessage(role: .developer, content: "unsupported")], QwenTemplateOracleVectors.noUserQueryError),
            ([ModelChatMessage(role: .user, content: "real query"), ModelChatMessage(role: .developer, content: "unsupported")], QwenTemplateOracleVectors.unexpectedRoleError),
            ([ModelChatMessage(role: .user, content: "real query"), ModelChatMessage(role: .other("alien"), content: "unsupported")], QwenTemplateOracleVectors.unexpectedRoleError),
            ([ModelChatMessage(role: .user, content: "  <tool_response>x</tool_response>  ")], QwenTemplateOracleVectors.noUserQueryError),
        ]
        for item in cases {
            let message = errorMessage {
                _ = try codec.renderPrompt(messages: item.messages, tools: [], options: .init())
            }
            #expect(message.contains(item.expected), "unexpected error: \(message)")
        }
    }

    @Test("Unsupported structured content has pinned refusal semantics")
    func unsupportedContentRefusals() {
        let systemImage = [
            ModelChatMessage(role: .system, content: .parts([.image()])),
            ModelChatMessage(role: .user, content: "q"),
        ]
        let malformed = [ModelChatMessage(role: .user, content: .parts([.unsupported(type: "unknown")]))]
        let audio = [ModelChatMessage(role: .user, content: .parts([.audio()]))]
        for (messages, expected) in [(systemImage, QwenTemplateOracleVectors.systemImageError), (malformed, QwenTemplateOracleVectors.malformedContentError), (audio, QwenTemplateOracleVectors.unsupportedAudioError)] {
            let message = errorMessage {
                _ = try codec.renderPrompt(messages: messages, tools: [], options: .init())
            }
            #expect(message.contains(expected), "unexpected error: \(message)")
        }
    }

    @Test("Structured video content is refused by the Phase 14 codec")
    func videoContent() {
        let messages = [ModelChatMessage(role: .user, content: .parts([.video()]))]
        let message = errorMessage {
            _ = try codec.renderPrompt(messages: messages, tools: [], options: .init())
        }
        #expect(message.contains(QwenTemplateOracleVectors.unsupportedVideoError))
    }

    @Test("Model-neutral Gemma adapter delegates text-only prompts unchanged")
    func gemmaAdapterDelegatesTextOnly() async throws {
        let gemma = try await GFTokenizer.load()
        let messages = [
            ModelChatMessage(role: .system, content: "Be concise."),
            ModelChatMessage(role: .user, content: "Adapter parity check."),
        ]
        let adapter = GemmaChatCodecAdapter(tokenizer: gemma)
        let actual = try adapter.encodePrompt(
            messages: messages, tools: [], options: .init())
        let expected = gemma.encode(
            try gemma.applyChatTemplate([
                GFTokenizer.Message(role: .system, content: "Be concise."),
                GFTokenizer.Message(role: .user, content: "Adapter parity check."),
            ]), addBOS: false)
        #expect(actual == expected)
    }

    @Test("Non-object tool arguments retain the pinned mapping error")
    func nonObjectToolArguments() {
        let messages = [
            ModelChatMessage(role: .user, content: "q"),
            ModelChatMessage(role: .assistant, content: "", toolCalls: [
                ModelChatToolCall(name: "bad", arguments: .array([.integer(1), .integer(2)])),
            ]),
        ]
        let message = errorMessage {
            _ = try codec.renderPrompt(messages: messages, tools: [], options: .init())
        }
        #expect(message.contains(QwenTemplateOracleVectors.nonObjectArgumentsError))
    }
}
