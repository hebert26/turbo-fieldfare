import TurboFieldfare

public enum CLIThinkingMode: String, Sendable, Equatable {
    case auto
    case on
    case off
}

public struct Args: Equatable, Sendable {
    public var model: String
    public var prompt: String?
    public var chatPrompt: String?
    public var messagesFile: String?
    public var images: [String]
    public var visionPack: String?
    public var visionResidency: VisionResidencyPolicy
    public var thinking: CLIThinkingMode
    public var toolsFile: String?
    public var showModelIdentity: Bool
    public var maxNew: Int
    public var maxContext: Int
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var seed: UInt64?
    public var stops: [String]
    public var quiet: Bool
    public var expertCacheSlots: Int
    public var expertCachePolicy: RuntimeExpertCachePolicy
    public var prefillPolicy: RuntimePrefillPolicy
    public var prefillChunkTokens: Int
    /// `--prefill-chunk-tokens auto`: the size is decided once the prompt length
    /// is known, which needs the tokenizer and, for images, their geometry.
    public var prefillChunkTokensAuto: Bool
    public var rdadvisePolicy: RDAdvicePolicyMode

    public init(model: String,
                prompt: String? = nil,
                chatPrompt: String? = nil,
                messagesFile: String? = nil,
                images: [String] = [],
                visionPack: String? = nil,
                visionResidency: VisionResidencyPolicy = .defaultPolicy,
                thinking: CLIThinkingMode = .auto,
                toolsFile: String? = nil,
                showModelIdentity: Bool = false,
                maxNew: Int = 1_024,
                maxContext: Int = 8192,
                temperature: Float = 0.2,
                topK: Int? = 64,
                topP: Float? = 0.95,
                repetitionPenalty: Float = 1.0,
                seed: UInt64? = nil,
                stops: [String] = [],
                quiet: Bool = false,
                expertCacheSlots: Int = RuntimeConfiguration.production.expertCacheSlots,
                expertCachePolicy: RuntimeExpertCachePolicy = RuntimeConfiguration.production.expertCachePolicy,
                prefillPolicy: RuntimePrefillPolicy = RuntimeConfiguration.production.prefillPolicy,
                prefillChunkTokens: Int = RuntimeConfiguration.production.prefillChunkTokens,
                prefillChunkTokensAuto: Bool = false,
                rdadvisePolicy: RDAdvicePolicyMode = RuntimeConfiguration.production.rdadvisePolicy) {
        self.model = model
        self.prompt = prompt
        self.chatPrompt = chatPrompt
        self.messagesFile = messagesFile
        self.images = images
        self.visionPack = visionPack
        self.visionResidency = visionResidency
        self.thinking = thinking
        self.toolsFile = toolsFile
        self.showModelIdentity = showModelIdentity
        self.maxNew = maxNew
        self.maxContext = maxContext
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.stops = stops
        self.quiet = quiet
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.prefillPolicy = prefillPolicy
        self.prefillChunkTokens = prefillChunkTokens
        self.prefillChunkTokensAuto = prefillChunkTokensAuto
        self.rdadvisePolicy = rdadvisePolicy
    }
}

public enum ArgsError: Error, Equatable, CustomStringConvertible {
    case helpRequested
    case unknownFlag(String)
    case missingValue(flag: String)
    case invalidValue(flag: String, value: String)
    case requiredMissing(String)
    case mutuallyExclusive(String, String)
    case modeMissing
    case unsupported(String)
    case imagePromptNeedsExpertCacheSlots(have: Int, need: Int)

    public var description: String {
        switch self {
        case .helpRequested: return "help requested"
        case .unknownFlag(let flag): return "unknown flag: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .invalidValue(let flag, let value): return "invalid value for \(flag): \(value)"
        case .requiredMissing(let flag): return "required flag missing: \(flag)"
        case .mutuallyExclusive(let a, let b): return "\(a) and \(b) are mutually exclusive"
        case .modeMissing:
            return "one of --prompt, --chat-prompt or --messages-file is required"
        case .unsupported(let detail): return detail
        case .imagePromptNeedsExpertCacheSlots(let have, let need):
            return "--image requires chunked prefill, which needs at least \(need) "
                + "expert-cache slots; --expert-cache-slots \(have) cannot serve it: "
                + "raise --expert-cache-slots to \(need) or more, or drop --image"
        }
    }
}

extension Args {
    public static let usage = """
    TurboFieldfareCLI — Gemma and verified Qwen generation

    usage: TurboFieldfareCLI --model <dir>
           (--prompt <string> | --chat-prompt <string> | --messages-file <path>) [options]

    required:
      --model <dir>             Path to a .gturbo model directory.
      --prompt <string>         Raw-completion prompt.
      --chat-prompt <string>    Single-turn instruction chat; pairs with --image
      --messages-file <path>    JSON chat messages with role and content fields.
      --image <path>            Attach a still image; repeatable. Needs --chat-prompt
                                and an installed companion pack.
      --vision-pack <dir>       Companion pack (default beside the model).
      --vision-residency <on-demand|keep-ready>
                                Routed-expert residency during vision (default on-demand).
      --video <path>            Unsupported. Video input is rejected before model load.

    options:
      --thinking <auto|on|off>  Chat thinking mode. Explicit on/off requires verified Qwen;
                                default auto preserves each family's existing behavior.
      --tools-file <path>       OpenAI-shaped Qwen function definitions for chat. Calls are
                                returned for the caller to approve; the CLI never runs them.
      --show-model-identity     Print loaded family, model, revision, and format to stderr.
      --max-new <int>            Generated-token limit (default 1024).
      --max-context <int>        Context limit in tokens (default 8192).
      --temperature <float>      Sampling temperature (default 0.2; 0 = greedy).
      --top-k <int>              Top-k truncation, 1...256 (default 64; 0 = off).
      --top-p <float>            Nucleus truncation (default 0.95).
      --repetition-penalty <f>   Repetition penalty (default 1.0).
      --seed <uint64>            Deterministic sampling seed (default off).
      --stop <string>            Stop substring (repeatable).
      --quiet                    Suppress the timing footer.
      --expert-cache-slots <n>   Expert-cache slots: 8, 16, 24, 32, 48, 64, 96, or 128
                                 (default 16).
      --expert-cache-policy <s>  Expert-cache policy: lfu or lru (default lfu).
      --prefill on|off           Enable or disable chunked prompt prefill (default on).
                                 Chunked prefill requires 16 or more cache slots.
      --prefill-chunk-tokens <n|auto>
                                 Prefill chunk size: 32, 64, 128, 256, or auto
                                 (default 128). Each chunk re-reads the routed
                                 expert pool, so larger chunks read less; auto
                                 picks the smallest size that covers the prompt.
      --rdadvise <s>             Read-advice policy: off, default, bounded, or adaptive (default off).
      --help                     Show this message.
    """

    public func resolvedRuntimeConfiguration(
        forceLogitsHead: Bool,
        imagePrompt: Bool = false) throws -> RuntimeConfiguration {
        // Images reach the runtime two ways, `--image` and `image_file` parts
        // inside `--messages-file`, and only the first fills `images`. Asking
        // the caller keeps the second from surviving validation and dying in
        // the routed tile scheduler after the model load and every GPU encode.
        guard !imagePrompt
                || expertCacheSlots
                >= RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill else {
            throw ArgsError.imagePromptNeedsExpertCacheSlots(
                have: expertCacheSlots,
                need: RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill)
        }
        guard RuntimeConfiguration.allowedExpertCacheSlots.contains(expertCacheSlots) else {
            throw ArgsError.invalidValue(
                flag: "--expert-cache-slots", value: "\(expertCacheSlots)")
        }
        guard RuntimeConfiguration.allowedPrefillChunkTokens.contains(prefillChunkTokens) else {
            throw ArgsError.invalidValue(
                flag: "--prefill-chunk-tokens", value: "\(prefillChunkTokens)")
        }
        let chunkedPrefillSupported = expertCacheSlots >=
            RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill
        guard prefillPolicy == .off || chunkedPrefillSupported else {
            throw ArgsError.invalidValue(
                flag: "--expert-cache-slots",
                value: "\(expertCacheSlots) requires --prefill off")
        }
        return RuntimeConfiguration(
            expertCacheSlots: expertCacheSlots,
            expertCachePolicy: expertCachePolicy,
            rdadvisePolicy: rdadvisePolicy,
            prefillEnabled: prefillPolicy == .chunked,
            prefillChunkTokens: prefillChunkTokens,
            forceLogitsHead: forceLogitsHead)
    }

    public static func parse(_ argv: [String]) throws -> Args {
        var model: String?
        var prompt: String?
        var chatPrompt: String?
        var messagesFile: String?
        var images: [String] = []
        var visionPack: String?
        var visionResidency: VisionResidencyPolicy = .defaultPolicy
        var thinking: CLIThinkingMode = .auto
        var toolsFile: String?
        var showModelIdentity = false
        var maxNew = 1_024
        var maxContext = 8192
        var temperature: Float = 0.2
        var topK: Int? = 64
        var topP: Float? = 0.95
        var repetitionPenalty: Float = 1.0
        var seed: UInt64?
        var stops: [String] = []
        var quiet = false
        let runtimeDefaults = RuntimeConfiguration.production
        var expertCacheSlots = runtimeDefaults.expertCacheSlots
        var expertCachePolicy = runtimeDefaults.expertCachePolicy
        var prefillPolicy = runtimeDefaults.prefillPolicy
        var prefillChunkTokens = runtimeDefaults.prefillChunkTokens
        var prefillChunkTokensAuto = false
        var rdadvisePolicy = runtimeDefaults.rdadvisePolicy

        var index = 0
        while index < argv.count {
            let flag = argv[index]
            switch flag {
            case "--help":
                throw ArgsError.helpRequested
            case "--quiet":
                quiet = true
                index += 1
            case "--show-model-identity":
                showModelIdentity = true
                index += 1
            case "--model":
                model = try takeValue(argv, &index, flag: flag)
            case "--prompt":
                prompt = try takeValue(argv, &index, flag: flag)
            case "--chat-prompt":
                chatPrompt = try takeValue(argv, &index, flag: flag)
            case "--image":
                images.append(try takeValue(argv, &index, flag: flag))
            case "--vision-pack":
                visionPack = try takeValue(argv, &index, flag: flag)
            case "--vision-residency":
                let value = try takeValue(argv, &index, flag: flag)
                guard let policy = VisionResidencyPolicy(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                visionResidency = policy
            case "--thinking":
                let value = try takeValue(argv, &index, flag: flag)
                guard let mode = CLIThinkingMode(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                thinking = mode
            case "--tools-file":
                toolsFile = try takeValue(argv, &index, flag: flag)
            case "--video":
                _ = try takeValue(argv, &index, flag: flag)
                throw ArgsError.unsupported("--video is not supported")
            case "--messages-file":
                messagesFile = try takeValue(argv, &index, flag: flag)
            case "--max-new":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                maxNew = parsed
            case "--max-context":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                maxContext = parsed
            case "--temperature":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed.isFinite, parsed >= 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                temperature = parsed
            case "--top-k":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), (0...256).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                topK = parsed == 0 ? nil : parsed
            case "--top-p":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed > 0, parsed <= 1 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                topP = parsed
            case "--repetition-penalty":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed.isFinite, parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                repetitionPenalty = parsed
            case "--seed":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = UInt64(value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                seed = parsed
            case "--stop":
                stops.append(try takeValue(argv, &index, flag: flag))
            case "--expert-cache-slots":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedExpertCacheSlots.contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                expertCacheSlots = parsed
            case "--expert-cache-policy":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = RuntimeExpertCachePolicy(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                expertCachePolicy = parsed
            case "--prefill":
                let value = try takeValue(argv, &index, flag: flag)
                switch value {
                case "on": prefillPolicy = .chunked
                case "off": prefillPolicy = .off
                default: throw ArgsError.invalidValue(flag: flag, value: value)
                }
            case "--prefill-chunk-tokens":
                let value = try takeValue(argv, &index, flag: flag)
                if value == "auto" {
                    prefillChunkTokensAuto = true
                    break
                }
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedPrefillChunkTokens.contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                prefillChunkTokensAuto = false
                prefillChunkTokens = parsed
            case "--rdadvise":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = RDAdvicePolicyMode(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                rdadvisePolicy = parsed
            default:
                throw ArgsError.unknownFlag(flag)
            }
        }

        guard let model else { throw ArgsError.requiredMissing("--model") }
        if prompt != nil && messagesFile != nil {
            throw ArgsError.mutuallyExclusive("--prompt", "--messages-file")
        }
        if prompt != nil, chatPrompt != nil {
            throw ArgsError.mutuallyExclusive("--prompt", "--chat-prompt")
        }
        if messagesFile != nil, chatPrompt != nil {
            throw ArgsError.mutuallyExclusive("--chat-prompt", "--messages-file")
        }
        if prompt != nil, !images.isEmpty {
            throw ArgsError.mutuallyExclusive("--prompt", "--image")
        }
        if prompt != nil, toolsFile != nil {
            throw ArgsError.mutuallyExclusive("--prompt", "--tools-file")
        }
        if prompt != nil, thinking != .auto {
            throw ArgsError.mutuallyExclusive("--prompt", "--thinking")
        }
        let minimumSlots = RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill
        if !images.isEmpty, expertCacheSlots < minimumSlots {
            throw ArgsError.imagePromptNeedsExpertCacheSlots(
                have: expertCacheSlots, need: minimumSlots)
        }
        if !images.isEmpty, chatPrompt == nil {
            // Naming the flag the user actually left out. Reporting a conflict
            // with `--messages-file` sent them looking for a flag they never
            // passed, in the most likely first-run mistake there is.
            throw messagesFile == nil
                ? ArgsError.requiredMissing("--chat-prompt")
                : ArgsError.mutuallyExclusive("--image", "--messages-file")
        }
        if prompt == nil && chatPrompt == nil && messagesFile == nil {
            throw ArgsError.modeMissing
        }
        if temperature > 0, topK == nil, let topP, topP < 1 {
            throw ArgsError.invalidValue(
                flag: "--top-p",
                value: "\(topP) requires --top-k between 1 and 256")
        }
        let arguments = Args(model: model,
                             prompt: prompt,
                             chatPrompt: chatPrompt,
                             messagesFile: messagesFile,
                             images: images,
                             visionPack: visionPack,
                             visionResidency: visionResidency,
                             thinking: thinking,
                             toolsFile: toolsFile,
                             showModelIdentity: showModelIdentity,
                             maxNew: maxNew,
                             maxContext: maxContext,
                             temperature: temperature,
                             topK: topK,
                             topP: topP,
                             repetitionPenalty: repetitionPenalty,
                             seed: seed,
                             stops: stops,
                             quiet: quiet,
                             expertCacheSlots: expertCacheSlots,
                             expertCachePolicy: expertCachePolicy,
                             prefillPolicy: prefillPolicy,
                             prefillChunkTokens: prefillChunkTokens,
                             prefillChunkTokensAuto: prefillChunkTokensAuto,
                             rdadvisePolicy: rdadvisePolicy)
        _ = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: false)
        return arguments
    }

    private static func takeValue(_ argv: [String],
                                  _ index: inout Int,
                                  flag: String) throws -> String {
        guard index + 1 < argv.count else { throw ArgsError.missingValue(flag: flag) }
        let value = argv[index + 1]
        index += 2
        return value
    }
}
