import Foundation
import TurboFieldfare

private final class VisionCaptureNoRedirectDelegate: NSObject,
    URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}

struct VisionCaptureMCPResult: Sendable {
    /// The already-decoded tools/call result. Keeping this structured avoids
    /// encoding a response of up to 4 MB and then decoding it repeatedly while
    /// the host extracts the manifest, target, session, and sanitized facts.
    let value: JSONValue
    let isError: Bool
    let refusalCode: String?
    let dispatchAttempted: Bool?
    let hasConflictingDispatchAttemptEvidence: Bool
    let isGuardedTargetRejectedBeforeSubmission: Bool
    let isStaleActionCapabilityBeforeDispatch: Bool
    let isSourceLayoutChangedBeforeRevalidation: Bool
    let isActionAuthorizationExpiredBeforeDispatch: Bool
    let isObservedTapTargetUnavailableBeforeDispatch: Bool
    let isDeliveredTransitionContinuation: Bool
    var isObservationTopologyRefreshBeforeDispatch: Bool = false

    /// Format the existing response directly into a bounded display excerpt.
    /// Never retain another payload tree or encode image bytes for the UI.
    var displayExcerpt: String {
        let maximumBytes = 8 * 1_024
        let maximumStringBytes = 1_024
        let maximumDepth = 8
        let maximumEntries = 64
        var remaining = maximumBytes
        var text = ""
        var truncated = false
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]

        func append(_ value: String) {
            let bytes = value.utf8
            if bytes.count > remaining { truncated = true }
            text += AppAgentActivity.displayPrefix(value, maximumUTF8Bytes: remaining)
            remaining -= min(bytes.count, remaining)
        }
        func quoted(_ value: String) -> String {
            let prefix = AppAgentActivity.displayPrefix(value, maximumUTF8Bytes: maximumStringBytes)
            let displayed: String
            if prefix.contains("base64,")
                || (value.utf8.count > maximumStringBytes
                    && !prefix.contains(where: \.isWhitespace)) {
                displayed = "[Encoded or oversized unbroken value omitted]"
            } else {
                displayed = prefix + (value.utf8.count > maximumStringBytes
                    ? " [String truncated at 1 KiB]" : "")
            }
            guard let data = try? encoder.encode(displayed) else {
                return "\"[Value unavailable for display]\""
            }
            return String(decoding: data, as: UTF8.self)
        }
        func render(_ value: JSONValue, depth: Int) {
            guard remaining > 0 else { truncated = true; return }
            guard depth < maximumDepth else {
                append("\"[Nested fields omitted at display depth limit]\"")
                return
            }
            let indent = String(repeating: "  ", count: depth)
            switch value {
            case .object(let object):
                if object["type"] == .string("image") || object["type"] == .string("audio") {
                    append("\"[Image/audio block omitted; use the attachment preview]\"")
                    return
                }
                let priority = ["isError", "payload", "proof", "interaction_evidence"]
                let keys = priority.filter { object[$0] != nil }
                    + object.keys.filter { !priority.contains($0) }.sorted()
                append("{")
                for (index, key) in keys.prefix(maximumEntries).enumerated() {
                    guard remaining > 0 else { truncated = true; break }
                    append((index == 0 ? "\n" : ",\n") + indent + "  " + quoted(key) + ": ")
                    if key == "content", object["payload"] != nil {
                        append("\"[MCP content blocks omitted; structured payload shown above]\"")
                    } else if ["data", "blob", "base64", "image_data", "image_base64"].contains(key) {
                        append("\"[Binary/data field omitted from display]\"")
                    } else if key == "text", case .string(let embedded)? = object[key],
                              let first = embedded.first(where: { !$0.isWhitespace }),
                              first == "{" || first == "[" {
                        append("\"[Embedded JSON text omitted from display]\"")
                    } else if let child = object[key] {
                        render(child, depth: depth + 1)
                    }
                }
                if keys.count > maximumEntries { append("\n" + indent + "[Additional fields omitted]") }
                append("\n" + indent + "}")
            case .array(let values):
                append("[")
                for (index, child) in values.prefix(maximumEntries).enumerated() {
                    guard remaining > 0 else { truncated = true; break }
                    append((index == 0 ? "\n" : ",\n") + indent + "  ")
                    render(child, depth: depth + 1)
                }
                if values.count > maximumEntries { append("\n" + indent + "[Additional items omitted]") }
                append("\n" + indent + "]")
            case .string(let value): append(quoted(value))
            default:
                if let data = try? encoder.encode(value) {
                    append(String(decoding: data, as: UTF8.self))
                } else { append("\"[Value unavailable for display]\"") }
            }
        }
        render(value, depth: 0)
        if truncated { text += "\n[Display truncated at 8 KiB.]" }
        return text
    }

    /// The public boundary enriches payload.proof.reason after producing any
    /// embedded content copies. Read its canonical proof, with root fallback,
    /// rather than treating an older text copy as the displayed server verdict.
    var serverOutcome: VisionCaptureServerOutcome {
        let root = value.objectValue
        let rawProof = root?["payload"]?.objectValue?["proof"] ?? root?["proof"]
        let proof = rawProof?.objectValue
        func boundedString(_ key: String, limit: Int) -> String? {
            guard case .string(let rawText)? = proof?[key] else { return nil }
            let text = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return nil }
            return text.count > limit ? String(text.prefix(limit)) + "…" : text
        }
        return VisionCaptureServerOutcome(
            verdict: boundedString("verdict", limit: 64),
            reason: boundedString("reason", limit: 1_024),
            reasonCode: boundedString("reason_code", limit: 160)
                ?? refusalCode.map { String($0.prefix(160)) },
            dispatchAttempted: dispatchAttempted)
    }

    var isRecoverableColdMiss: Bool {
        isError
            && refusalCode == "CACHE_COLD_MISS"
            && dispatchAttempted == false
    }

    var isSystemAlertDeliveryUnknown: Bool {
        refusalCode == "SYSTEM_ALERT_TAP_DELIVERY_UNKNOWN"
    }
}

enum VisionCaptureMCPError: Error, CustomStringConvertible {
    case invalidProtocol
    case executeUnavailable
    case rpc(String)

    var description: String {
        switch self {
        case .invalidProtocol:
            "VisionCapture returned an invalid MCP response."
        case .executeUnavailable:
            "VisionCapture did not publish exactly one execute tool."
        case .rpc(let message):
            "VisionCapture refused the MCP request: \(message)"
        }
    }
}

actor VisionCaptureMCPClient {
    private static let maximumRequestBytes = 1_000_000
    private static let maximumResponseBytes = 4 * 1_024 * 1_024
    private let endpoint: URL
    private let session: URLSession
    private var nextID: Int64 = 1
    private var isPrepared = false
    private var typedExecuteActions: Set<String>?

    init(port: Int) throws {
        guard (1...65_535).contains(port),
              let endpoint = URL(string: "http://127.0.0.1:\(port)/mcp") else {
            throw VisionCaptureMCPError.invalidProtocol
        }
        self.endpoint = endpoint
        self.session = URLSession(
            configuration: .ephemeral,
            delegate: VisionCaptureNoRedirectDelegate(),
            delegateQueue: nil)
    }

    func prepare() async throws {
        guard !isPrepared else { return }
        let initialization = try await send(
            method: "initialize",
            params: .object([
                "protocolVersion": .string("2024-11-05"),
                "capabilities": .object([:]),
                "clientInfo": .object([
                    "name": .string("TurboFieldfare Agent Mode"),
                    "version": .string("0.1"),
                ]),
            ]))
        guard case .object(let initializationObject) = initialization,
              case .string(let protocolVersion)? = initializationObject["protocolVersion"],
              !protocolVersion.isEmpty else {
            throw VisionCaptureMCPError.invalidProtocol
        }

        let listing = try await send(method: "tools/list", params: .object([:]))
        guard case .object(let listingObject) = listing,
              case .array(let tools)? = listingObject["tools"] else {
            throw VisionCaptureMCPError.invalidProtocol
        }
        let names = tools.compactMap { tool -> String? in
            guard case .object(let object) = tool,
                  case .string(let name)? = object["name"] else { return nil }
            return name
        }
        guard names == ["execute"] else {
            throw VisionCaptureMCPError.executeUnavailable
        }
        if case .object(let tool)? = tools.first,
           case .object(let schema)? = tool["inputSchema"],
           case .array(let required)? = schema["required"],
           required.contains(.string("action")) {
            guard case .object(let properties)? = schema["properties"],
                  case .object(let action)? = properties["action"],
                  case .array(let values)? = action["enum"] else {
                throw VisionCaptureMCPError.invalidProtocol
            }
            let actions = values.compactMap { value -> String? in
                guard case .string(let name) = value else { return nil }
                return name
            }
            guard !actions.isEmpty, actions.count == values.count else {
                throw VisionCaptureMCPError.invalidProtocol
            }
            typedExecuteActions = Set(actions)
        }
        isPrepared = true
    }

    /// Adapt only the outbound contract. Host proof and identity checks keep the
    /// original request, while the server resolves the session kind from its ID.
    static func typedExecuteArguments(
        _ arguments: JSONValue,
        supportedActions: Set<String>
    ) throws -> JSONValue {
        func refused(_ reason: String) -> VisionCaptureMCPError {
            .rpc("Unsupported typed execute request: \(reason). No action was sent.")
        }
        guard case .object(var object) = arguments,
              case .string(let request)? = object["request"],
              case .object(var parameters)? = object["parameters"] else {
            throw refused("request and parameters are required")
        }
        let allowedKeys: Set<String> = [
            "request", "bundle_id", "parameters", "session_id", "session_kind", "mode",
        ]
        guard Set(object.keys).isSubset(of: allowedKeys) else {
            throw refused("unexpected top-level field")
        }
        if let kind = object["session_kind"] {
            guard kind == .string("flow") || kind == .string("learning"),
                  case .string(let session)? = object["session_id"],
                  !session.isEmpty else {
                throw refused("session kind is not bound to a known session")
            }
            object.removeValue(forKey: "session_kind")
        }
        let fixed: [String: String] = [
            "launch app": "launch_app",
            "describe screen": "describe_screen",
            "take a screenshot": "take_screenshot",
            "inspect cache": "inspect_cache",
            "tap cached action": "tap_cached_action",
            "execute cached action": "execute_cached_action",
            "revalidate cached action": "revalidate_cached_action",
            "execute observed action": "execute_observed_action",
            "describe system alert": "describe_system_alert",
            "press system alert button": "press_system_alert_button",
            "go back": "go_back",
            "tap coordinates": "tap_coordinates",
            "activate computer use": "activate_computer_use",
            "click pointer": "click_pointer",
            "hide pointer": "hide_pointer",
        ]
        let action: String
        if let mapped = fixed[request] {
            action = mapped
        } else {
            let forms = [("tap ", "tap", "query"),
                         ("type ", "type", "text"),
                         ("swipe ", "swipe", "direction")]
            guard let form = forms.first(where: { request.hasPrefix($0.0) }) else {
                throw refused("unknown operation")
            }
            let value = String(request.dropFirst(form.0.count))
            guard !value.isEmpty,
                  form.1 != "tap" || !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  form.1 != "swipe" || ["up", "down", "left", "right"].contains(value),
                  parameters[form.2] == nil else {
                throw refused("empty or conflicting embedded operation value")
            }
            action = form.1
            parameters[form.2] = .string(value)
        }
        guard supportedActions.contains(action) else {
            throw refused("operation is not advertised by the server")
        }
        object.removeValue(forKey: "request")
        object["action"] = .string(action)
        object["parameters"] = .object(parameters)
        return .object(object)
    }

    func execute(
        arguments: JSONValue,
        beforeDispatch: @Sendable () async throws -> Void = {}
    ) async throws -> VisionCaptureMCPResult {
        try await prepare()
        let wireArguments = try typedExecuteActions.map {
            try Self.typedExecuteArguments(arguments, supportedActions: $0)
        } ?? arguments
        let isScreenshot: Bool
        if case .object(let request) = arguments {
            isScreenshot = request["request"] == .string("take a screenshot")
        } else {
            isScreenshot = false
        }
        // This callback is the final admission point. No suspension occurs
        // between its return and entering `send`, so a user instruction that
        // wins this boundary prevents the request from reaching VisionCapture.
        try await beforeDispatch()
        let result = try await send(
            method: "tools/call",
            params: .object([
                "name": .string("execute"),
                "arguments": wireArguments,
            ]),
            responseLimit: isScreenshot ? VisionCaptureScreenshot.maximumResponseBytes
                : Self.maximumResponseBytes)
        guard case .object(let object) = result else {
            throw VisionCaptureMCPError.invalidProtocol
        }
        let isError = object["isError"] == .bool(true)
        let dispatchAttempts = Self.findBooleanValues(
            named: "dispatch_attempted",
            in: result)
        let refusalCode = Self.findRefusalCode(in: result)
        return VisionCaptureMCPResult(
            value: result,
            isError: isError,
            refusalCode: refusalCode,
            dispatchAttempted: dispatchAttempts.count == 1
                ? dispatchAttempts.first : nil,
            hasConflictingDispatchAttemptEvidence: dispatchAttempts.count > 1,
            isGuardedTargetRejectedBeforeSubmission:
                isError
                && ["GUARDED_TARGET_REJECTED", "NO_BACK_TARGET"].contains(refusalCode)
                && !dispatchAttempts.contains(true)
                && dispatchAttempts.count <= 1
                && Self.provesGuardedTargetRejectionBeforeSubmission(in: result),
            isStaleActionCapabilityBeforeDispatch:
                isError
                && refusalCode == "CACHE_ACTION_CAPABILITY_STALE"
                && dispatchAttempts == [false],
            isSourceLayoutChangedBeforeRevalidation:
                isError
                && refusalCode == "CACHE_REVALIDATION_CAPABILITY_STALE"
                && Self.provesSourceLayoutChangedBeforeRevalidation(
                    in: result, arguments: arguments),
            isActionAuthorizationExpiredBeforeDispatch:
                isError
                && refusalCode == "CACHE_ACTION_CAPABILITY_INVALID"
                && Self.provesActionAuthorizationExpiredBeforeDispatch(
                    in: result, arguments: arguments),
            isObservedTapTargetUnavailableBeforeDispatch:
                isError
                && refusalCode == "CACHE_ACTION_CAPABILITY_INVALID"
                && dispatchAttempts == [false]
                && Self.provesObservedTapTargetUnavailableBeforeDispatch(
                    in: result, arguments: arguments),
            isDeliveredTransitionContinuation:
                isError
                && refusalCode == "DISCOVERY_SCREEN_TRANSITION_CONTRADICTED"
                && Self.provesDeliveredTransitionContinuation(
                    in: result, arguments: arguments),
            isObservationTopologyRefreshBeforeDispatch:
                isError && refusalCode == "CACHE_ACTION_CAPABILITY_STALE"
                && dispatchAttempts == [false]
                && Self.provesObservationTopologyRefresh(in: result, arguments: arguments))
    }

    /// A complete retired-grant refusal permits reads only. Missing copies,
    /// unexpected input evidence, or conflicting target/session facts fail closed.
    private static func provesObservationTopologyRefresh(
        in value: JSONValue, arguments: JSONValue
    ) -> Bool {
        let code = "CACHE_ACTION_CAPABILITY_STALE"
        let expected: [String: JSONValue] = [
            "error": .string(code),
            "cache_authorization_phase": .string("core_observation_completion"),
            "cache_authorization_reason": .string("live_screen_binding_mismatch"),
            "recovery_action": .string("observe_and_inspect"),
            "recovery_reason": .string("observation_topology_changed_before_completion"),
            "observation_grant_retired": .bool(true),
            "cache_used": .bool(false), "cache_revalidation_used": .bool(false),
            "revalidation_verified": .bool(false), "fresh_authority_recorded": .bool(false),
            "dispatch_attempted": .bool(false),
        ]
        guard let request = arguments.objectValue,
              request["request"] == .string("describe screen"),
              case .string(let bundle)? = request["bundle_id"], !bundle.isEmpty,
              case .string(let session)? = request["session_id"], !session.isEmpty,
              request["session_kind"] == .string("flow"),
              let parameters = request["parameters"]?.objectValue,
              (Set(parameters.keys) == ["udid", "observation_grant"]
                || (Set(parameters.keys) == ["udid", "observation_grant", "describe"]
                    && parameters["describe"] == .object([
                        "include_values": .bool(true),
                        "redaction": .string("balanced"),
                    ]))),
              case .string(let udid)? = parameters["udid"], !udid.isEmpty,
              case .string(let grant)? = parameters["observation_grant"], !grant.isEmpty,
              let root = value.objectValue, root["isError"] == .bool(true),
              expected.allSatisfy({ root[$0.key] == $0.value }),
              let payload = root["payload"]?.objectValue,
              expected.allSatisfy({ payload[$0.key] == $0.value }),
              case .array(let content)? = root["content"],
              content.contains(where: { block in
                  guard case .string(let text)? = block.objectValue?["text"] else { return false }
                  return embeddedJSONValues(in: text).contains { copy in
                      guard let object = copy.objectValue else { return false }
                      return expected.allSatisfy { object[$0.key] == $0.value }
                  }
              }) else { return false }
        var inspectedTexts: Set<String> = []
        func isConsistent(_ value: JSONValue) -> Bool {
            switch value {
            case .object(let object):
                for key in ["proof", "dispatch", "interaction_evidence", "system_alert", "delivery",
                            "delivery_status", "dispatch_status", "continuation_action", "continuation_reason",
                            "success", "verdict", "status", "outcome", "observation_grant", "action_capability"] {
                    if object[key] != nil { return false }
                }
                if let returned = object["isError"], returned != .bool(true) { return false }
                if let returned = object["is_error"], returned != .bool(true) { return false }
                if let returned = object["request"], returned != request["request"] { return false }
                for (key, expectedValue) in expected {
                    if let returned = object[key], returned != expectedValue { return false }
                }
                if object["recovery_action"] != nil || object["recovery_reason"] != nil
                    || object["observation_grant_retired"] != nil {
                    guard expected.allSatisfy({ object[$0.key] == $0.value }) else { return false }
                }
                for key in ["submission_started", "delivery_acknowledged", "mutation_sent"] {
                    if let returned = object[key], returned != .bool(false) { return false }
                }
                for key in ["reason_code", "code", "error_code", "refusal_code"] {
                    if let returned = object[key], returned != .string(code) { return false }
                }
                for key in ["bundle_id", "requested_bundle_id", "observed_bundle_id",
                            "bundle_id_requested", "bundle_id_active"] {
                    if let returned = object[key], returned != request["bundle_id"] { return false }
                }
                for key in ["udid", "requested_udid", "bound_udid"] {
                    if let returned = object[key], returned != parameters["udid"] { return false }
                }
                if object["session_id"] != nil || object["session_kind"] != nil {
                    guard object["session_id"] == request["session_id"],
                          object["session_kind"] == request["session_kind"] else { return false }
                }
                return object.values.allSatisfy(isConsistent)
            case .array(let array): return array.allSatisfy(isConsistent)
            case .string(let text):
                if text.hasPrefix("Error ["), !text.hasPrefix("Error [\(code)]:") { return false }
                guard inspectedTexts.insert(text).inserted else { return true }
                return embeddedJSONValues(in: text).allSatisfy(isConsistent)
            default: return true
            }
        }
        return isConsistent(value)
    }

    /// A granted observed tap can become invalid when the live target changes
    /// before dispatch. Accept only VisionCapture's exact no-dispatch contract;
    /// generic invalid-capability failures remain terminal.
    private static func provesObservedTapTargetUnavailableBeforeDispatch(
        in value: JSONValue, arguments: JSONValue
    ) -> Bool {
        guard let request = arguments.objectValue,
              case .string(let rawRequest)? = request["request"],
              rawRequest.hasPrefix("tap "),
              !rawRequest.dropFirst(4).isEmpty,
              case .string(let bundle)? = request["bundle_id"], !bundle.isEmpty,
              case .string(let session)? = request["session_id"], !session.isEmpty,
              request["session_kind"] == .string("flow"),
              let parameters = request["parameters"]?.objectValue,
              Set(parameters.keys) == ["udid", "observation_grant"],
              case .string(let udid)? = parameters["udid"], !udid.isEmpty,
              case .string(let grant)? = parameters["observation_grant"], !grant.isEmpty,
              let root = value.objectValue,
              root["isError"] == .bool(true),
              root["cache_authorization_phase"] == .string("cold_tap_live_validation"),
              root["cache_authorization_reason"] == .string("live_target_not_actionable"),
              root["dispatch_attempted"] == .bool(false),
              let payload = root["payload"]?.objectValue,
              let proof = payload["proof"]?.objectValue,
              proof["verdict"] == .string("failed"),
              proof["action"] == .string("tap"),
              proof["target"] == .string(String(rawRequest.dropFirst(4))),
              proof["checks"] == .object([:]),
              case .string(let reason)? = proof["reason"], !reason.isEmpty
        else { return false }

        for key in ["submission_started", "delivery_acknowledged", "mutation_sent"] {
            if findBooleanValues(named: key, in: value).contains(true) { return false }
        }
        return findBooleanValues(named: "dispatch_attempted", in: value) == [false]
    }

    /// Only the publisher's complete expired-unused action contract permits
    /// observation. Generic invalid capabilities never enter this path.
    private static func provesActionAuthorizationExpiredBeforeDispatch(
        in value: JSONValue, arguments: JSONValue
    ) -> Bool {
        let code = "CACHE_ACTION_CAPABILITY_INVALID"
        let expected: [String: JSONValue] = [
            "error": .string(code),
            "recovery_action": .string("observe_and_inspect"),
            "recovery_reason": .string("authorization_expired_before_dispatch"),
            "cache_used": .bool(false), "cache_revalidation_used": .bool(false),
            "revalidation_verified": .bool(false), "fresh_authority_recorded": .bool(false),
            "dispatch_attempted": .bool(false),
        ]
        guard let request = arguments.objectValue,
              request["request"] == .string("tap cached action"),
              let parameters = request["parameters"]?.objectValue,
              Set(parameters.keys) == ["udid", "action_capability"],
              case .string(let capability)? = parameters["action_capability"], !capability.isEmpty,
              let root = value.objectValue, root["isError"] == .bool(true),
              expected.allSatisfy({ root[$0.key] == $0.value }),
              let payload = root["payload"]?.objectValue,
              expected.allSatisfy({ payload[$0.key] == $0.value }),
              let proof = payload["proof"]?.objectValue,
              Set(proof.keys) == ["verdict", "reason", "action", "target", "checks"],
              proof["verdict"] == .string("failed"),
              proof["action"] == .string("tap"), proof["target"] == .string("cached action"),
              proof["checks"] == .object([:]),
              case .string(let reason)? = proof["reason"], !reason.isEmpty,
              case .array(let content)? = root["content"],
              content.contains(where: { block in
                  guard case .string(let text)? = block.objectValue?["text"] else { return false }
                  return embeddedJSONValues(in: text).contains { copy in
                      guard let object = copy.objectValue else { return false }
                      return expected.allSatisfy { object[$0.key] == $0.value }
                  }
              }) else { return false }
        var inspectedTexts: Set<String> = []
        func isConsistent(_ value: JSONValue) -> Bool {
            switch value {
            case .object(let object):
                for key in ["dispatch", "interaction_evidence", "system_alert", "delivery",
                            "delivery_status", "dispatch_status", "continuation_action", "continuation_reason"] {
                    if object[key] != nil { return false }
                }
                if let returned = object["proof"], returned != .object(proof) { return false }
                if let returned = object["isError"], returned != .bool(true) { return false }
                for (key, expectedValue) in expected {
                    if let returned = object[key], returned != expectedValue { return false }
                }
                if object["recovery_action"] != nil || object["recovery_reason"] != nil {
                    guard expected.allSatisfy({ object[$0.key] == $0.value }) else { return false }
                }
                for key in ["submission_started", "delivery_acknowledged", "mutation_sent"] {
                    if let returned = object[key], returned != .bool(false) { return false }
                }
                for key in ["reason_code", "code", "error_code", "refusal_code"] {
                    if let returned = object[key], returned != .string(code) { return false }
                }
                for key in ["bundle_id", "requested_bundle_id", "observed_bundle_id",
                            "bundle_id_requested", "bundle_id_active"] {
                    if let returned = object[key], returned != request["bundle_id"] { return false }
                }
                for key in ["udid", "requested_udid", "bound_udid"] {
                    if let returned = object[key], returned != parameters["udid"] { return false }
                }
                if object["session_id"] != nil || object["session_kind"] != nil {
                    guard object["session_id"] != nil, object["session_kind"] != nil,
                          object["session_id"] == request["session_id"],
                          object["session_kind"] == request["session_kind"] else { return false }
                }
                return object.values.allSatisfy(isConsistent)
            case .array(let array): return array.allSatisfy(isConsistent)
            case .string(let text):
                if text.hasPrefix("Error ["), !text.hasPrefix("Error [\(code)]:") { return false }
                guard inspectedTexts.insert(text).inserted else { return true }
                return embeddedJSONValues(in: text).allSatisfy(isConsistent)
            default: return true
            }
        }
        return isConsistent(value)
    }

    /// A delivered failed transition may permit observation of its new screen.
    /// This contract never means the action succeeded or may be replayed.
    private static func provesDeliveredTransitionContinuation(
        in value: JSONValue, arguments: JSONValue
    ) -> Bool {
        let code = "DISCOVERY_SCREEN_TRANSITION_CONTRADICTED"
        guard let request = arguments.objectValue,
              let parameters = request["parameters"]?.objectValue else { return false }
        let isWarmTap = request["request"] == .string("tap cached action")
        guard isWarmTap || request["request"] == .string("revalidate cached action") else { return false }
        let expected: [String: JSONValue] = [
            "continuation_action": .string("observe_and_inspect"),
            "continuation_reason": .string("delivered_transition_contradicted"),
            "cache_used": .bool(isWarmTap), "cache_revalidation_used": .bool(!isWarmTap),
            "revalidation_verified": .bool(false), "fresh_authority_recorded": .bool(false),
            "dispatch_attempted": .bool(true),
        ]
        guard let root = value.objectValue,
              expected.allSatisfy({ root[$0.key] == $0.value }),
              let payload = root["payload"]?.objectValue,
              let proof = payload["proof"]?.objectValue,
              proof["verdict"] == .string("failed"),
              proof["reason_code"] == .string(code),
              proof["verdict_source"] == .string("interaction_truth_judge"),
              let evidence = payload["interaction_evidence"]?.objectValue,
              let binding = evidence["binding"]?.objectValue,
              binding["udid"] == parameters["udid"],
              binding["requested_bundle_id"] == request["bundle_id"],
              binding["observed_bundle_id"] == request["bundle_id"],
              case .integer(let pid)? = binding["observed_pid"], pid > 0,
              let dispatch = evidence["dispatch"]?.objectValue,
              dispatch["status"] == .string("acknowledged_by_driver"),
              dispatch["submission_started"] == .bool(true),
              dispatch["delivery_acknowledged"] == .bool(true),
              let outcome = evidence["outcome"]?.objectValue,
              outcome["status"] == .string("failed"),
              outcome["reason_code"] == .string(code),
              outcome["scope"] == .string("discovery_screen_transition"),
              let observations = outcome["observations"]?.objectValue,
              let before = observations["before"]?.objectValue,
              case .array(let after)? = observations["after"], after.count >= 2,
              let terminal = after.last?.objectValue,
              case .string(let topology)? = terminal["topology_digest"], !topology.isEmpty,
              case .string(let layout)? = terminal["layout_digest"], !layout.isEmpty,
              case .string(let sourceTopology)? = before["topology_digest"],
              !sourceTopology.isEmpty, topology != sourceTopology,
              after.allSatisfy({
                  $0.objectValue?["topology_digest"] == .string(topology)
                      && $0.objectValue?["layout_digest"] == .string(layout)
              }) else { return false }
        var inspectedTexts: Set<String> = []
        func isConsistent(_ value: JSONValue) -> Bool {
            switch value {
            case .object(let object):
                if object["system_alert"] != nil || object["recovery_action"] != nil
                    || object["revalidation_evidence_code"] != nil
                    || object["fresh_authority_error_code"] != nil { return false }
                if let isError = object["isError"], isError != .bool(true) { return false }
                for (key, expectedValue) in expected {
                    if let returned = object[key], returned != expectedValue { return false }
                }
                if object["continuation_action"] != nil || object["continuation_reason"] != nil {
                    guard expected.allSatisfy({ object[$0.key] == $0.value }) else { return false }
                }
                for key in ["error", "error_code", "refusal_code", "code"] {
                    if let returned = object[key], returned != .string(code) { return false }
                }
                if let returned = object["dispatch"], returned != .object(dispatch) { return false }
                if let returned = object["interaction_evidence"], returned != .object(evidence) { return false }
                if let returned = object["proof"] {
                    guard let proof = returned.objectValue,
                          proof["verdict"] == .string("failed"),
                          proof["reason_code"] == .string(code),
                          proof["verdict_source"] == .string("interaction_truth_judge") else { return false }
                }
                for key in ["bundle_id", "requested_bundle_id", "observed_bundle_id",
                            "bundle_id_requested", "bundle_id_active"] {
                    if let returned = object[key], returned != request["bundle_id"] { return false }
                }
                for key in ["udid", "requested_udid", "bound_udid"] {
                    if let returned = object[key], returned != parameters["udid"] { return false }
                }
                if object["session_id"] != nil || object["session_kind"] != nil {
                    guard object["session_id"] != nil, object["session_kind"] != nil,
                          object["session_id"] == request["session_id"],
                          object["session_kind"] == request["session_kind"] else { return false }
                }
                return object.values.allSatisfy(isConsistent)
            case .array(let array): return array.allSatisfy(isConsistent)
            case .string(let text):
                if text.hasPrefix("Error ["), !text.hasPrefix("Error [\(code)]:") { return false }
                guard inspectedTexts.insert(text).inserted else { return true }
                return embeddedJSONValues(in: text).allSatisfy(isConsistent)
            default: return true
            }
        }
        return isConsistent(value)
    }

    /// Only the explicit pre-claim layout refusal permits a fresh observation.
    /// The publisher repeats its envelope in payload and JSON text. Require
    /// every copy and every named field to agree, including their JSON types.
    private static func provesSourceLayoutChangedBeforeRevalidation(
        in value: JSONValue,
        arguments: JSONValue
    ) -> Bool {
        let expected: [String: JSONValue] = [
            "recovery_action": .string("observe_and_inspect"),
            "recovery_reason": .string("source_layout_changed_before_claim"),
            "cache_used": .bool(false),
            "cache_revalidation_used": .bool(false),
            "revalidation_verified": .bool(false),
            "fresh_authority_recorded": .bool(false),
            "dispatch_attempted": .bool(false),
        ]
        guard let request = arguments.objectValue,
              let parameters = request["parameters"]?.objectValue,
              let root = value.objectValue,
              expected.allSatisfy({ root[$0.key] == $0.value }) else { return false }
        var inspectedTexts: Set<String> = []
        func isConsistent(_ value: JSONValue) -> Bool {
            switch value {
            case .object(let object):
                // This narrow refusal has no delivery or native-alert result.
                // Their presence would make the pre-dispatch claim ambiguous.
                if object["dispatch"] != nil || object["system_alert"] != nil
                    || object["proof"] != nil { return false }
                if let isError = object["isError"], isError != .bool(true) { return false }
                for key in ["submission_started", "delivery_acknowledged", "mutation_sent"] {
                    if let returned = object[key], returned != .bool(false) { return false }
                }
                for key in ["bundle_id", "requested_bundle_id", "observed_bundle_id",
                            "bundle_id_requested", "bundle_id_active"] {
                    if let returned = object[key], returned != request["bundle_id"] { return false }
                }
                for key in ["udid", "requested_udid", "bound_udid"] {
                    if let returned = object[key], returned != parameters["udid"] { return false }
                }
                if object["session_id"] != nil || object["session_kind"] != nil {
                    guard object["session_id"] != nil, object["session_kind"] != nil,
                          object["session_id"] == request["session_id"],
                          object["session_kind"] == request["session_kind"] else { return false }
                }
                for (key, expectedValue) in expected {
                    if let returned = object[key], returned != expectedValue { return false }
                }
                if object["recovery_action"] != nil || object["recovery_reason"] != nil {
                    guard expected.allSatisfy({ object[$0.key] == $0.value }) else { return false }
                }
                for key in ["reason_code", "code", "error_code", "refusal_code", "error"] {
                    if case .string(let code)? = object[key], isPublicErrorCode(code),
                       code != "CACHE_REVALIDATION_CAPABILITY_STALE" { return false }
                }
                return object.values.allSatisfy(isConsistent)
            case .array(let array):
                return array.allSatisfy(isConsistent)
            case .string(let text):
                if text.hasPrefix("Error ["),
                   !text.hasPrefix("Error [CACHE_REVALIDATION_CAPABILITY_STALE]:") { return false }
                guard inspectedTexts.insert(text).inserted else { return true }
                return embeddedJSONValues(in: text).allSatisfy(isConsistent)
            default:
                return true
            }
        }
        return isConsistent(value)
    }

    private func send(method: String, params: JSONValue,
                      responseLimit: Int = maximumResponseBytes) async throws -> JSONValue {
        try Task.checkCancellation()
        let id = nextID
        nextID += 1
        let request: JSONValue = .object([
            "jsonrpc": .string("2.0"),
            "id": .integer(id),
            "method": .string(method),
            "params": params,
        ])
        let body = try JSONEncoder().encode(request)
        guard body.count <= Self.maximumRequestBytes else {
            throw VisionCaptureMCPError.rpc("request exceeded the local size limit")
        }
        var urlRequest = URLRequest(url: endpoint)
        urlRequest.httpMethod = "POST"
        urlRequest.timeoutInterval = 120
        urlRequest.httpBody = body
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (bytes, response) = try await session.bytes(for: urlRequest)
        defer { bytes.task.cancel() }
        guard let httpResponse = response as? HTTPURLResponse,
              (200..<300).contains(httpResponse.statusCode),
              httpResponse.url == endpoint,
              response.expectedContentLength <= Int64(responseLimit) else {
            throw VisionCaptureMCPError.invalidProtocol
        }
        var responseData = Data()
        for try await byte in bytes {
            if responseData.count % (64 * 1_024) == 0 { try Task.checkCancellation() }
            guard responseData.count < responseLimit else {
                throw VisionCaptureMCPError.invalidProtocol
            }
            responseData.append(byte)
        }
        let envelope = try JSONDecoder().decode(JSONValue.self, from: responseData)
        guard case .object(let object) = envelope,
              object["jsonrpc"] == .string("2.0"),
              object["id"] == .integer(id) else {
            throw VisionCaptureMCPError.invalidProtocol
        }
        if case .object(let error)? = object["error"] {
            let code: String
            switch error["code"] {
            case .integer(let value): code = String(value)
            case .string(let value): code = value
            default: code = "unknown"
            }
            let message: String
            if case .string(let value)? = error["message"] {
                message = String(value.prefix(300))
            } else {
                message = "JSON-RPC error \(code)"
            }
            throw VisionCaptureMCPError.rpc("\(code): \(message)")
        }
        guard let result = object["result"] else {
            throw VisionCaptureMCPError.invalidProtocol
        }
        return result
    }

    private static func findRefusalCode(in value: JSONValue) -> String? {
        var inspectedEmbeddedTexts: Set<String> = []
        return findRefusalCode(
            in: value,
            inspectedEmbeddedTexts: &inspectedEmbeddedTexts)
    }

    private static func findRefusalCode(
        in value: JSONValue,
        inspectedEmbeddedTexts: inout Set<String>
    ) -> String? {
        switch value {
        case .object(let object):
            for key in ["reason_code", "code", "error_code", "refusal_code", "error"] {
                if case .string(let code)? = object[key], isPublicErrorCode(code) {
                    return code
                }
            }
            for child in object.values {
                if let code = findRefusalCode(
                    in: child,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts
                ) { return code }
            }
        case .array(let array):
            for child in array {
                if let code = findRefusalCode(
                    in: child,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts
                ) { return code }
            }
        case .string(let text):
            let prefix = "Error ["
            if text.hasPrefix(prefix),
               let close = text[text.index(
                text.startIndex,
                offsetBy: prefix.count)...].firstIndex(of: "]") {
                let start = text.index(text.startIndex, offsetBy: prefix.count)
                let code = String(text[start..<close])
                if isPublicErrorCode(code) {
                    return code
                }
            }
            guard inspectedEmbeddedTexts.insert(text).inserted else {
                return nil
            }
            for embedded in embeddedJSONValues(in: text) {
                if let code = findRefusalCode(
                    in: embedded,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts
                ) { return code }
            }
        default:
            break
        }
        return nil
    }

    private static func isPublicErrorCode(_ value: String) -> Bool {
        let allowed = CharacterSet(
            charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
        return !value.isEmpty
            && value.unicodeScalars.allSatisfy(allowed.contains)
    }

    private static func findBooleanValues(
        named key: String,
        in value: JSONValue
    ) -> Set<Bool> {
        var inspectedEmbeddedTexts: Set<String> = []
        return findBooleanValues(
            named: key,
            in: value,
            inspectedEmbeddedTexts: &inspectedEmbeddedTexts)
    }

    private static func findBooleanValues(
        named key: String,
        in value: JSONValue,
        inspectedEmbeddedTexts: inout Set<String>
    ) -> Set<Bool> {
        switch value {
        case .object(let object):
            var values: Set<Bool> = []
            if case .bool(let value)? = object[key] {
                values.insert(value)
            }
            for child in object.values {
                values.formUnion(findBooleanValues(
                    named: key,
                    in: child,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts))
            }
            return values
        case .array(let array):
            return array.reduce(into: Set<Bool>()) { values, child in
                values.formUnion(findBooleanValues(
                    named: key,
                    in: child,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts))
            }
        case .string(let text):
            guard inspectedEmbeddedTexts.insert(text).inserted else {
                return []
            }
            return embeddedJSONValues(in: text).reduce(into: Set<Bool>()) {
                values, embedded in
                values.formUnion(findBooleanValues(
                    named: key,
                    in: embedded,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts))
            }
        default:
            return []
        }
    }

    private struct GuardedDispatchEvidence: Hashable {
        let status: String?
        let submissionStarted: Bool?
        let deliveryAcknowledged: Bool?
    }

    /// The refusal code alone is not enough: VisionCapture uses a different
    /// code when submission may have begun. Require the complete public
    /// pre-submission envelope and reject mixed or contradictory evidence.
    private static func provesGuardedTargetRejectionBeforeSubmission(
        in value: JSONValue
    ) -> Bool {
        var evidence: Set<GuardedDispatchEvidence> = []
        var reasonCodes: Set<String> = []
        var inspectedEmbeddedTexts: Set<String> = []
        collectGuardedDispatchEvidence(
            in: value,
            inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
            into: &evidence,
            reasonCodes: &reasonCodes)
        let expected = GuardedDispatchEvidence(
            status: "rejected_before_submission",
            submissionStarted: false,
            deliveryAcknowledged: false)
        return evidence == [expected]
            && reasonCodes.contains("DISPATCH_REJECTED_BEFORE_SUBMISSION")
    }

    private static func collectGuardedDispatchEvidence(
        in value: JSONValue,
        inspectedEmbeddedTexts: inout Set<String>,
        into evidence: inout Set<GuardedDispatchEvidence>,
        reasonCodes: inout Set<String>
    ) {
        switch value {
        case .object(let object):
            if case .string(let reasonCode)? = object["reason_code"] {
                reasonCodes.insert(reasonCode)
            }
            if case .object(let dispatch)? = object["dispatch"] {
                let status: String? = if case .string(let value)? = dispatch["status"] {
                    value
                } else { nil }
                let submissionStarted: Bool? = if case .bool(let value)? =
                    dispatch["submission_started"] { value } else { nil }
                let deliveryAcknowledged: Bool? = if case .bool(let value)? =
                    dispatch["delivery_acknowledged"] { value } else { nil }
                if status != nil || submissionStarted != nil
                    || deliveryAcknowledged != nil {
                    evidence.insert(GuardedDispatchEvidence(
                        status: status,
                        submissionStarted: submissionStarted,
                        deliveryAcknowledged: deliveryAcknowledged))
                }
            }
            for child in object.values {
                collectGuardedDispatchEvidence(
                    in: child,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    into: &evidence,
                    reasonCodes: &reasonCodes)
            }
        case .array(let array):
            for child in array {
                collectGuardedDispatchEvidence(
                    in: child,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    into: &evidence,
                    reasonCodes: &reasonCodes)
            }
        case .string(let text):
            guard inspectedEmbeddedTexts.insert(text).inserted else { return }
            for embedded in embeddedJSONValues(in: text) {
                collectGuardedDispatchEvidence(
                    in: embedded,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    into: &evidence,
                    reasonCodes: &reasonCodes)
            }
        default:
            break
        }
    }

    private static func embeddedJSONValues(in text: String) -> [JSONValue] {
        let bytes = Array(text.utf8)
        var values: [JSONValue] = []
        var index = 0
        while index < bytes.count {
            guard bytes[index] == 0x7B || bytes[index] == 0x5B else {
                index += 1
                continue
            }
            let start = index
            var closingBytes: [UInt8] = [bytes[index] == 0x7B ? 0x7D : 0x5D]
            var inString = false
            var isEscaped = false
            index += 1
            while index < bytes.count, !closingBytes.isEmpty {
                let byte = bytes[index]
                if inString {
                    if isEscaped {
                        isEscaped = false
                    } else if byte == 0x5C {
                        isEscaped = true
                    } else if byte == 0x22 {
                        inString = false
                    }
                } else {
                    switch byte {
                    case 0x22: inString = true
                    case 0x7B: closingBytes.append(0x7D)
                    case 0x5B: closingBytes.append(0x5D)
                    default:
                        if byte == closingBytes.last { closingBytes.removeLast() }
                    }
                }
                index += 1
            }
            guard closingBytes.isEmpty else { break }
            let block = Data(bytes[start..<index])
            if let value = try? JSONDecoder().decode(JSONValue.self, from: block) {
                values.append(value)
            }
        }
        return values
    }
}
