import AppKit
import Foundation
import TurboFieldfareAppCore

@MainActor
public final class InstructionTranscriptDocumentController {
    public enum Mutation: Equatable {
        case none
        case rebuilt
        case appended
        /// The open block was re-rendered in place. The completed blocks above
        /// it did not move, so the reader keeps their position.
        case tailReplaced
        case finalized
    }

    /// The stretch of the document this update rewrote, in the coordinates the
    /// document had *before* it. Everything below `previous.location` is the
    /// same text it was, which is what lets a selection there keep its exact
    /// indices instead of being clamped onto different characters.
    public struct ReplacedRange: Equatable {
        public let previous: NSRange
        public let length: Int

        public init(previous: NSRange, length: Int) {
            self.previous = previous
            self.length = length
        }

        /// The part of `previous` that actually changed.
        ///
        /// A tail re-render rewrites the same sentence with a few characters
        /// added, and reporting the whole tail as replaced collapsed every
        /// selection inside it — including one over text that had not moved.
        /// Trimming the common prefix and suffix leaves only what differs;
        /// nil when the two are the same text.
        public static func differing(
            previous: NSRange,
            old: NSString,
            new: NSString
        ) -> ReplacedRange? {
            let shorter = min(old.length, new.length)
            var prefix = 0
            while prefix < shorter, old.character(at: prefix) == new.character(at: prefix) {
                prefix += 1
            }
            guard prefix < old.length || prefix < new.length else { return nil }
            var suffix = 0
            while suffix < shorter - prefix,
                  old.character(at: old.length - 1 - suffix)
                    == new.character(at: new.length - 1 - suffix) {
                suffix += 1
            }

            // Neither boundary may land inside a composed character sequence,
            // or the replacement writes half a character.
            var start = prefix
            if start > 0, start < old.length {
                start = min(start, old.rangeOfComposedCharacterSequence(at: start).location)
            }
            if start > 0, start < new.length {
                start = min(start, new.rangeOfComposedCharacterSequence(at: start).location)
            }
            var oldEnd = max(start, old.length - suffix)
            var newEnd = max(start, new.length - suffix)
            if oldEnd > start, oldEnd < old.length {
                let sequence = old.rangeOfComposedCharacterSequence(at: oldEnd)
                if sequence.location < oldEnd { oldEnd = sequence.upperBound }
            }
            if newEnd > start, newEnd < new.length {
                let sequence = new.rangeOfComposedCharacterSequence(at: newEnd)
                if sequence.location < newEnd { newEnd = sequence.upperBound }
            }
            // The two ends move together or the suffixes they leave behind
            // stop matching.
            let kept = min(old.length - oldEnd, new.length - newEnd)
            return ReplacedRange(
                previous: NSRange(
                    location: previous.location + start,
                    length: old.length - kept - start),
                length: new.length - kept - start)
        }
    }

    public struct UpdateResult {
        public let mutation: Mutation
        public let assistantRange: NSRange
        public let replaced: ReplacedRange?

        public init(
            mutation: Mutation,
            assistantRange: NSRange,
            replaced: ReplacedRange? = nil
        ) {
            self.mutation = mutation
            self.assistantRange = assistantRange
            self.replaced = replaced
        }
    }

    public private(set) var prompt = ""
    public private(set) var activities: [AppAgentActivity] = []
    public private(set) var promptPrefixIdentifier = ""
    public private(set) var response = ""
    public private(set) var isFinalized = false
    public private(set) var showsPrefillPlaceholder = false
    private var generationStatusText: String?
    public private(set) var assistantRange = NSRange(location: 0, length: 0)
    /// Length of the completed turns above the live one. Everything below this
    /// offset is drawn once and never looked at again — which is the whole
    /// point: a chat that re-rendered its history on every token would make
    /// TextKit lay out the entire conversation per tick.
    public private(set) var frozenLength = 0
    /// Where each finished turn sits in the document, and the answer it holds.
    ///
    /// One floating copy button over a transcript of several turns reads as
    /// belonging to the first one. Knowing which turn a click landed in is what
    /// lets the menu offer that turn's answer instead of guessing.
    private var sealedTurns: [(range: NSRange, answer: String)] = []
    private var sealedRequestCount = 0
    private var latestRequestMarkerRange: NSRange?
    /// Absolute UTF-16 starts plus the end of the live activity section.
    /// The rendered cards stay in text storage; only their boundaries are kept here.
    private var activityBoundaries: [Int] = []
    private var prefillPlaceholderRange: NSRange?
    private var prefillDotCount = 0

    private let renderer: any TranscriptBlockRendering
    private let progressiveRendering: Bool
    private var progressive = ProgressiveState()

    public init(
        renderer: any TranscriptBlockRendering = ResponseMarkdownRenderer(),
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        self.renderer = renderer
        progressiveRendering = environment["TURBO_FIELDFARE_PROGRESSIVE_RENDER"] != "0"
    }

    /// The part of the assistant range that is still being written. Only this
    /// range is rewritten while an answer streams; in the raw streaming mode
    /// the whole answer is unrendered and the two ranges are the same.
    public var tailRange: NSRange {
        NSRange(
            location: assistantRange.location + progressive.prefixLength,
            length: max(0, assistantRange.length - progressive.prefixLength))
    }

    public static func clampedRanges(
        _ ranges: [NSRange],
        toLength length: Int
    ) -> [NSRange] {
        ranges.map { range in
            let location = min(max(range.location, 0), length)
            let available = max(0, length - location)
            return NSRange(location: location, length: min(max(range.length, 0), available))
        }
    }

    /// Where a selection lands after part of the document was rewritten.
    ///
    /// Clamping the old ranges onto the new length kept their lengths and so
    /// kept selecting *something* — whatever now sits at those indices. When
    /// the open tail is redrawn on a streaming tick that is different text,
    /// so a selection that reached into the tail loses the part that no longer
    /// exists and keeps the part above it, which has not moved.
    public static func adjustedRanges(
        _ ranges: [NSRange],
        replacing replaced: NSRange,
        newLength: Int
    ) -> [NSRange] {
        let end = replaced.location + newLength
        let delta = newLength - replaced.length
        return ranges.map { range in
            guard range.upperBound > replaced.location else { return range }
            // Below the rewrite the characters are the same ones, just moved.
            // Collapsing them too dropped a selection over a heading the
            // finalize pass never touched.
            guard range.location < replaced.upperBound else {
                return NSRange(location: range.location + delta, length: range.length)
            }
            guard range.location < replaced.location else {
                return NSRange(location: min(range.location, end), length: 0)
            }
            return NSRange(
                location: range.location,
                length: replaced.location - range.location)
        }
    }

    public static func shouldScrollToBottom(
        wasAtBottom: Bool,
        mutation: Mutation
    ) -> Bool {
        wasAtBottom || mutation == .finalized
    }

    public static func shouldRunPrefillAnimation(
        response: String,
        isTerminal: Bool,
        requested: Bool
    ) -> Bool {
        requested && response.isEmpty && !isTerminal
    }

    /// Whether `response` starts with the exact bytes of `previous`.
    /// `hasPrefix` compares by canonical equivalence, so a decomposed prefix
    /// that arrives recomposed passes it while the UTF-8 offsets the delta is
    /// cut at no longer line up, and the append writes the wrong characters
    /// onto a document drawn from different ones.
    private static func extendsExactly(_ response: String, _ previous: String) -> Bool {
        let count = previous.utf8.count
        guard count > 0 else { return true }
        guard response.utf8.count >= count else { return false }
        // Through contiguous storage, not an element-wise walk of two UTF8
        // views: this runs on every streaming tick, and iterating one costs a
        // measurable fraction of the tick by the end of a long answer.
        return Self.withUTF8(response) { response in
            Self.withUTF8(previous) { previous in
                guard let left = response.baseAddress, let right = previous.baseAddress else {
                    return response.prefix(count).elementsEqual(previous)
                }
                return memcmp(left, right, count) == 0
            }
        }
    }

    private static func withUTF8<R>(
        _ text: String,
        _ body: (UnsafeBufferPointer<UInt8>) -> R
    ) -> R {
        if let result = text.utf8.withContiguousStorageIfAvailable(body) { return result }
        var copy = text
        return copy.withUTF8 { body($0) }
    }

    /// The text `response` adds on top of `previous`, or nil when the two do
    /// not share a Character-aligned UTF-8 prefix and the document must be
    /// rebuilt instead. Grapheme-based `count`/`dropFirst` walk the whole
    /// response on every streaming tick, which is quadratic across a
    /// generation; UTF-8 offsets are constant time on native strings.
    private static func appendedSuffix(of response: String, after previous: String) -> String? {
        let previousBytes = previous.utf8.count
        guard response.utf8.count > previousBytes else { return nil }
        let boundary = response.utf8.index(
            response.utf8.startIndex,
            offsetBy: previousBytes)
        guard let start = boundary.samePosition(in: response) else { return nil }
        return String(response[start...])
    }

    @discardableResult
    public func synchronize(
        storage: NSMutableAttributedString,
        prompt: String,
        activities: [AppAgentActivity] = [],
        response: String,
        isTerminal: Bool,
        showsPrefillPlaceholder: Bool = false,
        promptPrefix: NSAttributedString = NSAttributedString(),
        promptPrefixIdentifier: String = "",
        generationStatusText: String? = nil
    ) -> UpdateResult {
        let statusChanged = generationStatusText != self.generationStatusText
        self.generationStatusText = generationStatusText
        let responseChanged = response != self.response
        let displaysPrefillPlaceholder = Self.shouldRunPrefillAnimation(
            response: response,
            isTerminal: isTerminal,
            requested: showsPrefillPlaceholder)
        var needsRebuild = prompt != self.prompt
            || activities != self.activities
            || promptPrefixIdentifier != self.promptPrefixIdentifier
            || !Self.extendsExactly(response, self.response)
            || (isFinalized && !isTerminal)
            || displaysPrefillPlaceholder != self.showsPrefillPlaceholder
        var appendedDelta: String?
        if !needsRebuild, response.utf8.count > self.response.utf8.count {
            appendedDelta = Self.appendedSuffix(of: response, after: self.response)
            needsRebuild = appendedDelta == nil
        }

        var mutation: Mutation = .none
        var replaced: ReplacedRange?
        var rebuiltSuffix: (previous: NSRange, text: NSString)?
        // In progressive mode a terminal response is closed block by block, so
        // the tick that carries it is an ordinary extension. The raw mode is
        // replaced wholesale below and has nothing to gain from writing the
        // delta first.
        // `frozenLength`, not zero: with completed turns above it the live turn
        // is empty at `storage.length == frozenLength`, and comparing against
        // zero meant the first tick of turn two never drew its prompt.
        if needsRebuild
            || storage.length == frozenLength
                && (!prompt.isEmpty || promptPrefix.length > 0
                    || !response.isEmpty || displaysPrefillPlaceholder) {
            rebuiltSuffix = rebuild(
                storage: storage,
                prompt: prompt,
                promptPrefix: promptPrefix,
                promptPrefixIdentifier: promptPrefixIdentifier,
                activities: activities,
                response: response,
                showsPrefillPlaceholder: displaysPrefillPlaceholder,
                closingTail: isTerminal)
            mutation = .rebuilt
        } else if let delta = appendedDelta {
            if progressiveRendering {
                let update = extendProgressiveRender(
                    storage: storage,
                    response: response,
                    closingTail: isTerminal)
                mutation = update.mutation
                replaced = update.replaced
            } else if !isTerminal {
                mutation = appendRaw(delta, to: storage)
            }
        } else if statusChanged, displaysPrefillPlaceholder, let range = prefillPlaceholderRange {
            let text = NSAttributedString(
                string: generationStatusText ?? Self.prefillPlaceholder(dotCount: prefillDotCount),
                attributes: Self.prefillPlaceholderAttributes())
            storage.replaceCharacters(in: range, with: text)
            prefillPlaceholderRange = NSRange(location: range.location, length: text.length)
            replaced = ReplacedRange(previous: range, length: text.length)
            mutation = .tailReplaced
        }

        self.prompt = prompt
        self.activities = activities
        self.promptPrefixIdentifier = promptPrefixIdentifier
        self.response = response
        self.showsPrefillPlaceholder = displaysPrefillPlaceholder

        if isTerminal && (!isFinalized || responseChanged) {
            if progressiveRendering {
                // Finalize closes the open block and leaves everything above it
                // exactly as it was drawn. Re-rendering the whole answer put it
                // back through message-wide gates the streaming pass applies
                // per block — an unclosed fence anywhere, a table in a list, a
                // block-level HTML tag — so an answer that streamed styled
                // flipped to raw source the moment it finished.
                if let update = closeTail(storage: storage, response: response) {
                    replaced = update
                }
            } else {
                let rendered = renderer.render(response, typesetsMath: true).attributedString
                replaced = ReplacedRange(previous: assistantRange, length: rendered.length)
                storage.replaceCharacters(in: assistantRange, with: rendered)
                assistantRange.length = rendered.length
                progressive.reset()
            }
            isFinalized = true
            mutation = .finalized
        } else if !isTerminal {
            isFinalized = false
        }

        // Terminal rendering may also change the answer after the activity
        // suffix moved it. Report one change in the original coordinates.
        if let rebuiltSuffix {
            let current = NSRange(
                location: rebuiltSuffix.previous.location,
                length: storage.length - rebuiltSuffix.previous.location)
            replaced = ReplacedRange.differing(
                previous: rebuiltSuffix.previous,
                old: rebuiltSuffix.text,
                new: storage.mutableString.substring(with: current) as NSString)
        }

        return UpdateResult(
            mutation: mutation,
            assistantRange: assistantRange,
            replaced: replaced)
    }

    @discardableResult
    public func advancePrefillAnimation(
        storage: NSMutableAttributedString
    ) -> Bool {
        guard showsPrefillPlaceholder, var range = prefillPlaceholderRange else {
            return false
        }
        prefillDotCount = (prefillDotCount + 1) % 4
        let replacement = NSAttributedString(
            string: generationStatusText ?? Self.prefillPlaceholder(dotCount: prefillDotCount),
            attributes: Self.prefillPlaceholderAttributes())
        storage.replaceCharacters(in: range, with: replacement)
        range.length = replacement.length
        prefillPlaceholderRange = range
        return true
    }

    private func appendRaw(
        _ delta: String,
        to storage: NSMutableAttributedString
    ) -> Mutation {
        storage.append(NSAttributedString(
            string: delta,
            attributes: Self.responseAttributes()))
        assistantRange.length += (delta as NSString).length
        return .appended
    }

    private func rebuild(
        storage: NSMutableAttributedString,
        prompt: String,
        promptPrefix: NSAttributedString,
        promptPrefixIdentifier: String,
        activities: [AppAgentActivity],
        response: String,
        showsPrefillPlaceholder: Bool,
        closingTail: Bool
    ) -> (previous: NSRange, text: NSString)? {
        let canKeepPrefix = prompt == self.prompt
            && promptPrefixIdentifier == self.promptPrefixIdentifier
            && activityBoundaries.count == self.activities.count + 1
            && activityBoundaries.first.map { $0 >= frozenLength } == true
            && activityBoundaries.last.map {
                $0 <= assistantRange.location && $0 <= storage.length
            } == true
        var keptActivities = 0
        if canKeepPrefix {
            while keptActivities < min(activities.count, self.activities.count),
                  activities[keptActivities] == self.activities[keptActivities] {
                keptActivities += 1
            }
            // An unchanged request still needs redrawing when it loses Latest.
            let latestID = activities.last(where: { $0.kind == .mcpRequest })?.id
            if let oldLatest = self.activities.lastIndex(where: { $0.kind == .mcpRequest }),
               self.activities[oldLatest].id != latestID {
                keptActivities = min(keptActivities, oldLatest)
            }
        }
        let replacementStart = canKeepPrefix
            ? activityBoundaries[keptActivities] : frozenLength
        let previous = NSRange(
            location: replacementStart, length: storage.length - replacementStart)
        let oldText: NSString? = canKeepPrefix
            ? storage.mutableString.substring(with: previous) as NSString : nil
        let document = NSMutableAttributedString()
        if !canKeepPrefix, !prompt.isEmpty || promptPrefix.length > 0 {
            document.append(NSAttributedString(
                string: "You\n",
                attributes: Self.userLabelAttributes()))
            if promptPrefix.length > 0 {
                document.append(promptPrefix)
                if !prompt.isEmpty {
                    document.append(NSAttributedString(
                        string: "\n\n",
                        attributes: Self.promptAttributes()))
                }
            }
            if !prompt.isEmpty {
                document.append(NSAttributedString(
                    string: prompt,
                    attributes: Self.promptAttributes()))
            }
            document.append(NSAttributedString(
                string: "\n\n",
                attributes: Self.promptAttributes()))
        }
        let activityStart = replacementStart + document.length
        let precedingRequests = activities.prefix(keptActivities)
            .reduce(0) { $0 + ($1.kind == .mcpRequest ? 1 : 0) }
        let requests = Self.activityDocument(
            activities, startingNumber: sealedRequestCount + precedingRequests + 1,
            from: keptActivities)
        activityBoundaries = Array(activityBoundaries.prefix(keptActivities))
            + requests.boundaries.map { activityStart + $0 }
        if let marker = requests.latestMarkerRange {
            latestRequestMarkerRange = NSRange(
                location: activityStart + marker.location, length: marker.length)
        } else if let marker = latestRequestMarkerRange,
                  marker.location >= replacementStart {
            latestRequestMarkerRange = nil
        }
        document.append(requests.document)
        document.append(NSAttributedString(
            string: "Answer\n",
            attributes: Self.assistantLabelAttributes()))
        assistantRange = NSRange(location: replacementStart + document.length, length: 0)
        let assistant = progressiveRendering
            ? progressiveRender(response, closingTail: closingTail)
            : NSAttributedString(string: response, attributes: Self.responseAttributes())
        document.append(assistant)
        assistantRange.length = assistant.length
        prefillDotCount = 0
        prefillPlaceholderRange = nil
        if showsPrefillPlaceholder {
            let placeholder = NSAttributedString(
                string: generationStatusText ?? Self.prefillPlaceholder(dotCount: prefillDotCount),
                attributes: Self.prefillPlaceholderAttributes())
            prefillPlaceholderRange = NSRange(
                location: replacementStart + document.length,
                length: placeholder.length)
            document.append(placeholder)
        }
        // Finished turns and unchanged live cards retain their existing AppKit
        // objects. Incompatible prompt/image changes still rebuild the live turn.
        storage.replaceCharacters(in: previous, with: document)
        isFinalized = false
        return oldText.map { (previous: previous, text: $0) }
    }

    /// Freezes the live turn into the history above it and starts an empty one.
    ///
    /// Called when a turn is committed. Apart from retiring its current-request
    /// marker, sealing is bookkeeping, not a re-render of the finished answer.
    public func sealTurn(storage: NSMutableAttributedString) {
        guard storage.length > frozenLength else { return }
        // Only the current turn may claim Latest or Running. Remove its small
        // marker once at archival, without re-rendering earlier requests.
        if let range = latestRequestMarkerRange {
            storage.deleteCharacters(in: range)
            assistantRange.location -= range.length
        }
        latestRequestMarkerRange = nil
        activityBoundaries.removeAll(keepingCapacity: true)
        sealedRequestCount += activities.filter { $0.kind == .mcpRequest }.count
        let start = frozenLength
        let answer = response
        storage.append(NSAttributedString(
            string: "\n\n", attributes: Self.promptAttributes()))
        frozenLength = storage.length
        sealedTurns.append((
            range: NSRange(location: start, length: frozenLength - start),
            answer: answer))
        prompt = ""
        activities = []
        promptPrefixIdentifier = ""
        response = ""
        isFinalized = false
        showsPrefillPlaceholder = false
        assistantRange = NSRange(location: frozenLength, length: 0)
        prefillPlaceholderRange = nil
        prefillDotCount = 0
        progressive.reset()
    }

    /// The answer of the turn containing `index`, or nil if the index is not in
    /// a turn that has one. The live turn is included: it is the one a reader is
    /// most likely to want, and it is not sealed until the next run starts.
    public func answer(at index: Int) -> String? {
        for turn in sealedTurns where NSLocationInRange(index, turn.range) {
            return turn.answer.isEmpty ? nil : turn.answer
        }
        guard index >= frozenLength, !response.isEmpty else { return nil }
        return response
    }

    /// Marks where the model's context begins.
    ///
    /// Everything above this line is on screen but no longer in the KV — a
    /// reload or an unload took it. Saying so is the difference between a
    /// transcript the user can trust and one that quietly implies the model
    /// remembers what it cannot.
    public func appendContextBreak(storage: NSMutableAttributedString, text: String) {
        guard storage.length == frozenLength else { return }
        storage.append(NSAttributedString(
            string: "\(text)\n\n", attributes: Self.contextBreakAttributes()))
        frozenLength = storage.length
        assistantRange = NSRange(location: frozenLength, length: 0)
    }

    /// Empties the transcript, history included. New chat, not a new turn.
    public func resetTranscript(storage: NSMutableAttributedString) {
        storage.setAttributedString(NSAttributedString())
        TranscriptImageLoader.clearCache()
        frozenLength = 0
        sealedTurns.removeAll()
        sealedRequestCount = 0
        latestRequestMarkerRange = nil
        activityBoundaries.removeAll()
        prompt = ""
        activities = []
        promptPrefixIdentifier = ""
        response = ""
        isFinalized = false
        showsPrefillPlaceholder = false
        assistantRange = NSRange(location: 0, length: 0)
        prefillPlaceholderRange = nil
        prefillDotCount = 0
        progressive.reset()
        // Whole-chat resets release rendered blocks; normal turn resets reuse them.
        progressive.cache.removeAll(keepingCapacity: false)
    }

    // MARK: - Progressive rendering

    /// What the transcript shows while an answer is still arriving: every
    /// completed block already styled, and the block being written re-rendered
    /// from an auto-closed copy of itself. Finalize closes that last block the
    /// same way, so nothing above it is ever looked at twice.
    ///
    /// The cost of rendering block by block is that a block cannot see the
    /// rest of the answer: a `[label][ref]` whose definition sits in a later
    /// block keeps its brackets. Resolving it per tick is the quadratic shape
    /// the timing gate forbids, and resolving it only at finalize would
    /// reintroduce the flip this pass removed.
    private struct ProgressiveState {
        var split = ResponseBlockSplitter.Split()
        /// Completed blocks that are already drawn.
        var renderedBlocks = 0
        /// NSString length of those blocks and the separators between them,
        /// including the separator that precedes the open tail.
        var prefixLength = 0
        /// UTF-8 offset of the open block the drawn tail was made from.
        var tailStart: Int?
        var fence: FenceTail?
        /// Rendered blocks keyed by position and text. Position is part of the
        /// key because two identical tables must not share one `NSTextTable`,
        /// which TextKit would draw as a single merged table.
        var cache: [BlockKey: NSAttributedString] = [:]

        mutating func reset() {
            split = ResponseBlockSplitter.Split()
            renderedBlocks = 0
            prefixLength = 0
            tailStart = nil
            fence = nil
        }
    }

    private struct BlockKey: Hashable {
        let start: Int
        let text: String
    }

    /// A fenced block that is still arriving. It is drawn directly rather than
    /// through the markdown pass because it is the one block with no size
    /// bound: re-parsing a 10 KB listing on each of the ~1,200 ticks it takes
    /// to stream is quadratic, while appending the bytes that arrived is not.
    private struct FenceTail {
        let attributes: [NSAttributedString.Key: Any]
        let bodyStart: Int
        var bodyEnd: Int
        var filter: CodeBodyFilter
        /// The drawn tail ends with a newline this controller added because the
        /// body has none of its own yet; the next delta replaces it.
        var synthetic: Bool
    }

    /// Turns raw fenced-body bytes into what the markdown parser would have
    /// produced: CRLF and a lone CR become LF, and the opening fence's own
    /// indentation comes off every line. Deltas arrive one at a time, so the CR
    /// that ends one and the LF that starts the next still collapse into a
    /// single line break.
    private struct CodeBodyFilter {
        let indent: Int
        private var pendingCarriageReturn = false
        /// Columns of the current line consumed while the opening fence's own
        /// indentation comes off. A tab is four columns wide, so one that
        /// reaches past the fence's indent keeps the columns it owns beyond
        /// it: dropping the whole tab lost the listing's indentation until the
        /// answer finalized and the markdown pass drew it properly.
        private var column = 0
        private var stripping: Bool
        private(set) var endsWithNewline = true

        init(indent: Int) {
            self.indent = indent
            stripping = indent > 0
        }

        /// Scalars, not characters: Swift reads CRLF as one `Character`, so a
        /// character loop would carry the carriage return straight through into
        /// the transcript.
        mutating func append(_ raw: String) -> String {
            var output = String.UnicodeScalarView()
            output.reserveCapacity(raw.unicodeScalars.count)
            for scalar in raw.unicodeScalars {
                if scalar == "\n" {
                    if pendingCarriageReturn {
                        pendingCarriageReturn = false
                        continue
                    }
                    output.append("\n")
                    startLine()
                    endsWithNewline = true
                    continue
                }
                pendingCarriageReturn = false
                if scalar == "\r" {
                    pendingCarriageReturn = true
                    output.append("\n")
                    startLine()
                    endsWithNewline = true
                    continue
                }
                if stripping, scalar == " " {
                    column += 1
                    stripping = column < indent
                    continue
                }
                if stripping, scalar == "\t" {
                    let reached = column + 4 - column % 4
                    column = reached
                    stripping = false
                    if reached > indent {
                        for _ in 0..<(reached - indent) { output.append(" ") }
                        endsWithNewline = false
                    }
                    continue
                }
                stripping = false
                endsWithNewline = false
                output.append(scalar)
            }
            return String(output)
        }

        private mutating func startLine() {
            column = 0
            stripping = indent > 0
        }
    }

    private func progressiveRender(
        _ response: String,
        closingTail: Bool
    ) -> NSAttributedString {
        progressive.reset()
        let split = ResponseBlockSplitter.split(response)
        progressive.split = split
        let content = NSMutableAttributedString()
        var trailing = 0
        for block in split.completed {
            // With the separator the incremental path writes. Without it a
            // rebuild mid-answer — an image thumbnail landing and changing the
            // prompt prefix — glued every completed block to the one above it
            // and left the answer that way until finalize.
            append(
                block,
                of: response,
                to: content,
                trailing: &trailing,
                isFirst: content.length == 0)
        }
        progressive.renderedBlocks = split.completed.count
        if let open = split.open {
            content.append(separator(trailingNewlines: trailing, isFirst: content.length == 0))
            progressive.prefixLength = content.length
            if closingTail {
                // A terminal rebuild draws the last block finished. Rendering it
                // as a tail and then re-rendering the whole answer cost N+2
                // passes over an answer that was already on screen.
                progressive.tailStart = open.start
                progressive.fence = nil
                content.append(renderedBlock(open, of: response))
            } else {
                content.append(renderTail(open, of: response))
            }
        } else {
            progressive.prefixLength = content.length
        }
        return content
    }

    private struct ProgressiveUpdate {
        let mutation: Mutation
        var replaced: ReplacedRange?
    }

    private func extendProgressiveRender(
        storage: NSMutableAttributedString,
        response: String,
        closingTail: Bool
    ) -> ProgressiveUpdate {
        let split = ResponseBlockSplitter.split(response, resuming: progressive.split)
        progressive.split = split

        if !closingTail,
           split.completed.count == progressive.renderedBlocks,
           let open = split.open,
           open.start == progressive.tailStart,
           let fence = open.fence,
           let drawn = progressive.fence,
           fence.bodyStart == drawn.bodyStart {
            return growFence(fence, storage: storage, response: response)
        }

        var replacedTail = false
        let drawnTail = tailRange
        let drawnTailText = (storage.string as NSString).substring(with: NSRange(
            location: min(drawnTail.location, storage.length),
            length: min(drawnTail.length, max(0, storage.length - drawnTail.location)))) as NSString
        if progressive.tailStart != nil {
            if let promoted = split.completed[safe: progressive.renderedBlocks],
               promoted.start == progressive.tailStart,
               keepsDrawnBytes(promoted) {
                // The tail was a fenced block drawn from exactly this body; its
                // closing line adds nothing to the render, so the bytes stay
                // where they are and only the boundary moves.
                progressive.prefixLength += drawnTail.length
                progressive.renderedBlocks += 1
            } else if drawnTail.length > 0 {
                storage.deleteCharacters(in: drawnTail)
                assistantRange.length = progressive.prefixLength
                replacedTail = true
            }
            progressive.tailStart = nil
            progressive.fence = nil
        }

        let addition = NSMutableAttributedString()
        var trailing = trailingNewlines(in: storage)
        while progressive.renderedBlocks < split.completed.count {
            let block = split.completed[progressive.renderedBlocks]
            append(
                block,
                of: response,
                to: addition,
                trailing: &trailing,
                isFirst: assistantRange.length == 0 && addition.length == 0)
            progressive.renderedBlocks += 1
        }
        if let open = split.open {
            addition.append(separator(
                trailingNewlines: trailing,
                isFirst: assistantRange.length == 0 && addition.length == 0))
            progressive.prefixLength = assistantRange.length + addition.length
            if closingTail {
                progressive.tailStart = open.start
                progressive.fence = nil
                addition.append(renderedBlock(open, of: response))
            } else {
                addition.append(renderTail(open, of: response))
            }
        } else {
            progressive.prefixLength = assistantRange.length + addition.length
        }
        guard addition.length > 0 else {
            guard replacedTail else { return ProgressiveUpdate(mutation: .none) }
            return ProgressiveUpdate(
                mutation: .tailReplaced,
                replaced: ReplacedRange(previous: drawnTail, length: 0))
        }
        storage.replaceCharacters(
            in: NSRange(location: assistantRange.upperBound, length: 0),
            with: addition)
        assistantRange.length += addition.length
        guard replacedTail else { return ProgressiveUpdate(mutation: .appended) }
        return ProgressiveUpdate(
            mutation: .tailReplaced,
            replaced: ReplacedRange.differing(
                previous: drawnTail,
                old: drawnTailText,
                new: addition.string as NSString)
                ?? ReplacedRange(previous: drawnTail, length: addition.length))
    }

    /// Closes the block that was still being written, in place.
    ///
    /// The blocks above it are the ones already drawn: nothing above the open
    /// block is looked at again, so a message-wide gate cannot turn a styled
    /// answer back into raw source at the moment it finishes.
    private func closeTail(
        storage: NSMutableAttributedString,
        response: String
    ) -> ReplacedRange? {
        let update = extendProgressiveRender(
            storage: storage,
            response: response,
            closingTail: true)
        return update.replaced
    }

    /// Appends one completed block and remembers how many newlines it left
    /// behind, so the next separator matches what a whole-document render puts
    /// between the same two blocks.
    private func append(
        _ block: ResponseBlockSplitter.Block,
        of response: String,
        to content: NSMutableAttributedString,
        trailing: inout Int,
        isFirst: Bool
    ) {
        let rendered = renderedBlock(block, of: response)
        // A block that draws nothing — an empty fence — takes no separator
        // with it either, or the gap around it is written twice.
        guard rendered.length > 0 else { return }
        let separator = self.separator(trailingNewlines: trailing, isFirst: isFirst)
        content.append(separator)
        content.append(rendered)
        // The separator's own newlines count too. Seeding with the previous
        // block's count instead left a fence whose body is one blank line with
        // an extra paragraph gap under it that the whole render does not have.
        trailing = min(2, Self.trailingNewlines(
            rendered.string,
            seed: trailing + separator.length))
    }

    private func separator(trailingNewlines: Int, isFirst: Bool) -> NSAttributedString {
        guard !isFirst else { return NSAttributedString() }
        return renderer.blockSeparator(trailingNewlines: trailingNewlines)
    }

    private func renderedBlock(
        _ block: ResponseBlockSplitter.Block,
        of response: String
    ) -> NSAttributedString {
        // Empty body included: a fence with nothing in it renders nothing.
        // Sending it to the markdown pass instead produced an empty document,
        // which the renderer turns into raw fallback, so a rebuild mid-answer
        // put "```" back on screen where the streamed render had drawn nothing.
        if let fence = block.fence {
            return code(fence, of: response, attributes: renderer.streamingCodeAttributes()).text
        }
        let text = ResponseBlockSplitter.text(block.utf8Range, in: response)
        let key = BlockKey(start: block.start, text: text)
        if let cached = progressive.cache[key] { return cached }
        let rendered = renderer.render(text, typesetsMath: true).attributedString
        // A session cannot grow this without bound; block renders are already
        // held by the storage, and dropping the whole table beats ageing it.
        if progressive.cache.count >= 512 { progressive.cache.removeAll(keepingCapacity: true) }
        progressive.cache[key] = rendered
        return rendered
    }

    /// The open block, styled. Math stays as source here: a half-typed equation
    /// cannot typeset, so asking per tick only produces a failed parse and a
    /// log line, and the block is typeset once when it completes.
    private func renderTail(
        _ block: ResponseBlockSplitter.Block,
        of response: String
    ) -> NSAttributedString {
        progressive.tailStart = block.start
        if let fence = block.fence {
            let drawn = code(
                fence,
                of: response,
                attributes: renderer.streamingCodeAttributes())
            progressive.fence = drawn.state
            return drawn.text
        }
        progressive.fence = nil
        let text = ResponseBlockSplitter.text(block.utf8Range, in: response)
        let closed = TailAutoClose.close(text, typesetsMath: false)
        return renderer.render(closed, typesetsMath: false).attributedString
    }

    private func code(
        _ fence: ResponseBlockSplitter.Fence,
        of response: String,
        attributes: [NSAttributedString.Key: Any]
    ) -> (text: NSAttributedString, state: FenceTail) {
        var filter = CodeBodyFilter(indent: fence.indent)
        var body = filter.append(ResponseBlockSplitter.text(
            fence.bodyStart..<fence.bodyEnd,
            in: response))
        // Every code line is a paragraph; the last one needs its terminator or
        // the container closes mid-line.
        let synthetic = !filter.endsWithNewline
        if synthetic { body.append("\n") }
        return (
            NSAttributedString(string: body, attributes: attributes),
            FenceTail(
                attributes: attributes,
                bodyStart: fence.bodyStart,
                bodyEnd: fence.bodyEnd,
                filter: filter,
                synthetic: synthetic))
    }

    private func growFence(
        _ fence: ResponseBlockSplitter.Fence,
        storage: NSMutableAttributedString,
        response: String
    ) -> ProgressiveUpdate {
        guard var drawn = progressive.fence else { return ProgressiveUpdate(mutation: .none) }
        guard fence.bodyEnd >= drawn.bodyEnd else {
            // A closing fence line took bytes back out of the body. Redrawing
            // costs one pass over a block that is about to stop growing.
            let redrawn = code(fence, of: response, attributes: drawn.attributes)
            progressive.fence = redrawn.state
            let previous = tailRange
            let before = (storage.string as NSString).substring(with: previous) as NSString
            storage.replaceCharacters(in: previous, with: redrawn.text)
            assistantRange.length = progressive.prefixLength + redrawn.text.length
            return ProgressiveUpdate(
                mutation: .tailReplaced,
                replaced: ReplacedRange.differing(
                    previous: previous,
                    old: before,
                    new: redrawn.text.string as NSString)
                    ?? ReplacedRange(previous: previous, length: redrawn.text.length))
        }
        let raw = ResponseBlockSplitter.text(drawn.bodyEnd..<fence.bodyEnd, in: response)
        guard !raw.isEmpty else { return ProgressiveUpdate(mutation: .none) }
        var delta = drawn.filter.append(raw)
        let synthetic = !drawn.filter.endsWithNewline
        if synthetic { delta.append("\n") }
        // The newline that goes away is the one the previous tick added, not
        // the one this tick is about to.
        let removal = NSRange(
            location: assistantRange.upperBound - (drawn.synthetic ? 1 : 0),
            length: drawn.synthetic ? 1 : 0)
        drawn.bodyEnd = fence.bodyEnd
        drawn.synthetic = synthetic
        progressive.fence = drawn
        guard !delta.isEmpty || removal.length > 0 else {
            return ProgressiveUpdate(mutation: .none)
        }
        storage.replaceCharacters(
            in: removal,
            with: NSAttributedString(string: delta, attributes: drawn.attributes))
        assistantRange.length += (delta as NSString).length - removal.length
        return ProgressiveUpdate(mutation: .appended)
    }

    /// True when the drawn tail is already exactly what this newly completed
    /// block renders to, so its bytes can stay in place.
    private func keepsDrawnBytes(_ block: ResponseBlockSplitter.Block) -> Bool {
        guard let fence = block.fence, let drawn = progressive.fence else { return false }
        return fence.bodyStart == drawn.bodyStart && fence.bodyEnd == drawn.bodyEnd
    }

    private func trailingNewlines(in storage: NSMutableAttributedString) -> Int {
        guard assistantRange.length > 0 else { return 0 }
        let text = storage.string as NSString
        var count = 0
        var index = min(assistantRange.upperBound, text.length) - 1
        while index >= assistantRange.location, count < 2,
              text.character(at: index) == 10 {
            count += 1
            index -= 1
        }
        return count
    }

    private static func trailingNewlines(_ text: String, seed: Int) -> Int {
        var count = 0
        for character in text.reversed() {
            guard character == "\n" else { return count }
            count += 1
        }
        return count + seed
    }

    private static func activityDocument(
        _ activities: [AppAgentActivity],
        startingNumber: Int,
        from startIndex: Int
    ) -> (document: NSAttributedString, latestMarkerRange: NSRange?, boundaries: [Int]) {
        let document = NSMutableAttributedString()
        let latestRequestID = activities.last(where: { $0.kind == .mcpRequest })?.id
        var number = startingNumber
        var latestMarkerRange: NSRange?
        var boundaries: [Int] = []
        for activity in activities.dropFirst(startIndex) {
            boundaries.append(document.length)
            switch activity.kind {
            case .mcpRequest:
                document.append(NSAttributedString(
                    string: "VisionCapture MCP request #\(number)",
                    attributes: activityLabelAttributes(
                        color: TurboFieldfareMacTheme.accentNSColor)))
                number += 1
                if activity.id == latestRequestID {
                    let marker = activity.status == .dispatching ? " · Running" : " · Latest"
                    latestMarkerRange = NSRange(
                        location: document.length, length: (marker as NSString).length)
                    document.append(NSAttributedString(
                        string: marker,
                        attributes: activityLabelAttributes(
                            color: TurboFieldfareMacTheme.accentNSColor)))
                }
                document.append(NSAttributedString(string: "\n"))
                let status = activityStatus(
                    activity.status,
                    elapsedSeconds: activity.elapsedSeconds)
                document.append(activityCodeCard(
                    body: activity.body,
                    status: status))
                if let response = activity.responseBody {
                    document.append(NSAttributedString(
                        string: "\nVisionCapture response to request #\(number - 1) · display excerpt\n",
                        attributes: activityLabelAttributes(color: .secondaryLabelColor)))
                    document.append(NSAttributedString(
                        string: "Actual MCP result received by the host. Embedded content and binary fields are omitted where marked. This is not the sanitized result sent to Gemma.\n",
                        attributes: activityBodyAttributes()))
                    document.append(activityCodeCard(body: response, cardStyle: .response))
                }
                if let preview = activity.screenshotPreview {
                    document.append(NSAttributedString(
                        string: "\nScreenshot preview\n",
                        attributes: activityLabelAttributes(color: .secondaryLabelColor)))
                    let budget = TranscriptImageLoader.Budget(
                        maximumSourcePixels: 512 * 512, maximumSourceDimension: 512,
                        maximumDecodedBytes: 2 * 1_024 * 1_024,
                        allowedTypeIdentifiers: ["public.png"])
                    if let image = TranscriptImageLoader.thumbnail(
                        at: preview.fileURL, maximumPixelSize: 512,
                        budget: budget, cacheKey: preview.sha256) {
                        let size = TranscriptImageTile.fittedSize(
                            image.size, within: CGSize(width: 360, height: 360))
                        let attachment = NSTextAttachment()
                        attachment.image = image
                        attachment.bounds = CGRect(origin: .zero, size: size)
                        document.append(NSAttributedString(attachment: attachment))
                    } else {
                        document.append(NSAttributedString(
                            string: "Screenshot preview unavailable.", attributes: activityBodyAttributes()))
                    }
                } else if let reason = activity.screenshotPreviewUnavailable {
                    document.append(NSAttributedString(
                        string: "\n\(reason)", attributes: activityBodyAttributes()))
                }
                document.append(NSAttributedString(string: "\n\n"))
            case .modelInput(let attempt, let isFormatCorrection):
                document.append(NSAttributedString(
                    string: "Sent to Gemma · input attempt #\(attempt)\(isFormatCorrection ? " · format correction" : "")\n",
                    attributes: activityLabelAttributes(color: modelInputAccentColor)))
                document.append(NSAttributedString(
                    string: "Complete request text below. Prior conversation is reused separately.\nDeveloper/tool configuration may already be retained.\nTool results are host-sanitized; raw MCP responses have their own cards.\nImages are supplied separately; references are listed below.\nJSON indentation is display-only. Original request text and values are unchanged.\n",
                    attributes: activityBodyAttributes()))
                document.append(activityCodeCard(
                    body: formattedModelInputBody(activity.body), cardStyle: .modelInput))
                document.append(NSAttributedString(string: "\n\n"))
            case .modelResult(let callID, let toolName, let imageCount):
                document.append(NSAttributedString(
                    string: "Host → Gemma · tool result\n",
                    attributes: activityLabelAttributes(color: TurboFieldfareMacTheme.accentNSColor)))
                document.append(NSAttributedString(
                    string: "\(toolName) · call \(callID)\nSanitized result · small JSON formatted for display. Original model input is unchanged. This result may combine several MCP responses.\n",
                    attributes: activityBodyAttributes()))
                document.append(activityCodeCard(body: activity.body, cardStyle: .response))
                if imageCount > 0 {
                    document.append(NSAttributedString(
                        string: "\(imageCount) image attachment(s) included separately in this model input. Image bytes are omitted from this text; see the screenshot preview above.\n",
                        attributes: activityBodyAttributes()))
                }
                document.append(NSAttributedString(string: "\n\n"))
            case .localProposal:
                document.append(NSAttributedString(
                    string: "Not sent · Gemma proposal (not an MCP request)\n",
                    attributes: activityLabelAttributes(color: .systemOrange)))
                document.append(NSAttributedString(
                    string: activity.body,
                    attributes: activityBodyAttributes()))
                if case .notSent(let reason) = activity.status {
                    document.append(NSAttributedString(
                        string: "\nReason: \(reason)\n\n",
                        attributes: activityStatusAttributes(color: .systemOrange)))
                } else {
                    let status = activityStatus(
                        activity.status,
                        elapsedSeconds: activity.elapsedSeconds)
                    document.append(NSAttributedString(
                        string: "\nStatus: \(status.text)\n\n",
                        attributes: activityStatusAttributes(color: status.color)))
                }
            }
        }
        boundaries.append(document.length)
        return (document, latestMarkerRange, boundaries)
    }

    private enum ActivityCodeCardStyle {
        case request
        case response
        case modelInput
    }

    private static var modelInputAccentColor: NSColor {
        NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                ? NSColor(srgbRed: 0.89, green: 0.51, blue: 0.51, alpha: 1)
                : NSColor(srgbRed: 0.64, green: 0.20, blue: 0.20, alpha: 1)
        }
    }

    private static func formattedModelInputBody(_ body: String) -> String {
        var result = ""
        var followsToolHeader = false
        var expectsJSON = false
        body.enumerateSubstrings(in: body.startIndex..<body.endIndex, options: .byLines) {
            line, range, enclosingRange, _ in
            guard let line else { return }
            if expectsJSON, let formatted = indentedJSONLine(line) {
                result += formatted
                result += body[range.upperBound..<enclosingRange.upperBound]
            } else {
                result += body[enclosingRange]
            }
            expectsJSON = followsToolHeader && line.hasPrefix("Call: ")
            followsToolHeader = line.hasPrefix("Tool result: ")
        }
        return result
    }

    /// Add whitespace outside strings only: preserve every original JSON token,
    /// including number spelling and escapes. Large/deep results remain complete
    /// and wrap normally instead of creating an oversized formatted copy.
    private static func indentedJSONLine(_ line: String) -> String? {
        guard line.utf8.count <= 16 * 1_024, line.first == "{",
              (try? JSONSerialization.jsonObject(with: Data(line.utf8))) != nil else { return nil }
        let bytes = Array(line.utf8)
        var output: [UInt8] = []
        var depth = 0
        var inString = false
        var escaped = false
        var previous: UInt8?
        func newline() {
            output.append(10)
            output.append(contentsOf: repeatElement(32, count: min(depth, 12) * 2))
        }
        for byte in bytes {
            if inString {
                output.append(byte)
                if escaped { escaped = false }
                else if byte == 92 { escaped = true }
                else if byte == 34 { inString = false }
            } else {
                switch byte {
                case 9, 10, 13, 32: continue
                case 34:
                    inString = true
                    output.append(byte)
                case 123, 91:
                    output.append(byte)
                    depth += 1
                    newline()
                case 125, 93:
                    depth -= 1
                    if previous == 123 || previous == 91 {
                        output.removeLast(1 + min(depth + 1, 12) * 2)
                    } else {
                        newline()
                    }
                    output.append(byte)
                case 44:
                    output.append(byte)
                    newline()
                case 58: output.append(contentsOf: [58, 32])
                default: output.append(byte)
                }
            }
            previous = byte
            guard output.count <= 64 * 1_024 else { return nil }
        }
        return String(decoding: output, as: UTF8.self)
    }

    private static func activityCodeCard(
        body: String,
        status: (text: String, color: NSColor)? = nil,
        cardStyle: ActivityCodeCardStyle = .request
    ) -> NSAttributedString {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 2
        style.paragraphSpacing = 0
        if cardStyle == .modelInput { style.lineBreakMode = .byCharWrapping }

        // Match the native table-cell mechanism used by Markdown code blocks.
        // One shared cell joins all JSON paragraphs into a full-width panel.
        let table = NSTextTable()
        table.numberOfColumns = 1
        table.collapsesBorders = true
        if cardStyle == .modelInput {
            table.layoutAlgorithm = .fixedLayoutAlgorithm
            table.setContentWidth(100, type: .percentageValueType)
        }
        let block = NSTextTableBlock(
            table: table, startingRow: 0, rowSpan: 1,
            startingColumn: 0, columnSpan: 1)
        block.setContentWidth(100, type: .percentageValueType)
        if cardStyle == .modelInput {
            block.setValue(0, type: .absoluteValueType, for: .minimumWidth)
            block.setValue(100, type: .percentageValueType, for: .maximumWidth)
        }
        block.backgroundColor = NSColor(name: nil) { appearance in
            if cardStyle == .modelInput {
                return appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                    ? NSColor(srgbRed: 0.28, green: 0.15, blue: 0.17, alpha: 1)
                    : NSColor(srgbRed: 1, green: 0.93, blue: 0.93, alpha: 1)
            }
            if cardStyle == .response {
                return appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                    ? NSColor(srgbRed: 0.12, green: 0.20, blue: 0.30, alpha: 1)
                    : NSColor(srgbRed: 0.89, green: 0.94, blue: 1, alpha: 1)
            }
            return NSColor(calibratedWhite:
                appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? 0.18 : 0.94,
                alpha: 1)
        }
        block.setWidth(9, type: .absoluteValueType, for: .padding)
        block.setWidth(1, type: .absoluteValueType, for: .border)
        block.setBorderColor(NSColor(name: nil) { appearance in
            if cardStyle == .modelInput {
                return appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                    ? NSColor(srgbRed: 0.66, green: 0.36, blue: 0.38, alpha: 1)
                    : NSColor(srgbRed: 0.76, green: 0.42, blue: 0.42, alpha: 1)
            }
            if cardStyle == .response {
                return appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                    ? NSColor(srgbRed: 0.34, green: 0.61, blue: 0.91, alpha: 1)
                    : NSColor(srgbRed: 0.24, green: 0.46, blue: 0.74, alpha: 1)
            }
            return NSColor(calibratedWhite:
                appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? 0.40 : 0.76,
                alpha: 1)
        })
        style.textBlocks = [block]

        let card = NSMutableAttributedString(
            string: body,
            attributes: [
                .font: NSFont.monospacedSystemFont(
                    ofSize: NSFont.smallSystemFontSize, weight: .regular),
                .foregroundColor: NSColor.labelColor,
                .paragraphStyle: style,
            ])
        if let status {
            card.append(NSAttributedString(
                string: "\nStatus: \(status.text)\n",
                attributes: [
                    .font: NSFont.monospacedSystemFont(
                        ofSize: NSFont.smallSystemFontSize, weight: .semibold),
                    .foregroundColor: status.color,
                    .paragraphStyle: style,
                ]))
        } else {
            card.append(NSAttributedString(string: "\n", attributes: [.paragraphStyle: style]))
        }
        return card
    }

    private static func activityStatus(
        _ status: AppAgentActivity.Status,
        elapsedSeconds: Double?
    ) -> (text: String, color: NSColor) {
        let presentation: (text: String, color: NSColor) = switch status {
        case .dispatching:
            ("Dispatching", .secondaryLabelColor)
        case .succeeded:
            ("Succeeded", .systemGreen)
        case .recoverablePreDispatchRefusal(let outcome):
            ("Recoverable pre-dispatch refusal · \(outcome.description)", .systemOrange)
        case .serverOutcome(let outcome):
            (outcome.description, outcome.verdict == "verified" ? .systemGreen
                : outcome.verdict == "inconclusive" ? .systemOrange : .systemRed)
        case .localFailure(let reason):
            (reason, .systemRed)
        case .cancelled:
            ("Cancelled", .secondaryLabelColor)
        case .notSent:
            ("Not sent", .systemOrange)
        }
        guard let elapsedSeconds else { return presentation }
        return (
            "\(presentation.text) · \(String(format: "%.2f s", elapsedSeconds))",
            presentation.color)
    }

    private static func activityLabelAttributes(
        color: NSColor
    ) -> [NSAttributedString.Key: Any] {
        labelAttributes(color: color)
    }

    private static func activityBodyAttributes() -> [NSAttributedString.Key: Any] {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 2
        return [
            .font: NSFont.monospacedSystemFont(
                ofSize: NSFont.smallSystemFontSize, weight: .regular),
            .foregroundColor: NSColor.labelColor,
            .paragraphStyle: style,
        ]
    }

    private static func activityStatusAttributes(
        color: NSColor
    ) -> [NSAttributedString.Key: Any] {
        let style = NSMutableParagraphStyle()
        style.paragraphSpacing = 4
        return [
            .font: NSFont.systemFont(
                ofSize: NSFont.smallSystemFontSize, weight: .medium),
            .foregroundColor: color,
            .paragraphStyle: style,
        ]
    }

    private static func userLabelAttributes() -> [NSAttributedString.Key: Any] {
        labelAttributes(color: .secondaryLabelColor)
    }

    private static func assistantLabelAttributes() -> [NSAttributedString.Key: Any] {
        labelAttributes(color: TurboFieldfareMacTheme.accentNSColor)
    }

    private static func labelAttributes(
        color: NSColor
    ) -> [NSAttributedString.Key: Any] {
        let style = NSMutableParagraphStyle()
        style.paragraphSpacing = 5
        return [
            .font: NSFont.systemFont(
                ofSize: NSFont.smallSystemFontSize,
                weight: .semibold),
            .foregroundColor: color,
            .paragraphStyle: style,
        ]
    }

    private static func contextBreakAttributes() -> [NSAttributedString.Key: Any] {
        let style = NSMutableParagraphStyle()
        style.alignment = .center
        style.paragraphSpacingBefore = 8
        style.paragraphSpacing = 8
        return [
            .font: NSFont.systemFont(
                ofSize: NSFont.smallSystemFontSize, weight: .regular),
            .foregroundColor: NSColor.tertiaryLabelColor,
            .paragraphStyle: style,
        ]
    }

    private static func promptAttributes() -> [NSAttributedString.Key: Any] {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 3
        return [
            .font: NSFont.systemFont(ofSize: NSFont.systemFontSize),
            .foregroundColor: NSColor.secondaryLabelColor,
            .paragraphStyle: style,
        ]
    }

    private static func responseAttributes() -> [NSAttributedString.Key: Any] {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 3
        style.paragraphSpacing = 6
        return [
            .font: NSFont.systemFont(ofSize: NSFont.systemFontSize),
            .foregroundColor: NSColor.labelColor,
            .paragraphStyle: style,
        ]
    }

    private static func prefillPlaceholderAttributes() -> [NSAttributedString.Key: Any] {
        var attributes = responseAttributes()
        attributes[.foregroundColor] = NSColor.secondaryLabelColor
        return attributes
    }

    private static func prefillPlaceholder(dotCount: Int) -> String {
        "Processing your prompt" + String(repeating: ".", count: dotCount)
    }
}

extension Array {
    fileprivate subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
