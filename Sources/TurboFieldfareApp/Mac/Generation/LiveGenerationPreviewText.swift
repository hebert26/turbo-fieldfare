import AppKit
import SwiftUI
import TurboFieldfareMacPresentation

/// SwiftUI measures a fixed viewport. The native text system owns layout and
/// selection for the existing bounded preview, without changing model text.
struct LiveGenerationPreviewText: NSViewRepresentable {
    let text: String
    let monospaced: Bool
    let accessibilityID: String
    @Binding var followsLatest: Bool

    func makeCoordinator() -> Coordinator { Coordinator(followsLatest: $followsLatest) }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.drawsBackground = false
        let contentSize = scroll.contentSize
        let view = NSTextView(frame: NSRect(origin: .zero, size: contentSize))
        view.minSize = NSSize(width: 0, height: contentSize.height)
        view.maxSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude)
        view.isEditable = false
        view.isSelectable = true
        view.isRichText = false
        view.allowsUndo = false
        view.drawsBackground = false
        view.isVerticallyResizable = true
        view.isHorizontallyResizable = false
        view.autoresizingMask = [.width]
        view.textContainerInset = .zero
        view.textContainer?.widthTracksTextView = true
        view.textContainer?.containerSize = NSSize(
            width: contentSize.width,
            height: CGFloat.greatestFiniteMagnitude)
        view.textContainer?.lineFragmentPadding = 0
        // This short preview must finish layout before its height changes.
        view.layoutManager?.allowsNonContiguousLayout = false
        view.isAutomaticLinkDetectionEnabled = false
        view.isAutomaticDataDetectionEnabled = false
        view.setAccessibilityIdentifier(accessibilityID)
        view.setAccessibilityLabel("Recent generation text")
        scroll.documentView = view
        context.coordinator.attach(scroll: scroll, textView: view)
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        context.coordinator.followsLatest = $followsLatest
        let bodyFont = NSFont.preferredFont(forTextStyle: .body)
        let font = monospaced
            ? NSFont.monospacedSystemFont(ofSize: bodyFont.pointSize, weight: .regular) : bodyFont
        context.coordinator.update(text: text, font: font)
    }

    static func dismantleNSView(_ scroll: NSScrollView, coordinator: Coordinator) {
        coordinator.detach()
    }

    @MainActor
    final class Coordinator: NSObject, NSTextViewDelegate {
        var followsLatest: Binding<Bool>
        private weak var scroll: NSScrollView?
        private weak var textView: NSTextView?
        private var updating = false
        private var lastFollow = false

        init(followsLatest: Binding<Bool>) { self.followsLatest = followsLatest }

        func attach(scroll: NSScrollView, textView: NSTextView) {
            self.scroll = scroll
            self.textView = textView
            textView.delegate = self
            for name in [NSScrollView.willStartLiveScrollNotification, NSScrollView.didLiveScrollNotification] {
                NotificationCenter.default.addObserver(
                    self, selector: #selector(readerTookOver), name: name, object: scroll)
            }
        }

        func detach() {
            NotificationCenter.default.removeObserver(self)
            textView?.delegate = nil
            textView = nil
            scroll = nil
        }

        @objc private func readerTookOver() {
            guard !updating else { return }
            lastFollow = false
            if followsLatest.wrappedValue { followsLatest.wrappedValue = false }
        }

        func textViewDidChangeSelection(_ notification: Notification) { readerTookOver() }

        func update(text: String, font: NSFont) {
            guard let view = textView, let storage = view.textStorage else { return }
            let changed = !storage.string.utf8.elementsEqual(text.utf8)
            let fontChanged = view.font != font
            let shouldFollow = followsLatest.wrappedValue
            let resumed = shouldFollow && !lastFollow
            lastFollow = shouldFollow
            guard changed || fontChanged || resumed else { return }
            updating = true
            defer { updating = false }
            let origin = scroll?.contentView.bounds.origin
            var selection = view.selectedRanges.map(\.rangeValue)
            if changed {
                let old = Array(storage.string.utf16)
                let new = Array(text.utf16)
                let overlap = Self.retainedOverlap(old: old, new: new)
                let incoming = text as NSString
                let attributes: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.labelColor]
                storage.beginEditing()
                if overlap > 0 {
                    // Streaming append, including a rolling prefix eviction.
                    let removed = old.count - overlap
                    storage.deleteCharacters(in: NSRange(location: 0, length: removed))
                    storage.append(NSAttributedString(string: incoming.substring(from: overlap), attributes: attributes))
                    selection = selection.map {
                        let start = max(0, $0.location - removed)
                        let end = max(0, NSMaxRange($0) - removed)
                        return NSRange(location: start, length: end - start)
                    }
                } else {
                    // A new step or changed draft may retain only its prefix.
                    var prefix = 0
                    while prefix < min(old.count, new.count), old[prefix] == new[prefix] { prefix += 1 }
                    if prefix > 0, prefix < new.count, (0xDC00...0xDFFF).contains(new[prefix]) {
                        prefix -= 1
                    }
                    let replaced = NSRange(location: prefix, length: old.count - prefix)
                    let replacement = incoming.substring(from: prefix)
                    storage.replaceCharacters(in: replaced, with: NSAttributedString(string: replacement, attributes: attributes))
                    selection = InstructionTranscriptDocumentController.adjustedRanges(
                        selection, replacing: replaced, newLength: new.count - prefix)
                }
                storage.endEditing()
                view.selectedRanges = InstructionTranscriptDocumentController.clampedRanges(
                    selection, toLength: storage.length).map(NSValue.init(range:))
            }
            if fontChanged { view.font = font }
            if shouldFollow {
                view.scrollRangeToVisible(NSRange(location: storage.length, length: 0))
            } else if let origin, let scroll {
                scroll.contentView.scroll(to: origin)
                scroll.reflectScrolledClipView(scroll.contentView)
            }
        }

        /// Longest old suffix equal to a new prefix, in linear time and bounded
        /// scratch. This preserves selected characters when the 8 KiB rolls.
        private static func retainedOverlap(old: [UInt16], new: [UInt16]) -> Int {
            guard !new.isEmpty else { return 0 }
            var fallback = [Int](repeating: 0, count: new.count)
            var matched = 0
            for index in new.indices.dropFirst() {
                while matched > 0, new[index] != new[matched] { matched = fallback[matched - 1] }
                if new[index] == new[matched] { matched += 1 }
                fallback[index] = matched
            }
            matched = 0
            for unit in old {
                while matched > 0, matched == new.count || unit != new[matched] {
                    matched = fallback[matched - 1]
                }
                if unit == new[matched] { matched += 1 }
            }
            // Never split a surrogate pair when identifying retained text.
            if matched > 0, matched < new.count, (0xDC00...0xDFFF).contains(new[matched]) { return 0 }
            return matched
        }
    }
}
