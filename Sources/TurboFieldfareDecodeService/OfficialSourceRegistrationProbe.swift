import Darwin
import Foundation
import TurboFieldfareFormat
import TurboFieldfareOfficialQwenSource

/// Retains protected registration and source-root directory handles across
/// service load awaits. The digest describes the pinned descriptor only; model
/// trust still belongs to the independently admitted runtime and is checked
/// again against its own receipt immediately before ready publication.
/// OfficialSourceHandle owns immutable FDs and synchronizes its accepted-file
/// identities; this wrapper never exposes either handle or arbitrary reads.
final class OfficialSourceRegistrationProbe: @unchecked Sendable {
    enum Rejection: Error, Equatable {
        case markerUnavailable(Int32)
        case descriptorChanged
    }

    let contentDigest: String
    private let handle: OfficialSourceHandle

    private init(handle: OfficialSourceHandle, contentDigest: String) {
        self.handle = handle
        self.contentDigest = contentDigest
    }

    /// A genuinely absent marker denotes a packed candidate. A symlink,
    /// inaccessible marker, or changed source is never downgraded to packed.
    /// The runtime separately classifies and rejects ambiguous directories.
    static func openIfSource(directoryURL: URL) throws -> Self? {
        let marker = directoryURL.appendingPathComponent(
            OfficialSourceDescriptor.markerFilename).path
        var info = stat()
        if lstat(marker, &info) != 0 {
            let failure = errno
            if failure == ENOENT { return nil }
            throw Rejection.markerUnavailable(failure)
        }
        let handle = try OfficialSourceHandle(registrationURL: directoryURL)
        let descriptor = try handle.registeredDescriptor()
        try handle.validateBinding()
        return Self(handle: handle, contentDigest: descriptor.contentSHA256)
    }

    /// Reopens the literal names and compares their identities and capped
    /// marker bytes with the retained FDs; never interprets a mutable path as
    /// a replacement for the source that the model actually admitted.
    func revalidate() throws {
        try handle.validateBinding()
        guard try handle.registeredDescriptor().contentSHA256 == contentDigest else {
            throw Rejection.descriptorChanged
        }
        try handle.validateBinding()
    }
}
