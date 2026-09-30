import Darwin
import Foundation
import TurboFieldfareFormat

/// Metadata-only registration. A marker identifies a pinned source inventory,
/// not files authenticated at its sourceRoot or a model ready for loading.
public enum OfficialSourceRegistration {
    public enum RegistrationError: Error, Sendable, Equatable {
        case unsafePath(String)
        case conflict(String)
        case invalidLayout(String)
        case io(path: String, errno: Int32)
        /// The atomic rename committed a complete marker, but syncing the
        /// parent failed. The caller must inspect before deciding what to do;
        /// it must not retry as though no registration were published.
        case publishedDurabilityUnknown(path: String, errno: Int32)
    }

    package enum PublicationCheckpoint: Sendable {
        case stageCreated
        case markerSynced
        case beforePublish
    }

    /// Publish exactly one bounded marker in a new logical model directory.
    /// The source directory is checked by metadata only; no child is opened.
    public static func register(markerData: Data, at logicalModelURL: URL) throws -> OfficialSourceDescriptor {
        let support = try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: false)
        return try register(markerData: markerData, at: logicalModelURL,
                            checkpoint: { _ in }, syncParent: { fsync($0) },
                            allowedApplicationSupportURL: support)
    }

    /// Package-only deterministic interruption seam. All checkpoints precede
    /// the atomic commit; an injected parent-sync failure is reported as
    /// publishedDurabilityUnknown, never as an unpublished failure. Neither
    /// operation substitutes a descriptor, digest, source pin or payload proof.
    package static func register(
        markerData: Data,
        at logicalModelURL: URL,
        checkpoint: (PublicationCheckpoint) throws -> Void,
        syncParent: (Int32) -> Int32 = { fsync($0) },
        allowedApplicationSupportURL: URL? = nil
    ) throws -> OfficialSourceDescriptor {
        try Task.checkCancellation()
        let descriptor = try OfficialSourceDescriptor.decodeStrict(data: markerData)
        try OfficialSourceDescriptorValidation.validate(descriptor)
        let destination = try components(logicalModelURL)
        let source = try components(URL(fileURLWithPath: descriptor.sourceRoot, isDirectory: true))
        guard !isPrefix(source, of: destination), !isPrefix(destination, of: source) else {
            throw RegistrationError.unsafePath("source and logical model directories overlap")
        }
        let sourceFD = try openDirectory(source)
        defer { close(sourceFD) }
        guard let leaf = destination.last else {
            throw RegistrationError.unsafePath("missing destination basename")
        }
        let parentParts = Array(destination.dropLast())
        let parentFD = try openRegistrationParent(
            parentParts, destinationLeaf: leaf,
            allowedApplicationSupportURL: allowedApplicationSupportURL)
        defer { close(parentFD) }
        // Match RepackCore InstallLock: the same sibling lock filename and
        // nonblocking exclusive flock, held across conflict checks and rename.
        let lockName = leaf.precomposedStringWithCanonicalMapping + ".install.lock"
        let lockFD = try acquireInstallerLock(parentFD: parentFD, name: lockName)
        defer { _ = flock(lockFD, LOCK_UN); close(lockFD) }
        try requireAvailable(leaf, parentFD: parentFD)
        let encoded = try JSONEncoder().encode(descriptor)
        guard UInt64(encoded.count) <= OfficialSourceDescriptor.maximumMarkerBytes else {
            throw RegistrationError.invalidLayout("marker exceeds bounded metadata limit")
        }
        _ = try OfficialSourceDescriptor.decodeStrict(data: encoded)
        try Task.checkCancellation()
        let stage = ".official-source-stage-\(UUID().uuidString)"
        guard mkdirat(parentFD, stage, 0o700) == 0 else {
            throw RegistrationError.io(path: stage, errno: errno)
        }
        let stageFD = openat(parentFD, stage, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard stageFD >= 0 else {
            let problem = errno
            _ = unlinkat(parentFD, stage, AT_REMOVEDIR)
            throw RegistrationError.io(path: stage, errno: problem)
        }
        var ownsStage = true
        defer {
            if ownsStage {
                var held = stat(), named = stat()
                if fstat(stageFD, &held) == 0,
                   fstatat(parentFD, stage, &named, AT_SYMLINK_NOFOLLOW) == 0,
                   held.st_dev == named.st_dev, held.st_ino == named.st_ino {
                    _ = unlinkat(stageFD, OfficialSourceDescriptor.markerFilename, 0)
                    _ = unlinkat(parentFD, stage, AT_REMOVEDIR)
                }
            }
            close(stageFD)
        }
        try checkpoint(.stageCreated)
        let marker = OfficialSourceDescriptor.markerFilename
        let markerFD = openat(stageFD, marker, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard markerFD >= 0 else { throw RegistrationError.io(path: marker, errno: errno) }
        var writtenMarker = stat()
        do {
            try encoded.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                var offset = 0
                while offset < raw.count {
                    try Task.checkCancellation()
                    let count = write(markerFD, base.advanced(by: offset), raw.count - offset)
                    if count < 0 && errno == EINTR { continue }
                    guard count > 0 else { throw RegistrationError.io(path: marker, errno: count < 0 ? errno : EIO) }
                    offset += count
                }
            }
            guard fsync(markerFD) == 0 else { throw RegistrationError.io(path: marker, errno: errno) }
            guard fstat(markerFD, &writtenMarker) == 0 else {
                throw RegistrationError.io(path: marker, errno: errno)
            }
            guard (writtenMarker.st_mode & S_IFMT) == S_IFREG,
                  writtenMarker.st_size == off_t(encoded.count) else {
                throw RegistrationError.invalidLayout("written marker changed during staging")
            }
        } catch {
            close(markerFD)
            throw error
        }
        close(markerFD)
        guard fsync(stageFD) == 0 else { throw RegistrationError.io(path: stage, errno: errno) }
        try checkpoint(.markerSynced)
        try Task.checkCancellation()
        try checkpoint(.beforePublish)
        try Task.checkCancellation()
        try checkSameDirectory(parentParts, fd: parentFD)
        try checkSameDirectory(source, fd: sourceFD)
        try checkLock(parentFD: parentFD, name: lockName, fd: lockFD)
        try requireAvailable(leaf, parentFD: parentFD)
        try checkSameDirectoryEntry(parentFD: parentFD, leaf: stage, heldFD: stageFD)
        try requireSingleMarker(stageFD, marker: marker)
        var namedMarker = stat()
        guard fstatat(stageFD, marker, &namedMarker, AT_SYMLINK_NOFOLLOW) == 0 else {
            throw RegistrationError.io(path: marker, errno: errno)
        }
        guard sameFile(writtenMarker, namedMarker) else {
            throw RegistrationError.invalidLayout("staged marker changed before publication")
        }
        // Unlike rename(), RENAME_EXCL cannot replace an existing installation,
        // even when another process creates it between the check and publication.
        guard renameatx_np(parentFD, stage, parentFD, leaf, UInt32(RENAME_EXCL)) == 0 else {
            if errno == EEXIST { throw RegistrationError.conflict(leaf) }
            throw RegistrationError.io(path: leaf, errno: errno)
        }
        ownsStage = false
        // Publication is the cancellation commit point. After this point no
        // cancellation check or staging cleanup can imply the rename failed.
        errno = 0
        if syncParent(parentFD) != 0 {
            throw RegistrationError.publishedDurabilityUnknown(
                path: logicalModelURL.path, errno: errno == 0 ? EIO : errno)
        }
        return descriptor
    }

    /// Read back a marker-only registration or the defined Task 6.8
    /// marker-plus-receipt layout. A receipt's presence does not imply trust:
    /// only OfficialSourceTrust can validate its binding and fingerprints.
    public static func inspect(at logicalModelURL: URL) throws -> OfficialSourceDescriptor {
        try inspect(at: logicalModelURL, observeIO: nil)
    }

    /// Same production inspection, with package-only syscall observations.
    /// The observer cannot provide descriptors, bytes, or verified status.
    package static func inspect(
        at logicalModelURL: URL,
        observeIO: ((OfficialSourceTrustIOEvent) -> Void)?
    ) throws -> OfficialSourceDescriptor {
        try Task.checkCancellation()
        let destination = try components(logicalModelURL)
        let directoryFD = try openDirectory(destination)
        defer { close(directoryFD) }
        let marker = OfficialSourceDescriptor.markerFilename
        try requireRegistrationEntries(directoryFD, marker: marker,
                                       path: logicalModelURL.path,
                                       observeIO: observeIO)
        let markerPath = logicalModelURL.path + "/" + marker
        observeIO?(.openFile(path: markerPath))
        let fd = openat(directoryFD, marker, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else { throw RegistrationError.io(path: marker, errno: errno) }
        defer { close(fd) }
        var before = stat()
        guard fstat(fd, &before) == 0 else { throw RegistrationError.io(path: marker, errno: errno) }
        guard (before.st_mode & S_IFMT) == S_IFREG, before.st_size >= 0,
              UInt64(before.st_size) <= OfficialSourceDescriptor.maximumMarkerBytes else {
            throw RegistrationError.invalidLayout("marker is not a bounded regular file")
        }
        var bytes = Data(count: Int(before.st_size))
        try bytes.withUnsafeMutableBytes { raw in
            var offset = 0
            while offset < raw.count {
                try Task.checkCancellation()
                observeIO?(.readFile(path: markerPath, offset: UInt64(offset),
                                     requestedBytes: raw.count - offset))
                let count = pread(fd, raw.baseAddress?.advanced(by: offset), raw.count - offset, off_t(offset))
                if count < 0 && errno == EINTR { continue }
                guard count > 0 else { throw RegistrationError.io(path: marker, errno: count < 0 ? errno : EIO) }
                offset += count
            }
        }
        var after = stat()
        var entry = stat()
        observeIO?(.statFile(path: markerPath))
        guard fstat(fd, &after) == 0,
              fstatat(directoryFD, marker, &entry, AT_SYMLINK_NOFOLLOW) == 0 else {
            throw RegistrationError.io(path: marker, errno: errno)
        }
        guard sameFile(before, after), sameFile(before, entry) else {
            throw RegistrationError.invalidLayout("marker changed during read")
        }
        let descriptor = try OfficialSourceDescriptor.decodeStrict(data: bytes)
        try OfficialSourceDescriptorValidation.validate(descriptor)
        let source = try components(URL(fileURLWithPath: descriptor.sourceRoot, isDirectory: true))
        guard !isPrefix(source, of: destination), !isPrefix(destination, of: source) else {
            throw RegistrationError.unsafePath("source and logical model directories overlap")
        }
        let sourceFD = try openDirectory(source)
        close(sourceFD)
        try Task.checkCancellation()
        return descriptor
    }

    private static func components(_ url: URL) throws -> [String] {
        guard url.isFileURL else { throw RegistrationError.unsafePath("not a file URL") }
        let path = url.path
        guard path.hasPrefix("/"), path != "/", !path.contains("\0"),
              !path.hasSuffix("/"), !path.contains("//") else {
            throw RegistrationError.unsafePath("not a safe absolute directory path")
        }
        let pieces = path.dropFirst().split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard pieces.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw RegistrationError.unsafePath("path contains dot or empty component")
        }
        return pieces
    }

    private static func isPrefix(_ prefix: [String], of path: [String]) -> Bool {
        guard prefix.count <= path.count else { return false }
        return zip(prefix, path).allSatisfy {
            GTurboPathValidator.appleFilesystemKey($0.0)
                == GTurboPathValidator.appleFilesystemKey($0.1)
        }
    }

    /// Create only the missing app-owned TurboFieldfare directory under an
    /// existing, no-follow Application Support root. Explicit/checkout parents
    /// must already exist. A created parent is retained on later failure: it
    /// may have acquired unrelated content and is never task-owned staging.
    private static func openRegistrationParent(
        _ pieces: [String], destinationLeaf: String,
        allowedApplicationSupportURL: URL?
    ) throws -> Int32 {
        do { return try openDirectory(pieces) }
        catch {
            guard let registrationError = error as? RegistrationError,
                  case .io(_, let code) = registrationError, code == ENOENT,
                  let support = allowedApplicationSupportURL,
                  let name = pieces.last, name == "TurboFieldfare",
                  destinationLeaf == "qwen3.6-35b-a3b.gturbo" else { throw error }
            let supportParts = try components(support)
            guard Array(pieces.dropLast()) == supportParts else { throw error }
            let supportFD = try openDirectory(supportParts)
            defer { close(supportFD) }
            let result = mkdirat(supportFD, name, 0o700)
            if result != 0 && errno != EEXIST {
                throw RegistrationError.io(path: support.path + "/" + name, errno: errno)
            }
            let parentFD = try openDirectory(pieces)
            do {
                if result == 0 && fsync(supportFD) != 0 {
                    throw RegistrationError.io(path: support.path, errno: errno)
                }
                return parentFD
            } catch {
                close(parentFD)
                throw error
            }
        }
    }

    private static func acquireInstallerLock(parentFD: Int32, name: String) throws -> Int32 {
        let fd = openat(parentFD, name, O_RDWR | O_CREAT | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw RegistrationError.io(path: name, errno: errno) }
        do {
            var info = stat()
            guard fstat(fd, &info) == 0 else {
                throw RegistrationError.io(path: name, errno: errno)
            }
            guard (info.st_mode & S_IFMT) == S_IFREG, info.st_nlink == 1 else {
                throw RegistrationError.invalidLayout("installer lock is not a single-link regular file")
            }
            guard flock(fd, LOCK_EX | LOCK_NB) == 0 else {
                if errno == EWOULDBLOCK { throw RegistrationError.conflict(name) }
                throw RegistrationError.io(path: name, errno: errno)
            }
            try checkLock(parentFD: parentFD, name: name, fd: fd)
            return fd
        } catch {
            _ = flock(fd, LOCK_UN)
            close(fd)
            throw error
        }
    }

    private static func checkLock(parentFD: Int32, name: String, fd: Int32) throws {
        var held = stat(), named = stat()
        guard fstat(fd, &held) == 0,
              fstatat(parentFD, name, &named, AT_SYMLINK_NOFOLLOW) == 0 else {
            throw RegistrationError.io(path: name, errno: errno)
        }
        guard (held.st_mode & S_IFMT) == S_IFREG, named.st_mode == held.st_mode,
              held.st_nlink == 1, named.st_nlink == 1,
              held.st_dev == named.st_dev, held.st_ino == named.st_ino else {
            throw RegistrationError.invalidLayout("installer lock changed or is not regular")
        }
    }

    private static func openDirectory(_ pieces: [String]) throws -> Int32 {
        var fd = open("/", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else { throw RegistrationError.io(path: "/", errno: errno) }
        var path = ""
        for piece in pieces {
            path += "/" + piece
            let next = openat(fd, piece, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
            if next < 0 {
                let problem = errno
                close(fd)
                throw RegistrationError.io(path: path, errno: problem)
            }
            close(fd)
            fd = next
        }
        return fd
    }

    private static func checkSameDirectory(_ pieces: [String], fd: Int32) throws {
        let current = try openDirectory(pieces)
        defer { close(current) }
        var a = stat(), b = stat()
        guard fstat(fd, &a) == 0, fstat(current, &b) == 0 else {
            throw RegistrationError.io(path: "parent", errno: errno)
        }
        guard a.st_dev == b.st_dev, a.st_ino == b.st_ino else {
            throw RegistrationError.invalidLayout("parent directory changed before publication")
        }
    }

    private static func checkSameDirectoryEntry(parentFD: Int32, leaf: String, heldFD: Int32) throws {
        var held = stat(), named = stat()
        guard fstat(heldFD, &held) == 0,
              fstatat(parentFD, leaf, &named, AT_SYMLINK_NOFOLLOW) == 0 else {
            throw RegistrationError.io(path: leaf, errno: errno)
        }
        guard (held.st_mode & S_IFMT) == S_IFDIR,
              held.st_dev == named.st_dev, held.st_ino == named.st_ino,
              held.st_mode == named.st_mode else {
            throw RegistrationError.invalidLayout("staging directory changed before publication")
        }
    }

    private static func requireAvailable(_ leaf: String, parentFD: Int32) throws {
        // The pre-existing packed installer uses these siblings for work in
        // progress. Never publish a source marker beside ambiguous resume state.
        for name in [leaf, leaf + ".partial", leaf + ".resume.json"] {
            try requireAbsent(name, parentFD: parentFD)
        }
    }

    private static func requireAbsent(_ leaf: String, parentFD: Int32) throws {
        var info = stat()
        if fstatat(parentFD, leaf, &info, AT_SYMLINK_NOFOLLOW) == 0 {
            throw RegistrationError.conflict(leaf)
        }
        guard errno == ENOENT else { throw RegistrationError.io(path: leaf, errno: errno) }
    }

    private static func requireSingleMarker(_ fd: Int32, marker: String) throws {
        try checkEntries(fd, marker: marker, allowReceipt: false)
    }

    private static func requireRegistrationEntries(
        _ fd: Int32, marker: String, path: String,
        observeIO: ((OfficialSourceTrustIOEvent) -> Void)?
    ) throws {
        try checkEntries(fd, marker: marker, allowReceipt: true,
                         path: path, observeIO: observeIO)
    }

    private static func checkEntries(
        _ fd: Int32, marker: String, allowReceipt: Bool,
        path: String? = nil,
        observeIO: ((OfficialSourceTrustIOEvent) -> Void)? = nil
    ) throws {
        let copy = dup(fd)
        guard copy >= 0 else { throw RegistrationError.io(path: marker, errno: errno) }
        guard let stream = fdopendir(copy) else {
            let problem = errno
            close(copy)
            throw RegistrationError.io(path: marker, errno: problem)
        }
        defer { closedir(stream) }
        // dup shares the retained directory offset; every inspection must
        // enumerate from the beginning, including a second readback.
        rewinddir(stream)
        var seen = false
        var seenReceipt = false
        errno = 0
        while let item = readdir(stream) {
            let name = withUnsafeBytes(of: item.pointee.d_name) { raw -> String in
                let bytes = raw.prefix { $0 != 0 }
                return String(decoding: bytes, as: UTF8.self)
            }
            if name == "." || name == ".." { continue }
            if name == marker, !seen {
                seen = true
            } else if allowReceipt, name == OfficialSourceTrust.receiptFilename,
                      !seenReceipt {
                var receipt = stat()
                if let path { observeIO?(.statFile(path: path + "/" + name)) }
                guard fstatat(fd, name, &receipt, AT_SYMLINK_NOFOLLOW) == 0 else {
                    throw RegistrationError.io(path: name, errno: errno)
                }
                guard (receipt.st_mode & S_IFMT) == S_IFREG,
                      receipt.st_nlink == 1, receipt.st_size >= 0,
                      UInt64(receipt.st_size) <= OfficialSourceTrust.maximumReceiptBytes else {
                    throw RegistrationError.invalidLayout("source receipt is not a bounded regular file")
                }
                seenReceipt = true
            } else {
                throw RegistrationError.invalidLayout("source registration contains another entry")
            }
            errno = 0
        }
        guard errno == 0 else { throw RegistrationError.io(path: marker, errno: errno) }
        guard seen else { throw RegistrationError.invalidLayout("source marker is missing") }
    }

    private static func sameFile(_ a: stat, _ b: stat) -> Bool {
        a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_mode == b.st_mode
            && a.st_size == b.st_size
            && a.st_mtimespec.tv_sec == b.st_mtimespec.tv_sec
            && a.st_mtimespec.tv_nsec == b.st_mtimespec.tv_nsec
            && a.st_ctimespec.tv_sec == b.st_ctimespec.tv_sec
            && a.st_ctimespec.tv_nsec == b.st_ctimespec.tv_nsec
    }
}
