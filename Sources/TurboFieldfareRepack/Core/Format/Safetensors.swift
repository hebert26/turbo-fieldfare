import Foundation
import Darwin
import TurboFieldfareOfficialQwenSource

/// Compatibility adapter for installer callers. The shared source module owns
/// Safetensors metadata validation; RepackCore retains its tensor and error API.
enum Safetensors {
    static let maxHeaderBytes = OfficialSafetensorsSource.maximumHeaderBytes

    struct Header {
        let path: String
        let payloadBaseOffset: UInt64
        let tensors: [SourceTensor]
    }

    /// Remote installer temporary files retain their historical open policy.
    static func parseHeader(path: String) throws -> Header {
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_size >= 0 else {
            throw RepackError.fileStatFailed(path: path, errno: errno == 0 ? EINVAL : errno)
        }
        return try convert(readSharedHeader(fd: fd, path: path, fileSize: UInt64(info.st_size)))
    }

    /// A local untrusted leaf cannot follow a symlink or block on a FIFO.
    static func parseLocalHeader(path: String) throws -> Header {
        try convert(parseLocalHeaderShared(path: path))
    }

    static func parseLocalHeaderShared(path: String) throws -> OfficialSafetensorsSource.Header {
        let fd = open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        if fd < 0 { throw RepackError.fileOpenFailed(path: path, errno: errno) }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG,
              info.st_size >= 0 else {
            throw RepackError.fileStatFailed(path: path, errno: errno == 0 ? EINVAL : errno)
        }
        return try readSharedHeader(fd: fd, path: path, fileSize: UInt64(info.st_size))
    }

    private static func readSharedHeader(fd: Int32, path: String,
                                         fileSize: UInt64) throws -> OfficialSafetensorsSource.Header {
        do {
            let parsed = try OfficialSafetensorsSource.readHeader(
                path: path, fileSize: fileSize) { offset, count in
                    var bytes = Data(count: count)
                    try bytes.withUnsafeMutableBytes { raw in
                        guard let base = raw.baseAddress, !raw.isEmpty else { return }
                        try Posix.preadAll(fd: fd, path: path, buf: base,
                                           count: count, offset: offset)
                    }
                    return bytes
                }
            return parsed
        } catch let error as OfficialSafetensorsSource.ValidationError {
            throw map(error, path: path)
        }
    }

    static func parseHeaderBytes(path: String, fileSize: UInt64,
                                 headerBytes: Data) throws -> Header {
        do {
            return try convert(OfficialSafetensorsSource.parseHeader(
                path: path, fileSize: fileSize, headerBytes: headerBytes))
        } catch let error as OfficialSafetensorsSource.ValidationError {
            throw map(error, path: path)
        }
    }

    static func convert(_ shared: OfficialSafetensorsSource.Header) throws -> Header {
        let tensors = try shared.tensors.map { tensor -> SourceTensor in
            let dtype: SourceTensor.Dtype
            switch tensor.dtype {
            case "U32": dtype = .u32
            case "BF16": dtype = .bf16
            case "F16": dtype = .fp16
            case "F32": dtype = .fp32
            default: throw RepackError.safetensorsUnknownDtype(
                path: shared.path, dtype: tensor.dtype)
            }
            return SourceTensor(name: tensor.name, shardPath: shared.path,
                                dtype: dtype, shape: tensor.shape,
                                absoluteOffset: tensor.absoluteOffset,
                                sizeBytes: tensor.sizeBytes)
        }
        return Header(path: shared.path, payloadBaseOffset: shared.payloadBaseOffset,
                      tensors: tensors)
    }

    static func map(_ error: OfficialSafetensorsSource.ValidationError,
                    path: String) -> RepackError {
        switch error {
        case .headerTooLarge(let size):
            .safetensorsHeaderTooLarge(path: path, size: size)
        case .invalidHeader(let detail):
            .safetensorsHeaderInvalid(path: path, detail: detail)
        case .unknownDtype(let dtype):
            .safetensorsUnknownDtype(path: path, dtype: dtype)
        case .tensorOutOfRange(let name, let end, let size):
            .safetensorsTensorOutOfRange(path: path, name: name, end: end, fileSize: size)
        case .shapeMismatch(let name, let detail):
            .shapeMismatch(name: name, detail: detail)
        case .invalidIndex(let detail):
            .indexJsonInvalid(path: path, detail: detail)
        case .shortRead(let expected, let got):
            .preadShort(path: path, expected: expected, got: got, errno: 0)
        }
    }
}
