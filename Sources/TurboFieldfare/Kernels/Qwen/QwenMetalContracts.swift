import Foundation

/// Host representation of one packed quantized value consumed by Qwen Metal kernels.
typealias QwenMetalPackedValue = UInt8

/// Host storage for one BF16 affine scale or bias consumed by Qwen Metal kernels.
typealias QwenMetalAffineMetadata = UInt16

/// Host representation of dimensions, strides, and indices in the Qwen Metal ABI.
typealias QwenMetalDimension = UInt32

enum QwenMetalABI {
    static let packedValueByteWidth = MemoryLayout<QwenMetalPackedValue>.stride
    static let affineMetadataByteWidth = MemoryLayout<QwenMetalAffineMetadata>.stride
    static let dimensionByteWidth = MemoryLayout<QwenMetalDimension>.stride
    static let affineGroupSize = 64
    static let maximumAddressableBytes = UInt64(UInt32.max)
}

/// Stable buffer roles shared by Qwen-specific Metal entry points.
///
/// Individual kernels may use only a subset. Later kernel phases must not
/// reinterpret an occupied slot with another role.
enum QwenMetalBufferIndex: Int, CaseIterable, Sendable {
    case parameters = 0
    case input = 1
    case weights = 2
    case scales = 3
    case biases = 4
    case output = 5
    case scratch = 6
    case state = 7
}

enum QwenMetalFeature: UInt32, CaseIterable, Sendable {
    case bfloat16 = 0
    case simdgroupMatrix = 1
    case nonuniformThreadgroups = 2
}

struct QwenMetalFeatureSet: OptionSet, Sendable {
    let rawValue: UInt32

    static let bfloat16 = Self(rawValue: 1 << QwenMetalFeature.bfloat16.rawValue)
    static let simdgroupMatrix = Self(rawValue: 1 << QwenMetalFeature.simdgroupMatrix.rawValue)
    static let nonuniformThreadgroups = Self(
        rawValue: 1 << QwenMetalFeature.nonuniformThreadgroups.rawValue)
}

enum QwenMetalContractError: Error, Equatable, Sendable {
    case invalidDimension(name: String, value: Int)
    case dimensionOutOfRange(name: String, value: Int)
    case unsupportedBitWidth(Int)
    case arithmeticOverflow(operation: String)
    case bufferLengthOverflow(component: String, requiredBytes: UInt64)
    case addressRangeOverflow(component: String, requiredBytes: UInt64)
    case unsupportedFeature(QwenMetalFeature)
}

/// Row-local affine layout shared verbatim with `qwen_common.metal`.
///
/// Values use packed unsigned 4- or 8-bit storage. Each logical row starts a
/// new group sequence, with BF16 scale and bias metadata for every group of 64
/// values. All shader address arithmetic remains 32-bit, so construction
/// rejects a component whose complete byte range would exceed `UInt32.max`.
struct QwenMetalAffineLayout: Sendable, Equatable {
    var rowCount: UInt32
    var columnCount: UInt32
    var valuesRowStrideBytes: UInt32
    var metadataRowStrideBytes: UInt32
    var groupsPerRow: UInt32
    var bitWidth: UInt32
    var groupSize: UInt32
    var reserved: UInt32

    init(rows: Int, columns: Int, bitWidth: Int) throws {
        let checkedRows = try Self.checkedDimension(rows, named: "rows")
        let checkedColumns = try Self.checkedDimension(columns, named: "columns")
        guard bitWidth == 4 || bitWidth == 8 else {
            throw QwenMetalContractError.unsupportedBitWidth(bitWidth)
        }

        let packedBits = try Self.checkedMultiply(
            UInt64(checkedColumns), UInt64(bitWidth), operation: "packed row bits")
        let valuesStride = Self.ceilingDivide(packedBits, by: 8)
        let groups = Self.ceilingDivide(
            UInt64(checkedColumns), by: UInt64(QwenMetalABI.affineGroupSize))
        let metadataStride = try Self.checkedMultiply(
            groups, UInt64(QwenMetalABI.affineMetadataByteWidth),
            operation: "metadata row bytes")

        guard valuesStride <= UInt64(UInt32.max), metadataStride <= UInt64(UInt32.max) else {
            throw QwenMetalContractError.arithmeticOverflow(operation: "row stride")
        }

        let valueBytes = try Self.checkedMultiply(
            UInt64(checkedRows), valuesStride, operation: "values buffer length")
        let metadataBytes = try Self.checkedMultiply(
            UInt64(checkedRows), metadataStride, operation: "metadata buffer length")
        try Self.validateBufferLength(valueBytes, component: "values")
        try Self.validateBufferLength(metadataBytes, component: "metadata")

        rowCount = checkedRows
        columnCount = checkedColumns
        valuesRowStrideBytes = UInt32(valuesStride)
        metadataRowStrideBytes = UInt32(metadataStride)
        groupsPerRow = UInt32(groups)
        self.bitWidth = UInt32(bitWidth)
        groupSize = UInt32(QwenMetalABI.affineGroupSize)
        reserved = 0
    }

    static func require(
        _ required: QwenMetalFeatureSet,
        available: QwenMetalFeatureSet
    ) throws {
        for feature in QwenMetalFeature.allCases {
            let flag = QwenMetalFeatureSet(rawValue: 1 << feature.rawValue)
            if required.contains(flag), !available.contains(flag) {
                throw QwenMetalContractError.unsupportedFeature(feature)
            }
        }
    }

    private static func checkedDimension(_ value: Int, named name: String) throws -> UInt32 {
        guard value > 0 else {
            throw QwenMetalContractError.invalidDimension(name: name, value: value)
        }
        guard let result = UInt32(exactly: value) else {
            throw QwenMetalContractError.dimensionOutOfRange(name: name, value: value)
        }
        return result
    }

    private static func checkedMultiply(
        _ lhs: UInt64,
        _ rhs: UInt64,
        operation: String
    ) throws -> UInt64 {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw QwenMetalContractError.arithmeticOverflow(operation: operation)
        }
        return result
    }

    private static func ceilingDivide(_ value: UInt64, by divisor: UInt64) -> UInt64 {
        value / divisor + (value.isMultiple(of: divisor) ? 0 : 1)
    }

    private static func validateBufferLength(_ bytes: UInt64, component: String) throws {
        guard bytes <= UInt64(Int.max) else {
            throw QwenMetalContractError.bufferLengthOverflow(
                component: component, requiredBytes: bytes)
        }
        guard bytes <= QwenMetalABI.maximumAddressableBytes else {
            throw QwenMetalContractError.addressRangeOverflow(
                component: component, requiredBytes: bytes)
        }
    }
}
