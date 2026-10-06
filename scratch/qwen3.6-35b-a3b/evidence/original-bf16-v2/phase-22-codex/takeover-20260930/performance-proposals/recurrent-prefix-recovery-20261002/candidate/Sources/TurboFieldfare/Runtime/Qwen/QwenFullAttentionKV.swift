import Foundation
import Metal

enum QwenFullAttentionKVError: Error, Equatable, Sendable {
    case invalidLayerMaskCount(Int)
    case invalidLayerMaskValue(layer: Int, value: UInt8)
    case invalidFullLayerCount(Int)
    case invalidGeometry(field: String, value: Int)
    case arithmeticOverflow(operation: String)
    case allocationFailed(layer: Int, kind: String, bytes: Int)
    case notFullAttentionLayer(Int)
    case invalidTokenCount(Int)
    case outOfOrderWrite(layer: Int, expected: Int, actual: Int)
    case capacityExceeded(start: Int, count: Int, capacity: Int)
    case writeAlreadyPending(layer: Int)
    case unknownWrite
    case writeAlreadySubmitted
    case readAlreadySubmitted
    case activeGPUUse
    case invalidSnapshotOwner
    case invalidSnapshotLineage
    case snapshotAheadOfCommitted(layer: Int, snapshot: Int, committed: Int)
    case inconsistentCommittedPositions
    case commandBufferAlreadySubmitted
    case invalidStagedBuffer(kind: String, required: Int, actual: Int)
}

struct QwenFullAttentionKVView: @unchecked Sendable {
    let key: MTLBuffer
    let value: MTLBuffer
    let keyOffset: Int
    let valueOffset: Int
    let strideBytes: Int
    let validTokenCount: Int
}

struct QwenFullAttentionKVWrite: @unchecked Sendable {
    let layer: Int
    let position: Int
    let tokenCount: Int
    let key: MTLBuffer
    let value: MTLBuffer
    let keyOffset: Int
    let valueOffset: Int
    let strideBytes: Int
    fileprivate let identifier: UInt64
}

struct QwenFullAttentionKVRead: Sendable, Equatable {
    let layers: [Int]
    fileprivate let identifier: UInt64
}

struct QwenFullAttentionKVSnapshot: Sendable, Equatable {
    let position: Int
    fileprivate let owner: UUID
    fileprivate let lineage: UInt64
    fileprivate let positions: [Int: Int]
}

/// Linear FP32 K/V storage for Qwen's ten manifest-declared full-attention
/// layers. Admission and completion state are lock-guarded; command-buffer
/// completion handlers retain this owner until the GPU's actual last use.
final class QwenFullAttentionKV: @unchecked Sendable {
    private struct LayerStorage {
        let key: MTLBuffer
        let value: MTLBuffer
        var committedPosition: Int
        var pendingWrite: PendingWrite?
        var activeReads: Int
    }

    private struct PendingWrite {
        let identifier: UInt64
        let position: Int
        let tokenCount: Int
        var submitted: Bool
    }

    private struct ReadLease {
        let layers: [Int]
        var submitted: Bool
    }

    let fullLayerIndices: [Int]
    let maxContext: Int
    let keyValueHeadCount: Int
    let headDimension: Int
    let strideBytes: Int

    var fullLayerCount: Int { fullLayerIndices.count }

    private let owner = UUID()
    private let lock = NSLock()
    private var layers: [Int: LayerStorage]
    private var reads: [UInt64: ReadLease] = [:]
    private var nextIdentifier: UInt64 = 1
    private var lineage: UInt64 = 0
    private var replacementCheckpointID: UUID?

    init(
        device: MTLDevice,
        fullAttentionLayerMask: [UInt8],
        maxContext: Int,
        keyValueHeadCount: Int,
        headDimension: Int,
        expectedLayerCount: Int = 40,
        expectedFullLayerCount: Int = 10
    ) throws {
        // Defaults preserve the official Phase 9 geometry. Only the internal
        // two-layer source fixture supplies reduced expected counts.
        guard expectedLayerCount > 0,
              fullAttentionLayerMask.count == expectedLayerCount else {
            throw QwenFullAttentionKVError.invalidLayerMaskCount(fullAttentionLayerMask.count)
        }
        for (layer, value) in fullAttentionLayerMask.enumerated() where value != 0 && value != 1 {
            throw QwenFullAttentionKVError.invalidLayerMaskValue(layer: layer, value: value)
        }
        let selectedLayers = fullAttentionLayerMask.indices.filter {
            fullAttentionLayerMask[$0] == 1
        }
        guard expectedFullLayerCount > 0,
              selectedLayers.count == expectedFullLayerCount else {
            throw QwenFullAttentionKVError.invalidFullLayerCount(selectedLayers.count)
        }
        guard maxContext > 0 else {
            throw QwenFullAttentionKVError.invalidGeometry(field: "maxContext", value: maxContext)
        }
        guard keyValueHeadCount > 0 else {
            throw QwenFullAttentionKVError.invalidGeometry(
                field: "keyValueHeadCount", value: keyValueHeadCount)
        }
        guard headDimension > 0 else {
            throw QwenFullAttentionKVError.invalidGeometry(
                field: "headDimension", value: headDimension)
        }

        let elementsPerToken = try Self.checkedMultiply(
            keyValueHeadCount, headDimension, operation: "KV elements per token")
        let stride = try Self.checkedMultiply(
            elementsPerToken, MemoryLayout<Float>.stride, operation: "KV token stride")
        let bufferBytes = try Self.checkedMultiply(
            maxContext, stride, operation: "KV buffer bytes")
        guard maxContext <= Int(UInt32.max), bufferBytes <= device.maxBufferLength else {
            throw QwenFullAttentionKVError.invalidGeometry(
                field: "KV device/position limit", value: bufferBytes)
        }

        var storage: [Int: LayerStorage] = [:]
        storage.reserveCapacity(selectedLayers.count)
        for layer in selectedLayers {
            guard let key = device.makeBuffer(length: bufferBytes, options: .storageModeShared) else {
                throw QwenFullAttentionKVError.allocationFailed(
                    layer: layer, kind: "key", bytes: bufferBytes)
            }
            guard let value = device.makeBuffer(length: bufferBytes, options: .storageModeShared) else {
                throw QwenFullAttentionKVError.allocationFailed(
                    layer: layer, kind: "value", bytes: bufferBytes)
            }
            key.label = "qwen.fullKV.K.layer\(layer)"
            value.label = "qwen.fullKV.V.layer\(layer)"
            storage[layer] = LayerStorage(
                key: key,
                value: value,
                committedPosition: 0,
                pendingWrite: nil,
                activeReads: 0)
        }

        fullLayerIndices = selectedLayers
        self.maxContext = maxContext
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimension = headDimension
        strideBytes = stride
        layers = storage
    }

    func committedPosition(layer: Int) throws -> Int {
        try withLock {
            try requireLayer(layer).committedPosition
        }
    }

    func view(layer: Int) throws -> QwenFullAttentionKVView {
        try withLock {
            let storage = try requireLayer(layer)
            return QwenFullAttentionKVView(
                key: storage.key,
                value: storage.value,
                keyOffset: 0,
                valueOffset: 0,
                strideBytes: strideBytes,
                validTokenCount: storage.committedPosition)
        }
    }

    /// Reserves an append-only range before the caller writes or encodes into it.
    /// Every failure path leaves the committed cursor and buffers unexposed.
    func reserveWrite(
        layer: Int,
        position: Int,
        tokenCount: Int
    ) throws -> QwenFullAttentionKVWrite {
        try withLock {
            guard tokenCount > 0 else {
                throw QwenFullAttentionKVError.invalidTokenCount(tokenCount)
            }
            var storage = try requireLayer(layer)
            guard storage.pendingWrite == nil else {
                throw QwenFullAttentionKVError.writeAlreadyPending(layer: layer)
            }
            guard position == storage.committedPosition else {
                throw QwenFullAttentionKVError.outOfOrderWrite(
                    layer: layer, expected: storage.committedPosition, actual: position)
            }
            let end = try Self.checkedAdd(position, tokenCount, operation: "KV write end")
            guard position >= 0, end <= maxContext else {
                throw QwenFullAttentionKVError.capacityExceeded(
                    start: position, count: tokenCount, capacity: maxContext)
            }
            let offset = try Self.checkedMultiply(position, strideBytes, operation: "KV write offset")
            let identifier = allocateIdentifier()
            storage.pendingWrite = PendingWrite(
                identifier: identifier,
                position: position,
                tokenCount: tokenCount,
                submitted: false)
            layers[layer] = storage
            return QwenFullAttentionKVWrite(
                layer: layer,
                position: position,
                tokenCount: tokenCount,
                key: storage.key,
                value: storage.value,
                keyOffset: offset,
                valueOffset: offset,
                strideBytes: strideBytes,
                identifier: identifier)
        }
    }

    /// Registers the producer command buffer's actual last use. Successful GPU
    /// completion publishes the appended rows; failure only releases admission.
    func trackLastUse(
        of write: QwenFullAttentionKVWrite,
        on commandBuffer: MTLCommandBuffer
    ) throws {
        try withLock {
            guard commandBuffer.status == .notEnqueued else {
                throw QwenFullAttentionKVError.commandBufferAlreadySubmitted
            }
            var storage = try requireLayer(write.layer)
            guard var pending = storage.pendingWrite,
                  pending.identifier == write.identifier,
                  pending.position == write.position,
                  pending.tokenCount == write.tokenCount else {
                throw QwenFullAttentionKVError.unknownWrite
            }
            guard !pending.submitted else {
                throw QwenFullAttentionKVError.writeAlreadySubmitted
            }
            pending.submitted = true
            storage.pendingWrite = pending
            layers[write.layer] = storage
        }
        commandBuffer.addCompletedHandler { [self] completed in
            completeWrite(
                layer: write.layer,
                identifier: write.identifier,
                succeeded: completed.status == .completed && completed.error == nil)
        }
    }

    /// Publish two completed, shared FP32 rows and the cursor as one locked
    /// operation. Neither source buffer may alias a committed KV buffer. All
    /// geometry, ownership and capacity checks precede the first byte copy;
    /// the bounded synchronous copies have no suspension or throwing point.
    /// Failed GPU work must never call this method: abandon the reservation.
    func commitStaged(_ write: QwenFullAttentionKVWrite,
                      key: MTLBuffer, value: MTLBuffer) throws {
        try withLock {
            var storage = try requireLayer(write.layer)
            guard let pending = storage.pendingWrite,
                  pending.identifier == write.identifier,
                  pending.position == write.position,
                  pending.tokenCount == write.tokenCount,
                  !pending.submitted,
                  storage.committedPosition == write.position else {
                throw QwenFullAttentionKVError.unknownWrite
            }
            guard storage.activeReads == 0 else {
                throw QwenFullAttentionKVError.activeGPUUse
            }
            let bytes = try Self.checkedMultiply(
                write.tokenCount, strideBytes, operation: "staged KV bytes")
            let offset = try Self.checkedMultiply(
                write.position, strideBytes, operation: "staged KV offset")
            let end = try Self.checkedAdd(offset, bytes, operation: "staged KV end")
            guard write.keyOffset == offset, write.valueOffset == offset,
                  write.strideBytes == strideBytes, end <= storage.key.length,
                  end <= storage.value.length, key.length == bytes,
                  storage.key.storageMode == .shared,
                  storage.value.storageMode == .shared,
                  key.storageMode == .shared, key.device === storage.key.device,
                  key !== value, key !== storage.key, key !== storage.value else {
                throw QwenFullAttentionKVError.invalidStagedBuffer(
                    kind: "key", required: bytes, actual: key.length)
            }
            guard value.length == bytes, value.storageMode == .shared,
                  value.device === storage.value.device,
                  value !== storage.key, value !== storage.value else {
                throw QwenFullAttentionKVError.invalidStagedBuffer(
                    kind: "value", required: bytes, actual: value.length)
            }
            let next = try Self.checkedAdd(
                write.position, write.tokenCount, operation: "staged KV position")
            storage.key.contents().advanced(by: offset).copyMemory(
                from: key.contents(), byteCount: bytes)
            storage.value.contents().advanced(by: offset).copyMemory(
                from: value.contents(), byteCount: bytes)
            storage.committedPosition = next
            storage.pendingWrite = nil
            layers[write.layer] = storage
        }
    }

    /// Releases an unsubmitted reservation. Bytes possibly written by the CPU
    /// remain outside the committed range and are never exposed to attention.
    func abandon(_ write: QwenFullAttentionKVWrite) throws {
        try withLock {
            var storage = try requireLayer(write.layer)
            guard let pending = storage.pendingWrite,
                  pending.identifier == write.identifier else {
                throw QwenFullAttentionKVError.unknownWrite
            }
            guard !pending.submitted else {
                throw QwenFullAttentionKVError.writeAlreadySubmitted
            }
            storage.pendingWrite = nil
            layers[write.layer] = storage
        }
    }

    /// Retains a stable committed read view until its command buffer completes.
    func reserveRead(layers requestedLayers: [Int]? = nil) throws -> QwenFullAttentionKVRead {
        try withLock {
            let selected = requestedLayers ?? fullLayerIndices
            guard !selected.isEmpty, Set(selected).count == selected.count else {
                throw QwenFullAttentionKVError.invalidFullLayerCount(selected.count)
            }
            for layer in selected {
                _ = try requireLayer(layer)
            }
            let identifier = allocateIdentifier()
            for layer in selected {
                var storage = try requireLayer(layer)
                storage.activeReads += 1
                layers[layer] = storage
            }
            reads[identifier] = ReadLease(layers: selected, submitted: false)
            return QwenFullAttentionKVRead(layers: selected, identifier: identifier)
        }
    }

    func trackLastUse(
        of read: QwenFullAttentionKVRead,
        on commandBuffer: MTLCommandBuffer
    ) throws {
        try withLock {
            guard commandBuffer.status == .notEnqueued else {
                throw QwenFullAttentionKVError.commandBufferAlreadySubmitted
            }
            guard var lease = reads[read.identifier], lease.layers == read.layers else {
                throw QwenFullAttentionKVError.activeGPUUse
            }
            guard !lease.submitted else {
                throw QwenFullAttentionKVError.readAlreadySubmitted
            }
            lease.submitted = true
            reads[read.identifier] = lease
        }
        commandBuffer.addCompletedHandler { [self] _ in
            completeRead(identifier: read.identifier)
        }
    }

    func abandon(_ read: QwenFullAttentionKVRead) throws {
        try withLock {
            guard let lease = reads[read.identifier], lease.layers == read.layers else {
                throw QwenFullAttentionKVError.activeGPUUse
            }
            guard !lease.submitted else {
                throw QwenFullAttentionKVError.readAlreadySubmitted
            }
            releaseRead(identifier: read.identifier)
        }
    }

    func snapshot() throws -> QwenFullAttentionKVSnapshot {
        try withLock {
            try requireNoActiveUse()
            let positions = Dictionary(uniqueKeysWithValues: try fullLayerIndices.map { layer in
                (layer, try requireLayer(layer).committedPosition)
            })
            let uniquePositions = Set(positions.values)
            guard uniquePositions.count == 1, let position = uniquePositions.first else {
                throw QwenFullAttentionKVError.inconsistentCommittedPositions
            }
            return QwenFullAttentionKVSnapshot(
                position: position,
                owner: owner,
                lineage: lineage,
                positions: positions)
        }
    }

    func restore(_ snapshot: QwenFullAttentionKVSnapshot) throws {
        try withLock {
            try requireNoActiveUse()
            guard snapshot.owner == owner else {
                throw QwenFullAttentionKVError.invalidSnapshotOwner
            }
            guard snapshot.lineage == lineage else {
                throw QwenFullAttentionKVError.invalidSnapshotLineage
            }
            var createsBranch = false
            for layer in fullLayerIndices {
                let storage = try requireLayer(layer)
                guard let target = snapshot.positions[layer], target <= storage.committedPosition else {
                    throw QwenFullAttentionKVError.snapshotAheadOfCommitted(
                        layer: layer,
                        snapshot: snapshot.positions[layer] ?? -1,
                        committed: storage.committedPosition)
                }
                createsBranch = createsBranch || target < storage.committedPosition
            }
            if createsBranch, lineage == UInt64.max {
                throw QwenFullAttentionKVError.arithmeticOverflow(operation: "snapshot lineage")
            }
            for layer in fullLayerIndices {
                var storage = try requireLayer(layer)
                storage.committedPosition = snapshot.positions[layer] ?? 0
                layers[layer] = storage
            }
            if createsBranch {
                lineage += 1
            }
        }
    }

    // Diagnostic prefix truncation. Return the ORIGINAL rollback position with refreshed lineage.
    func truncateDiagnosticPrefix(position: Int, baseline: QwenFullAttentionKVSnapshot,
                                  expected: QwenFullAttentionKVSnapshot) throws -> QwenFullAttentionKVSnapshot {
        try withLock {
            try requireNoActiveUse()
            guard baseline.owner == owner, expected.owner == owner else {
                throw QwenFullAttentionKVError.invalidSnapshotOwner
            }
            guard baseline.lineage == lineage, expected.lineage == lineage else {
                throw QwenFullAttentionKVError.invalidSnapshotLineage
            }
            guard Set(baseline.positions.keys) == Set(fullLayerIndices),
                  Set(expected.positions.keys) == Set(fullLayerIndices),
                  baseline.position >= 0, position >= baseline.position,
                  position <= expected.position,
                  baseline.positions.values.allSatisfy({ $0 == baseline.position }),
                  expected.positions.values.allSatisfy({ $0 == expected.position }) else {
                throw QwenFullAttentionKVError.inconsistentCommittedPositions
            }
            for layer in fullLayerIndices {
                guard try requireLayer(layer).committedPosition == expected.position else {
                    throw QwenFullAttentionKVError.inconsistentCommittedPositions
                }
            }
            let createsBranch = position < expected.position
            guard !createsBranch || lineage < UInt64.max else {
                throw QwenFullAttentionKVError.arithmeticOverflow(operation: "diagnostic prefix lineage")
            }
            for layer in fullLayerIndices {
                var storage = layers[layer]!
                storage.committedPosition = position
                layers[layer] = storage
            }
            if createsBranch { lineage += 1 }
            return QwenFullAttentionKVSnapshot(position: baseline.position,
                owner: owner, lineage: lineage, positions: baseline.positions)
        }
    }

    struct ReplacementCheckpoint: Sendable {
        fileprivate let owner: UUID
        fileprivate let identifier: UUID
        fileprivate let keys: [Int: Data]
        fileprivate let values: [Int: Data]
        fileprivate let positions: [Int: Int]
    }

    // Cursor snapshots cannot protect a replacement that overwrites position zero.
    // Retain only the accepted prefixes, after every GPU lease has settled.
    func retainReplacementCheckpoint() throws -> ReplacementCheckpoint {
        try withLock {
            try requireNoActiveUse()
            var keys: [Int: Data] = [:]
            var values: [Int: Data] = [:]
            var positions: [Int: Int] = [:]
            for layer in fullLayerIndices {
                let storage = try requireLayer(layer)
                let bytes = storage.committedPosition * strideBytes
                keys[layer] = Data(bytes: storage.key.contents(), count: bytes)
                values[layer] = Data(bytes: storage.value.contents(), count: bytes)
                positions[layer] = storage.committedPosition
            }
            let identifier = UUID()
            replacementCheckpointID = identifier
            return ReplacementCheckpoint(owner: owner, identifier: identifier,
                keys: keys, values: values, positions: positions)
        }
    }

    func restoreReplacementCheckpoint(_ checkpoint: ReplacementCheckpoint) throws {
        try withLock {
            try requireNoActiveUse()
            guard checkpoint.owner == owner else {
                throw QwenFullAttentionKVError.invalidSnapshotOwner
            }
            guard checkpoint.identifier == replacementCheckpointID else {
                throw QwenFullAttentionKVError.invalidSnapshotLineage
            }
            guard lineage < UInt64.max else {
                throw QwenFullAttentionKVError.arithmeticOverflow(operation: "replacement lineage")
            }
            for layer in fullLayerIndices {
                let storage = try requireLayer(layer)
                guard let position = checkpoint.positions[layer], position >= 0, position <= maxContext,
                      let key = checkpoint.keys[layer], let value = checkpoint.values[layer],
                      key.count == position * strideBytes, value.count == key.count,
                      key.count <= storage.key.length, value.count <= storage.value.length else {
                    throw QwenFullAttentionKVError.invalidSnapshotOwner
                }
            }
            for layer in fullLayerIndices {
                var storage = try requireLayer(layer)
                checkpoint.keys[layer]!.withUnsafeBytes { bytes in
                    if let base = bytes.baseAddress { memcpy(storage.key.contents(), base, bytes.count) }
                }
                checkpoint.values[layer]!.withUnsafeBytes { bytes in
                    if let base = bytes.baseAddress { memcpy(storage.value.contents(), base, bytes.count) }
                }
                storage.committedPosition = checkpoint.positions[layer]!
                layers[layer] = storage
            }
            lineage += 1
        }
    }

    func discardReplacementCheckpoint() {
        lock.lock()
        replacementCheckpointID = nil
        lock.unlock()
    }

    func reset() throws {
        try withLock {
            try requireNoActiveUse()
            guard lineage < UInt64.max else {
                throw QwenFullAttentionKVError.arithmeticOverflow(operation: "snapshot lineage")
            }
            for layer in fullLayerIndices {
                var storage = try requireLayer(layer)
                storage.committedPosition = 0
                layers[layer] = storage
            }
            lineage += 1
        }
    }

    private func completeWrite(layer: Int, identifier: UInt64, succeeded: Bool) {
        lock.lock()
        defer { lock.unlock() }
        guard var storage = layers[layer],
              let pending = storage.pendingWrite,
              pending.identifier == identifier else { return }
        if succeeded {
            storage.committedPosition = pending.position + pending.tokenCount
        }
        storage.pendingWrite = nil
        layers[layer] = storage
    }

    private func completeRead(identifier: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        releaseRead(identifier: identifier)
    }

    private func releaseRead(identifier: UInt64) {
        guard let lease = reads.removeValue(forKey: identifier) else { return }
        for layer in lease.layers {
            guard var storage = layers[layer] else { continue }
            storage.activeReads -= 1
            layers[layer] = storage
        }
    }

    private func requireNoActiveUse() throws {
        for layer in fullLayerIndices {
            let storage = try requireLayer(layer)
            if storage.pendingWrite != nil || storage.activeReads != 0 {
                throw QwenFullAttentionKVError.activeGPUUse
            }
        }
    }

    private func requireLayer(_ layer: Int) throws -> LayerStorage {
        guard let storage = layers[layer] else {
            throw QwenFullAttentionKVError.notFullAttentionLayer(layer)
        }
        return storage
    }

    private func allocateIdentifier() -> UInt64 {
        let identifier = nextIdentifier
        nextIdentifier &+= 1
        if nextIdentifier == 0 { nextIdentifier = 1 }
        return identifier
    }

    private func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

    private static func checkedMultiply(
        _ lhs: Int, _ rhs: Int, operation: String
    ) throws -> Int {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw QwenFullAttentionKVError.arithmeticOverflow(operation: operation)
        }
        return result
    }

    private static func checkedAdd(
        _ lhs: Int, _ rhs: Int, operation: String
    ) throws -> Int {
        let (result, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else {
            throw QwenFullAttentionKVError.arithmeticOverflow(operation: operation)
        }
        return result
    }
}
