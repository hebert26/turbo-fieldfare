import Foundation

package enum OfficialSourceMetadataCostSite {
    case total, binding, receiptNames, root, listing, fileOpen
    case heldStat, namedStat, fileCheckAndRemember, fileClose
}

package struct OfficialSourceMetadataCostCounter: Codable, Sendable {
    package var count: UInt64 = 0
    package var wallNanoseconds: UInt64 = 0
    mutating func add(_ duration: UInt64) {
        count += 1
        wallNanoseconds += duration
    }
}

package struct OfficialSourceMetadataCostSnapshot: Codable, Sendable {
    package var total = OfficialSourceMetadataCostCounter()
    package var binding = OfficialSourceMetadataCostCounter()
    package var receiptNames = OfficialSourceMetadataCostCounter()
    package var root = OfficialSourceMetadataCostCounter()
    package var listing = OfficialSourceMetadataCostCounter()
    package var fileOpen = OfficialSourceMetadataCostCounter()
    package var heldStat = OfficialSourceMetadataCostCounter()
    package var namedStat = OfficialSourceMetadataCostCounter()
    package var fileCheckAndRemember = OfficialSourceMetadataCostCounter()
    package var fileClose = OfficialSourceMetadataCostCounter()
}

// One synchronous command owns this object. It has no callbacks or locks.
package final class OfficialSourceMetadataCost {
    package let clockEnabled: Bool
    private var totals = OfficialSourceMetadataCostSnapshot()
    package init(clockEnabled: Bool) { self.clockEnabled = clockEnabled }
    package func start() -> UInt64? {
        clockEnabled ? DispatchTime.now().uptimeNanoseconds : nil
    }
    package func end(_ site: OfficialSourceMetadataCostSite, start: UInt64?) {
        let duration: UInt64
        if let start { duration = DispatchTime.now().uptimeNanoseconds - start }
        else { duration = 0 }
        switch site {
        case .total: totals.total.add(duration)
        case .binding: totals.binding.add(duration)
        case .receiptNames: totals.receiptNames.add(duration)
        case .root: totals.root.add(duration)
        case .listing: totals.listing.add(duration)
        case .fileOpen: totals.fileOpen.add(duration)
        case .heldStat: totals.heldStat.add(duration)
        case .namedStat: totals.namedStat.add(duration)
        case .fileCheckAndRemember: totals.fileCheckAndRemember.add(duration)
        case .fileClose: totals.fileClose.add(duration)
        }
    }
    package func merge(_ value: OfficialSourceMetadataCostSnapshot) {
        func sum(_ a: inout OfficialSourceMetadataCostCounter, _ b: OfficialSourceMetadataCostCounter) {
            a.count += b.count
            a.wallNanoseconds += b.wallNanoseconds
        }
        sum(&totals.total, value.total)
        sum(&totals.binding, value.binding)
        sum(&totals.receiptNames, value.receiptNames)
        sum(&totals.root, value.root)
        sum(&totals.listing, value.listing)
        sum(&totals.fileOpen, value.fileOpen)
        sum(&totals.heldStat, value.heldStat)
        sum(&totals.namedStat, value.namedStat)
        sum(&totals.fileCheckAndRemember, value.fileCheckAndRemember)
        sum(&totals.fileClose, value.fileClose)
    }
    package func snapshot() -> OfficialSourceMetadataCostSnapshot { totals }
}
