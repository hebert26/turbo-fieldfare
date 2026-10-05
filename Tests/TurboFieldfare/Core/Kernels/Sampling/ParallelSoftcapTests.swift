import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct ParallelSoftcapTests {
    @Test(arguments: [16_384, 16_385, 262_144], [0, 1, 2, 3])
    func preservesProbabilityBits(vocab: Int, shape: Int) throws {
        let context = try MetalContext()
        let serial = try LogitSoftcapSoftmax(context: context, useParallel: false)
        let parallel = try LogitSoftcapSoftmax(context: context)
        var state: UInt64 = 71
        let values = (0..<vocab).map { index -> Float16 in
            state = state &* 6_364_136_223_846_793_005 &+ 1
            switch shape {
            case 1: return 0.1234
            case 2: return index == 42 ? 1000 : 0
            case 3: return index.isMultiple(of: 4) ? -.infinity : Float16(Int(state >> 48) % 101 - 50)
            default: return Float16(Float(Int(state >> 48) - 32_768) / 512)
            }
        }
        let input = try #require(context.device.makeBuffer(bytes: values,
            length: vocab * 2, options: .storageModeShared))
        let expected = try #require(context.device.makeBuffer(length: vocab * 2, options: .storageModeShared))
        let actual = try #require(context.device.makeBuffer(length: vocab * 2, options: .storageModeShared))
        let command = try #require(context.queue.makeCommandBuffer())
        let cap: Float = shape == 2 ? 5 : shape == 3 ? 1e9 : 30
        serial.encode(commandBuffer: command, logits: input, probs: expected, v: UInt32(vocab), softcap: cap)
        parallel.encode(commandBuffer: command, logits: input, probs: actual, v: UInt32(vocab), softcap: cap)
        command.commit()
        command.waitUntilCompleted()
        try checkCommandBufferError(command)
        #expect(Data(bytes: actual.contents(), count: vocab * 2)
            == Data(bytes: expected.contents(), count: vocab * 2))
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TURBOFIELDFARE_SOFTMAX_PROFILE"] == "1"))
    func measuresLargeVocabulary() throws {
        let context = try MetalContext()
        let values = (0..<262_144).map { Float16(Float($0 % 997) / 31 - 15) }
        let input = try #require(context.device.makeBuffer(bytes: values,
            length: values.count * 2, options: .storageModeShared))
        let output = try #require(context.device.makeBuffer(length: values.count * 2, options: .storageModeShared))
        for enabled in [false, true, false, true] {
            let kernel = try LogitSoftcapSoftmax(context: context, useParallel: enabled)
            let command = try #require(context.queue.makeCommandBuffer())
            for _ in 0..<16 {
                kernel.encode(commandBuffer: command, logits: input, probs: output, v: UInt32(values.count))
            }
            command.commit()
            command.waitUntilCompleted()
            try checkCommandBufferError(command)
            print("SOFTMAX_GPU parallel=\(enabled) milliseconds=\((command.gpuEndTime - command.gpuStartTime) * 1000 / 16)")
        }
    }
}
