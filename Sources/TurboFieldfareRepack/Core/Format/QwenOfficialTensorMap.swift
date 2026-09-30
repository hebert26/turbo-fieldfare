import TurboFieldfareOfficialQwenSource

struct QwenOfficialTensorDescriptor: Sendable, Equatable {
    let name: String
    let dataType: SourceTensor.Dtype
}

enum QwenOfficialTensorCategory: Sendable, Equatable {
    case textResident
    case routedExpert
    case vision
    case mtpOmitted
}

/// RepackCore compatibility adapter. The shared module owns the sole pinned
/// name definition; RepackCore retains its dtype, categories and error cases.
enum QwenOfficialTensorMap {
    static func classify(
        _ tensors: [QwenOfficialTensorDescriptor]
    ) throws -> [QwenOfficialTensorCategory] {
        let shared = tensors.map {
            OfficialQwenTensorDescriptor(name: $0.name, dataType: $0.dataType.safetensorsToken)
        }
        do {
            return try OfficialQwenTensorMap.classify(shared).map { category -> QwenOfficialTensorCategory in
                switch category {
                case .textResident: .textResident
                case .routedExpert: .routedExpert
                case .vision: .vision
                case .mtpPresentUnsupported: .mtpOmitted
                }
            }
        } catch let error as OfficialQwenTensorMapError {
            switch error {
            case .invalidTensorSet: throw QwenOfficialValidationError.invalidTensorSet
            case .invalidTensor: throw QwenOfficialValidationError.invalidTensor
            }
        }
    }
}

private extension SourceTensor.Dtype {
    var safetensorsToken: String {
        switch self {
        case .u32: "U32"
        case .bf16: "BF16"
        case .fp16: "F16"
        case .fp32: "F32"
        }
    }
}
