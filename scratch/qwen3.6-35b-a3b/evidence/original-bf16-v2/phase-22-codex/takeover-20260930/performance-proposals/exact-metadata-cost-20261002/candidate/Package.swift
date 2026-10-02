// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "TurboFieldfare",
    platforms: [
        .macOS(.v26),
        .iOS(.v26),
    ],
    products: [
        .executable(name: "TurboFieldfareExactMetadataCost", targets: ["TurboFieldfareExactMetadataCost"]),
        .library(name: "TurboFieldfare", targets: ["TurboFieldfare"]),
        .executable(name: "TurboFieldfareRepack", targets: ["TurboFieldfareRepack"]),
        .executable(name: "TurboFieldfareCLI", targets: ["TurboFieldfareCLI"]),
        .executable(name: "TurboFieldfareMac", targets: ["TurboFieldfareMac"]),
        .executable(name: "TurboFieldfareDecodeService", targets: ["TurboFieldfareDecodeService"]),
        .executable(name: "TurboFieldfareServer", targets: ["TurboFieldfareServer"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.101.3"),
        // Pinned by revision: tag 1.7.3 predates the `\left...\right`
        // duplication fix and typesets 433 of the 607-string coverage sweep
        // against this revision's 479.
        .package(
            url: "https://github.com/mgriebling/SwiftMath.git",
            revision: "1d2c90827e9c3908269d810d055fb03b7da5fd53"),
    ],
    targets: [
        .executableTarget(name: "TurboFieldfareExactMetadataCost", dependencies: ["TurboFieldfareOfficialQwenSource"], path: "Sources/TurboFieldfareExactMetadataCost"),
        .binaryTarget(
            name: "TurboFieldfareLibJPEG",
            path: "ThirdParty/libjpeg-turbo/Artifacts/LibJPEG.xcframework"
        ),
        .target(
            name: "TurboFieldfareJPEGBridge",
            dependencies: ["TurboFieldfareLibJPEG"],
            path: "Sources/TurboFieldfareJPEGBridge",
            publicHeadersPath: "include"
        ),
        .target(
            name: "TurboFieldfareSourceTopK",
            path: "Sources/TurboFieldfareSourceTopK",
            publicHeadersPath: "include"
        ),
        .target(
            name: "TurboFieldfareFormat",
            path: "Sources/TurboFieldfareFormat"
        ),
        .target(
            name: "TurboFieldfareOfficialQwenSource",
            dependencies: ["TurboFieldfareFormat"],
            path: "Sources/TurboFieldfareOfficialQwenSource"
        ),
        .target(
            name: "TurboFieldfare",
            dependencies: [
                "TurboFieldfareFormat",
                "TurboFieldfareOfficialQwenSource",
                "TurboFieldfareJPEGBridge",
                "TurboFieldfareSourceTopK",
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/TurboFieldfare",
            resources: [
                .copy("Metal"),
            ]
        ),
        .target(
            name: "TurboFieldfareRepackCore",
            dependencies: ["TurboFieldfareFormat", "TurboFieldfareOfficialQwenSource"],
            path: "Sources/TurboFieldfareRepack/Core"
        ),
        .executableTarget(
            name: "TurboFieldfareRepack",
            dependencies: ["TurboFieldfareRepackCore"],
            path: "Sources/TurboFieldfareRepack/Command"
        ),
        .target(
            name: "TurboFieldfareCLICore",
            dependencies: ["TurboFieldfare"],
            path: "Sources/TurboFieldfareCLI",
            exclude: ["Command"]
        ),
        .executableTarget(
            name: "TurboFieldfareCLI",
            dependencies: ["TurboFieldfareCLICore"],
            path: "Sources/TurboFieldfareCLI/Command"
        ),
        .target(
            name: "TurboFieldfareAppCore",
            dependencies: [
                "TurboFieldfare", "TurboFieldfareRepackCore", "TurboFieldfareDecodeProtocol",
                "TurboFieldfareOfficialQwenSource", "TurboFieldfareFormat",
            ],
            path: "Sources/TurboFieldfareApp/Core",
            resources: [
                .copy("Resources/app-prompts.json"),
            ],
            linkerSettings: [
                .linkedLibrary("bsm"),
            ]
        ),
        .target(
            name: "TurboFieldfareMacPresentation",
            dependencies: [
                "TurboFieldfareAppCore",
                .product(name: "SwiftMath", package: "SwiftMath"),
            ],
            path: "Sources/TurboFieldfareApp/MacPresentation"
        ),
        .target(
            name: "TurboFieldfareDecodeProtocol",
            path: "Sources/TurboFieldfareDecodeProtocol"
        ),
        .executableTarget(
            name: "TurboFieldfareDecodeService",
            dependencies: [
                "TurboFieldfareAppCore", "TurboFieldfareDecodeProtocol",
                "TurboFieldfareFormat", "TurboFieldfareOfficialQwenSource",
            ],
            path: "Sources/TurboFieldfareDecodeService"
        ),
        .target(
            name: "TurboFieldfareServerCore",
            dependencies: [
                "TurboFieldfare",
                "TurboFieldfareOfficialQwenSource",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            path: "Sources/TurboFieldfareServer/Core"
        ),
        .executableTarget(
            name: "TurboFieldfareServer",
            dependencies: ["TurboFieldfareServerCore"],
            path: "Sources/TurboFieldfareServer/Command"
        ),
        .executableTarget(
            name: "TurboFieldfareMac",
            dependencies: ["TurboFieldfareAppCore", "TurboFieldfareMacPresentation"],
            path: "Sources/TurboFieldfareApp/Mac",
            resources: [
                .copy("Resources/turbofieldfare-app-icon.png"),
            ]
        ),
        .target(
            name: "TurboFieldfareValidationSupport",
            dependencies: ["TurboFieldfare"],
            path: "Sources/TurboFieldfareValidation/Support"
        ),
        .testTarget(
            name: "TurboFieldfareOfficialQwenSourceTests",
            dependencies: ["TurboFieldfareOfficialQwenSource"],
            path: "Tests/TurboFieldfareOfficialQwenSource"
        ),
        .testTarget(
            name: "TurboFieldfareFormatTests",
            dependencies: ["TurboFieldfareFormat"],
            path: "Tests/TurboFieldfareFormat"
        ),
        .testTarget(
            name: "TurboFieldfareFormatCompatibilityTests",
            dependencies: ["TurboFieldfareFormat", "TurboFieldfare", "TurboFieldfareRepackCore"],
            path: "Tests/TurboFieldfareFormatCompatibility",
            resources: [.copy("Fixtures")]
        ),
        .testTarget(
            name: "TurboFieldfareTestsCore",
            dependencies: [
                "TurboFieldfare",
                "TurboFieldfareValidationSupport",
                "TurboFieldfareRepackCore",
                "TurboFieldfareCLICore",
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Tests/TurboFieldfare/Core",
            resources: [
                .copy("Runtime/Vision/Fixtures/images"),
                .copy("QwenFixtures/qwen36-tiny-fixtures.json"),
                .copy("QwenFixtures/qwen36-tiny-text-model-fixtures.json"),
                .copy("QwenFixtures/official-bf16-reference-cases.json"),
                .copy("QwenFixtures/BF16ProjectionOracle"),
                .copy("QwenFixtures/full-attention-source"),
                .copy("QwenFixtures/attention-cached-128"),
                .copy("QwenFixtures/large-moe-parity"),
                .copy("QwenFixtures/vision-jpeg-decode"),
                .copy("QwenFixtures/vision-source-layernorm"),
                .copy("QwenFixtures/vision-patch-projection"),
                .copy("QwenFixtures/vision-source-rotary"),
                .copy("QwenFixtures/vision-source-attention"),
                .copy("QwenFixtures/source-topk"),
            ]
        ),
        .testTarget(
            name: "TurboFieldfareRepackTests",
            dependencies: ["TurboFieldfareFormat", "TurboFieldfareRepackCore"],
            path: "Tests/TurboFieldfareRepack/Core"
        ),
        .testTarget(
            name: "TurboFieldfareAppCoreTests",
            dependencies: [
                "TurboFieldfareAppCore", "TurboFieldfare", "TurboFieldfareRepackCore",
                "TurboFieldfareDecodeProtocol", "TurboFieldfareOfficialQwenSource",
            ],
            path: "Tests/TurboFieldfareApp/Core"
        ),
        .testTarget(
            name: "TurboFieldfareDecodeServiceTests",
            dependencies: [
                "TurboFieldfareDecodeService", "TurboFieldfareAppCore",
                "TurboFieldfareDecodeProtocol", "TurboFieldfareFormat",
                "TurboFieldfareOfficialQwenSource",
            ],
            path: "Tests/TurboFieldfareDecodeService"
        ),
        .testTarget(
            name: "TurboFieldfareMacPresentationTests",
            dependencies: ["TurboFieldfareAppCore", "TurboFieldfareMacPresentation"],
            path: "Tests/TurboFieldfareApp/MacPresentation",
            resources: [.copy("Fixtures")]
        ),
        .testTarget(
            name: "TurboFieldfareServerTests",
            dependencies: [
                "TurboFieldfareServerCore",
                "TurboFieldfareFormat", "TurboFieldfareOfficialQwenSource",
                .product(name: "NIOEmbedded", package: "swift-nio"),
            ],
            path: "Tests/TurboFieldfareServer",
            resources: [.copy("Fixtures")]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
