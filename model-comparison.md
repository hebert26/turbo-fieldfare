# TurboFieldfare: model comparison and next trial

Evidence checked: 14 September 2026  
Project: `/Users/dev-machine/dev/turbo-fieldfare-personal`  
Status: next model selected. Local Qwen testing and integration have not started.

**The next model to try is Qwen3.6-35B-A3B: 35 billion total parameters, approximately 3 billion active per generated step. Preserve expert streaming and vision support.**

## What the model needs to do

The model drives exploration of an iOS application through VisionCapture's MCP gateway. MCP is the connection that lets the model request actions from the testing tools.

Hebert provides the testing goal, device or Simulator identifier, and application identifier. The model must inspect the current screen, choose useful actions, tap and type through the gateway, check the results, and report problems with supporting evidence. Gemma already performs this role in the current app.

## Models being compared

The ranking is a benchmark-based estimate for this workflow. It is not a measured ranking of the compressed models on Hebert's Mac.

| Model | Total parameters | Active per step | Compressed model files | MoE | Tool use | Vision | Decision for iOS exploration |
|---|---:|---:|---|---|---|---|---|
| **Qwen3.6-35B-A3B** | 35 billion | **3 billion** | **22.3 GB + vision file** | Yes | Yes | Yes | **1 — Selected for the next trial.** Strong published evidence for MCP tool use and planning. |
| **Qwen3.5-122B-A10B** | 122 billion | 10 billion | 77.6 GB + vision file | Yes | Yes | Yes | **2 — Larger alternative.** Strong broad reasoning and visual capabilities, with more work per step. |
| **Gemma 4 26B-A4B — current** | Approximately 26 billion | Approximately 4 billion | 14.3 GB + 1.1 GB vision | Yes | Yes | Yes | **3 — Comparison baseline.** Already integrated with TurboFieldfare and VisionCapture. |

MoE means **mixture of experts**: the model selects a subset of its expert blocks for each step. All three also retain shared parts that every step needs. Parameter counts are rounded. Model-native tool and vision support still require the runner and gateway to support them together.

The Qwen sizes refer to the published **Q4_K_M 4-bit GGUF** builds, rounded from 22.29 GB and 77.62 GB. GGUF is a model file format. Their matching vision files are additional, and their total installed sizes have not been measured here. These are disk sizes, not requirements to keep every expert in memory. [35B files](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF), [122B files](https://huggingface.co/bartowski/Qwen_Qwen3.5-122B-A10B-GGUF)

Gemma's installed configuration and storage figures come from the [project README](/Users/dev-machine/dev/turbo-fieldfare-personal/README.md) and [pinned model source](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfareRepack/Core/Remote/SupportedModelSource.swift). Native architectures and capabilities are documented in the [Gemma model card](https://ai.google.dev/gemma/docs/core/model_card_4), [Qwen3.6 model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), and [Qwen3.5-122B model card](https://huggingface.co/Qwen/Qwen3.5-122B-A10B).

## Why try 3 billion active instead of 4 billion active?

**Active parameter count is not an intelligence score.** Training and architecture affect what a model can do. A model with fewer active parameters can perform better on a particular task, while its total expert set is larger. Fewer active parameters also do not guarantee a particular generation speed.

The following publisher results support trying Qwen3.6 for this workflow. Higher scores are better in each row.

| Benchmark and relevance | Qwen3.6-35B-A3B | Qwen3.5-122B-A10B | Gemma 4 26B-A4B |
|---|---:|---:|---:|
| MCPMark — completing tasks through connected tools | **37.0** | Not reported in the reviewed card | 14.2 |
| MCP-Atlas — working with connected tools | **62.8** | Not reported in the reviewed card | 50.0 |
| DeepPlanning — planning several steps | **25.9** | 24.1 | 16.2 |
| CC-OCR — reading text in images | **81.9** | 81.8 | 74.5 |

The Qwen3.6 and Gemma scores appear in the same [Qwen3.6 release comparison](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The 122B scores come from its [separate release evaluation](https://huggingface.co/Qwen/Qwen3.5-122B-A10B). Protocols can differ between releases, so small differences do not establish a winner.

These results show substantial gains over Gemma on the listed tool and planning benchmarks. They do **not** prove better iOS exploration, better bug detection, or equal results after compression. No direct comparison of these three models through Hebert's gateway has been run.

## Preserve TurboFieldfare's benefit

The intended approach keeps shared weights and the image-processing component in memory, streams the selected experts from disk, and uses a bounded expert cache. Machines with more available memory can retain more experts and reduce disk reads. Actual speed also depends on the processor, storage, screenshots, and conversation length.

The [current runtime](/Users/dev-machine/dev/turbo-fieldfare-personal/Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift) is specific to Gemma 4 26B-A4B. Qwen needs a compatible execution path. It cannot be enabled by replacing the existing `.gturbo` file.

**gmlx is a candidate runner to evaluate**, because its documentation explicitly supports Qwen3.5/3.6 vision models with expert streaming. Its vision component stays on the GPU while the language model's selected experts stream. This combination has not been tested on this Mac or connected to this app. [Vision support](https://github.com/asher/gmlx/blob/main/docs/vlm.md), [expert streaming](https://github.com/asher/gmlx/blob/main/docs/streaming.md)

## What the trial needs to establish

Compare Qwen3.6 with the current Gemma using the same application state, testing goal, Simulator, and VisionCapture tools. Judge completed exploration, valid tool calls, correct screen interpretation, and reproducible bug reports. Record false reports, repeated actions, memory use, response time, and generation speed.

The trial must demonstrate that image input and tool use work while experts are streamed. A text-only run or a run with the full model kept in memory would not establish the intended result.

**Decision: try Qwen3.6-35B-A3B next, with 3 billion active parameters, vision, tool use, and expert streaming. Keep Gemma as the baseline and Qwen3.5-122B-A10B as the larger alternative.**
