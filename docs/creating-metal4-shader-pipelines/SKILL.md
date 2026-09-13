---
name: creating-metal4-shader-pipelines
description: ALWAYS use when creating Metal pipeline state objects with Metal 4 descriptors (MTL4RenderPipelineDescriptor, MTL4ComputePipelineDescriptor, MTL4MeshRenderPipelineDescriptor) via MTL4Compiler. Covers function descriptors, function constant specialization, flexible/unspecialized PSOs, color-attachment mapping, async compilation (MTL4CompilerTask), pipeline caching (MTL4Archive, MTL4PipelineDataSetSerializer), static/binary shader linking, and pipeline reflection. Do NOT trigger for HLSL/DXIL → metallib — use compiling-with-metal-shaderconverter.
---

# Pipeline Creation

## Overview

Covers Metal 4 pipeline state creation when porting games: loading pre-compiled metallibs, building render/compute/mesh PSOs via `MTL4Compiler`, function constants, flexible PSOs, color-attachment mapping, async compilation, and pipeline caching.

For compiling HLSL/DXIL shaders to metallibs, see the `compiling-with-metal-shaderconverter` skill.
For compiling Metal shader language (MSL) source, see MSL Source Compilation below.

## References

**Read the relevant Metal 4 SDK header before writing pipeline code** — the headers are the source of truth for property names, types, and method signatures.

- Metal 4 SDK headers - `$(xcrun --show-sdk-path)/System/Library/Frameworks/Metal.framework/Headers/` — focus on `MTL4Compiler.h`, `MTL4RenderPipeline.h`, `MTL4ComputePipeline.h`, `MTL4MeshRenderPipeline.h`, `MTL4TileRenderPipeline.h`, `MTL4LinkingDescriptor.h`
- Apple documentation - [Using the Metal 4 compilation API](https://developer.apple.com/documentation/metal/using-the-metal-4-compilation-api)

## Cross-API PSO Compatibility

Metal 3 PSOs work on Metal 4 encoders (and vice versa). This enables incremental porting — pipeline creation can be migrated to `MTL4Compiler` independently of encoder migration.

**Cross-API equivalents (metallib → PSO):**

| Concept | D3D12 | Vulkan | Metal 4 |
|---|---|---|---|
| Bytecode → loaded library | embedded `D3D12_SHADER_BYTECODE` | `vkCreateShaderModule(SPIR-V)` | `-[device newLibraryWithURL:]` (load metallib) |
| Function reference for PSO | bytecode + entry-point name | `VkPipelineShaderStageCreateInfo` | `MTL4LibraryFunctionDescriptor` (library + name) |
| Render PSO create | `CreateGraphicsPipelineState(D3D12_GRAPHICS_PIPELINE_STATE_DESC)` | `vkCreateGraphicsPipelines(VkGraphicsPipelineCreateInfo)` | `-[MTL4Compiler newRenderPipelineStateWithDescriptor:](MTL4RenderPipelineDescriptor)` |
| Compute PSO create | `CreateComputePipelineState(D3D12_COMPUTE_PIPELINE_STATE_DESC)` | `vkCreateComputePipelines(VkComputePipelineCreateInfo)` | `-[MTL4Compiler newComputePipelineStateWithDescriptor:](MTL4ComputePipelineDescriptor)` |
| Mesh-shader PSO create | `ID3D12Device2::CreatePipelineState` (stream w/ MS subobjects) | `vkCreateGraphicsPipelines` (mesh stages) | render selector above — pass `MTL4MeshRenderPipelineDescriptor` |
| Specialization | none (ship compiled permutations) | `VkSpecializationInfo` | `MTL4SpecializedFunctionDescriptor` (function constants); flexible PSOs for format/blend |
| Pipeline cache (disk) | `ID3D12PipelineLibrary` | `VkPipelineCache` | `MTL4Archive` + `MTL4PipelineDataSetSerializer` |

**Notes:**
- `-[MTL4Compiler newRenderPipelineStateWithDescriptor:]` is polymorphic: render, mesh, and tile pipeline descriptors all derive from `MTL4PipelineDescriptor` and use the same selector.
- Each PSO-creation selector has an async variant taking `completionHandler:` and returning `MTL4CompilerTask`.
- Producing a metallib from HLSL/DXIL is covered by the `compiling-with-metal-shaderconverter` skill.
- For in-process MSL source compilation, see **MSL Source Compilation** below.

## MTL4Compiler Workflow

Metal 4 uses `MTL4Compiler` for explicit pipeline compilation, replacing Metal 3's `[device newRenderPipelineStateWithDescriptor:error:]`. Pipelines use `FunctionDescriptor` (not `MTLFunction`), support flexible (unspecialized) states for reduced compilation time, and color-attachment mapping for PSO reuse across render pass configurations.

Synchronous (blocks the caller):

```objc
// 1. Create compiler
MTL4CompilerDescriptor* compilerDescriptor = [[MTL4CompilerDescriptor alloc] init];
id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:compilerDescriptor error:&error];

// 2. Load library (most common: from pre-compiled metallib)
id<MTLLibrary> library = [device newLibraryWithURL:metallibURL error:&error];

// 3. Create function descriptors (NOT MTLFunction)
MTL4LibraryFunctionDescriptor* vertexFunction = [[MTL4LibraryFunctionDescriptor alloc] init];
vertexFunction.library = library;
vertexFunction.name = @"vertexShader";

// 4. Create pipeline
MTL4RenderPipelineDescriptor* pipelineDescriptor = [[MTL4RenderPipelineDescriptor alloc] init];
pipelineDescriptor.vertexFunctionDescriptor = vertexFunction;
pipelineDescriptor.fragmentFunctionDescriptor = fragmentFunction;
pipelineDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;  // width/height inferred from render pass
id<MTLRenderPipelineState> pipelineState = [compiler newRenderPipelineStateWithDescriptor:pipelineDescriptor compilerTaskOptions:nil error:&error];
```

Asynchronous (returns immediately; preferred for AAA games and large projects):

```objc
MTL4CompilerTaskOptions* taskOptions = [[MTL4CompilerTaskOptions alloc] init];
id<MTL4CompilerTask> task = [compiler newRenderPipelineStateWithDescriptor:pipelineDescriptor
                                                        compilerTaskOptions:taskOptions
                                                          completionHandler:^(id<MTLRenderPipelineState> pipelineState, NSError* error) {
    if (pipelineState) { /* cache PSO */ }
}];
```

Multithreaded by default; configure QoS via `MTL4CompilerTaskOptions`.

## MSL Source Compilation

For native MSL, use `MTL4Compiler`:

```objc
MTL4LibraryDescriptor* libraryDescriptor = [[MTL4LibraryDescriptor alloc] init];
libraryDescriptor.source = mslSourceString;
id<MTLLibrary> library = [compiler newLibraryWithDescriptor:libraryDescriptor error:&error];
```

Offline alternative: `xcrun metal`.

## Function Descriptor Types

| Type | Use Case |
|---|---|
| `MTL4LibraryFunctionDescriptor` | Standard: library + function name |
| `MTL4SpecializedFunctionDescriptor` | Function constants specialization |
| `MTL4BinaryFunctionDescriptor` | Pre-compiled binary functions for binary linking — see below |

## Render Pipeline Descriptor Properties

See `MTL4RenderPipeline.h` for the full property list (pipeline functions, color attachments, vertex input, rasterization, features, linking).

**Options and reflection.** `MTL4PipelineOptions` configures reflection capture and validation at compile time:

```objc
MTL4PipelineOptions* pipelineOptions = [[MTL4PipelineOptions alloc] init];
pipelineOptions.shaderReflection = MTL4ShaderReflectionBindingInfo | MTL4ShaderReflectionBufferTypeInfo;
pipelineOptions.shaderValidation = MTLShaderValidationEnabled;
pipelineDescriptor.options = pipelineOptions;

id<MTLRenderPipelineState> pipelineState = [compiler newRenderPipelineStateWithDescriptor:pipelineDescriptor
                                                              compilerTaskOptions:nil
                                                                            error:&error];
```

When debugging binding mismatches, use reflection to inspect pipeline bindings at runtime.

## Mesh Shader Pipeline

See `MTL4MeshRenderPipeline.h`. `MTL4MeshRenderPipelineDescriptor` adds object/mesh function descriptors and per-threadgroup limits on top of the render pipeline base.

## Compute Pipeline

See `MTL4ComputePipeline.h`. Set `threadGroupSizeIsMultipleOfThreadExecutionWidth = YES` only when the threadgroup size is guaranteed to be a multiple of execution width — this enables SIMD optimizations.

## Specialization

Metal 4 offers three ways to make one PSO serve many cases: function constants (compile-time specialization on values), flexible pipeline states (deferred format/blend selection), and color-attachment mapping (output index remapping).

### Function Constants

Metal's mechanism for compile-time shader specialization — eliminates runtime branches entirely.

```objc
MTLFunctionConstantValues* functionConstants = [[MTLFunctionConstantValues alloc] init];
BOOL enableLighting = YES;
[functionConstants setConstantValue:&enableLighting type:MTLDataTypeBool atIndex:0];

MTL4LibraryFunctionDescriptor* baseFunction = [[MTL4LibraryFunctionDescriptor alloc] init];
baseFunction.library = library;
baseFunction.name = @"fragmentShader";

MTL4SpecializedFunctionDescriptor* specializedFunction = [[MTL4SpecializedFunctionDescriptor alloc] init];
specializedFunction.functionDescriptor = baseFunction;
specializedFunction.constantValues = functionConstants;
// specializedFunction.specializedName = @"optimizedName"; // optional — names the specialized variant
```

**Key considerations:**
- Not all shaders have function constants — check shader reflection or MSL source for `[[function_constant(N)]]` declarations
- Specialization triggers recompilation — cache the resulting pipelines
- In MSL: `constant bool &flag [[function_constant(0)]]` — branches on this are eliminated at pipeline creation time

### Flexible (Unspecialized) Pipeline States

Compile once at launch without committing to pixel format or blend state, then specialize at runtime without recompiling shader code. Use this when the same shader must work with many pixel format or blend state combinations — common in engines with configurable render targets, multiple output formats, or runtime-variable blend modes. If the pipeline configuration is known and fixed, full compilation is simpler and relatively as fast at draw time.

```objc
// 1. Compile flexible PSO at launch
pipelineDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatUnspecialized;
pipelineDescriptor.colorAttachments[0].blendingState = MTL4BlendStateUnspecialized;
id<MTLRenderPipelineState> flexiblePipelineState = [compiler newRenderPipelineStateWithDescriptor:pipelineDescriptor
                                                                  compilerTaskOptions:nil
                                                                                error:&error];

// 2. Specialize at runtime (fast — no shader recompile)
MTL4RenderPipelineDescriptor* specializationDescriptor = [[MTL4RenderPipelineDescriptor alloc] init];
specializationDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
specializationDescriptor.colorAttachments[0].blendingState = MTL4BlendStateEnabled;
// set blend factors...
id<MTLRenderPipelineState> specializedPipelineState = [compiler newRenderPipelineStateBySpecializationWithDescriptor:specializationDescriptor
                                                                                       pipeline:flexiblePipelineState
                                                                                          error:&error];
```

Benefits:
- Shared shader code across specializations (saves memory)
- Faster runtime specialization than full recompilation
- Reduces total pipeline count

### Color-Attachment Mapping

Allows remapping a shader's logical `[[color(N)]]` outputs to different physical render pass attachment indices at draw time. A single PSO works across different render pass output configurations without recompilation:

```objc
MTL4RenderPipelineDescriptor* pipelineDescriptor = [[MTL4RenderPipelineDescriptor alloc] init];
// Pipeline: inherit mapping from encoder (not baked into PSO)
pipelineDescriptor.colorAttachmentMappingState = MTL4LogicalToPhysicalColorAttachmentMappingStateInherited;

// Render pass: enable mapping
renderPassDescriptor.supportColorAttachmentMapping = YES;

// Encoder: set mapping at draw time
// Encoder: remap logical [[color(0)]] to physical attachment 2
MTLLogicalToPhysicalColorAttachmentMap* attachmentMap = [[MTLLogicalToPhysicalColorAttachmentMap alloc] init];
[attachmentMap setPhysicalIndex:2 forLogicalIndex:0];
[renderEncoder setColorAttachmentMap:attachmentMap];
```

Benefits:
- Compile one pipeline state instead of many for different render pass configurations
- Consolidate render commands into fewer passes
- Reduce CPU overhead from pipeline state management

## Shader Linking

Metal 4 provides two mechanisms for linking pre-compiled shader functions into a pipeline. These are advanced features — most ports don't need them initially, but engines with modular shader architectures (material systems, effect graphs) may benefit.

**Static linking** links additional shader functions at Metal IR level during PSO creation. Because linking occurs at compile time, the compiler can inline and optimize across function boundaries. Configured per-stage on the pipeline descriptor via `vertexStaticLinkingDescriptor` / `fragmentStaticLinkingDescriptor` (render) or `staticLinkingDescriptor` (compute/tile), using `MTL4StaticLinkingDescriptor`.

**Binary linking** links pre-compiled binary functions (`MTL4BinaryFunction`) to a pipeline. Since the functions are already compiled to machine code, no cross-function inlining or optimization is possible — but compilation is faster because the linked functions don't need recompilation. Configured via `MTL4PipelineStageDynamicLinkingDescriptor` passed to `newRenderPipelineState:dynamicLinkingDescriptor:` (or the compute equivalent). To later add binary functions to an existing pipeline, set `supportVertexBinaryLinking` / `supportFragmentBinaryLinking` on the pipeline descriptor at creation time.

**When to use binary linking:** Binary functions save compilation time when the same function is reused across many PSOs — the function is compiled to machine code once and linked without recompilation. However, because the function call cannot be inlined, there is a runtime cost: call frame maintenance and potential stack spilling. Profile to ensure the compilation time savings justify the runtime overhead for your workload. Static linking is preferred when runtime performance matters more than compilation time.

## Pipeline Caching

### Default cache
Metal maintains a system-wide shader cache - details of which are not specified - that reuses compiled pipelines across runs of the same app. When creating pipeline descriptors, populate input arrays in the same order across runs (for example, the `MTL4FunctionDescriptor` arrays in `MTL4StaticLinkingDescriptor`). Reordering changes the cache key which results in a cache miss, forcing recompilation.

### Explicit cache
For explicit cache control: on the app's first launch, attach an `MTL4PipelineDataSetSerializer` to the compiler so it captures pipeline data as PSOs are built, then flush the serializer to an `MTL4Archive` file. On subsequent app launches, pass the archive via `MTL4CompilerTaskOptions.lookupArchives`; PSOs whose descriptors match load from disk instead of recompiling.

```objc
// First launch — attach serializer so the compiler captures pipeline data as PSOs are built
// serializer comes from [device newPipelineDataSetSerializerWithDescriptor:serializerDescriptor error:&error]
compilerDescriptor.pipelineDataSetSerializer = serializer;
id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:compilerDescriptor error:&error];
// ... create PSOs ...
[serializer serializeAsArchiveAndFlushToURL:archiveURL error:&error];

// Subsequent launches — load the archive and let the compiler look up cached PSOs
id<MTL4Archive> archive = [device newArchiveWithURL:archiveURL error:&error];
MTL4CompilerTaskOptions* taskOptions = [[MTL4CompilerTaskOptions alloc] init];
taskOptions.lookupArchives = @[archive];
```

Binary archives are compatible across Metal 3 and Metal 4.

## Recommended Practices

- **Enable shader validation during development** — see `man MetalValidation` for environment variables. Use `MTL4PipelineOptions.shaderValidation` for fine-grained per-pipeline control.
- **Compile pipelines async or pre-launch.** Synchronous runtime compilation hitches the render thread. `MTL4CompilerTask` parallelizes compilation off the render thread.
- **Use Metal 3 pipeline descriptors for incremental porting.** Metal 3 PSOs work on Metal 4 encoders — don't block other porting work on pipeline migration. Move to `MTL4Compiler` when you need its features (flexible PSOs, caching, binary linking).
- **Use binary archives** (`MTL4Archive`) to cache compiled pipelines across app launches. First-launch compilation can be slow; harvest pipeline states, serialize, and load on subsequent runs.
- **Use color-attachment mapping** when the same shader outputs to different render-pass configurations — one PSO instead of many.
- **Use flexible PSOs for format/blend variance.** Compile once unspecialized, specialize at runtime. Avoids the per-variant PSO explosion; faster than full recompilation and uses less memory.
- **Pipeline reflection is a debugging aid.** Binding slot assignments should be known ahead of time from the shader. Reach for reflection only to resolve binding mismatches.
