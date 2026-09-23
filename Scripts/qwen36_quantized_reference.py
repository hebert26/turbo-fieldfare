#!/usr/bin/env python3
"""Independent offline Qwen 3.6 reference over an exact .gturbo v2 pack.

This program deliberately does not import TurboFieldfare. It validates and
decodes the production pack format itself, then supplies the decoded Float32
weights to the pinned official Transformers Qwen3_5Moe decoder one layer at a
time. The full BF16 or Float32 model is never constructed.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import gc
import hashlib
import importlib
import importlib.util
import json
import mmap
import os
import platform
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NoReturn


REQUEST_SCHEMA = "qwen36-quantized-reference-request-v1"
RESULT_SCHEMA = "qwen36-quantized-reference-v1"
READER_SCHEMA = "gturbo-v2-qwen-text-independent-v1"

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
SOURCE_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
SOURCE_INDEX_SHA256 = "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
CONFIG_SHA256 = "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"

TRANSFORMERS_COMMIT = "bd15bc95a89e728bbc1224084eb3b5829428c353"
TRANSFORMERS_TREE = "80eb369e589827bc7ed45b3a1f0ead5457097535"
MODELING_SOURCE_SHA256 = "971b08ed3eb7452f3f5f1b0f8ab8e602fc4c4e2cc10d0a4e99ec1e1830776074"
CONFIGURATION_SOURCE_SHA256 = "9f68bcddc54b4e512802e18ec8a242514d7f746795b28373805d3e05a981f573"

PYTHON_VERSION = "3.12.3"
TORCH_VERSION = "2.10.0"
TRANSFORMERS_VERSION = "5.18.0.dev0"
NUMPY_VERSION = "2.4.3"

ALIGNMENT = 16_384
GROUP_SIZE = 64
VOCABULARY_SIZE = 248_320
HIDDEN_SIZE = 2_048
LAYER_COUNT = 40
EXPERT_COUNT = 256
TOP_K = 8
ROUTED_INTERMEDIATE = 512
SHARED_INTERMEDIATE = 512
EOS_TOKEN_ID = 248_044
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_LAYOUT_BYTES = 16 * 1024 * 1024
MAX_REQUEST_BYTES = 64 * 1024
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_TRANSIENT_DECODE_BYTES = 64 * 1024 * 1024
HEAD_CHUNK_ROWS = 2_048
MIN_PHYSICAL_MEMORY_BYTES = 24 * 1024 * 1024 * 1024
MIN_OUTPUT_FREE_BYTES = 512 * 1024 * 1024

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CASE_ID_RE = re.compile(r"^[\x21-\x7e]{1,64}$")

OFFLINE_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "USE_HUB_KERNELS": "0",
}

SIDECAR_SHA256 = {
    "config.json": CONFIG_SHA256,
    "configuration.json": "c1b09db419119513247e9b8b912c4b9897106c9b20c6cada7e107d993c5435eb",
    "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
    "model.safetensors.index.json": SOURCE_INDEX_SHA256,
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
}

EXPECTED_LAYER_TYPES = [
    "fullAttention" if (index + 1) % 4 == 0 else "linearAttention"
    for index in range(LAYER_COUNT)
]

EXPECTED_ARCHITECTURE = {
    "hiddenSize": HIDDEN_SIZE,
    "numLayers": LAYER_COUNT,
    "layerTypes": EXPECTED_LAYER_TYPES,
    "numAttentionHeads": 16,
    "numKeyValueHeads": 2,
    "headDimension": 256,
    "attentionOutputGate": True,
    "linearConvolutionKernel": 4,
    "linearKeyHeads": 16,
    "linearKeyHeadDimension": 128,
    "linearValueHeads": 32,
    "linearValueHeadDimension": 128,
    "recurrentStateType": "fp32",
    "partialRotaryFactor": 0.25,
    "ropeTheta": 10_000_000,
    "mropeInterleaved": True,
    "mropeSections": [11, 11, 10],
    "numberOfExperts": EXPERT_COUNT,
    "expertsPerToken": TOP_K,
    "routedExpertIntermediateSize": ROUTED_INTERMEDIATE,
    "sharedExpertIntermediateSize": SHARED_INTERMEDIATE,
    "vocabularySize": VOCABULARY_SIZE,
    "tiedWordEmbeddings": False,
    "hiddenActivation": "silu",
    "bosTokenID": EOS_TOKEN_ID,
    "eosTokenID": EOS_TOKEN_ID,
    "imageTokenID": 248_056,
    "videoTokenID": 248_057,
    "visionStartTokenID": 248_053,
    "visionEndTokenID": 248_054,
}

EXPECTED_QUANTIZATION = {
    "embedding": ("affineInt8", 64, "bf16", "bf16"),
    "attention": ("affineInt4", 64, "bf16", "bf16"),
    "linearAttention": ("affineInt8", 64, "bf16", "bf16"),
    "router": ("affineInt8", 64, "bf16", "bf16"),
    "sharedExpert": ("affineInt4", 64, "bf16", "bf16"),
    "routedExpert": ("affineInt4", 64, "bf16", "bf16"),
    "outputHead": ("affineInt8", 64, "bf16", "bf16"),
    "normalization": ("bf16", None, None, None),
    "recurrentState": ("fp32", None, None, None),
}

IGNORED_MTP_SUFFIXES = {
    "fc.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.mlp.experts.down_proj",
    "layers.0.mlp.experts.gate_up_proj",
    "layers.0.mlp.gate.weight",
    "layers.0.mlp.shared_expert.down_proj.weight",
    "layers.0.mlp.shared_expert.gate_proj.weight",
    "layers.0.mlp.shared_expert.up_proj.weight",
    "layers.0.mlp.shared_expert_gate.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.k_norm.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.self_attn.q_norm.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
}

TOLERANCES = {
    "exactInteger": {"absolute": 0.0, "relative": 0.0},
    "fp32": {
        "absolute": 1e-5,
        "relative": 1e-5,
        "comparisonRule": (
            "maxAbs <= absolute + relative * "
            "max(maxAbsReference,maxAbsCandidate)"
        ),
        "source": "fixed before Swift runtime implementation; packed-byte CPU oracle",
        "fixtureSHA256": "e07372907e09f7deab72abd64f417b6f514953a098ca51d53845dc832ae6b1ec",
    },
    "stateFP32": {
        "absolute": 2e-5,
        "relative": 2e-5,
        "source": "fixed before Swift runtime implementation; chunk/recurrent reassociation",
    },
}


class ReferenceFailure(Exception):
    def __init__(self, category: str, exit_code: int, detail: str):
        super().__init__(detail)
        self.category = category
        self.exit_code = exit_code
        self.detail = bounded_detail(detail)


class ReferenceCancelled(ReferenceFailure):
    def __init__(self, detail: str):
        super().__init__("cancelled", 8, detail)


class StrictArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ReferenceFailure("usage", 2, message)


@dataclasses.dataclass(frozen=True)
class RequestCase:
    identifier: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int


@dataclasses.dataclass(frozen=True)
class FileRecord:
    relative_path: str
    path: Path
    size: int
    sha256: str
    descriptor: int


@dataclasses.dataclass(frozen=True)
class TensorRegion:
    name: str
    file: str
    offset: int
    size: int
    shape: tuple[int, ...]
    storage: str
    category: str


@dataclasses.dataclass(frozen=True)
class AffineSource:
    name: str
    role: str
    shape: tuple[int, ...]
    values_offset: int
    values_size: int
    scales_offset: int
    scales_size: int
    biases_offset: int
    biases_size: int

    @property
    def end(self) -> int:
        return self.biases_offset + self.biases_size


@dataclasses.dataclass(frozen=True)
class ExpertLayer:
    layer: int
    path: str
    stride: int
    sources: dict[str, AffineSource]


@dataclasses.dataclass
class ValidatedPack:
    root: Path
    manifest_sha256: str
    policy_sha256: str
    source_revision: str
    source_index_sha256: str
    files: dict[str, FileRecord]
    regions: dict[str, TensorRegion]
    expert_regions: dict[tuple[int, int], TensorRegion]
    expert_layers: tuple[ExpertLayer, ...]
    expert_stride: int

    def close(self) -> None:
        for record in self.files.values():
            try:
                os.close(record.descriptor)
            except OSError:
                pass


@dataclasses.dataclass(frozen=True)
class ReferenceModules:
    np: Any
    torch: Any
    functional: Any
    dynamic_cache: Any
    decoder_layer: Any
    rms_norm: Any
    rotary_embedding: Any
    text_config: Any
    create_causal_mask: Any
    create_recurrent_attention_mask: Any
    environment_record: dict[str, Any]


def bounded_detail(value: object, maximum: int = 500) -> str:
    text = " ".join(str(value).replace("\x00", "\\0").splitlines()).strip()
    return text[:maximum] if text else "unspecified failure"


def fail(category: str, exit_code: int, detail: str) -> NoReturn:
    raise ReferenceFailure(category, exit_code, detail)


def require(condition: bool, category: str, exit_code: int, detail: str) -> None:
    if not condition:
        fail(category, exit_code, detail)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def exact_keys(value: Any, expected: set[str], label: str, category: str = "identityOrFormat") -> dict[str, Any]:
    require(isinstance(value, dict), category, 5 if category == "identityOrFormat" else 4,
            f"{label} must be an object")
    actual = set(value)
    require(actual == expected, category, 5 if category == "identityOrFormat" else 4,
            f"{label} keys mismatch: expected {sorted(expected)}, got {sorted(actual)}")
    return value


def duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def decode_json(data: bytes, label: str, category: str) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=duplicate_rejecting_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        fail(category, 5 if category == "identityOrFormat" else 4,
             f"{label}: {error}")


def read_bounded(path: Path, maximum: int, label: str, category: str) -> bytes:
    try:
        info = path.stat()
        require(stat.S_ISREG(info.st_mode), category, 5 if category == "identityOrFormat" else 4,
                f"{label} is not a regular file")
        require(0 < info.st_size <= maximum, category,
                5 if category == "identityOrFormat" else 4,
                f"{label} size {info.st_size} is outside 1...{maximum}")
        return path.read_bytes()
    except ReferenceFailure:
        raise
    except OSError as error:
        fail(category, 5 if category == "identityOrFormat" else 4,
             f"cannot read {label}: {error}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_sha256(value: Any, label: str, category: str = "identityOrFormat") -> str:
    code = 5 if category == "identityOrFormat" else 4
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            category, code, f"{label} must be 64 lowercase hexadecimal characters")
    return value


def product(shape: tuple[int, ...]) -> int:
    result = 1
    for extent in shape:
        require(is_int(extent) and extent > 0, "identityOrFormat", 5,
                f"invalid tensor shape {shape}")
        result *= extent
        require(result <= sys.maxsize, "resource", 6,
                f"tensor shape is not addressable: {shape}")
    return result


def affine_component_sizes(shape: tuple[int, ...], bit_width: int) -> tuple[int, int, int]:
    require(bit_width in (4, 8), "identityOrFormat", 5, "invalid affine bit width")
    count = product(shape)
    columns = shape[-1]
    rows = count // columns
    values_per_row = (columns * bit_width + 7) // 8
    groups_per_row = (columns + GROUP_SIZE - 1) // GROUP_SIZE
    values = rows * values_per_row
    metadata = rows * groups_per_row * 2
    return values, metadata, values + 2 * metadata


def expected_payload_paths() -> set[str]:
    return {
        "model_weights.bin",
        "packed_experts/layout.json",
        *(f"packed_experts/layer_{layer:02d}.bin" for layer in range(LAYER_COUNT)),
    }


def expected_resident_tensors() -> dict[str, tuple[tuple[int, ...], str, str]]:
    result: dict[str, tuple[tuple[int, ...], str, str]] = {
        "model.language_model.embed_tokens.weight": (
            (VOCABULARY_SIZE, HIDDEN_SIZE), "affineInt8", "embedding"),
        "model.language_model.norm.weight": ((HIDDEN_SIZE,), "bf16", "normalization"),
        "lm_head.weight": ((VOCABULARY_SIZE, HIDDEN_SIZE), "affineInt8", "outputHead"),
    }
    for layer in range(LAYER_COUNT):
        prefix = f"model.language_model.layers.{layer}."
        result[prefix + "input_layernorm.weight"] = ((HIDDEN_SIZE,), "bf16", "normalization")
        result[prefix + "post_attention_layernorm.weight"] = ((HIDDEN_SIZE,), "bf16", "normalization")
        result[prefix + "mlp.gate.weight"] = ((EXPERT_COUNT, HIDDEN_SIZE), "affineInt8", "router")
        result[prefix + "mlp.shared_expert_gate.weight"] = ((1, HIDDEN_SIZE), "affineInt8", "router")
        result[prefix + "mlp.shared_expert.gate_proj.weight"] = (
            (SHARED_INTERMEDIATE, HIDDEN_SIZE), "affineInt4", "sharedExpert")
        result[prefix + "mlp.shared_expert.up_proj.weight"] = (
            (SHARED_INTERMEDIATE, HIDDEN_SIZE), "affineInt4", "sharedExpert")
        result[prefix + "mlp.shared_expert.down_proj.weight"] = (
            (HIDDEN_SIZE, SHARED_INTERMEDIATE), "affineInt4", "sharedExpert")
        if layer % 4 == 3:
            result[prefix + "self_attn.q_proj.weight"] = ((8192, HIDDEN_SIZE), "affineInt4", "attention")
            result[prefix + "self_attn.k_proj.weight"] = ((512, HIDDEN_SIZE), "affineInt4", "attention")
            result[prefix + "self_attn.v_proj.weight"] = ((512, HIDDEN_SIZE), "affineInt4", "attention")
            result[prefix + "self_attn.o_proj.weight"] = ((HIDDEN_SIZE, 4096), "affineInt4", "attention")
            result[prefix + "self_attn.q_norm.weight"] = ((256,), "bf16", "normalization")
            result[prefix + "self_attn.k_norm.weight"] = ((256,), "bf16", "normalization")
        else:
            result[prefix + "linear_attn.dt_bias"] = ((32,), "affineInt8", "linearAttention")
            result[prefix + "linear_attn.A_log"] = ((32,), "affineInt8", "linearAttention")
            result[prefix + "linear_attn.conv1d.weight"] = ((8192, 1, 4), "affineInt8", "linearAttention")
            result[prefix + "linear_attn.norm.weight"] = ((128,), "bf16", "normalization")
            result[prefix + "linear_attn.out_proj.weight"] = ((HIDDEN_SIZE, 4096), "affineInt8", "linearAttention")
            result[prefix + "linear_attn.in_proj_qkv.weight"] = ((8192, HIDDEN_SIZE), "affineInt8", "linearAttention")
            result[prefix + "linear_attn.in_proj_z.weight"] = ((4096, HIDDEN_SIZE), "affineInt8", "linearAttention")
            result[prefix + "linear_attn.in_proj_b.weight"] = ((32, HIDDEN_SIZE), "affineInt8", "linearAttention")
            result[prefix + "linear_attn.in_proj_a.weight"] = ((32, HIDDEN_SIZE), "affineInt8", "linearAttention")
    require(len(result) == 613, "execution", 7,
            f"internal expected resident count is {len(result)}, not 613")
    return result


def validate_request(path: Path) -> tuple[RequestCase, ...]:
    root = decode_json(read_bounded(path, MAX_REQUEST_BYTES, "request", "request"),
                       "request", "request")
    root = exact_keys(root, {"schemaVersion", "cases"}, "request", "request")
    require(root["schemaVersion"] == REQUEST_SCHEMA, "request", 4,
            f"unsupported request schema {root['schemaVersion']!r}")
    cases = root["cases"]
    require(isinstance(cases, list) and 1 <= len(cases) <= 2, "request", 4,
            "request cases must contain one or two entries")
    identifiers: set[str] = set()
    decoded: list[RequestCase] = []
    for index, raw in enumerate(cases):
        item = exact_keys(raw, {"id", "promptTokenIDs", "maxNewTokens"},
                          f"request.cases[{index}]", "request")
        identifier = item["id"]
        require(isinstance(identifier, str) and CASE_ID_RE.fullmatch(identifier) is not None,
                "request", 4, f"case {index} id is not bounded printable ASCII")
        require(identifier not in identifiers, "request", 4,
                f"duplicate case id {identifier!r}")
        identifiers.add(identifier)
        token_ids = item["promptTokenIDs"]
        require(isinstance(token_ids, list) and 1 <= len(token_ids) <= 32,
                "request", 4, f"case {identifier} prompt length is outside 1...32")
        for token_index, token_id in enumerate(token_ids):
            require(is_int(token_id) and 0 <= token_id < VOCABULARY_SIZE,
                    "request", 4,
                    f"case {identifier} token {token_index} is outside the vocabulary")
        max_new_tokens = item["maxNewTokens"]
        require(is_int(max_new_tokens) and 1 <= max_new_tokens <= 2,
                "request", 4, f"case {identifier} maxNewTokens must be 1 or 2")
        decoded.append(RequestCase(identifier, tuple(token_ids), max_new_tokens))
    return tuple(decoded)


def safe_payload_path(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and relative, "identityOrFormat", 5,
            "manifest contains an empty payload path")
    require("\\" not in relative and "\x00" not in relative and not relative.startswith("/"),
            "identityOrFormat", 5, f"unsafe payload path {relative!r}")
    parts = relative.split("/")
    require(all(part not in ("", ".", "..") for part in parts),
            "identityOrFormat", 5, f"unsafe payload path {relative!r}")
    candidate = root.joinpath(*parts)
    cursor = root
    for part in parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), "identityOrFormat", 5,
                f"payload path contains a symbolic link: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        fail("identityOrFormat", 5, f"payload path escapes model directory: {relative}: {error}")
    return resolved


def open_and_hash_payload(root: Path, relative: str, declared_size: int,
                          declared_sha256: str) -> FileRecord:
    path = safe_payload_path(root, relative)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), "identityOrFormat", 5,
                f"payload is not a regular file: {relative}")
        require(info.st_size == declared_size, "identityOrFormat", 5,
                f"payload size mismatch for {relative}: {info.st_size} != {declared_size}")
        digest = hashlib.sha256()
        offset = 0
        while offset < declared_size:
            chunk = os.pread(descriptor, min(8 * 1024 * 1024, declared_size - offset), offset)
            require(bool(chunk), "identityOrFormat", 5,
                    f"short read while hashing {relative}")
            digest.update(chunk)
            offset += len(chunk)
        observed = digest.hexdigest()
        require(observed == declared_sha256, "identityOrFormat", 5,
                f"payload SHA-256 mismatch for {relative}")
        return FileRecord(relative, path, declared_size, observed, descriptor)
    except ReferenceFailure:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        fail("identityOrFormat", 5, f"cannot verify payload {relative}: {error}")


def validate_quantization(raw: Any) -> None:
    require(isinstance(raw, list) and len(raw) == len(EXPECTED_QUANTIZATION),
            "identityOrFormat", 5, "manifest quantization groups are incomplete")
    observed: dict[str, tuple[str, int | None, str | None, str | None]] = {}
    for index, item_raw in enumerate(raw):
        require(isinstance(item_raw, dict), "identityOrFormat", 5,
                f"manifest.quantization[{index}] must be an object")
        storage = item_raw.get("storage")
        affine = storage in ("affineInt4", "affineInt8")
        keys = {"category", "storage", "groupSize", "scaleType", "biasType"} if affine \
            else {"category", "storage"}
        item = exact_keys(item_raw, keys, f"manifest.quantization[{index}]")
        category = item["category"]
        require(isinstance(category, str) and category not in observed,
                "identityOrFormat", 5, "duplicate or invalid quantization category")
        observed[category] = (
            storage,
            item.get("groupSize"),
            item.get("scaleType"),
            item.get("biasType"),
        )
    require(observed == EXPECTED_QUANTIZATION, "identityOrFormat", 5,
            "manifest quantization profile differs from frozen Phase 4 policy")


def validate_ignored_tensors(raw: Any) -> None:
    require(isinstance(raw, list), "identityOrFormat", 5,
            "manifest ignoredTensors must be an array")
    observed: set[str] = set()
    for index, item_raw in enumerate(raw):
        item = exact_keys(item_raw, {"name", "reason"},
                          f"manifest.ignoredTensors[{index}]")
        require(item["reason"] == "unsupportedMTP" and isinstance(item["name"], str),
                "identityOrFormat", 5, "invalid ignored tensor record")
        require(item["name"] not in observed, "identityOrFormat", 5,
                f"duplicate ignored tensor {item['name']!r}")
        observed.add(item["name"])
    expected = {"mtp." + suffix for suffix in IGNORED_MTP_SUFFIXES}
    require(observed == expected, "identityOrFormat", 5,
            "manifest ignored tensor set differs from the exact MTP omission set")


def parse_manifest_files(root: Path, raw: Any) -> dict[str, FileRecord]:
    require(isinstance(raw, dict) and set(raw) == expected_payload_paths(),
            "identityOrFormat", 5,
            "manifest payload path set differs from the exact text pack set")
    records: dict[str, FileRecord] = {}
    try:
        for relative in sorted(raw):
            item = exact_keys(raw[relative], {"size", "sha256"},
                              f"manifest.files.{relative}")
            size = item["size"]
            require(is_int(size) and 0 < size <= sys.maxsize,
                    "identityOrFormat", 5, f"invalid payload size for {relative}")
            digest = validate_sha256(item["sha256"], f"manifest.files.{relative}.sha256")
            records[relative] = open_and_hash_payload(root, relative, size, digest)
        return records
    except BaseException:
        for record in records.values():
            try:
                os.close(record.descriptor)
            except OSError:
                pass
        raise


def parse_tensor_regions(raw: Any, files: dict[str, FileRecord]) -> tuple[
    dict[str, TensorRegion], dict[tuple[int, int], TensorRegion]
]:
    require(isinstance(raw, list) and len(raw) == 613 + LAYER_COUNT * EXPERT_COUNT,
            "identityOrFormat", 5,
            "manifest tensor region count is not 10853")
    expected_resident = expected_resident_tensors()
    resident: dict[str, TensorRegion] = {}
    expert: dict[tuple[int, int], TensorRegion] = {}
    ranges: dict[str, list[tuple[int, int, str]]] = {path: [] for path in files}
    expert_name = re.compile(r"^qwen\.layer\.(\d+)\.expert\.(\d+)$")
    for index, item_raw in enumerate(raw):
        item = exact_keys(
            item_raw,
            {"name", "file", "offset", "size", "shape", "storage", "quantizationCategory"},
            f"manifest.tensorRegions[{index}]",
        )
        name = item["name"]
        relative = item["file"]
        offset = item["offset"]
        size = item["size"]
        shape_raw = item["shape"]
        storage = item["storage"]
        category = item["quantizationCategory"]
        require(isinstance(name, str) and name and "\x00" not in name,
                "identityOrFormat", 5, f"invalid tensor region name at index {index}")
        require(relative in files, "identityOrFormat", 5,
                f"tensor {name} references an unknown file")
        require(is_int(offset) and offset >= 0 and offset % ALIGNMENT == 0,
                "identityOrFormat", 5, f"tensor {name} has an invalid offset")
        require(is_int(size) and size > 0 and size <= files[relative].size - offset,
                "identityOrFormat", 5, f"tensor {name} range exceeds {relative}")
        require(isinstance(shape_raw, list) and shape_raw and
                all(is_int(value) and value > 0 for value in shape_raw),
                "identityOrFormat", 5, f"tensor {name} has an invalid shape")
        shape = tuple(shape_raw)
        region = TensorRegion(name, relative, offset, size, shape, storage, category)
        ranges[relative].append((offset, offset + size, name))
        match = expert_name.fullmatch(name)
        if match is None:
            require(name in expected_resident and name not in resident,
                    "identityOrFormat", 5, f"unknown or duplicate resident tensor {name}")
            expected_shape, expected_storage, expected_category = expected_resident[name]
            require((shape, storage, category) ==
                    (expected_shape, expected_storage, expected_category),
                    "identityOrFormat", 5,
                    f"resident tensor shape/storage/category mismatch for {name}")
            if storage == "bf16":
                expected_size = product(shape) * 2
            elif storage == "fp32":
                expected_size = product(shape) * 4
            else:
                bit_width = 4 if storage == "affineInt4" else 8
                expected_size = affine_component_sizes(shape, bit_width)[2]
            require(size == expected_size and relative == "model_weights.bin",
                    "identityOrFormat", 5,
                    f"resident tensor byte layout mismatch for {name}")
            resident[name] = region
        else:
            layer = int(match.group(1))
            expert_id = int(match.group(2))
            key = (layer, expert_id)
            require(0 <= layer < LAYER_COUNT and 0 <= expert_id < EXPERT_COUNT and
                    key not in expert,
                    "identityOrFormat", 5, f"invalid or duplicate expert region {name}")
            require(relative == f"packed_experts/layer_{layer:02d}.bin" and
                    storage == "affineInt4" and category == "routedExpert" and
                    shape == (size,),
                    "identityOrFormat", 5, f"expert region metadata mismatch for {name}")
            expert[key] = region
    require(set(resident) == set(expected_resident), "identityOrFormat", 5,
            "resident tensor set is incomplete")
    require(len(expert) == LAYER_COUNT * EXPERT_COUNT, "identityOrFormat", 5,
            "expert region set is incomplete")
    for relative, file_ranges in ranges.items():
        ordered = sorted(file_ranges)
        for previous, current in zip(ordered, ordered[1:]):
            require(previous[1] <= current[0], "identityOrFormat", 5,
                    f"overlapping tensor regions {previous[2]} and {current[2]} in {relative}")
    return resident, expert


def parse_expert_layout(data: bytes, files: dict[str, FileRecord],
                        expert_stride: int,
                        expert_regions: dict[tuple[int, int], TensorRegion]) -> tuple[ExpertLayer, ...]:
    raw = decode_json(data, "packed_experts/layout.json", "identityOrFormat")
    root = exact_keys(raw, {"version", "layers"}, "packed_experts/layout.json")
    require(root["version"] == 2, "identityOrFormat", 5,
            "packed expert layout version is not 2")
    layers_raw = root["layers"]
    require(isinstance(layers_raw, list) and len(layers_raw) == LAYER_COUNT,
            "identityOrFormat", 5, "packed expert layout must contain 40 layers")
    layers: dict[int, ExpertLayer] = {}
    seen_paths: set[str] = set()
    source_keys = {
        "name", "role", "shape", "valuesOffset", "valuesSize",
        "scalesOffset", "scalesSize", "biasesOffset", "biasesSize",
    }
    for position, layer_raw in enumerate(layers_raw):
        item = exact_keys(layer_raw, {"layer", "path", "experts", "stride", "sources"},
                          f"layout.layers[{position}]")
        layer = item["layer"]
        path = item["path"]
        require(is_int(layer) and 0 <= layer < LAYER_COUNT and layer not in layers,
                "identityOrFormat", 5, "layout has duplicate or invalid layer index")
        require(path == f"packed_experts/layer_{layer:02d}.bin" and path not in seen_paths,
                "identityOrFormat", 5, f"layout path mismatch for layer {layer}")
        seen_paths.add(path)
        require(item["experts"] == EXPERT_COUNT and item["stride"] == expert_stride,
                "identityOrFormat", 5, f"layout dimensions mismatch for layer {layer}")
        require(files[path].size == EXPERT_COUNT * expert_stride,
                "identityOrFormat", 5, f"expert file size mismatch for layer {layer}")
        sources_raw = item["sources"]
        require(isinstance(sources_raw, list) and len(sources_raw) == 2,
                "identityOrFormat", 5, f"layer {layer} must have two routed sources")
        sources: dict[str, AffineSource] = {}
        occupied: list[tuple[int, int, str]] = []
        prefix = f"model.language_model.layers.{layer}.mlp.experts."
        for source_index, source_raw in enumerate(sources_raw):
            source_item = exact_keys(source_raw, source_keys,
                                     f"layout.layers[{layer}].sources[{source_index}]")
            role = source_item["role"]
            expected_shape = (1024, HIDDEN_SIZE) if role == "gate_up" \
                else (HIDDEN_SIZE, ROUTED_INTERMEDIATE) if role == "down" else None
            require(expected_shape is not None and role not in sources,
                    "identityOrFormat", 5, f"layer {layer} has duplicate or unknown source role")
            require(source_item["name"] == prefix +
                    ("gate_up_proj" if role == "gate_up" else "down_proj"),
                    "identityOrFormat", 5, f"layer {layer} source name mismatch for {role}")
            require(source_item["shape"] == list(expected_shape),
                    "identityOrFormat", 5, f"layer {layer} source shape mismatch for {role}")
            values_size, metadata_size, _ = affine_component_sizes(expected_shape, 4)
            numeric_names = (
                "valuesOffset", "valuesSize", "scalesOffset", "scalesSize",
                "biasesOffset", "biasesSize",
            )
            require(all(is_int(source_item[name]) and source_item[name] >= 0
                        for name in numeric_names),
                    "identityOrFormat", 5, f"layer {layer} source offsets are invalid")
            source = AffineSource(
                source_item["name"], role, expected_shape,
                source_item["valuesOffset"], source_item["valuesSize"],
                source_item["scalesOffset"], source_item["scalesSize"],
                source_item["biasesOffset"], source_item["biasesSize"],
            )
            require(source.values_size == values_size and
                    source.scales_size == metadata_size and
                    source.biases_size == metadata_size and
                    source.scales_offset == source.values_offset + source.values_size and
                    source.biases_offset == source.scales_offset + source.scales_size and
                    source.scales_offset % 2 == 0 and source.biases_offset % 2 == 0 and
                    source.end <= expert_stride,
                    "identityOrFormat", 5,
                    f"layer {layer} affine component layout mismatch for {role}")
            occupied.extend([
                (source.values_offset, source.values_offset + source.values_size, role + ".values"),
                (source.scales_offset, source.scales_offset + source.scales_size, role + ".scales"),
                (source.biases_offset, source.biases_offset + source.biases_size, role + ".biases"),
            ])
            sources[role] = source
        require(set(sources) == {"gate_up", "down"}, "identityOrFormat", 5,
                f"layer {layer} routed roles are incomplete")
        ordered_ranges = sorted(occupied)
        require(ordered_ranges[0][0] == 0, "identityOrFormat", 5,
                f"layer {layer} components do not start at zero")
        for previous, current in zip(ordered_ranges, ordered_ranges[1:]):
            require(previous[1] == current[0], "identityOrFormat", 5,
                    f"layer {layer} components are not contiguous")
        used = ordered_ranges[-1][1]
        for expert_id in range(EXPERT_COUNT):
            region = expert_regions[(layer, expert_id)]
            require(region.offset == expert_id * expert_stride and region.size == used,
                    "identityOrFormat", 5,
                    f"layer {layer} expert {expert_id} region disagrees with layout")
        layers[layer] = ExpertLayer(layer, path, expert_stride, sources)
    require(set(layers) == set(range(LAYER_COUNT)), "identityOrFormat", 5,
            "packed expert layout layer set is incomplete")
    return tuple(layers[index] for index in range(LAYER_COUNT))


def validate_official_config(path: Path, architecture: dict[str, Any]) -> dict[str, Any]:
    data = read_bounded(path, MAX_MANIFEST_BYTES, "official config", "identityOrFormat")
    require(sha256_bytes(data) == CONFIG_SHA256, "identityOrFormat", 5,
            "official config SHA-256 differs from pinned provenance")
    root = decode_json(data, "official config", "identityOrFormat")
    require(isinstance(root, dict) and isinstance(root.get("text_config"), dict),
            "identityOrFormat", 5, "official config has no text_config object")
    text = root["text_config"]
    layer_types = [
        "fullAttention" if value == "full_attention" else
        "linearAttention" if value == "linear_attention" else "invalid"
        for value in text.get("layer_types", [])
    ]
    observed = {
        "hiddenSize": text.get("hidden_size"),
        "numLayers": text.get("num_hidden_layers"),
        "layerTypes": layer_types,
        "numAttentionHeads": text.get("num_attention_heads"),
        "numKeyValueHeads": text.get("num_key_value_heads"),
        "headDimension": text.get("head_dim"),
        "attentionOutputGate": text.get("attn_output_gate"),
        "linearConvolutionKernel": text.get("linear_conv_kernel_dim"),
        "linearKeyHeads": text.get("linear_num_key_heads"),
        "linearKeyHeadDimension": text.get("linear_key_head_dim"),
        "linearValueHeads": text.get("linear_num_value_heads"),
        "linearValueHeadDimension": text.get("linear_value_head_dim"),
        "recurrentStateType": (
            "fp32" if text.get("mamba_ssm_dtype") == "float32"
            else text.get("mamba_ssm_dtype")
        ),
        "partialRotaryFactor": text.get("partial_rotary_factor"),
        "ropeTheta": text.get("rope_parameters", {}).get("rope_theta"),
        "mropeInterleaved": text.get("rope_parameters", {}).get("mrope_interleaved"),
        "mropeSections": text.get("rope_parameters", {}).get("mrope_section"),
        "numberOfExperts": text.get("num_experts"),
        "expertsPerToken": text.get("num_experts_per_tok"),
        "routedExpertIntermediateSize": text.get("moe_intermediate_size"),
        "sharedExpertIntermediateSize": text.get("shared_expert_intermediate_size"),
        "vocabularySize": text.get("vocab_size"),
        "tiedWordEmbeddings": text.get("tie_word_embeddings"),
        "hiddenActivation": text.get("hidden_act"),
        "bosTokenID": text.get("bos_token_id"),
        "eosTokenID": text.get("eos_token_id"),
        "imageTokenID": root.get("image_token_id"),
        "videoTokenID": root.get("video_token_id"),
        "visionStartTokenID": root.get("vision_start_token_id"),
        "visionEndTokenID": root.get("vision_end_token_id"),
    }
    require(observed == architecture == EXPECTED_ARCHITECTURE,
            "identityOrFormat", 5,
            "official config, manifest architecture, and pinned architecture disagree")
    require(text.get("rope_parameters", {}).get("rope_type") == "default" and
            text.get("rope_parameters", {}).get("partial_rotary_factor") == 0.25 and
            text.get("attention_bias") is False and
            text.get("attention_dropout") == 0.0 and
            text.get("rms_norm_eps") == 1e-6 and
            text.get("use_cache") is True,
            "identityOrFormat", 5, "official text execution configuration is not pinned")
    return root


def validate_manifest(model_directory: Path, official_config: Path,
                      expected_manifest_sha256: str,
                      expected_policy_sha256: str) -> tuple[ValidatedPack, dict[str, Any]]:
    try:
        root = model_directory.resolve(strict=True)
    except OSError as error:
        fail("identityOrFormat", 5, f"model directory cannot be resolved: {error}")
    require(root.is_dir() and not root.is_symlink(), "identityOrFormat", 5,
            "model directory must be a real directory")
    manifest_path = root / "manifest.json"
    require(not manifest_path.is_symlink(), "identityOrFormat", 5,
            "manifest.json must not be a symbolic link")
    manifest_data = read_bounded(
        manifest_path, MAX_MANIFEST_BYTES, "manifest.json", "identityOrFormat")
    manifest_sha256 = sha256_bytes(manifest_data)
    require(manifest_sha256 == expected_manifest_sha256,
            "identityOrFormat", 5,
            "manifest SHA-256 differs from the explicit Phase 21 receipt identity")
    manifest = decode_json(manifest_data, "manifest.json", "identityOrFormat")
    manifest = exact_keys(manifest, {
        "magic", "versionMajor", "versionMinor", "family", "requiredFeatures",
        "modelID", "architecture", "provenance", "quantization", "ignoredTensors",
        "files", "tensorRegions", "expertsPerLayer", "numLayers", "expertStride",
    }, "manifest")
    require(manifest["magic"] == "GTURBO" and manifest["versionMajor"] == 2 and
            manifest["versionMinor"] == 0 and manifest["family"] == "qwen3_6" and
            manifest["modelID"] == MODEL_ID,
            "identityOrFormat", 5, "manifest header or family identity is not pinned Qwen v2")
    require(isinstance(manifest["requiredFeatures"], list) and
            all(isinstance(value, str) for value in manifest["requiredFeatures"]) and
            set(manifest["requiredFeatures"]) == {
        "familyDispatch", "verifiedIdentity", "qwenHybridAttention", "qwenMTPExcluded"
    } and len(manifest["requiredFeatures"]) == 4,
            "identityOrFormat", 5, "manifest required feature set is invalid")
    architecture_wire = exact_keys(
        manifest["architecture"], {"family", "configuration"}, "manifest.architecture")
    require(architecture_wire["family"] == "qwen3_6" and
            architecture_wire["configuration"] == EXPECTED_ARCHITECTURE,
            "identityOrFormat", 5, "manifest architecture is not exact pinned Qwen")
    provenance = exact_keys(
        manifest["provenance"],
        {"sourceRepository", "sourceRevision", "sourceIndexSHA256", "sidecarSHA256",
         "quantizationPolicySHA256"},
        "manifest.provenance",
    )
    policy_sha256 = validate_sha256(
        provenance["quantizationPolicySHA256"],
        "manifest.provenance.quantizationPolicySHA256")
    require(provenance["sourceRepository"] == MODEL_ID and
            provenance["sourceRevision"] == SOURCE_REVISION and
            provenance["sourceIndexSHA256"] == SOURCE_INDEX_SHA256 and
            provenance["sidecarSHA256"] == SIDECAR_SHA256 and
            policy_sha256 == expected_policy_sha256,
            "identityOrFormat", 5, "manifest provenance differs from pinned receipt identity")
    validate_quantization(manifest["quantization"])
    validate_ignored_tensors(manifest["ignoredTensors"])
    require(manifest["expertsPerLayer"] == EXPERT_COUNT and
            manifest["numLayers"] == LAYER_COUNT and
            is_int(manifest["expertStride"]) and manifest["expertStride"] > 0 and
            manifest["expertStride"] % ALIGNMENT == 0 and
            manifest["expertStride"] <= 0xFFFF_FFFF,
            "identityOrFormat", 5, "manifest expert streaming dimensions are invalid")
    config_document = validate_official_config(
        official_config, architecture_wire["configuration"])

    files: dict[str, FileRecord] = {}
    try:
        files = parse_manifest_files(root, manifest["files"])
        regions, expert_regions = parse_tensor_regions(manifest["tensorRegions"], files)
        layout_file = files["packed_experts/layout.json"]
        require(layout_file.size <= MAX_LAYOUT_BYTES, "identityOrFormat", 5,
                "packed expert layout exceeds its bounded metadata limit")
        layout_data = os.pread(layout_file.descriptor, layout_file.size, 0)
        require(len(layout_data) == layout_file.size, "identityOrFormat", 5,
                "short read for packed expert layout")
        expert_layers = parse_expert_layout(
            layout_data, files, manifest["expertStride"], expert_regions)
        return ValidatedPack(
            root=root,
            manifest_sha256=manifest_sha256,
            policy_sha256=policy_sha256,
            source_revision=provenance["sourceRevision"],
            source_index_sha256=provenance["sourceIndexSHA256"],
            files=files,
            regions=regions,
            expert_regions=expert_regions,
            expert_layers=expert_layers,
            expert_stride=manifest["expertStride"],
        ), config_document
    except BaseException:
        for record in files.values():
            try:
                os.close(record.descriptor)
            except OSError:
                pass
        raise


def validate_paths(
    model_directory: Path,
    official_config: Path,
    transformers_checkout: Path,
    request: Path,
    output: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    try:
        model = model_directory.resolve(strict=True)
        config = official_config.resolve(strict=True)
        checkout = transformers_checkout.resolve(strict=True)
        request_path = request.resolve(strict=True)
        output_parent = output.parent.resolve(strict=True)
        output_path = output_parent / output.name
    except OSError as error:
        fail("request", 4, f"input or output parent cannot be resolved: {error}")
    require(output.name not in ("", ".", "..") and "/" not in output.name,
            "request", 4, "output must name a file in an existing directory")
    require(output_path not in {
        config, request_path, checkout, model / "manifest.json",
        Path(__file__).resolve(),
    },
            "request", 4, "output path must be distinct from every input")
    try:
        output_path.relative_to(model)
        inside_model = True
    except ValueError:
        inside_model = False
    require(not inside_model, "request", 4,
            "output path must be outside the model directory")
    try:
        output_path.relative_to(checkout)
        inside_transformers_checkout = True
    except ValueError:
        inside_transformers_checkout = False
    require(not inside_transformers_checkout, "request", 4,
            "output path must be outside the Transformers checkout")
    repository_root: Path | None = None
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "Package.swift").is_file():
            repository_root = candidate
            break
    require(repository_root is not None, "environment", 3,
            "cannot identify repository root for output isolation")
    for source_tree in ("Sources", "Tests", "Scripts"):
        try:
            output_path.relative_to(repository_root / source_tree)
            inside_source = True
        except ValueError:
            inside_source = False
        require(not inside_source, "request", 4,
                f"output path must be outside repository {source_tree}/")
    require(not os.path.lexists(output_path), "request", 4,
            "output path already exists and is preserved")
    try:
        free = shutil.disk_usage(output_parent).free
    except OSError as error:
        fail("resource", 6, f"cannot inspect output disk: {error}")
    require(free >= MIN_OUTPUT_FREE_BYTES, "resource", 6,
            f"output volume has only {free} free bytes")
    return model, config, checkout, request_path, output_path


def physical_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, KeyError) as error:
        fail("resource", 6, f"cannot determine physical memory: {error}")


def git_output(repository: Path, arguments: list[str]) -> bytes:
    git_environment = {
        name: value for name, value in os.environ.items()
        if not name.startswith("GIT_")
    }
    try:
        completed = subprocess.run(
            ["/usr/bin/git", "--no-replace-objects", "-C", str(repository),
             *arguments],
            check=False, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=30, env=git_environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        fail("environment", 3,
             f"cannot verify pinned Transformers source tree: {error}")
    require(len(completed.stdout) <= 4 * 1024 * 1024 and
            len(completed.stderr) <= 4 * 1024 * 1024,
            "environment", 3, "Transformers Git verification output is unbounded")
    require(completed.returncode == 0, "environment", 3,
            "pinned Transformers source-tree verification failed")
    return completed.stdout


def validate_transformers_checkout(checkout: Path) -> tuple[Path, str, str]:
    try:
        repository = checkout.resolve(strict=True)
    except OSError as error:
        fail("environment", 3,
             f"cannot resolve pinned Transformers checkout: {error}")
    require(repository.is_dir(), "environment", 3,
            "pinned Transformers checkout is not a directory")
    try:
        observed_repository = Path(git_output(repository, [
            "rev-parse", "--show-toplevel",
        ]).decode("utf-8", errors="strict").strip()).resolve(strict=True)
    except (OSError, UnicodeError) as error:
        fail("environment", 3,
             f"cannot resolve pinned Transformers checkout root: {error}")
    require(observed_repository == repository,
            "environment", 3,
            "pinned Transformers path is not the Git checkout root")
    try:
        source_root = (repository / "src").resolve(strict=True)
        package_init = (source_root / "transformers" / "__init__.py").resolve(
            strict=True)
    except OSError as error:
        fail("environment", 3,
             f"cannot resolve pinned Transformers source directory: {error}")
    require(package_init.parent == source_root / "transformers",
            "environment", 3,
            "pinned Transformers package has an unexpected source layout")

    commit = git_output(repository, ["rev-parse", "HEAD^{commit}"]).decode(
        "ascii", errors="strict").strip()
    tree = git_output(repository, ["rev-parse", "HEAD^{tree}"]).decode(
        "ascii", errors="strict").strip()
    require(commit == TRANSFORMERS_COMMIT and tree == TRANSFORMERS_TREE,
            "environment", 3,
            "Transformers checkout differs from the pinned commit/tree")
    tracked_status = git_output(repository, [
        "status", "--porcelain=v1", "--untracked-files=no", "--",
        "src/transformers",
    ])
    require(not tracked_status, "environment", 3,
            "Transformers tracked sources have local modifications")
    untracked = git_output(repository, [
        "ls-files", "--others", "--exclude-standard", "--", "src/transformers",
    ]).splitlines()
    unexpected_untracked = [
        entry for entry in untracked
        if b"/__pycache__/" not in entry and not entry.endswith(b".pyc")
    ]
    require(not unexpected_untracked, "environment", 3,
            "Transformers source tree contains untracked runtime files")
    return source_root, commit, tree


def select_transformers_source(checkout: Path) -> tuple[Path, str, str]:
    source_root, commit, tree = validate_transformers_checkout(checkout)
    preloaded = sorted(
        name for name in sys.modules
        if name == "transformers" or name.startswith("transformers.")
    )
    require(not preloaded, "environment", 3,
            "Transformers was imported before the pinned source was selected")
    source_path = str(source_root)
    sys.path[:] = [
        entry for entry in sys.path
        if entry != source_path
    ]
    sys.path.insert(0, source_path)
    importlib.invalidate_caches()
    return source_root, commit, tree


def validate_transformers_source_tree(
    transformers: Any,
    expected_repository: Path,
) -> tuple[str, str]:
    try:
        package_init = Path(transformers.__file__).resolve(strict=True)
        package_root = package_init.parent
        package_locations = tuple(
            Path(location).resolve(strict=True) for location in transformers.__path__)
    except (AttributeError, OSError, TypeError) as error:
        fail("environment", 3,
             f"cannot resolve executed Transformers package: {error}")
    require(package_init.name == "__init__.py" and
            package_locations == (package_root,),
            "environment", 3,
            "executed Transformers package has an unexpected import path")

    source_root, commit, tree = validate_transformers_checkout(expected_repository)
    repository = source_root.parent
    expected_package_root = source_root / "transformers"
    require(package_root == expected_package_root,
            "environment", 3,
            "executed Transformers package is outside the selected checkout")

    imported_paths: set[str] = set()
    for name, module in tuple(sys.modules.items()):
        if name != "transformers" and not name.startswith("transformers."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            imported_path = Path(module_file).resolve(strict=True)
            imported_path.relative_to(package_root)
            imported_paths.add(str(imported_path.relative_to(repository)))
        except (OSError, ValueError) as error:
            fail("environment", 3,
                 f"executed Transformers module {name} is outside pinned sources: {error}")
    require(str(package_init.relative_to(repository)) in imported_paths,
            "environment", 3,
            "executed Transformers package source was not observed")
    git_output(repository, [
        "ls-files", "--error-unmatch", "--", *sorted(imported_paths),
    ])
    return commit, tree


def validate_environment(
    torch_threads: int,
    transformers_checkout: Path,
) -> ReferenceModules:
    for name, expected in OFFLINE_ENVIRONMENT.items():
        require(os.environ.get(name) == expected, "environment", 3,
                f"{name} must equal {expected!r}")
    require(platform.system() == "Darwin", "environment", 3,
            "reference execution is pinned to Darwin")
    require(platform.python_version() == PYTHON_VERSION, "environment", 3,
            f"Python must be {PYTHON_VERSION}, got {platform.python_version()}")
    require(1 <= torch_threads <= 12, "resource", 6,
            "torch thread count must be in 1...12")
    require(physical_memory_bytes() >= MIN_PHYSICAL_MEMORY_BYTES,
            "resource", 6, "physical memory is below the 24 GiB streaming minimum")
    for package in ("flash_attn", "causal_conv1d", "mlx", "mlx_lm"):
        require(importlib.util.find_spec(package) is None, "environment", 3,
                f"optional execution package must be absent: {package}")
    _, selected_transformers_commit, selected_transformers_tree = (
        select_transformers_source(transformers_checkout))
    try:
        import numpy as np
        import torch
        import torch.nn.functional as functional
        import transformers
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import (
            create_causal_mask,
            create_recurrent_attention_mask,
        )
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
            Qwen3_5MoeTextConfig,
        )
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeDecoderLayer,
            Qwen3_5MoeRMSNorm,
            Qwen3_5MoeTextRotaryEmbedding,
        )
        import transformers.models.qwen3_5_moe.configuration_qwen3_5_moe as configuration_module
        import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as modeling_module
    except ReferenceFailure:
        raise
    except Exception as error:
        fail("environment", 3, f"pinned reference imports failed: {type(error).__name__}: {error}")
    require(np.__version__ == NUMPY_VERSION and torch.__version__ == TORCH_VERSION and
            transformers.__version__ == TRANSFORMERS_VERSION,
            "environment", 3,
            f"reference versions differ: numpy={np.__version__}, torch={torch.__version__}, "
            f"transformers={transformers.__version__}")
    observed_transformers_commit, observed_transformers_tree = (
        validate_transformers_source_tree(transformers, transformers_checkout))
    require(observed_transformers_commit == selected_transformers_commit and
            observed_transformers_tree == selected_transformers_tree,
            "environment", 3,
            "Transformers checkout changed while importing the runtime")
    modeling_path = Path(modeling_module.__file__).resolve()
    configuration_path = Path(configuration_module.__file__).resolve()
    try:
        modeling_digest = sha256_bytes(modeling_path.read_bytes())
        configuration_digest = sha256_bytes(configuration_path.read_bytes())
    except OSError as error:
        fail("environment", 3, f"cannot hash installed Transformers source: {error}")
    require(modeling_digest == MODELING_SOURCE_SHA256 and
            configuration_digest == CONFIGURATION_SOURCE_SHA256,
            "environment", 3, "installed Qwen Transformers sources differ from the pinned commit")
    try:
        torch.set_num_threads(torch_threads)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cpu")
    except ReferenceFailure:
        raise
    except Exception as error:
        fail("environment", 3, f"cannot establish deterministic CPU execution: {error}")
    require(torch.get_num_threads() == torch_threads and
            torch.get_num_interop_threads() == 1 and
            torch.are_deterministic_algorithms_enabled(),
            "environment", 3, "deterministic Torch settings did not take effect")
    environment_record = {
        "pythonVersion": platform.python_version(),
        "torchVersion": torch.__version__,
        "transformersVersion": transformers.__version__,
        "numpyVersion": np.__version__,
        "transformersCommit": observed_transformers_commit,
        "transformersTree": observed_transformers_tree,
        "modelingSourceSHA256": modeling_digest,
        "configurationSourceSHA256": configuration_digest,
        "platform": platform.system(),
        "processor": platform.processor() or platform.machine(),
        "device": "cpu",
        "dtype": "float32",
        "attentionImplementation": "eager",
        "expertsImplementation": "eager",
        "offline": dict(OFFLINE_ENVIRONMENT),
        "deterministicAlgorithms": True,
        "torchThreads": torch_threads,
        "torchInteropThreads": 1,
    }
    return ReferenceModules(
        np=np,
        torch=torch,
        functional=functional,
        dynamic_cache=DynamicCache,
        decoder_layer=Qwen3_5MoeDecoderLayer,
        rms_norm=Qwen3_5MoeRMSNorm,
        rotary_embedding=Qwen3_5MoeTextRotaryEmbedding,
        text_config=Qwen3_5MoeTextConfig,
        create_causal_mask=create_causal_mask,
        create_recurrent_attention_mask=create_recurrent_attention_mask,
        environment_record=environment_record,
    )


class ReadOnlyMapping:
    def __init__(self, record: FileRecord):
        self.record = record
        self.value: mmap.mmap | None = None

    def __enter__(self) -> mmap.mmap:
        try:
            current = os.fstat(self.record.descriptor)
            require(current.st_size == self.record.size, "identityOrFormat", 5,
                    f"payload size changed after validation: {self.record.relative_path}")
            self.value = mmap.mmap(self.record.descriptor, 0, access=mmap.ACCESS_READ)
            return self.value
        except ReferenceFailure:
            raise
        except (OSError, ValueError) as error:
            fail("resource", 6,
                 f"cannot map {self.record.relative_path} read-only: {error}")

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        if self.value is not None:
            self.value.close()
            self.value = None


class TensorDecoder:
    def __init__(self, modules: ReferenceModules, pack: ValidatedPack):
        self.modules = modules
        self.pack = pack
        self.weights_mapping_owner = ReadOnlyMapping(pack.files["model_weights.bin"])
        self.weights_mapping: mmap.mmap | None = None

    def __enter__(self) -> "TensorDecoder":
        self.weights_mapping = self.weights_mapping_owner.__enter__()
        return self

    def __exit__(self, kind: Any, value: Any, traceback: Any) -> None:
        self.weights_mapping_owner.__exit__(kind, value, traceback)
        self.weights_mapping = None

    def resident(self, name: str) -> TensorRegion:
        try:
            return self.pack.regions[name]
        except KeyError:
            fail("identityOrFormat", 5, f"missing validated resident tensor {name}")

    def fill_resident(self, destination: Any, name: str) -> None:
        require(self.weights_mapping is not None, "execution", 7,
                "resident mapping is not open")
        region = self.resident(name)
        expected_shape = tuple(int(value) for value in destination.shape)
        require(expected_shape == region.shape, "execution", 7,
                f"destination shape mismatch for {name}: {expected_shape} != {region.shape}")
        if region.storage == "bf16":
            self._fill_unquantized(destination, self.weights_mapping, region.offset, "bf16")
        elif region.storage == "fp32":
            self._fill_unquantized(destination, self.weights_mapping, region.offset, "fp32")
        elif region.storage in ("affineInt4", "affineInt8"):
            bit_width = 4 if region.storage == "affineInt4" else 8
            values_size, metadata_size, _ = affine_component_sizes(region.shape, bit_width)
            self._fill_affine_rows(
                destination, self.weights_mapping, region.shape, bit_width,
                region.offset,
                region.offset + values_size,
                region.offset + values_size + metadata_size,
                0,
            )
        else:
            fail("identityOrFormat", 5, f"unsupported storage for {name}: {region.storage}")

    def embedding_rows(self, token_ids: tuple[int, ...]) -> Any:
        torch = self.modules.torch
        region = self.resident("model.language_model.embed_tokens.weight")
        require(self.weights_mapping is not None, "execution", 7,
                "resident mapping is not open")
        values_size, metadata_size, _ = affine_component_sizes(region.shape, 8)
        result = torch.empty((len(token_ids), HIDDEN_SIZE), dtype=torch.float32, device="cpu")
        cached: dict[int, Any] = {}
        for position, token_id in enumerate(token_ids):
            if token_id not in cached:
                row = torch.empty((1, HIDDEN_SIZE), dtype=torch.float32, device="cpu")
                self._fill_affine_rows(
                    row, self.weights_mapping, region.shape, 8,
                    region.offset,
                    region.offset + values_size,
                    region.offset + values_size + metadata_size,
                    token_id,
                )
                cached[token_id] = row
            result[position].copy_(cached[token_id][0])
        return result.unsqueeze(0)

    def output_logits(self, hidden: Any) -> Any:
        torch = self.modules.torch
        functional = self.modules.functional
        region = self.resident("lm_head.weight")
        require(self.weights_mapping is not None, "execution", 7,
                "resident mapping is not open")
        values_size, metadata_size, _ = affine_component_sizes(region.shape, 8)
        logits = torch.empty((VOCABULARY_SIZE,), dtype=torch.float32, device="cpu")
        for start in range(0, VOCABULARY_SIZE, HEAD_CHUNK_ROWS):
            count = min(HEAD_CHUNK_ROWS, VOCABULARY_SIZE - start)
            weight = torch.empty((count, HIDDEN_SIZE), dtype=torch.float32, device="cpu")
            self._fill_affine_rows(
                weight, self.weights_mapping, region.shape, 8,
                region.offset,
                region.offset + values_size,
                region.offset + values_size + metadata_size,
                start,
            )
            logits[start:start + count].copy_(functional.linear(hidden, weight).reshape(-1))
            del weight
        require(bool(torch.isfinite(logits).all().item()), "execution", 7,
                "output logits contain a non-finite value")
        return logits

    def fill_experts(self, layer_module: Any, layer_layout: ExpertLayer) -> None:
        torch = self.modules.torch
        parameters = dict(layer_module.named_parameters())
        gate_up = parameters.get("mlp.experts.gate_up_proj")
        down = parameters.get("mlp.experts.down_proj")
        require(gate_up is not None and down is not None, "execution", 7,
                f"official layer {layer_layout.layer} routed parameters are missing")
        with ReadOnlyMapping(self.pack.files[layer_layout.path]) as expert_mapping:
            for expert_id in range(EXPERT_COUNT):
                base = expert_id * layer_layout.stride
                for role, destination in (("gate_up", gate_up[expert_id]),
                                          ("down", down[expert_id])):
                    source = layer_layout.sources[role]
                    self._fill_affine_rows(
                        destination, expert_mapping, source.shape, 4,
                        base + source.values_offset,
                        base + source.scales_offset,
                        base + source.biases_offset,
                        0,
                    )

    def _fill_unquantized(self, destination: Any, mapping: mmap.mmap,
                          offset: int, storage: str) -> None:
        np = self.modules.np
        torch = self.modules.torch
        flat = destination.reshape(-1)
        item_bytes = 2 if storage == "bf16" else 4
        block_values = max(1, MAX_TRANSIENT_DECODE_BYTES // 8)
        for start in range(0, flat.numel(), block_values):
            count = min(block_values, flat.numel() - start)
            if storage == "bf16":
                bits = np.frombuffer(mapping, dtype="<u2", count=count,
                                     offset=offset + start * item_bytes).astype(np.uint32)
                values = np.left_shift(bits, 16).view(np.float32)
            else:
                values = np.frombuffer(mapping, dtype="<f4", count=count,
                                       offset=offset + start * item_bytes).copy()
            require(bool(np.isfinite(values).all()), "identityOrFormat", 5,
                    "decoded unquantized tensor contains a non-finite value")
            flat[start:start + count].copy_(torch.from_numpy(values))

    def _fill_affine_rows(self, destination: Any, mapping: mmap.mmap,
                          full_shape: tuple[int, ...], bit_width: int,
                          values_base: int, scales_base: int, biases_base: int,
                          source_row_start: int) -> None:
        np = self.modules.np
        torch = self.modules.torch
        columns = full_shape[-1]
        total_rows = product(full_shape) // columns
        destination_rows = destination.numel() // columns
        require(destination.numel() == destination_rows * columns and
                source_row_start >= 0 and source_row_start + destination_rows <= total_rows,
                "resource", 6, "affine destination row range is invalid")
        values_per_row = (columns * bit_width + 7) // 8
        groups_per_row = (columns + GROUP_SIZE - 1) // GROUP_SIZE
        estimated_row_scratch = max(1, columns * 10 + values_per_row + groups_per_row * 8)
        rows_per_block = max(1, MAX_TRANSIENT_DECODE_BYTES // estimated_row_scratch)
        destination_2d = destination.reshape(destination_rows, columns)
        for local_start in range(0, destination_rows, rows_per_block):
            row_count = min(rows_per_block, destination_rows - local_start)
            source_start = source_row_start + local_start
            packed = np.frombuffer(
                mapping, dtype=np.uint8, count=row_count * values_per_row,
                offset=values_base + source_start * values_per_row,
            ).reshape(row_count, values_per_row)
            if bit_width == 8:
                codes = packed[:, :columns]
            else:
                codes = np.empty((row_count, columns), dtype=np.uint8)
                codes[:, 0::2] = packed[:, :(columns + 1) // 2] & 0x0F
                codes[:, 1::2] = packed[:, :columns // 2] >> 4
            scale_bits = np.frombuffer(
                mapping, dtype="<u2", count=row_count * groups_per_row,
                offset=scales_base + source_start * groups_per_row * 2,
            ).astype(np.uint32)
            bias_bits = np.frombuffer(
                mapping, dtype="<u2", count=row_count * groups_per_row,
                offset=biases_base + source_start * groups_per_row * 2,
            ).astype(np.uint32)
            scales = np.left_shift(scale_bits, 16).view(np.float32).reshape(
                row_count, groups_per_row)
            biases = np.left_shift(bias_bits, 16).view(np.float32).reshape(
                row_count, groups_per_row)
            require(bool(np.isfinite(scales).all()) and bool((scales > 0).all()) and
                    bool(np.isfinite(biases).all()),
                    "identityOrFormat", 5, "affine metadata is non-finite or non-positive")
            decoded = np.empty((row_count, columns), dtype=np.float32)
            for group in range(groups_per_row):
                first = group * GROUP_SIZE
                last = min(columns, first + GROUP_SIZE)
                decoded[:, first:last] = (
                    codes[:, first:last].astype(np.float32)
                    * scales[:, group:group + 1]
                    + biases[:, group:group + 1]
                )
            require(bool(np.isfinite(decoded).all()), "identityOrFormat", 5,
                    "decoded affine tensor contains a non-finite value")
            destination_2d[local_start:local_start + row_count].copy_(
                torch.from_numpy(decoded))


class LayerStreamingReference:
    def __init__(self, modules: ReferenceModules, pack: ValidatedPack,
                 config_document: dict[str, Any]):
        self.modules = modules
        self.pack = pack
        try:
            self.config = modules.text_config(**dict(config_document["text_config"]))
            self.config._attn_implementation = "eager"
            self.config._experts_implementation = "eager"
            self.config.use_cache = True
            self.rotary = modules.rotary_embedding(self.config, device="cpu")
            self.rotary.eval()
        except ReferenceFailure:
            raise
        except Exception as error:
            fail("environment", 3,
                 f"cannot construct pinned Qwen text configuration: {type(error).__name__}: {error}")
        require(list(self.config.layer_types) == [
            "full_attention" if value == "fullAttention" else "linear_attention"
            for value in EXPECTED_LAYER_TYPES
        ], "identityOrFormat", 5, "Transformers layer schedule differs from manifest")
        self.maximum_live_decoded_layer_bytes = 0

    def run(self, cases: tuple[RequestCase, ...]) -> tuple[list[dict[str, Any]], int]:
        results: list[dict[str, Any]] = []
        with TensorDecoder(self.modules, self.pack) as decoder:
            for case in cases:
                results.append(self._run_case(decoder, case))
        return results, self.maximum_live_decoded_layer_bytes

    def _run_case(self, decoder: TensorDecoder, case: RequestCase) -> dict[str, Any]:
        try:
            cache = self.modules.dynamic_cache(config=self.config)
        except ReferenceFailure:
            raise
        except Exception as error:
            fail("execution", 7,
                 f"case {case.identifier} cache construction failed: {type(error).__name__}: {error}")
        generated: list[int] = []
        steps: list[dict[str, Any]] = []
        input_ids = case.prompt_token_ids
        expected_sequence_length = 0
        stop_reason = "maxTokens"
        for step_index in range(case.max_new_tokens):
            logits, sequence_length = self._forward(decoder, input_ids, cache)
            expected_sequence_length += len(input_ids)
            require(sequence_length == expected_sequence_length, "execution", 7,
                    f"case {case.identifier} cache length {sequence_length} != {expected_sequence_length}")
            logits_record, greedy, top_k = self._serialize_logits(logits)
            generated.append(greedy)
            steps.append({
                "step": step_index,
                "inputTokenIDs": list(input_ids),
                "sequenceLengthAfterForward": sequence_length,
                "logits": logits_record,
                "greedyTokenID": greedy,
                "topK": top_k,
            })
            del logits
            if greedy == EOS_TOKEN_ID:
                stop_reason = "eos"
                break
            input_ids = (greedy,)
        digest_input = b"".join(struct.pack("<i", token_id)
                                 for token_id in case.prompt_token_ids)
        return {
            "id": case.identifier,
            "promptTokenIDs": list(case.prompt_token_ids),
            "promptTokenIDSHA256": sha256_bytes(digest_input),
            "maxNewTokens": case.max_new_tokens,
            "generatedTokenIDs": generated,
            "stopReason": stop_reason,
            "steps": steps,
        }

    def _forward(self, decoder: TensorDecoder, input_ids: tuple[int, ...],
                 cache: Any) -> tuple[Any, int]:
        torch = self.modules.torch
        try:
            with torch.inference_mode():
                hidden_states = decoder.embedding_rows(input_ids)
                past_seen = int(cache.get_seq_length())
                positions = torch.arange(
                    hidden_states.shape[1], dtype=torch.long, device="cpu") + past_seen
                position_ids = positions.view(1, 1, -1).expand(
                    4, hidden_states.shape[0], -1)
                text_position_ids = position_ids[0]
                mrope_position_ids = position_ids[1:]
                position_embeddings = self.rotary(hidden_states, mrope_position_ids)
                full_attention_mask = self.modules.create_causal_mask(
                    config=self.config,
                    inputs_embeds=hidden_states,
                    attention_mask=None,
                    past_key_values=cache,
                    position_ids=text_position_ids,
                )
                linear_attention_mask = self.modules.create_recurrent_attention_mask(
                    config=self.config,
                    inputs_embeds=hidden_states,
                    attention_mask=None,
                    past_key_values=cache,
                )
                for layer_index in range(LAYER_COUNT):
                    layer = self._materialize_layer(decoder, layer_index)
                    attention_mask = full_attention_mask if layer_index % 4 == 3 \
                        else linear_attention_mask
                    hidden_states = layer(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        position_ids=text_position_ids,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    require(bool(torch.isfinite(hidden_states).all().item()),
                            "execution", 7,
                            f"non-finite hidden state after layer {layer_index}")
                    del layer
                    gc.collect()
                final_norm = self.modules.rms_norm(
                    HIDDEN_SIZE, eps=float(self.config.rms_norm_eps))
                final_norm.requires_grad_(False)
                decoder.fill_resident(
                    dict(final_norm.named_parameters())["weight"],
                    "model.language_model.norm.weight",
                )
                hidden_states = final_norm(hidden_states)
                require(bool(torch.isfinite(hidden_states).all().item()),
                        "execution", 7, "non-finite final normalized hidden state")
                final_row = hidden_states[:, -1, :].contiguous()
                logits = decoder.output_logits(final_row)
                sequence_length = int(cache.get_seq_length())
                del final_norm, hidden_states, final_row
                return logits, sequence_length
        except ReferenceFailure:
            raise
        except KeyboardInterrupt:
            raise ReferenceCancelled("keyboard interrupt during model execution")
        except Exception as error:
            fail("execution", 7,
                 f"official layer execution failed: {type(error).__name__}: {error}")

    def _materialize_layer(self, decoder: TensorDecoder, layer_index: int) -> Any:
        torch = self.modules.torch
        try:
            with torch.device("meta"):
                layer = self.modules.decoder_layer(self.config, layer_index)
            layer.to_empty(device="cpu")
            layer.requires_grad_(False)
            layer.eval()
            parameters = dict(layer.named_parameters())
            expected_local_names = {
                name.removeprefix(f"model.language_model.layers.{layer_index}.")
                for name in decoder.pack.regions
                if name.startswith(f"model.language_model.layers.{layer_index}.")
            }
            expected_local_names.update({
                "mlp.experts.gate_up_proj", "mlp.experts.down_proj",
            })
            require(set(parameters) == expected_local_names, "identityOrFormat", 5,
                    f"official parameter set differs from pack mapping in layer {layer_index}")
            prefix = f"model.language_model.layers.{layer_index}."
            for local_name, parameter in parameters.items():
                require(parameter.device.type == "cpu" and parameter.dtype == torch.float32,
                        "environment", 3,
                        f"layer {layer_index} parameter {local_name} is not CPU Float32")
                if local_name.startswith("mlp.experts."):
                    continue
                decoder.fill_resident(parameter, prefix + local_name)
            decoder.fill_experts(layer, decoder.pack.expert_layers[layer_index])
            live_bytes = sum(parameter.numel() * parameter.element_size()
                             for parameter in parameters.values())
            self.maximum_live_decoded_layer_bytes = max(
                self.maximum_live_decoded_layer_bytes, live_bytes)
            require(live_bytes < 4 * 1024 * 1024 * 1024,
                    "resource", 6,
                    f"decoded layer {layer_index} exceeds the 4 GiB live-weight bound")
            return layer
        except ReferenceFailure:
            raise
        except Exception as error:
            fail("execution", 7,
                 f"layer {layer_index} materialization failed: {type(error).__name__}: {error}")

    def _serialize_logits(self, logits: Any) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
        np = self.modules.np
        values = logits.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        require(values.shape == (VOCABULARY_SIZE,) and bool(np.isfinite(values).all()),
                "execution", 7, "logit vector shape or finiteness is invalid")
        greedy = int(np.argmax(values))
        token_ids = np.arange(VOCABULARY_SIZE, dtype=np.int64)
        top_indices = np.lexsort((token_ids, -values.astype(np.float64)))[:5]
        top_k = [
            {"tokenID": int(index), "logit": float(values[index])}
            for index in top_indices
        ]
        raw = values.tobytes(order="C")
        require(len(raw) == VOCABULARY_SIZE * 4, "execution", 7,
                "serialized logit byte count is invalid")
        return ({
            "dtype": "float32-le",
            "encoding": "base64",
            "count": VOCABULARY_SIZE,
            "sha256": sha256_bytes(raw),
            "bytes": base64.b64encode(raw).decode("ascii"),
        }, greedy, top_k)


def revalidate_payloads(pack: ValidatedPack) -> None:
    for relative in sorted(pack.files):
        record = pack.files[relative]
        try:
            info = os.fstat(record.descriptor)
            require(info.st_size == record.size, "identityOrFormat", 5,
                    f"payload size changed during execution: {relative}")
            digest = hashlib.sha256()
            offset = 0
            while offset < record.size:
                chunk = os.pread(record.descriptor,
                                 min(8 * 1024 * 1024, record.size - offset), offset)
                require(bool(chunk), "identityOrFormat", 5,
                        f"short read while revalidating {relative}")
                digest.update(chunk)
                offset += len(chunk)
            require(digest.hexdigest() == record.sha256, "identityOrFormat", 5,
                    f"payload changed during execution: {relative}")
        except ReferenceFailure:
            raise
        except OSError as error:
            fail("identityOrFormat", 5,
                 f"cannot revalidate payload {relative}: {error}")


def identity_record(pack: ValidatedPack) -> dict[str, Any]:
    return {
        "manifestSHA256": pack.manifest_sha256,
        "quantizationPolicySHA256": pack.policy_sha256,
        "modelID": MODEL_ID,
        "sourceRevision": pack.source_revision,
        "sourceIndexSHA256": pack.source_index_sha256,
        "configSHA256": CONFIG_SHA256,
        "files": [
            {"path": relative, "size": pack.files[relative].size,
             "sha256": pack.files[relative].sha256}
            for relative in sorted(pack.files)
        ],
    }


def write_atomic_json(output: Path, value: dict[str, Any]) -> None:
    temporary: Path | None = None
    output_identity: tuple[int, int] | None = None
    output_created = False
    directory_descriptor: int | None = None
    cancellation_mask = {signal.SIGINT, signal.SIGTERM}
    previous_mask: set[signal.Signals] | None = None
    try:
        encoded = (json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ) + "\n").encode("utf-8")
        require(len(encoded) <= MAX_OUTPUT_BYTES, "resource", 6,
                f"result JSON exceeds {MAX_OUTPUT_BYTES} bytes")
        descriptor, name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            temporary_stat = temporary.stat(follow_symlinks=False)
            directory_descriptor = os.open(output.parent, os.O_RDONLY)
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, cancellation_mask)
            try:
                os.link(temporary, output)
                output_created = True
                output_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
                output_stat = output.stat(follow_symlinks=False)
                require((output_stat.st_dev, output_stat.st_ino) == output_identity,
                        "outputWrite", 9,
                        "published output is not the owned temporary inode")
                os.fsync(directory_descriptor)
                temporary.unlink()
                temporary = None
                os.fsync(directory_descriptor)
                os.close(directory_descriptor)
                directory_descriptor = None
                mask_to_restore = previous_mask
                previous_mask = None
                signal.pthread_sigmask(signal.SIG_SETMASK, mask_to_restore)
            except BaseException as publication_error:
                cleanup_error: BaseException | None = None
                try:
                    if previous_mask is None:
                        previous_mask = signal.pthread_sigmask(
                            signal.SIG_BLOCK, cancellation_mask)
                    if output_created and output_identity is not None:
                        current = output.stat(follow_symlinks=False)
                        require((current.st_dev, current.st_ino) == output_identity,
                                "outputWrite", 9,
                                "refusing to remove a substituted output after failure")
                        output.unlink()
                        output_created = False
                        output_identity = None
                        cleanup_descriptor = os.open(output.parent, os.O_RDONLY)
                        try:
                            os.fsync(cleanup_descriptor)
                        finally:
                            os.close(cleanup_descriptor)
                except BaseException as error:
                    cleanup_error = error
                if cleanup_error is not None:
                    fail("outputWrite", 9,
                         "publication failed and owned output rollback failed: "
                         f"{type(cleanup_error).__name__}: {cleanup_error}")
                raise publication_error
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
    except ReferenceFailure:
        raise
    except (OSError, TypeError, ValueError) as error:
        fail("outputWrite", 9, f"cannot atomically write result: {error}")
    finally:
        if previous_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        if directory_descriptor is not None:
            try:
                os.close(directory_descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def parser() -> StrictArgumentParser:
    value = StrictArgumentParser(
        prog="qwen36_quantized_reference.py",
        description="Offline same-pack Qwen v2 quantized reference",
        allow_abbrev=False,
    )
    value.add_argument("--model-directory", required=True, type=Path)
    value.add_argument("--official-config", required=True, type=Path)
    value.add_argument("--transformers-checkout", required=True, type=Path)
    value.add_argument("--request", required=True, type=Path)
    value.add_argument("--expected-manifest-sha256", required=True)
    value.add_argument("--expected-policy-sha256", required=True)
    value.add_argument("--torch-threads", required=True, type=int)
    value.add_argument("--output", required=True, type=Path)
    return value


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    if "--help" in arguments or "-h" in arguments:
        return parser().parse_args(arguments)
    option_names = (
        "--model-directory", "--official-config", "--transformers-checkout", "--request",
        "--expected-manifest-sha256", "--expected-policy-sha256",
        "--torch-threads", "--output",
    )
    for option in option_names:
        require(arguments.count(option) == 1, "usage", 2,
                f"{option} must appear exactly once")
    parsed = parser().parse_args(arguments)
    for option, value in (
        ("--expected-manifest-sha256", parsed.expected_manifest_sha256),
        ("--expected-policy-sha256", parsed.expected_policy_sha256),
    ):
        require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
                "usage", 2,
                f"{option} must be 64 lowercase hexadecimal characters")
    return parsed


def install_cancellation_handlers() -> None:
    def cancel(signum: int, _frame: Any) -> NoReturn:
        raise ReferenceCancelled(f"received signal {signum}")

    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)


def execute(arguments: list[str]) -> None:
    parsed = parse_arguments(arguments)
    install_cancellation_handlers()
    model, official_config, transformers_checkout, request_path, output = validate_paths(
        parsed.model_directory, parsed.official_config,
        parsed.transformers_checkout, parsed.request, parsed.output)
    cases = validate_request(request_path)
    modules = validate_environment(parsed.torch_threads, transformers_checkout)
    initial_script_sha256 = sha256_bytes(Path(__file__).resolve().read_bytes())
    pack: ValidatedPack | None = None
    try:
        pack, config_document = validate_manifest(
            model, official_config,
            parsed.expected_manifest_sha256,
            parsed.expected_policy_sha256,
        )
        reference = LayerStreamingReference(modules, pack, config_document)
        case_results, maximum_live_layer_bytes = reference.run(cases)
        require(maximum_live_layer_bytes > 0, "execution", 7,
                "reference executed no decoder layer")
        revalidate_payloads(pack)
        final_script_sha256 = sha256_bytes(Path(__file__).resolve().read_bytes())
        require(final_script_sha256 == initial_script_sha256,
                "identityOrFormat", 5, "reference script changed during execution")
        require(not any(name.startswith("TurboFieldfare") for name in sys.modules),
                "environment", 3, "TurboFieldfare runtime module was imported")
        result = {
            "schemaVersion": RESULT_SCHEMA,
            "status": "complete",
            "identity": identity_record(pack),
            "referenceEnvironment": modules.environment_record,
            "reader": {
                "schema": READER_SCHEMA,
                "scriptSHA256": final_script_sha256,
                "maximumTransientDecodeBytes": MAX_TRANSIENT_DECODE_BYTES,
                "maximumLiveDecodedLayerBytes": maximum_live_layer_bytes,
                "runtimeModulesImported": False,
                "bf16CheckpointRead": False,
            },
            "tolerances": TOLERANCES,
            "cases": case_results,
        }
        write_atomic_json(output, result)
    finally:
        if pack is not None:
            pack.close()


def entrypoint() -> int:
    try:
        execute(sys.argv[1:])
        return 0
    except ReferenceFailure as error:
        sys.stderr.write(
            f"qwen36_quantized_reference: {error.category}: {error.detail}\n")
        sys.stderr.flush()
        return error.exit_code
    except KeyboardInterrupt:
        sys.stderr.write(
            "qwen36_quantized_reference: cancelled: keyboard interrupt\n")
        sys.stderr.flush()
        return 8
    except SystemExit:
        raise
    except BaseException as error:
        sys.stderr.write(
            "qwen36_quantized_reference: execution: "
            f"{type(error).__name__}: {bounded_detail(error)}\n")
        sys.stderr.flush()
        return 7


if __name__ == "__main__":
    raise SystemExit(entrypoint())
