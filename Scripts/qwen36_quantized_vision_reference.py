#!/usr/bin/env python3
"""Independent offline Qwen 3.6 vision reference over verified v2 packs.

Imports are deliberately standard-library only.  Heavy modules and the frozen
Phase 22 text oracle are loaded only after their paths, bytes, and environment
have been validated by ``execute``.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import gc
import hashlib
import importlib.util
import json
import mmap
import os
import re
import shutil
import signal
import stat
import struct
import sys
import types
from pathlib import Path
from typing import Any, NoReturn


REQUEST_SCHEMA = "qwen36-quantized-vision-reference-request-v1"
RESULT_SCHEMA = "qwen36-quantized-vision-reference-v1"
READER_SCHEMA = "gturbo-v2-qwen-vision-independent-v1"
TEXT_HELPER_SHA256 = "9b2463fd8935a17c049b3399f9ced52b62c554defaf3038894f6c21e4815a50d"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
SOURCE_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
SOURCE_INDEX_SHA256 = "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
PROCESSOR_SHA256 = "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
CONVERTER_VERSION = "TurboFieldfareRepack/QwenBF16Affine/v2"
ALIGNMENT = 16_384
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_REQUEST_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_PROMPT_BYTES = 64 * 1024
MAX_PATCH_ROWS = 2_520
MAX_MERGED_ROWS = 630
VISION_DECODED_WEIGHT_BYTES = 1_786_284_992
VISION_START_TOKEN_ID = 248_053
VISION_END_TOKEN_ID = 248_054
IMAGE_PAD_TOKEN_ID = 248_056
EOS_TOKEN_ID = 248_044
HIDDEN_SIZE = 2_048
LAYER_COUNT = 40
VOCABULARY_SIZE = 248_320
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CASE_ID_RE = re.compile(r"^[\x21-\x7e]{1,64}$")
IMAGE_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
OFFLINE_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "USE_HUB_KERNELS": "0",
}
NETWORK_AUDIT_EVENTS = frozenset({
    "socket.bind", "socket.connect", "socket.connect_ex", "socket.getaddrinfo",
    "socket.gethostbyaddr", "socket.gethostbyname",
})


class VisionReferenceFailure(Exception):
    def __init__(self, category: str, exit_code: int, detail: str):
        super().__init__(detail)
        self.category = category
        self.exit_code = exit_code
        compact = " ".join(str(detail).replace("\x00", "\\0").splitlines())
        self.detail = compact[:1_024]


class VisionReferenceCancelled(VisionReferenceFailure):
    def __init__(self, detail: str):
        super().__init__("cancelled", 8, detail)


class StrictArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise VisionReferenceFailure("usage", 2, message)


@dataclasses.dataclass(frozen=True)
class VisionRequestCase:
    identifier: str
    image: Path
    image_sha256: str
    image_size: int
    image_descriptor: int
    prompt: str
    max_new_tokens: int


@dataclasses.dataclass(frozen=True)
class VisionTensorContract:
    name: str
    shape: tuple[int, ...]
    storage: str = "bf16"

    @property
    def size(self) -> int:
        total = 2
        for extent in self.shape:
            total *= extent
        return total


@dataclasses.dataclass(frozen=True)
class OpenFile:
    relative_path: str
    path: Path
    size: int
    sha256: str
    descriptor: int


@dataclasses.dataclass(frozen=True)
class ValidatedRequestDocument:
    cases: tuple[VisionRequestCase, ...]
    file: OpenFile


@dataclasses.dataclass(frozen=True)
class VisionRegion:
    name: str
    offset: int
    size: int
    shape: tuple[int, ...]


@dataclasses.dataclass
class ValidatedVisionPack:
    root: Path
    manifest_sha256: str
    receipt_sha256: str
    compatible_text_manifest_sha256: str
    files: dict[str, OpenFile]
    regions: tuple[VisionRegion, ...]
    receipt: dict[str, Any]

    def close(self) -> None:
        for record in self.files.values():
            try:
                os.close(record.descriptor)
            except OSError:
                pass


def fail(category: str, exit_code: int, detail: str) -> NoReturn:
    raise VisionReferenceFailure(category, exit_code, detail)


def require(condition: bool, category: str, exit_code: int, detail: str) -> None:
    if not condition:
        fail(category, exit_code, detail)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def exact_keys(value: Any, keys: set[str], label: str,
               category: str = "identityOrFormat", exit_code: int = 5) -> dict[str, Any]:
    require(isinstance(value, dict), category, exit_code, f"{label} must be an object")
    require(set(value) == keys, category, exit_code,
            f"{label} keys differ: expected={sorted(keys)} observed={sorted(value)}")
    return value


def decode_json(data: bytes, label: str, category: str, exit_code: int) -> Any:
    try:
        text = data.decode("utf-8")
        return json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value {value}")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        fail(category, exit_code, f"invalid {label} JSON: {error}")


def read_bounded(path: Path, maximum: int, label: str,
                 category: str, exit_code: int) -> bytes:
    data, record = open_bounded_regular(
        path, maximum, label, category, exit_code)
    try:
        return data
    finally:
        try:
            os.close(record.descriptor)
        except OSError:
            pass


def open_bounded_regular(path: Path, maximum: int, label: str,
                         category: str, exit_code: int) -> tuple[bytes, OpenFile]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), category, exit_code,
                f"{label} must be a regular non-link file")
        require(0 <= info.st_size <= maximum, category, exit_code,
                f"{label} exceeds {maximum} bytes")
        output = bytearray()
        digest = hashlib.sha256()
        offset = 0
        while offset < info.st_size:
            chunk = os.pread(descriptor, min(1024 * 1024, info.st_size - offset), offset)
            require(bool(chunk), category, exit_code, f"short read while reading {label}")
            output.extend(chunk)
            digest.update(chunk)
            offset += len(chunk)
        return bytes(output), OpenFile(
            path.name, path, info.st_size, digest.hexdigest(), descriptor)
    except VisionReferenceFailure:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        fail(category, exit_code, f"cannot read {label}: {error}")


def canonical_existing(path: Path, label: str, *, directory: bool = False) -> Path:
    require(path.is_absolute(), "request", 4, f"{label} must be absolute")
    require(not path.is_symlink(), "request", 4, f"{label} must not be a symbolic link")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        fail("request", 4, f"cannot resolve {label}: {error}")
    require(resolved == path, "request", 4, f"{label} must be a canonical physical path")
    require(resolved.is_dir() if directory else resolved.is_file(), "request", 4,
            f"{label} has the wrong type")
    return resolved


def open_hashed_regular(path: Path, expected_sha256: str, label: str,
                        category: str = "identityOrFormat", exit_code: int = 5,
                        expected_size: int | None = None) -> OpenFile:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), category, exit_code,
                f"{label} is not a regular file")
        if expected_size is not None:
            require(info.st_size == expected_size, category, exit_code,
                    f"{label} size mismatch: {info.st_size} != {expected_size}")
        digest = hashlib.sha256()
        offset = 0
        while offset < info.st_size:
            chunk = os.pread(descriptor, min(8 * 1024 * 1024, info.st_size - offset), offset)
            require(bool(chunk), category, exit_code, f"short read while hashing {label}")
            digest.update(chunk)
            offset += len(chunk)
        observed = digest.hexdigest()
        require(observed == expected_sha256, category, exit_code,
                f"{label} SHA-256 mismatch")
        return OpenFile(path.name, path, info.st_size, observed, descriptor)
    except VisionReferenceFailure:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        fail(category, exit_code, f"cannot verify {label}: {error}")


def _parser() -> StrictArgumentParser:
    value = StrictArgumentParser(
        prog="qwen36_quantized_vision_reference.py",
        description="Offline same-pack Qwen v2 quantized vision reference",
        allow_abbrev=False,
    )
    value.add_argument("--text-model-directory", required=True, type=Path)
    value.add_argument("--vision-model-directory", required=True, type=Path)
    value.add_argument("--official-source-directory", required=True, type=Path)
    value.add_argument("--transformers-checkout", required=True, type=Path)
    value.add_argument("--text-reference-script", required=True, type=Path)
    value.add_argument("--request", required=True, type=Path)
    value.add_argument("--expected-text-manifest-sha256", required=True)
    value.add_argument("--expected-vision-manifest-sha256", required=True)
    value.add_argument("--expected-policy-sha256", required=True)
    value.add_argument("--torch-threads", required=True, type=int)
    value.add_argument("--output", required=True, type=Path)
    return value


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    if "--help" in arguments or "-h" in arguments:
        return _parser().parse_args(arguments)
    options = (
        "--text-model-directory", "--vision-model-directory",
        "--official-source-directory", "--transformers-checkout",
        "--text-reference-script", "--request",
        "--expected-text-manifest-sha256", "--expected-vision-manifest-sha256",
        "--expected-policy-sha256", "--torch-threads", "--output",
    )
    for option in options:
        require(arguments.count(option) == 1, "usage", 2,
                f"{option} must appear exactly once")
    parsed = _parser().parse_args(arguments)
    for option, value in (
        ("--expected-text-manifest-sha256", parsed.expected_text_manifest_sha256),
        ("--expected-vision-manifest-sha256", parsed.expected_vision_manifest_sha256),
        ("--expected-policy-sha256", parsed.expected_policy_sha256),
    ):
        require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
                "usage", 2, f"{option} must be 64 lowercase hexadecimal characters")
    require(is_int(parsed.torch_threads) and 1 <= parsed.torch_threads <= 12,
            "usage", 2, "--torch-threads must be in 1...12")
    return parsed


def _validate_request_document(path: Path) -> ValidatedRequestDocument:
    request_path = canonical_existing(path, "request")
    request_data, request_file = open_bounded_regular(
        request_path, MAX_REQUEST_BYTES, "request", "request", 4)
    opened_descriptors: list[int] = []
    try:
        document = decode_json(request_data, "request", "request", 4)
        document = exact_keys(document, {"schemaVersion", "cases"}, "request", "request", 4)
        require(document["schemaVersion"] == REQUEST_SCHEMA, "request", 4,
                f"unsupported request schema {document['schemaVersion']!r}")
        raw_cases = document["cases"]
        require(isinstance(raw_cases, list) and 1 <= len(raw_cases) <= 2,
                "request", 4, "request must contain one or two cases")
        identifiers: set[str] = set()
        images: set[Path] = set()
        cases: list[VisionRequestCase] = []
        for index, raw in enumerate(raw_cases):
            item = exact_keys(raw, {"id", "image", "imageSHA256", "prompt", "maxNewTokens"},
                              f"request.cases[{index}]", "request", 4)
            identifier = item["id"]
            require(isinstance(identifier, str) and CASE_ID_RE.fullmatch(identifier) is not None,
                    "request", 4, f"case {index} id is not bounded printable ASCII")
            require(identifier not in identifiers, "request", 4,
                    f"duplicate case id {identifier!r}")
            identifiers.add(identifier)
            require(isinstance(item["image"], str), "request", 4,
                    f"case {identifier} image must be a path string")
            image = canonical_existing(Path(item["image"]), f"case {identifier} image")
            require(image not in images, "request", 4,
                    f"duplicate physical image path {image}")
            images.add(image)
            expected_digest = item["imageSHA256"]
            require(isinstance(expected_digest, str) and SHA256_RE.fullmatch(expected_digest) is not None,
                    "request", 4, f"case {identifier} imageSHA256 is invalid")
            record = open_hashed_regular(image, expected_digest, f"case {identifier} image",
                                         "request", 4)
            opened_descriptors.append(record.descriptor)
            prompt = item["prompt"]
            require(isinstance(prompt, str), "request", 4,
                    f"case {identifier} prompt must be a string")
            try:
                prompt_bytes = prompt.encode("utf-8", "strict")
            except UnicodeEncodeError as error:
                fail("request", 4, f"case {identifier} prompt is invalid UTF-8: {error}")
            require(0 < len(prompt_bytes) <= MAX_PROMPT_BYTES, "request", 4,
                    f"case {identifier} prompt must contain 1...{MAX_PROMPT_BYTES} UTF-8 bytes")
            require(prompt.count(IMAGE_MARKER) == 1 and
                    prompt.count("<|vision_start|>") == 1 and
                    prompt.count("<|image_pad|>") == 1 and
                    prompt.count("<|vision_end|>") == 1,
                    "request", 4, f"case {identifier} prompt must contain one contiguous image marker")
            maximum = item["maxNewTokens"]
            require(is_int(maximum) and 1 <= maximum <= 2, "request", 4,
                    f"case {identifier} maxNewTokens must be 1 or 2")
            cases.append(VisionRequestCase(
                identifier, image, expected_digest, record.size, record.descriptor,
                prompt, maximum))
        return ValidatedRequestDocument(tuple(cases), request_file)
    except BaseException:
        for descriptor in opened_descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            os.close(request_file.descriptor)
        except OSError:
            pass
        raise


def validate_request(path: Path) -> tuple[VisionRequestCase, ...]:
    validated = _validate_request_document(path)
    try:
        return validated.cases
    finally:
        try:
            os.close(validated.file.descriptor)
        except OSError:
            pass


def expected_vision_tensor_contract() -> tuple[VisionTensorContract, ...]:
    hidden = 1_152
    intermediate = 4_304
    members = (
        ("attn.proj.weight", (hidden, hidden)),
        ("attn.proj.bias", (hidden,)),
        ("attn.qkv.weight", (3 * hidden, hidden)),
        ("attn.qkv.bias", (3 * hidden,)),
        ("mlp.linear_fc1.weight", (intermediate, hidden)),
        ("mlp.linear_fc1.bias", (intermediate,)),
        ("mlp.linear_fc2.weight", (hidden, intermediate)),
        ("mlp.linear_fc2.bias", (hidden,)),
        ("norm1.weight", (hidden,)),
        ("norm1.bias", (hidden,)),
        ("norm2.weight", (hidden,)),
        ("norm2.bias", (hidden,)),
    )
    patch = [
        VisionTensorContract("model.visual.patch_embed.proj.weight", (hidden, 3, 2, 16, 16)),
        VisionTensorContract("model.visual.patch_embed.proj.bias", (hidden,)),
    ]
    blocks: list[VisionTensorContract] = []
    for layer in range(27):
        for member, shape in members:
            blocks.append(VisionTensorContract(f"model.visual.blocks.{layer}.{member}", shape))
    merger = [
        VisionTensorContract("model.visual.merger.norm.weight", (hidden,)),
        VisionTensorContract("model.visual.merger.norm.bias", (hidden,)),
        VisionTensorContract("model.visual.merger.linear_fc1.weight", (4 * hidden, 4 * hidden)),
        VisionTensorContract("model.visual.merger.linear_fc1.bias", (4 * hidden,)),
        VisionTensorContract("model.visual.merger.linear_fc2.weight", (2_048, 4 * hidden)),
        VisionTensorContract("model.visual.merger.linear_fc2.bias", (2_048,)),
    ]
    ordered = (
        sorted(patch, key=lambda item: item.name)
        + [VisionTensorContract("model.visual.pos_embed.weight", (2_304, hidden))]
        + [item for layer in range(27) for item in sorted(
            (value for value in blocks if value.name.startswith(f"model.visual.blocks.{layer}.")),
            key=lambda value: value.name)]
        + sorted(merger, key=lambda item: item.name)
    )
    require(len(ordered) == 333, "execution", 7,
            f"internal vision tensor contract has {len(ordered)} entries")
    return tuple(ordered)


def float32_le_to_bf16_rne(data: bytes) -> bytes:
    require(len(data) % 4 == 0, "identityOrFormat", 5,
            "Float32 input length must be a multiple of four")
    output = bytearray(len(data) // 2)
    for index in range(0, len(data), 4):
        bits = struct.unpack_from("<I", data, index)[0]
        exponent = bits & 0x7F80_0000
        fraction = bits & 0x007F_FFFF
        require(exponent != 0x7F80_0000, "identityOrFormat", 5,
                "Float32 input contains a non-finite value")
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_FFFF
        struct.pack_into("<H", output, index // 2, (rounded >> 16) & 0xFFFF)
    return bytes(output)


def bf16_le_to_float32_le(data: bytes) -> bytes:
    require(len(data) % 2 == 0, "identityOrFormat", 5,
            "BF16 input length must be a multiple of two")
    output = bytearray(len(data) * 2)
    for index in range(0, len(data), 2):
        bits = struct.unpack_from("<H", data, index)[0]
        exponent = bits & 0x7F80
        fraction = bits & 0x007F
        require(exponent != 0x7F80 or fraction == 0, "identityOrFormat", 5,
                "BF16 input contains a NaN")
        require(exponent != 0x7F80, "identityOrFormat", 5,
                "BF16 input contains an infinity")
        struct.pack_into("<I", output, index * 2, bits << 16)
    return bytes(output)


def encode_result_json(value: dict[str, Any]) -> bytes:
    try:
        encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        fail("outputWrite", 9, f"cannot encode result JSON: {error}")
    require(len(encoded) <= MAX_OUTPUT_BYTES, "resource", 6,
            f"result JSON exceeds {MAX_OUTPUT_BYTES} bytes")
    return encoded


def write_atomic_result(output: Path, value: dict[str, Any], helper: Any) -> None:
    encode_result_json(value)
    require(callable(getattr(helper, "write_atomic_json", None)), "outputWrite", 9,
            "accepted helper has no atomic JSON writer")
    helper.write_atomic_json(output, value)


def _close_cases(cases: tuple[VisionRequestCase, ...]) -> None:
    for case in cases:
        try:
            os.close(case.image_descriptor)
        except OSError:
            pass


def _close_open_file(record: OpenFile | None) -> None:
    if record is None:
        return
    try:
        os.close(record.descriptor)
    except OSError:
        pass


def _terminal_atomic_commit(output: Path, value: dict[str, Any], helper: Any) -> None:
    cancellation_mask = {signal.SIGINT, signal.SIGTERM}
    try:
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, cancellation_mask)
        pending = signal.sigpending()
    except OSError as error:
        fail("outputWrite", 9, f"cannot establish terminal publication boundary: {error}")
    if pending.intersection(cancellation_mask):
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        raise VisionReferenceCancelled("cancellation was pending before output commit")
    # Signals that arrive from here onward belong to the terminal commit.  Keep
    # them blocked through process exit; the accepted helper performs its own
    # exclusive link/fsync transaction and rollback on publication failure.
    write_atomic_result(output, value, helper)


def _canonical_output(parsed: argparse.Namespace, inputs: tuple[Path, ...]) -> Path:
    output = parsed.output
    require(output.is_absolute() and output.name not in ("", ".", ".."), "request", 4,
            "output must be an absolute file path")
    try:
        parent = output.parent.resolve(strict=True)
    except OSError as error:
        fail("request", 4, f"output parent cannot be resolved: {error}")
    candidate = parent / output.name
    require(not os.path.lexists(candidate), "request", 4,
            "output path already exists and is preserved")
    try:
        free = shutil.disk_usage(parent).free
    except OSError as error:
        fail("resource", 6, f"cannot inspect output volume: {error}")
    require(free >= 512 * 1024 * 1024, "resource", 6,
            f"output volume has only {free} free bytes")
    for path in inputs:
        require(candidate != path, "request", 4, "output must differ from every input")
        if path.is_dir():
            try:
                candidate.relative_to(path)
                inside = True
            except ValueError:
                inside = False
            require(not inside, "request", 4, f"output must be outside {path}")
    repository_root = next((item for item in Path(__file__).resolve().parents
                            if (item / "Package.swift").is_file()), None)
    require(repository_root is not None, "environment", 3,
            "cannot identify repository root for output isolation")
    for relative in ("Sources", "Tests", "Scripts"):
        try:
            candidate.relative_to(repository_root / relative)
            inside = True
        except ValueError:
            inside = False
        require(not inside, "request", 4,
                f"output must be outside repository {relative}/")
    return candidate


def _install_offline_process_boundary() -> None:
    for name, expected in OFFLINE_ENVIRONMENT.items():
        require(os.environ.get(name) == expected, "environment", 3,
                f"{name} must equal {expected!r}")
    require("PYTHONPATH" not in os.environ, "environment", 3,
            "ambient PYTHONPATH must be absent")
    preloaded = sorted(name for name in sys.modules if name in {
        "numpy", "torch", "PIL", "transformers",
    } or name.startswith(("numpy.", "torch.", "PIL.", "transformers.")))
    require(not preloaded, "environment", 3,
            f"heavy runtime module was preloaded: {preloaded[:8]}")

    def reject_network(event: str, _arguments: tuple[Any, ...]) -> None:
        if event in NETWORK_AUDIT_EVENTS:
            fail("environment", 3,
                 f"Python network operation is forbidden in offline reference: {event}")

    try:
        sys.addaudithook(reject_network)
    except Exception as error:
        fail("environment", 3,
             f"cannot install Python offline network guard: {type(error).__name__}: {error}")


def _load_helper(path: Path) -> tuple[Any, OpenFile]:
    helper_path = canonical_existing(path, "text reference script")
    data, record = open_bounded_regular(
        helper_path, 4 * 1024 * 1024,
        "text reference script", "identityOrFormat", 5)
    module: Any = None
    try:
        require(record.sha256 == TEXT_HELPER_SHA256, "identityOrFormat", 5,
                "text reference script differs from the accepted Phase 22 helper")
        specification = importlib.util.spec_from_loader(
            "_qwen36_phase22_reference", loader=None, origin=str(helper_path))
        require(specification is not None, "environment", 3,
                "cannot create text helper import specification")
        module = importlib.util.module_from_spec(specification)
        module.__file__ = str(helper_path)
        sys.modules[specification.name] = module
        compiled = compile(data, str(helper_path), "exec", dont_inherit=True)
        exec(compiled, module.__dict__)
        return module, record
    except VisionReferenceFailure:
        if module is not None and sys.modules.get("_qwen36_phase22_reference") is module:
            del sys.modules["_qwen36_phase22_reference"]
        try:
            os.close(record.descriptor)
        except OSError:
            pass
        raise
    except Exception as error:
        if module is not None and sys.modules.get("_qwen36_phase22_reference") is module:
            del sys.modules["_qwen36_phase22_reference"]
        try:
            os.close(record.descriptor)
        except OSError:
            pass
        fail("environment", 3,
             f"cannot import accepted text helper: {type(error).__name__}: {error}")


def _validate_sidecars(official: Path, helper: Any) -> None:
    require(set(helper.SIDECAR_SHA256) >= {
        "config.json", "model.safetensors.index.json", "preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json",
    }, "identityOrFormat", 5, "accepted helper sidecar inventory is incomplete")
    for relative, expected in sorted(helper.SIDECAR_SHA256.items()):
        path = official / relative
        record = open_hashed_regular(path, expected, f"official sidecar {relative}")
        os.close(record.descriptor)


def _receipt_path_matches(root: Path, claimed: Any) -> bool:
    if not isinstance(claimed, str) or not claimed.startswith("/") or "\x00" in claimed:
        return False
    try:
        return Path(claimed).resolve(strict=True) == root.resolve(strict=True)
    except OSError:
        return False


def _validate_receipt(root: Path, manifest_data: bytes, files: dict[str, OpenFile],
                      expected_policy_sha256: str, compatible: str | None) -> tuple[str, dict[str, Any]]:
    data = read_bounded(root / "verified-install.json", MAX_METADATA_BYTES,
                        "verified-install.json", "identityOrFormat", 5)
    receipt_sha = sha256_bytes(data)
    receipt = decode_json(data, "verified-install.json", "identityOrFormat", 5)
    keys = {
        "schemaVersion", "manifestSha256", "modelDirectoryPath", "sourceRepoID",
        "sourceRevision", "verificationTimestamp", "toolVersion", "files",
        "sourceIndexSHA256", "sourcePayloadSHA256", "planFingerprint",
        "quantizationPolicySHA256", "converterVersion", "verificationTimePolicy",
    }
    if compatible is not None:
        keys.add("compatibleTextManifestSHA256")
    receipt = exact_keys(receipt, keys, "verified-install.json")
    require(receipt["schemaVersion"] == 1 and
            receipt["manifestSha256"] == sha256_bytes(manifest_data) and
            _receipt_path_matches(root, receipt["modelDirectoryPath"]) and
            receipt["sourceRepoID"] == MODEL_ID and
            receipt["sourceRevision"] == SOURCE_REVISION and
            receipt["verificationTimestamp"] == "1970-01-01T00:00:00Z" and
            receipt["toolVersion"] == CONVERTER_VERSION and
            receipt["sourceIndexSHA256"] == SOURCE_INDEX_SHA256 and
            isinstance(receipt["sourcePayloadSHA256"], str) and
            SHA256_RE.fullmatch(receipt["sourcePayloadSHA256"]) is not None and
            isinstance(receipt["planFingerprint"], str) and
            SHA256_RE.fullmatch(receipt["planFingerprint"]) is not None and
            receipt["quantizationPolicySHA256"] == expected_policy_sha256 and
            receipt["converterVersion"] == CONVERTER_VERSION and
            receipt["verificationTimePolicy"] == "deterministic-epoch; see operational evidence",
            "identityOrFormat", 5, "verified-install provenance is invalid")
    if compatible is not None:
        require(receipt["compatibleTextManifestSHA256"] == compatible,
                "identityOrFormat", 5, "vision receipt text binding is invalid")
    raw_files = receipt["files"]
    expected_paths = set(files) | {"manifest.json"}
    require(isinstance(raw_files, dict) and set(raw_files) == expected_paths,
            "identityOrFormat", 5, "verified-install file inventory is invalid")
    expected_records = {
        **{name: (record.size, record.sha256) for name, record in files.items()},
        "manifest.json": (len(manifest_data), sha256_bytes(manifest_data)),
    }
    for name in sorted(expected_records):
        item = exact_keys(raw_files[name], {"size", "sha256"},
                          f"verified-install.files.{name}")
        require((item["size"], item["sha256"]) == expected_records[name],
                "identityOrFormat", 5, f"verified-install binding is invalid for {name}")
    return receipt_sha, receipt


def _validate_vision_pack(root: Path, expected_manifest_sha256: str,
                          text_manifest_sha256: str,
                          expected_policy_sha256: str) -> ValidatedVisionPack:
    manifest_path = root / "manifest.json"
    manifest_data = read_bounded(manifest_path, MAX_METADATA_BYTES,
                                 "vision manifest", "identityOrFormat", 5)
    require(sha256_bytes(manifest_data) == expected_manifest_sha256,
            "identityOrFormat", 5, "vision manifest SHA-256 differs from explicit identity")
    raw = decode_json(manifest_data, "vision manifest", "identityOrFormat", 5)
    raw = exact_keys(raw, {
        "magic", "artifactKind", "versionMajor", "versionMinor", "family", "modelID",
        "sourceRevision", "processorProfile", "processorConfigSHA256",
        "compatibleTextManifestSHA256", "visionPayloadSHA256",
        "supportsStillImages", "supportsVideo", "files", "tensorRegions",
    }, "vision manifest")
    profile = exact_keys(raw["processorProfile"], {
        "processorClass", "imageProcessorType", "patchSize",
        "temporalPatchSize", "spatialMergeSize",
    }, "vision manifest processorProfile")
    require(raw["magic"] == "GTURBO-VISION" and
            raw["artifactKind"] == "qwen3_6_vision_companion" and
            raw["versionMajor"] == 2 and raw["versionMinor"] == 0 and
            raw["family"] == "qwen3_6" and raw["modelID"] == MODEL_ID and
            raw["sourceRevision"] == SOURCE_REVISION and
            profile == {"processorClass": "Qwen3VLProcessor",
                        "imageProcessorType": "Qwen2VLImageProcessorFast",
                        "patchSize": 16, "temporalPatchSize": 2,
                        "spatialMergeSize": 2} and
            raw["processorConfigSHA256"] == PROCESSOR_SHA256 and
            raw["compatibleTextManifestSHA256"] == text_manifest_sha256 and
            raw["supportsStillImages"] is True and raw["supportsVideo"] is False,
            "identityOrFormat", 5, "vision manifest identity or processor contract is invalid")
    raw_files = raw["files"]
    require(isinstance(raw_files, dict) and
            set(raw_files) == {"vision_weights.bin", "preprocessor_config.json"},
            "identityOrFormat", 5, "vision manifest file set is invalid")
    files: dict[str, OpenFile] = {}
    try:
        for name in sorted(raw_files):
            item = exact_keys(raw_files[name], {"size", "sha256"},
                              f"vision manifest files.{name}")
            require(is_int(item["size"]) and item["size"] > 0 and
                    isinstance(item["sha256"], str) and SHA256_RE.fullmatch(item["sha256"]) is not None,
                    "identityOrFormat", 5, f"invalid vision file record for {name}")
            files[name] = open_hashed_regular(
                root / name, item["sha256"], f"vision payload {name}",
                expected_size=item["size"])
            files[name] = dataclasses.replace(files[name], relative_path=name)
        require(files["preprocessor_config.json"].sha256 == PROCESSOR_SHA256 and
                files["vision_weights.bin"].sha256 == raw["visionPayloadSHA256"],
                "identityOrFormat", 5, "vision payload digest binding is invalid")
        contract = expected_vision_tensor_contract()
        regions_raw = raw["tensorRegions"]
        require(isinstance(regions_raw, list) and len(regions_raw) == len(contract),
                "identityOrFormat", 5, "vision tensor count is not exactly 333")
        regions: list[VisionRegion] = []
        previous_end = 0
        for index, (item_raw, expected) in enumerate(zip(regions_raw, contract)):
            item = exact_keys(item_raw, {
                "name", "file", "offset", "size", "shape", "storage",
                "quantizationCategory",
            }, f"vision tensorRegions[{index}]")
            require(item["name"] == expected.name and item["file"] == "vision_weights.bin" and
                    is_int(item["offset"]) and item["offset"] % ALIGNMENT == 0 and
                    item["offset"] >= previous_end and item["size"] == expected.size and
                    item["shape"] == list(expected.shape) and item["storage"] == "bf16" and
                    item["quantizationCategory"] is None and
                    item["offset"] + item["size"] <= files["vision_weights.bin"].size,
                    "identityOrFormat", 5,
                    f"vision tensor contract mismatch at index {index}: {expected.name}")
            previous_end = item["offset"] + item["size"]
            regions.append(VisionRegion(expected.name, item["offset"], item["size"], expected.shape))
        require(sum(region.size for region in regions) == VISION_DECODED_WEIGHT_BYTES // 2,
                "identityOrFormat", 5, "vision tensor raw byte total is invalid")
        receipt_sha, receipt = _validate_receipt(
            root, manifest_data, files, expected_policy_sha256, text_manifest_sha256)
        return ValidatedVisionPack(root, expected_manifest_sha256, receipt_sha,
                                   text_manifest_sha256, files, tuple(regions), receipt)
    except BaseException:
        for record in files.values():
            try:
                os.close(record.descriptor)
            except OSError:
                pass
        raise


def _revalidate_file(record: OpenFile, label: str) -> None:
    try:
        info = os.fstat(record.descriptor)
        require(info.st_size == record.size, "identityOrFormat", 5,
                f"{label} size changed during execution")
        digest = hashlib.sha256()
        offset = 0
        while offset < record.size:
            chunk = os.pread(record.descriptor, min(8 * 1024 * 1024, record.size - offset), offset)
            require(bool(chunk), "identityOrFormat", 5,
                    f"short read while revalidating {label}")
            digest.update(chunk)
            offset += len(chunk)
        require(digest.hexdigest() == record.sha256, "identityOrFormat", 5,
                f"{label} changed during execution")
    except VisionReferenceFailure:
        raise
    except OSError as error:
        fail("identityOrFormat", 5, f"cannot revalidate {label}: {error}")


class VisionWeightDecoder:
    def __init__(self, modules: Any, pack: ValidatedVisionPack):
        self.modules = modules
        self.pack = pack
        self.mapping: mmap.mmap | None = None
        self.by_name = {region.name: region for region in pack.regions}

    def __enter__(self) -> "VisionWeightDecoder":
        try:
            self.mapping = mmap.mmap(
                self.pack.files["vision_weights.bin"].descriptor, 0, access=mmap.ACCESS_READ)
            return self
        except (OSError, ValueError) as error:
            fail("resource", 6, f"cannot map vision payload read-only: {error}")

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None

    def fill(self, destination: Any, name: str) -> None:
        require(self.mapping is not None and name in self.by_name,
                "execution", 7, f"missing mapped vision tensor {name}")
        region = self.by_name[name]
        require(tuple(int(value) for value in destination.shape) == region.shape,
                "execution", 7, f"vision destination shape mismatch for {name}")
        np = self.modules.np
        torch = self.modules.torch
        flat = destination.reshape(-1)
        block = max(1, 64 * 1024 * 1024 // 8)
        for start in range(0, flat.numel(), block):
            count = min(block, flat.numel() - start)
            bits = np.frombuffer(self.mapping, dtype="<u2", count=count,
                                 offset=region.offset + start * 2).astype(np.uint32)
            values = np.left_shift(bits, 16).view(np.float32)
            require(bool(np.isfinite(values).all()), "identityOrFormat", 5,
                    f"vision tensor contains non-finite values: {name}")
            flat[start:start + count].copy_(torch.from_numpy(values))


def _extend_environment(helper: Any, modules: Any, official: Path,
                        checkout: Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    try:
        from PIL import Image, ImageOps
        import PIL
        from transformers import Qwen3VLProcessor
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeModel, Qwen3_5MoeVisionModel,
        )
    except Exception as error:
        fail("environment", 3,
             f"pinned vision imports failed: {type(error).__name__}: {error}")
    require(PIL.__version__ == "10.3.0", "environment", 3,
            f"Pillow must be 10.3.0, got {PIL.__version__}")
    observed_commit, observed_tree = helper.validate_transformers_source_tree(
        sys.modules["transformers"], checkout)
    require(observed_commit == helper.TRANSFORMERS_COMMIT and
            observed_tree == helper.TRANSFORMERS_TREE,
            "environment", 3, "Transformers source changed during vision imports")
    try:
        processor = Qwen3VLProcessor.from_pretrained(
            str(official), local_files_only=True)
    except Exception as error:
        fail("environment", 3,
             f"cannot construct pinned local processor: {type(error).__name__}: {error}")
    environment = dict(modules.environment_record)
    environment.update({
        "pillowVersion": PIL.__version__,
        "processorClass": "Qwen3VLProcessor",
        "visionModelClass": "Qwen3_5MoeVisionModel",
        "processorMode": "pinned-local-only",
        "executedTransformersSources": _executed_transformers_sources(checkout),
    })
    return (processor, (Image, ImageOps),
            (Qwen3_5MoeConfig, Qwen3_5MoeModel, Qwen3_5MoeVisionModel), environment)


def _executed_transformers_sources(checkout: Path) -> list[dict[str, str]]:
    repository = checkout.resolve(strict=True)
    package = repository / "src" / "transformers"
    records: dict[str, str] = {}
    for name, module in tuple(sys.modules.items()):
        if name != "transformers" and not name.startswith("transformers."):
            continue
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        try:
            path = Path(source).resolve(strict=True)
            path.relative_to(package)
            relative = str(path.relative_to(repository))
            data = read_bounded(path, 16 * 1024 * 1024,
                                f"executed Transformers source {relative}",
                                "environment", 3)
            records[relative] = sha256_bytes(data)
        except (OSError, ValueError) as error:
            fail("environment", 3,
                 f"executed Transformers module {name} is outside pinned sources: {error}")
    require(bool(records), "environment", 3,
            "no executed Transformers source files were observed")
    return [{"path": path, "sha256": records[path]} for path in sorted(records)]


def _read_image(case: VisionRequestCase,
                image_modules: tuple[Any, Any]) -> tuple[Any, dict[str, Any]]:
    _revalidate_file(OpenFile(case.image.name, case.image, case.image_size,
                              case.image_sha256, case.image_descriptor),
                     f"case {case.identifier} image")
    try:
        image_module, image_ops = image_modules
        duplicate = os.dup(case.image_descriptor)
        with os.fdopen(duplicate, "rb", closefd=True) as stream:
            image = image_module.open(stream)
            try:
                require(getattr(image, "n_frames", 1) == 1 and
                        not getattr(image, "is_animated", False),
                        "request", 4, f"case {case.identifier} image must have one still frame")
                image.load()
                orientation = image.getexif().get(274, 1)
                require(is_int(orientation) and 1 <= orientation <= 8,
                        "request", 4,
                        f"case {case.identifier} EXIF orientation is outside 1...8")
                record = {
                    "basename": case.image.name,
                    "bytes": case.image_size,
                    "width": int(image.width),
                    "height": int(image.height),
                    "format": str(image.format),
                    "mode": str(image.mode),
                    "frameCount": 1,
                    "orientation": int(orientation),
                    "sha256": case.image_sha256,
                }
                oriented = image_ops.exif_transpose(image)
                try:
                    prepared = oriented.convert("RGB")
                finally:
                    if oriented is not image:
                        oriented.close()
                record.update({
                    "orientedWidth": int(prepared.width),
                    "orientedHeight": int(prepared.height),
                })
                return prepared, record
            finally:
                image.close()
    except VisionReferenceFailure:
        raise
    except Exception as error:
        fail("request", 4,
             f"case {case.identifier} image decode failed: {type(error).__name__}: {error}")


def _tensor_record(tensor: Any, modules: Any, *, include_bytes: bool = False) -> dict[str, Any]:
    np = modules.np
    values = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    require(bool(np.isfinite(values).all()), "execution", 7,
            "reference output contains a non-finite value")
    raw = values.tobytes(order="C")
    result: dict[str, Any] = {
        "shape": [int(value) for value in values.shape],
        "dtype": "float32-le",
        "byteCount": len(raw),
        "sha256": sha256_bytes(raw),
    }
    if include_bytes:
        result.update({"encoding": "base64", "bytes": base64.b64encode(raw).decode("ascii")})
    return result


def _materialize_vision_model(modules: Any, model_class: Any, config: Any,
                              decoder: VisionWeightDecoder) -> tuple[Any, int]:
    torch = modules.torch
    try:
        with torch.device("meta"):
            model = model_class(config.vision_config)
        model.to_empty(device="cpu")
        inv_freq, attention_scaling = (
            model.rotary_pos_emb.compute_axial_rope_parameters(
                config.vision_config, device="cpu"))
        model.rotary_pos_emb.inv_freq = torch.nn.Buffer(inv_freq, persistent=False)
        model.rotary_pos_emb.original_inv_freq = torch.nn.Buffer(
            inv_freq.clone(), persistent=False)
        model.rotary_pos_emb.attention_scaling = attention_scaling
        model.requires_grad_(False)
        model.eval()
        parameters = dict(model.named_parameters())
        expected = {item.name.removeprefix("model.visual.")
                    for item in expected_vision_tensor_contract()}
        require(set(parameters) == expected, "identityOrFormat", 5,
                "official vision parameter set differs from companion contract")
        buffers = dict(model.named_buffers())
        require(set(buffers) == {
            "rotary_pos_emb.inv_freq", "rotary_pos_emb.original_inv_freq",
        } and all(value.device.type == "cpu" and value.dtype == torch.float32
                  for value in buffers.values()),
                "environment", 3, "official vision derived buffers are not pinned CPU Float32")
        for local_name, parameter in parameters.items():
            require(parameter.device.type == "cpu" and parameter.dtype == torch.float32,
                    "environment", 3,
                    f"vision parameter {local_name} is not CPU Float32")
            decoder.fill(parameter, "model.visual." + local_name)
        parameter_bytes = sum(parameter.numel() * parameter.element_size()
                              for parameter in parameters.values())
        require(parameter_bytes == VISION_DECODED_WEIGHT_BYTES, "execution", 7,
                f"decoded vision weights total {parameter_bytes}, expected {VISION_DECODED_WEIGHT_BYTES}")
        live = parameter_bytes + sum(value.numel() * value.element_size()
                                     for value in buffers.values())
        return model, live
    except VisionReferenceFailure:
        raise
    except Exception as error:
        fail("execution", 7,
             f"vision materialization failed: {type(error).__name__}: {error}")


class VisionTextReference:
    def __init__(self, helper: Any, modules: Any, pack: Any,
                 config_document: dict[str, Any]):
        self.helper = helper
        self.base = helper.LayerStreamingReference(modules, pack, config_document)
        self.modules = modules

    @property
    def maximum_live_decoded_layer_bytes(self) -> int:
        return self.base.maximum_live_decoded_layer_bytes

    def run_case(self, decoder: Any, case: VisionRequestCase,
                 token_ids: tuple[int, ...], pad_rows: tuple[int, ...],
                 merger: Any, position_ids: Any, delta: int) -> dict[str, Any]:
        torch = self.modules.torch
        try:
            cache = self.modules.dynamic_cache(config=self.base.config)
            generated: list[int] = []
            steps: list[dict[str, Any]] = []
            input_ids = token_ids
            expected_length = 0
            stop_reason = "maxTokens"
            for step_index in range(case.max_new_tokens):
                with torch.inference_mode():
                    hidden = decoder.embedding_rows(input_ids)
                    past = int(cache.get_seq_length())
                    if step_index == 0:
                        require(past == 0 and hidden.shape[1] == position_ids.shape[2],
                                "execution", 7, "prefill cache or position shape is invalid")
                        require(tuple(int(value) for value in merger.shape) ==
                                (len(pad_rows), HIDDEN_SIZE),
                                "execution", 7, "merger shape does not match image pad rows")
                        hidden[0, list(pad_rows), :].copy_(merger)
                        positions = position_ids
                    else:
                        scalar = past + delta
                        positions = torch.full((3, 1, len(input_ids)), scalar,
                                               dtype=torch.long, device="cpu")
                    logits, sequence_length = self._forward_embeds(
                        decoder, hidden, positions, cache)
                expected_length += len(input_ids)
                require(sequence_length == expected_length, "execution", 7,
                        f"case {case.identifier} cache length is invalid")
                logits_record, greedy, top_k = self.base._serialize_logits(logits)
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
            return {"generatedTokenIDs": generated, "stopReason": stop_reason, "steps": steps}
        except VisionReferenceFailure:
            raise
        except KeyboardInterrupt:
            raise VisionReferenceCancelled("keyboard interrupt during text continuation")
        except Exception as error:
            fail("execution", 7,
                 f"vision text continuation failed: {type(error).__name__}: {error}")

    def _forward_embeds(self, decoder: Any, hidden: Any, positions: Any,
                        cache: Any) -> tuple[Any, int]:
        torch = self.modules.torch
        rope_positions = positions
        position_embeddings = self.base.rotary(hidden, rope_positions)
        full_mask = self.modules.create_causal_mask(
            config=self.base.config, inputs_embeds=hidden, attention_mask=None,
            past_key_values=cache, position_ids=None)
        recurrent_mask = self.modules.create_recurrent_attention_mask(
            config=self.base.config, inputs_embeds=hidden, attention_mask=None,
            past_key_values=cache)
        for layer_index in range(LAYER_COUNT):
            layer = self.base._materialize_layer(decoder, layer_index)
            hidden = layer(
                hidden, position_embeddings=position_embeddings,
                attention_mask=full_mask if layer_index % 4 == 3 else recurrent_mask,
                position_ids=None, past_key_values=cache, use_cache=True)
            require(bool(torch.isfinite(hidden).all().item()), "execution", 7,
                    f"non-finite hidden state after layer {layer_index}")
            del layer
            gc.collect()
        norm = self.modules.rms_norm(HIDDEN_SIZE, eps=float(self.base.config.rms_norm_eps))
        norm.requires_grad_(False)
        decoder.fill_resident(dict(norm.named_parameters())["weight"],
                              "model.language_model.norm.weight")
        hidden = norm(hidden)
        final_row = hidden[:, -1, :].contiguous()
        logits = decoder.output_logits(final_row)
        length = int(cache.get_seq_length())
        del norm, hidden, final_row
        return logits, length


def _official_positions(model_class: Any, config: Any, token_ids: Any,
                        token_types: Any, grid: Any, modules: Any) -> tuple[Any, int]:
    holder = types.SimpleNamespace(config=config)
    holder.get_vision_position_ids = types.MethodType(model_class.get_vision_position_ids, holder)
    holder.get_rope_index = types.MethodType(model_class.get_rope_index, holder)
    try:
        attention = modules.torch.ones_like(token_ids, dtype=modules.torch.long)
        positions, deltas = holder.get_rope_index(
            input_ids=token_ids, image_grid_thw=grid, video_grid_thw=None,
            second_per_grid_ts=None, attention_mask=attention,
            mm_token_type_ids=token_types)
        require(tuple(int(value) for value in positions.shape) ==
                (3, 1, int(token_ids.shape[1])) and
                tuple(int(value) for value in deltas.shape) == (1, 1),
                "execution", 7, "official M-RoPE output shape is invalid")
        return positions.to(dtype=modules.torch.long, device="cpu"), int(deltas.item())
    except VisionReferenceFailure:
        raise
    except Exception as error:
        fail("execution", 7,
             f"official M-RoPE execution failed: {type(error).__name__}: {error}")


def _process_cases(cases: tuple[VisionRequestCase, ...], processor: Any,
                   image_module: Any, official_classes: tuple[Any, Any, Any],
                   modules: Any, helper: Any, text_pack: Any,
                   text_config_document: dict[str, Any], vision_pack: ValidatedVisionPack
                   ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    config_class, model_class, vision_model_class = official_classes
    try:
        config = config_class(**dict(text_config_document))
        config._attn_implementation = "eager"
        config.vision_config._attn_implementation = "eager"
        config.text_config._attn_implementation = "eager"
        config.text_config._experts_implementation = "eager"
    except Exception as error:
        fail("environment", 3,
             f"cannot construct pinned multimodal configuration: {type(error).__name__}: {error}")
    results: list[dict[str, Any]] = []
    prepared: list[tuple[VisionRequestCase, dict[str, Any], Any, Any, Any, tuple[int, ...], dict[str, Any], dict[str, Any]]] = []
    maximum_patches = 0
    maximum_merged = 0
    maximum_retained_vision_tensor_bytes = 0
    with VisionWeightDecoder(modules, vision_pack) as vision_decoder:
        vision_model, live_vision_bytes = _materialize_vision_model(
            modules, vision_model_class, config, vision_decoder)
        try:
            for case in cases:
                image, image_record = _read_image(case, image_module)
                try:
                    batch = processor(text=[case.prompt], images=[image],
                                      padding=False, return_tensors="pt")
                finally:
                    image.close()
                required = {"pixel_values", "image_grid_thw", "input_ids",
                            "attention_mask", "mm_token_type_ids"}
                require(required.issubset(batch), "execution", 7,
                        f"processor output is missing {sorted(required - set(batch))}")
                pixels = batch["pixel_values"].to(dtype=modules.torch.float32, device="cpu").contiguous()
                grid = batch["image_grid_thw"].to(dtype=modules.torch.long, device="cpu").contiguous()
                token_ids_tensor = batch["input_ids"].to(dtype=modules.torch.long, device="cpu").contiguous()
                token_types_tensor = batch["mm_token_type_ids"].to(
                    dtype=modules.torch.long, device="cpu").contiguous()
                attention_tensor = batch["attention_mask"].to(
                    dtype=modules.torch.long, device="cpu").contiguous()
                require(tuple(int(value) for value in attention_tensor.shape) ==
                        tuple(int(value) for value in token_ids_tensor.shape) and
                        bool((attention_tensor == 1).all().item()),
                        "execution", 7, "unpadded processor attention mask is invalid")
                require(tuple(int(value) for value in grid.shape) == (1, 3),
                        "execution", 7, "processor image grid shape is invalid")
                grid_values = tuple(int(value) for value in grid[0].tolist())
                patch_rows = grid_values[0] * grid_values[1] * grid_values[2]
                require(0 < patch_rows <= MAX_PATCH_ROWS and patch_rows % 4 == 0,
                        "resource", 6, "processor patch row count is outside production limits")
                merged_rows = patch_rows // 4
                require(merged_rows <= MAX_MERGED_ROWS and
                        tuple(int(value) for value in pixels.shape) == (patch_rows, 1_536),
                        "resource", 6, "processor patch tensor shape is outside production limits")
                image_record.update({
                    "targetWidth": grid_values[2] * 16,
                    "targetHeight": grid_values[1] * 16,
                })
                float_raw = pixels.numpy().astype("<f4", copy=False).tobytes(order="C")
                bf16_raw = float32_le_to_bf16_rne(float_raw)
                roundtrip_raw = bf16_le_to_float32_le(bf16_raw)
                np_values = modules.np.frombuffer(roundtrip_raw, dtype="<f4").copy().reshape(
                    tuple(int(value) for value in pixels.shape))
                roundtrip = modules.torch.from_numpy(np_values)
                ids = tuple(int(value) for value in token_ids_tensor[0].tolist())
                types_values = tuple(int(value) for value in token_types_tensor[0].tolist())
                require(len(ids) == len(types_values), "execution", 7,
                        "token and modality arrays differ in length")
                pads = tuple(index for index, (token, kind) in enumerate(zip(ids, types_values))
                             if token == IMAGE_PAD_TOKEN_ID and kind == 1)
                require(len(pads) == merged_rows and pads == tuple(range(pads[0], pads[-1] + 1)),
                        "execution", 7, "image pad rows are not the expected contiguous span")
                require(pads[0] > 0 and pads[-1] + 1 < len(ids) and
                        ids[pads[0] - 1] == VISION_START_TOKEN_ID and
                        ids[pads[-1] + 1] == VISION_END_TOKEN_ID,
                        "execution", 7, "image pad span is not bounded by vision markers")
                positions, delta = _official_positions(
                    model_class, config, token_ids_tensor, token_types_tensor, grid, modules)
                with modules.torch.inference_mode():
                    output = vision_model(hidden_states=roundtrip, grid_thw=grid,
                                          return_dict=True)
                tower = output.last_hidden_state
                merger = output.pooler_output
                require(tuple(int(value) for value in tower.shape) == (patch_rows, 1_152) and
                        tuple(int(value) for value in merger.shape) == (merged_rows, HIDDEN_SIZE),
                        "execution", 7, "official tower or merger output shape is invalid")
                live_tensors = (pixels, roundtrip, grid, token_ids_tensor,
                                token_types_tensor, attention_tensor,
                                positions, tower, merger)
                live_case_bytes = live_vision_bytes + sum(
                    int(value.numel()) * int(value.element_size()) for value in live_tensors)
                maximum_retained_vision_tensor_bytes = max(
                    maximum_retained_vision_tensor_bytes, live_case_bytes)
                float_record = {"shape": [patch_rows, 1_536], "dtype": "float32-le",
                                "byteCount": len(float_raw), "sha256": sha256_bytes(float_raw)}
                bf16_record = {"shape": [patch_rows, 1_536], "dtype": "bfloat16-le-rne",
                               "byteCount": len(bf16_raw), "sha256": sha256_bytes(bf16_raw)}
                prepared.append((case, image_record, token_ids_tensor, token_types_tensor,
                                 grid, pads, _tensor_record(tower, modules),
                                 {"processorFloat32": float_record,
                                  "processorBF16": bf16_record,
                                  "merger": merger.detach().cpu().contiguous(),
                                  "positionIDs": positions.detach().cpu().contiguous(),
                                  "delta": delta, "ids": ids, "types": types_values}))
                maximum_patches = max(maximum_patches, patch_rows)
                maximum_merged = max(maximum_merged, merged_rows)
                del output, tower, merger, pixels, roundtrip
        finally:
            del vision_model
            gc.collect()
    text_reference = VisionTextReference(helper, modules, text_pack, text_config_document)
    with helper.TensorDecoder(modules, text_pack) as text_decoder:
        for case, image_record, _ids_tensor, _types_tensor, grid, pads, tower_record, state in prepared:
            continuation = text_reference.run_case(
                text_decoder, case, state["ids"], pads, state["merger"],
                state["positionIDs"], state["delta"])
            try:
                decoded = processor.tokenizer.decode(
                    continuation["generatedTokenIDs"], skip_special_tokens=False)
            except Exception as error:
                fail("execution", 7,
                     f"case {case.identifier} token decode failed: {type(error).__name__}: {error}")
            prompt_bytes = case.prompt.encode("utf-8")
            results.append({
                "id": case.identifier,
                "image": image_record,
                "prompt": {"utf8Bytes": len(prompt_bytes), "sha256": sha256_bytes(prompt_bytes)},
                "effectiveTokenIDs": list(state["ids"]),
                "mmTokenTypeIDs": list(state["types"]),
                "imageGridTHW": [int(value) for value in grid[0].tolist()],
                "padRows": list(pads),
                "processorFloat32": state["processorFloat32"],
                "processorBF16": state["processorBF16"],
                "towerOutput": tower_record,
                "mergerOutput": _tensor_record(state["merger"], modules, include_bytes=True),
                "positionIDs": state["positionIDs"].tolist(),
                "mropePositionDelta": state["delta"],
                "maxNewTokens": case.max_new_tokens,
                "generatedTokenIDs": continuation["generatedTokenIDs"],
                "decodedText": decoded,
                "stopReason": continuation["stopReason"],
                "steps": continuation["steps"],
            })
    return results, {
        "decodedFloat32VisionWeightBytes": VISION_DECODED_WEIGHT_BYTES,
        "maximumRetainedOfficialVisionTensorBytes": maximum_retained_vision_tensor_bytes,
        "maximumLiveDecodedTextLayerBytes": text_reference.maximum_live_decoded_layer_bytes,
        "maximumObservedPatchRows": maximum_patches,
        "maximumObservedMergedRows": maximum_merged,
    }


def _vision_identity(pack: ValidatedVisionPack) -> dict[str, Any]:
    return {
        "manifestSHA256": pack.manifest_sha256,
        "receiptSHA256": pack.receipt_sha256,
        "artifactKind": "qwen3_6_vision_companion",
        "modelID": MODEL_ID,
        "sourceRevision": SOURCE_REVISION,
        "compatibleTextManifestSHA256": pack.compatible_text_manifest_sha256,
        "processorProfile": {"processorClass": "Qwen3VLProcessor",
                             "imageProcessorType": "Qwen2VLImageProcessorFast",
                             "patchSize": 16, "temporalPatchSize": 2,
                             "spatialMergeSize": 2},
        "processorConfigSHA256": PROCESSOR_SHA256,
        "visionPayloadSHA256": pack.files["vision_weights.bin"].sha256,
        "supportsStillImages": True,
        "supportsVideo": False,
        "files": [{"path": name, "size": pack.files[name].size,
                   "sha256": pack.files[name].sha256} for name in sorted(pack.files)],
    }


def _comparison_rules() -> dict[str, Any]:
    return {
        "referenceAppliesPassFail": False,
        "preprocessingAdmission": {
            "rule": "exact-bf16-patch-shape-byte-count-and-sha256",
            "hardGateBeforeFeaturesOrLogits": True,
        },
        "exact": ["rawImageIdentity", "orientedAndTargetGeometry", "imageGridTHW",
                  "effectiveTokenIDs", "mmTokenTypeIDs", "padRows", "positionIDs",
                  "mropePositionDelta", "sequenceLength", "greedyTokenID", "stopReason"],
        "visionFeatures": {
            "rule": "elementwise",
            "absolute": 1e-5,
            "relative": 1e-5,
            "predicate": "abs(candidate-reference)<=absolute+relative*abs(reference)",
            "provenance": "P16",
        },
        "logits": {
            "rule": "vector-global",
            "absolute": 1e-5,
            "relative": 1e-5,
            "predicate": "maxAbs<=absolute+relative*max(maxAbsCandidate,maxAbsReference)",
            "requiresExactFirstArgmax": True,
            "provenance": "P3/P22",
        },
        "policy": "preregistered-authentic-cap; no-post-result-widening",
    }


def _execute(arguments: list[str]) -> None:
    parsed = parse_arguments(arguments)
    install_cancellation_handlers()
    text_root = canonical_existing(parsed.text_model_directory, "text model directory", directory=True)
    vision_root = canonical_existing(parsed.vision_model_directory, "vision model directory", directory=True)
    official = canonical_existing(parsed.official_source_directory, "official source directory", directory=True)
    checkout = canonical_existing(parsed.transformers_checkout, "Transformers checkout", directory=True)
    helper_path = canonical_existing(parsed.text_reference_script, "text reference script")
    request_path = canonical_existing(parsed.request, "request")
    script_path = canonical_existing(Path(__file__).resolve(), "vision reference script")
    cases: tuple[VisionRequestCase, ...] = ()
    text_pack = None
    vision_pack = None
    script_file: OpenFile | None = None
    helper_file: OpenFile | None = None
    request_file: OpenFile | None = None
    try:
        _install_offline_process_boundary()
        _script_data, script_file = open_bounded_regular(
            script_path, 4 * 1024 * 1024,
            "vision reference script", "identityOrFormat", 5)
        del _script_data
        helper, helper_file = _load_helper(helper_path)
        _validate_sidecars(official, helper)
        validated_request = _validate_request_document(request_path)
        cases = validated_request.cases
        request_file = validated_request.file
        output = _canonical_output(parsed, (text_root, vision_root, official, checkout,
                                            helper_path, request_path, script_path,
                                            *(case.image for case in cases)))
        modules = helper.validate_environment(parsed.torch_threads, checkout)
        text_pack, config_document = helper.validate_manifest(
            text_root, official / "config.json", parsed.expected_text_manifest_sha256,
            parsed.expected_policy_sha256)
        text_manifest_data = read_bounded(text_root / "manifest.json", MAX_METADATA_BYTES,
                                          "text manifest", "identityOrFormat", 5)
        _text_receipt_sha, text_receipt = _validate_receipt(
            text_root, text_manifest_data, text_pack.files,
            parsed.expected_policy_sha256, None)
        vision_pack = _validate_vision_pack(
            vision_root, parsed.expected_vision_manifest_sha256,
            parsed.expected_text_manifest_sha256, parsed.expected_policy_sha256)
        require(all(vision_pack.receipt[field] == text_receipt[field] for field in (
            "sourceIndexSHA256", "sourcePayloadSHA256", "planFingerprint",
            "quantizationPolicySHA256", "converterVersion",
        )), "identityOrFormat", 5,
                "text and vision receipt conversion provenance differs")
        processor, image_module, official_classes, environment = _extend_environment(
            helper, modules, official, checkout)
        case_results, resources = _process_cases(
            cases, processor, image_module, official_classes, modules, helper,
            text_pack, config_document, vision_pack)
        helper.revalidate_payloads(text_pack)
        for record in vision_pack.files.values():
            _revalidate_file(record, f"vision payload {record.relative_path}")
        for case in cases:
            _revalidate_file(OpenFile(case.image.name, case.image, case.image_size,
                                      case.image_sha256, case.image_descriptor),
                             f"case {case.identifier} image")
        require(request_file is not None and helper_file is not None and script_file is not None,
                "execution", 7, "validated boundary file identity is missing")
        _revalidate_file(request_file, "request")
        _revalidate_file(helper_file, "text reference script")
        _revalidate_file(script_file, "vision reference script")
        _validate_sidecars(official, helper)
        observed_commit, observed_tree = helper.validate_transformers_source_tree(
            sys.modules["transformers"], checkout)
        require(observed_commit == helper.TRANSFORMERS_COMMIT and
                observed_tree == helper.TRANSFORMERS_TREE,
                "environment", 3, "Transformers source changed during execution")
        environment["executedTransformersSources"] = _executed_transformers_sources(checkout)
        require(helper_file.sha256 == TEXT_HELPER_SHA256,
                "identityOrFormat", 5, "text helper changed during execution")
        require(not any(name.startswith("TurboFieldfare") for name in sys.modules),
                "environment", 3, "TurboFieldfare runtime module was imported")
        result: dict[str, Any] = {
            "schemaVersion": RESULT_SCHEMA,
            "status": "complete",
            "textIdentity": helper.identity_record(text_pack),
            "visionIdentity": _vision_identity(vision_pack),
            "referenceEnvironment": environment,
            "reader": {
                "schema": READER_SCHEMA,
                "scriptSHA256": script_file.sha256,
                "textReferenceScriptSHA256": TEXT_HELPER_SHA256,
                "maximumTransientDecodeBytes": helper.MAX_TRANSIENT_DECODE_BYTES,
                "maximumLiveDecodedTextLayerBytes": resources["maximumLiveDecodedTextLayerBytes"],
                "runtimeModulesImported": False,
                "bf16CheckpointRead": False,
            },
            "request": {"sha256": request_file.sha256,
                        "caseCount": len(cases)},
            "comparisonRules": _comparison_rules(),
            "cases": case_results,
            "resources": {**resources, "finalJSONBytes": 0},
        }
        for _ in range(16):
            observed = len(encode_result_json(result))
            if observed == result["resources"]["finalJSONBytes"]:
                break
            result["resources"]["finalJSONBytes"] = observed
        require(result["resources"]["finalJSONBytes"] == len(encode_result_json(result)),
                "outputWrite", 9, "result byte-count fixed point did not converge")
    finally:
        _close_cases(cases)
        if vision_pack is not None:
            vision_pack.close()
        if text_pack is not None:
            for record in text_pack.files.values():
                try:
                    os.close(record.descriptor)
                except OSError:
                    pass
        _close_open_file(request_file)
        _close_open_file(helper_file)
        _close_open_file(script_file)
    _terminal_atomic_commit(output, result, helper)


def install_cancellation_handlers() -> None:
    def cancel(signum: int, _frame: Any) -> NoReturn:
        raise VisionReferenceCancelled(f"received signal {signum}")
    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)


def main() -> NoReturn:
    try:
        _execute(sys.argv[1:])
    except VisionReferenceFailure as error:
        print(f"qwen36_quantized_vision_reference: {error.category}: {error.detail}",
              file=sys.stderr, flush=True)
        raise SystemExit(error.exit_code)
    except KeyboardInterrupt:
        print("qwen36_quantized_vision_reference: cancelled: keyboard interrupt",
              file=sys.stderr, flush=True)
        raise SystemExit(8)
    except BaseException as error:
        category = getattr(error, "category", None)
        exit_code = getattr(error, "exit_code", None)
        detail_value = getattr(error, "detail", None)
        if isinstance(category, str) and isinstance(exit_code, int) and isinstance(detail_value, str):
            print(f"qwen36_quantized_vision_reference: {category}: {detail_value}",
                  file=sys.stderr, flush=True)
            raise SystemExit(exit_code)
        detail = " ".join(f"{type(error).__name__}: {error}".splitlines())[:1_024]
        print(f"qwen36_quantized_vision_reference: execution: {detail}",
              file=sys.stderr, flush=True)
        raise SystemExit(7)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
