#!/usr/bin/env python3
"""Standard-library tests for the staged Qwen 3.6 vision reference seams.

The test module deliberately uses arbitrary image bytes.  Request validation
is required to authenticate those bytes without decoding pixels, so these
tests do not import Pillow, NumPy, Torch, or Transformers.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock


SCRIPT_PATH_ENVIRONMENT = "QWEN36_QUANTIZED_VISION_REFERENCE_SCRIPT"
REFERENCE_MODULE_NAME = "_qwen36_quantized_vision_reference_unit_subject"
REFERENCE_SCRIPT_NAME = "qwen36_quantized_vision_reference.py"
IMAGE_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"


def reference_script_path() -> Path:
    explicit = os.environ.get(SCRIPT_PATH_ENVIRONMENT)
    if explicit:
        return Path(explicit).expanduser().resolve()
    # staging/Tests/Python and the promoted repository Tests/Python both have
    # the staged/promoted Scripts directory three levels above this file.
    return Path(__file__).resolve().parents[2] / "Scripts" / REFERENCE_SCRIPT_NAME


def load_reference_module() -> types.ModuleType:
    script = reference_script_path()
    spec = importlib.util.spec_from_file_location(REFERENCE_MODULE_NAME, script)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load reference script at {script}")
    subject = importlib.util.module_from_spec(spec)
    sys.modules[REFERENCE_MODULE_NAME] = subject
    spec.loader.exec_module(subject)
    return subject


def record_field(record: object, *names: str) -> object:
    for name in names:
        if isinstance(record, dict) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    raise AssertionError(f"record {record!r} has none of {names!r}")


def expected_tensor_names() -> list[str]:
    names = [
        "model.visual.patch_embed.proj.bias",
        "model.visual.patch_embed.proj.weight",
        "model.visual.pos_embed.weight",
    ]
    block_members = [
        "attn.proj.bias", "attn.proj.weight", "attn.qkv.bias", "attn.qkv.weight",
        "mlp.linear_fc1.bias", "mlp.linear_fc1.weight",
        "mlp.linear_fc2.bias", "mlp.linear_fc2.weight",
        "norm1.bias", "norm1.weight", "norm2.bias", "norm2.weight",
    ]
    for block in range(27):
        names.extend(f"model.visual.blocks.{block}.{member}" for member in block_members)
    names.extend([
        "model.visual.merger.linear_fc1.bias",
        "model.visual.merger.linear_fc1.weight",
        "model.visual.merger.linear_fc2.bias",
        "model.visual.merger.linear_fc2.weight",
        "model.visual.merger.norm.bias",
        "model.visual.merger.norm.weight",
    ])
    return names


def expected_tensor_shapes() -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = [
        (1152,), (1152, 3, 2, 16, 16), (2304, 1152),
    ]
    block_shapes = [
        (1152,), (1152, 1152), (3456,), (3456, 1152),
        (4304,), (4304, 1152), (1152,), (1152, 4304),
        (1152,), (1152,), (1152,), (1152,),
    ]
    for _ in range(27):
        shapes.extend(block_shapes)
    shapes.extend([
        (4608,), (4608, 4608), (2048,), (2048, 4608),
        (1152,), (1152,),
    ])
    return shapes


def f32_bits(bits: int) -> bytes:
    return struct.pack("<I", bits)


class VisionReferenceUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subject = load_reference_module()

    @classmethod
    def tearDownClass(cls) -> None:
        if sys.modules.get(REFERENCE_MODULE_NAME) is cls.subject:
            del sys.modules[REFERENCE_MODULE_NAME]

    def assert_subject_failure(
        self, operation: object, *, category: str | None = None,
        exit_code: int | None = None,
    ) -> BaseException:
        with self.assertRaises(Exception) as raised:
            operation()  # type: ignore[operator]
        failure = raised.exception
        if category is not None:
            self.assertEqual(getattr(failure, "category", None), category)
        if exit_code is not None:
            self.assertEqual(getattr(failure, "exit_code", None), exit_code)
        return failure

    @staticmethod
    def cli_arguments() -> list[str]:
        return [
            "--text-model-directory", "/tmp/qwen.gturbo",
            "--vision-model-directory", "/tmp/qwen.vision.gturbo",
            "--official-source-directory", "/tmp/official",
            "--transformers-checkout", "/tmp/transformers",
            "--text-reference-script", "/tmp/qwen36_quantized_reference.py",
            "--request", "/tmp/request.json",
            "--expected-text-manifest-sha256", "a" * 64,
            "--expected-vision-manifest-sha256", "b" * 64,
            "--expected-policy-sha256", "c" * 64,
            "--torch-threads", "12",
            "--output", "/tmp/result.json",
        ]

    def test_import_has_no_heavy_runtime_modules_or_side_effect_exports(self) -> None:
        for name in ("torch", "numpy", "PIL", "transformers"):
            self.assertNotIn(name, self.subject.__dict__)
        self.assertTrue(hasattr(self.subject, "parse_arguments"))
        self.assertTrue(hasattr(self.subject, "validate_request"))

    def test_parse_arguments_accepts_all_required_options_once(self) -> None:
        parsed = self.subject.parse_arguments(self.cli_arguments())
        self.assertEqual(Path(parsed.output), Path("/tmp/result.json"))
        self.assertEqual(Path(parsed.vision_model_directory), Path("/tmp/qwen.vision.gturbo"))
        self.assertEqual(parsed.torch_threads, 12)

    def test_parse_arguments_rejects_missing_duplicate_and_unknown_options(self) -> None:
        missing = self.cli_arguments()
        index = missing.index("--output")
        del missing[index:index + 2]
        self.assert_subject_failure(
            lambda: self.subject.parse_arguments(missing),
            category="usage", exit_code=2,
        )

        duplicate = self.cli_arguments() + ["--output", "/tmp/second.json"]
        self.assert_subject_failure(
            lambda: self.subject.parse_arguments(duplicate),
            category="usage", exit_code=2,
        )

        unknown = self.cli_arguments() + ["--unexpected", "value"]
        self.assert_subject_failure(
            lambda: self.subject.parse_arguments(unknown),
            category="usage", exit_code=2,
        )

    @staticmethod
    def write_request(root: Path, cases: list[dict[str, object]]) -> Path:
        request = root / "request.json"
        request.write_text(json.dumps({
            "schemaVersion": "qwen36-quantized-vision-reference-request-v1",
            "cases": cases,
        }), encoding="utf-8")
        return request

    @staticmethod
    def request_case(image: Path, identifier: str = "square-png") -> dict[str, object]:
        image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
        return {
            "id": identifier,
            "image": str(image),
            "imageSHA256": image_hash,
            "prompt": f"Describe this image {IMAGE_MARKER}",
            "maxNewTokens": 2,
        }

    def test_validate_request_accepts_arbitrary_bytes_without_image_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            image = root / "arbitrary.png"
            image.write_bytes(b"not a decoded image; digest validation is the boundary")
            cases = self.subject.validate_request(self.write_request(
                root, [self.request_case(image)]))
            self.assertEqual(len(cases), 1)
            self.assertEqual(
                record_field(cases[0], "identifier", "id"), "square-png",
            )
            self.assertEqual(
                int(record_field(cases[0], "max_new_tokens", "maxNewTokens")), 2,
            )

    def test_validate_request_rejects_digest_unknown_keys_duplicates_and_marker_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            image = root / "fixture.png"
            image.write_bytes(b"arbitrary fixture bytes")
            valid = self.request_case(image)

            changed_digest = dict(valid)
            changed_digest["imageSHA256"] = "0" * 64
            self.assert_subject_failure(
                lambda: self.subject.validate_request(
                    self.write_request(root, [changed_digest])),
                category="request", exit_code=4,
            )

            unknown_key = dict(valid)
            unknown_key["unexpected"] = True
            self.assert_subject_failure(
                lambda: self.subject.validate_request(
                    self.write_request(root, [unknown_key])),
                category="request", exit_code=4,
            )

            duplicate_id = dict(valid)
            duplicate_id["image"] = str(root / "second.png")
            second = Path(duplicate_id["image"])
            second.write_bytes(b"second arbitrary fixture")
            duplicate_id["imageSHA256"] = hashlib.sha256(second.read_bytes()).hexdigest()
            self.assert_subject_failure(
                lambda: self.subject.validate_request(
                    self.write_request(root, [valid, duplicate_id])),
                category="request", exit_code=4,
            )

            repeated_marker = dict(valid)
            repeated_marker["prompt"] = IMAGE_MARKER + IMAGE_MARKER
            self.assert_subject_failure(
                lambda: self.subject.validate_request(
                    self.write_request(root, [repeated_marker])),
                category="request", exit_code=4,
            )

            invalid_budget = dict(valid)
            invalid_budget["maxNewTokens"] = 3
            self.assert_subject_failure(
                lambda: self.subject.validate_request(
                    self.write_request(root, [invalid_budget])),
                category="request", exit_code=4,
            )

    def test_validate_request_rejects_image_symlink_and_duplicate_image_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            image = root / "real.bin"
            image.write_bytes(b"arbitrary bytes")
            link = root / "link.png"
            link.symlink_to(image)
            linked = self.request_case(link)
            self.assert_subject_failure(
                lambda: self.subject.validate_request(
                    self.write_request(root, [linked])),
                category="request", exit_code=4,
            )

            duplicate_path = dict(self.request_case(image, "second"))
            self.assert_subject_failure(
                lambda: self.subject.validate_request(
                    self.write_request(root, [self.request_case(image), duplicate_path])),
                category="request", exit_code=4,
            )

    def test_bounded_descriptor_reads_reject_symlinks_and_oversized_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            target = root / "target.bin"
            target.write_bytes(b"target bytes")
            link = root / "link.bin"
            link.symlink_to(target)
            self.assert_subject_failure(
                lambda: self.subject.open_bounded_regular(
                    link, 64, "linked fixture", "request", 4),
                category="request", exit_code=4,
            )
            self.assert_subject_failure(
                lambda: self.subject.open_hashed_regular(
                    link, hashlib.sha256(target.read_bytes()).hexdigest(), "hashed fixture"),
                category="identityOrFormat", exit_code=5,
            )

            oversized = root / "oversized.bin"
            oversized.write_bytes(b"0123456789")
            failure = self.assert_subject_failure(
                lambda: self.subject.open_bounded_regular(
                    oversized, 4, "bounded fixture", "request", 4),
                category="request", exit_code=4,
            )
            self.assertIn("exceeds 4 bytes", str(failure))
            size_mismatch = self.assert_subject_failure(
                lambda: self.subject.open_hashed_regular(
                    oversized, hashlib.sha256(oversized.read_bytes()).hexdigest(),
                    "hashed fixture", expected_size=4),
                category="identityOrFormat", exit_code=5,
            )
            self.assertIn("size mismatch", str(size_mismatch))

    def test_validated_request_keeps_descriptor_identity_when_path_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            image = root / "image.bin"
            image_bytes = b"descriptor-bound image bytes"
            image.write_bytes(image_bytes)
            request = self.write_request(root, [self.request_case(image)])
            original_request_bytes = request.read_bytes()
            document = self.subject._validate_request_document(request)
            try:
                replacement = root / "replacement.json"
                replacement.write_bytes(b"replacement at the old request path")
                os.replace(replacement, request)

                self.assertEqual(
                    os.pread(document.file.descriptor, document.file.size, 0),
                    original_request_bytes,
                )
                self.assertEqual(
                    document.file.sha256,
                    hashlib.sha256(original_request_bytes).hexdigest(),
                )
                self.assertEqual(
                    os.pread(
                        document.cases[0].image_descriptor,
                        document.cases[0].image_size,
                        0,
                    ),
                    image_bytes,
                )
            finally:
                for case in document.cases:
                    os.close(case.image_descriptor)
                os.close(document.file.descriptor)

    def test_expected_vision_tensor_contract_is_exact_and_ordered(self) -> None:
        contract = self.subject.expected_vision_tensor_contract()
        self.assertIsInstance(contract, tuple)
        self.assertEqual(len(contract), 333)
        names = [str(record_field(item, "name")) for item in contract]
        self.assertEqual(names, expected_tensor_names())
        self.assertEqual(len(set(names)), 333)

        shapes = [tuple(record_field(item, "shape")) for item in contract]
        self.assertEqual(shapes, expected_tensor_shapes())
        for item in contract:
            storage = str(record_field(item, "storage", "dtype")).lower()
            self.assertIn(storage, {"bf16", "bfloat16", "bfloat16-le"})

    def test_float32_to_bf16_uses_little_endian_round_to_nearest_even(self) -> None:
        values = b"".join([
            f32_bits(0x3F800000),  # 1.0
            f32_bits(0xC0200000),  # -2.5
            f32_bits(0x3F808000),  # exact tie, even high bit: round down
            f32_bits(0x3F818000),  # exact tie, odd high bit: round up
            f32_bits(0x3F81FFFF),  # carry after the tie
        ])
        expected = b"".join([
            struct.pack("<H", 0x3F80), struct.pack("<H", 0xC020),
            struct.pack("<H", 0x3F80), struct.pack("<H", 0x3F82),
            struct.pack("<H", 0x3F82),
        ])
        self.assertEqual(self.subject.float32_le_to_bf16_rne(values), expected)

    def test_bf16_to_float32_expands_little_endian_words_exactly(self) -> None:
        bf16 = struct.pack("<HH", 0x3F80, 0xC020)
        self.assertEqual(
            self.subject.bf16_le_to_float32_le(bf16),
            struct.pack("<II", 0x3F800000, 0xC0200000),
        )

    def test_bf16_converters_reject_invalid_lengths_and_nonfinite_float32(self) -> None:
        for operation, data in (
            (self.subject.float32_le_to_bf16_rne, b"\x00"),
            (self.subject.float32_le_to_bf16_rne, b"\x00" * 6),
            (self.subject.bf16_le_to_float32_le, b"\x00"),
            (self.subject.bf16_le_to_float32_le, b"\x00" * 3),
        ):
            with self.subTest(operation=operation.__name__, length=len(data)):
                with self.assertRaises(Exception):
                    operation(data)
        for bits in (0x7F800000, 0xFF800000, 0x7FC00000):
            with self.subTest(bits=hex(bits)):
                with self.assertRaises(Exception):
                    self.subject.float32_le_to_bf16_rne(f32_bits(bits))

    def test_encode_result_json_is_sorted_compact_ascii_finite_and_newline_terminated(self) -> None:
        value = {"z": "café", "a": [1, True]}
        self.assertEqual(
            self.subject.encode_result_json(value),
            b'{"a":[1,true],"z":"caf\\u00e9"}\n',
        )
        for nonfinite in (math.nan, math.inf, -math.inf):
            with self.subTest(nonfinite=nonfinite):
                self.assert_subject_failure(
                    lambda nonfinite=nonfinite: self.subject.encode_result_json(
                        {"value": nonfinite}),
                )

    def test_encode_result_json_enforces_the_bounded_result_cap(self) -> None:
        cap_name = next(
            (name for name in ("MAX_OUTPUT_BYTES", "MAX_RESULT_BYTES", "MAX_RESULT_JSON_BYTES")
             if hasattr(self.subject, name)),
            None,
        )
        self.assertIsNotNone(cap_name, "reference must expose its result cap")
        with mock.patch.object(self.subject, cap_name, 64):  # type: ignore[arg-type]
            self.assert_subject_failure(
                lambda: self.subject.encode_result_json({"payload": "x" * 128}),
                category="resource", exit_code=6,
            )

    def test_write_atomic_result_delegates_to_exclusive_helper_and_preserves_existing_output(self) -> None:
        subject = self.subject

        class ExclusiveHelper:
            def __init__(self) -> None:
                self.calls: list[tuple[Path, dict[str, object]]] = []

            def write_atomic_json(self, output: Path, value: dict[str, object]) -> None:
                output = Path(output)
                self.calls.append((output, value))
                if output.exists() or output.is_symlink():
                    raise FileExistsError(output)
                output.write_bytes(subject.encode_result_json(value))

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            output = root / "result.json"
            value = {"status": "complete", "caseCount": 1}
            helper = ExclusiveHelper()
            subject.write_atomic_result(output, value, helper)
            self.assertEqual(output.read_bytes(), subject.encode_result_json(value))
            self.assertEqual(helper.calls, [(output, value)])

            original = b'{"status":"prior"}\n'
            output.write_bytes(original)
            self.assert_subject_failure(
                lambda: subject.write_atomic_result(output, value, helper),
            )
            self.assertEqual(output.read_bytes(), original)

    def test_atomic_publication_does_not_reopen_output_after_helper_returns(self) -> None:
        subject = self.subject

        class CommitOnlyHelper:
            def __init__(self) -> None:
                self.calls = 0

            def write_atomic_json(self, output: Path, value: dict[str, object]) -> None:
                self.calls += 1
                with Path(output).open("wb") as stream:
                    stream.write(subject.encode_result_json(value))

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory).resolve() / "result.json"
            value = {"status": "complete", "caseCount": 1}
            helper = CommitOnlyHelper()
            with mock.patch.object(
                Path, "read_bytes", side_effect=AssertionError("post-commit read"),
            ):
                subject.write_atomic_result(output, value, helper)
            self.assertEqual(helper.calls, 1)
            self.assertEqual(
                output.read_bytes(), subject.encode_result_json(value),
            )


if __name__ == "__main__":
    unittest.main()
