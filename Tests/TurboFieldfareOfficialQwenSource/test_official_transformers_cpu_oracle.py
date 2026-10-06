"""Weight-free tests for the standalone official Safetensors reader.

The fixtures contain only synthetic metadata and twelve payload bytes. They do
not read checkpoint shards or load model code. Expected values are literals
specified here, not values produced by the implementation under test.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import struct
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = (
    REPO_ROOT
    / "scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/reference"
)
HELPER_PATH = REFERENCE_DIR / "official_safetensors_reader.py"
SHARD_NAME = "model-00001-of-00001.safetensors"
INDEX_NAME = "model.safetensors.index.json"

# Independent synthetic values: F32[2] is bytes [0, 8), then BF16[2] is
# bytes [8, 12). These bytes are never interpreted as candidate output.
PAYLOAD = bytes.fromhex("0000803f00000040013f0240")
INDEX_MAP = {
    "layer.0.weight": SHARD_NAME,
    "layer.0.bias": SHARD_NAME,
}
INDEX_OBJECT = {
    "metadata": {"total_size": 12},
    "weight_map": INDEX_MAP,
}
HEADER_OBJECT = {
    "layer.0.weight": {
        "dtype": "F32",
        "shape": [2],
        "data_offsets": [0, 8],
    },
    "layer.0.bias": {
        "dtype": "BF16",
        "shape": [2],
        "data_offsets": [8, 12],
    },
    "__metadata__": {"format": "pt"},
}


def _load_helper():
    """Import the separately owned helper by its repository path."""
    reference_path = str(REFERENCE_DIR)
    if reference_path not in sys.path:
        sys.path.insert(0, reference_path)
    spec = importlib.util.spec_from_file_location(
        "official_safetensors_reader", HELPER_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import helper at {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _write_index(
    root: Path,
    *,
    raw: bytes | None = None,
    value: dict[str, Any] | None = None,
) -> Path:
    path = root / INDEX_NAME
    if raw is None:
        raw = _json_bytes(INDEX_OBJECT if value is None else value)
    path.write_bytes(raw)
    return path


def _write_shard(
    root: Path,
    *,
    header: dict[str, Any] | None = None,
    raw_header: bytes | None = None,
    payload: bytes = PAYLOAD,
    declared_header_length: int | None = None,
    filename: str = SHARD_NAME,
) -> Path:
    path = root / filename
    encoded_header = _json_bytes(HEADER_OBJECT if header is None else header)
    if raw_header is not None:
        encoded_header = raw_header
    header_length = (
        len(encoded_header)
        if declared_header_length is None
        else declared_header_length
    )
    path.write_bytes(struct.pack("<Q", header_length) + encoded_header + payload)
    return path


def _record(module, shard_path: Path, tensor_name: str = "layer.0.weight"):
    return module.read_header(shard_path)[tensor_name]


class OfficialSafetensorsReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = _load_helper()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def _assert_format_error(self, operation: Callable[[], Any]):
        with self.assertRaises(self.reader.SourceFormatError):
            operation()

    def test_load_index_returns_literal_tensor_to_basename_mapping(self):
        _write_index(self.root)
        self.assertEqual(self.reader.load_index(self.root), INDEX_MAP)

    def test_load_index_checks_expected_sha256_over_exact_index_bytes(self):
        raw = _json_bytes(INDEX_OBJECT)
        _write_index(self.root, raw=raw)
        expected = hashlib.sha256(raw).hexdigest()
        self.assertEqual(
            self.reader.load_index(self.root, expected_sha256=expected), INDEX_MAP
        )
        with self.assertRaises(self.reader.SourceFormatError):
            self.reader.load_index(self.root, expected_sha256="0" * 64)

    def test_load_index_rejects_malformed_json(self):
        for raw in (b'{"metadata":', _json_bytes(INDEX_OBJECT) + b" false"):
            with self.subTest(raw=raw):
                _write_index(self.root, raw=raw)
                self._assert_format_error(lambda: self.reader.load_index(self.root))

    def test_load_index_rejects_duplicate_top_level_metadata_in_both_orders(self):
        shard = SHARD_NAME.encode("ascii")
        variants = (
            b'{"metadata":{"total_size":12},"metadata":{"total_size":12},'
            b'"weight_map":{"x":"' + shard + b'"}}',
            b'{"weight_map":{"x":"' + shard
            + b'"},"metadata":{"total_size":12},'
            b'"metadata":{"total_size":12}}',
        )
        for raw in variants:
            with self.subTest(raw=raw):
                _write_index(self.root, raw=raw)
                self._assert_format_error(lambda: self.reader.load_index(self.root))

    def test_load_index_rejects_duplicate_metadata_fields(self):
        shard = SHARD_NAME.encode("ascii")
        raw = (
            b'{"metadata":{"total_size":12,"total_size":13},'
            b'"weight_map":{"x":"' + shard + b'"}}'
        )
        _write_index(self.root, raw=raw)
        self._assert_format_error(lambda: self.reader.load_index(self.root))

    def test_load_index_rejects_duplicate_weight_map_names(self):
        shard = SHARD_NAME.encode("ascii")
        raw = (
            b'{"metadata":{"total_size":12},"weight_map":{"x":"'
            + shard + b'","x":"' + shard + b'"}}'
        )
        _write_index(self.root, raw=raw)
        self._assert_format_error(lambda: self.reader.load_index(self.root))

    def test_load_index_requires_metadata_total_size_and_nonempty_weight_map(self):
        invalid_indexes = (
            {"weight_map": INDEX_MAP},
            {"metadata": {"total_size": 12}},
            {"metadata": {}, "weight_map": INDEX_MAP},
            {"metadata": {"total_size": 12}, "weight_map": {}},
        )
        for value in invalid_indexes:
            with self.subTest(value=value):
                _write_index(self.root, value=value)
                self._assert_format_error(lambda: self.reader.load_index(self.root))

    def test_load_index_rejects_unsafe_shard_names(self):
        for shard_name in (
            "../outside.safetensors",
            "nested/shard.safetensors",
            "/tmp/outside.safetensors",
            "..",
            "_leading-underscore.safetensors",
        ):
            with self.subTest(shard_name=shard_name):
                value = {
                    "metadata": {"total_size": 12},
                    "weight_map": {"tensor": shard_name},
                }
                _write_index(self.root, value=value)
                self._assert_format_error(lambda: self.reader.load_index(self.root))

    def test_load_index_does_not_open_shard_payloads_or_require_their_presence(self):
        value = {
            "metadata": {"total_size": 4},
            "weight_map": {"tensor": "not-present.safetensors"},
        }
        _write_index(self.root, value=value)
        self.assertEqual(
            self.reader.load_index(self.root),
            {"tensor": "not-present.safetensors"},
        )

    def test_read_json_file_checks_correct_and_wrong_expected_sha256(self):
        raw = b'{"revision":"fixture-r1","count":2}'
        path = self.root / "small-metadata.json"
        path.write_bytes(raw)
        expected = hashlib.sha256(raw).hexdigest()
        self.assertEqual(
            self.reader.read_json_file(
                path, max_bytes=len(raw), expected_sha256=expected
            ),
            {"revision": "fixture-r1", "count": 2},
        )
        with self.assertRaises(self.reader.SourceFormatError):
            self.reader.read_json_file(
                path, max_bytes=len(raw), expected_sha256="f" * 64
            )
        alternate_encoding = b'{ "revision" : "fixture-r1", "count" : 2 }'
        path.write_bytes(alternate_encoding)
        with self.assertRaises(self.reader.SourceFormatError):
            self.reader.read_json_file(
                path, max_bytes=len(alternate_encoding), expected_sha256=expected
            )

    def test_read_json_file_rejects_symlink_without_following_it(self):
        target = self.root / "metadata-target.json"
        target.write_bytes(b'{"source":"synthetic"}')
        link = self.root / "metadata-link.json"
        link.symlink_to(target)
        with self.assertRaises(OSError):
            self.reader.read_json_file(link, max_bytes=1024)

    def test_read_json_file_keeps_opened_inode_if_path_is_replaced_during_read(self):
        path = self.root / "metadata.json"
        original = b'{"identity":"opened-file"}'
        replacement = b'{"identity":"replacement-file"}'
        replacement_path = self.root / "replacement.json"
        path.write_bytes(original)
        replacement_path.write_bytes(replacement)
        swapped = False

        def replace_then_read(fd: int, size: int, offset: int) -> bytes:
            nonlocal swapped
            if not swapped:
                os.replace(replacement_path, path)
                swapped = True
            return os.pread(fd, size, offset)

        parsed = self.reader.read_json_file(
            path,
            max_bytes=len(original),
            expected_sha256=hashlib.sha256(original).hexdigest(),
            read_at=replace_then_read,
        )
        self.assertTrue(swapped)
        self.assertEqual(parsed, {"identity": "opened-file"})
        self.assertEqual(path.read_bytes(), replacement)

    def test_read_header_returns_literal_dtype_shape_and_absolute_byte_bounds(self):
        shard = _write_shard(self.root)
        records = self.reader.read_header(shard)
        data_start = 8 + len(_json_bytes(HEADER_OBJECT))
        weight = records["layer.0.weight"]
        bias = records["layer.0.bias"]
        self.assertEqual((weight.name, weight.dtype, weight.shape),
                         ("layer.0.weight", "F32", (2,)))
        self.assertEqual((weight.start, weight.end), (data_start, data_start + 8))
        self.assertEqual((bias.name, bias.dtype, bias.shape),
                         ("layer.0.bias", "BF16", (2,)))
        self.assertEqual((bias.start, bias.end),
                         (data_start + 8, data_start + 12))
        self.assertEqual(weight.nbytes, 8)

    def test_read_header_accepts_injected_short_reads(self):
        shard = _write_shard(self.root)
        requests: list[tuple[int, int]] = []

        def short_read(fd: int, size: int, offset: int) -> bytes:
            requests.append((size, offset))
            return os.pread(fd, min(size, 2), offset)

        records = self.reader.read_header(shard, read_at=short_read)
        self.assertEqual(records["layer.0.weight"].nbytes, 8)
        expected_header = _json_bytes(HEADER_OBJECT)
        expected_requests = [
            (8 - offset, offset) for offset in range(0, 8, 2)
        ] + [
            (len(expected_header) - offset, 8 + offset)
            for offset in range(0, len(expected_header), 2)
        ]
        self.assertEqual(requests, expected_requests)
        self.assertTrue(all(size <= 16 * 1024 * 1024 for size, _ in requests))

    def test_read_header_honors_explicit_header_cap(self):
        shard = _write_shard(self.root)
        encoded_header_size = len(_json_bytes(HEADER_OBJECT))
        exact_limit = self.reader.read_header(
            shard, max_header_bytes=encoded_header_size
        )
        self.assertEqual(exact_limit["layer.0.bias"].nbytes, 4)
        self._assert_format_error(
            lambda: self.reader.read_header(
                shard, max_header_bytes=encoded_header_size - 1
            )
        )

    def test_read_header_rejects_truncated_length_prefix_and_body(self):
        prefix_only = self.root / SHARD_NAME
        prefix_only.write_bytes(b"\x00" * 7)
        self._assert_format_error(lambda: self.reader.read_header(prefix_only))

        truncated_body = self.root / SHARD_NAME
        truncated_body.write_bytes(struct.pack("<Q", 5) + b"{}")
        self._assert_format_error(lambda: self.reader.read_header(truncated_body))

    def test_read_header_rejects_oversized_declared_header_before_body_read(self):
        path = self.root / SHARD_NAME
        path.write_bytes(struct.pack("<Q", 16 * 1024 * 1024 + 1) + b"xx")
        requests: list[tuple[int, int]] = []

        def spy(fd: int, size: int, offset: int) -> bytes:
            requests.append((size, offset))
            return os.pread(fd, size, offset)

        self._assert_format_error(
            lambda: self.reader.read_header(path, read_at=spy)
        )
        self.assertEqual(requests, [(8, 0)])

    def test_read_header_rejects_malformed_json(self):
        cases = {
            "incomplete-object": b'{"x":',
            "trailing-comma": b'{"x":{},}',
            "trailing-content": b'{} false',
            "bad-escape": b'{"x":"\\q"}',
        }
        for name, raw in cases.items():
            with self.subTest(name=name):
                path = _write_shard(self.root, raw_header=raw)
                self._assert_format_error(lambda: self.reader.read_header(path))

    def test_read_header_rejects_duplicate_tensor_and_metadata_json_members(self):
        cases = {
            "duplicate-tensor": (
                b'{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},'
                b'"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
            ),
            "escaped-equivalent-tensor": (
                b'{"layer.0.weight":{"dtype":"F32","shape":[2],'
                b'"data_offsets":[0,8]},"layer.0.w\\u0065ight":'
                b'{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
            ),
            "duplicate-metadata-member": (
                b'{"__metadata__":{"format":"pt","format":"numpy"},'
                b'"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
            ),
        }
        for name, raw in cases.items():
            with self.subTest(name=name):
                path = _write_shard(self.root, raw_header=raw)
                self._assert_format_error(lambda: self.reader.read_header(path))

    def test_read_header_rejects_nonobject_or_empty_tensor_inventory(self):
        for raw in (b"[]", b"null", b"{}"):
            with self.subTest(raw=raw):
                path = _write_shard(self.root, raw_header=raw)
                self._assert_format_error(lambda: self.reader.read_header(path))

    def test_read_header_rejects_invalid_offsets(self):
        for offsets in ([0, -1], [5, 4], [0, 13], [-1, 4]):
            with self.subTest(offsets=offsets):
                header = {
                    "x": {"dtype": "F32", "shape": [2],
                          "data_offsets": offsets}
                }
                path = _write_shard(self.root, header=header)
                self._assert_format_error(lambda: self.reader.read_header(path))

    def test_read_header_rejects_overlapping_and_gapped_intervals(self):
        cases = (
            {
                "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                "b": {"dtype": "BF16", "shape": [2], "data_offsets": [4, 8]},
            },
            {
                "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                "b": {"dtype": "BF16", "shape": [2], "data_offsets": [5, 9]},
            },
        )
        for header in cases:
            with self.subTest(header=header):
                path = _write_shard(self.root, header=header)
                self._assert_format_error(lambda: self.reader.read_header(path))

    def test_read_header_rejects_trailing_unindexed_payload(self):
        path = _write_shard(self.root, payload=PAYLOAD + b"x")
        self._assert_format_error(lambda: self.reader.read_header(path))

    def test_read_header_rejects_unknown_dtype_and_invalid_dimensions(self):
        cases = (
            ("F17", [2], [0, 8]),
            ("F32", [-1], [0, 8]),
            ("F32", [2.0], [0, 8]),
            ("F32", [True], [0, 8]),
            ("F32", "2", [0, 8]),
        )
        for dtype, shape, offsets in cases:
            with self.subTest(dtype=dtype, shape=shape):
                header = {"x": {"dtype": dtype, "shape": shape,
                                "data_offsets": offsets}}
                path = _write_shard(self.root, header=header)
                self._assert_format_error(lambda: self.reader.read_header(path))

    def test_read_header_accepts_scalar_and_rejects_shape_byte_size_mismatch(self):
        scalar = {
            "scalar": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}
        }
        path = _write_shard(self.root, header=scalar, payload=PAYLOAD[:4])
        self.assertEqual(self.reader.read_header(path)["scalar"].nbytes, 4)

        mismatch = {
            "x": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 4]}
        }
        path = _write_shard(self.root, header=mismatch)
        self._assert_format_error(lambda: self.reader.read_header(path))

    def test_iter_tensor_chunks_reads_exact_requested_ranges_with_chunk_cap(self):
        shard = _write_shard(self.root)
        record = _record(self.reader, shard)
        requests: list[tuple[int, int]] = []

        def spy(fd: int, size: int, offset: int) -> bytes:
            requests.append((size, offset))
            return os.pread(fd, size, offset)

        chunks = [
            bytes(chunk)
            for chunk in self.reader.iter_tensor_chunks(
                shard, record, chunk_bytes=3, read_at=spy
            )
        ]
        expected_start = 8 + len(_json_bytes(HEADER_OBJECT))
        self.assertEqual(chunks, [PAYLOAD[:3], PAYLOAD[3:6], PAYLOAD[6:8]])
        self.assertEqual(
            requests,
            [(3, expected_start), (3, expected_start + 3),
             (2, expected_start + 6)],
        )
        self.assertTrue(all(0 < size <= 3 for size, _ in requests))

    def test_iter_tensor_chunks_rejects_record_from_another_shard(self):
        first = _write_shard(self.root, filename=SHARD_NAME)
        record = _record(self.reader, first)
        second = _write_shard(self.root, filename="model-00002-of-00002.safetensors")
        with self.assertRaises(self.reader.SourceReadError):
            list(self.reader.iter_tensor_chunks(second, record))

    def test_iter_tensor_chunks_rejects_out_of_file_record_bounds(self):
        shard = _write_shard(self.root)
        record = _record(self.reader, shard)
        forged = replace(record, end=record.file_size + 1)
        self._assert_format_error(
            lambda: list(self.reader.iter_tensor_chunks(shard, forged))
        )

    def test_iter_tensor_chunks_rejects_nonpositive_memory_budget(self):
        shard = _write_shard(self.root)
        record = _record(self.reader, shard)
        for chunk_bytes in (0, -1):
            with self.subTest(chunk_bytes=chunk_bytes):
                self._assert_format_error(
                    lambda: list(self.reader.iter_tensor_chunks(
                        shard, record, chunk_bytes=chunk_bytes
                    ))
                )

    def test_iter_tensor_chunks_reports_short_payload_read(self):
        shard = _write_shard(self.root)
        record = _record(self.reader, shard)
        called = False

        def short_read(fd: int, size: int, offset: int) -> bytes:
            nonlocal called
            called = True
            return b""

        with self.assertRaises(self.reader.SourceReadError):
            list(self.reader.iter_tensor_chunks(
                shard, record, chunk_bytes=3, read_at=short_read
            ))
        self.assertTrue(called)


if __name__ == "__main__":
    unittest.main()
