#!/usr/bin/env python3
"""Standard-library tests for the independent Qwen 3.6 reference seams.

Set QWEN36_QUANTIZED_REFERENCE_SCRIPT to target a staged or alternate script.
After promotion the default resolves to the repository's Scripts directory.
The test module and the reference script are the only imports performed here.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


SCRIPT_PATH_ENVIRONMENT = "QWEN36_QUANTIZED_REFERENCE_SCRIPT"
REFERENCE_MODULE_NAME = "_qwen36_quantized_reference_unit_subject"
REFERENCE_SCRIPT_NAME = "qwen36_quantized_reference.py"


def reference_script_path() -> Path:
    explicit = os.environ.get(SCRIPT_PATH_ENVIRONMENT)
    if explicit:
        return Path(explicit).expanduser().resolve()
    # staging/Tests/Python and the promoted repository Tests/Python both have
    # the script three directory levels above this file.
    return (Path(__file__).resolve().parents[2] / "Scripts" /
            REFERENCE_SCRIPT_NAME)


def load_reference_module() -> types.ModuleType:
    script = reference_script_path()
    spec = importlib.util.spec_from_file_location(REFERENCE_MODULE_NAME, script)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load reference script at {script}")
    subject = importlib.util.module_from_spec(spec)
    sys.modules[REFERENCE_MODULE_NAME] = subject
    spec.loader.exec_module(subject)
    return subject


class ReferenceUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subject = load_reference_module()

    @classmethod
    def tearDownClass(cls) -> None:
        if sys.modules.get(REFERENCE_MODULE_NAME) is cls.subject:
            del sys.modules[REFERENCE_MODULE_NAME]

    def assert_reference_failure(
        self, operation: object, *, category: str, exit_code: int,
    ) -> object:
        with self.assertRaises(self.subject.ReferenceFailure) as raised:
            operation()  # type: ignore[operator]
        failure = raised.exception
        self.assertEqual(failure.category, category)
        self.assertEqual(failure.exit_code, exit_code)
        return failure

    @staticmethod
    def assert_no_temporary_links(directory: Path, output: Path) -> None:
        temporary_links = list(directory.glob(f".{output.name}.*.tmp"))
        if temporary_links:
            raise AssertionError(f"temporary publication links remain: {temporary_links}")

    def test_write_atomic_json_publishes_canonical_document_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            output = directory / "result.json"
            value = {"status": "complete", "count": 2}

            self.subject.write_atomic_json(output, value)

            self.assertEqual(output.read_bytes(), b'{"count":2,"status":"complete"}\n')
            self.assert_no_temporary_links(directory, output)

    def test_write_atomic_json_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            output = directory / "result.json"
            original = b'{"status":"prior"}\n'
            output.write_bytes(original)

            self.assert_reference_failure(
                lambda: self.subject.write_atomic_json(
                    output, {"status": "complete"}),
                category="outputWrite", exit_code=9,
            )

            self.assertEqual(output.read_bytes(), original)
            self.assert_no_temporary_links(directory, output)

    def test_write_atomic_json_preserves_raced_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            output = directory / "result.json"
            raced = b'{"status":"raced"}\n'

            def publish_race(_temporary: Path, destination: Path) -> None:
                destination.write_bytes(raced)
                raise FileExistsError("simulated concurrent publisher")

            with mock.patch.object(
                self.subject.os, "link", side_effect=publish_race,
            ):
                self.assert_reference_failure(
                    lambda: self.subject.write_atomic_json(
                        output, {"status": "complete"}),
                    category="outputWrite", exit_code=9,
                )

            self.assertEqual(output.read_bytes(), raced)
            self.assert_no_temporary_links(directory, output)

    def test_write_atomic_json_rolls_back_after_directory_fsync_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            output = directory / "result.json"
            real_fsync = self.subject.os.fsync
            calls = 0

            def fail_publication_fsync(descriptor: int) -> None:
                nonlocal calls
                calls += 1
                # The first fsync is the temporary file. The second is the
                # parent directory after the exclusive hard link is created.
                if calls == 2:
                    raise OSError("simulated publication directory fsync failure")
                real_fsync(descriptor)

            with mock.patch.object(
                self.subject.os, "fsync", side_effect=fail_publication_fsync,
            ):
                self.assert_reference_failure(
                    lambda: self.subject.write_atomic_json(
                        output, {"status": "complete"}),
                    category="outputWrite", exit_code=9,
                )

            self.assertFalse(output.exists())
            self.assert_no_temporary_links(directory, output)

    def test_write_atomic_json_rolls_back_after_temporary_unlink_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            output = directory / "result.json"
            real_unlink = self.subject.Path.unlink
            failed_once = False

            def fail_temporary_unlink(
                path: Path, *arguments: object, **keywords: object,
            ) -> None:
                nonlocal failed_once
                is_temporary = (
                    path.parent == directory and
                    path.name.startswith(f".{output.name}.") and
                    path.suffix == ".tmp"
                )
                if is_temporary and not failed_once:
                    failed_once = True
                    raise OSError("simulated temporary-link unlink failure")
                real_unlink(path, *arguments, **keywords)

            with mock.patch.object(
                self.subject.Path, "unlink", new=fail_temporary_unlink,
            ):
                self.assert_reference_failure(
                    lambda: self.subject.write_atomic_json(
                        output, {"status": "complete"}),
                    category="outputWrite", exit_code=9,
                )

            self.assertTrue(failed_once)
            self.assertFalse(output.exists())
            self.assert_no_temporary_links(directory, output)

    def test_write_atomic_json_cancellation_before_publication_has_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            output = directory / "result.json"

            def cancel_before_link(_temporary: Path, _destination: Path) -> None:
                raise self.subject.ReferenceCancelled("cancelled before publication")

            with mock.patch.object(
                self.subject.os, "link", side_effect=cancel_before_link,
            ):
                self.assert_reference_failure(
                    lambda: self.subject.write_atomic_json(
                        output, {"status": "complete"}),
                    category="cancelled", exit_code=8,
                )

            self.assertFalse(output.exists())
            self.assert_no_temporary_links(directory, output)

    def test_write_atomic_json_cancellation_after_publication_rolls_back_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            output = directory / "result.json"
            real_fsync = self.subject.os.fsync
            calls = 0

            def cancel_after_link(descriptor: int) -> None:
                nonlocal calls
                calls += 1
                # The second fsync is reached only after the output hard link
                # and the first publication audit have succeeded.
                if calls == 2:
                    raise self.subject.ReferenceCancelled(
                        "cancelled after publication")
                real_fsync(descriptor)

            with mock.patch.object(
                self.subject.os, "fsync", side_effect=cancel_after_link,
            ):
                self.assert_reference_failure(
                    lambda: self.subject.write_atomic_json(
                        output, {"status": "complete"}),
                    category="cancelled", exit_code=8,
                )

            self.assertFalse(output.exists())
            self.assert_no_temporary_links(directory, output)

    @staticmethod
    def parser_arguments() -> list[str]:
        return [
            "--model-directory", "model",
            "--official-config", "config.json",
            "--transformers-checkout", "transformers-checkout",
            "--request", "request.json",
            "--expected-manifest-sha256", "a" * 64,
            "--expected-policy-sha256", "b" * 64,
            "--torch-threads", "1",
            "--output", "result.json",
        ]

    def test_parse_arguments_requires_exactly_one_transformers_checkout(self) -> None:
        missing = self.parser_arguments()
        checkout_index = missing.index("--transformers-checkout")
        del missing[checkout_index:checkout_index + 2]
        failure = self.assert_reference_failure(
            lambda: self.subject.parse_arguments(missing),
            category="usage", exit_code=2,
        )
        self.assertIn("--transformers-checkout must appear exactly once", failure.detail)

        duplicate = self.parser_arguments()
        duplicate.extend(["--transformers-checkout", "second-checkout"])
        failure = self.assert_reference_failure(
            lambda: self.subject.parse_arguments(duplicate),
            category="usage", exit_code=2,
        )
        self.assertIn("--transformers-checkout must appear exactly once", failure.detail)

    def test_parse_arguments_records_explicit_transformers_checkout(self) -> None:
        parsed = self.subject.parse_arguments(self.parser_arguments())
        self.assertEqual(parsed.transformers_checkout, Path("transformers-checkout"))

    def test_select_transformers_source_prepends_only_selected_source_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            transformers, modules = self.make_transformers_fixture(
                root, ("configuration_utils.py",),
            )
            del transformers, modules
            source_root = (root / "src").resolve()
            source_path = str(source_root)
            fake_sys = types.SimpleNamespace(
                modules={},
                path=[source_path, "/other", source_path, "/last"],
            )
            git_output = self.make_git_output_stub(root)
            with mock.patch.object(self.subject, "sys", fake_sys), mock.patch.object(
                self.subject, "git_output", side_effect=git_output,
            ), mock.patch.object(
                self.subject.importlib, "invalidate_caches",
            ) as invalidate_caches:
                result = self.subject.select_transformers_source(root)

            self.assertEqual(
                result,
                (source_root, self.subject.TRANSFORMERS_COMMIT,
                 self.subject.TRANSFORMERS_TREE),
            )
            self.assertEqual(fake_sys.path, [source_path, "/other", "/last"])
            invalidate_caches.assert_called_once_with()

    def test_select_transformers_source_rejects_preloaded_transformers_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            transformers, modules = self.make_transformers_fixture(
                root, ("configuration_utils.py",),
            )
            del transformers, modules
            fake_sys = types.SimpleNamespace(
                modules={
                    "transformers": object(),
                    "transformers.configuration_utils": object(),
                },
                path=["/existing"],
            )
            failure = self.assert_reference_failure(
                lambda: self.run_selection_with_fake_sys(
                    root, fake_sys,
                ),
                category="environment", exit_code=3,
            )
            self.assertIn("imported before", failure.detail)
            self.assertEqual(fake_sys.path, ["/existing"])

    def run_selection_with_fake_sys(
        self, root: Path, fake_sys: types.SimpleNamespace,
    ):
        with mock.patch.object(self.subject, "sys", fake_sys), mock.patch.object(
            self.subject, "git_output", side_effect=self.make_git_output_stub(root),
        ):
            return self.subject.select_transformers_source(root)

    def make_transformers_fixture(
        self, root: Path, imported_relative_paths: tuple[str, ...],
    ) -> tuple[types.SimpleNamespace, dict[str, types.SimpleNamespace]]:
        repository = root
        package_root = repository / "src" / "transformers"
        package_root.mkdir(parents=True)
        (repository / ".git").mkdir()
        package_init = package_root / "__init__.py"
        package_init.write_text("# test package\n", encoding="utf-8")
        modules: dict[str, types.SimpleNamespace] = {
            "transformers": types.SimpleNamespace(
                __file__=str(package_init),
                __path__=(str(package_root),),
            ),
        }
        for index, relative_path in enumerate(imported_relative_paths):
            path = package_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# support module {index}\n", encoding="utf-8")
            module_name = f"transformers.{Path(relative_path).stem}"
            modules[module_name] = types.SimpleNamespace(
                __file__=str(path),
            )
        return modules["transformers"], modules

    def make_git_output_stub(
        self,
        repository: Path,
        *,
        expected_candidate: Path | None = None,
        show_toplevel: Path | None = None,
        commit: str | None = None,
        tree: str | None = None,
        tracked_status: bytes = b"",
        untracked: bytes = b"",
        tracked_files: bytes = b"tracked\n",
    ):
        expected_commit = commit or self.subject.TRANSFORMERS_COMMIT
        expected_tree = tree or self.subject.TRANSFORMERS_TREE
        candidate_root = expected_candidate or repository
        observed_root = show_toplevel or repository

        def git_output(candidate: Path, arguments: list[str]) -> bytes:
            self.assertEqual(candidate.resolve(), candidate_root.resolve())
            key = tuple(arguments)
            if key == ("rev-parse", "--show-toplevel"):
                return (str(observed_root.resolve()) + "\n").encode("utf-8")
            if key == ("rev-parse", "HEAD^{commit}"):
                return (expected_commit + "\n").encode("ascii")
            if key == ("rev-parse", "HEAD^{tree}"):
                return (expected_tree + "\n").encode("ascii")
            if key == (
                "status", "--porcelain=v1", "--untracked-files=no", "--",
                "src/transformers",
            ):
                return tracked_status
            if key == (
                "ls-files", "--others", "--exclude-standard", "--",
                "src/transformers",
            ):
                return untracked
            if key[:3] == ("ls-files", "--error-unmatch", "--"):
                return tracked_files
            raise AssertionError(f"unexpected git_output request: {arguments}")

        return git_output

    def run_validator(
        self,
        root: Path,
        transformers: types.SimpleNamespace,
        modules: dict[str, types.SimpleNamespace],
        git_output,
    ):
        fake_sys = types.SimpleNamespace(modules=modules)
        with mock.patch.object(self.subject, "sys", fake_sys), mock.patch.object(
            self.subject, "git_output", side_effect=git_output,
        ):
            return self.subject.validate_transformers_source_tree(transformers, root)

    def test_validate_transformers_checkout_rejects_non_root_checkout_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            transformers, modules = self.make_transformers_fixture(
                root, ("configuration_utils.py",),
            )
            del transformers, modules
            nested = root / "nested"
            nested.mkdir()
            with mock.patch.object(
                self.subject,
                "git_output",
                side_effect=self.make_git_output_stub(
                    root, expected_candidate=nested, show_toplevel=root,
                ),
            ):
                failure = self.assert_reference_failure(
                    lambda: self.subject.validate_transformers_checkout(nested),
                    category="environment", exit_code=3,
                )
            self.assertIn("not the Git checkout root", failure.detail)

    def test_validate_accepts_coherent_small_mocked_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            transformers, modules = self.make_transformers_fixture(
                root, ("configuration_utils.py", "modeling_utils.py"),
            )
            result = self.run_validator(
                root, transformers, modules,
                self.make_git_output_stub(root),
            )

            self.assertEqual(
                result,
                (self.subject.TRANSFORMERS_COMMIT, self.subject.TRANSFORMERS_TREE),
            )

    def test_validate_rejects_changed_support_module_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            transformers, modules = self.make_transformers_fixture(
                root, ("configuration_utils.py",),
            )
            support_module = root / "src" / "transformers" / "configuration_utils.py"
            support_module.write_text(
                support_module.read_text(encoding="utf-8") + "# changed support module\n",
                encoding="utf-8",
            )
            failure = self.assert_reference_failure(
                lambda: self.run_validator(
                    root, transformers, modules,
                    self.make_git_output_stub(
                        root,
                        tracked_status=b" M src/transformers/configuration_utils.py\n",
                    ),
                ),
                category="environment", exit_code=3,
            )
            self.assertIn("local modifications", failure.detail)

    def test_validate_rejects_wrong_commit_or_tree(self) -> None:
        for wrong_field in ("commit", "tree"):
            with self.subTest(wrong_field=wrong_field):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    transformers, modules = self.make_transformers_fixture(
                        root, ("configuration_utils.py",),
                    )
                    kwargs = {
                        "commit": self.subject.TRANSFORMERS_COMMIT,
                        "tree": self.subject.TRANSFORMERS_TREE,
                    }
                    kwargs[wrong_field] = "0" * 40
                    failure = self.assert_reference_failure(
                        lambda: self.run_validator(
                            root, transformers, modules,
                            self.make_git_output_stub(root, **kwargs),
                        ),
                        category="environment", exit_code=3,
                    )
                    self.assertIn("differs from the pinned", failure.detail)

    def test_validate_rejects_module_outside_supported_package_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            transformers, modules = self.make_transformers_fixture(
                root, ("configuration_utils.py",),
            )
            outside = root / "outside_support.py"
            outside.write_text("# unsupported module root\n", encoding="utf-8")
            modules["transformers.external"] = types.SimpleNamespace(
                __file__=str(outside),
            )

            failure = self.assert_reference_failure(
                lambda: self.run_validator(
                    root, transformers, modules,
                    self.make_git_output_stub(root),
                ),
                category="environment", exit_code=3,
            )
            self.assertIn("outside pinned sources", failure.detail)


if __name__ == "__main__":
    unittest.main()
