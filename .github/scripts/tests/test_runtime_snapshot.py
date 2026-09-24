import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("runtime_snapshot", SCRIPTS / "runtime_snapshot.py")
SNAPSHOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SNAPSHOT)


class RuntimeSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "bundle"

    def test_collect_builds_before_install_without_adding_upgrade_or_constraints(self):
        commands = []

        def pip(arguments):
            commands.append(arguments)
            if "--report" in arguments:
                Path(arguments[arguments.index("--report") + 1]).write_text(
                    json.dumps({"install": [{"download_info": {"url": "file:///fixture.whl"}}]})
                )
            elif arguments[0] == "wheel":
                (self.root / "0" / "fixture.whl").touch()

        with mock.patch.object(SNAPSHOT, "run_pip", side_effect=pip):
            SNAPSHOT.collect(
                self.root,
                [
                    "fixture",
                    "--no-deps",
                    "--ignore-requires-python",
                    "--break-system-packages",
                    "--index-url",
                    "https://example.invalid/simple",
                    "--no-cache-dir",
                ],
            )
        self.assertEqual([command[0] for command in commands], ["install", "wheel", "install"])
        self.assertIn("--dry-run", commands[0])
        self.assertIn("--no-deps", commands[0])
        self.assertIn("https://example.invalid/simple", commands[1])
        self.assertNotIn("--break-system-packages", commands[1])
        self.assertIn("--break-system-packages", commands[2])
        self.assertIn("--no-index", commands[2])
        self.assertIn("--no-deps", commands[2])
        for command in commands:
            for flag in ("--upgrade", "--force-reinstall", "--constraint", "--no-cache-dir"):
                self.assertNotIn(flag, command)

    def test_editable_source_replacement_is_preserved_for_its_batch(self):
        def pip(arguments):
            if "--report" in arguments:
                Path(arguments[arguments.index("--report") + 1]).write_text(
                    json.dumps(
                        {
                            "install": [
                                {
                                    "download_info": {
                                        "url": "file:///source",
                                        "dir_info": {"editable": True},
                                    }
                                }
                            ]
                        }
                    )
                )
            elif arguments[0] == "wheel":
                (self.root / "0/source.whl").touch()

        with mock.patch.object(SNAPSHOT, "run_pip", side_effect=pip) as run:
            SNAPSHOT.collect(self.root, ["-e", "/source"])
        self.assertIn("--force-reinstall", run.call_args.args[0])

    def test_satisfied_dependencies_do_not_build_or_install(self):
        def pip(arguments):
            Path(arguments[arguments.index("--report") + 1]).write_text('{"install": []}')

        with mock.patch.object(SNAPSHOT, "run_pip", side_effect=pip) as run:
            SNAPSHOT.collect(self.root, ["already-installed"])
        self.assertEqual(run.call_count, 1)
        self.assertFalse((self.root / "install.json").exists())

    def test_source_references_use_resolved_commit_and_hash(self):
        self.assertEqual(
            SNAPSHOT.wheel_source(
                {
                    "download_info": {
                        "url": "https://example/repo",
                        "vcs_info": {"vcs": "git", "commit_id": "abc"},
                        "subdirectory": "pkg",
                    }
                }
            ),
            "git+https://example/repo@abc#subdirectory=pkg",
        )
        self.assertEqual(
            SNAPSHOT.wheel_source(
                {
                    "download_info": {
                        "url": "https://example/pkg.whl",
                        "archive_info": {"hashes": {"sha256": "abc"}},
                    }
                }
            ),
            "https://example/pkg.whl#sha256=abc",
        )
        self.assertEqual(
            SNAPSHOT.wheel_source(
                {"download_info": {"url": "file:///source", "dir_info": {"editable": True}}}
            ),
            "file:///source",
        )

    def test_prepare_only_copies_te_wheel_and_publishes_name(self):
        te = self.root.parent / "te"
        te.mkdir()
        (te / "transformer_engine-1.whl").write_bytes(b"existing-wheel")
        output = self.root.parent / "output"
        with (
            mock.patch.dict(
                os.environ,
                {
                    "TE_FL_CACHE_KEY": "te-key",
                    "TE_FL_WHEEL_DIR": str(te),
                    "GITHUB_OUTPUT": str(output),
                    "CI_UNIT_SNAPSHOT_NAME": "per-run-bundle",
                },
            ),
            mock.patch.object(SNAPSHOT, "run_pip") as pip,
        ):
            SNAPSHOT.prepare(self.root)
        pip.assert_not_called()
        self.assertEqual(
            (self.root / "te-fl/transformer_engine-1.whl").read_bytes(), b"existing-wheel"
        )
        self.assertEqual(output.read_text(), "name=per-run-bundle\n")

    def test_install_replays_order_and_does_not_change_later_pip_settings(self):
        self.root.mkdir()
        batches = [
            {"args": [], "wheels": ["0/first.whl"]},
            {"args": ["--ignore-requires-python"], "wheels": ["1/second.whl"]},
        ]
        (self.root / "install.json").write_text(json.dumps(batches))
        env_file = self.root.parent / "env"
        with mock.patch.dict(os.environ, {"GITHUB_ENV": str(env_file), "TE_FL_CACHE_KEY": ""}):
            with mock.patch.object(SNAPSHOT, "run_pip") as pip:
                SNAPSHOT.install(self.root)
            self.assertEqual(pip.call_args_list[0].args[0][-1], str(self.root / "0/first.whl"))
            self.assertEqual(pip.call_args_list[1].args[0][-1], str(self.root / "1/second.whl"))
            self.assertEqual(env_file.read_text(), "CI_UNIT_SNAPSHOT_READY=true\n")
            env_file.unlink()
            with mock.patch.object(SNAPSHOT, "run_pip", side_effect=RuntimeError("install failed")):
                with self.assertRaises(RuntimeError):
                    SNAPSHOT.install(self.root)
            self.assertFalse(env_file.exists())


class RuntimeHelperTests(unittest.TestCase):
    def run_helper(self, script, **environment):
        result = subprocess.run(
            ["bash", "-c", 'source "$HELPER"; ' + script],
            env={**os.environ, "HELPER": str(SCRIPTS / "set_env_common.sh"), **environment},
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_producer_uses_collection_and_other_modes_keep_original_pip_arguments(self):
        for suite, directory, expected in (
            ("unit", "/bundle", "collect"),
            ("unit", "", "install"),
            ("functional", "/bundle", "install"),
        ):
            with self.subTest(suite=suite, directory=directory):
                output = self.run_helper(
                    'mock_python() { printf "%s\\n" "$@"; }; ci_install_unit_packages --no-cache-dir example',
                    CI_TEST_SUITE=suite,
                    CI_UNIT_SNAPSHOT_DIRECTORY=directory,
                    CI_PYTHON_BIN="mock_python",
                    CI_UNIT_SNAPSHOT_READY="false",
                )
                self.assertIn(expected, output)
                self.assertIn("--no-cache-dir", output)
                self.assertNotIn("--upgrade", output)

    def test_runtime_packages_do_not_gain_upgrade_flags(self):
        output = self.run_helper(
            'ci_install_unit_packages() { printf "%s\\n" "$@"; }; ci_install_runtime_packages',
            CI_TEST_SUITE="unit",
            CI_UNIT_SNAPSHOT_DIRECTORY="/bundle",
            CI_UNIT_SNAPSHOT_READY="false",
            CI_PYTHON_BIN=sys.executable,
            CI_RUNTIME_PIP_PACKAGES_JSON='["flag-gems"]',
            CI_RUNTIME_PIP_INSTALL_ARGS_JSON='["--ignore-requires-python"]',
        )
        self.assertIn("flag-gems", output)
        self.assertIn("--ignore-requires-python", output)
        self.assertNotIn("--upgrade", output)

    def test_ready_consumers_skip_dependency_resolution(self):
        self.run_helper(
            "ci_install_unit_packages example; ci_install_runtime_packages",
            CI_TEST_SUITE="unit",
            CI_UNIT_SNAPSHOT_READY="true",
            CI_PYTHON_BIN="/does-not-exist",
            CI_RUNTIME_PIP_PACKAGES_JSON="invalid",
        )


class WheelBundleIntegrationTests(unittest.TestCase):
    def test_producer_and_consumer_reuse_wheels_without_upgrading_satisfied_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = {
                **os.environ,
                "CI_UNIT_SNAPSHOT_DIRECTORY": str(root / "bundle"),
                "CI_UNIT_SNAPSHOT_NAME": "bundle",
                "GITHUB_OUTPUT": str(root / "output"),
                "GITHUB_ENV": str(root / "env"),
                "TE_FL_CACHE_KEY": "",
                "PIP_CONFIG_FILE": os.devnull,
                "PIP_NO_INDEX": "1",
                "PIP_INDEX_URL": "http://127.0.0.1:1/simple",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "CI_TEST_SUITE": "unit",
                "CI_UNIT_SNAPSHOT_READY": "false",
            }
            environment.pop("PIP_CONSTRAINT", None)
            environment.pop("PYTHONPATH", None)

            def run(arguments):
                result = subprocess.run(
                    arguments, cwd=root, env=environment, capture_output=True, text=True, timeout=45
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout.strip()

            for version in ("1.0", "2.0"):
                wheel = root / f"bundle_fixture-{version}-py3-none-any.whl"
                info = f"bundle_fixture-{version}.dist-info"
                files = {
                    "bundle_fixture.py": f'VERSION = "{version}"\n',
                    f"{info}/METADATA": f"Metadata-Version: 2.1\nName: bundle-fixture\nVersion: {version}\n",
                    f"{info}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
                }
                files[f"{info}/RECORD"] = (
                    "".join(f"{name},,\n" for name in files) + f"{info}/RECORD,,\n"
                )
                with zipfile.ZipFile(wheel, "w") as archive:
                    for name, content in files.items():
                        archive.writestr(name, content)
            for name in ("producer", "consumer"):
                run([sys.executable, "-m", "venv", str(root / name)])
                run(
                    [
                        str(root / name / "bin/python"),
                        "-m",
                        "pip",
                        "install",
                        "--no-deps",
                        str(root / "bundle_fixture-1.0-py3-none-any.whl"),
                    ]
                )
            producer = str(root / "producer/bin/python")
            consumer = str(root / "consumer/bin/python")
            environment["CI_PYTHON_BIN"] = producer

            def collect(requirement):
                run(
                    [
                        "bash",
                        "-c",
                        'source "$1"; ci_install_unit_packages "$2" --find-links "$3" --no-deps --no-cache-dir',
                        "test",
                        str(SCRIPTS / "set_env_common.sh"),
                        requirement,
                        str(root),
                    ]
                )

            collect("bundle-fixture")
            self.assertFalse((root / "bundle/install.json").exists())
            collect("bundle-fixture==2.0")
            collect("bundle-fixture==2.0")
            self.assertEqual(len(json.loads((root / "bundle/install.json").read_text())), 1)
            script = str(SCRIPTS / "runtime_snapshot.py")
            run([producer, script, "prepare"])
            run([consumer, script, "install"])
            for python in (producer, consumer):
                self.assertEqual(
                    run([python, "-c", "import bundle_fixture; print(bundle_fixture.VERSION)"]),
                    "2.0",
                )
            self.assertEqual((root / "env").read_text(), "CI_UNIT_SNAPSHOT_READY=true\n")


if __name__ == "__main__":
    unittest.main()
