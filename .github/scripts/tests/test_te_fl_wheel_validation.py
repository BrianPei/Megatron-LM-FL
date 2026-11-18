import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "validate_te_fl_wheel.py"
SPEC = importlib.util.spec_from_file_location("validate_te_fl_wheel", SCRIPT)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)
COMMIT = "f7b8171d94d31e40b7822838b0a8c4d300125f4d"
VERSION = "2.17.0+f7b8171"


class WheelValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source"
        (self.source / "build_tools").mkdir(parents=True)
        (self.source / "build_tools" / "VERSION.txt").write_text("2.17.0\n")
        self.wheels = self.root / "wheels"
        self.wheels.mkdir()
        mock = patch.object(VALIDATOR.subprocess, "check_output", side_effect=[COMMIT, "f7b8171"])
        mock.start()
        self.addCleanup(mock.stop)

    def wheel(
        self,
        version=VERSION,
        filename_version=None,
        metadata_name="transformer_engine",
        metadata_version=None,
    ):
        name = f"transformer_engine-{filename_version or version}-py3-none-any.whl"
        path = self.wheels / name
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                f"transformer_engine-{version}.dist-info/METADATA",
                f"Metadata-Version: 2.1\nName: {metadata_name}\nVersion: {metadata_version or version}\n",
            )
        return path

    def validate(self):
        return VALIDATOR.validate_wheel(self.wheels, self.source, COMMIT)

    def test_matching_wheel_is_accepted_without_importing_package(self):
        wheel = self.wheel()
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr(
                "transformer_engine/__init__.py", "raise RuntimeError('must not import')"
            )
        self.assertEqual(self.validate(), wheel)

    def test_daily_hygon_wrong_commit_is_rejected(self):
        self.wheel(version="2.17.0+8f8a003")
        with self.assertRaisesRegex(ValueError, "does not match checkout"):
            self.validate()

    def test_renaming_stale_wheel_does_not_hide_wrong_metadata(self):
        self.wheel(version="2.17.0+8f8a003", filename_version=VERSION)
        with self.assertRaisesRegex(ValueError, "does not match checkout"):
            self.validate()

    def test_wrong_filename_is_rejected(self):
        self.wheel(filename_version="2.17.0+8f8a003")
        with self.assertRaisesRegex(ValueError, "filename"):
            self.validate()

    def test_missing_or_multiple_wheels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "found 0"):
            self.validate()
        self.wheel()
        self.wheel(version="2.17.0+8f8a003")
        with patch.object(VALIDATOR.subprocess, "check_output", side_effect=[COMMIT, "f7b8171"]):
            with self.assertRaisesRegex(ValueError, "found 2"):
                self.validate()

    def test_other_distribution_is_rejected(self):
        self.wheel(metadata_name="unrelated-project")
        with self.assertRaisesRegex(ValueError, "not the transformer-engine"):
            self.validate()

    def test_version_without_source_revision_is_rejected(self):
        self.wheel(version="2.17.0")
        with self.assertRaisesRegex(ValueError, "does not match checkout"):
            self.validate()

    def test_missing_metadata_is_rejected(self):
        with zipfile.ZipFile(
            self.wheels / f"transformer_engine-{VERSION}-py3-none-any.whl", "w"
        ) as archive:
            archive.writestr("transformer_engine/__init__.py", "")
        with self.assertRaisesRegex(ValueError, "metadata file"):
            self.validate()

    def test_source_changed_after_cache_key_resolution_is_rejected(self):
        self.wheel()
        with patch.object(VALIDATOR.subprocess, "check_output", return_value="a" * 40):
            with self.assertRaisesRegex(ValueError, "checkout changed"):
                self.validate()

    def test_invalid_expected_commit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "full TE-FL"):
            VALIDATOR.validate_wheel(self.wheels, self.source, "f7b8171")


if __name__ == "__main__":
    unittest.main()
