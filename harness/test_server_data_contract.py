"""Regression tests for the backend-neutral server-data measurement contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import utils


class ServerDataContractTest(unittest.TestCase):
    def test_measurement_sums_arbitrary_nested_regular_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            server_data = Path(temporary) / "server_data"
            nested = server_data / "backend" / "cache"
            nested.mkdir(parents=True)
            (server_data / "model.bin").write_bytes(b"abc")
            (nested / "tables.dat").write_bytes(b"defgh")

            with patch.object(utils, "_bandwidth", {}):
                self.assertEqual(utils.log_server_data_size(server_data), 8)
                self.assertEqual(
                    utils._bandwidth["Packed model weights"], "8.0B"
                )

    def test_missing_or_non_directory_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing"
            regular_file = root / "model.bin"
            regular_file.write_bytes(b"model")

            with self.assertRaises(FileNotFoundError):
                utils.log_server_data_size(missing)
            with self.assertRaises(NotADirectoryError):
                utils.log_server_data_size(regular_file)

    def test_symbolic_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server_data = root / "server_data"
            server_data.mkdir()
            target = root / "model.bin"
            target.write_bytes(b"model")
            (server_data / "model-link").symlink_to(target)

            with self.assertRaisesRegex(ValueError, "symbolic links"):
                utils.log_server_data_size(server_data)

    def test_one_time_measurement_is_written_to_every_result(self):
        with tempfile.TemporaryDirectory() as temporary, patch.multiple(
            utils,
            _timestamps={"Encrypted model preprocessing": 1.0},
            _timestampsStr={"Encrypted model preprocessing": "1.0s"},
            _onetime_timestamps={},
            _onetime_timestampsStr={},
            _bandwidth={"Packed model weights": "8.0B"},
            _onetime_bandwidth={},
            _model_quality={},
        ):
            utils.reset_run_state()
            for run in (1, 2):
                result_path = Path(temporary) / f"results-{run}.json"
                utils.save_run(result_path)
                result = json.loads(result_path.read_text())
                self.assertEqual(
                    result["Bandwidth"]["Packed model weights"], "8.0B"
                )
                utils.reset_run_state()


if __name__ == "__main__":
    unittest.main()
