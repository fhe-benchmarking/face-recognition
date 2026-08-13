"""Regression test for the deterministic ArcFace detector contract."""

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from cleartext_impl import ARCFACE_DET_SIZE, load_arcface


class ArcFaceContractTest(unittest.TestCase):
    def test_detector_size_is_explicit(self):
        calls = {}

        class FakeFaceAnalysis:
            def __init__(self, **kwargs):
                calls["init"] = kwargs

            def prepare(self, **kwargs):
                calls["prepare"] = kwargs

        app = ModuleType("insightface.app")
        app.FaceAnalysis = FakeFaceAnalysis
        with patch.dict(sys.modules, {"insightface.app": app}):
            model = load_arcface()

        self.assertIsInstance(model, FakeFaceAnalysis)
        self.assertEqual(
            calls["init"],
            {
                "name": "buffalo_l",
                "providers": ["CPUExecutionProvider"],
            },
        )
        self.assertEqual(
            calls["prepare"],
            {
                "ctx_id": -1,
                "det_size": ARCFACE_DET_SIZE,
            },
        )
        self.assertEqual(ARCFACE_DET_SIZE, (640, 640))


if __name__ == "__main__":
    unittest.main()
