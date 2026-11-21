"""Exercise production inference/gate guards with deterministic stub logits."""
import contextlib
import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HttpError(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail


class FakeApp:
    def __init__(self, **kwargs):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: lambda func: func


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def unsqueeze(self, axis):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.value)


class FakeModel:
    def __init__(self, callback=None):
        self.lstm = types.SimpleNamespace(input_size=2)
        self.calls = 0
        self.callback = callback

    def __call__(self, data):
        self.calls += 1
        if self.callback:
            self.callback()
        return np.array([0.8])


class NeuralExecutionIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.http = types.ModuleType("requests")
        self.http.post = Mock()
        with patch.dict(sys.modules, {"requests": self.http}):
            self.gate = load_file("test_ml_gate", "scripts/ml_gate.py")
        nn = types.ModuleType("torch.nn")
        nn.Module = object
        torch = types.ModuleType("torch")
        torch.nn = nn
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.float32 = None
        torch.tensor = lambda value, **kwargs: FakeTensor(value)
        torch.no_grad = contextlib.nullcontext
        torch.sigmoid = lambda value: FakeTensor(value)
        fastapi = types.ModuleType("fastapi")
        fastapi.FastAPI, fastapi.HTTPException = FakeApp, HttpError
        pydantic = types.ModuleType("pydantic")
        pydantic.BaseModel = object
        with patch.dict(sys.modules, {"torch": torch, "torch.nn": nn, "fastapi": fastapi,
                                     "pydantic": pydantic, "joblib": types.ModuleType("joblib")}):
            self.api = load_file("test_inference_api", "scripts/infer_api.py")
        self.api.pathlib = types.SimpleNamespace(Path=Mock(side_effect=OSError("no test logs")))
        self.model = FakeModel()
        self.api.S = types.SimpleNamespace(lock=threading.RLock(), tag="loaded-v1", model=self.model,
                                          scaler=types.SimpleNamespace(transform=lambda value: value))
        self.req = types.SimpleNamespace(tag="", threshold=0.55, features=[[1.0, 2.0]])

    def test_inference_failure_denies(self):
        self.http.post.side_effect = TimeoutError("offline")
        self.assertEqual(self.gate.infer_gate(pd.DataFrame([[1, 2]])), (False, 0.0, None))

    def test_missing_allow_or_wrong_tag_denies(self):
        self.http.post.return_value.json.return_value = {"p_up": 0.9, "tag": "v1"}
        self.assertFalse(self.gate.infer_gate(pd.DataFrame([[1, 2]]))[0])
        self.http.post.return_value.json.return_value = {"allow": True, "p_up": 0.9, "tag": "wrong"}
        self.assertFalse(self.gate.infer_gate(pd.DataFrame([[1, 2]]), tag="v1")[0])

    def test_nonfinite_probability_or_features_deny(self):
        self.http.post.return_value.json.return_value = {"allow": True, "p_up": float("nan"), "tag": "v1"}
        self.assertFalse(self.gate.infer_gate(pd.DataFrame([[1, 2]]))[0])
        self.assertFalse(self.gate.infer_gate(pd.DataFrame([[float("nan"), 2]]))[0])

    def test_positive_gate_requires_matching_finite_score(self):
        self.http.post.return_value.json.return_value = {"allow": True, "p_up": 0.8, "tag": "v1"}
        self.assertEqual(self.gate.infer_gate(pd.DataFrame([[1, 2]]), tag="v1"), (True, 0.8, "v1"))
        self.http.post.return_value.json.return_value["p_up"] = 0.2
        self.assertFalse(self.gate.infer_gate(pd.DataFrame([[1, 2]]), tag="v1")[0])

    def test_requested_unloaded_model_is_rejected(self):
        self.req.tag = "not-loaded"
        with self.assertRaises(HttpError) as caught:
            self.api.infer(self.req)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(self.model.calls, 0)

    def test_input_shape_nonfinite_and_threshold_fail_before_inference(self):
        for field, value in (("features", []), ("features", [[1.0]]),
                             ("features", [[float("inf"), 2.0]]), ("threshold", float("nan"))):
            req = types.SimpleNamespace(**vars(self.req))
            setattr(req, field, value)
            with self.assertRaises(HttpError):
                self.api.infer(req)
        self.assertEqual(self.model.calls, 0)

    def test_prediction_uses_consistent_model_scaler_and_tag_snapshot(self):
        def change_active_state():
            self.api.S.tag = "new-v2"
            self.api.S.model = FakeModel()
        self.model.callback = change_active_state
        result = self.api.infer(self.req)
        self.assertEqual(result["tag"], "loaded-v1")
        self.assertAlmostEqual(result["p_up"], 0.8)
        self.assertEqual(self.model.calls, 1)


if __name__ == "__main__":
    unittest.main()
