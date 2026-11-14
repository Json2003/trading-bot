#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import time
import threading
from typing import List, Optional

import numpy as np
import joblib
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
import pathlib

REGISTRY_PATH = "models/registry.json"
ACTIVE_TAG_PATH = "models/active_tag.txt"  # write a tag here to switch models

app = FastAPI(title="Model Inference API", version="1.0")
_device = "cuda" if torch.cuda.is_available() else "cpu"


class InferenceRequest(BaseModel):
    tag: str = ""  # optional: force specific tag
    features: List[List[float]]  # shape: [T, F]
    threshold: float = 0.5


class LSTMClassifier(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            n_feat, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.head(h)


class _State:
    def __init__(self):
        self.lock = threading.RLock()
        self.tag: Optional[str] = None
        self.model: Optional[LSTMClassifier] = None
        self.scaler = None
        self.meta = None


S = _State()


def _load_latest(tag: Optional[str] = None) -> str:
    if not os.path.exists(REGISTRY_PATH):
        raise RuntimeError("Registry not found")
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    if not isinstance(reg, dict) or not reg:
        raise RuntimeError("Registry is empty or invalid")

    chosen_tag = tag or ""
    if not chosen_tag:
        # try active tag
        if os.path.exists(ACTIVE_TAG_PATH):
            try:
                with open(ACTIVE_TAG_PATH) as f:
                    chosen_tag = f.read().strip()
            except Exception:
                chosen_tag = ""
    if not chosen_tag:
        # fallback: most recent by filename modtime among existing ckpts
        existing = [(t, m) for t, m in reg.items() if os.path.exists(m.get("ckpt", ""))]
        if not existing:
            raise RuntimeError("No valid checkpoints found in registry")
        chosen_tag = max(existing, key=lambda kv: os.path.getmtime(kv[1]["ckpt"]))[0]

    if chosen_tag not in reg:
        raise RuntimeError(f"Tag '{chosen_tag}' not found in registry")
    meta = reg[chosen_tag]
    ckpt_path = meta.get("ckpt")
    scaler_path = meta.get("scaler")
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise RuntimeError(f"Checkpoint missing for tag '{chosen_tag}' -> {ckpt_path}")
    if not scaler_path or not os.path.exists(scaler_path):
        raise RuntimeError(f"Scaler missing for tag '{chosen_tag}' -> {scaler_path}")

    ckpt = torch.load(ckpt_path, map_location=_device)
    model = LSTMClassifier(
        n_feat=int(ckpt["n_feat"]),
        hidden=int(ckpt["hidden"]),
        layers=int(ckpt["layers"]),
        dropout=float(ckpt["dropout"]),
    )
    model.load_state_dict(ckpt["state_dict"])  # type: ignore[arg-type]
    model.eval().to(_device)
    scaler = joblib.load(scaler_path)
    with S.lock:
        S.tag, S.model, S.scaler, S.meta = chosen_tag, model, scaler, meta
    return chosen_tag


def watch_active_tag():
    last = None
    while True:
        try:
            cur = open(ACTIVE_TAG_PATH).read().strip() if os.path.exists(ACTIVE_TAG_PATH) else None
            if cur and cur != last:
                _load_latest(cur)
                last = cur
        except Exception as e:
            print("watcher error:", e)
        time.sleep(2)


@app.on_event("startup")
def _startup():
    _load_latest(None)
    threading.Thread(target=watch_active_tag, daemon=True).start()


@app.get("/health")
def health():
    with S.lock:
        return {"ok": S.model is not None, "tag": S.tag}


@app.post("/infer")
def infer(req: InferenceRequest):
    with S.lock:
        active_tag, model, scaler = S.tag, S.model, S.scaler
    if model is None or scaler is None or not active_tag:
        raise HTTPException(500, "Model not loaded")
    if req.tag and req.tag != active_tag:
        raise HTTPException(409, "Requested model tag is not the loaded model")
    tag = active_tag
    if not np.isfinite(req.threshold) or not 0 <= req.threshold <= 1:
        raise HTTPException(400, "threshold must be finite and between zero and one")
    try:
        X = np.array(req.features, dtype=np.float32)  # (T,F)
    except (TypeError, ValueError):
        raise HTTPException(400, "features must be a rectangular numeric matrix") from None
    if X.ndim != 2 or not X.size or not np.isfinite(X).all():
        raise HTTPException(400, "features must be nonempty finite shape [T,F]")
    if X.shape[1] != model.lstm.input_size:
        raise HTTPException(400, "feature count does not match loaded model")
    try:
        Xs = scaler.transform(X)  # scale the same model/scaler snapshot
    except Exception as e:
        raise HTTPException(400, f"Scaler transform failed: {e}")
    if np.shape(Xs) != X.shape or not np.isfinite(Xs).all():
        raise HTTPException(400, "Scaler output is invalid")
    xs = torch.tensor(Xs, dtype=torch.float32, device=_device).unsqueeze(0)  # (1,T,F)
    with torch.no_grad():
        p = torch.sigmoid(model(xs)).cpu().numpy().ravel()[0]
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise HTTPException(500, "Model returned an invalid probability")
    out = {
        "tag": tag,
        "p_up": float(p),
        "allow": bool(p >= float(req.threshold)),
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    # Append simple JSONL log for monitoring
    try:
        log_dir = pathlib.Path("artifacts")
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "ml_infer.log", "a") as f:
            f.write(
                json.dumps(
                    {
                        "ts": out["ts"],
                        "tag": out["tag"],
                        "p_up": out["p_up"],
                        "allow": out["allow"],
                        "T": int(X.shape[0]),
                        "F": int(X.shape[1]),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    return out
