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
from datetime import datetime
import pathlib

REGISTRY_PATH = "models/registry.json"
ACTIVE_TAG_PATH = "models/active_tag.txt"  # write a tag here to switch models

app = FastAPI(title="Model Inference API", version="1.0")
_device = "cuda" if torch.cuda.is_available() else "cpu"


class InferenceRequest(BaseModel):
    tag: str = ""                # optional: force specific tag
    features: List[List[float]]  # shape: [T, F]
    threshold: float = 0.5


class LSTMClassifier(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.head(h)


class _State:
    def __init__(self):
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
    return {"ok": S.model is not None, "tag": S.tag}


@app.post("/infer")
def infer(req: InferenceRequest):
    if S.model is None:
        raise HTTPException(500, "Model not loaded")
    tag = req.tag or S.tag or ""
    X = np.array(req.features, dtype=np.float32)  # (T,F)
    if X.ndim != 2:
        raise HTTPException(400, "features must be shape [T,F]")
    try:
        Xs = S.scaler.transform(X)  # scale
    except Exception as e:
        raise HTTPException(400, f"Scaler transform failed: {e}")
    xs = torch.tensor(Xs, dtype=torch.float32, device=_device).unsqueeze(0)  # (1,T,F)
    with torch.no_grad():
        p = torch.sigmoid(S.model(xs)).cpu().numpy().ravel()[0]
    out = {
        "tag": tag,
        "p_up": float(p),
        "allow": bool(p >= float(req.threshold)),
        "ts": datetime.utcnow().isoformat() + "Z",
    }
    # Append simple JSONL log for monitoring
    try:
        log_dir = pathlib.Path("artifacts")
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "ml_infer.log", "a") as f:
            f.write(json.dumps({
                "ts": out["ts"], "tag": out["tag"], "p_up": out["p_up"],
                "allow": out["allow"], "T": int(X.shape[0]), "F": int(X.shape[1]),
            }) + "\n")
    except Exception:
        pass
    return out
