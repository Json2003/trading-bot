"""Lightweight import wrapper for the inference API.

Expose FastAPI app at module path `models.predict_api:app` so you can run:
  uvicorn models.predict_api:app --host 0.0.0.0 --port 8000

The actual implementation lives in `scripts/infer_api.py`.
"""
from __future__ import annotations

from scripts.infer_api import app  # re-export

__all__ = ["app"]

