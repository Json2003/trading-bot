"""Helpers to promote candidate batch models to production with safety checks.

Promotion requires:
- `ALLOW_MODEL_PROMOTE=true` in env
- presence of file `allow_live_confirm.txt` in project root
This prevents accidental live swaps.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def can_promote(env: dict) -> bool:
    if env.get('ALLOW_MODEL_PROMOTE', 'false').lower() != 'true':
        return False
    confirm = ROOT / 'allow_live_confirm.txt'
    return confirm.exists()

def promote_model(candidate_path: str, dest_path: str, env: dict) -> bool:
    if not can_promote(env):
        return False
    Path(candidate_path).replace(dest_path)
    return True
