#!/usr/bin/env python3
from __future__ import annotations

import os, json, argparse, time
from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Use the repo's feature store (already guards pandas under the hood)
from data.feature_store import get_supervised_dataset


# Guarded pandas import to avoid repo-local stub (pandas.py)
import sys as _sys, os as _os, importlib as _importlib
_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
def _pd():
    mod = _sys.modules.get("pandas")
    if mod is not None:
        mod_file = getattr(mod, "__file__", "") or ""
        try:
            if _REPO_ROOT in _os.path.abspath(mod_file):
                del _sys.modules["pandas"]
        except Exception:
            pass
    original = _sys.path.copy()
    try:
        repo_paths = {p for p in original if _REPO_ROOT in _os.path.abspath(p)}
        non_repo = [p for p in original if p not in repo_paths]
        _sys.path = non_repo + [p for p in original if p in repo_paths]
        return _importlib.import_module("pandas")
    finally:
        _sys.path = original


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LSTMClassifier(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        h = out[:, -1, :]  # last timestep
        logits = self.head(h)
        return logits


def build_sequences(X_df, y, lookback: int) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    pd = _pd()
    # align & scale features
    df = pd.DataFrame(X_df).dropna().copy()
    y = pd.Series(y).reindex(df.index).dropna()
    df = df.loc[y.index]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df.values)
    # create sequences
    seqs, targets = [], []
    for i in range(lookback, len(Xs)):
        seqs.append(Xs[i - lookback : i, :])
        targets.append(1.0 if float(y.iloc[i]) > 0.0 else 0.0)  # binary: up next bar
    return np.array(seqs), np.array(targets), scaler


def train_loop(model, train_loader, val_loader, epochs=15, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    best_val = -1e9
    best = None
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE).unsqueeze(1)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward(); opt.step()
        # val
        model.eval()
        with torch.no_grad():
            corr, tot = 0, 0
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE).unsqueeze(1)
                p = torch.sigmoid(model(xb))
                pred = (p > 0.5).float()
                corr += (pred.eq(yb)).sum().item()
                tot += yb.numel()
        acc = corr / max(tot, 1)
        if acc > best_val:
            best_val = acc
            best = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"[epoch {ep}] val_acc={acc:.4f}")
    if best is not None:
        model.load_state_dict(best)
    return model, best_val


def split_train_val(X, y, val_frac=0.2):
    n = len(X)
    k = int(n * (1.0 - val_frac))
    return (X[:k], y[:k]), (X[k:], y[k:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--lookback", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--tag", default=None, help="Model tag, defaults to auto timestamp")
    ap.add_argument("--outdir", default="models/checkpoints")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X_df, y = get_supervised_dataset(
        args.symbol,
        args.timeframe,
        args.start,
        args.end,
        lookback_bars=args.lookback,
    )
    X, y, scaler = build_sequences(X_df, y, args.lookback)
    if len(X) < 10:
        raise RuntimeError("Not enough sequences after preprocessing. Increase date range or lower lookback.")
    (Xtr, ytr), (Xv, yv) = split_train_val(X, y)
    train_ds, val_ds = SeqDataset(Xtr, ytr), SeqDataset(Xv, yv)
    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    model = LSTMClassifier(n_feat=X.shape[-1], hidden=args.hidden, layers=args.layers, dropout=args.dropout)
    model, val_acc = train_loop(model, tl, vl, epochs=args.epochs)

    tag = args.tag or f"lstm_{args.symbol.replace('/', '-')}_{args.timeframe}_{int(time.time())}"
    ckpt_path = os.path.join(args.outdir, f"{tag}.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_feat": X.shape[-1],
            "lookback": args.lookback,
            "hidden": args.hidden,
            "layers": args.layers,
            "dropout": args.dropout,
        },
        ckpt_path,
    )

    # save scaler
    import joblib
    scaler_path = os.path.join(args.outdir, f"{tag}.scaler.gz")
    joblib.dump(scaler, scaler_path)

    # update registry
    reg_path = os.path.join("models", "registry.json")
    reg = {}
    if os.path.exists(reg_path):
        try:
            with open(reg_path) as f:
                reg = json.load(f) or {}
        except Exception:
            reg = {}
    reg[tag] = {
        "ckpt": ckpt_path,
        "scaler": scaler_path,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
    }
    with open(reg_path, "w") as f:
        json.dump(reg, f, indent=2)
    print(json.dumps({"tag": tag, "val_acc": float(val_acc), "ckpt": ckpt_path, "scaler": scaler_path}, indent=2))


if __name__ == "__main__":
    main()

