"""Transformer + mixture-of-experts classifier with uncertainty estimation."""

from __future__ import annotations

import dataclasses
from typing import Callable, Iterable, List, Optional, Tuple


try:  # pragma: no cover - heavy dependency optional
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception as exc:  # pragma: no cover - guidance for environments without torch
    raise ImportError(
        "PyTorch is required for TransformerMoEClassifier. "
        "Install torch>=2.0 to enable nextgen models."
    ) from exc

try:  # pragma: no cover - optional dependency
    import pytorch_lightning as pl
except Exception:  # pragma: no cover - fallback training harness
    pl = None


@dataclasses.dataclass
class TransformerMoEConfig:
    """Configuration for the Transformer + MoE classifier."""

    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_experts: int = 4
    dropout: float = 0.1
    num_classes: int = 2
    uncertainty_passes: int = 10


class TransformerMoEClassifier(nn.Module):
    """Transformer encoder with Mixture-of-Experts head and MC-dropout inference."""

    def __init__(self, config: TransformerMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.dropout = nn.Dropout(config.dropout)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, config.num_classes),
                )
                for _ in range(config.num_experts)
            ]
        )
        self.gating = nn.Linear(config.hidden_dim, config.num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, input_dim]
        h = self.input_proj(x)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        pooled = self.dropout(pooled)
        logits_per_expert = torch.stack([expert(pooled) for expert in self.experts], dim=1)
        gates = F.softmax(self.gating(pooled), dim=-1).unsqueeze(-1)
        logits = torch.sum(logits_per_expert * gates, dim=1)
        return logits

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor, passes: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean probability and predictive entropy using MC-dropout."""
        self.train()  # ensure dropout active
        passes = passes or self.config.uncertainty_passes
        probs: List[torch.Tensor] = []
        for _ in range(passes):
            logits = self.forward(x)
            probs.append(F.softmax(logits, dim=-1))
        stacked = torch.stack(probs, dim=0)
        mean_prob = stacked.mean(dim=0)
        entropy = -(mean_prob * torch.log(mean_prob + 1e-9)).sum(dim=-1)
        return mean_prob, entropy


class LightningTransformerMoE(pl.LightningModule if pl else object):  # type: ignore[misc]
    """PyTorch Lightning wrapper enabling standardised training loops."""

    def __init__(
        self,
        model: TransformerMoEClassifier,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        lr: float = 1e-3,
    ) -> None:
        if pl is None:  # pragma: no cover - guidance
            raise RuntimeError("pytorch-lightning must be installed to use LightningTransformerMoE")
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore[override]
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        acc = (preds == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc})

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]


def build_default_model(input_dim: int, num_classes: int = 2) -> TransformerMoEClassifier:
    """Convenience factory used by other services."""
    config = TransformerMoEConfig(input_dim=input_dim, num_classes=num_classes)
    return TransformerMoEClassifier(config)
