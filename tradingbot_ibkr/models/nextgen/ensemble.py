"""Utilities for blending uncertainty-aware model predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol, Tuple

import torch


class ProbabilisticModel(Protocol):
    """Protocol capturing predict_with_uncertainty behaviour."""

    def predict_with_uncertainty(
        self, x: torch.Tensor, passes: int | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


@dataclass
class EnsembleOutput:
    mean_prob: torch.Tensor
    entropy: torch.Tensor
    member_entropies: List[torch.Tensor]


class UncertaintyEnsembler:
    """Aggregate predictions from multiple models via entropy-weighted blending."""

    def __init__(self, models: Iterable[ProbabilisticModel]) -> None:
        self.models = list(models)
        if not self.models:
            raise ValueError("UncertaintyEnsembler requires at least one model")

    def predict(self, x: torch.Tensor, passes: int | None = None) -> EnsembleOutput:
        probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        for model in self.models:
            mean_prob, entropy = model.predict_with_uncertainty(x, passes=passes)
            probs.append(mean_prob)
            entropies.append(entropy)

        stacked_probs = torch.stack(probs, dim=0)  # [members, batch, classes]
        stacked_entropy = torch.stack(entropies, dim=0)  # [members, batch]

        inv_entropy = 1.0 / (stacked_entropy + 1e-6)
        weights = inv_entropy / inv_entropy.sum(dim=0, keepdim=True)
        blended = (weights.unsqueeze(-1) * stacked_probs).sum(dim=0)

        final_entropy = -(blended * torch.log(blended + 1e-9)).sum(dim=-1)
        return EnsembleOutput(mean_prob=blended, entropy=final_entropy, member_entropies=entropies)
