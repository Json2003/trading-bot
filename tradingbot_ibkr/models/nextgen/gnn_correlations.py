"""Graph neural network model for cross-asset correlations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

try:  # pragma: no cover - optional torch
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for CorrelationGraphModel") from exc

try:  # pragma: no cover - optional PyG
    from torch_geometric.nn import GraphConv
    from torch_geometric.data import Data
except Exception:  # pragma: no cover - fallback simple aggregator
    GraphConv = None
    Data = None


@dataclass
class GraphBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: Optional[torch.Tensor]


def build_correlation_graph(features: Dict[str, np.ndarray], threshold: float = 0.5) -> GraphBatch:
    """Create a graph from asset feature matrix using Pearson correlations."""
    assets = list(features.keys())
    matrix = np.stack([features[a] for a in assets], axis=0)
    corr = np.corrcoef(matrix)
    edge_src = []
    edge_dst = []
    edge_weight = []
    num_assets = len(assets)
    for i in range(num_assets):
        for j in range(num_assets):
            if i == j:
                continue
            weight = corr[i, j]
            if abs(weight) >= threshold:
                edge_src.append(i)
                edge_dst.append(j)
                edge_weight.append(weight)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_weight_tensor = torch.tensor(edge_weight, dtype=torch.float32) if edge_weight else None
    node_features = torch.tensor(matrix, dtype=torch.float32)
    return GraphBatch(
        x=node_features,
        edge_index=edge_index,
        edge_weight=edge_weight_tensor,
    )


class CorrelationGraphModel(nn.Module):
    """GNN wrapper that uses PyG when available, otherwise mean-aggregator fallback."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 16) -> None:
        super().__init__()
        self.use_pyg = GraphConv is not None
        if self.use_pyg:
            self.conv1 = GraphConv(input_dim, hidden_dim)
            self.conv2 = GraphConv(hidden_dim, output_dim)
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data: GraphBatch) -> torch.Tensor:  # type: ignore[override]
        if self.use_pyg:
            pyg_data = Data(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight)
            h = self.conv1(pyg_data.x, pyg_data.edge_index, pyg_data.edge_weight)
            h = torch.relu(h)
            h = self.conv2(h, pyg_data.edge_index, pyg_data.edge_weight)
            return torch.relu(h)
        # fallback aggregator: mean of neighbours weighted by correlation
        x = data.x
        h = torch.relu(self.fc1(x))
        if data.edge_weight is not None and data.edge_weight.numel() > 0:
            weight_matrix = torch.zeros((x.size(0), x.size(0)), dtype=torch.float32, device=x.device)
            weight_matrix[data.edge_index[0], data.edge_index[1]] = data.edge_weight
            neigh = weight_matrix @ h
            h = h + neigh
        return torch.relu(self.fc2(h))
