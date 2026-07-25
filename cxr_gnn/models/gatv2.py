"""
models/gatv2.py -- GATv2-based graph classifier.

Architecture:
    BN(input) -> dropout -> GATv2Conv_1 (K heads, concat) -> BN -> ELU
              -> DropEdge -> GATv2Conv_2 (1 head) -> BN -> ELU
              -> [mean_pool || max_pool] -> Linear -> ELU -> Dropout -> Linear

Why mean + max pooling?
    mean_pool summarises global texture/intensity; max_pool picks out the single
    most salient region, which is what a focal pathology looks like. Concatenating
    keeps both.

Why DropEdge?
    Randomly dropping edges regularises the graph structure, counteracting
    over-smoothing. Train only -- disabled in eval mode.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.utils import dropout_edge

from cxr_gnn.config import Config, NUM_CLASSES


class GATv2Classifier(nn.Module):
    """
    Args:
        node_feat_dim: input node feature dim (140 hybrid, 128 deep-only, 12 hand-only)
        edge_feat_dim: input edge feature dim (2, or None to disable edge features)
        cfg:           frozen Config
        n_classes:     number of output classes
        heads:         attention heads in layer 1; defaults to cfg.heads
                       (overridden by the single-head ablation arm)
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: Optional[int],
        cfg: Config,
        n_classes: int = NUM_CLASSES,
        heads: Optional[int] = None,
    ) -> None:
        super().__init__()

        H = cfg.hidden
        K = cfg.heads if heads is None else heads
        self._dropout = cfg.dropout
        self._in_drop = cfg.in_dropout
        self._dropedge = cfg.dropedge
        self._edge_dim = edge_feat_dim

        self.in_bn = nn.BatchNorm1d(node_feat_dim)

        self.gat1 = GATv2Conv(node_feat_dim, H, heads=K, edge_dim=edge_feat_dim,
                              dropout=cfg.dropout, concat=True)
        self.bn1 = nn.BatchNorm1d(H * K)

        self.gat2 = GATv2Conv(H * K, H, heads=1, edge_dim=edge_feat_dim,
                              dropout=cfg.dropout, concat=True)
        self.bn2 = nn.BatchNorm1d(H)

        self.head = nn.Sequential(
            nn.Linear(H * 2, H),          # H*2: mean_pool || max_pool
            nn.ELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(H, n_classes),
        )

    def _trunk(self, x, edge_index, edge_attr, ei=None, ea=None):
        """Shared body of forward()/embed() so the two can never diverge."""
        x = self.in_bn(x)
        x = F.dropout(x, p=self._in_drop, training=self.training)
        ei = edge_index if ei is None else ei
        ea = edge_attr if ea is None else ea
        x = F.elu(self.bn1(self.gat1(x, ei, ea)))
        return x

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        batch: Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            x:          (total_nodes, node_feat_dim)
            edge_index: (2, total_edges)
            edge_attr:  (total_edges, edge_feat_dim) or None
            batch:      (total_nodes,) graph assignment vector

        Returns:
            logits, or (logits, (edge_index, attention_weights)) when
            return_attention=True.
        """
        # DropEdge (train only). The edge mask must be applied to edge_attr as
        # well, or attributes and edges desynchronise.
        ei, ea = edge_index, edge_attr
        if self.training and self._dropedge > 0:
            ei, mask = dropout_edge(edge_index, p=self._dropedge)
            if ea is not None:
                ea = ea[mask]

        x = self._trunk(x, edge_index, edge_attr, ei, ea)

        if return_attention:
            # Attention is reported on the FULL edge set: a saliency map should
            # not depend on which edges a random DropEdge draw happened to keep.
            x, (attn_edge_index, attn_weights) = self.gat2(
                x, edge_index, edge_attr, return_attention_weights=True,
            )
        else:
            x = self.gat2(x, ei, ea)
            attn_edge_index = attn_weights = None

        x = F.elu(self.bn2(x))

        h = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        logits = self.head(h)

        if return_attention:
            return logits, (attn_edge_index, attn_weights)
        return logits

    @torch.no_grad()
    def embed(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor],
              batch: Tensor) -> Tensor:
        """Graph-level embedding before the classification head (for t-SNE).

        Does not flip the module's train/eval state -- callers set that; the
        original version called self.eval() as a side effect and left it there.
        """
        x = self._trunk(x, edge_index, edge_attr)
        x = F.elu(self.bn2(self.gat2(x, edge_index, edge_attr)))
        return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)


class GATv2SingleHead(GATv2Classifier):
    """Ablation arm: one attention head instead of cfg.heads."""

    def __init__(self, node_feat_dim, edge_feat_dim, cfg, n_classes=NUM_CLASSES):
        super().__init__(node_feat_dim, edge_feat_dim, cfg, n_classes, heads=1)
