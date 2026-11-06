from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gensid.config.graph_build_config import AdjacentCoocGraphConfig


class StubGraphDataset:
    """Minimal graph dataset exposing what clustering SID methods/trainers consume."""

    def __init__(
        self,
        x_graph=None,
        edge_index=None,
        edge_weight=None,
        num_items=6,
        item_ids=None,
        embeddings_path="emb.npy",
        interactions_path="inter.json",
        skip_n_last=2,
    ):
        if x_graph is None:
            x_graph = torch.linspace(0, 5, num_items * 4).reshape(num_items, 4)
        if edge_index is None:
            edge_index = torch.tensor([list(range(num_items - 1)), list(range(1, num_items))], dtype=torch.long)
        self.x_graph = x_graph
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_items = num_items
        self.item_ids = item_ids if item_ids is not None else np.arange(num_items, dtype=np.int64)
        self.embeddings_path = embeddings_path
        self.interactions_path = interactions_path
        self.config = SimpleNamespace(
            graph_building=AdjacentCoocGraphConfig(),
            skip_n_last=skip_n_last,
        )

    def to(self, device):
        if torch.is_tensor(self.x_graph):
            self.x_graph = self.x_graph.to(device)
        if torch.is_tensor(self.edge_index):
            self.edge_index = self.edge_index.to(device)
        if self.edge_weight is not None and torch.is_tensor(self.edge_weight):
            self.edge_weight = self.edge_weight.to(device)
        return self

    def __len__(self):
        return self.num_items


@pytest.fixture
def graph_stub_dataset():
    """A deterministic 6-node path graph with the attrs clustering methods read."""
    return StubGraphDataset()


@pytest.fixture
def graph_stub_loader(graph_stub_dataset):
    """A DataLoader shim exposing ``.dataset`` and ``.batch_size``."""
    return SimpleNamespace(dataset=graph_stub_dataset, batch_size=16)
