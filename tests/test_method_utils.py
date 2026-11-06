import math

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from gensid.method.utils import RecursiveClusteringMixin, dense_dmon_local_loss, spectral_cluster_subset


class TestDenseDmonLocalLoss:
    def test_edgeless_uniform_assignment_is_entropy_only(self):
        # Edgeless graph + uniform soft assignment: no modularity/balance penalty,
        # only the entropy regularizer (0.01 * entropy of a uniform [N, 2] row).
        loss, metrics = dense_dmon_local_loss(
            adj=torch.zeros(4, 4),
            logits=torch.zeros(4, 2),
            degree=torch.zeros(4),
            m2=1e-12,
        )
        assert metrics["modularity"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["balance"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["entropy"] == pytest.approx(math.log(2), rel=1e-5)
        assert loss == pytest.approx(0.01 * math.log(2), rel=1e-5)

    def test_edgeless_one_hot_balance_penalty(self):
        # 3 edgeless nodes split 2/1: balance term = (1/3-1/2)^2 * 2 = 1/18.
        logits = torch.tensor([[10.0, -10.0], [10.0, -10.0], [-10.0, 10.0]])
        loss, metrics = dense_dmon_local_loss(
            adj=torch.zeros(3, 3),
            logits=logits,
            degree=torch.zeros(3),
            m2=1e-12,
        )
        assert metrics["balance"] == pytest.approx(1 / 18, abs=1e-4)
        assert metrics["entropy"] < 1e-3
        assert loss == pytest.approx(1 / 18, abs=1e-4)

    @staticmethod
    def _two_clique_adj():
        adj = torch.zeros(8, 8)
        for comp in (range(0, 4), range(4, 8)):
            for i in comp:
                for j in comp:
                    if i != j:
                        adj[i, j] = 1.0
        return adj

    def test_modularity_rewards_true_cliques_and_penalizes_split(self):
        adj = self._two_clique_adj()
        degree = adj.sum(dim=1)
        m2 = degree.sum()
        good = torch.tensor([[10.0, -10.0]] * 4 + [[-10.0, 10.0]] * 4)
        bad = torch.tensor([[10.0, -10.0], [10.0, -10.0], [-10.0, 10.0], [-10.0, 10.0]] * 2)
        _, mgood = dense_dmon_local_loss(adj, good, degree, m2)
        _, mbad = dense_dmon_local_loss(adj, bad, degree, m2)
        # Perfect split: (edge_term - null_term) / m2 = (24 - 12) / 24.
        assert mgood["modularity"] == pytest.approx(0.5, abs=1e-4)
        # Split that mixes the cliques: (8 - 12) / 24.
        assert mbad["modularity"] == pytest.approx(-1 / 6, abs=1e-4)
        assert mgood["modularity"] > mbad["modularity"]


class TestSpectralClusterSubset:
    @staticmethod
    def _two_clique_csr():
        adj = np.zeros((8, 8))
        for comp in (range(0, 4), range(4, 8)):
            for i in comp:
                for j in comp:
                    if i != j:
                        adj[i, j] = 1.0
        return sp.csr_matrix(adj)

    def test_empty_subset(self):
        labels = spectral_cluster_subset(self._two_clique_csr(), np.array([], dtype=np.int64), 3, seed=0)
        assert labels.shape == (0,)
        assert labels.dtype == np.int64

    def test_k_le_one_returns_zeros(self):
        labels = spectral_cluster_subset(self._two_clique_csr(), np.arange(8), k=1, seed=0)
        assert labels.tolist() == [0] * 8

    def test_fewer_nodes_than_clusters_is_identity(self):
        labels = spectral_cluster_subset(self._two_clique_csr(), np.arange(3), k=4, seed=0)
        assert labels.tolist() == [0, 1, 2]

    def test_edgeless_subgraph_splits_by_position(self):
        # Only one edge, between node 0 and node 9; the subset ``0..7`` induces no edges.
        a = sp.lil_matrix((10, 10))
        a[0, 9] = a[9, 0] = 1.0
        labels = spectral_cluster_subset(a.tocsr(), np.arange(8), k=3, seed=0)
        assert labels.tolist() == [0, 1, 2, 0, 1, 2, 0, 1]

    def test_two_cliques_are_separated(self):
        labels = spectral_cluster_subset(self._two_clique_csr(), np.arange(8), k=2, seed=0)
        (a,) = set(labels[:4].tolist())
        (b,) = set(labels[4:].tolist())
        assert a != b

    def test_eigsh_failure_falls_back_to_adjacency_rows(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr("gensid.method.utils.eigsh", _boom)
        labels = spectral_cluster_subset(self._two_clique_csr(), np.arange(8), k=2, seed=0)
        assert labels.shape == (8,)
        assert labels.dtype == np.int64
        (a,) = set(labels[:4].tolist())
        (b,) = set(labels[4:].tolist())
        assert a != b


class TestRecursiveClusteringMixin:
    class StubClusterer(RecursiveClusteringMixin):
        def forward(self, x_all, A, subset, k, level, parent):
            return np.arange(len(subset)) % k

    @staticmethod
    def _path_edge_index(num_nodes=6):
        return torch.tensor([list(range(num_nodes - 1)), list(range(1, num_nodes))], dtype=torch.long)

    def test_two_level_hierarchy_labels(self):
        hierarchy = self.StubClusterer().recursive_clustering(
            torch.randn(6, 4),
            self._path_edge_index(),
            None,
            recursive_levels=2,
            recursive_factor=2,
        )
        assert len(hierarchy) == 2
        assert hierarchy[0].tolist() == [0, 1, 0, 1, 0, 1]
        assert hierarchy[1].tolist() == [0, 2, 1, 3, 0, 2]

    def test_level_cluster_counts(self):
        hierarchy = self.StubClusterer().recursive_clustering(
            torch.randn(6, 4),
            self._path_edge_index(),
            None,
            recursive_levels=2,
            recursive_factor=2,
        )
        counts = [len(np.unique(h[h >= 0])) for h in hierarchy]
        assert counts == [2, 4]

    def test_single_level_requested(self):
        hierarchy = self.StubClusterer().recursive_clustering(
            torch.randn(6, 4),
            self._path_edge_index(),
            None,
            recursive_levels=1,
            recursive_factor=2,
        )
        assert len(hierarchy) == 1

    def test_weighted_edges_accepted(self):
        edge_index = self._path_edge_index()
        edge_weight = torch.ones(edge_index.size(1))
        hierarchy = self.StubClusterer().recursive_clustering(
            torch.randn(6, 4), edge_index, edge_weight, recursive_levels=1, recursive_factor=2
        )
        assert len(hierarchy) == 1
        assert hierarchy[0].shape == (6,)
