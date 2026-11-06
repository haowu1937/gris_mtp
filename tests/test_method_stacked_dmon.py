import pytest
import torch

from gensid.config.model_config import StackedDMoNModelConfig
from gensid.method.stacked_dmon import StackedDMoN


def _config(**kwargs):
    kwargs.setdefault("in_dim", 4)
    kwargs.setdefault("num_emb_list", (3, 2))
    kwargs.setdefault("e_dim", 4)
    kwargs.setdefault("dmon_hidden_dim", 8)
    kwargs.setdefault("assign_hidden_dim", 8)
    kwargs.setdefault("dropout", 0.0)
    return StackedDMoNModelConfig(**kwargs)


class TestDmonPool:
    def test_pools_clusters_and_coarsens_adjacency(self):
        # Path 0-1-2-3, one-hot assignment {0,1} -> A, {2,3} -> B.
        x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        adj = torch.tensor([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=torch.float32)
        logits = torch.tensor([[10.0, -10.0], [10.0, -10.0], [-10.0, 10.0], [-10.0, 10.0]])

        x_next, adj_next, s = StackedDMoN.dmon_pool(x, adj, logits)

        # Cluster-wise means: A = mean(0, 1) = 0.5, B = mean(2, 3) = 2.5.
        assert x_next.shape == (2, 1)
        assert torch.allclose(x_next[:, 0], torch.tensor([0.5, 2.5]), atol=1e-4)
        # Coarsened adjacency: off-diagonal inter-cluster weight (1-2 edge), zero diagonal.
        assert torch.allclose(adj_next, torch.tensor([[0.0, 1.0], [1.0, 0.0]]), atol=1e-4)
        assert adj_next.shape == (2, 2)
        # Soft assignment rows are (near) one-hot and sum to 1.
        assert s.shape == (4, 2)
        assert torch.allclose(s.sum(dim=1), torch.ones(4), atol=1e-4)
        assert torch.allclose(s[0], torch.tensor([1.0, 0.0]), atol=1e-4)


class TestComposeAssignments:
    def test_compose_assignments_to_indices(self):
        p0 = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        p1 = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        composed = StackedDMoN.compose_assignments_to_indices([p0, p1])
        # argmax(P0) = [0,0,1]; argmax(P0 @ P1) = [1,1,0].
        assert composed.tolist() == [[0, 1], [0, 1], [1, 0]]

    def test_level_probs_to_f2c_cf2_indices(self):
        p0 = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        p1 = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        model = StackedDMoN(_config())
        f2c, c2f = model.level_probs_to_f2c_cf2_indices([{"s_probs": p0}, {"s_probs": p1}])
        assert f2c.tolist() == [[0, 1], [0, 1], [1, 0]]
        assert c2f.tolist() == [[1, 0], [1, 0], [0, 1]]


class TestStackedDMoN:
    def _model(self, **kwargs):
        return StackedDMoN(_config(**kwargs))

    def test_codebook_sizes_are_coarse_to_fine(self):
        model = self._model()
        assert model.n_codebooks == 2
        # cluster_sizes are fine-to-coarse (3, 2); codebooks are the reverse.
        assert model.codebook_sizes == (2, 3)

    def test_forward_runs_all_levels_and_accumulates_loss(self, graph_stub_dataset):
        model = self._model()
        z0, levels, total_loss = model(graph_stub_dataset)

        assert z0.shape == (6, 4)
        assert len(levels) == 2
        assert levels[0]["s_probs"].shape == (6, 3)
        assert levels[1]["s_probs"].shape == (3, 2)
        assert torch.allclose(levels[0]["s_probs"].sum(dim=1), torch.ones(6), atol=1e-5)
        assert torch.allclose(levels[1]["s_probs"].sum(dim=1), torch.ones(3), atol=1e-5)
        for lvl in levels:
            for key in ("level", "s_logits", "s_probs", "loss", "modularity", "balance", "entropy"):
                assert key in lvl
        assert total_loss.item() == pytest.approx(levels[0]["loss"] + levels[1]["loss"], abs=1e-5)

    def test_generate_returns_coarse_to_fine_matrix_and_artifacts(self, graph_stub_dataset):
        model = self._model()
        coarse, artifacts = model.generate(graph_stub_dataset)

        assert coarse.shape == (6, 2)
        assert coarse.dtype == torch.int64
        # col 0 is coarsest (level 2, k=2); col 1 is finest (level 1, k=3).
        assert coarse[:, 0].max() < 2
        assert coarse[:, 1].max() < 3
        assert set(artifacts) == {
            "encoded_nodes",
            "level1_s_probs",
            "level1_s_logits",
            "level2_s_probs",
            "level2_s_logits",
        }
        assert artifacts["encoded_nodes"].shape == (6, 4)
        assert artifacts["level1_s_probs"].shape == (6, 3)
        assert artifacts["level2_s_probs"].shape == (3, 2)

    def test_level_with_more_clusters_than_nodes_raises(self, graph_stub_dataset):
        model = self._model(num_emb_list=(8,))
        with pytest.raises(ValueError, match="k=8"):
            model(graph_stub_dataset)
