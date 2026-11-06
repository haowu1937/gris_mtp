import numpy as np
import pytest
import torch

from gensid.config.model_config import RecDMoNModelConfig, SpectralModelConfig
from gensid.data.utils import edge_index_to_sparse_adj
from gensid.method.recursive_hierarchy import RecDMoN, Spectral


def _path_edge_index(num_nodes=6):
    return torch.tensor([list(range(num_nodes - 1)), list(range(1, num_nodes))], dtype=torch.long)


def _path_adj(num_nodes=6):
    return edge_index_to_sparse_adj(_path_edge_index(num_nodes), None, num_nodes)


def _recdmon_config(**kwargs):
    kwargs.setdefault("recursive_epochs", 2)
    kwargs.setdefault("assign_hidden_dim", 4)
    kwargs.setdefault("dropout", 0.0)
    kwargs.setdefault("max_dense_nodes", 10000)
    kwargs.setdefault("recursive_factor", 2)
    kwargs.setdefault("recursive_levels", 2)
    kwargs.setdefault("lr", 1e-3)
    kwargs.setdefault("weight_decay", 0.0)
    kwargs.setdefault("grad_clip", None)
    return RecDMoNModelConfig(**kwargs)


class TestRecDMoN:
    def test_config_validation_raises(self):
        with pytest.raises(ValueError, match="recursive_factor"):
            RecDMoN(_recdmon_config(recursive_factor=1))
        with pytest.raises(ValueError, match="recursive_levels"):
            RecDMoN(_recdmon_config(recursive_levels=0))

    def test_forward_clusters_subset_into_children(self):
        torch.manual_seed(0)
        model = RecDMoN(_recdmon_config())
        labels = model.forward(torch.randn(6, 4), _path_adj(), np.arange(6), k=2, level=1, parent=0)
        assert labels.shape == (6,)
        assert labels.dtype == np.int64
        assert labels.min() >= 0 and labels.max() < 2

    def test_fit_generate_and_state_round_trip(self, graph_stub_dataset):
        torch.manual_seed(0)
        model = RecDMoN(_recdmon_config())
        model.fit(graph_stub_dataset)
        indices = model.generate(graph_stub_dataset)
        assert indices.shape == (6, 2)
        assert indices.dtype == torch.int64
        assert model.n_codebooks == 2
        assert len(model.codebook_sizes) == 2

        other = RecDMoN(_recdmon_config())
        other.load_state_dict(model.state_dict())
        assert torch.equal(other.generate(graph_stub_dataset), indices)

    def test_save_round_trip(self, graph_stub_dataset, tmp_path):
        torch.manual_seed(0)
        model = RecDMoN(_recdmon_config())
        model.fit(graph_stub_dataset)
        path = str(tmp_path / "recdmon.pt")
        model.save(path)
        other = RecDMoN(_recdmon_config())
        other.load_state_dict(torch.load(path, weights_only=False))
        assert torch.equal(other.generate(graph_stub_dataset), model.generate(graph_stub_dataset))

    def test_indices_raises_before_fit(self):
        model = RecDMoN(_recdmon_config())
        with pytest.raises(AttributeError, match="indices"):
            _ = model.indices

    def test_max_dense_nodes_raises(self):
        model = RecDMoN(_recdmon_config(max_dense_nodes=3))
        with pytest.raises(ValueError, match="max_dense_nodes"):
            model.forward(torch.randn(6, 4), _path_adj(), np.arange(6), k=2, level=1, parent=0)

    def test_to_sets_device(self):
        model = RecDMoN(_recdmon_config())
        assert model.to("cpu") is model
        assert model.device == torch.device("cpu")


class TestSpectral:
    def _spectral(self, **kwargs):
        kwargs.setdefault("seed", 0)
        kwargs.setdefault("recursive_factor", 2)
        kwargs.setdefault("recursive_levels", 2)
        return Spectral(SpectralModelConfig(**kwargs))

    def test_forward_is_deterministic(self):
        model = self._spectral()
        args = (torch.randn(6, 4), _path_adj(), np.arange(6), 2, 1, 0)
        first = model.forward(*args)
        second = model.forward(*args)
        assert np.array_equal(first, second)
        assert first.min() >= 0 and first.max() < 2

    def test_fit_generate_and_state_round_trip(self, graph_stub_dataset):
        model = self._spectral()
        model.fit(graph_stub_dataset)
        indices = model.generate(graph_stub_dataset)
        assert indices.shape == (6, 2)
        assert indices.dtype == torch.int64
        assert model.n_codebooks == 2
        assert len(model.codebook_sizes) == 2

        other = self._spectral()
        other.load_state_dict(model.state_dict())
        assert torch.equal(other.generate(graph_stub_dataset), indices)

    def test_save_round_trip(self, graph_stub_dataset, tmp_path):
        model = self._spectral()
        model.fit(graph_stub_dataset)
        path = str(tmp_path / "spectral.pt")
        model.save(path)
        other = self._spectral()
        other.load_state_dict(torch.load(path, weights_only=False))
        assert torch.equal(other.generate(graph_stub_dataset), model.generate(graph_stub_dataset))

    def test_indices_raises_before_fit(self):
        model = self._spectral()
        with pytest.raises(AttributeError, match="indices"):
            _ = model.indices

    def test_to_sets_device(self):
        model = self._spectral()
        assert model.to("cpu") is model
        assert model.device == torch.device("cpu")
