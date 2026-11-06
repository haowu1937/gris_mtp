import argparse
import logging

import pytest
import torch

from gensid.config.model_config import OPQModelConfig, PQModelConfig
from gensid.method.pq import OPQMethod, PQMethod


@pytest.fixture
def data():
    torch.manual_seed(0)
    return torch.randn(400, 16)


@pytest.fixture
def structured_data():
    """Low-rank data: the OPQ rotation must reduce distortion vs plain PQ."""
    torch.manual_seed(0)
    latent = torch.randn(400, 2)
    proj = torch.randn(2, 16)
    return latent @ proj


def _pq_config(**kwargs):
    kwargs.setdefault("in_dim", 16)
    kwargs.setdefault("n_subquantizers", 4)
    kwargs.setdefault("n_bits", 4)
    kwargs.setdefault("quantizer_backend", "torch")
    kwargs.setdefault("kmeans_iters", 10)
    return PQModelConfig(**kwargs)


def _opq_config(**kwargs):
    kwargs.setdefault("in_dim", 16)
    kwargs.setdefault("n_subquantizers", 4)
    kwargs.setdefault("n_bits", 4)
    kwargs.setdefault("quantizer_backend", "torch")
    kwargs.setdefault("niter", 3)
    kwargs.setdefault("kmeans_iters", 10)
    return OPQModelConfig(**kwargs)


class TestPQMethod:
    def test_codebook_sizes_and_count(self):
        method = PQMethod(_pq_config(n_subquantizers=4, n_bits=8))
        assert method.n_codebooks == 4
        assert method.codebook_sizes == [256, 256, 256, 256]

    def test_fit_generate_shape_dtype_bounds(self, data):
        method = PQMethod(_pq_config())
        method.fit(data)
        codes = method.generate(data)
        assert codes.shape == (400, 4)
        assert codes.dtype == torch.int64
        assert codes.min() >= 0 and codes.max() < 16

    def test_generate_before_fit_raises(self, data):
        method = PQMethod(_pq_config())
        with pytest.raises(ValueError, match="not fitted"):
            method.generate(data)

    def test_more_bits_reduce_distortion(self, data):
        low = PQMethod(_pq_config(n_bits=4)).fit(data)
        high = PQMethod(_pq_config(n_bits=8)).fit(data)
        assert high.quantizer.distortion(data) < low.quantizer.distortion(data)

    def test_state_dict_roundtrip_identical_codes(self, data):
        method = PQMethod(_pq_config())
        method.fit(data)
        codes_before = method.generate(data)
        state = method.state_dict()
        assert set(state.keys()) == {"centroids", "name", "quantizer_backend"}

        loaded = PQMethod(_pq_config())
        loaded.load_state_dict(state)
        assert torch.equal(loaded.generate(data), codes_before)

    def test_from_args(self):
        args = argparse.Namespace(in_dim=32, n_subquantizers=8, n_bits=4, quantizer_backend="faiss", kmeans_iters=15)
        config = PQModelConfig.from_args(args)
        assert config == PQModelConfig(
            in_dim=32, n_subquantizers=8, n_bits=4, quantizer_backend="faiss", kmeans_iters=15
        )

    def test_from_args_missing_fields_fall_back(self):
        config = PQModelConfig.from_args(argparse.Namespace(n_subquantizers=2))
        assert config.n_subquantizers == 2
        assert config.n_bits == 8
        assert config.quantizer_backend == "torch"

    def test_n_subquantizers_must_divide_in_dim(self, data):
        method = PQMethod(_pq_config(in_dim=16, n_subquantizers=3))
        with pytest.raises(ValueError, match="divisible"):
            method.fit(data)

    def test_faiss_backend_dispatch(self, data):
        pytest.importorskip("faiss")
        from gensid.nn.pq import FaissPQ

        method = PQMethod(_pq_config(quantizer_backend="faiss"))
        method.fit(data)
        assert isinstance(method.quantizer, FaissPQ)
        codes = method.generate(data)
        assert codes.shape == (400, 4)
        assert codes.min() >= 0 and codes.max() < 16


class TestOPQMethod:
    def test_codebook_sizes_and_count(self):
        method = OPQMethod(_opq_config(n_subquantizers=4, n_bits=8))
        assert method.n_codebooks == 4
        assert method.codebook_sizes == [256, 256, 256, 256]

    def test_fit_generate_shape_dtype_bounds(self, structured_data):
        method = OPQMethod(_opq_config())
        method.fit(structured_data)
        codes = method.generate(structured_data)
        assert codes.shape == (400, 4)
        assert codes.dtype == torch.int64
        assert codes.min() >= 0 and codes.max() < 16

    def test_rotation_reduces_distortion(self, structured_data):
        pq = PQMethod(_pq_config()).fit(structured_data)
        opq = OPQMethod(_opq_config()).fit(structured_data)
        assert opq.quantizer.distortion(structured_data) < pq.quantizer.distortion(structured_data)

    def test_state_dict_roundtrip_identical_codes(self, structured_data):
        method = OPQMethod(_opq_config())
        method.fit(structured_data)
        codes_before = method.generate(structured_data)
        state = method.state_dict()
        assert set(state.keys()) == {"A", "centroids", "name", "quantizer_backend"}

        loaded = OPQMethod(_opq_config())
        loaded.load_state_dict(state)
        assert torch.equal(loaded.generate(structured_data), codes_before)

    def test_to_after_load_state_dict_moves_rotation(self, structured_data):
        # regression: generation on CUDA failed because load_state_dict builds the
        # quantizer on CPU and .to(device) only moved the centroids, not A
        method = OPQMethod(_opq_config())
        method.fit(structured_data)
        loaded = OPQMethod(_opq_config())
        loaded.load_state_dict(method.state_dict())
        loaded.to(torch.device("meta"))
        assert loaded.quantizer.A.device.type == "meta"
        assert loaded.quantizer.centroids.device.type == "meta"

    def test_from_args(self):
        args = argparse.Namespace(
            in_dim=32, n_subquantizers=8, n_bits=4, quantizer_backend="faiss", opq_niter=7, kmeans_iters=15
        )
        config = OPQModelConfig.from_args(args)
        assert config == OPQModelConfig(
            in_dim=32, n_subquantizers=8, n_bits=4, quantizer_backend="faiss", niter=7, kmeans_iters=15
        )

    def test_faiss_backend_dispatch(self, structured_data):
        pytest.importorskip("faiss")
        from gensid.nn.pq import FaissOPQ

        method = OPQMethod(_opq_config(quantizer_backend="faiss"))
        method.fit(structured_data)
        assert isinstance(method.quantizer, FaissOPQ)
        assert method.generate(structured_data).shape == (400, 4)


class TestPQCheckpointLoadValidation:
    def test_state_dict_persists_method_metadata(self, data):
        pq = PQMethod(_pq_config()).fit(data)
        state = pq.state_dict()
        assert state["name"] == "pq"
        assert state["quantizer_backend"] == "torch"

        opq = OPQMethod(_opq_config()).fit(data)
        state = opq.state_dict()
        assert state["name"] == "opq"
        assert state["quantizer_backend"] == "torch"

    def test_load_pq_checkpoint_into_opq_raises(self, data):
        pq_state = PQMethod(_pq_config()).fit(data).state_dict()
        with pytest.raises(ValueError, match="SID method 'pq' but the model is 'opq'"):
            OPQMethod(_opq_config()).load_state_dict(pq_state)

    def test_load_opq_checkpoint_into_pq_raises(self, data):
        opq_state = OPQMethod(_opq_config()).fit(data).state_dict()
        with pytest.raises(ValueError, match="SID method 'opq' but the model is 'pq'"):
            PQMethod(_pq_config()).load_state_dict(opq_state)

    def test_load_pq_state_without_metadata_into_opq_raises(self, data):
        # pre-metadata checkpoints carry no 'name'; the rotation check must catch them
        pq_state = PQMethod(_pq_config()).fit(data).state_dict()
        del pq_state["name"]
        del pq_state["quantizer_backend"]
        with pytest.raises(ValueError, match="missing the 'A' rotation"):
            OPQMethod(_opq_config()).load_state_dict(pq_state)

    def test_load_opq_state_without_metadata_into_pq_raises(self, data):
        opq_state = OPQMethod(_opq_config()).fit(data).state_dict()
        del opq_state["name"]
        del opq_state["quantizer_backend"]
        with pytest.raises(ValueError, match="which has no rotation"):
            PQMethod(_pq_config()).load_state_dict(opq_state)

    def test_load_method_name_mismatch_raises(self, data):
        state = PQMethod(_pq_config()).fit(data).state_dict()
        state["name"] = "rqvae"
        with pytest.raises(ValueError, match="SID method 'rqvae'.*'pq'"):
            PQMethod(_pq_config()).load_state_dict(state)

    def test_load_legacy_faiss_index_format_raises(self, data):
        legacy = {"faiss_index": torch.zeros(64, dtype=torch.uint8)}
        with pytest.raises(ValueError, match="legacy faiss_index format"):
            PQMethod(_pq_config()).load_state_dict(legacy)

    def test_load_missing_centroids_raises(self):
        with pytest.raises(ValueError, match="missing 'centroids'"):
            PQMethod(_pq_config()).load_state_dict({"A": torch.eye(16)})

    def test_load_backend_mismatch_warns_but_loads(self, data, caplog):
        method = PQMethod(_pq_config())
        method.fit(data)
        state = method.state_dict()
        state["quantizer_backend"] = "faiss"
        with caplog.at_level(logging.WARNING, logger="gensid.method.pq"):
            loaded = PQMethod(_pq_config())
            loaded.load_state_dict(state)
        assert any("quantizer_backend='faiss' differs" in r.message for r in caplog.records)
        assert torch.equal(loaded.generate(data), method.generate(data))
