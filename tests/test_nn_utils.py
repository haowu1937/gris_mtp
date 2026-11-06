import builtins
import math
from unittest import mock

import numpy as np
import pytest
import torch
from torch import nn

from gensid.nn.utils import (
    DISTANCE_FUNCTIONS,
    MEAN_FUNCTIONS,
    _get_faiss,
    activation_layer,
    custom_constrained_km,
    cosine_distance,
    normalized_mean,
    sinkhorn_algorithm,
)


_real_import = builtins.__import__


class TestCosineDistance:
    def test_normalized_vectors_distance(self):
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[1.0, 0.0]])
        assert cosine_distance(a, b)[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b = torch.tensor([[0.0, 1.0]])
        dist = cosine_distance(a, b)
        assert dist[0, 0] == pytest.approx(1.0, abs=1e-6)  # (1,0) vs (0,1)
        assert dist[1, 0] == pytest.approx(0.0, abs=1e-6)  # (0,1) vs (0,1)

    def test_opposite_vectors(self):
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[-1.0, 0.0]])
        assert cosine_distance(a, b)[0, 0] == pytest.approx(2.0, abs=1e-6)

    def test_output_shape_is_b_x_k(self):
        a = torch.randn(5, 3)
        b = torch.randn(7, 3)
        assert cosine_distance(a, b).shape == (5, 7)

    def test_scale_invariance(self):
        a = torch.tensor([[2.0, 0.0]])
        b = torch.tensor([[1.0, 0.0]])
        assert cosine_distance(a, b)[0, 0] == pytest.approx(0.0, abs=1e-6)


class TestNormalizedMean:
    def test_returns_unit_norm_vector(self):
        x = torch.randn(5, 4)
        out = normalized_mean(x)
        assert out.shape == (4,)
        assert out.norm().item() == pytest.approx(1.0, abs=1e-6)

    def test_matches_manual_normalization(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mean = x.mean(dim=0)
        expected = mean / mean.norm()
        assert normalized_mean(x).allclose(expected)


class TestDistanceFunctionRegistry:
    def test_registry_contains_expected_entries(self):
        assert set(DISTANCE_FUNCTIONS) == {"cosine", "euclidean"}

    def test_euclidean_matches_cdist(self):
        a = torch.randn(4, 3)
        b = torch.randn(5, 3)
        assert torch.allclose(DISTANCE_FUNCTIONS["euclidean"](a, b), torch.cdist(a, b))


class TestMeanFunctionRegistry:
    def test_registry_entries(self):
        assert set(MEAN_FUNCTIONS) == {"mean", "normalized_mean"}
        x = torch.randn(5, 3)
        assert MEAN_FUNCTIONS["mean"](x).allclose(x.mean(dim=0))


class TestActivationLayer:
    @pytest.mark.parametrize(
        ("name", "expected_cls"),
        [
            ("sigmoid", nn.Sigmoid),
            ("tanh", nn.Tanh),
            ("relu", nn.ReLU),
            ("leakyrelu", nn.LeakyReLU),
            ("gelu", nn.GELU),
            ("elu", nn.ELU),
        ],
    )
    def test_named_activations(self, name, expected_cls):
        assert isinstance(activation_layer(name), expected_cls)

    def test_case_insensitive(self):
        assert isinstance(activation_layer("ReLU"), nn.ReLU)

    def test_none_returns_none(self):
        assert activation_layer(None) is None

    def test_unknown_string_returns_none(self):
        assert activation_layer("bogus") is None

    def test_module_class_passed_instantiates(self):
        assert isinstance(activation_layer(nn.ELU), nn.ELU)

    def test_unexpected_type_raises(self):
        with pytest.raises(NotImplementedError):
            activation_layer(12345)


class TestSinkhornAlgorithm:
    def test_output_is_assignment_like(self):
        torch.manual_seed(0)
        distances = torch.randn(8, 4).abs()
        Q = sinkhorn_algorithm(distances, epsilon=0.01, sinkhorn_iterations=10)
        assert Q.shape == (8, 4)
        assert torch.isfinite(Q).all()
        # soft-assignment mass must be non-negative
        assert (Q >= 0).all()
        # no degenerate all-zero columns
        assert (Q.sum(dim=0) > 0).all()

    def test_closer_points_get_more_mass(self):
        # Two points, two assignments; point 0 close to centroid 0, point 1 close to centroid 1
        distances = torch.tensor([[0.1, 10.0], [10.0, 0.1]], dtype=torch.float64)
        Q = sinkhorn_algorithm(distances, epsilon=0.001, sinkhorn_iterations=100)
        assert Q[0, 0] > Q[0, 1]
        assert Q[1, 1] > Q[1, 0]


class TestCustomConstrainedKm:
    def test_returns_centers_and_labels(self):
        torch.manual_seed(0)
        # Two well separated blobs so constrained k-means finds them
        blob_a = torch.randn(10, 2) + torch.tensor([5.0, 5.0])
        blob_b = torch.randn(10, 2) - torch.tensor([5.0, 5.0])
        data = torch.cat([blob_a, blob_b])
        centers, labels = custom_constrained_km(data, n_clusters=2, max_iter=10, n_init=2)
        assert centers.shape == (2, 2)
        assert len(labels) == len(data)
        assert set(labels) == {0, 1}

    def test_centers_are_close_to_blob_centroids(self):
        blob_a = torch.randn(20, 2) + torch.tensor([5.0, 5.0])
        blob_b = torch.randn(20, 2) - torch.tensor([5.0, 5.0])
        data = torch.cat([blob_a, blob_b])
        centers, _ = custom_constrained_km(data, n_clusters=2, max_iter=10, n_init=2)
        # one center near each blob
        distances = torch.cdist(centers, torch.stack([blob_a.mean(0), blob_b.mean(0)]))
        assert distances.min(0).values.max() < 1.0

    def test_tight_cap_never_runs_out_of_capacity(self):
        # All points in one blob so plain KMeans over-fills a single cluster; the greedy
        # enforcement must cap it without ever running out of non-full clusters (which
        # used to raise "argmin of an empty sequence" when size_max was too small).
        torch.manual_seed(0)
        data = torch.randn(100, 2) + torch.tensor([5.0, 5.0])
        n_clusters = 4
        centers, labels = custom_constrained_km(data, n_clusters=n_clusters, max_size_multiplier=1.0, n_init=2)
        counts = np.bincount(labels, minlength=n_clusters)
        size_max = max(n_clusters * 1.0, math.ceil(len(data) / n_clusters))
        assert centers.shape == (n_clusters, 2)
        assert len(labels) == len(data)
        assert set(labels) <= set(range(n_clusters))
        assert counts.max() <= size_max
        # every point must have been assigned to a non-empty cluster
        assert counts.sum() == len(data)


class TestGetFaiss:
    def _patch_faiss_import(self, exc):
        def fake_import(name, *args, **kwargs):
            if name == "faiss":
                raise exc
            return _real_import(name, *args, **kwargs)

        return mock.patch("builtins.__import__", side_effect=fake_import)

    def test_not_installed_raises_install_hint(self):
        # A plain "No module named 'faiss'" means the package really is missing.
        err = ModuleNotFoundError("No module named 'faiss'", name="faiss")
        with self._patch_faiss_import(err):
            with pytest.raises(ImportError, match="install it with"):
                _get_faiss()

    def test_installed_but_broken_surfaces_real_error(self):
        # faiss present but failing to load (e.g. missing shared library / numpy ABI
        # mismatch) must NOT be reported as "not installed" -- the real cause is shown.
        err = ImportError("libgomp.so.1: cannot open shared object file")
        with self._patch_faiss_import(err):
            with pytest.raises(ImportError, match="failed to load") as exc_info:
                _get_faiss()
        message = str(exc_info.value)
        assert "libgomp.so.1: cannot open shared object file" in message
        assert "ImportError" in message
        assert "install it with" not in message
