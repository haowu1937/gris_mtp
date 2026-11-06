import pytest
import torch

from gensid.config.model_config import HierarchicalRKMeansModelConfig, RKMeansModelConfig
from gensid.method.rkmeans import HierarchicalRKMeans, RKMeans


class TestRKMeans:
    @pytest.fixture
    def blobs(self):
        torch.manual_seed(0)
        a = torch.randn(20, 2) + torch.tensor([5.0, 5.0])
        b = torch.randn(20, 2) - torch.tensor([5.0, 5.0])
        return torch.cat([a, b])

    def _config(self, **kwargs):
        kwargs.setdefault("depth", 2)
        kwargs.setdefault("n_clusters", 2)
        kwargs.setdefault("distance_function", "euclidean")
        kwargs.setdefault("mean_function", "mean")
        kwargs.setdefault("norm_input", False)
        return RKMeansModelConfig(**kwargs)

    def _make(self, **kwargs):
        return RKMeans(self._config(**kwargs))

    def test_normalize_residuals_for_cosine(self):
        km = RKMeans(
            RKMeansModelConfig(depth=1, n_clusters=2, distance_function="cosine", mean_function="normalized_mean")
        )
        assert km.normalize_residuals is True
        assert km.norm_input is True

    def test_norm_input_override(self):
        km = self._make(norm_input=False)
        assert km.norm_input is False

    def test_fit_and_generate_index_shape(self, blobs):
        km = self._make()
        km._fit(blobs)
        indexes = km.generate(blobs)
        assert indexes.shape == (40, 2)
        assert indexes.dtype == torch.int64
        assert indexes.min() >= 0 and indexes.max() < 2

    def test_single_cluster_centroid_equals_mean(self):
        # k=1 k-means must converge to the data mean (mean_function="mean", no input norm).
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
        km = self._make(depth=1, n_clusters=1)
        km._fit(data)
        assert torch.allclose(km.k_means[0].centroids, data.mean(dim=0, keepdim=True), atol=1e-5)

    def test_generate_is_predict_at_depth_one(self, blobs):
        # no residual subtraction at depth 1, so generate == raw k-means prediction
        km = self._make(depth=1, n_clusters=2)
        km._fit(blobs)
        assert torch.equal(km.generate(blobs), km.k_means[0].predict(blobs).unsqueeze(-1))

    def test_separated_blobs_get_distinct_clusters(self, blobs):
        # two tight blobs far apart: every point of a blob maps to a single, distinct cluster
        km = self._make(depth=1, n_clusters=2)
        km._fit(blobs)
        indexes = km.generate(blobs).squeeze(-1)
        blob_a = set(indexes[:20].tolist())
        blob_b = set(indexes[20:].tolist())
        assert len(blob_a) == 1 and len(blob_b) == 1
        assert blob_a != blob_b

    def test_cosine_pipeline_runs(self, blobs):
        km = self._make(distance_function="cosine", mean_function="normalized_mean")
        km._fit(blobs)
        assert km.generate(blobs).shape == (40, 2)

    def test_fit_mini_batch(self, blobs):
        km = self._make()
        km.fit_mini_batch(blobs, batch_size=8)
        assert len(km.k_means) == 2
        assert km.generate(blobs).shape == (40, 2)

    def test_state_dict_keys_are_depth_indices(self, blobs):
        km = self._make()
        km._fit(blobs)
        sd = km.state_dict()
        assert set(sd.keys()) == {0, 1}
        assert sd[0].shape[0] == 2

    def test_save_load_round_trip(self, blobs, tmp_path):
        km = self._make()
        km._fit(blobs)
        path = str(tmp_path / "kmeans.pt")
        km.save(path)
        loaded = self._make()
        loaded.load(path)
        assert len(loaded.k_means) == 2
        assert torch.equal(loaded.generate(blobs), km.generate(blobs))

    def test_load_mismatched_depth_raises(self, blobs, tmp_path):
        km = self._make()
        km._fit(blobs)
        path = str(tmp_path / "kmeans.pt")
        km.save(path)
        shallow = RKMeans(
            RKMeansModelConfig(
                depth=1, n_clusters=2, distance_function="euclidean", mean_function="mean", norm_input=False
            )
        )
        with pytest.raises(AssertionError):
            shallow.load(path)

    def test_load_state_dict(self, blobs):
        km = self._make()
        km._fit(blobs)
        sd = km.state_dict()
        other = self._make()
        other.load_state_dict(sd)
        assert torch.equal(other.generate(blobs), km.generate(blobs))

    def test_to_returns_self(self, blobs):
        km = self._make()
        km._fit(blobs)
        assert km.to("cpu") is km

    def test_faiss_backend_matches_torch_backend(self, blobs):
        pytest.importorskip("faiss")
        torch.manual_seed(0)
        torch_km = self._make()
        torch_km._fit(blobs)
        faiss_km = self._make(kmeans_backend="faiss")
        faiss_km._fit(blobs)
        assert torch.equal(faiss_km.generate(blobs), torch_km.generate(blobs))
        assert faiss_km.k_means[0].device.type == "cpu"

    def test_faiss_backend_matches_torch_mini_batch(self, blobs):
        # fit_mini_batch routes every batch's assignment through _make_kmeans -> FaissKMeans
        pytest.importorskip("faiss")
        torch.manual_seed(0)
        torch_km = self._make(fit_mini_batch=True)
        torch_km.fit(blobs, batch_size=8)
        torch.manual_seed(0)
        faiss_km = self._make(fit_mini_batch=True, kmeans_backend="faiss")
        faiss_km.fit(blobs, batch_size=8)
        assert torch.equal(faiss_km.generate(blobs), torch_km.generate(blobs))
        assert all(k.device.type == "cpu" for k in faiss_km.k_means)

    def test_kmeans_backend_defaults_to_torch(self):
        assert self._make().kmeans_backend == "torch"

    def test_kmeanspp_exact_defaults_to_false(self):
        assert self._make().kmeanspp_exact is False

    def test_kmeanspp_exact_forces_full_matrix_init(self, blobs):
        # with kmeanspp_exact=True the method's k-means objects use the exact init path
        km = self._make(kmeanspp_exact=True)
        km._fit(blobs)
        assert all(k.kmeanspp_exact for k in km.k_means)
        assert km.generate(blobs).shape == (40, 2)


class TestHierarchicalRKMeans:
    @pytest.fixture
    def blobs(self):
        torch.manual_seed(0)
        a = torch.randn(15, 2) + torch.tensor([5.0, 5.0])
        b = torch.randn(15, 2) - torch.tensor([5.0, 5.0])
        return torch.cat([a, b])

    def _make(self, n_clusters=2, depth=1, **kwargs):
        if isinstance(n_clusters, int):
            n_clusters = [n_clusters] * depth
        kwargs.setdefault("distance_function", "euclidean")
        kwargs.setdefault("mean_function", "mean")
        kwargs.setdefault("norm_input", False)
        return HierarchicalRKMeans(HierarchicalRKMeansModelConfig(depth=depth, n_clusters=n_clusters, **kwargs))

    def test_int_n_clusters_expanded(self):
        km = HierarchicalRKMeans(
            HierarchicalRKMeansModelConfig(
                depth=3, n_clusters=4, distance_function="euclidean", mean_function="mean", norm_input=False
            )
        )
        assert km.n_clusters == [4, 4, 4]

    def test_bad_n_clusters_length_raises(self):
        with pytest.raises(ValueError, match="length `depth`"):
            HierarchicalRKMeans(
                HierarchicalRKMeansModelConfig(
                    depth=3, n_clusters=[4, 4], distance_function="euclidean", mean_function="mean", norm_input=False
                )
            )

    def test_fit_generate(self, blobs):
        km = self._make()
        km._fit(blobs)
        indexes = km.generate(blobs)
        assert indexes.shape == (30, 1)
        assert set(indexes.squeeze().tolist()) == {0, 1}

    def test_fit_generate_two_levels(self, blobs):
        km = self._make(n_clusters=2, depth=2)
        km._fit(blobs)
        indexes = km.generate(blobs)
        assert indexes.shape == (30, 2)
        assert indexes.min() >= 0 and indexes.max() < 2

    def test_fit_with_unique_values_fewer_than_clusters(self):
        # crafted data with exactly 2 unique rows and n_clusters=4 -> unique-branch
        data = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        km = self._make(n_clusters=4)
        km._fit(data)
        assert km.generate(data).shape == (4, 1)

    def test_fit_mini_batch_not_implemented(self, blobs):
        km = self._make()
        with pytest.raises(NotImplementedError):
            km.fit_mini_batch(blobs, batch_size=8)

    def test_faiss_backend_matches_torch_generate(self, blobs):
        # recursive branch: per-prefix fit_predict via _make_kmeans -> FaissKMeans
        pytest.importorskip("faiss")
        torch.manual_seed(0)
        torch_km = self._make(n_clusters=2, depth=2)
        torch_km._fit(blobs)
        torch.manual_seed(0)
        faiss_km = self._make(n_clusters=2, depth=2, kmeans_backend="faiss")
        faiss_km._fit(blobs)
        assert torch.equal(faiss_km.generate(blobs), torch_km.generate(blobs))
        assert all(k.device.type == "cpu" for k in faiss_km.k_means.values())

    def test_faiss_backend_matches_torch_unique_branch(self):
        # n_unique < n_clusters branch: centroids set directly, then predict via FaissKMeans
        pytest.importorskip("faiss")
        data = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        torch_km = self._make(n_clusters=4)
        torch_km._fit(data)
        faiss_km = self._make(n_clusters=4, kmeans_backend="faiss")
        faiss_km._fit(data)
        assert torch.equal(faiss_km.generate(data), torch_km.generate(data))
        assert all(k.device.type == "cpu" for k in faiss_km.k_means.values())

    def test_state_dict_prefix_keys(self, blobs):
        km = self._make()
        km._fit(blobs)
        sd = km.state_dict()
        assert () in sd
        assert sd[()].shape[0] == 2

    def test_state_dict_save_load(self, blobs, tmp_path):
        km = self._make(n_clusters=2, depth=1)
        km._fit(blobs)
        path = str(tmp_path / "hkmeans.pt")
        km.save(path)
        other = self._make(n_clusters=2, depth=1)
        other.load(path)
        assert torch.equal(other.generate(blobs), km.generate(blobs))

    def test_load_state_dict(self, blobs):
        km = self._make()
        km._fit(blobs)
        other = self._make()
        other.load_state_dict(km.state_dict())
        assert torch.equal(other.generate(blobs), km.generate(blobs))

    def test_to_returns_self(self, blobs):
        km = self._make()
        km._fit(blobs)
        assert km.to("cpu") is km
