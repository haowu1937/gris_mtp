import pytest
import torch

from gensid.nn.layers import FaissKMeans, MLPLayers, ResidualVectorQuantizer, TorchKMeans, VectorQuantizer
from gensid.nn.utils import MEAN_FUNCTIONS, cosine_distance


class TestVectorQuantizer:
    def _make(self, **kwargs):
        kwargs.setdefault("n_e", 8)
        kwargs.setdefault("e_dim", 4)
        kwargs.setdefault("sk_epsilon", 0.0)  # disable sinkhorn for speed
        return VectorQuantizer(**kwargs)

    def test_forward_shapes_and_index_bounds(self):
        torch.manual_seed(0)
        vq = self._make()
        x = torch.randn(5, 4)
        x_q, loss, indices, x_q_codebook = vq(x, label=None, idx=0, use_sk=False)
        assert x_q.shape == x.shape
        assert x_q_codebook.shape == x.shape
        assert indices.shape == (5,)
        assert indices.min() >= 0 and indices.max() < vq.n_e
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_get_codebook_returns_embedding_weight(self):
        vq = self._make()
        assert torch.equal(vq.get_codebook(), vq.embedding.weight)

    def test_get_codebook_entry(self):
        vq = self._make()
        idx = torch.tensor([0, 1, 2])
        z = vq.get_codebook_entry(idx)
        assert z.shape == (3, 4)
        assert torch.equal(z, vq.embedding.weight[idx])

    def test_get_codebook_entry_with_shape(self):
        vq = self._make()
        z = vq.get_codebook_entry(torch.tensor([1, 2]), shape=(2, 4))
        assert z.shape == (2, 4)

    def test_kmeans_init_sets_initialized_flag(self, monkeypatch):
        from gensid.nn import layers as layers_mod

        n_e, e_dim = 8, 4

        # init_emb hardcodes n_clusters=256; stub custom_constrained_km so it returns n_e
        # centers matching the embedding weight shape.
        def fake_custom_constrained_km(data, n_clusters, max_size_multiplier=10, **kwargs):
            return data[:n_e].clone(), torch.zeros(data.shape[0], dtype=torch.long)

        monkeypatch.setattr(layers_mod, "custom_constrained_km", fake_custom_constrained_km)

        vq = self._make(kmeans_init=True)
        assert vq.initialized is False
        data = torch.randn(20, e_dim)
        vq.vq_initialization(data, use_sk=False)
        assert vq.initialized is True
        assert torch.isfinite(vq.embedding.weight).all()
        assert torch.allclose(vq.embedding.weight, data[:n_e])

    def test_vq_initialization_argmin_assigns_nearest_code(self):
        torch.manual_seed(0)
        vq = self._make()
        x = torch.randn(6, 4)
        out = vq.vq_initialization(x, use_sk=False)
        # argmin path quantizes to exactly the nearest codebook row
        closest = vq.get_codebook()[torch.cdist(x, vq.get_codebook()).argmin(-1)]
        assert torch.allclose(out, closest)

    def test_center_distance_for_constraint_centers_and_scales(self):
        distances = torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
        centered = VectorQuantizer.center_distance_for_constraint(distances)
        # min=1, max=6 -> middle=3.5, amplitude=6-3.5 ~ 2.5; centered = (d - 3.5) / 2.5
        assert torch.allclose(centered, (distances - 3.5) / 2.5, atol=1e-4)


class TestResidualVectorQuantizer:
    def _make(self):
        return ResidualVectorQuantizer(
            n_e_list=[6, 6],
            e_dim=4,
            sk_epsilons=[0.0, 0.0],
            beta=0.25,
        )

    def test_get_codebook_stacks_all_layers(self):
        rq = self._make()
        codebook = rq.get_codebook()
        assert codebook.shape == (2, 6, 4)

    def test_forward_output_shapes(self):
        torch.manual_seed(0)
        rq = self._make()
        x = torch.randn(8, 4)
        x_q, loss, indices, x_q_codebooks, x_res_list = rq(x, labels=None, use_sk=False)
        assert x_q.shape == (8, 4)
        assert loss.ndim == 0
        assert indices.shape == (8, 2)
        assert x_q_codebooks.shape == (8, 2, 4)
        assert x_res_list.shape == (8, 2, 4)
        assert indices.min() >= 0 and indices.max() < 6

    def test_forward_use_sk(self):
        rq = self._make()
        rq.sk_epsilons = [0.01, 0.01]
        x = torch.randn(8, 4)
        x_q, loss, indices, _, _ = rq(x, labels=None, use_sk=True)
        assert x_q.shape == (8, 4)
        assert indices.shape == (8, 2)

    def test_vq_initialization_runs(self):
        rq = self._make()
        rq.vq_initialization(torch.randn(16, 4), use_sk=False)
        assert len(rq.vq_layers) == 2


class TestMLPLayers:
    def test_multi_layer_forward_with_relu_is_linear_at_zero(self):
        # biases init to 0 and relu(0)=0, so a 3-layer relu MLP maps zero input to exactly zero
        mlp = MLPLayers(layers=[4, 8, 2], dropout=0.0, activation="relu", bn=False)
        x = torch.zeros(3, 4)
        out = mlp(x)
        assert torch.allclose(out, torch.zeros(3, 2), atol=1e-6)
        assert torch.allclose(out, mlp(x))  # deterministic

    def test_batchnorm_layers_are_created(self):
        mlp = MLPLayers(layers=[4, 8], dropout=0.0, activation="relu", bn=True)
        assert any(isinstance(m, torch.nn.BatchNorm1d) for m in mlp.mlp_layers)

    def test_no_activation_on_last_layer_output_is_linear(self):
        mlp = MLPLayers(layers=[4, 2], dropout=0.0, activation="relu", bn=False)
        x = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        out = mlp(x)
        # zero activation layer is not applied to the final output; biases initialized to 0
        assert torch.allclose(out, torch.zeros(1, 2), atol=1e-6)

    def test_linear_weights_initialized(self):
        mlp = MLPLayers(layers=[4, 8], dropout=0.0)
        for m in mlp.mlp_layers:
            if isinstance(m, torch.nn.Linear):
                assert torch.isfinite(m.weight).all()
                assert torch.allclose(m.bias, torch.zeros_like(m.bias))


class _ReferenceKMeans:
    """Faithful copy of the pre-vectorization ``TorchKMeans`` algorithm.

    Kept here so the parity tests can prove the vectorized reimplementation
    (same RNG stream, same inputs) produces identical outputs.
    """

    def __init__(self, n_clusters, distance_function, mean_function, device):
        self.num_clusters = n_clusters
        self.distance_function = distance_function
        self.mean_function = mean_function
        self.centroids = None
        self.device = device

    def fit(self, data, max_iterations=300, tolerance=1e-4, batch_size=None, init_mode="kmeans++"):
        if batch_size is None:
            data = data.to(self.device)
        self._initialize(data, init_mode=init_mode, batch_size=batch_size)
        for _ in range(max_iterations):
            if batch_size is None:
                distances = self.distance_function(data, self.centroids)
                assignments = torch.argmin(distances, dim=1)
            else:
                n_batches = (data.shape[0] + batch_size - 1) // batch_size
                assignments = torch.zeros(data.shape[0], dtype=torch.long, device=data.device)
                for j in range(n_batches):
                    start_idx, end_idx = j * batch_size, min((j + 1) * batch_size, data.shape[0])
                    batch_data = data[start_idx:end_idx].to(self.device)
                    batch_distances = self.distance_function(batch_data, self.centroids)
                    assignments[start_idx:end_idx] = torch.argmin(batch_distances, dim=1)
            max_change = 0
            for i in range(self.num_clusters):
                mask = assignments == i
                if mask.sum() == 0:
                    continue
                new_centroid = self.mean_function(data[mask.to(data.device)].to(self.device))
                change = self.distance_function(torch.unsqueeze(new_centroid, 0), torch.unsqueeze(self.centroids[i], 0))
                max_change = max(max_change, change)
                self.centroids[i] = new_centroid
            if max_change < tolerance:
                break

    def fit_mini_batch(self, data, num_epochs, batch_size, init_mode="kmeans++", max_steps=None, init_data=None):
        data = data[torch.randperm(data.shape[0], device=data.device)]
        self._initialize(data if init_data is None else init_data, init_mode=init_mode, batch_size=batch_size)
        assignment_counts = torch.zeros(self.num_clusters, device=self.device)
        steps = 0
        for _ in range(num_epochs):
            data = data[torch.randperm(data.shape[0], device=data.device)]
            for batch_i in range(0, data.shape[0], batch_size):
                batch = data[batch_i : batch_i + batch_size].to(self.device)
                distances = self.distance_function(batch, self.centroids)
                assignments = torch.argmin(distances, dim=1)
                assignment_counts += torch.bincount(assignments, minlength=self.num_clusters)
                for i in range(self.num_clusters):
                    mask = assignments == i
                    if mask.sum() == 0:
                        continue
                    learning_rate = 1 / assignment_counts[i]
                    update_vector = self.mean_function(batch[mask.to(batch.device)].to(self.device))
                    new_centroid = (1 - learning_rate) * self.centroids[i] + learning_rate * update_vector
                    new_centroid = self.mean_function(torch.unsqueeze(new_centroid, dim=0))
                    self.centroids[i] = new_centroid
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    return

    def predict(self, data, batch_size=None):
        if batch_size is None:
            data = data.to(self.device)
            distances = self.distance_function(data, self.centroids)
        else:
            batch_distances = [
                self.distance_function(data[j : j + batch_size].to(self.device), self.centroids)
                for j in range(0, data.shape[0], batch_size)
            ]
            distances = torch.cat(batch_distances)
        return torch.argmin(distances, dim=1)

    def _initialize(self, init_data, init_mode="kmeans++", batch_size=None):
        if init_mode == "random":
            self._initialize_random(init_data)
        elif init_mode == "kmeans++":
            self._initialize_kmeanspp(init_data, batch_size)
        else:
            raise ValueError(f"Unknown k-means init_mode: {init_mode}")

    def _initialize_kmeanspp(self, data, batch_size=None):
        n, d = data.shape
        self.centroids = torch.zeros(self.num_clusters, d).to(self.device)
        if batch_size is None:
            data = data.to(self.device)
        self.centroids[0] = data[torch.randint(0, n, (1,))].to(self.device)
        for i in range(1, self.num_clusters):
            if batch_size is None:
                distances = self.distance_function(data, self.centroids[:i])
                min_distances = distances.min(dim=1)[0]
            else:
                batch_distances = [
                    self.distance_function(data[j : j + batch_size].to(self.device), self.centroids[:i])
                    for j in range(0, data.shape[0], batch_size)
                ]
                batch_min_distances = [bd.min(dim=1)[0] for bd in batch_distances]
                min_distances = torch.cat(batch_min_distances)
            min_distances.clamp_(min=1e-6)
            probabilities = min_distances**2
            probabilities = probabilities / probabilities.sum()
            index = torch.multinomial(probabilities, 1)
            self.centroids[i] = data[index.to(data.device)].to(self.device)

    def _initialize_random(self, data):
        n, _ = data.shape
        indices = torch.randperm(n)[: self.num_clusters]
        self.centroids = data[indices.to(data.device)].to(self.device)


class TestTorchKMeansParity:
    """Prove the vectorized TorchKMeans matches the original algorithm output-for-output."""

    @pytest.fixture
    def data(self):
        torch.manual_seed(7)
        a = torch.randn(40, 3) + torch.tensor([5.0, 5.0, 5.0])
        b = torch.randn(40, 3) - torch.tensor([5.0, 5.0, 5.0])
        c = torch.randn(40, 3)
        return torch.cat([a, b, c])

    def _run_pair(self, data, n_clusters, distance_function, mean_function, seed, **fit_kwargs):
        torch.manual_seed(seed)
        ref = _ReferenceKMeans(n_clusters, distance_function, mean_function, "cpu")
        ref.fit(data.clone(), **fit_kwargs)
        torch.manual_seed(seed)
        km = TorchKMeans(n_clusters, distance_function, mean_function, "cpu")
        km.fit(data.clone(), **fit_kwargs)
        return ref, km

    def test_kmeanspp_init_is_identical(self, data):
        n_clusters = 6
        torch.manual_seed(0)
        ref = _ReferenceKMeans(n_clusters, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        ref._initialize_kmeanspp(data.clone())
        torch.manual_seed(0)
        km = TorchKMeans(n_clusters, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        km._initialize_kmeanspp(data.clone())
        assert torch.equal(km.centroids, ref.centroids)

    def test_incremental_init_default_matches_reference_when_k_lt_n(self, data):
        # default path (kmeanspp_exact=False, K < N) uses the fast incremental init;
        # it must still be bit-identical to the reference full-matrix computation.
        n_clusters = 6
        assert n_clusters < data.shape[0]
        torch.manual_seed(0)
        ref = _ReferenceKMeans(n_clusters, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        ref._initialize_kmeanspp(data.clone())
        torch.manual_seed(0)
        km = TorchKMeans(n_clusters, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu", kmeanspp_exact=False)
        km._initialize_kmeanspp(data.clone())
        assert torch.equal(km.centroids, ref.centroids)

    def test_kmeanspp_exact_flag_uses_full_matrix_init(self, data):
        # kmeanspp_exact=True forces the exact full-matrix init even when K < N.
        n_clusters = 6
        torch.manual_seed(0)
        ref = _ReferenceKMeans(n_clusters, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        ref._initialize_kmeanspp(data.clone())
        torch.manual_seed(0)
        km = TorchKMeans(n_clusters, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu", kmeanspp_exact=True)
        km._initialize_kmeanspp(data.clone())
        assert torch.equal(km.centroids, ref.centroids)

    def test_k_gt_n_auto_uses_exact_full_matrix_init(self, data):
        # K > N auto-falls back to the exact full-matrix init (bit-identical).
        n_clusters = 140
        torch.manual_seed(0)
        ref = _ReferenceKMeans(n_clusters, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        ref._initialize_kmeanspp(data.clone())
        torch.manual_seed(0)
        km = TorchKMeans(n_clusters, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        km._initialize_kmeanspp(data.clone())
        assert torch.equal(km.centroids, ref.centroids)

    def test_random_init_is_identical(self, data):
        n_clusters = 6
        torch.manual_seed(0)
        ref = _ReferenceKMeans(n_clusters, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        ref._initialize_random(data.clone())
        torch.manual_seed(0)
        km = TorchKMeans(n_clusters, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        km._initialize_random(data.clone())
        assert torch.equal(km.centroids, ref.centroids)

    def test_fit_matches_reference(self, data):
        ref, km = self._run_pair(
            data, 6, torch.cdist, MEAN_FUNCTIONS["mean"], seed=0, max_iterations=100, tolerance=1e-5
        )
        assert torch.equal(km.predict(data), ref.predict(data))
        assert torch.allclose(km.centroids, ref.centroids, atol=1e-4)

    def test_fit_batch_matches_reference(self, data):
        ref, km = self._run_pair(
            data, 6, torch.cdist, MEAN_FUNCTIONS["mean"], seed=1, max_iterations=100, tolerance=1e-5, batch_size=16
        )
        assert torch.equal(km.predict(data), ref.predict(data))
        assert torch.allclose(km.centroids, ref.centroids, atol=1e-4)

    def test_fit_cosine_matches_reference(self, data):
        from gensid.nn.utils import cosine_distance

        ref, km = self._run_pair(
            data, 6, cosine_distance, MEAN_FUNCTIONS["normalized_mean"], seed=2, max_iterations=100, tolerance=1e-5
        )
        assert torch.equal(km.predict(data), ref.predict(data))
        assert torch.allclose(km.centroids, ref.centroids, atol=1e-4)

    def test_fit_mini_batch_matches_reference(self, data):
        torch.manual_seed(3)
        ref = _ReferenceKMeans(6, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        ref.fit_mini_batch(data.clone(), num_epochs=5, batch_size=16, max_steps=30)
        torch.manual_seed(3)
        km = TorchKMeans(6, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        km.fit_mini_batch(data.clone(), num_epochs=5, batch_size=16, max_steps=30)
        assert torch.equal(km.predict(data), ref.predict(data))
        assert torch.allclose(km.centroids, ref.centroids, atol=1e-4)

    def test_more_clusters_than_points_keeps_empty_cluster(self, data):
        # K > N: some clusters are empty; empty clusters must keep their init centroid (old behavior).
        n_clusters = 140
        torch.manual_seed(0)
        ref = _ReferenceKMeans(n_clusters, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        ref.fit(data.clone(), max_iterations=20)
        torch.manual_seed(0)
        km = TorchKMeans(n_clusters, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        km.fit(data.clone(), max_iterations=20)
        assert torch.equal(km.predict(data), ref.predict(data))
        assert torch.allclose(km.centroids, ref.centroids, atol=1e-4)


class TestFaissKMeans:
    faiss = pytest.importorskip("faiss")

    def _data(self):
        torch.manual_seed(0)
        a = torch.randn(60, 4) + torch.tensor([5.0, 5.0, 5.0, 5.0])
        b = torch.randn(60, 4) - torch.tensor([5.0, 5.0, 5.0, 5.0])
        c = torch.randn(60, 4)
        return torch.cat([a, b, c])

    def _run_pair(self, data, n_clusters, distance_function, mean_function, seed, **fit_kwargs):
        torch.manual_seed(seed)
        fa = FaissKMeans(n_clusters, distance_function, mean_function, "cpu")
        fa.fit(data.clone(), **fit_kwargs)
        torch.manual_seed(seed)
        km = TorchKMeans(n_clusters, distance_function, mean_function, "cpu")
        km.fit(data.clone(), **fit_kwargs)
        return fa, km

    def test_fit_matches_torch_euclidean(self):
        data = self._data()
        fa, km = self._run_pair(data, 6, torch.cdist, MEAN_FUNCTIONS["mean"], seed=0, max_iterations=50, tolerance=1e-6)
        assert torch.equal(fa.predict(data), km.predict(data))
        assert torch.allclose(fa.centroids, km.centroids, atol=1e-4)

    def test_fit_matches_torch_cosine(self):
        data = self._data()
        fa, km = self._run_pair(
            data, 6, cosine_distance, MEAN_FUNCTIONS["normalized_mean"], seed=1, max_iterations=50, tolerance=1e-6
        )
        assert torch.equal(fa.predict(data), km.predict(data))
        assert torch.allclose(fa.centroids, km.centroids, atol=1e-4)

    def test_fit_with_batch_size_matches_torch(self):
        data = self._data()
        fa, km = self._run_pair(
            data, 6, torch.cdist, MEAN_FUNCTIONS["mean"], seed=2, max_iterations=50, tolerance=1e-6, batch_size=32
        )
        assert torch.equal(fa.predict(data), km.predict(data))
        assert torch.allclose(fa.centroids, km.centroids, atol=1e-4)

    def test_centroids_set_directly_then_predict_matches_torch(self):
        # HierarchicalRKMeans / load() set `.centroids` directly then call predict.
        data = self._data()
        torch.manual_seed(0)
        km = TorchKMeans(6, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        km.fit(data.clone(), max_iterations=30)
        fa = FaissKMeans(6, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        fa.centroids = km.centroids.clone()
        assert torch.equal(fa.predict(data), km.predict(data))

    def test_deterministic_given_same_seed(self):
        data = self._data()
        torch.manual_seed(0)
        fa1 = FaissKMeans(6, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        fa1.fit(data.clone(), max_iterations=30)
        torch.manual_seed(0)
        fa2 = FaissKMeans(6, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        fa2.fit(data.clone(), max_iterations=30)
        assert torch.equal(fa1.predict(data), fa2.predict(data))
        assert torch.allclose(fa1.centroids, fa2.centroids, atol=1e-6)

    def test_to_ignores_requested_device_and_stays_cpu(self):
        data = self._data()
        fa = FaissKMeans(6, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        fa.fit(data, max_iterations=10)
        fa.to("cuda:0")
        assert fa.device == torch.device("cpu")
        assert fa.centroids.device.type == "cpu"
        assert fa.predict(data).device.type == "cpu"

    def test_unsupported_distance_function_raises(self):
        with pytest.raises(ValueError, match="supports only euclidean"):
            FaissKMeans(4, lambda x, y: torch.zeros(x.shape[0], y.shape[0]), MEAN_FUNCTIONS["mean"], "cpu")


class TestTorchKMeans:
    def _data(self):
        torch.manual_seed(0)
        a = torch.randn(30, 2) + torch.tensor([5.0, 5.0])
        b = torch.randn(30, 2) - torch.tensor([5.0, 5.0])
        return torch.cat([a, b])

    def test_predict_before_fit_raises(self):
        km = TorchKMeans(2, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        with pytest.raises(ValueError, match="not fitted"):
            km.predict(self._data())

    def test_fit_predict_recovers_blobs(self):
        data = self._data()
        km = TorchKMeans(2, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        labels = km.fit_predict(data, max_iterations=100, tolerance=1e-6)
        # first 30 points are blob A -> one cluster; second 30 -> the other
        first_cluster = set(labels[:30].tolist())
        second_cluster = set(labels[30:].tolist())
        assert len(first_cluster) == 1 and len(second_cluster) == 1
        assert first_cluster != second_cluster

    def test_fit_predict_with_batch_size(self):
        data = self._data()
        km = TorchKMeans(2, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        labels = km.fit_predict(data, max_iterations=50, batch_size=16)
        assert labels.shape == (60,)
        assert set(labels.tolist()) == {0, 1}

    def test_fit_mini_batch_assigns_all_points(self):
        data = self._data()
        km = TorchKMeans(2, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        km.fit_mini_batch(data, num_epochs=5, batch_size=16, init_mode="kmeans++", max_steps=20)
        labels = km.predict(data)
        assert labels.shape == (60,)
        assert set(labels.tolist()) == {0, 1}

    def test_random_init_mode(self):
        # init_mode is an argument of `fit`, not `fit_predict`
        data = self._data()
        km = TorchKMeans(2, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        km.fit(data, max_iterations=100, init_mode="random")
        labels = km.predict(data)
        assert labels.shape == (60,)

    def test_unknown_init_mode_raises(self):
        km = TorchKMeans(2, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        with pytest.raises(ValueError, match="Unknown k-means init_mode"):
            km.fit(self._data(), max_iterations=1, init_mode="bogus")

    def test_to_returns_none_after_fit(self):
        data = self._data()
        km = TorchKMeans(2, torch.cdist, MEAN_FUNCTIONS["mean"], "cpu")
        km.fit(data, max_iterations=10)
        # `to` is a no-op returning None; ensure it doesn't raise
        assert km.to("cpu") is None
