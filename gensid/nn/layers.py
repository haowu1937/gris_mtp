import logging
import random
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_normal_
from tqdm import tqdm

from gensid.nn.utils import (
    _get_faiss,
    MEAN_FUNCTIONS,
    activation_layer,
    custom_constrained_km,
    cosine_distance,
    sinkhorn_algorithm,
)

logger = logging.getLogger(__name__)


class LocalAssignNet(nn.Module):
    """Small two-layer MLP mapping node features to cluster-assignment logits.

    Used as the local DMoN assigner: ``Linear(in_dim, hidden_dim) -> ReLU -> Dropout
    -> Linear(hidden_dim, k)``, producing one raw logit per cluster.

    Args:
        in_dim (int): Input feature dimensionality.
        hidden_dim (int): Hidden layer size.
        k (int): Number of clusters / output logits.
        dropout (float): Dropout probability applied after the ReLU.
    """

    def __init__(self, in_dim: int, hidden_dim: int, k: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, k),
        )

    def forward(self, x):
        return self.net(x)


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        n_e_list: list[int],
        e_dim: int,
        sk_epsilons: list[float],
        beta: int = 1,
        kmeans_init: bool = False,
        kmeans_iters: int = 100,
        sk_iters: int = 100,
    ):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.vq_layers = nn.ModuleList(
            [
                VectorQuantizer(
                    n_e,
                    e_dim,
                    beta=beta,
                    kmeans_init=self.kmeans_init,
                    kmeans_iters=self.kmeans_iters,
                    sk_epsilon=sk_epsilon,
                    sk_iters=sk_iters,
                )
                for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)
            ]
        )

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def vq_initialization(self, x, use_sk):
        x_q = 0
        residual = x
        for idx, quantizer in enumerate(self.vq_layers):
            x_res = quantizer.vq_initialization(residual, use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

    def forward(self, x, labels, use_sk=True):
        all_losses = []
        all_indices = []
        all_quantized = []
        all_res = []

        x_q = 0
        residual = x

        for idx, quantizer in enumerate(self.vq_layers):
            # NOTE part of slow loss
            # label = labels[str(idx)]
            label = None

            x_res, loss, indices, x_res_codebook = quantizer(residual, label, idx, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)
            all_quantized.append(x_res_codebook)
            all_res.append(x_res)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        all_quantized = torch.stack(all_quantized, dim=1)
        all_res = torch.stack(all_res, dim=1)

        return x_q, mean_losses, all_indices, all_quantized, all_res


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_e: int,
        e_dim: int,
        mu: float = 0.25,
        beta: int = 1,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        sk_epsilon: int = 0.01,
        sk_iters: int = 100,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.mu = mu
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initialized = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initialized = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):
        centers, _ = custom_constrained_km(data, n_clusters=self.n_e, max_size_multiplier=4)
        self.embedding.weight.data.copy_(centers)
        self.initialized = True

    def diversity_loss(self, x_q, indices, indices_cluster, indices_list):
        emb = self.embedding.weight
        temp = 1

        pos_list = [indices_list[i] for i in indices_cluster]
        pos_sample = []
        for idx, pos in enumerate(pos_list):
            random_element = random.choice(pos)

            while random_element == indices[idx]:
                random_element = random.choice(pos)
            pos_sample.append(random_element)

        y_true = torch.tensor(pos_sample, device=x_q.device)

        # sim = F.cosine_similarity(x_q, emb, dim=-1)
        sim = torch.matmul(x_q, emb.t())

        # sampled_ids = torch.multinomial(best_scores, num_samples=1)
        sim_self = torch.zeros_like(sim)
        for idx, row in enumerate(sim_self):
            sim_self[idx, indices[idx]] = 1e12
        sim = sim - sim_self
        sim = sim / temp
        loss = F.cross_entropy(sim, y_true)

        return loss

    def diversity_loss_main_entry(self, x, x_q, indices, labels):

        indices_cluster = [labels[idx.item()] for idx in indices]
        target_numbers = list(range(10))
        indices_list = {}
        for target_number in target_numbers:
            indices_list[target_number] = [index for index, num in enumerate(labels) if num == target_number]

        diversity_loss = self.diversity_loss(x_q, indices, indices_cluster, indices_list)

        return diversity_loss

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def vq_initialization(self, x, use_sk=True):
        latent = x.view(-1, self.e_dim)

        if not self.initialized:
            self.init_emb(latent)

        _distance_flag = "distance"

        if _distance_flag == "distance":
            d = (
                torch.sum(latent**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
                - 2 * torch.matmul(latent, self.embedding.weight.t())
            )
        else:
            # Calculate Cosine Similarity
            d = latent @ self.embedding.weight.t()

        if not use_sk or self.sk_epsilon <= 0:
            if _distance_flag == "distance":
                indices = torch.argmin(d, dim=-1)
            else:
                indices = torch.argmax(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()

            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                logger.warning("Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        return x_q

    def forward(self, x, label, idx, use_sk=True):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initialized and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        _distance_flag = "distance"

        if _distance_flag == "distance":
            d = (
                torch.sum(latent**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
                - 2 * torch.matmul(latent, self.embedding.weight.t())
            )
        else:
            # Calculate Cosine Similarity
            d = latent @ self.embedding.weight.t()
        if not use_sk or self.sk_epsilon <= 0:
            if _distance_flag == "distance":
                if idx != -1:
                    indices = torch.argmin(d, dim=-1)
                else:
                    temp = 1.0
                    prob_dist = F.softmax(-d / temp, dim=1)
                    indices = torch.multinomial(prob_dist, 1).squeeze()
            else:
                indices = torch.argmax(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()

            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
            # print(Q.sum(0)[:10])
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                logger.warning("Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        # indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)
        x_q_codebook = x_q

        # Diversity
        # diversity_loss = self.diversity_loss_main_entry(x, x_q, indices, label)
        # wandb.log({'diversity_loss': diversity_loss})

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())

        loss = codebook_loss + self.mu * commitment_loss  # + self.beta * diversity_loss

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices, x_q_codebook


class MLPLayers(nn.Module):
    def __init__(self, layers, dropout=0.0, activation="relu", bn=False):
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation)
            if activation_func is not None and idx != (len(self.layers) - 2):
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


def _apply_mean_rows(mean_function, x: torch.Tensor) -> torch.Tensor:
    """Apply a k-means ``mean_function`` to every row of a ``(K, D)`` tensor.

    The supported mean functions (``MEAN_FUNCTIONS["mean"]`` and
    ``MEAN_FUNCTIONS["normalized_mean"]``) operate on a single ``(N, D)`` slice
    via ``torch.mean(..., dim=0)``; applied row-wise to precomputed cluster means
    they reduce to the identity and to row L2 normalization respectively. Any
    other callable is applied to each row independently (slow fallback).
    """
    if mean_function is MEAN_FUNCTIONS["normalized_mean"]:
        return F.normalize(x, dim=1)
    if mean_function is MEAN_FUNCTIONS["mean"]:
        return x
    return torch.stack([mean_function(row.unsqueeze(0)) for row in x])


def _centroid_change(distance_function, new_centroids, old_centroids, active) -> torch.Tensor:
    """Vectorized per-cluster centroid movement, mirroring the old per-cluster ``distance_function`` loop.

    For the two supported distance functions (``torch.cdist`` and ``cosine_distance``)
    the value matches the original 1x1 ``distance_function`` call exactly; any other
    function falls back to the L2 norm of the delta.
    """
    if not active.any():
        return torch.tensor(0.0, device=new_centroids.device)
    if distance_function is cosine_distance:
        change = 1.0 - F.cosine_similarity(new_centroids, old_centroids, dim=1)
    else:
        change = torch.linalg.vector_norm(new_centroids - old_centroids, dim=1)
    return change[active].max()


class TorchKMeans:
    class KMeansDefaults:
        MAX_ITERATIONS = 300
        TOLERANCE = 1e-4

    class KMeansInitalizationMode:
        KMEANS_PP = "kmeans++"
        RANDOM = "random"

    def __init__(
        self,
        n_clusters: int,
        distance_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        mean_function: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device | str,
        kmeanspp_exact: bool = False,
    ):
        self.num_clusters = n_clusters
        self.distance_function = distance_function
        self.mean_function = mean_function
        self.centroids = None
        self.device = device
        self.kmeanspp_exact = kmeanspp_exact

    def fit(
        self,
        data: torch.Tensor,
        max_iterations: int = KMeansDefaults.MAX_ITERATIONS,
        tolerance: float = KMeansDefaults.TOLERANCE,
        batch_size: int | None = None,
        init_mode: Literal["kmeans++", "random"] | None = "kmeans++",
    ):
        """
        Fit the k-means algorithm to data.

        :param data: Data to fit.
        :param max_iterations: Maximum number of iterations.
        :param tolerance: Tolerance for convergence.
        :param batch_size: Optional batch size.
        :param init_mode: Centroids initialization mode.
        """
        if batch_size is None:
            data = data.to(self.device)

        self._initialize(data, init_mode=init_mode, batch_size=batch_size)

        dim = data.shape[-1]
        # Fit loop
        for iteration in tqdm(
            range(max_iterations), desc="TorchKMeans fit", total=max_iterations, leave=False, position=1
        ):
            if batch_size is None:
                assignments = self._assign(data)
                sums = torch.zeros(self.num_clusters, dim, dtype=data.dtype, device=self.device)
                sums.scatter_add_(0, assignments[:, None].expand(-1, dim), data)
                counts = torch.bincount(assignments, minlength=self.num_clusters).to(self.device)
            else:
                sums = torch.zeros(self.num_clusters, dim, dtype=data.dtype, device=self.device)
                counts = torch.zeros(self.num_clusters, dtype=torch.long, device=self.device)
                n_batches = (data.shape[0] + batch_size - 1) // batch_size
                for j in range(n_batches):
                    start_idx, end_idx = j * batch_size, min((j + 1) * batch_size, data.shape[0])
                    batch = data[start_idx:end_idx].to(self.device)
                    batch_assign = self._assign(batch)
                    sums.scatter_add_(0, batch_assign[:, None].expand(-1, dim), batch)
                    counts += torch.bincount(batch_assign, minlength=self.num_clusters).to(self.device)

            new_centroids, active = self._update_centroids(sums, counts)
            max_change = _centroid_change(self.distance_function, new_centroids, self.centroids, active)
            self.centroids = new_centroids

            # Check for convergence
            if max_change < tolerance:
                break

    def fit_mini_batch(
        self,
        data: torch.Tensor,
        num_epochs: int,
        batch_size: int,
        init_mode: Literal["kmeans++", "random"] | None = "kmeans++",
        max_steps: int | None = None,
        init_data: torch.Tensor | None = None,
    ):
        """Fits the k-means algorithm using the mini-batch k-means algorithm (10.1145/1772690.1772862)."""
        # Shuffle data before initialization
        data = data[torch.randperm(data.shape[0], device=data.device)]

        self._initialize(data if init_data is None else init_data, init_mode=init_mode, batch_size=batch_size)

        assignment_counts = torch.zeros(self.num_clusters, device=self.device)
        dim = data.shape[-1]

        steps = 0
        for epoch in tqdm(
            range(num_epochs), desc="TorchKMeans mini-batch fit", total=num_epochs, leave=False, position=1
        ):
            # Shuffle data before epoch
            data = data[torch.randperm(data.shape[0], device=data.device)]
            for batch_i in range(0, data.shape[0], batch_size):
                batch = data[batch_i : batch_i + batch_size]
                batch = batch.to(self.device)

                assignments = self._assign(batch)
                counts_batch = torch.bincount(assignments, minlength=self.num_clusters).to(self.device)
                assignment_counts += counts_batch

                sums = torch.zeros(self.num_clusters, dim, dtype=batch.dtype, device=self.device)
                sums.scatter_add_(0, assignments[:, None].expand(-1, dim), batch)
                active = counts_batch > 0
                batch_means = _apply_mean_rows(self.mean_function, sums / counts_batch.clamp_min(1).unsqueeze(1))
                learning_rate = 1.0 / assignment_counts.clamp_min(1)
                new_centroids = (1.0 - learning_rate)[:, None] * self.centroids + learning_rate[:, None] * batch_means
                # Apply mean function one more time to renormalize in case of normalized function
                new_centroids = _apply_mean_rows(self.mean_function, new_centroids)
                self.centroids = torch.where(active[:, None], new_centroids, self.centroids)

                # Break out of function if max steps has been reached
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    return

    def predict(self, data: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
        """
        Predict labels for data.
        Will fail if k-means has not been initialized.

        :param data: Data to predict labels for.
        :param batch_size: Optional batch size.
        :return: Indexes for closest centroids as predicted labels.
        """
        if self.centroids is None:
            raise ValueError("Model not fitted yet.")

        if batch_size is None:
            return self._assign(data.to(self.device))
        return torch.cat(
            [self._assign(data[j : j + batch_size].to(self.device)) for j in range(0, data.shape[0], batch_size)]
        )

    def fit_predict(
        self,
        data: torch.Tensor,
        max_iterations: int = KMeansDefaults.MAX_ITERATIONS,
        tolerance: float = KMeansDefaults.TOLERANCE,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Fits k-means algorithm to data and predicts labels for data."""
        self.fit(data, max_iterations, tolerance, batch_size)
        return self.predict(data, batch_size)

    def to(self, device: torch.device | str):
        self.device = device
        if self.centroids is not None:
            self.centroids = self.centroids.to(device)

    def _initialize(self, init_data, init_mode="kmeans++", batch_size=None):
        match init_mode:
            case "random":
                self._initialize_random(init_data)
            case "kmeans++":
                self._initialize_kmeanspp(init_data, batch_size)
            case _:
                raise ValueError(f"Unknown k-means init_mode: {init_mode}")

    def _assign(self, data: torch.Tensor) -> torch.Tensor:
        """Nearest-centroid assignment for one device-resident tensor."""
        distances = self.distance_function(data, self.centroids)
        return torch.argmin(distances, dim=1)

    def _update_centroids(self, sums: torch.Tensor, counts: torch.Tensor):
        """Vectorized centroid update; empty clusters keep their previous centroid."""
        active = counts > 0
        denom = counts.clamp_min(1).to(sums.dtype).unsqueeze(1)
        means = _apply_mean_rows(self.mean_function, sums / denom)
        new_centroids = torch.where(active[:, None], means, self.centroids)
        return new_centroids, active

    def _initialize_kmeanspp(self, data: torch.Tensor, batch_size: int | None = None):
        """Initialize the k-means algorithm based on the method from k-means++.

        Uses the fast O(N*K*D) incremental seeding by default (Arthur & Vassilvitskii
        2007, SODA; cf. scikit-learn's ``kmeans_plusplus``), which maintains the
        running distance to the nearest chosen center with ``torch.minimum`` instead
        of recomputing the full ``N x i`` matrix. It is mathematically equivalent to
        the full-matrix form and matches it bit-for-bit in the realistic ``K < N``
        case, but floating-point rounding differs when ``K > N``; the exact
        full-matrix form is used then (or when ``kmeanspp_exact`` is set) to preserve
        the original RNG-draw behavior.
        """
        n, d = data.shape
        self.centroids = torch.zeros(self.num_clusters, d, device=self.device)

        if batch_size is None:
            data = data.to(self.device)

        # Assign first centroid randomly
        first_index = torch.randint(0, n, (1,), device=data.device)
        self.centroids[0] = data[first_index].to(self.device)

        if self.kmeanspp_exact or self.num_clusters > n:
            self._kmeanspp_full_matrix(data, batch_size)
        else:
            self._kmeanspp_incremental(data, batch_size)

    def _kmeanspp_full_matrix(self, data: torch.Tensor, batch_size: int | None):
        """k-means++ seeding recomputing the full ``N x i`` distance matrix per step.

        Mirrors the original implementation exactly (same ``distance_function`` calls
        and RNG draw order), so results are bit-identical to the pre-vectorization
        code. Used only when ``kmeanspp_exact`` is set or ``K > N``.
        """
        # Assign other centroids iteratively by probability based on distance
        for i in range(1, self.num_clusters):
            if batch_size is None:
                distances = self.distance_function(data, self.centroids[:i])
                min_distances = distances.min(dim=1)[0]
            else:
                batch_distances = [
                    self.distance_function(data[j : j + batch_size].to(self.device), self.centroids[:i])
                    for j in range(0, data.shape[0], batch_size)
                ]
                min_distances = torch.cat([bd.min(dim=1)[0] for bd in batch_distances])
            self._sample_kmeanspp_centroid(min_distances, data, i)

    def _kmeanspp_incremental(self, data: torch.Tensor, batch_size: int | None):
        """Fast O(N*K*D) k-means++ seeding (Arthur & Vassilvitskii 2007; cf. scikit-learn ``kmeans_plusplus``).

        Each new center is sampled with probability proportional to its squared
        distance to the nearest already-chosen center; that running minimum is
        maintained incrementally with ``torch.minimum`` instead of recomputing the
        full ``N x i`` distance matrix.
        """
        min_dist = None
        for i in range(1, self.num_clusters):
            if batch_size is None:
                d_new = self.distance_function(data, self.centroids[i - 1 : i])[:, 0]
            else:
                parts = [
                    self.distance_function(data[j : j + batch_size].to(self.device), self.centroids[i - 1 : i])[:, 0]
                    for j in range(0, data.shape[0], batch_size)
                ]
                d_new = torch.cat(parts)
            min_dist = d_new if min_dist is None else torch.minimum(min_dist, d_new)
            self._sample_kmeanspp_centroid(min_dist, data, i)

    def _sample_kmeanspp_centroid(self, min_distances: torch.Tensor, data: torch.Tensor, i: int):
        """Sample one new centroid with probability proportional to ``min_distances**2``."""
        # Ensure distances are never 0 even with multiple identical vectors in the data
        min_distances.clamp_(min=1e-6)
        probabilities = min_distances**2
        probabilities = probabilities / probabilities.sum()
        index = torch.multinomial(probabilities, 1)
        self.centroids[i] = data[index.to(data.device)].to(self.device)

    def _initialize_random(self, data: torch.Tensor):
        """Initialize the k-means centroid by picking random vectors from data."""
        n, _ = data.shape

        indices = torch.randperm(n)[: self.num_clusters]
        self.centroids = data[indices.to(data.device)].to(self.device)


class FaissKMeans(TorchKMeans):
    """faiss-cpu backed k-means exposing the same API as :class:`TorchKMeans`.

    The nearest-centroid assignment (the ``O(N*K*D)`` part of every iteration) is
    delegated to a faiss ``IndexFlatL2`` / ``IndexFlatIP`` running entirely on
    CPU, which is far faster than torch's ``cdist``/matmul kernels on CPU and NPU.
    Centroids are kept as torch tensors so the surrounding ``RKMeans`` /
    ``HierarchicalRKMeans`` code (residual subtraction, ``state_dict``, ``load``)
    works unchanged. The loop structure (``max_iterations``/``tolerance`` early
    stop, explicit ``init_mode``) is inherited from :class:`TorchKMeans`.
    """

    def __init__(
        self,
        n_clusters: int,
        distance_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        mean_function: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device | str,
        kmeanspp_exact: bool = False,
    ):
        if distance_function is not torch.cdist and distance_function is not cosine_distance:
            raise ValueError("FaissKMeans supports only euclidean (torch.cdist) and cosine (cosine_distance)")
        super().__init__(n_clusters, distance_function, mean_function, torch.device("cpu"), kmeanspp_exact)
        self._spherical = distance_function is cosine_distance

    def to(self, device: torch.device | str):
        # faiss always runs on CPU; ignore the requested device (kept for the API).
        self.device = torch.device("cpu")
        if self.centroids is not None:
            self.centroids = self.centroids.to("cpu")

    def _faiss_centroids(self):
        faiss = _get_faiss()
        c = np.asarray(self.centroids.detach().cpu().numpy(), dtype="float32")
        if self._spherical:
            c = np.array(c, copy=True)  # normalize_L2 mutates in place
            faiss.normalize_L2(c)
        return c

    def _faiss_query(self, data: torch.Tensor) -> np.ndarray:
        """Zero-copy float32 view of ``data`` for faiss search (copied only when normalized)."""
        faiss = _get_faiss()
        arr = np.asarray(data.detach().cpu().numpy(), dtype="float32")
        if self._spherical:
            arr = np.array(arr, copy=True)  # normalize_L2 mutates in place
            faiss.normalize_L2(arr)
        return arr

    def _assign(self, data: torch.Tensor) -> torch.Tensor:
        faiss = _get_faiss()
        x = self._faiss_query(data)
        c = self._faiss_centroids()
        index = faiss.IndexFlatIP(c.shape[1]) if self._spherical else faiss.IndexFlatL2(c.shape[1])
        index.add(c)
        _, labels = index.search(x, 1)
        return torch.from_numpy(labels[:, 0]).to(data.device)
