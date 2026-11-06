import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


def _get_faiss():
    """Return the faiss module, raising a clear error when it is not available.

    A failure at ``import faiss`` is not always "faiss not installed": a
    binary/ABI mismatch, a missing shared library or a dependency conflict
    (e.g. a ``faiss-cpu`` wheel built against a different numpy version) also
    raises at import time. Distinguish the two cases so the real cause is
    surfaced instead of always blaming a missing package.
    """
    try:
        import faiss
    except Exception as exc:
        if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", None) == "faiss":
            raise ImportError(
                "faiss is required for the 'faiss' kmeans backend; install it with "
                "`pip install faiss-cpu` (or `uv sync --extra faiss`)"
            ) from exc
        raise ImportError(
            "faiss appears to be installed but failed to load (binary/ABI mismatch, missing "
            f"shared library, or dependency conflict). Underlying error: {type(exc).__name__}: {exc}"
        ) from exc
    return faiss


def cosine_distance(a, b) -> torch.Tensor:
    if len(b.shape) == 1:
        b = torch.unsqueeze(b, -1)

    normalized_a = torch.nn.functional.normalize(a)
    normalized_b = torch.nn.functional.normalize(b)
    cosine_similarity = torch.mm(normalized_a, normalized_b.T)

    return 1 - cosine_similarity


def normalized_mean(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.mean(x, dim=0), dim=0)


DISTANCE_FUNCTIONS = {
    "cosine": cosine_distance,
    "euclidean": torch.cdist,
}

MEAN_FUNCTIONS = {
    "normalized_mean": normalized_mean,
    "mean": lambda d: torch.mean(d, dim=0),
}


def custom_constrained_km(
    data, n_clusters=10, max_size_multiplier: float = 4.0, max_iter: int = 10, n_init: int = 10, seed: int = 42
):
    # Convert torch tensor to numpy
    x = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)

    # 1. Standard KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init, random_state=seed)
    labels = kmeans.fit_predict(x)
    centers = kmeans.cluster_centers_

    # 2. Greedy constraint enforcement (Caps max size to protect limits)
    # size_max must be at least ceil(len(data) / n_clusters), otherwise total capacity
    # (n_clusters * size_max) is smaller than the number of points and the greedy loop
    # below runs out of non-full clusters (argmin over an empty sequence).
    size_max = max(n_clusters * max_size_multiplier, math.ceil(len(x) / n_clusters))
    counts = np.bincount(labels, minlength=n_clusters)

    if np.any(counts > size_max):
        distances = np.linalg.norm(x[:, np.newaxis] - centers, axis=2)
        sorted_points = np.argsort(np.min(distances, axis=1))  # Process closest points first

        counts = np.zeros(n_clusters, dtype=int)
        for idx in sorted_points:
            # Assign to closest cluster that isn't full
            allowed_clusters = np.where(counts < size_max)[0]
            best_cluster = allowed_clusters[np.argmin(distances[idx, allowed_clusters])]
            labels[idx] = best_cluster
            counts[best_cluster] += 1

    return torch.from_numpy(centers), labels.tolist()


def activation_layer(activation_name="relu"):
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        activation_name = activation_name.lower()
        match activation_name:
            case "sigmoid":
                activation = nn.Sigmoid()
            case "tanh":
                activation = nn.Tanh()
            case "relu":
                activation = nn.ReLU()
            case "leakyrelu":
                activation = nn.LeakyReLU()
            case "gelu":
                activation = nn.GELU()
            case "elu":
                activation = nn.ELU()
            case _:
                activation = None
    elif isinstance(activation_name, type) and issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(f"activation function {activation_name} is not implemented")

    return activation


# def kmeans(
#     samples,
#     num_clusters,
#     num_iters=10,
# ):
#     B, dim, dtype, device = (
#         samples.shape[0],
#         samples.shape[-1],
#         samples.dtype,
#         samples.device,
#     )
#     x = samples.cpu().detach().numpy()

#     cluster = KMeans(n_clusters=num_clusters, max_iter=num_iters).fit(x)

#     centers = cluster.cluster_centers_
#     tensor_centers = torch.from_numpy(centers).to(device)

#     return tensor_centers


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    Q = torch.exp(-distances / epsilon)

    B = Q.shape[0]  # number of samples to assign
    K = Q.shape[1]  # how many centroids per block (usually set to 256)

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    # print(Q.sum())
    for it in range(sinkhorn_iterations):
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q
