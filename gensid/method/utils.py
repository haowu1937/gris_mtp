import logging

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from tqdm import tqdm

from gensid.utils import relabel_compact

logger = logging.getLogger(__name__)


def dense_dmon_local_loss(
    adj: torch.Tensor,
    logits: torch.Tensor,
    degree: torch.Tensor,
    m2: float,
    balance_weight: float = 1.0,
    entropy_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DMoN modularity loss on one local (dense) adjacency slice.

    Maximizes the modularity of the soft cluster assignment ``S`` and adds two
    regularizers: a balance term that pushes each cluster toward ``1/K`` of the
    nodes, and an entropy term that sharpens the assignments.

    Args:
        adj (torch.Tensor): ``[N, N]`` dense adjacency of the local subset.
        logits (torch.Tensor): ``[N, K]`` raw assignment logits.
        degree (torch.Tensor): ``[N]`` degree of every node in ``adj``.
        m2 (float): Twice the number of edges (``degree.sum()``).
        balance_weight (float, optional): Weight of the balance regularizer.
            Defaults to 1.0.
        entropy_weight (float, optional): Weight of the entropy regularizer.
            Defaults to 0.01.

    Returns:
        (torch.Tensor, dict[str, torch.Tensor]): the scalar loss and a dict of
        detached metrics (``modularity``, ``balance``, ``entropy``).
    """
    S = F.softmax(logits, dim=-1)  # [N, K]
    # Exact trace(S.T @ A @ S), without constructing the K x K matrix.
    AS = adj @ S
    edge_term = (S * AS).sum(dtype=torch.float32)

    # Exact trace(S.T @ (dd.T / m2) @ S).
    cluster_degree = S.T @ degree
    null_term = cluster_degree.square().sum(dtype=torch.float32) / m2

    modularity = (edge_term - null_term) / m2

    # Simple collapse/balance regularizer. Keeps clusters from all collapsing.
    p = S.mean(dim=0)
    k = S.size(1)
    balance = ((p - 1.0 / k) ** 2).sum()

    # Positive entropy term encourages sharper assignments when minimized.
    entropy = -(S * torch.log(S.clamp_min(1e-12))).sum(dim=1).mean()

    loss = -modularity + balance_weight * balance + entropy_weight * entropy
    metrics = {
        "modularity": modularity.detach(),
        "balance": balance.detach(),
        "entropy": entropy.detach(),
    }
    return loss, metrics


def spectral_cluster_subset(A: sp.csr_matrix, subset: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Spectral cluster one subset of the interaction graph into ``k`` parts.

    Builds the symmetric normalized Laplacian of the induced subgraph, embeds the
    nodes on the smallest non-trivial eigenvectors and runs k-means on that
    embedding. Falls back to a deterministic modular split when the subgraph has
    no internal structure, and to the raw adjacency when eigendecomposition fails.

    Args:
        A (sp.csr_matrix): Full ``[N, N]`` (symmetrized) adjacency matrix.
        subset (np.ndarray): Indices of the nodes to cluster.
        k (int): Number of target clusters.
        seed (int): Random seed for the k-means step.

    Returns:
        np.ndarray: ``[len(subset)]`` int64 cluster labels in ``[0, k)``.
    """
    n = len(subset)
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    if k <= 1:
        return np.zeros((n,), dtype=np.int64)
    if n <= k:
        return np.arange(n, dtype=np.int64)

    A_sub = A[subset][:, subset].astype(np.float64).tocsr()
    if A_sub.nnz == 0:
        # No internal structure: deterministic-ish split by semantic order inside subset.
        return np.arange(n, dtype=np.int64) % k

    deg = np.asarray(A_sub.sum(axis=1)).reshape(-1)
    deg = np.maximum(deg, 1e-12)
    inv_sqrt = 1.0 / np.sqrt(deg)
    L = sp.eye(n, format="csr") - sp.diags(inv_sqrt) @ A_sub @ sp.diags(inv_sqrt)

    eig_k = min(k, n - 1)
    try:
        evals, evecs = eigsh(L, k=eig_k, which="SM")
        order = np.argsort(evals)
        Z = evecs[:, order]
        if Z.shape[1] > 1:
            Z = Z[:, 1:]
        if Z.shape[1] == 0:
            Z = evecs[:, order]
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] eigsh failed for subset size={n}, k={k}: {e}. Falling back to adjacency rows.")
        Z = A_sub.toarray()

    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Z)
    return labels.astype(np.int64)


class RecursiveClusteringMixin:
    """Stateless driver of top-down (recursive) graph clustering.

    Subclasses implement :meth:`forward`, which clusters a single parent subset
    into ``k`` child clusters and returns the per-node child labels. This mixin
    runs that local clustering level by level, coarsest first, and returns one
    label vector per hierarchy level.

    The mixin holds no state: every call to :meth:`recursive_clustering` runs the
    full loop and returns a fresh hierarchy. Results are cached by the concrete
    SID method (e.g. as ``_indices``).
    """

    def recursive_clustering(
        self,
        emb_data: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.Tensor,
        recursive_levels: int = 100,
        recursive_factor: int = 6,
    ) -> list[np.ndarray]:
        """Run the recursive clustering loop.

        Args:
            emb_data (torch.Tensor): ``[num_nodes, dim]`` node features; used only
                as a hint by the local clusterers (e.g. RecDMoN's neural net).
            edge_index (torch.LongTensor): ``[2, num_edges]`` graph adjacency.
            edge_weight (torch.Tensor or None): ``[num_edges]`` edge weights, or
                None for an unweighted graph.
            recursive_levels (int, optional): Number of hierarchy levels to build.
                Defaults to 100.
            recursive_factor (int, optional): Maximum number of child clusters per
                parent subset. Defaults to 6.

        Returns:
            list[np.ndarray]: one ``[num_nodes]`` int64 label vector per level,
            coarse-to-fine (``-1`` marks nodes not assigned at that level).
        """
        from gensid.data.utils import edge_index_to_sparse_adj

        num_items = emb_data.size(0)
        A = edge_index_to_sparse_adj(edge_index, edge_weight, num_items)

        parent_labels = np.zeros(num_items, dtype=np.int64)
        hierarchy: list[np.ndarray] = []

        for level in tqdm(range(1, recursive_levels + 1), desc="Hierarchical mincut levels", position=0):
            new_labels = np.full(num_items, -1, dtype=np.int64)
            next_cluster_id = 0
            parents = [p for p in np.unique(parent_labels).tolist() if p >= 0]

            for parent in tqdm(parents, desc="Hierarchical mincut parents", position=1, leave=False):
                subset = np.where(parent_labels == parent)[0].astype(np.int64)
                k = min(recursive_factor, len(subset))

                local_labels = self.forward(emb_data, A, subset, k, level, parent)

                unique_indices, inverse_indices = np.unique(local_labels, return_inverse=True)
                new_labels[subset] = next_cluster_id + inverse_indices
                next_cluster_id += len(unique_indices)

            new_labels = relabel_compact(new_labels)
            hierarchy.append(new_labels)
            parent_labels = new_labels
            logger.info(
                "recursive level %d: clusters=%d",
                level,
                len(np.unique(new_labels[new_labels >= 0])),
            )

        return hierarchy
