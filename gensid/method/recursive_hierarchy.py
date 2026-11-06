import logging
from time import time
from typing import Callable

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from tqdm import tqdm

from gensid.config.model_config import RecDMoNModelConfig, SpectralModelConfig
from gensid.method.base import SIDMethod
from gensid.method.utils import (
    RecursiveClusteringMixin,
    dense_dmon_local_loss,
    spectral_cluster_subset,
)
from gensid.nn.layers import LocalAssignNet

logger = logging.getLogger(__name__)


class RecDMoN(SIDMethod, RecursiveClusteringMixin):
    """Recursive DMoN hierarchical clustering SID method.

    Builds a coarse-to-fine hierarchy of SIDs by recursively applying the DMoN
    neural modularity objective: each parent cluster is split into at most
    ``recursive_factor`` child clusters by a small local assignment network
    trained on the induced adjacency slice.

    The recursion is driven by the stateless :class:`RecursiveClusteringMixin`;
    :meth:`forward` implements the per-subset neural clustering step.

    Args:
        config (RecDMoNModelConfig): Hyperparameters of the method. See
            ``gensid.config.model_config.RecDMoNModelConfig``.
    """

    name = "recdmon"
    trainer = "recursive_clustering"

    def __init__(self, config: RecDMoNModelConfig):
        """Validate and store the recursive DMoN hyperparameters."""
        super().__init__()

        if config.recursive_factor < 2:
            raise ValueError("--recursive_factor must be >= 2")
        if config.recursive_levels < 1:
            raise ValueError("--recursive_levels must be >= 1")

        self.recursive_factor = config.recursive_factor
        self.recursive_levels = config.recursive_levels
        self.recursive_epochs = config.recursive_epochs
        self.assign_hidden_dim = config.assign_hidden_dim
        self.dropout = config.dropout
        self.max_dense_nodes = config.max_dense_nodes
        self.dmon_balance_weight = config.dmon_balance_weight
        self.dmon_entropy_weight = config.dmon_entropy_weight

        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.grad_clip = config.grad_clip
        self.device = torch.device("cpu")

        self._indices = None

    def to(self, device: torch.device | str):
        """Move the method to ``device``."""
        self.device = torch.device(device)
        return super().to(device)

    @property
    def indices(self):
        """The cached ``[num_nodes, recursive_levels]`` SID matrix.

        Raises:
            AttributeError: If :meth:`fit` has not run yet.
        """
        if self._indices is None:
            raise AttributeError("Indices cannot be generated if clustering is not run by fit first!")

        return self._indices

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int | None = None,
        callback: Callable | None = None,
    ):
        """Run the recursive DMoN clustering and cache the coarse-to-fine SIDs.

        Args:
            dataset (torch.utils.data.Dataset): Dataset exposing ``x_graph``,
                ``edge_index`` and ``edge_weight`` (e.g. ``GraphClusteringEmbDataset``).
            batch_size (int or None, optional): Unused by this method; kept for
                interface compatibility. Defaults to None.
            callback (Callable or None, optional): Called per local-optimization
                epoch with ``(epoch, start_time, end_time, loss, **info)``.
                Defaults to None.
        """
        self._fit_callback = callback
        cluster_hierarchy = self.recursive_clustering(
            dataset.x_graph,
            dataset.edge_index,
            dataset.edge_weight,
            recursive_levels=self.recursive_levels,
            recursive_factor=self.recursive_factor,
        )
        delattr(self, "_fit_callback")
        self._codebook_sizes = [len(np.unique(h[h >= 0])) for h in cluster_hierarchy]
        self._indices = np.stack(cluster_hierarchy, axis=1)

    def forward(
        self,
        x_all: torch.Tensor,
        A: sp.csr_matrix,
        subset: np.ndarray,
        k: int,
        level: int,
        parent: int,
    ) -> np.ndarray:
        """Cluster one parent subset with a locally-trained DMoN assignment net.

        Trains a small MLP on the induced adjacency slice to maximize modularity,
        then assigns each node to its most confident cluster.

        Args:
            x_all (torch.Tensor): ``[num_nodes, dim]`` node features.
            A (sp.csr_matrix): Full symmetrized ``[num_nodes, num_nodes]`` adjacency.
            subset (np.ndarray): Node indices of the parent cluster.
            k (int): Number of child clusters to produce.
            level (int): Current hierarchy level (1-based), used for reporting.
            parent (int): Parent cluster id, used for reporting.

        Returns:
            np.ndarray: ``[len(subset)]`` int64 child labels in ``[0, k)``.
        """
        n = len(subset)
        if n == 0:
            return np.empty((0,), dtype=np.int64)
        if k <= 1:
            return np.zeros((n,), dtype=np.int64)
        if n <= k:
            return np.arange(n, dtype=np.int64)
        if n > self.max_dense_nodes:
            raise ValueError(f"RecDMoN local subset has {n} nodes, above --max_dense_nodes={self.max_dense_nodes}.")

        A_sub_np = A[subset][:, subset].astype(np.float32).toarray()
        x_sub = x_all[torch.as_tensor(subset, dtype=torch.long)].to(self.device)
        adj = torch.as_tensor(A_sub_np, dtype=torch.float32, device=self.device)

        model = LocalAssignNet(x_sub.size(1), self.assign_hidden_dim, k, self.dropout).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Cached since they are fixed
        degree = adj.sum(dim=1)
        m2 = degree.sum().clamp_min(1e-12)
        model.train()
        iter_epochs = tqdm(
            range(self.recursive_epochs),
            position=2,
            leave=False,
            desc="Neural cluster training",
        )
        for epoch in iter_epochs:
            opt.zero_grad()
            training_start_time = time()
            logits = model(x_sub)
            training_end_time = time()
            loss, metrics = dense_dmon_local_loss(
                adj=adj,
                logits=logits,
                degree=degree,
                m2=m2,
                balance_weight=self.dmon_balance_weight,
                entropy_weight=self.dmon_entropy_weight,
            )

            if torch.isnan(loss):
                raise ValueError("NaN loss in RecDMon local split")
            loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            opt.step()

            train_info = {
                "subset_size": n,
                "k": k,
                "level": level,
                "parent": parent,
                **{name: float(value.item()) for name, value in metrics.items()},
            }
            if getattr(self, "_fit_callback", None) is not None:
                self._fit_callback(epoch + 1, training_start_time, training_end_time, loss.item(), **train_info)

        model.eval()
        with torch.no_grad():
            labels = model(x_sub).argmax(dim=-1).cpu().numpy().astype(np.int64)

        return labels

    def generate(self, dataset: torch.utils.data.Dataset, batch_size: int | None = None) -> torch.Tensor:
        """Return the cached ``[num_nodes, recursive_levels]`` SID matrix.

        Args:
            dataset (torch.utils.data.Dataset): Ignored; indices were produced by
                :meth:`fit`. Kept for interface compatibility.
            batch_size (int or None, optional): Ignored. Defaults to None.
        """
        return torch.from_numpy(self._indices)

    def state_dict(self):
        """Return the cached SID matrix (no learned weights are persisted)."""
        return {"indices": self._indices}

    def save(self, path: str):
        """Persist the cached SID matrix with ``torch.save``."""
        torch.save(self.state_dict(), path)

    def load_state_dict(self, state_dict: dict, **kwargs):
        """Restore the cached SID matrix from a ``state_dict``."""
        self._indices = state_dict["indices"]
        self._codebook_sizes = [len(np.unique(self._indices[:, i])) for i in range(self._indices.shape[1])]


class Spectral(SIDMethod, RecursiveClusteringMixin):
    """Recursive spectral-clustering SID method.

    Same coarse-to-fine recursion as :class:`RecDMoN`, but each local split uses
    spectral clustering (:func:`spectral_cluster_subset`) instead of a learned
    assignment network, so it needs no training loop.

    Args:
        config (SpectralModelConfig): Hyperparameters of the method. See
            ``gensid.config.model_config.SpectralModelConfig``.
    """

    name = "spectral"
    trainer = "recursive_clustering"

    def __init__(self, config: SpectralModelConfig):
        """Store the recursive spectral clustering hyperparameters."""
        super().__init__()

        self.seed = config.seed
        self.recursive_factor = config.recursive_factor
        self.recursive_levels = config.recursive_levels
        self.device = torch.device("cpu")

        self._indices = None

    def to(self, device: torch.device | str):
        """Move the method to ``device``."""
        self.device = torch.device(device)
        return super().to(device)

    @property
    def indices(self):
        """The cached ``[num_nodes, recursive_levels]`` SID matrix.

        Raises:
            AttributeError: If :meth:`fit` has not run yet.
        """
        if self._indices is None:
            raise AttributeError("Indices cannot be generated if clustering is not run by fit first!")

        return self._indices

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int | None = None,
        callback: Callable | None = None,
    ):
        """Run the recursive spectral clustering and cache the coarse-to-fine SIDs.

        Args:
            dataset (torch.utils.data.Dataset): Dataset exposing ``x_graph``,
                ``edge_index`` and ``edge_weight`` (e.g. ``GraphClusteringEmbDataset``).
            batch_size (int or None, optional): Unused; kept for interface
                compatibility. Defaults to None.
            callback (Callable or None, optional): Unused; spectral clustering has
                no training loop. Defaults to None.
        """
        cluster_hierarchy = self.recursive_clustering(
            dataset.x_graph,
            dataset.edge_index,
            dataset.edge_weight,
            recursive_levels=self.recursive_levels,
            recursive_factor=self.recursive_factor,
        )
        self._codebook_sizes = [len(np.unique(h[h >= 0])) for h in cluster_hierarchy]
        self._indices = np.stack(cluster_hierarchy, axis=1)

    def forward(
        self,
        x_all: torch.Tensor,
        A: sp.csr_matrix,
        subset: np.ndarray,
        k: int,
        level: int,
        parent: int,
    ) -> np.ndarray:
        """Cluster one parent subset with spectral clustering.

        Args:
            x_all (torch.Tensor): ``[num_nodes, dim]`` node features. Unused by
                spectral clustering; kept for interface consistency.
            A (sp.csr_matrix): Full symmetrized ``[num_nodes, num_nodes]`` adjacency.
            subset (np.ndarray): Node indices of the parent cluster.
            k (int): Number of child clusters to produce.
            level (int): Current hierarchy level (1-based), mixed into the seed.
            parent (int): Parent cluster id, mixed into the seed.

        Returns:
            np.ndarray: ``[len(subset)]`` int64 child labels in ``[0, k)``.
        """
        seed = (self.seed or 0) + level * 1009 + int(parent)
        return spectral_cluster_subset(A, subset, k=k, seed=seed)

    def generate(self, dataset: torch.utils.data.Dataset, batch_size: int | None = None) -> torch.Tensor:
        """Return the cached ``[num_nodes, recursive_levels]`` SID matrix.

        Args:
            dataset (torch.utils.data.Dataset): Ignored; indices were produced by
                :meth:`fit`. Kept for interface compatibility.
            batch_size (int or None, optional): Ignored. Defaults to None.
        """
        return torch.from_numpy(self._indices)

    def state_dict(self):
        """Return the cached SID matrix (no learned weights are persisted)."""
        return {"indices": self._indices}

    def save(self, path: str):
        """Persist the cached SID matrix with ``torch.save``."""
        torch.save(self.state_dict(), path)

    def load_state_dict(self, state_dict: dict, **kwargs):
        """Restore the cached SID matrix from a ``state_dict``."""
        self._indices = state_dict["indices"]
        self._codebook_sizes = [len(np.unique(self._indices[:, i])) for i in range(self._indices.shape[1])]
