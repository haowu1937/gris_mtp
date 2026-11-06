import dataclasses
import logging
import os
import random

import numpy as np
import torch
from torch.utils import data

from gensid.config.dataset_config import (
    GRQVAEFusionDatasetConfig,
    GraphClusteringDatasetConfig,
    GraphDatasetConfig,
    GraphFusionDatasetConfig,
    MixedDepthDatasetConfig,
    PlainDatasetConfig,
)
from gensid.config.encoder_config import EncoderConfig
from gensid.nn import get_encoder
from gensid.data.graph import InteractionGraphBuilder, compute_graph_spectral_vectors
from gensid.data.utils import load_embeddings, load_graph_cluster_labels, load_train_item_mask

logger = logging.getLogger(__name__)


def _override_propagations(encoder_config: EncoderConfig, n_propagations: int) -> EncoderConfig:
    """Return a copy of ``encoder_config`` with ``n_propagations`` set (when supported)."""
    if "n_propagations" in {f.name for f in dataclasses.fields(encoder_config)}:
        return dataclasses.replace(encoder_config, n_propagations=n_propagations)
    return encoder_config


class EmbDataset(data.Dataset):
    """Raw semantic embeddings (no graph).

    Args:
        config (PlainDatasetConfig): Dataset configuration.
        embeddings_path (str): Path to the ``.npy`` embedding matrix.
        interactions_path (str or None, optional): Path to the interaction JSON.
        eval_mode (bool, optional): If True (or ``interactions_path`` is None), all
            items are used. Defaults to True.
    """

    name = "plain"

    def __init__(
        self,
        config: PlainDatasetConfig,
        embeddings_path: str,
        interactions_path: str = None,
        eval_mode: bool = True,
    ):
        self.config = config
        self.data_path = embeddings_path
        self.eval_mode = eval_mode

        full_embeddings = load_embeddings(embeddings_path)
        self.dim = full_embeddings.shape[-1]
        num_items = full_embeddings.shape[0]

        if self.eval_mode or interactions_path is None:
            self.train_item_ids = np.arange(num_items, dtype=np.int64)
            self.embeddings = full_embeddings
        else:
            self.train_item_ids = load_train_item_mask(
                interactions_path,
                num_items,
                skip_n_last=config.skip_n_last,
            )
            self.embeddings = full_embeddings[self.train_item_ids]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.as_tensor(emb, dtype=torch.float32)
        return {"emb": tensor_emb, "index": index, "target": tensor_emb}

    def __len__(self):
        return len(self.embeddings)


class GraphClusteringEmbDataset(data.Dataset):
    """Graph-aware dataset for the recursive graph-clustering SID methods.

    Builds the interaction graph once and exposes the node embeddings
    (``x_graph``), the adjacency (``edge_index`` / ``edge_weight``) and the
    mapping from graph-node order to global item ids (``item_ids``). The
    clustering methods consume these tensors directly in ``fit``; the
    ``__getitem__`` view exists so the dataset can also feed a plain
    ``DataLoader``.

    Args:
        config (GraphClusteringDatasetConfig): Dataset configuration.
        embeddings_path (str): Path to the ``.npy`` embedding matrix.
        interactions_path (str): Path to the interaction JSON.
        eval_mode (bool, optional): Unused by clustering methods (they always
            cluster the whole graph); kept for interface compatibility.
            Defaults to False.
    """

    name = "graph_clustering"

    def __init__(
        self,
        config: GraphClusteringDatasetConfig,
        embeddings_path: str,
        interactions_path: str,
        eval_mode: bool = False,
    ):
        self.config = config
        self.interactions_path = interactions_path
        self.embeddings_path = embeddings_path
        self.eval_mode = eval_mode
        self.graph_cluster_labels = None

        if interactions_path is not None and ("prod" not in config.graph_building.graph_type):
            full_raw_embeddings = load_embeddings(embeddings_path)

            if full_raw_embeddings.ndim != 2:
                raise ValueError(f"Expected embeddings of shape [N, D], got {tuple(full_raw_embeddings.shape)}")

            self.num_items, self.dim = full_raw_embeddings.shape

            self.train_item_ids = load_train_item_mask(
                interactions_path,
                self.num_items,
                skip_n_last=config.skip_n_last,
            )
            self.num_train_items = len(self.train_item_ids)
        else:
            full_raw_embeddings = load_embeddings(embeddings_path)
            if full_raw_embeddings.ndim != 2:
                raise ValueError(f"Expected embeddings of shape [N, D], got {tuple(full_raw_embeddings.shape)}")
            self.num_items, self.dim = full_raw_embeddings.shape

        self.graph_builder = InteractionGraphBuilder(
            interactions_path=interactions_path,
            graph_building_config=config.graph_building,
            skip_n_last=config.skip_n_last,
            num_items=self.num_items,
        )

        self.pyg_data = self.graph_builder.pyg_data
        # polars ``to_numpy`` yields a read-only buffer; copy so torch can index with it
        node_map = self.graph_builder.node_map.sort("node_id")
        self.item_ids = node_map["item_id"].to_numpy().copy()

        if len(self.item_ids) == 0:
            raise NotImplementedError("The built graph has no item nodes. Consider using `EmbDataset`")

        self.x_graph = torch.as_tensor(full_raw_embeddings[self.item_ids], dtype=torch.float32)

    @property
    def edge_index(self):
        return self.pyg_data.edge_index

    @property
    def edge_weight(self):
        return getattr(self.pyg_data, "edge_weight", None)

    def to(self, device):
        self.x_graph = self.x_graph.to(device)
        self.pyg_data = self.pyg_data.to(device)
        return self

    def __getitem__(self, index):
        emb = self.x_graph[index]
        return {"emb": emb, "index": index, "target": emb}

    def __len__(self):
        return len(self.x_graph)


class GraphAwareEmbDataset(data.Dataset):
    """
    Shared base for graph-aware embedding datasets.

    Builds the interaction graph once, exposes it as ``pyg_data`` (with the
    semantic embeddings attached as node features, ``pyg_data.x``), and resolves
    the item embeddings used for training / generation:

    - ``train_item_ids``: items kept for training (``skip_n_last`` leave-n-out)
    - ``item_ids``: item ids covered by the graph (aligned with ``pyg_data`` nodes)

    Subclasses implement :meth:`_build_fused_train`, which returns the fused
    train-item embeddings in ``train_item_ids`` order, or ``None`` when no fused
    representation is available yet (online-GNN path before the trainer has
    persisted the post-training fusion cache).

    Pipeline:
        1. Build co-occurrence graph with InteractionGraphBuilder, or reuse a provided one
        2. Convert to / use PyG graph
        3. Attach semantic embeddings as graph node features
        4. Let the subclass produce the fused train embeddings (offline or online)
        5. Scatter fused embeddings back to original item_id order (eval mode)
    Adds optional graph spectral vectors:
        self.graph_spectral_vectors : [num_items, spectral_dim]
    """

    def __init__(
        self,
        config: GraphDatasetConfig,
        embeddings_path: str,
        interactions_path: str,
        eval_mode: bool = False,
        graph_builder: InteractionGraphBuilder = None,
    ):
        self.config = config
        self.interactions_path = interactions_path
        self.embeddings_path = embeddings_path
        self.spectral_dim = config.spectral_dim
        self.eval_mode = eval_mode
        self.graph_builder = graph_builder
        self.graph_cluster_labels = None

        full_raw_embeddings = load_embeddings(embeddings_path)
        x0 = torch.as_tensor(full_raw_embeddings, dtype=torch.float32)

        if x0.ndim != 2:
            raise ValueError(f"Expected embeddings of shape [N, D], got {tuple(x0.shape)}")

        self.num_items, self.dim = x0.shape

        self.train_item_ids = load_train_item_mask(
            interactions_path,
            self.num_items,
            skip_n_last=config.skip_n_last,
        )
        self.num_train_items = len(self.train_item_ids)

        if graph_builder is None:
            graph_builder = InteractionGraphBuilder(
                interactions_path=interactions_path,
                graph_building_config=config.graph_building,
                skip_n_last=config.skip_n_last,
            )

        self.graph_builder = graph_builder
        self.pyg_data = self.graph_builder.pyg_data
        # polars ``to_numpy`` yields a read-only buffer; copy so torch can index with it
        self.item_ids = self.graph_builder.node_map["item_id"].to_numpy().copy()

        if len(self.item_ids) == 0:
            raise NotImplementedError("The built graph has no item nodes. Consider using `EmbDataset`")

        # Semantic embeddings as graph node features; consumed by the online-GNN
        # path (NeighborLoader) and unused by the offline-fusion path.
        self.pyg_data.x = x0[self.item_ids]

        fused_train = self._build_fused_train(full_raw_embeddings, x0)
        if fused_train is None:
            if self.eval_mode:
                raise FileNotFoundError(
                    f"No fused embedding cache at {self.config.fused_cache_path}; train the model first"
                )
            self.embeddings = full_raw_embeddings[self.train_item_ids]
            self.raw_embeddings = self.embeddings
        elif self.eval_mode:
            restored = full_raw_embeddings.copy()
            restored[self.train_item_ids] = fused_train
            self.embeddings = restored
            self.raw_embeddings = full_raw_embeddings
        else:
            self.embeddings = fused_train
            self.raw_embeddings = full_raw_embeddings[self.train_item_ids]

        self.prepare_additional_input(config.cluster_labels_path)

    def _build_fused_train(self, full_raw_embeddings: np.ndarray, x0: torch.Tensor) -> np.ndarray | None:
        """Return fused train-item embeddings (``train_item_ids`` order) or None."""
        raise NotImplementedError

    def _to_train_item_order(self, fused_node_order: np.ndarray) -> np.ndarray:
        """Reorder per-node fused embeddings (graph node order) into ``train_item_ids`` order."""
        fused_full = np.zeros((self.num_items, self.dim), dtype=np.float32)
        fused_full[self.item_ids] = fused_node_order
        return fused_full[self.train_item_ids]

    def _load_fused_cache(self, full_raw_embeddings: np.ndarray) -> np.ndarray:
        logger.info("loading cache path from %s", self.config.fused_cache_path)
        fused_train = np.load(self.config.fused_cache_path)

        expected_shape = (self.num_train_items, self.dim)
        if fused_train.shape != expected_shape:
            raise ValueError(
                f"Cached fused embeddings shape {fused_train.shape} "
                f"does not match expected train-only shape {expected_shape}"
            )

        return fused_train

    def prepare_additional_input(self, cluster_labels_path: str | None):
        if cluster_labels_path is not None:
            self.graph_cluster_labels = load_graph_cluster_labels(
                cluster_labels_path,
                num_items=self.num_items,
            )
            self.graph_cluster_labels = torch.tensor(self.graph_cluster_labels).T

        if self.spectral_dim > 0 and not self.eval_mode:
            spectral_train = compute_graph_spectral_vectors(
                edge_index=self.pyg_data.edge_index,
                edge_weight=self.pyg_data.edge_weight,
                num_nodes=self.pyg_data.num_nodes,
                k=self.spectral_dim,
            )

            self.graph_spectral_vectors = spectral_train.astype(np.float32)
        else:
            self.graph_spectral_vectors = np.zeros((self.num_train_items, max(self.spectral_dim, 0)), dtype=np.float32)

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        emb = self.embeddings[index]
        emb = torch.as_tensor(emb, dtype=torch.float32)

        sample = {"emb": emb, "index": index, "target": emb}

        if self.spectral_dim > 0 and not self.eval_mode:
            spectral_vec = self.graph_spectral_vectors[index]
            sample["spectral_vectors"] = torch.as_tensor(spectral_vec, dtype=torch.float32)
        elif self.graph_cluster_labels is not None:
            sample["cluster_labels"] = self.graph_cluster_labels[index]

        return sample

    def __len__(self):
        return len(self.embeddings)


class GNNFusionEmbDataset(GraphAwareEmbDataset):
    """
    Offline graph-fused embedding dataset (``--graph_signal`` / RQVAE).

    Runs the propagation conv (APPNP/S2GR) once over the built graph under
    ``torch.no_grad()``; the fused train embeddings are cached to
    ``config.fused_cache_path`` (when set) and reloaded on later runs.
    """

    name = "graph_fusion"

    def __init__(
        self,
        config: GraphFusionDatasetConfig,
        embeddings_path: str,
        interactions_path: str,
        eval_mode: bool = False,
        graph_builder: InteractionGraphBuilder = None,
    ):
        super().__init__(
            config,
            embeddings_path,
            interactions_path,
            eval_mode=eval_mode,
            graph_builder=graph_builder,
        )

    def _build_fused_train(self, full_raw_embeddings: np.ndarray, x0: torch.Tensor) -> np.ndarray | None:
        if self.config.fused_cache_path is not None and os.path.exists(self.config.fused_cache_path):
            return self._load_fused_cache(full_raw_embeddings)

        edge_index = self.pyg_data.edge_index
        edge_weight = self.pyg_data.edge_weight if self.config.use_edge_weight else None

        if self.config.encoder is None:
            self.encoder = None
            x_fused = x0[self.item_ids]
        else:
            self.encoder = get_encoder(self.config.encoder.name)(self.config.encoder)
            with torch.no_grad():
                x_fused = self.encoder(x0[self.item_ids], edge_index, edge_weight)

        fused_train = self._to_train_item_order(x_fused.cpu().numpy())

        if self.config.fused_cache_path is not None:
            fused_dir_name = os.path.dirname(self.config.fused_cache_path)
            os.makedirs(fused_dir_name, exist_ok=True)
            np.save(self.config.fused_cache_path, fused_train)

        return fused_train


class GRQVAEFusionEmbDataset(GraphAwareEmbDataset):
    """
    Online-GNN embedding dataset for GRQVAE-style models (``--sid_model grqvae``).

    Mirror of ``GNNFusionEmbDataset`` for the learned-graph-encoder path: during
    training the model consumes graph batches from ``pyg_data`` (via a
    ``NeighborLoader``) while ``__getitem__`` serves the raw semantic train
    embeddings (weak reconstruction signal). After training the trainer fuses
    the whole graph with the trained ``graph_encoder`` (:meth:`save_fused_cache`)
    and generation reloads that cache, applying the same eval restore logic.
    """

    name = "grqvae_fusion"

    def __init__(
        self,
        config: GRQVAEFusionDatasetConfig,
        embeddings_path: str,
        interactions_path: str,
        eval_mode: bool = False,
        graph_builder: InteractionGraphBuilder = None,
    ):
        super().__init__(
            config,
            embeddings_path,
            interactions_path,
            eval_mode=eval_mode,
            graph_builder=graph_builder,
        )

    def _build_fused_train(self, full_raw_embeddings: np.ndarray, x0: torch.Tensor) -> np.ndarray | None:
        if not self.eval_mode:
            # Training consumes the raw semantic embeddings; the post-train
            # fusion cache is written by the trainer after fitting.
            return None
        if self.config.fused_cache_path is None or not os.path.exists(self.config.fused_cache_path):
            raise FileNotFoundError(
                f"Missing fused embedding cache at {self.config.fused_cache_path}. "
                "Train the GRQVAE model first so the trainer can persist the post-training fusion."
            )
        return self._load_fused_cache(full_raw_embeddings)

    @torch.no_grad()
    def save_fused_cache(self, model, device):
        """Fuse the whole training graph with a trained ``graph_encoder`` and cache it.

        The cache holds the train-item fused embeddings (``train_item_ids``
        order, shape ``[num_train_items, dim]``), matching the layout expected
        by :meth:`_load_fused_cache` at generation time.
        """
        fused_cache_path = self.config.fused_cache_path
        if fused_cache_path is None:
            raise ValueError("GRQVAEFusionEmbDataset has no fused_cache_path; cannot save post-train fusion")
        if not hasattr(model, "graph_encoder"):
            raise ValueError("Model has no graph_encoder; cannot compute post-train fused embeddings")

        graph_data = self.pyg_data.to(device)
        graph_data.batch_size = graph_data.num_nodes
        model.eval()
        x_fused = model.graph_encoder(graph_data)
        fused_train = self._to_train_item_order(x_fused.detach().cpu().numpy())

        os.makedirs(os.path.dirname(fused_cache_path), exist_ok=True)
        np.save(fused_cache_path, fused_train)
        logger.info("saved fused embeddings cache to %s", fused_cache_path)


class MixedFusionDepthDataset(data.Dataset):
    """
    A mixed dataset that combines embeddings from multiple graph propagation depths (hops).

    This dataset allows for stochastic sampling of embeddings generated by GNN fusion
    at different hop distances (including the original 0-hop embeddings). It is useful
    for training models that need to be robust to varying levels of graph context or
    for data augmentation purposes.

    The dataset internally creates multiple `GNNFusionEmbDataset` instances (one for each
    non-zero hop) and one `EmbDataset` instance (for 0-hop). During training, a specific
    hop's dataset is selected probabilistically based on the provided `probs`.

    Notes
    -----
    - The dataset assumes that all underlying datasets (for different hops) have the
      same number of items and consistent ordering. A sanity check is performed during
      initialization to ensure this.
    - For hop 0, the raw embeddings are used (`EmbDataset`).
    - For hops > 0, `GNNFusionEmbDataset` is used to generate graph-fused embeddings.
    - To optimize performance, the graph builder is shared among all non-zero hop
      datasets. The first non-zero hop dataset is created first, and its graph builder
      is reused for subsequent hops.
    - The `probs` are normalized to sum to 1. If `probs` is `None`, uniform probabilities
      are assigned to each hop.
    """

    name = "mixed_depth"

    def __init__(
        self,
        config: MixedDepthDatasetConfig,
        embeddings_path: str,
        interactions_path: str,
        eval_mode: bool = False,
    ):
        hops = config.hops
        if not hops:
            raise ValueError("hops must be non-empty")
        raw_hops = [int(h) for h in hops]

        # --- probs + dedup ---
        probs = config.probs
        if probs is None:
            raw_probs = [1.0] * len(raw_hops)
        else:
            if len(probs) != len(raw_hops):
                raise ValueError("probs must match hops length")
            raw_probs = [float(p) for p in probs]

        hop_to_prob = {}
        for h, p in zip(raw_hops, raw_probs):
            hop_to_prob[h] = hop_to_prob.get(h, 0.0) + p

        self.hops = sorted(hop_to_prob.keys())
        self.probs = [hop_to_prob[h] for h in self.hops]

        total = sum(self.probs)
        if total <= 0:
            raise ValueError("sum(probs) must be > 0")
        self.probs = [p / total for p in self.probs]

        self.return_hop = config.return_hop
        self.rng = random.Random(config.seed)

        def get_cache_path(h):
            if config.fused_cache_root is None or h == 0:
                return None
            os.makedirs(config.fused_cache_root, exist_ok=True)
            return os.path.join(config.fused_cache_root, f"fused_depth{h}.npy")

        self.datasets = []

        # --- find first non-zero hop ---
        first_nonzero = next((h for h in self.hops if h > 0), None)

        def hop_fusion_config(h):
            return GraphFusionDatasetConfig(
                skip_n_last=config.skip_n_last,
                graph_building=config.graph_building,
                use_edge_weight=config.use_edge_weight,
                spectral_dim=config.spectral_dim,
                cluster_labels_path=config.cluster_labels_path,
                fused_cache_path=get_cache_path(h),
                encoder=_override_propagations(config.encoder, h),
            )

        shared_graph_builder = None

        # --- build anchor graph dataset ---
        if first_nonzero is not None:
            anchor_ds = GNNFusionEmbDataset(
                config=hop_fusion_config(first_nonzero),
                embeddings_path=embeddings_path,
                interactions_path=interactions_path,
                eval_mode=eval_mode,
                graph_builder=None,
            )
            shared_graph_builder = anchor_ds.graph_builder

        # --- build all datasets ---
        for h in self.hops:
            if h == 0:
                ds = EmbDataset(
                    config=PlainDatasetConfig(skip_n_last=config.skip_n_last),
                    embeddings_path=embeddings_path,
                    interactions_path=interactions_path,
                    eval_mode=eval_mode,
                )
            elif h == first_nonzero:
                ds = anchor_ds
            else:
                ds = GNNFusionEmbDataset(
                    config=hop_fusion_config(h),
                    embeddings_path=embeddings_path,
                    interactions_path=interactions_path,
                    eval_mode=eval_mode,
                    graph_builder=shared_graph_builder,
                )
            self.datasets.append(ds)

        # --- sanity ---
        lengths = [len(ds) for ds in self.datasets]
        if len(set(lengths)) != 1:
            raise ValueError(f"Inconsistent dataset lengths: {lengths}")

        self.dim = self.datasets[0].dim

    def __len__(self):
        return len(self.datasets[0])

    def _sample_dataset_id(self):
        return self.rng.choices(range(len(self.datasets)), weights=self.probs, k=1)[0]

    def __getitem__(self, idx):
        ds_id = self._sample_dataset_id()
        # hop = self.hops[ds_id]

        return self.datasets[ds_id][idx]
