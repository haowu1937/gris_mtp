import os
from typing import Any

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import NeighborLoader

from gensid.config.dataset_config import (
    DatasetConfig,
    GRQVAEFusionDatasetConfig,
    GraphClusteringDatasetConfig,
    GraphFusionDatasetConfig,
    MixedDepthDatasetConfig,
    PlainDatasetConfig,
)
from gensid.data import EdgeWeightCollator
from gensid.registry import REGISTRY


def get_dataset_config(args: Any) -> DatasetConfig:
    """Map the CLI flags to a typed :class:`DatasetConfig`.

    ``--sid_model`` is the primary switch: GRQVAE-style methods always get the
    online-GNN dataset (``GRQVAEFusionDatasetConfig``). For the others the
    mutually-exclusive ``--graph_signal`` / ``--graph_augmentation`` flags select
    the offline-fusion or multi-hop datasets, falling back to plain embeddings.
    """
    graph_building = REGISTRY.get_graph_building_config(args.graph_type).from_args(args)

    if args.sid_model in ("recdmon", "spectral", "stacked_dmon"):
        return GraphClusteringDatasetConfig(
            graph_building=graph_building,
            skip_n_last=args.skip_n_last,
            # graph_builder_load_path=args.graph_builder_load_path
        )

    if getattr(args, "sid_model") == "grqvae":
        return GRQVAEFusionDatasetConfig(
            graph_building=graph_building,
            fused_cache_path=os.path.join(args.ckpt_dir, "graph.npy"),
            use_edge_weight=args.use_edge_weight,
            skip_n_last=args.skip_n_last,
            spectral_dim=args.spectral_dim,
            cluster_labels_path=args.cluster_labels_path,
        )

    if args.graph_augmentation:
        return MixedDepthDatasetConfig(
            graph_building=graph_building,
            encoder=REGISTRY.get_encoder_config(args.convolution_type).from_args(args),
            hops=tuple(args.hops),
            probs=tuple(args.probs) if args.probs is not None else None,
            fused_cache_root=os.path.join(args.ckpt_dir, "multihop/"),
            use_edge_weight=args.use_edge_weight,
            skip_n_last=args.skip_n_last,
            spectral_dim=args.spectral_dim,
            cluster_labels_path=args.cluster_labels_path,
            seed=args.seed,
        )

    if args.graph_signal:
        return GraphFusionDatasetConfig(
            graph_building=graph_building,
            encoder=REGISTRY.get_encoder_config(args.convolution_type).from_args(args),
            fused_cache_path=os.path.join(args.ckpt_dir, "graph.npy"),
            use_edge_weight=args.use_edge_weight,
            skip_n_last=args.skip_n_last,
            spectral_dim=args.spectral_dim,
            cluster_labels_path=args.cluster_labels_path,
        )

    return PlainDatasetConfig(skip_n_last=args.skip_n_last)


def build_dataset(config: DatasetConfig, embeddings_path: str, interactions_path: str, eval_mode: bool = False):
    """Build the :class:`Dataset` selected by ``config.name``."""
    dataset_cls = REGISTRY.get_dataset(config.name)
    return dataset_cls(
        config,
        embeddings_path=embeddings_path,
        interactions_path=interactions_path,
        eval_mode=eval_mode,
    )


def build_data_loader(
    dataset, config: DatasetConfig, args: Any, eval_mode: bool = False, dist_state=None
) -> DataLoader:
    """Wrap a dataset into the train/eval loader.

    The online-GNN (GRQVAE) dataset is trained on a ``NeighborLoader`` over its
    ``pyg_data``; evaluation always uses a plain ``DataLoader`` so generation runs
    on the (possibly fused) embedding matrix like every other method.

    Under distributed training (``dist_state`` with ``world_size > 1``) the train
    loader shards the dataset with a ``DistributedSampler``; eval stays full-data.
    """
    batch_size = args.eval_batch_size if eval_mode else args.train_batch_size
    shuffle = not eval_mode

    if isinstance(config, GRQVAEFusionDatasetConfig) and not eval_mode:
        return NeighborLoader(dataset.pyg_data, num_neighbors=[30] * 2, batch_size=batch_size, shuffle=True)

    sampler = None
    distributed = dist_state is not None and dist_state.world_size > 1
    if distributed and not eval_mode:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_state.world_size,
            rank=dist_state.rank,
            shuffle=True,
        )
        shuffle = False

    collate_fn = (
        EdgeWeightCollator(dataset.pyg_data, num_anchors=args.num_anchors)
        if args.use_edge_reconstruction_loss
        else default_collate
    )
    return DataLoader(
        dataset,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if not eval_mode else None,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=args.persistent_workers if not eval_mode else False,
        collate_fn=collate_fn,
    )
