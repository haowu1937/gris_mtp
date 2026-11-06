from dataclasses import dataclass, field
from typing import Literal

from gensid.config.encoder_config import APPNPEncoderConfig, EncoderConfig
from gensid.config.graph_build_config import AdjacentCoocGraphConfig, GraphBuildingConfig


@dataclass(frozen=True)
class DatasetConfig:
    """Base for SID dataset configs: how embeddings + interactions become a Dataset.

    A dataset config owns the *data-handling* concerns: the train/eval split
    (``skip_n_last``), how the interaction graph is built and canonicalized
    (``graph_building``), fusion
    caching, and dataset-specific sampling knobs (hops, subgraph size, ...).

    The actual embedding refinement is *not* a dataset concern: it is delegated
    to a composed ``EncoderConfig`` (APPNP/S2GR propagation) or, for the learned
    GNN path, handled by the SID method itself (e.g. ``GRQVAE``).

    Args:
        name (str or None, optional): Registered name of the dataset, used to
            look up the concrete implementation. Defaults to None.

        skip_n_last (int, optional): Number of trailing items dropped from each
            user sequence (kept for leave-n-out evaluation). Defaults to 2.
    """

    name: str | None = None
    skip_n_last: int = 2

    def to_kwargs(self) -> dict:
        return {attr: value for attr, value in self.__dict__.items()}

    @classmethod
    def from_args(cls, args) -> "DatasetConfig":
        return cls(**{k: getattr(args, k, v) for k, v in cls().to_kwargs().items()})


@dataclass(frozen=True)
class PlainDatasetConfig(DatasetConfig):
    """Raw semantic embeddings, no graph (``EmbDataset``).

    Args:
        name (Literal["plain"], optional): Registered name of the dataset, fixed
            to "plain". Defaults to "plain".

        skip_n_last (int, optional): Number of trailing items dropped from each
            user sequence. Defaults to 2.
    """

    name: Literal["plain"] = "plain"


@dataclass(frozen=True)
class GraphClusteringDatasetConfig(DatasetConfig):
    """Graph-aware dataset currently used only as a storage for clustering methods like RecDMoN.

    Args:
        name (str or None, optional): Registered name of the dataset. Defaults to
            None.

        skip_n_last (int, optional): Number of trailing items dropped from each
            user sequence. Defaults to 2.

        graph_building (GraphBuildingConfig, optional): How the interaction graph
            is built (strategy + knobs). Defaults to AdjacentCoocGraphConfig.
    """

    name: Literal["graph_clustering"] = "graph_clustering"
    graph_building: GraphBuildingConfig = field(default_factory=AdjacentCoocGraphConfig)


@dataclass(frozen=True)
class GraphDatasetConfig(DatasetConfig):
    """Knobs shared by every graph-aware dataset (they all build an interaction graph).

    Args:
        name (str or None, optional): Registered name of the dataset. Defaults to
            None.

        skip_n_last (int, optional): Number of trailing items dropped from each
            user sequence. Defaults to 2.

        graph_building (GraphBuildingConfig, optional): How the interaction graph
            is built (strategy + knobs). Defaults to AdjacentCoocGraphConfig.

        use_edge_weight (bool, optional): If True, use the graph edge weights
            during fusion/propagation. Defaults to False.

        spectral_dim (int, optional): If > 0, attach ``spectral_dim`` graph
            spectral vectors to every sample. Defaults to 0.

        cluster_labels_path (str or None, optional): Path to hierarchical graph
            cluster labels used as extra supervision. Defaults to None.
    """

    name: str | None = None
    graph_building: GraphBuildingConfig = field(default_factory=AdjacentCoocGraphConfig)
    use_edge_weight: bool = False
    spectral_dim: int = 0
    cluster_labels_path: str | None = None


@dataclass(frozen=True)
class GraphFusionDatasetConfig(GraphDatasetConfig):
    """Offline graph-fused embeddings (``--graph_signal`` -> ``GNNFusionEmbDataset``).

    Runs the propagation conv (APPNP/S2GR) once over the built graph and trains
    the SID method on the fused embeddings.

    Args:
        name (Literal["graph_fusion"], optional): Registered name of the dataset,
            fixed to "graph_fusion". Defaults to "graph_fusion".

        skip_n_last (int, optional): Number of trailing items dropped from each
            user sequence. Defaults to 2.

        graph_building (GraphBuildingConfig, optional): How the interaction graph
            is built. Defaults to AdjacentCoocGraphConfig.

        use_edge_weight (bool, optional): If True, use the graph edge weights
            during propagation. Defaults to False.

        spectral_dim (int, optional): If > 0, attach spectral vectors. Defaults to 0.

        cluster_labels_path (str or None, optional): Path to hierarchical graph
            cluster labels. Defaults to None.

        fused_cache_path (str or None, optional): Where the fused embeddings are
            cached (as ``.npy``); None disables caching. Defaults to None.

        encoder (EncoderConfig, optional): Offline propagation operator applied to
            the item embeddings (APPNP or S2GR). Defaults to APPNPEncoderConfig.
    """

    name: Literal["graph_fusion"] = "graph_fusion"
    fused_cache_path: str | None = None
    encoder: EncoderConfig = field(default_factory=APPNPEncoderConfig)


@dataclass(frozen=True)
class MixedDepthDatasetConfig(GraphDatasetConfig):
    """Stochastic multi-hop fusion (``--graph_augmentation`` -> ``MixedFusionDepthDataset``).

    Per sample, a propagation depth is drawn from ``hops`` (0 = raw embeddings)
    and the corresponding fused representation is retrieved.

    Args:
        name (Literal["mixed_depth"], optional): Registered name of the dataset,
            fixed to "mixed_depth". Defaults to "mixed_depth".

        skip_n_last (int, optional): Number of trailing items dropped from each
            user sequence. Defaults to 2.

        graph_building (GraphBuildingConfig, optional): How the interaction graph
            is built. Defaults to AdjacentCoocGraphConfig.

        use_edge_weight (bool, optional): If True, use the graph edge weights
            during propagation. Defaults to False.

        spectral_dim (int, optional): If > 0, attach spectral vectors. Defaults to 0.

        cluster_labels_path (str or None, optional): Path to hierarchical graph
            cluster labels. Defaults to None.

        hops (tuple of int, optional): Propagation depths to combine; 0 stands for
            the raw (unfused) embeddings. Defaults to (0,).

        probs (tuple of float or None, optional): Sampling probability per hop;
            entries are normalized to sum to 1, and None selects uniform
            probabilities. Defaults to None.

        fused_cache_root (str or None, optional): Directory used to cache per-hop
            fused embeddings as ``fused_depth{h}.npy``; None disables caching.
            Defaults to None.

        return_hop (bool, optional): Reserved; whether the sampled hop should be
            returned alongside the sample. Defaults to False.

        seed (int or None, optional): Seed of the per-sample hop sampler.
            Defaults to None.

        encoder (EncoderConfig, optional): Propagation operator used for the
            non-zero hops (APPNP or S2GR); its ``n_propagations`` is overridden
            per hop. Defaults to APPNPEncoderConfig.
    """

    name: Literal["mixed_depth"] = "mixed_depth"
    hops: tuple[int, ...] = (0,)
    probs: tuple[float, ...] | None = None
    fused_cache_root: str | None = None
    return_hop: bool = False
    seed: int | None = None
    encoder: EncoderConfig = field(default_factory=APPNPEncoderConfig)


@dataclass(frozen=True)
class GRQVAEFusionDatasetConfig(GraphDatasetConfig):
    """Online-GNN fused embeddings for GRQVAE-style models (``--sid_model grqvae``).

    The dataset itself does not fuse: during training a ``NeighborLoader`` feeds
    the graph batches (with semantic node features) to the model's own
    ``graph_encoder``; after training the trainer persists the fused embeddings
    to ``fused_cache_path``, and at generation time they are scattered back onto
    the training items exactly like ``GNNFusionEmbDataset`` does.

    Args:
        name (Literal["grqvae_fusion"], optional): Registered name of the dataset,
            fixed to "grqvae_fusion". Defaults to "grqvae_fusion".

        skip_n_last (int, optional): Number of trailing items dropped from each
            user sequence. Defaults to 2.

        graph_building (GraphBuildingConfig, optional): How the interaction graph
            is built. Defaults to AdjacentCoocGraphConfig.

        use_edge_weight (bool, optional): If True, use the graph edge weights
            during propagation. Defaults to False.

        spectral_dim (int, optional): If > 0, attach spectral vectors. Defaults to 0.

        cluster_labels_path (str or None, optional): Path to hierarchical graph
            cluster labels. Defaults to None.

        fused_cache_path (str or None, optional): Where the post-training fused
            embeddings are cached (as ``.npy``); None disables caching.
            Defaults to None.
    """

    name: Literal["grqvae_fusion"] = "grqvae_fusion"
    fused_cache_path: str | None = None
