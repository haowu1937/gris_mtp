from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ModelConfig:
    """Base for any SID method configuration.

    Args:
        name (str or None, optional): Registered name of the SID method, used as a
            discriminant and for lookups. Defaults to None.
    """

    name: str | None = None

    def to_kwargs(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "name"}

    @classmethod
    def from_args(cls, args) -> "ModelConfig":
        kwargs = {k: getattr(args, k, v) for k, v in cls().to_kwargs().items()}
        # `cf_embedding` is configured via the `--cf_emb` checkpoint path.
        if "cf_embedding" in kwargs and hasattr(args, "cf_emb") and kwargs["cf_embedding"] is None:
            kwargs["cf_embedding"] = getattr(args, "cf_emb")
        return cls(**kwargs)


@dataclass(frozen=True)
class RQVAEModelConfig(ModelConfig):
    """Configuration for the residual-quantized VAE (``RQVAE``) SID method.

    Args:
        name (Literal["rqvae"], optional): Registered name of the SID method,
            fixed to "rqvae". Defaults to "rqvae".

        in_dim (int, optional): Dimensionality of the input item embeddings.
            Defaults to 768.

        num_emb_list (tuple of int, optional): Number of embeddings (codebook size)
            for each residual VQ stage. Defaults to (256, 256, 256, 256).

        e_dim (int, optional): Dimensionality of the latent embedding space.
            Defaults to 32.

        layers (tuple of int, optional): Hidden layer sizes of the shared
            encoder/decoder MLPs. Defaults to (2048, 1024, 512, 256, 128, 64).

        dropout_prob (float, optional): Dropout probability for the MLP layers.
            Defaults to 0.0.

        bn (bool, optional): Whether to use batch normalization in the MLP layers.
            Defaults to False.

        loss_type (str, optional): Reconstruction loss type: "mse" or "l1".
            Defaults to "mse".

        quant_loss_weight (float, optional): Weight of the vector quantization
            commitment loss. Defaults to 1.0.

        kmeans_init (bool, optional): Whether to initialize codebooks with k-means.
            Defaults to True.

        kmeans_iters (int, optional): Number of k-means initialization iterations.
            Defaults to 100.

        sk_epsilons (tuple of float, optional): Per-stage Sinkhorn epsilon used to
            break ties/collisions during index generation. Defaults to (0.0, 0.0, 0.0, 0.003).

        sk_iters (int, optional): Number of Sinkhorn iterations during collision
            fixing. Defaults to 50.

        beta (float, optional): Commitment loss weight of the vector quantizer.
            Defaults to 0.1.

        alpha (float, optional): Weight of the contrastive CF loss.
            Defaults to 0.1.

        n_clusters (int, optional): Number of clusters used to label codebook
            entries (constrained k-means) at generation time. Defaults to 10.

        sample_strategy (str, optional): Strategy for sampling during training.
            Defaults to "all".

        cf_embedding (Any, optional): Pre-computed contrastive embeddings used by
            the CF loss. Defaults to None.

        use_cf_loss (bool, optional): Whether to include the contrastive CF loss in
            the total loss. Defaults to True.

        cf_loss_use_codebook_rep (bool, optional): Whether to compute the CF loss
            on aggregated codebook representations. Defaults to False.

        cf_loss_last_m_codebooks (int or None, optional): If set, use only the last
            m codebooks when aggregating the CF-loss representation.
            Defaults to None.

        alpha_graph_cluster (float, optional): Weight of the hierarchical graph
            clustering loss. Defaults to 0.0.

        graph_cluster_tau (float, optional): Temperature for the graph-cluster
            similarity in ``graph_cluster_loss``. Defaults to 0.1.

        num_anchors (int, optional): Number of anchor points used to bound the
            graph/cluster loss computation. Defaults to 1024.

        lognorm_weights (bool, optional): If True, apply log1p to the edge
            weights inside the edge-reconstruction loss (compresses the skewed
            co-occurrence counts). Defaults to False.
    """

    name: Literal["rqvae"] = "rqvae"
    in_dim: int = 768
    num_emb_list: tuple[int, ...] = (256, 256, 256, 256)
    e_dim: int = 32
    layers: tuple[int, ...] = (2048, 1024, 512, 256, 128, 64)
    dropout_prob: float = 0.0
    bn: bool = False
    loss_type: str = "mse"
    quant_loss_weight: float = 1.0
    kmeans_init: bool = True
    kmeans_iters: int = 100
    sk_epsilons: tuple[float, ...] = (0.0, 0.0, 0.0, 0.003)
    sk_iters: int = 50
    beta: float = 0.1
    alpha: float = 0.1
    n_clusters: int = 10
    sample_strategy: str = "all"
    cf_embedding: Any = None
    use_cf_loss: bool = True
    cf_loss_use_codebook_rep: bool = False
    cf_loss_last_m_codebooks: int | None = None
    alpha_graph_cluster: float = 0.0
    graph_cluster_tau: float = 0.1
    num_anchors: int = 1024
    lognorm_weights: bool = False


@dataclass(frozen=True)
class GRQVAEModelConfig(RQVAEModelConfig):
    """Configuration for the graph-informed RQVAE variant (``GRQVAE``).

    Extends ``RQVAEModelConfig`` with the knobs of the ``GNNFusionEncoder`` that is
    prepended to the VAE.

    Args:
        name (Literal["grqvae"], optional): Registered name of the SID method,
            fixed to "grqvae". Defaults to "grqvae".

        in_dim (int, optional): Dimensionality of the input item embeddings.
            Defaults to 768.

        num_emb_list (tuple of int, optional): Number of embeddings (codebook size)
            for each residual VQ stage. Defaults to (256, 256, 256, 256).

        e_dim (int, optional): Dimensionality of the latent embedding space.
            Defaults to 32.

        layers (tuple of int, optional): Hidden layer sizes of the shared
            encoder/decoder MLPs. Defaults to (2048, 1024, 512, 256, 128, 64).

        dropout_prob (float, optional): Dropout probability for the MLP layers.
            Defaults to 0.0.

        bn (bool, optional): Whether to use batch normalization in the MLP layers.
            Defaults to False.

        loss_type (str, optional): Reconstruction loss type: "mse" or "l1".
            Defaults to "mse".

        quant_loss_weight (float, optional): Weight of the vector quantization
            commitment loss. Defaults to 1.0.

        kmeans_init (bool, optional): Whether to initialize codebooks with k-means.
            Defaults to True.

        kmeans_iters (int, optional): Number of k-means initialization iterations.
            Defaults to 100.

        sk_epsilons (tuple of float, optional): Per-stage Sinkhorn epsilon used to
            break ties/collisions during index generation. Defaults to (0.0, 0.0, 0.0, 0.003).

        sk_iters (int, optional): Number of Sinkhorn iterations during collision
            fixing. Defaults to 50.

        beta (float, optional): Commitment loss weight of the vector quantizer.
            Defaults to 0.1.

        alpha (float, optional): Weight of the contrastive CF loss.
            Defaults to 0.1.

        n_clusters (int, optional): Number of clusters used to label codebook
            entries (constrained k-means) at generation time. Defaults to 10.

        sample_strategy (str, optional): Strategy for sampling during training.
            Defaults to "all".

        cf_embedding (Any, optional): Pre-computed contrastive embeddings used by
            the CF loss. Defaults to None.

        use_cf_loss (bool, optional): Whether to include the contrastive CF loss in
            the total loss. Defaults to True.

        cf_loss_use_codebook_rep (bool, optional): Whether to compute the CF loss
            on aggregated codebook representations. Defaults to False.

        cf_loss_last_m_codebooks (int or None, optional): If set, use only the last
            m codebooks when aggregating the CF-loss representation.
            Defaults to None.

        alpha_graph_cluster (float, optional): Weight of the hierarchical graph
            clustering loss. Defaults to 0.0.

        num_anchors (int, optional): Number of anchor points used to bound the
            graph/cluster loss computation. Defaults to 1024.

        gnn_conv_type (str, optional): Base convolutional layer of the GNN encoder:
            "sage", "gat", "gcn", "gin" or "gated_graph". Defaults to "gat".

        gnn_hidden_dim (int or None, optional): Hidden dimensionality of the GNN
            encoder; None defers to the model's own default. Defaults to None.

        gnn_dropout (float, optional): Dropout probability of the GNN encoder.
            Defaults to 0.0.

        gnn_heads (int, optional): Number of attention heads of the GNN encoder
            (GAT only). Defaults to 4.

        gnn_layers (int, optional): Number of GNN convolutional layers.
            Defaults to 2.
    """

    name: Literal["grqvae"] = "grqvae"
    gnn_conv_type: str = "gat"
    gnn_hidden_dim: int | None = None
    gnn_dropout: float = 0.0
    gnn_heads: int = 4
    gnn_layers: int = 2


@dataclass(frozen=True)
class RecDMoNModelConfig(ModelConfig):
    """Configuration for the recursive DMoN hierarchical SID method (``RecDMoN``).

    Recursively splits the interaction graph with a locally-trained DMoN
    assignment network, producing one codebook per recursion level.

    Args:
        name (Literal["recdmon"], optional): Registered name of the SID method,
            fixed to "recdmon". Defaults to "recdmon".

        recursive_epochs (int, optional): Number of optimization epochs run per
            local DMoN split. Defaults to 1000.

        assign_hidden_dim (int, optional): Hidden size of the local assignment
            MLP. Defaults to 256.

        dropout (float, optional): Dropout of the local assignment MLP.
            Defaults to 0.0.

        max_dense_nodes (int, optional): Hard cap on the size of a local subset;
            larger subsets raise an error because the DMoN loss needs a dense
            adjacency slice. Defaults to 100000.

        dmon_balance_weight (float, optional): Weight of the cluster-balance
            regularizer in the DMoN loss. Defaults to 1.0.

        dmon_entropy_weight (float, optional): Weight of the assignment-entropy
            regularizer in the DMoN loss. Defaults to 0.01.

        lr (float, optional): Learning rate of the local assignment MLP optimizer.
            Defaults to 3e-4.

        recursive_factor (int, optional): Maximum number of child clusters per
            parent subset. Defaults to 6.

        recursive_levels (int, optional): Number of hierarchy levels (codebooks).
            Defaults to 4.

        grad_clip (float or None, optional): Max gradient norm of the local
            assignment MLP; None disables clipping. Defaults to 5.0.
    """

    name: Literal["recdmon"] = "recdmon"
    recursive_epochs: int = 1000
    assign_hidden_dim: int = 256
    dropout: float = 0.0
    max_dense_nodes: int = 100000
    dmon_balance_weight: float = 1.0
    dmon_entropy_weight: float = 0.01
    lr: float = 3e-4
    weight_decay: float = 1e-4
    recursive_factor: int = 6
    recursive_levels: int = 4
    grad_clip: float | None = 5.0


@dataclass(frozen=True)
class SpectralModelConfig(ModelConfig):
    """Configuration for the recursive spectral-clustering SID method (``Spectral``).

    Recursively splits the interaction graph with spectral clustering (no learned
    parameters), producing one codebook per recursion level.

    Args:
        name (Literal["spectral"], optional): Registered name of the SID method,
            fixed to "spectral". Defaults to "spectral".

        seed (int or None, optional): Seed for the spectral k-means step.
            Defaults to None.

        recursive_factor (int, optional): Maximum number of child clusters per
            parent subset. Defaults to 6.

        recursive_levels (int, optional): Number of hierarchy levels (codebooks).
            Defaults to 4.
    """

    name: Literal["spectral"] = "spectral"
    seed: int | None = None
    recursive_factor: int = 6
    recursive_levels: int = 4


@dataclass(frozen=True)
class StackedDMoNModelConfig(ModelConfig):
    """Configuration for the stacked DMoN SID method (``StackedDMoN``).

    Args:
        name (Literal["stacked_dmon"], optional): Registered name of the SID method,
            fixed to "stacked_dmon". Defaults to "stacked_dmon".

        in_dim (int, optional): Dimensionality of the input item embeddings.
            Defaults to 768.

        num_emb_list (tuple of int, optional): Number of cluster (codebook size)
            for each DMoN level. Defaults to (128, 32).

        e_dim (int, optional): Dimensionality of the latent embedding space.
            Defaults to 32.

        dmon_hidden_dim (int, optional): Hidden size of the refiners/assigners.
            Defaults to 256.

        assign_hidden_dim (int, optional): Hidden size of the local assignment
            MLP. Defaults to 256.

        dropout (float, optional): Dropout of the local assignment MLP.
            Defaults to 0.0.

        dmon_balance_weight (float, optional): Weight of the cluster-balance
            regularizer in the DMoN loss. Defaults to 1.0.

        dmon_entropy_weight (float, optional): Weight of the assignment-entropy
            regularizer in the DMoN loss. Defaults to 0.01.
    """

    name: Literal["stacked_dmon"] = "stacked_dmon"
    in_dim: int = 768
    num_emb_list: tuple[int, ...] = (256, 256, 256, 256)
    e_dim: int = 32
    dmon_hidden_dim: int = 256
    assign_hidden_dim: int = 256
    dropout: float = 0.0
    dmon_balance_weight: float = 1.0
    dmon_entropy_weight: float = 0.01


@dataclass(frozen=True)
class RKMeansModelConfig(ModelConfig):
    """Configuration for the (recursive) k-means SID method (``RKMeans``).

    A flat or recursively-clustered set of codebooks: each level clusters the
    residual (or the input) with k-means and assigns one codebook per level.

    Args:
        name (Literal["rkmeans"], optional): Registered name of the SID method,
            fixed to "rkmeans". Defaults to "rkmeans".

        depth (int, optional): Number of hierarchical clustering levels (i.e. the
            number of codebooks / SID positions). Defaults to 4.

        n_clusters (int or list of int, optional): Number of clusters per level; a
            single int applies to every level, a list gives per-level sizes.
            Defaults to 256.

        distance_function (str, optional): Name of the distance function used for
            clustering (key into ``DISTANCE_FUNCTIONS``). Defaults to "euclidean".

        mean_function (str, optional): Name of the centroid aggregation function
            (key into ``MEAN_FUNCTIONS``). Defaults to "mean".

        norm_input (bool, optional): If True, normalize the input embeddings before
            clustering. Defaults to True.

        kmeans_backend (Literal["torch", "faiss"], optional): Which k-means
            implementation to use: "torch" (``TorchKMeans``) or "faiss"
            (``FaissKMeans``, runs on CPU). Defaults to "torch".

        kmeanspp_exact (bool, optional): Force the exact (bit-identical) full-matrix
            k-means++ initialization instead of the fast incremental variant.
            Defaults to False (incremental; exact is used automatically when
            ``n_clusters > n_points``).
    """

    name: Literal["rkmeans"] = "rkmeans"
    depth: int = 4
    n_clusters: int | list[int] = 256
    distance_function: str = "euclidean"
    mean_function: str = "mean"
    norm_input: bool = True
    fit_mini_batch: bool = False
    mini_batch_epochs: int = 100
    init_batch_size: int = 256
    max_steps: int | None = None
    init_mode: Literal["kmeans++", "random"] | None = "kmeans++"
    kmeans_backend: Literal["torch", "faiss"] = "torch"
    kmeanspp_exact: bool = False

    @classmethod
    def from_args(cls, args) -> "RKMeansModelConfig":
        num_emb_list = getattr(args, "num_emb_list", None)
        if num_emb_list is not None:
            n_clusters = num_emb_list[0] if len(set(num_emb_list)) == 1 else list(num_emb_list)
            depth = len(num_emb_list)
        else:
            n_clusters = getattr(args, "n_clusters", cls.n_clusters)
            depth = getattr(args, "depth", cls.depth)
        return cls(
            depth=depth,
            n_clusters=n_clusters,
            distance_function=getattr(args, "distance_function", cls.distance_function),
            mean_function=getattr(args, "mean_function", cls.mean_function),
            norm_input=bool(getattr(args, "norm_kmeans_input", cls.norm_input)),
            fit_mini_batch=bool(getattr(args, "fit_mini_batch", cls.fit_mini_batch)),
            mini_batch_epochs=getattr(args, "mini_batch_epochs", cls.mini_batch_epochs),
            init_batch_size=getattr(args, "init_batch_size", cls.init_batch_size),
            max_steps=getattr(args, "max_steps", cls.max_steps),
            init_mode=getattr(args, "init_mode", cls.init_mode),
            kmeans_backend=getattr(args, "kmeans_backend", cls.kmeans_backend),
            kmeanspp_exact=bool(getattr(args, "kmeanspp_exact", cls.kmeanspp_exact)),
        )


@dataclass(frozen=True)
class HierarchicalRKMeansModelConfig(ModelConfig):
    """Configuration for the hierarchical (recursive) k-means SID method.

    Same knobs as ``RKMeansModelConfig``, with ``n_clusters`` fixed to a per-level
    list of sizes.

    Args:
        name (Literal["hierarchical_rkmeans"], optional): Registered name of the
            SID method, fixed to "hierarchical_rkmeans".
            Defaults to "hierarchical_rkmeans".

        n_clusters (list of int, optional): Number of clusters per level, one entry
            per level. Defaults to [256, 256, 256, 256].

        depth (int, optional): Inherited from ``RKMeansModelConfig``; note it must
            stay consistent with ``n_clusters``. Defaults to 4.

        distance_function (str, optional): Name of the distance function used for
            clustering (key into ``DISTANCE_FUNCTIONS``). Defaults to "euclidean".

        mean_function (str, optional): Name of the centroid aggregation function
            (key into ``MEAN_FUNCTIONS``). Defaults to "mean".

        norm_input (bool, optional): If True, normalize the input embeddings before
            clustering. Defaults to True.

        kmeans_backend (Literal["torch", "faiss"], optional): Which k-means
            implementation to use: "torch" (``TorchKMeans``) or "faiss"
            (``FaissKMeans``, runs on CPU). Defaults to "torch".

        kmeanspp_exact (bool, optional): Force the exact (bit-identical) full-matrix
            k-means++ initialization instead of the fast incremental variant.
            Defaults to False (incremental; exact is used automatically when
            ``n_clusters > n_points``).
    """

    name: Literal["hierarchical_rkmeans"] = "hierarchical_rkmeans"
    depth: int = 4
    n_clusters: list[int] = field(default_factory=lambda: [256, 256, 256, 256])
    distance_function: str = "euclidean"
    mean_function: str = "mean"
    norm_input: bool = True
    kmeans_backend: Literal["torch", "faiss"] = "torch"
    kmeanspp_exact: bool = False

    @classmethod
    def from_args(cls, args) -> "HierarchicalRKMeansModelConfig":
        num_emb_list = getattr(args, "num_emb_list", None)
        if num_emb_list is not None:
            n_clusters = list(num_emb_list)
            depth = len(n_clusters)
        else:
            # n_clusters is a default_factory field (no class attr), so fall back explicitly
            n_raw = getattr(args, "n_clusters", None)
            if n_raw is None:
                n_raw = [256, 256, 256, 256]
            depth = getattr(args, "depth", cls.depth)
            n_clusters = [n_raw] * depth if isinstance(n_raw, int) else list(n_raw)
        return cls(
            depth=depth,
            n_clusters=n_clusters,
            distance_function=getattr(args, "distance_function", cls.distance_function),
            mean_function=getattr(args, "mean_function", cls.mean_function),
            norm_input=bool(getattr(args, "norm_kmeans_input", cls.norm_input)),
            kmeans_backend=getattr(args, "kmeans_backend", cls.kmeans_backend),
            kmeanspp_exact=bool(getattr(args, "kmeanspp_exact", cls.kmeanspp_exact)),
        )


@dataclass(frozen=True)
class PQModelConfig(ModelConfig):
    """Configuration for the product-quantizer SID method (``PQ``).

    A flat product quantizer: each ``D``-dim vector is split into
    ``n_subquantizers`` sub-vectors, each quantized with its own ``2**n_bits``-
    entry codebook. The SID is the ``n_subquantizers`` concatenated sub-codes,
    i.e. ``n_subquantizers`` codebooks of size ``2**n_bits``.

    Args:
        name (Literal["pq"], optional): Registered name of the SID method,
            fixed to "pq". Defaults to "pq".

        in_dim (int, optional): Dimensionality of the input item embeddings.
            Overwritten by the pipeline with the dataset dim. Defaults to 768.

        n_subquantizers (int, optional): Number of sub-quantizers (SID codebooks
            / positions). ``in_dim`` must be divisible by it. Defaults to 4.

        n_bits (int, optional): Bits per sub-code; each codebook holds
            ``2**n_bits`` entries. Defaults to 8.

        quantizer_backend (Literal["torch", "faiss"], optional): Which PQ
            implementation to use: "torch" (``TorchPQ``, device-agnostic) or
            "faiss" (``FaissPQ``, runs on CPU). Defaults to "torch".

        kmeans_iters (int, optional): Number of k-means iterations per subspace
            (torch backend only; faiss fixes this at 25). Defaults to 25.
    """

    name: Literal["pq"] = "pq"
    in_dim: int = 768
    n_subquantizers: int = 4
    n_bits: int = 8
    quantizer_backend: Literal["torch", "faiss"] = "torch"
    kmeans_iters: int = 25

    @classmethod
    def from_args(cls, args) -> "PQModelConfig":
        return cls(
            in_dim=getattr(args, "in_dim", cls.in_dim),
            n_subquantizers=getattr(args, "n_subquantizers", cls.n_subquantizers),
            n_bits=getattr(args, "n_bits", cls.n_bits),
            quantizer_backend=getattr(args, "quantizer_backend", cls.quantizer_backend),
            kmeans_iters=getattr(args, "kmeans_iters", cls.kmeans_iters),
        )


@dataclass(frozen=True)
class OPQModelConfig(ModelConfig):
    """Configuration for the optimized product-quantizer SID method (``OPQ``).

    Like ``PQModelConfig``, but learns an orthonormal rotation ``A`` before the
    per-subspace product quantization to reduce quantization distortion.

    Args:
        name (Literal["opq"], optional): Registered name of the SID method,
            fixed to "opq". Defaults to "opq".

        in_dim (int, optional): Dimensionality of the input item embeddings.
            Overwritten by the pipeline with the dataset dim. Defaults to 768.

        n_subquantizers (int, optional): Number of sub-quantizers (SID codebooks
            / positions). ``in_dim`` must be divisible by it. Defaults to 4.

        n_bits (int, optional): Bits per sub-code; each codebook holds
            ``2**n_bits`` entries. Defaults to 8.

        quantizer_backend (Literal["torch", "faiss"], optional): Which OPQ
            implementation to use: "torch" (``TorchOPQ``, device-agnostic) or
            "faiss" (``FaissOPQ``, runs on CPU). Defaults to "torch".

        niter (int, optional): Number of outer rotation-refinement iterations.
            Defaults to 25.

        kmeans_iters (int, optional): Number of k-means iterations per subspace
            (torch backend only; faiss fixes this at 25). Defaults to 25.
    """

    name: Literal["opq"] = "opq"
    in_dim: int = 768
    n_subquantizers: int = 4
    n_bits: int = 8
    quantizer_backend: Literal["torch", "faiss"] = "torch"
    niter: int = 25
    kmeans_iters: int = 25

    @classmethod
    def from_args(cls, args) -> "OPQModelConfig":
        return cls(
            in_dim=getattr(args, "in_dim", cls.in_dim),
            n_subquantizers=getattr(args, "n_subquantizers", cls.n_subquantizers),
            n_bits=getattr(args, "n_bits", cls.n_bits),
            quantizer_backend=getattr(args, "quantizer_backend", cls.quantizer_backend),
            niter=getattr(args, "opq_niter", cls.niter),
            kmeans_iters=getattr(args, "kmeans_iters", cls.kmeans_iters),
        )
