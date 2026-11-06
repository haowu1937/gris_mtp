from dataclasses import dataclass
from typing import Literal

ConvolutionType = Literal["APPNP", "S2GR"]
GNNConvType = Literal["sage", "gat", "gcn", "gin", "gated_graph"]


@dataclass(frozen=True)
class EncoderConfig:
    """Base for any embedding encoder that refines item representations.

    Subclasses carry only the knobs specific to the encoder family; every encoder
    is registered by name and applied as a pre-processing stage before the SID
    method is run.

    Args:
        name (str or None, optional): Registered name of the encoder, used to look
            up the concrete implementation. Defaults to None.
    """

    name: str | None = None

    def to_kwargs(self) -> dict:
        return {attr: value for attr, value in self.__dict__.items()}

    @classmethod
    def from_args(cls, args) -> "EncoderConfig":
        return cls(**{k: getattr(args, k, v) for k, v in cls().to_kwargs().items()})


@dataclass(frozen=True)
class GraphEncoderConfig(EncoderConfig):
    """Base for graph-informed propagation encoders (APPNP/S2GR).

    Only the propagation knobs live here; how the interaction graph is built and
    split (``graph_building``, ``skip_n_last``, ...) is a dataset concern handled
    by ``DatasetConfig`` (see ``gensid.config.dataset_config``).

    Args:
        name (ConvolutionType or None, optional): Registered name of the encoder.
            Defaults to None.
    """

    name: ConvolutionType | None = None


@dataclass(frozen=True)
class APPNPEncoderConfig(GraphEncoderConfig):
    """Offline APPNP graph-signal fusion.

    Restart-likelihood-based propagation (PageRank-style) applied once on the
    item embeddings, using the interaction graph built by the owning dataset.

    Args:
        name (Literal["APPNP"], optional): Registered name of the encoder, fixed to
            "APPNP". Defaults to "APPNP".

        teleport_prob (float, optional): Restart/teleport probability alpha of the
            APPNP propagation. Defaults to 0.1.

        n_propagations (int, optional): Number of propagation steps (K).
            Defaults to 10.

        decomposed_layers (int, optional): Number of layers used in the decomposed
            (low-memory) APPNP computation. Defaults to 16.

        add_self_loops (bool, optional): If True, add self loops when normalizing
            the adjacency matrix. Defaults to True.
    """

    name: Literal["APPNP"] = "APPNP"
    teleport_prob: float = 0.1
    n_propagations: int = 10
    decomposed_layers: int = 16
    add_self_loops: bool = True


@dataclass(frozen=True)
class S2GREncoderConfig(GraphEncoderConfig):
    """Offline S2GR graph-signal fusion.

    APPNP-style propagation with modified S2GR coefficients (see ``nn/gnn.S2GRConv``),
    applied once on the item embeddings.

    Args:
        name (Literal["S2GR"], optional): Registered name of the encoder, fixed to
            "S2GR". Defaults to "S2GR".

        teleport_prob (float, optional): Teleport probability alpha used by the
            S2GR coefficient expansion. Defaults to 0.1.

        n_propagations (int, optional): Number of propagation steps (K).
            Defaults to 10.

        add_self_loops (bool, optional): If True, add self loops when normalizing
            the adjacency matrix. Defaults to True.
    """

    name: Literal["S2GR"] = "S2GR"
    teleport_prob: float = 0.1
    n_propagations: int = 10
    add_self_loops: bool = True


@dataclass(frozen=True)
class GNNFusionEncoderConfig(EncoderConfig):
    """nn.GNNFusionEncoder, the graph neural encoder used by ``GRQVAE``.

    Args:
        name (Literal["GNNFusion"], optional): Registered name of the encoder,
            fixed to "GNNFusion". Defaults to "GNNFusion".

        convolution_type (GNNConvType, optional): Base convolutional layer:
            "sage", "gat", "gcn", "gin" or "gated_graph". Defaults to "gat".

        hidden_dim (int, optional): Hidden dimensionality of the graph conv layers.
            Defaults to 128.

        out_dim (int, optional): Output dimensionality of the refined item
            embeddings. Defaults to 64.

        layers (int, optional): Number of graph convolutional layers.
            Defaults to 2.

        heads (int, optional): Number of attention heads (GAT only).
            Defaults to 4.

        dropout (float, optional): Dropout probability applied after the conv
            layers. Defaults to 0.1.

        use_residual_fusion (bool, optional): If True, add a residual skip
            connection from the projected input embeddings. Defaults to True.

        act (str, optional): Activation function applied between conv layers.
            Defaults to "elu".

        gated_aggr (str, optional): Aggregation scheme of the gated graph conv
            ("add", "mean" or "max"). Defaults to "add".

        gin_eps (float, optional): Initial eps of the GIN conv. Defaults to 0.0.

        gin_train_eps (bool, optional): If True, learn the GIN eps during training.
            Defaults to False.
    """

    name: Literal["GNNFusion"] = "GNNFusion"
    convolution_type: GNNConvType = "gat"
    hidden_dim: int = 128
    out_dim: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    use_residual_fusion: bool = True
    act: str = "elu"
    gated_aggr: str = "add"
    gin_eps: float = 0.0
    gin_train_eps: bool = False


@dataclass(frozen=True)
class GraphSageEncoderConfig(EncoderConfig):
    """GraphSAGE + BPR item-representation encoder (mmgrec flow).

    Trained on ``(interactions x embeddings)``; the refined embeddings are then
    fed to the SID method (e.g. ``rqvae``).

    Args:
        name (Literal["GraphSage"], optional): Registered name of the encoder,
            fixed to "GraphSage". Defaults to "GraphSage".

        hidden_dim (int, optional): Hidden dimensionality of the GraphSAGE layers.
            Defaults to 128.

        out_dim (int, optional): Dimensionality of the refined item embeddings.
            Defaults to 64.

        dropout (float, optional): Dropout probability used while training the
            encoder. Defaults to 0.5.

        epochs (int, optional): Number of training epochs. Defaults to 100.

        batch_size (int, optional): Training batch size. Defaults to 3000.

        lr (float, optional): Learning rate for the BPR objective.
            Defaults to 0.001.

        weight_decay (float, optional): L2 weight decay for the optimizer.
            Defaults to 1e-06.
    """

    name: Literal["GraphSage"] = "GraphSage"
    hidden_dim: int = 128
    out_dim: int = 64
    dropout: float = 0.5
    epochs: int = 100
    batch_size: int = 3000
    lr: float = 1e-3
    weight_decay: float = 1e-6
