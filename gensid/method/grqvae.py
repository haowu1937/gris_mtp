from gensid.config.docstring import auto_docstring
from gensid.config.model_config import GRQVAEModelConfig
from gensid.method.rqvae import RQVAE
from gensid.nn.gnn import GNNFusionEncoder


@auto_docstring
class GRQVAE(RQVAE):
    """Graph-informed RQ-VAE SID method.

    Prepends a ``GNNFusionEncoder`` to the RQ-VAE: item embeddings are produced by
    the graph encoder from sampled subgraphs, then encoded/quantized/decoded by the
    VAE.

    Args:
        config (GRQVAEModelConfig): Hyperparameters of the model. See
            ``gensid.config.model_config.GRQVAEModelConfig``.
    """

    name = "grqvae"

    def __init__(self, config: GRQVAEModelConfig):
        super().__init__(config)

        hidden_dim = config.gnn_hidden_dim if config.gnn_hidden_dim is not None else config.in_dim
        self.graph_encoder = GNNFusionEncoder(
            conv_type=config.gnn_conv_type,
            semantic_dim=config.in_dim,
            hidden_dim=hidden_dim,
            out_dim=config.in_dim,
            heads=config.gnn_heads,
            dropout=config.gnn_dropout,
            layers=config.gnn_layers,
        )

    def forward(self, data, labels, use_sk=True):
        # TODO: Data is not used? In this way, this only encodes the graph ignoring the semantic embeddings
        data["emb"] = self.graph_encoder(data["graph_data"])
        # TODO: the graph_encoder should be pre-trained, otherwise the training would be instable, even if target fixed
        # TODO: should also target be assigned to the new fused graph data? RQ-VAE with APPNP doesn't do that
        # data["target"] = data["emb"]
        return super().forward(data, labels, use_sk=use_sk)
