import pytest
import torch
from torch_geometric.data import Data

from gensid.config.model_config import GRQVAEModelConfig
from gensid.method.grqvae import GRQVAE


class TestGRQVAE:
    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        return GRQVAE(
            GRQVAEModelConfig(
                in_dim=8,
                gnn_hidden_dim=8,
                gnn_conv_type="sage",
                gnn_heads=2,
                gnn_dropout=0.0,
                gnn_layers=2,
                num_emb_list=(4, 4),
                e_dim=4,
                layers=(8,),
                sk_epsilons=(0.0, 0.0),
                beta=0.25,
                num_anchors=8,
                kmeans_init=False,
            )
        )

    @pytest.fixture
    def graph_data(self):
        num_nodes = 6
        edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 3, 4, 5]], dtype=torch.long)
        data = Data(x=torch.randn(num_nodes, 8), edge_index=edge_index)
        data.batch_size = 2
        return data

    def test_forward_replaces_emb_with_graph_encoder_output(self, model, graph_data):
        # GRQVAE's only unique logic: forward must overwrite data["emb"] with the
        # GNN-fused center embeddings before the inherited RQ-VAE step runs.
        data = {"emb": torch.randn(2, 8), "graph_data": graph_data}
        with torch.no_grad():
            expected = model.graph_encoder(graph_data)
        model(data, labels=None, use_sk=False)
        assert torch.allclose(data["emb"], expected)
