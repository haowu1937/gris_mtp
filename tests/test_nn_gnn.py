import pytest
import torch
from torch_geometric.data import Data

from gensid.config.encoder_config import APPNPEncoderConfig, S2GREncoderConfig
from gensid.nn import get_encoder
from gensid.nn.gnn import APPNPConv, GNNFusionEncoder, GatedGraphNet, S2GRConv


class TestGetEncoder:
    def test_none_returns_identity(self):
        x = torch.randn(3, 4)
        out = get_encoder(None)(x)
        assert torch.equal(out, x)

    def test_unknown_encoder_raises(self):
        with pytest.raises(KeyError):
            get_encoder("bogus")

    @pytest.mark.parametrize(
        ("encoder_name", "config"),
        [
            ("appnp", APPNPEncoderConfig(n_propagations=3, teleport_prob=0.1, decomposed_layers=4)),
            ("APPNP", APPNPEncoderConfig(n_propagations=3, teleport_prob=0.1, decomposed_layers=4)),
            ("s2gr", S2GREncoderConfig(n_propagations=2, teleport_prob=0.1)),
        ],
    )
    def test_resolves_named_encoder_to_usable_component(self, encoder_name, config):
        # Contract the future Registry must keep: a name resolves to a component that,
        # given its typed config, produces a valid output on a graph. The test asserts
        # behavior rather than pinning the exact class identity returned by get_encoder.
        encoder = get_encoder(encoder_name)
        x = torch.randn(5, 4)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        out = encoder(config)(x, edge_index)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


class TestAPPNPConv:
    def test_forward_matches_manual_appnp_propagation(self):
        # 2-node chain; with self-loops the GCN-normalized adjacency is A_hat = [[0.5, 0.5], [0.5, 0.5]].
        # APPNP iterates H^{k} = (1-alpha) * A_hat H^{k-1} + alpha * X, K=1, alpha=0.5:
        #   node0: 0.5 * 0.5 * (2 + 6) + 0.5 * 2 = 3.0
        #   node1: 0.5 * 0.5 * (2 + 6) + 0.5 * 6 = 5.0
        config = APPNPEncoderConfig(n_propagations=1, teleport_prob=0.5, decomposed_layers=1)
        conv = APPNPConv(config)
        x = torch.tensor([[2.0], [6.0]])
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        assert torch.allclose(conv(x, edge_index), torch.tensor([[3.0], [5.0]]))


class TestS2GRConv:
    def test_forward_isolated_nodes_return_scaled_inputs(self):
        # Self-loops only -> A_hat = I, so F^K = (sum_t c_t) * X exactly.
        config = S2GREncoderConfig(n_propagations=2, teleport_prob=0.1, add_self_loops=False)
        conv = S2GRConv(config)
        x = torch.tensor([[1.0], [2.0], [3.0]])
        edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
        coeffs = conv._get_coeffs(device="cpu", dtype=torch.float32)
        assert torch.allclose(conv(x, edge_index), x * coeffs.sum())

    def test_repr(self):
        conv = S2GRConv(S2GREncoderConfig(n_propagations=4, teleport_prob=0.2))
        assert repr(conv) == "S2GRConv(K=4, alpha=0.2)"

    def test_coefficients_length(self):
        conv = S2GRConv(S2GREncoderConfig(n_propagations=3, teleport_prob=0.1))
        coeffs = conv._get_coeffs(device="cpu", dtype=torch.float32)
        assert coeffs.shape == (4,)
        assert (coeffs >= 0).all()
        assert coeffs[0] == pytest.approx(0.1 * (2 - 0.1 - (0.9) ** 4))


class TestGNNFusionEncoder:
    @pytest.fixture
    def small_graph(self):
        num_nodes = 6
        edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 3, 4, 5]], dtype=torch.long)
        graph_data = Data(x=torch.randn(num_nodes, 4), edge_index=edge_index)
        graph_data.batch_size = 2
        return graph_data

    @pytest.mark.parametrize("conv_type", ["sage", "gat", "gcn", "gin", "gated_graph"])
    def test_message_passing_uses_neighbor_features(self, conv_type):
        # Two identical center nodes; node 1 additionally aggregates a distinct neighbor (node 2).
        # Real message passing must make node 1's fused representation differ from node 0's.
        torch.manual_seed(0)
        enc = GNNFusionEncoder(
            semantic_dim=4,
            hidden_dim=8,
            out_dim=8,
            conv_type=conv_type,
            layers=2,
            heads=2,
            dropout=0.0,
        )
        center = torch.tensor([1.0, -1.0, 1.0, -1.0])
        graph_data = Data(
            x=torch.stack([center, center, torch.tensor([5.0, 5.0, 5.0, 5.0])]),
            edge_index=torch.tensor([[1, 2], [2, 1]], dtype=torch.long),
        )
        graph_data.batch_size = 2
        out = enc(graph_data)
        assert not torch.allclose(out[0], out[1], atol=1e-4)

    def test_residual_fusion_off(self, small_graph):
        enc = GNNFusionEncoder(
            semantic_dim=4,
            hidden_dim=8,
            out_dim=8,
            conv_type="sage",
            layers=1,
            use_residual_fusion=False,
        )
        assert not hasattr(enc, "semantic_skip")
        out = enc(small_graph)
        assert out.shape == (2, 8)

    def test_unsupported_conv_raises(self):
        with pytest.raises(ValueError, match="Unsupported conv_type"):
            GNNFusionEncoder(semantic_dim=4, hidden_dim=8, out_dim=8, conv_type="bogus")

    def test_layers_zero_raises(self):
        with pytest.raises(ValueError, match="layers"):
            GNNFusionEncoder(semantic_dim=4, hidden_dim=8, out_dim=8, conv_type="sage", layers=0)

    def test_gat_requires_divisible_out_dim(self):
        # layers=3 -> the middle (non-last) GAT layer outputs hidden_dim=7 with concat=True,
        # which must be divisible by heads=2 -> raise.
        with pytest.raises(ValueError, match="divisible by heads"):
            GNNFusionEncoder(
                semantic_dim=4,
                hidden_dim=7,
                out_dim=8,
                conv_type="gat",
                layers=3,
                heads=2,
            )

    def test_gin_train_eps_flag(self):
        enc = GNNFusionEncoder(
            semantic_dim=4,
            hidden_dim=8,
            out_dim=8,
            conv_type="gin",
            layers=2,
            gin_eps=0.2,
            gin_train_eps=True,
        )
        assert isinstance(enc.convs[0].eps, torch.nn.Parameter)
        assert enc.convs[0].eps.requires_grad

        enc_fixed = GNNFusionEncoder(
            semantic_dim=4,
            hidden_dim=8,
            out_dim=8,
            conv_type="gin",
            layers=2,
            gin_eps=0.2,
            gin_train_eps=False,
        )
        assert not isinstance(enc_fixed.convs[0].eps, torch.nn.Parameter)


class TestGatedGraphNet:
    def test_forward_multi_layer(self):
        # layers > 1 with hidden_dim != out_dim -> hidden-dim recurrent conv plus a
        # final projection back to out_dim
        net = GatedGraphNet(hidden_dim=8, out_dim=16, layers=2)
        assert net.post_gated_proj is not None
        torch.manual_seed(0)
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = net(x, edge_index)
        assert out.shape == (5, 16)

    def test_single_layer_no_post_proj(self):
        net = GatedGraphNet(hidden_dim=8, out_dim=8, layers=1)
        assert net.post_gated_proj is None
