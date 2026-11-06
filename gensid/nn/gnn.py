import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import (
    GATConv,
    GatedGraphConv,
    GCNConv,
    GINConv,
    SAGEConv,
)
from torch_geometric.nn.conv import APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from gensid.config.encoder_config import APPNPEncoderConfig, S2GREncoderConfig
from gensid.nn.utils import activation_layer


class GNNFusionEncoder(nn.Module):
    """
    GNN encoder that fuses:
      1. semantic node representations
      2. graph neighborhood information from a sampled subgraph

    Supported conv_type:
      - "sage"
      - "gat"
      - "gcn"
      - "gin"
      - "gated_graph"

    Expected inputs:
        graph_data.x          FloatTensor [N_sub, D_sem]
        graph_data.edge_index LongTensor [2, E]
        graph_data.batch_size int

    Returns:
        fused_center: FloatTensor [B, out_dim]
    """

    SUPPORTED_CONVS = ("sage", "gat", "gcn", "gin", "gated_graph")

    def __init__(
        self,
        semantic_dim: int,
        hidden_dim: int,
        out_dim: int,
        conv_type: str = "gat",
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        use_residual_fusion: bool = True,
        act: str = "elu",
        gated_aggr: str = "add",
        gin_eps: float = 0.0,
        gin_train_eps: bool = False,
    ):
        super().__init__()

        conv_type = conv_type.lower()
        if conv_type not in self.SUPPORTED_CONVS:
            raise ValueError(f"Unsupported conv_type={conv_type}. Supported: {self.SUPPORTED_CONVS}")
        if layers < 1:
            raise ValueError("layers must be >= 1")

        self.conv_type = conv_type
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.use_residual_fusion = use_residual_fusion
        self.act_func = activation_layer(act)

        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if conv_type == "gated_graph":
            # GatedGraphConv already performs num_layers recurrent propagation internally.
            # It requires input_dim <= out_channels, so projecting to hidden_dim first is important.
            self.gated_graph_net = GatedGraphNet(hidden_dim, out_dim, layers, aggr=gated_aggr)
        else:
            dims = [hidden_dim]
            if layers == 1:
                dims.append(out_dim)
            else:
                dims.extend([hidden_dim] * (layers - 1))
                dims.append(out_dim)

            for layer_idx in range(layers):
                in_dim = dims[layer_idx]
                target_dim = dims[layer_idx + 1]
                is_last = layer_idx == layers - 1

                conv = self._build_conv_layer(
                    conv_type=conv_type,
                    in_dim=in_dim,
                    out_dim=target_dim,
                    is_last=is_last,
                    heads=heads,
                    gin_eps=gin_eps,
                    gin_train_eps=gin_train_eps,
                    dropout=dropout,
                )
                self.convs.append(conv)
                self.norms.append(nn.LayerNorm(target_dim))

        if use_residual_fusion:
            self.semantic_skip = nn.Linear(hidden_dim, out_dim)

    def _make_gin_mlp(self, in_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _build_conv_layer(
        self,
        conv_type: str,
        in_dim: int,
        out_dim: int,
        is_last: bool,
        heads: int,
        gin_eps: float,
        gin_train_eps: bool,
        dropout: float,
    ):
        # TODO: norm_dim was removed because it was always assigned to out_dim. Was it intentional?
        if conv_type == "sage":
            conv = SAGEConv(in_dim, out_dim)
        elif conv_type == "gcn":
            conv = GCNConv(in_dim, out_dim)
        elif conv_type == "gin":
            mlp = self._make_gin_mlp(in_dim, out_dim)
            conv = GINConv(mlp, eps=gin_eps, train_eps=gin_train_eps)
        elif conv_type == "gat":
            if is_last:
                gat_kws = {"out_channels": out_dim, "heads": 1, "concat": False}
            else:
                if out_dim % heads != 0:
                    raise ValueError(
                        f"For GAT hidden layers, out_dim ({out_dim}) must be divisible by heads ({heads}) "
                        f"when concat=True."
                    )
                gat_kws = {
                    "out_channels": out_dim // heads,
                    "heads": heads,
                    "concat": True,
                }

            conv = GATConv(in_channels=in_dim, dropout=dropout, **gat_kws)
        else:
            raise ValueError(f"Unsupported conv_type={conv_type}")

        return conv

    def forward(self, graph_data: torch_geometric.data.Data) -> torch.Tensor:
        semantic_x = graph_data.x
        edge_index = graph_data.edge_index

        x0 = self.semantic_proj(semantic_x)

        if self.conv_type == "gated_graph":
            x = self.gated_graph_net(x0, edge_index)
        else:
            x = x0
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = self.norms[i](x)

                is_last = i == len(self.convs) - 1
                if not is_last:
                    x = self.act_func(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)

        if self.use_residual_fusion:
            x = x + self.semantic_skip(x0)

        x = F.dropout(x, p=self.dropout, training=self.training)

        return x[: graph_data.batch_size]


class GatedGraphNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        layers: int,
        aggr: str = "add",
        bias: bool = True,
        **kwargs,
    ):
        super().__init__()

        gated_out_dim = out_dim if layers == 1 else hidden_dim
        self.gated_graph_conv = GatedGraphConv(
            out_channels=gated_out_dim,
            num_layers=layers,
            aggr=aggr,
            bias=bias,
            **kwargs,
        )
        self.post_gated_proj = None
        if gated_out_dim != out_dim:
            self.post_gated_proj = nn.Linear(gated_out_dim, out_dim)

        self.final_norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        x = self.gated_graph_conv(x, edge_index)
        if self.post_gated_proj is not None:
            x = self.post_gated_proj(x)
        return self.final_norm(x)


# TODO: in RecDMoN replace the SparseGCNEncoder with the GNNFusionEncoder above (main difference is last dropout)
class SparseGCNEncoder(nn.Module):
    """Two-layer GCN encoder with layer-norm and ReLU between the layers.

    Maps sparse-graph node features to a refined embedding space; used as the
    shared encoder of :class:`StackedDMoN`.

    Args:
        in_dim (int): Input feature dimensionality.
        hidden_dim (int): Hidden GCN width.
        out_dim (int): Output embedding dimensionality.
        dropout (float): Dropout probability after the first ReLU.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = float(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.norm2(x)
        return x


class APPNPConv(APPNP):
    def __init__(self, config: APPNPEncoderConfig):
        super().__init__(
            config.n_propagations, config.teleport_prob, cached=False, decomposed_layers=config.decomposed_layers
        )


class S2GRConv(MessagePassing):
    """
    APPNP-style propagation with modified coefficients (no caching).

    F^(K) = sum_{t=0}^K c_t^(K) * A_hat^t X

    c_t^(K) = alpha * (1-alpha)^(2t) * [2 - alpha - (1-alpha)^(K-t+1)]
    """

    def __init__(self, config: S2GREncoderConfig):
        super().__init__(aggr="add")
        self.K = config.n_propagations
        self.alpha = config.teleport_prob
        self.add_self_loops = config.add_self_loops

    def _get_coeffs(self, device, dtype) -> torch.Tensor:
        t = torch.arange(self.K + 1, device=device, dtype=dtype)
        return self.alpha * (1.0 - self.alpha) ** (2 * t) * (2.0 - self.alpha - (1.0 - self.alpha) ** (self.K - t + 1))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Normalize adjacency ONCE per forward
        edge_index, edge_weight = gcn_norm(
            edge_index,
            edge_weight,
            num_nodes=x.size(0),
            add_self_loops=self.add_self_loops,
            dtype=x.dtype,
        )

        coeffs = self._get_coeffs(device=x.device, dtype=x.dtype)

        # U^(0)
        u = x
        out = coeffs[0] * u

        # iterative propagation
        for t in range(1, self.K + 1):
            u = self.propagate(edge_index, x=u, edge_weight=edge_weight)
            out = out + coeffs[t] * u

        return out

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(K={self.K}, alpha={self.alpha})"
