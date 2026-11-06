import torch
import torch.nn as nn
import torch.nn.functional as F

from gensid.config.model_config import StackedDMoNModelConfig
from gensid.data.utils import to_dense_adj
from gensid.nn.gnn import SparseGCNEncoder
from gensid.nn.layers import LocalAssignNet
from gensid.method.base import SIDMethod
from gensid.method.utils import dense_dmon_local_loss


class StackedDMoN(SIDMethod):
    """Stacked DMoN hierarchical clustering SID method.

    Learns a stack of DMoN modularity-assignment levels on top of a shared
    ``SparseGCNEncoder``: each level pools the (refined) nodes into ``k`` coarse
    clusters and records a soft assignment, so that a node's SID is the path of
    cluster indices from fine to coarse (``fine_to_coarse``) and back
    (``coarse_to_fine``).

    Unlike :class:`RecDMoN`, the levels are trained jointly (one forward pass per
    gradient step) and every node is refined before the next assignment, so the
    hierarchy is not built recursively.

    Args:
        config (StackedDMoNModelConfig): Hyperparameters of the method. See
            ``gensid.config.model_config.StackedDMoNModelConfig``.
    """

    name = "stacked_dmon"
    trainer = "stacked_dmon"

    def __init__(self, config: StackedDMoNModelConfig):
        """Store the hyperparameters and build the GCN encoder + assign/refine stacks."""
        super().__init__()
        self.cluster_sizes = config.num_emb_list
        self._codebook_sizes = self.cluster_sizes[::-1]
        self.in_dim = config.in_dim
        self.emb_dim = config.e_dim
        self.dmon_hidden_dim = config.dmon_hidden_dim
        self.assign_hidden_dim = config.assign_hidden_dim
        self.dropout = config.dropout

        self.balance_weight = config.dmon_balance_weight
        self.entropy_weight = config.dmon_entropy_weight

        self.encoder = SparseGCNEncoder(self.in_dim, self.dmon_hidden_dim, self.emb_dim, self.dropout)
        self.assigners = nn.ModuleList(
            [LocalAssignNet(self.emb_dim, self.assign_hidden_dim, k, self.dropout) for k in self.cluster_sizes]
        )
        self.refiners = nn.ModuleList(
            [LocalAssignNet(self.emb_dim, self.emb_dim, self.emb_dim, self.dropout) for _ in self.cluster_sizes]
        )
        # self.assigners = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             nn.Linear(embed_dim, assign_hidden_dim),
        #             nn.ReLU(),
        #             nn.Dropout(dropout),
        #             nn.Linear(assign_hidden_dim, k),
        #         )
        #         for k in self.cluster_sizes
        #     ]
        # )
        # self.refiners = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             nn.Linear(embed_dim, embed_dim),
        #             nn.ReLU(),
        #             nn.Dropout(dropout),
        #             nn.Linear(embed_dim, embed_dim),
        #         )
        #         for _ in self.cluster_sizes
        #     ]
        # )

    @staticmethod
    def dmon_pool(
        x: torch.Tensor, adj: torch.Tensor, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Average-pool nodes into clusters and coarsen the adjacency.

        Args:
            x (torch.Tensor): ``[N, dim]`` node features at this level.
            adj (torch.Tensor): ``[N, N]`` dense adjacency at this level.
            logits (torch.Tensor): ``[N, K]`` raw cluster-assignment logits.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: the pooled features
            ``[K, dim]`` (cluster-wise means), the coarsened adjacency
            ``[K, K]`` (symmetrized, diagonal zeroed) and the soft assignment
            matrix ``s = softmax(logits)`` ``[N, K]``.
        """
        s = F.softmax(logits, dim=-1)  # [N, K]
        cluster_mass = s.sum(dim=0).clamp_min(1e-12).unsqueeze(-1)  # [K, 1]
        x_next = (s.t() @ x) / cluster_mass
        adj_next = s.t() @ adj @ s
        adj_next = 0.5 * (adj_next + adj_next.t())
        adj_next.fill_diagonal_(0.0)
        return x_next, adj_next, s

    def forward(self, dataset: torch.utils.data.Dataset):
        """Run all DMoN levels on the full graph dataset.

        Encodes the nodes with the shared GCN, then for each level assigns,
        pools and refines, accumulating the per-level DMoN losses.

        Args:
            dataset (torch.utils.data.Dataset): Dataset exposing ``x_graph``,
                ``edge_index`` and ``edge_weight``.

        Returns:
            tuple[torch.Tensor, list[dict], torch.Tensor]: the GCN node
            embeddings ``z0`` ``[N, emb_dim]``, one dict per level (``level``,
            ``s_logits``, ``s_probs``, ``loss``, ``modularity``, ``balance``,
            ``entropy``) and the summed scalar loss over all levels.

        Raises:
            ValueError: If any level requests more clusters than there are
                (pooled) nodes at that point.
        """
        x = dataset.x_graph
        z0 = self.encoder(x, dataset.edge_index, dataset.edge_weight)
        x_curr = z0
        adj_curr = to_dense_adj(dataset.edge_index, x.size(0), dataset.edge_weight, x.device)

        levels = []
        total_loss = None
        for i, (assigner, refiner, k) in enumerate(zip(self.assigners, self.refiners, self.cluster_sizes), start=1):
            if k > x_curr.size(0):
                raise ValueError(f"Level {i}: k={k} > nodes={x_curr.size(0)}")

            logits = assigner(x_curr)
            degree = adj_curr.sum(dim=1)
            m2 = degree.sum().clamp_min(1e-12)
            level_loss, metrics = dense_dmon_local_loss(
                adj=adj_curr,
                logits=logits,
                degree=degree,
                m2=m2,
                balance_weight=self.balance_weight,
                entropy_weight=self.entropy_weight,
            )
            x_next, adj_next, s_probs = self.dmon_pool(x_curr, adj_curr, logits)
            x_next = refiner(x_next)

            if total_loss is None:
                total_loss = level_loss
            else:
                total_loss += level_loss
            levels.append(
                {
                    "level": i,
                    "s_logits": logits,
                    "s_probs": s_probs,
                    "loss": level_loss.item(),
                    "modularity": metrics["modularity"].item(),
                    "balance": metrics["balance"].item(),
                    "entropy": metrics["entropy"].item(),
                }
            )

            x_curr = x_next
            adj_curr = adj_next

        return z0, levels, total_loss

    @torch.no_grad()
    def generate(self, dataset: torch.utils.data.Dataset, batch_size: int | None = None):
        """Generate the coarse-to-fine SID matrix plus per-level artifacts.

        Args:
            dataset (torch.utils.data.Dataset): Dataset consumed by :meth:`forward`.
            batch_size (int or None, optional): Unused; kept for interface
                compatibility. Defaults to None.

        Returns:
            tuple[torch.Tensor, dict]: the ``[N, depth]`` coarse-to-fine index
            matrix (int64) and a dict bundling ``encoded_nodes`` and each
            level's ``level<i>_s_probs`` / ``level<i>_s_logits`` arrays.
        """
        encoded_nodes, levels, total_loss = self.forward(dataset)
        fine_to_coarse, coarse_to_fine = self.level_probs_to_f2c_cf2_indices(levels)
        artifacts = {"encoded_nodes": encoded_nodes.detach().cpu().numpy()}
        for i, lvl in enumerate(levels, start=1):
            artifacts[f"level{i}_s_probs"] = lvl["s_probs"].detach().cpu().numpy()
            artifacts[f"level{i}_s_logits"] = lvl["s_logits"].detach().cpu().numpy()
        return coarse_to_fine, artifacts

    @staticmethod
    def compose_assignments_to_indices(level_probs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Compose hierarchical soft assignments into per-level predicted indices.

        Projects node assignments through the chain ``P_0 @ P_1 @ ...``: the
        first entry is simply ``argmax(P_0)`` and each later entry is the
        argmax of the running product, giving one column per level (fine to
        coarse).

        Args:
            level_probs (list of torch.Tensor): Soft assignment matrices, one
                per level, ``[N_level, k_level]`` (level 0 is the finest).

        Returns:
            list[torch.Tensor]: per-level index vectors, fine to coarse.
        """
        projected = [level_probs[0].argmax(-1)]
        running = level_probs[0]
        for probs in level_probs[1:]:
            running = running @ probs
            projected.append(running.argmax(-1))
        return torch.stack(projected, dim=1)

    def level_probs_to_f2c_cf2_indices(self, levels: list[dict]):
        """Turn the per-level soft assignments into fine-to-coarse and coarse-to-fine indices.

        Args:
            levels (list of dict): Output of :meth:`forward`; each dict holds a
                ``s_probs`` entry ``[N_level, k_level]``, finest level first.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the ``[N, depth]`` fine-to-coarse
            and coarse-to-fine index matrices (the latter is the former flipped
            along the level axis).
        """
        level_probs = [lvl["s_probs"].detach().cpu() for lvl in levels]
        fine_to_coarse = self.compose_assignments_to_indices(level_probs)
        coarse_to_fine = fine_to_coarse.flip(dims=[1])

        return fine_to_coarse, coarse_to_fine
