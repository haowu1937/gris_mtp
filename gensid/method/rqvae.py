import torch
from torch.nn import functional as F

from gensid.config.docstring import auto_docstring
from gensid.config.model_config import RQVAEModelConfig
from gensid.method.base import SIDMethod
from gensid.nn.layers import MLPLayers, ResidualVectorQuantizer


@auto_docstring
class RQVAE(SIDMethod):
    """Residual-quantized VAE (RQ-VAE) SID method.

    Encodes the item embeddings with an MLP, quantizes them with a stack of
    residual vector-quantization layers (one codebook per level) and decodes back
    to the input space. Optionally trains auxiliary contrastive (CF) and graph
    cluster losses alongside the reconstruction.

    Args:
        config (RQVAEModelConfig): Hyperparameters of the model. See
            ``gensid.config.model_config.RQVAEModelConfig``.
    """

    name = "rqvae"
    trainer = "rqvae"

    def __init__(self, config: RQVAEModelConfig):
        super().__init__()

        self.config = config
        self.in_dim = config.in_dim
        self._codebook_sizes = list(config.num_emb_list)
        self.e_dim = config.e_dim
        self.layers = list(config.layers)
        self.dropout_prob = config.dropout_prob
        self.bn = config.bn
        self.loss_type = config.loss_type
        self.quant_loss_weight = config.quant_loss_weight
        self.kmeans_init = config.kmeans_init
        self.kmeans_iters = config.kmeans_iters
        self.sk_epsilons = config.sk_epsilons
        self.sk_iters = config.sk_iters
        self.cf_embedding = config.cf_embedding
        self.alpha = config.alpha
        self.alpha_graph_cluster = config.alpha_graph_cluster
        self.graph_cluster_tau = config.graph_cluster_tau
        self.beta = config.beta
        self.n_clusters = config.n_clusters
        self.sample_strategy = config.sample_strategy
        self.use_cf_loss = config.use_cf_loss
        self.cf_loss_use_codebook_rep = config.cf_loss_use_codebook_rep
        self.cf_loss_last_m_codebooks = config.cf_loss_last_m_codebooks

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims, dropout=self.dropout_prob, bn=self.bn)

        self.rq = ResidualVectorQuantizer(
            self._codebook_sizes,
            self.e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
        )

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims, dropout=self.dropout_prob, bn=self.bn)
        self.num_anchors = config.num_anchors
        self.lognorm_weights = config.lognorm_weights

    def forward(self, data, labels, use_sk=True):
        x = data["emb"]
        x = self.encoder(x)
        x_q, rq_loss, indices, x_q_codebooks, x_res_list = self.rq(x, labels, use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices, x_q, x_q_codebooks, x_res_list

    def CF_loss(self, quantized_rep, encoded_rep):
        batch_size = quantized_rep.size(0)
        labels = torch.arange(batch_size, dtype=torch.long, device=quantized_rep.device)
        similarities = torch.matmul(quantized_rep, encoded_rep.transpose(0, 1))
        cf_loss = F.cross_entropy(similarities, labels)
        return cf_loss

    def graph_loss(self, dense_out, edge_w, tau=0.1, num_anchors=1024):
        z = F.normalize(dense_out, dim=-1)
        edge_w = edge_w[:, :num_anchors]
        sim = z @ z[:num_anchors].t() / tau

        sim.fill_diagonal_(-1e9)  # mask self
        logp = F.log_softmax(sim, dim=1)

        w = edge_w.clone().to(dense_out.device)
        w.fill_diagonal_(0.0)
        # target = w / (w.sum(dim=1, keepdim=True) + 1e-12)

        mask = w.sum(dim=1) > 0  # only rows with positives
        if not mask.any():
            return dense_out.new_tensor(0.0)
        if self.lognorm_weights:
            w = w.log1p()
        # NOTE: number of elements in average might vary across batches, fix it
        return -(w * logp).sum(dim=1)[mask].mean()

    def graph_cluster_loss(self, dense_out, cluster_labels, tau=None, num_anchors=1024):
        if cluster_labels is None:
            return dense_out.new_tensor(0.0)

        tau = self.graph_cluster_tau if tau is None else tau
        labels = cluster_labels.to(dense_out.device, non_blocking=True).view(-1)
        valid_label_mask = labels >= 0
        if valid_label_mask.sum() <= 1:
            return dense_out.new_tensor(0.0)

        z = F.normalize(dense_out[valid_label_mask], dim=-1)
        labels = labels[valid_label_mask]
        # NOTE: B^2
        sim = z @ z[:num_anchors].t() / tau
        sim.fill_diagonal_(1e-9)

        pos_mask = labels[:, None] == labels[None, :num_anchors]
        pos_mask.fill_diagonal_(False)
        pos_count = pos_mask.sum(dim=1)
        valid_anchor_mask = pos_count > 0
        if not valid_anchor_mask.any():
            return dense_out.new_tensor(0.0)

        logp = F.log_softmax(sim, dim=1)
        per_anchor = -(logp * pos_mask).sum(dim=1) / pos_count.clamp_min(1)
        return per_anchor[valid_anchor_mask].mean()

    def hierarchical_graph_cluster_loss(self, dense_out_codebooks, cluster_labels, num_anchors=1024):
        if cluster_labels is None or dense_out_codebooks is None:
            return dense_out_codebooks.new_tensor(0.0) if dense_out_codebooks is not None else torch.tensor(0.0)
        num_levels = cluster_labels.size(-1)
        if num_levels == 0:
            return dense_out_codebooks.new_tensor(0.0)

        losses = []
        for level_idx in range(num_levels):
            losses.append(
                self.graph_cluster_loss(
                    dense_out_codebooks[:, : level_idx + 1, :].sum(dim=1),
                    cluster_labels[:, level_idx],
                    num_anchors=num_anchors,
                )
            )

        if not losses:
            return dense_out_codebooks.new_tensor(0.0)
        return torch.stack(losses).sum()

    def vq_initialization(self, x, use_sk=True):
        self.rq.vq_initialization(self.encoder(x), use_sk)

    @torch.no_grad()
    def generate(self, data, labels, use_sk=False):
        x_e = self.encoder(data["emb"])
        _, _, indices, _, _ = self.rq(x_e, labels, use_sk=use_sk)
        return indices

    def _aggregate_codebook_representations(self, dense_out_codebook):
        if dense_out_codebook is None:
            raise ValueError("dense_out_codebook is required when CF loss is based on codebooks")
        start_idx = 0
        if self.cf_loss_last_m_codebooks is not None:
            m = min(self.cf_loss_last_m_codebooks, dense_out_codebook.shape[1])
            if m < 0:
                raise ValueError("cf_loss_last_m_codebooks should be >= 0")
            start_idx = dense_out_codebook.shape[1] - m
        return dense_out_codebook[:, start_idx:, :].sum(dim=1)

    def compute_loss(
        self,
        out,
        quant_loss,
        emb_idx,
        dense_out,
        dense_out_codebooks,
        x_res_list,
        xs=None,
        edges_weights=None,
        spectral_vectors=None,
        cluster_labels=None,
    ):

        if self.loss_type == "mse":
            loss = F.mse_loss(out, xs, reduction="mean")
        elif self.loss_type == "l1":
            loss = F.l1_loss(out, xs, reduction="mean")
        else:
            raise ValueError("incompatible loss type")
        loss_recon = loss.detach().clone()

        loss += self.quant_loss_weight * quant_loss

        # CF_Loss (optional)
        cf_input_codebooks = None
        if self.cf_loss_use_codebook_rep or self.cf_loss_last_m_codebooks is not None:
            cf_input_codebooks = self._aggregate_codebook_representations(dense_out_codebooks)

        cf_loss = 0.0
        cf_codebook_loss = 0.0
        if any(add_data is not None for add_data in [self.cf_embedding, edges_weights, spectral_vectors]):
            if self.cf_embedding is not None:
                cf_embedding_in_batch = torch.from_numpy(self.cf_embedding[emb_idx]).to(dense_out.device)
                cf_loss_fn, cf_loss_args = self.CF_loss, {"encoded_rep": cf_embedding_in_batch}
            elif edges_weights is not None:
                cf_loss_fn, cf_loss_args = self.graph_loss, {"edge_w": edges_weights, "num_anchors": self.num_anchors}
            elif spectral_vectors is not None:
                cf_loss_fn, cf_loss_args = self.CF_loss, {"encoded_rep": spectral_vectors}

            cf_loss += cf_loss_fn(dense_out, **cf_loss_args)
            if cf_input_codebooks is not None:
                cf_codebook_loss += cf_loss_fn(cf_input_codebooks, **cf_loss_args)

        graph_cluster_loss = dense_out.new_tensor(0.0)
        if self.alpha_graph_cluster > 0.0:
            graph_cluster_loss_res = self.hierarchical_graph_cluster_loss(
                x_res_list, cluster_labels, num_anchors=self.num_anchors
            )
            graph_cluster_loss_codes = self.hierarchical_graph_cluster_loss(
                dense_out_codebooks, cluster_labels, num_anchors=self.num_anchors
            )
            graph_cluster_loss = graph_cluster_loss_codes + graph_cluster_loss_res
            loss += self.alpha_graph_cluster * graph_cluster_loss

        if self.use_cf_loss:
            loss += self.alpha * (cf_loss + cf_codebook_loss)

        aux_loss = cf_loss + cf_codebook_loss + graph_cluster_loss
        return loss, aux_loss, loss_recon, quant_loss
