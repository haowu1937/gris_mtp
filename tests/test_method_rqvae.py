import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gensid.config.model_config import RQVAEModelConfig
from gensid.method.rqvae import RQVAE


class TestRQVAE:
    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        return RQVAE(
            RQVAEModelConfig(
                in_dim=8,
                num_emb_list=(4, 4),
                e_dim=4,
                layers=(8,),
                dropout_prob=0.0,
                bn=False,
                sk_epsilons=(0.0, 0.0),
                beta=0.25,
                num_anchors=8,
                kmeans_init=False,
            )
        )

    @pytest.fixture
    def data(self):
        return {"emb": torch.randn(6, 8), "index": torch.arange(6)}

    def test_constructor_defaults(self):
        # num_emb_list/layers/sk_epsilons must be provided; everything else uses defaults.
        m = RQVAE(RQVAEModelConfig(num_emb_list=(4,), layers=(8,), sk_epsilons=(0.0,)))
        assert m.in_dim == 768
        assert m.e_dim == 32
        assert m.loss_type == "mse"
        assert m.alpha == 0.1
        assert m.beta == 0.1
        assert m.use_cf_loss is True
        assert m.num_anchors == 1024

    def test_codebook_properties(self, model):
        assert model.n_codebooks == 2
        assert model.codebook_sizes == [4, 4]

    def test_forward_outputs(self, model, data):
        out, rq_loss, indices, x_q, x_q_codebooks, x_res_list = model(data, labels=None, use_sk=False)
        assert out.shape == (6, 8)
        assert rq_loss.ndim == 0
        assert indices.shape == (6, 2)
        assert x_q.shape == (6, 4)
        assert x_q_codebooks.shape == (6, 2, 4)
        assert x_res_list.shape == (6, 2, 4)
        assert indices.min() >= 0 and indices.max() < 4
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize(
        "sk_epsilons",
        [
            pytest.param([0.0, 0.0], id="argmin"),
            pytest.param([0.01, 0.01], id="sinkhorn"),
        ],
    )
    def test_forward_with_sk(self, data, sk_epsilons):
        # sk_epsilons is set through the public constructor knob instead of mutating the
        # internal ResidualVectorQuantizer.
        model = RQVAE(
            RQVAEModelConfig(
                in_dim=8,
                num_emb_list=(4, 4),
                e_dim=4,
                layers=(8,),
                dropout_prob=0.0,
                bn=False,
                sk_epsilons=sk_epsilons,
                beta=0.25,
                num_anchors=8,
                kmeans_init=False,
            )
        )
        out, rq_loss, indices, *_ = model(data, labels=None, use_sk=True)
        assert out.shape == (6, 8)
        assert indices.shape == (6, 2)
        assert torch.isfinite(out).all()

    def test_generate(self, model, data):
        indices = model.generate(data, labels=None, use_sk=False)
        assert indices.shape == (6, 2)
        assert indices.dtype == torch.int64

    def test_generate_matches_manual_nearest_quantization(self, model, data):
        # generate() must reproduce the exact greedy residual-quantization: at each level,
        # pick the codebook entry closest to the residual, then subtract it.
        with torch.no_grad():
            z_e = model.encoder(data["emb"])
            residual = z_e
            expected = []
            for layer in model.rq.vq_layers:
                idx = torch.cdist(residual, layer.embedding.weight).argmin(dim=-1)
                expected.append(idx)
                residual = residual - layer.embedding.weight[idx]
            expected = torch.stack(expected, dim=-1)
        actual = model.generate(data, labels=None, use_sk=False)
        assert torch.equal(actual, expected)

    def test_forward_indices_match_generate(self, model, data):
        _, _, fwd_indices, *_ = model(data, labels=None, use_sk=False)
        assert torch.equal(fwd_indices, model.generate(data, labels=None, use_sk=False))

    def test_compute_loss_decomposition(self, model):
        # with no CF/graph/cluster inputs and alpha_graph_cluster=0:
        #   loss_recon = mse(out, xs); aux = 0; loss = loss_recon + quant_loss_weight * quant_loss
        torch.manual_seed(0)
        out = torch.randn(6, 8)
        xs = torch.randn(6, 8)
        quant_loss = torch.tensor(0.1)
        loss, aux, loss_recon, quant = model.compute_loss(
            out,
            quant_loss,
            torch.arange(6),
            torch.randn(6, 4),
            torch.randn(6, 2, 4),
            torch.randn(6, 2, 4),
            xs=xs,
        )
        assert torch.allclose(loss_recon, F.mse_loss(out, xs))
        assert float(aux) == 0.0
        assert torch.allclose(loss, loss_recon + model.quant_loss_weight * quant_loss)
        assert quant == quant_loss

    def test_vq_initialization(self, model, data):
        model.vq_initialization(data["emb"], use_sk=False)
        assert model.rq.vq_layers[0].initialized is True

    def test_cf_loss_diagonal_when_reps_match(self, model):
        rep = torch.randn(5, 4)
        loss = model.CF_loss(rep, rep)
        assert loss.ndim == 0
        assert torch.isfinite(loss).all()

    def test_graph_loss_positive_for_co_members(self, model):
        dense_out = torch.randn(5, 4)
        edge_w = torch.zeros(5, 5)
        edge_w[0, 1] = edge_w[1, 0] = 1.0
        loss = model.graph_loss(dense_out, edge_w, tau=0.1, num_anchors=5)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_graph_loss_no_positives_returns_nan_free(self, model):
        dense_out = torch.randn(4, 4)
        edge_w = torch.zeros(4, 4)
        loss = model.graph_loss(dense_out, edge_w, tau=0.1, num_anchors=4)
        assert loss.ndim == 0
        assert torch.isfinite(loss).all()

    def test_graph_cluster_loss_none_labels_returns_zero(self, model):
        dense_out = torch.randn(4, 4)
        assert model.graph_cluster_loss(dense_out, None) == 0.0

    def test_graph_cluster_loss_with_labels(self, model):
        dense_out = torch.randn(6, 4)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loss = model.graph_cluster_loss(dense_out, labels, tau=0.1, num_anchors=6)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_graph_cluster_loss_ignore_negative_labels(self, model):
        dense_out = torch.randn(4, 4)
        labels = torch.tensor([-1, -1, 0, 0])
        loss = model.graph_cluster_loss(dense_out, labels, tau=0.1, num_anchors=4)
        assert loss.ndim == 0
        assert torch.isfinite(loss).all()

    def test_hierarchical_graph_cluster_loss(self, model):
        codebooks = torch.randn(6, 2, 4)
        labels = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0], [1, 1]])
        loss = model.hierarchical_graph_cluster_loss(codebooks, labels, num_anchors=6)
        assert loss.ndim == 0
        assert torch.isfinite(loss).all()

    def test_hierarchical_loss_none_inputs(self, model):
        assert model.hierarchical_graph_cluster_loss(None, None) == 0.0

    def test_compute_loss_mse(self, model):
        torch.manual_seed(0)
        out = torch.randn(6, 8)
        quant_loss = torch.tensor(0.1, requires_grad=True)
        emb_idx = torch.arange(6)
        dense_out = torch.randn(6, 4)
        dense_out_codebooks = torch.randn(6, 2, 4)
        x_res_list = torch.randn(6, 2, 4)
        xs = torch.randn(6, 8)
        loss, aux_loss, loss_recon, quant = model.compute_loss(
            out, quant_loss, emb_idx, dense_out, dense_out_codebooks, x_res_list, xs=xs
        )
        assert loss.ndim == 0
        assert loss_recon.item() >= 0
        assert quant == quant_loss

    def test_compute_loss_l1(self, model):
        model.loss_type = "l1"
        out = torch.randn(4, 8)
        xs = torch.randn(4, 8)
        loss, *_ = model.compute_loss(
            out,
            torch.tensor(0.1),
            torch.arange(4),
            torch.randn(4, 4),
            torch.randn(4, 2, 4),
            torch.randn(4, 2, 4),
            xs=xs,
        )
        assert torch.isfinite(loss).all()

    def test_compute_loss_invalid_type_raises(self, model):
        model.loss_type = "bogus"
        with pytest.raises(ValueError, match="incompatible loss type"):
            model.compute_loss(
                torch.randn(4, 8),
                torch.tensor(0.1),
                torch.arange(4),
                torch.randn(4, 4),
                torch.randn(4, 2, 4),
                torch.randn(4, 2, 4),
                xs=torch.randn(4, 8),
            )

    @pytest.mark.parametrize(
        ("model_tweak", "extra_kwargs"),
        [
            pytest.param(
                lambda m: setattr(m, "cf_embedding", np.random.randn(10, 4).astype("float32")),
                {},
                id="cf-embedding",
            ),
            pytest.param(
                lambda m: setattr(m, "cf_loss_use_codebook_rep", True),
                {"edges_weights": torch.eye(4)},
                id="edges-with-codebook-rep",
            ),
            pytest.param(
                lambda m: None,
                # feature dim must match dense_out so CF_loss matmul is valid
                {"spectral_vectors": torch.randn(4, 4)},
                id="spectral-vectors",
            ),
            pytest.param(
                lambda m: setattr(m, "alpha_graph_cluster", 0.5),
                {"cluster_labels": torch.zeros(4, 2, dtype=torch.long)},
                id="graph-cluster",
            ),
            pytest.param(
                lambda m: setattr(m, "use_cf_loss", False),
                {},
                id="cf-loss-off",
            ),
        ],
    )
    def test_compute_loss_variants(self, model, model_tweak, extra_kwargs):
        torch.manual_seed(0)
        model_tweak(model)
        loss, *_ = model.compute_loss(
            torch.randn(4, 8),
            torch.tensor(0.1),
            torch.arange(4),
            torch.randn(4, 4),
            torch.randn(4, 2, 4),
            torch.randn(4, 2, 4),
            xs=torch.randn(4, 8),
            **extra_kwargs,
        )
        assert torch.isfinite(loss).all()

    def test_aggregate_codebook_representations(self, model):
        codebooks = torch.randn(4, 3, 4)
        out = model._aggregate_codebook_representations(codebooks)
        assert out.shape == (4, 4)
        assert torch.allclose(out, codebooks.sum(dim=1))

    def test_aggregate_codebook_none_raises(self, model):
        with pytest.raises(ValueError, match="dense_out_codebook"):
            model._aggregate_codebook_representations(None)

    def test_aggregate_codebook_last_m(self, model):
        model.cf_loss_last_m_codebooks = 2
        codebooks = torch.randn(4, 4, 3)
        out = model._aggregate_codebook_representations(codebooks)
        assert out.shape == (4, 3)
        assert torch.allclose(out, codebooks[:, 2:, :].sum(dim=1))

    def test_aggregate_codebook_last_m_invalid_raises(self, model):
        model.cf_loss_last_m_codebooks = -1
        with pytest.raises(ValueError, match=">= "):
            model._aggregate_codebook_representations(torch.randn(4, 4, 3))
