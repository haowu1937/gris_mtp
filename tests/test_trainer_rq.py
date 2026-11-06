import json

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data

from gensid.config.dataset_config import PlainDatasetConfig
from gensid.config.model_config import RQVAEModelConfig
from gensid.config.trainer_config import RQTrainerConfig
from gensid.data.dataset import EmbDataset
from gensid.method.rqvae import RQVAE
from gensid.trainer import rq_trainer as rq_trainer_mod
from gensid.trainer.rq_trainer import RQTrainer


def make_trainer(tmp_path):
    config = RQTrainerConfig(
        lr=0.01,
        learner="adamw",
        weight_decay=0.0,
        epochs=1,
        eval_step=1,
        compile=False,
        disable_sk=False,
        disable_vq_init=True,
        max_cluster_level=None,
        num_workers=0,
        max_sk_fix_collision_iters=5,
        save_json=True,
    )
    return RQTrainer(
        config,
        make_model(),
        device="cpu",
        ckpt_dir=str(tmp_path),
        saved_model_dir="test",
    )


def make_model():
    torch.manual_seed(0)
    return RQVAE(
        RQVAEModelConfig(
            in_dim=4,
            num_emb_list=(2,),
            e_dim=2,
            layers=(4,),
            dropout_prob=0.0,
            bn=False,
            sk_epsilons=(0.0,),
            beta=0.25,
            num_anchors=4,
            kmeans_init=False,
        )
    )


class TestRQTrainerPrepareBatch:
    def test_plain_batch_train_keeps_all_tensors(self, tmp_path):
        tr = make_trainer(tmp_path)
        batch = {"emb": torch.randn(3, 4), "index": torch.arange(3), "target": torch.randn(3, 4)}
        out = tr._prepare_batch(batch, train=True)
        assert torch.equal(out["emb"], batch["emb"])
        assert torch.equal(out["index"], batch["index"])
        assert torch.equal(out["target"], batch["target"])
        assert out["emb"].device == tr.device

    def test_plain_batch_val_keeps_keys(self, tmp_path):
        tr = make_trainer(tmp_path)
        batch = {"emb": torch.randn(3, 4), "index": torch.arange(3), "target": torch.randn(3, 4)}
        out = tr._prepare_batch(batch, train=False)
        assert torch.equal(out["emb"], batch["emb"])
        assert "index" in out and "target" in out

    def test_graph_batch_train_slices_seed_nodes(self, tmp_path):
        tr = make_trainer(tmp_path)
        gdata = Data(
            x=torch.randn(10, 4),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
            n_id=torch.arange(10),
            batch_size=6,
        )
        out = tr._prepare_batch(gdata, train=True)
        assert torch.equal(out["emb"], gdata.x[:6])
        assert torch.equal(out["index"], gdata.n_id[:6])
        assert torch.equal(out["target"], gdata.x[:6])
        assert out["graph_data"] is gdata

    def test_graph_batch_val_omits_target(self, tmp_path):
        tr = make_trainer(tmp_path)
        gdata = Data(
            x=torch.randn(10, 4),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
            n_id=torch.arange(10),
            batch_size=6,
        )
        out = tr._prepare_batch(gdata, train=False)
        assert torch.equal(out["emb"], gdata.x[:6])
        assert "index" not in out and "target" not in out
        assert out["graph_data"] is gdata


class TestRQTrainerGenerateIndices:
    """SID generation: single nearest-neighbour pass by default (GrIS_v1 parity);
    the Sinkhorn second pass runs only when ``fix_collisions`` is set."""

    @staticmethod
    def _make_model():
        torch.manual_seed(0)
        return RQVAE(
            RQVAEModelConfig(
                in_dim=2,
                num_emb_list=(4,),
                e_dim=2,
                layers=(2,),
                dropout_prob=0.0,
                bn=False,
                sk_epsilons=(0.0,),
                beta=0.25,
                num_anchors=4,
                kmeans_init=False,
            )
        )

    def _make_trainer(self, tmp_path, save_json=True, fix_collisions=False):
        config = RQTrainerConfig(
            lr=0.01,
            learner="adamw",
            weight_decay=0.0,
            epochs=1,
            eval_step=1,
            compile=False,
            disable_sk=False,
            disable_vq_init=True,
            max_cluster_level=None,
            num_workers=0,
            max_sk_fix_collision_iters=5,
            save_json=save_json,
            fix_collisions=fix_collisions,
        )
        return RQTrainer(config, self._make_model(), device="cpu", ckpt_dir=str(tmp_path), saved_model_dir="test")

    @staticmethod
    def _make_dataset(tmp_path):
        emb_path = tmp_path / "emb.npy"
        np.save(emb_path, np.random.default_rng(0).standard_normal((4, 2)).astype("float32"))
        return EmbDataset(PlainDatasetConfig(), str(emb_path), interactions_path=None, eval_mode=True)

    def _run(self, tmp_path, monkeypatch, generate_values, fix_collisions=False):
        """Run generate_indices with custom_constrained_km faked; return (trainer, calls, output_file)."""
        tr = self._make_trainer(tmp_path, fix_collisions=fix_collisions)
        monkeypatch.setattr(
            rq_trainer_mod, "custom_constrained_km", lambda emb, **kws: (None, np.zeros(emb.shape[0], dtype=int))
        )

        calls = {"use_sk": []}
        it = iter(generate_values)

        def fake_generate(batch, labels, use_sk):
            calls["use_sk"].append(use_sk)
            n = int(batch["emb"].shape[0])
            return next(it)(batch, n)

        monkeypatch.setattr(tr.model, "generate", fake_generate)

        dataset = self._make_dataset(tmp_path)
        loader = [{"emb": torch.zeros(2, 2)}, {"emb": torch.zeros(2, 2)}]
        output_file = str(tmp_path / "index.json")
        tr.generate_indices(dataset, loader, output_file)
        return tr, calls, output_file

    def test_single_pass_default_matches_gris_v1(self, tmp_path, monkeypatch):
        # Default (fix_collisions=False) is a single nearest-neighbour pass like
        # GrIS_v1: colliding SIDs are left unresolved and no Sinkhorn runs.
        def unif(value):
            def gen(batch, n):
                return torch.full((n, 1), value, dtype=torch.int64)

            return gen

        tr, calls, output_file = self._run(tmp_path, monkeypatch, [unif(0), unif(0)])
        assert calls["use_sk"] == [False, False]
        # sk_epsilon is never bumped (no fix pass) -> stays at the configured 0.0
        assert tr.model.rq.vq_layers[-1].sk_epsilon == 0.0
        with open(output_file) as f:
            mapping = json.load(f)
        assert mapping == {"0": [0], "1": [0], "2": [0], "3": [0]}

    def test_two_pass_resolves_collisions_in_place(self, tmp_path, monkeypatch):
        # pass 1 (use_sk=False) yields a colliding pair per batch; the fix pass
        # (use_sk=True) rewrites rows by global item id -> all unique.
        def unif(value):
            def gen(batch, n):
                return torch.full((n, 1), value, dtype=torch.int64)

            return gen

        def by_id(batch, n):
            ids = batch["index"]
            return torch.as_tensor(ids, dtype=torch.int64).view(-1, 1) % 4

        tr, calls, output_file = self._run(tmp_path, monkeypatch, [unif(0), unif(1), by_id, by_id], fix_collisions=True)
        assert calls["use_sk"] == [False, False, True, True]
        with open(output_file) as f:
            mapping = json.load(f)
        assert mapping == {"0": [0], "1": [1], "2": [2], "3": [3]}
        # the single VQ layer had epsilon 0.0 -> bumped for the Sinkhorn fix pass
        assert tr.model.rq.vq_layers[-1].sk_epsilon == 0.003

    def test_no_collisions_skips_sinkhorn_fix(self, tmp_path, monkeypatch):
        def arange(offset):
            def gen(batch, n):
                return torch.arange(offset, offset + n, dtype=torch.int64).view(-1, 1)

            return gen

        _, calls, output_file = self._run(tmp_path, monkeypatch, [arange(0), arange(2)], fix_collisions=True)
        # first pass already collision-free -> fix loop breaks immediately
        assert calls["use_sk"] == [False, False]
        with open(output_file) as f:
            mapping = json.load(f)
        assert mapping == {"0": [0], "1": [1], "2": [2], "3": [3]}

    def test_unresolvable_collisions_capped_by_max_iters(self, tmp_path, monkeypatch):
        def unif(value):
            def gen(batch, n):
                return torch.full((n, 1), value, dtype=torch.int64)

            return gen

        # never fixes: pass1 both batches collide into one group, fix pass keeps them equal
        values = [unif(0), unif(0)] + [unif(0)] * 5
        _, calls, output_file = self._run(tmp_path, monkeypatch, values, fix_collisions=True)
        # 2 pass-1 calls + 1 group fixed per iteration for 5 iterations
        assert calls["use_sk"] == [False, False] + [True] * 5
        with open(output_file) as f:
            mapping = json.load(f)
        assert mapping == {"0": [0], "1": [0], "2": [0], "3": [0]}

    def test_parquet_default_output_plus_one_shift(self, tmp_path, monkeypatch):
        # default (save_json=False) writes the +1-shifted int codes as parquet;
        # with an .npy embedding source item_id/raw_id fall back to identity.
        tr = self._make_trainer(tmp_path, save_json=False)
        monkeypatch.setattr(
            rq_trainer_mod, "custom_constrained_km", lambda emb, **kws: (None, np.zeros(emb.shape[0], dtype=int))
        )

        def fake_generate(batch, labels, use_sk):
            n = int(batch["emb"].shape[0])
            return torch.arange(n, dtype=torch.int64).view(-1, 1)

        monkeypatch.setattr(tr.model, "generate", fake_generate)
        dataset = self._make_dataset(tmp_path)
        output_file = str(tmp_path / "index.parquet")
        tr.generate_indices(dataset, [{"emb": torch.zeros(4, 2)}], output_file)

        df = pl.read_parquet(output_file)
        assert df.columns == ["item_id", "normalized_id", "semantic_id", "raw_id"]
        assert df["normalized_id"].to_list() == [0, 1, 2, 3]
        assert df["item_id"].to_list() == [0, 1, 2, 3]
        assert df["raw_id"].to_list() == [0, 1, 2, 3]
        assert df["semantic_id"].to_list() == [[1], [2], [3], [4]]

    def test_parquet_output_joins_embeddings_table(self, tmp_path, monkeypatch):
        # with a real embeddings parquet, item_id/raw_id come from the join.
        emb_path = tmp_path / "emb.parquet"
        pl.DataFrame(
            {"item_id": [10, 11, 12, 13], "normalized_id": [0, 1, 2, 3], "raw_id": ["r0", "r1", "r2", "r3"]}
        ).write_parquet(emb_path)

        tr = self._make_trainer(tmp_path, save_json=False)
        monkeypatch.setattr(
            rq_trainer_mod, "custom_constrained_km", lambda emb, **kws: (None, np.zeros(emb.shape[0], dtype=int))
        )

        def fake_generate(batch, labels, use_sk):
            n = int(batch["emb"].shape[0])
            return torch.arange(n, dtype=torch.int64).view(-1, 1)

        monkeypatch.setattr(tr.model, "generate", fake_generate)
        dataset = self._make_dataset(tmp_path)
        dataset.data_path = str(emb_path)
        output_file = str(tmp_path / "index.parquet")
        tr.generate_indices(dataset, [{"emb": torch.zeros(4, 2)}], output_file)

        df = pl.read_parquet(output_file)
        assert df["item_id"].to_list() == [10, 11, 12, 13]
        assert df["raw_id"].to_list() == ["r0", "r1", "r2", "r3"]
        assert df["normalized_id"].to_list() == [0, 1, 2, 3]
        assert df["semantic_id"].to_list() == [[1], [2], [3], [4]]
