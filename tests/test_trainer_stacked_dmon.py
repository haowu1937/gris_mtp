import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from gensid.config.model_config import StackedDMoNModelConfig
from gensid.config.trainer_config import StackedDMoNTrainerConfig
from gensid.method.stacked_dmon import StackedDMoN
from gensid.trainer.stacked_dmon_trainer import StackedDMoNTrainer


STACKED_SEED = 0


def _make_trainer(tmp_path, seed=STACKED_SEED, save_json=False):
    model = StackedDMoN(
        StackedDMoNModelConfig(
            num_emb_list=(2, 2),
            in_dim=4,
            e_dim=8,
            dmon_hidden_dim=8,
            assign_hidden_dim=8,
        )
    )
    trainer = StackedDMoNTrainer(
        StackedDMoNTrainerConfig(
            seed=seed,
            epochs=1,
            lr=0.01,
            weight_decay=0.0,
            eval_step=1,
            compile=False,
            grad_clip=5.0,
            save_json=save_json,
        ),
        model,
        device="cpu",
        ckpt_dir=str(tmp_path),
        saved_model_dir="trainer",
    )
    return trainer, model


class TestStackedDMoNTrainerSteps:
    def test_train_epoch_reports_per_level_dmon_metrics(self, tmp_path, graph_stub_dataset, graph_stub_loader):
        trainer, _ = _make_trainer(tmp_path)
        torch.manual_seed(0)

        loss, train_info = trainer._train_epoch(graph_stub_loader, 0)
        assert isinstance(loss, float)
        assert np.isfinite(loss)
        for level in (1, 2):
            for key in ("loss", "modularity", "balance", "entropy"):
                assert f"level{level}_{key}" in train_info

        loss2, _ = trainer._train_epoch(graph_stub_loader, 1)
        assert np.isfinite(loss2)

    def test_valid_epoch_returns_collision_rate(self, tmp_path, graph_stub_dataset, graph_stub_loader):
        trainer, _ = _make_trainer(tmp_path)
        torch.manual_seed(0)
        trainer._train_epoch(graph_stub_loader, 0)

        rate = trainer._valid_epoch(graph_stub_loader)
        assert 0.0 <= rate <= 1.0
        # 6 items compressed onto 2x2 codebooks -> duplicate SIDs are guaranteed
        assert rate > 0.0

    def test_fit_writes_best_collision_ckpt_and_trained_flag(self, tmp_path, graph_stub_dataset, graph_stub_loader):
        trainer, _ = _make_trainer(tmp_path)
        torch.manual_seed(0)

        best_loss, best_collision_rate = trainer.fit(graph_stub_dataset, graph_stub_loader)
        assert np.isfinite(best_loss)
        assert 0.0 <= best_collision_rate <= 1.0

        pth = Path(trainer.ckpt_dir) / "best_collision_model.pth"
        assert pth.exists()
        with open(pth.with_suffix(".json")) as f:
            sidecar = json.load(f)
        assert sidecar["fully_trained"] is True
        assert sidecar["epoch"] == 0
        assert sidecar["best_collision_rate"] == best_collision_rate


class TestStackedDMoNTrainerGenerateIndices:
    def test_generate_indices_writes_parquet_and_artifacts(self, tmp_path, graph_stub_dataset, graph_stub_loader):
        trainer, _ = _make_trainer(tmp_path)
        torch.manual_seed(0)

        output_file = str(tmp_path / "out.parquet")
        trainer.generate_indices(graph_stub_dataset, graph_stub_loader, output_file)

        output_dir = tmp_path / "mincut_runs" / f"stacked_dmon_seed{STACKED_SEED}"
        assert output_dir.is_dir()

        df = pl.read_parquet(output_file)
        assert df.columns == ["item_id", "semantic_id", "raw_id"]
        assert df["item_id"].to_list() == list(range(6))
        assert df["raw_id"].to_list() == list(range(6))
        assert all(len(sem) == 3 for sem in df["semantic_id"].to_list())
        assert all(all(v >= 1 for v in sem) for sem in df["semantic_id"].to_list())

        with open(output_dir / "graph_hierarchy_meta.json") as f:
            meta = json.load(f)
        assert meta["method"] == "stacked_dmon"
        assert meta["cluster_sizes_fine_to_coarse"] == [2, 2]
        assert meta["cluster_sizes_coarse_to_fine"] == [2, 2]
        assert meta["num_items"] == 6
        assert meta["num_graph_nodes_used"] == 6

        with open(output_dir / "graph_hierarchy_labels.json") as f:
            labels = json.load(f)
        for group in labels.values():
            assert set(group.keys()) == {"level_1", "level_2"}
            assert all(len(v) == 6 for v in group.values())

        with np.load(output_dir / "graph_hierarchy_artifacts.npz") as npz:
            assert npz["item_ids_graph_order"].tolist() == list(range(6))
            for key in (
                "encoded_nodes",
                "level1_s_probs",
                "level1_s_logits",
                "level2_s_probs",
                "level2_s_logits",
            ):
                assert key in npz.files
            assert npz["coarse_to_fine_level1_graph_labels"].shape == (6,)
            assert npz["coarse_to_fine_level2_graph_labels"].shape == (6,)

    def test_generate_indices_save_json_writes_token_codes(self, tmp_path, graph_stub_dataset, graph_stub_loader):
        trainer, _ = _make_trainer(tmp_path, save_json=True)
        torch.manual_seed(0)

        output_file = str(tmp_path / "out.json")
        trainer.generate_indices(graph_stub_dataset, graph_stub_loader, output_file)

        with open(output_file) as f:
            codes = json.load(f)
        assert sorted(codes) == [str(i) for i in range(6)]
        for tokens in codes.values():
            assert len(tokens) == 3
            assert tokens[0].startswith("<a_")
            assert tokens[1].startswith("<b_")
            assert tokens[2].startswith("<z_")
