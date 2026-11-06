import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from gensid.config.model_config import SpectralModelConfig
from gensid.config.trainer_config import RecursiveClusteringTrainerConfig
from gensid.method.recursive_hierarchy import Spectral
from gensid.trainer.recursive_clustering_trainer import RecursiveClusteringTrainer


SPECTRAL_SEED = 0


def _make_trainer(tmp_path, seed=SPECTRAL_SEED, save_json=False):
    model = Spectral(SpectralModelConfig(seed=seed, recursive_factor=2, recursive_levels=2))
    trainer = RecursiveClusteringTrainer(
        RecursiveClusteringTrainerConfig(
            seed=seed,
            epochs=1,
            lr=0.01,
            weight_decay=0.0,
            eval_step=1,
            compile=False,
            save_json=save_json,
        ),
        model,
        device="cpu",
        ckpt_dir=str(tmp_path),
        saved_model_dir="trainer",
    )
    return trainer, model


class TestRecursiveClusteringTrainerFit:
    def test_fit_writes_checkpoint_and_sets_trained_flag(self, tmp_path, graph_stub_dataset, graph_stub_loader):
        trainer, model = _make_trainer(tmp_path)

        best_loss, best_collision_rate = trainer.fit(graph_stub_dataset, graph_stub_loader)

        pth = Path(trainer.ckpt_dir) / "best_collision_model.pth"
        assert pth.exists()
        with open(pth.with_suffix(".json")) as f:
            sidecar = json.load(f)
        assert sidecar["fully_trained"] is True
        assert sidecar["epoch"] == 0
        assert sidecar["best_collision_rate"] == best_collision_rate
        assert best_loss == trainer.best_loss
        # 6 items compressed onto 2x2 codebooks -> at least one collision is guaranteed
        assert best_collision_rate > 0.0
        assert model.indices.shape == (6, 2)


class TestRecursiveClusteringTrainerGenerateIndices:
    def test_generate_indices_writes_parquet_and_hierarchy_artifacts(
        self, tmp_path, graph_stub_dataset, graph_stub_loader
    ):
        trainer, _ = _make_trainer(tmp_path)
        trainer.fit(graph_stub_dataset, graph_stub_loader)

        output_file = str(tmp_path / "out.parquet")
        trainer.generate_indices(graph_stub_dataset, graph_stub_loader, output_file)

        output_dir = tmp_path / "mincut_runs" / f"spectral_seed{SPECTRAL_SEED}"
        assert output_dir.is_dir()

        df = pl.read_parquet(output_file)
        assert df.columns == ["item_id", "semantic_id", "raw_id"]
        assert df["item_id"].to_list() == list(range(6))
        assert df["raw_id"].to_list() == list(range(6))
        assert all(len(sem) == 3 for sem in df["semantic_id"].to_list())
        # labels are 0-based -> +1 shift reserves 0 for padding; leaf rank is >= 1
        assert all(all(v >= 1 for v in sem) for sem in df["semantic_id"].to_list())

        with open(output_dir / "graph_hierarchy_labels.json") as f:
            labels = json.load(f)
        for group_key in (
            "coarse_to_fine_item_labels",
            "fine_to_coarse_item_labels",
            "coarse_to_fine_graph_labels",
        ):
            group = labels[group_key]
            assert set(group.keys()) == {"level_1", "level_2"}
            assert all(len(v) == 6 for v in group.values())
        graph_labels = labels["coarse_to_fine_graph_labels"]
        assert all(level != -1 for level in graph_labels["level_1"])
        assert all(level != -1 for level in graph_labels["level_2"])

        node_map = pd.read_csv(output_dir / "graph_hierarchy_node_map.csv")
        assert list(node_map.columns) == ["node_id", "item_id"]
        assert node_map["item_id"].tolist() == list(range(6))

        with open(output_dir / "graph_hierarchy_meta.json") as f:
            meta = json.load(f)
        assert meta["method"] == "spectral"
        assert meta["recursive_factor"] == 2
        assert meta["recursive_levels"] == 2
        assert meta["num_items"] == 6
        assert meta["num_graph_nodes_used"] == 6
        assert meta["num_edges_used"] == 5
        assert meta["skip_n_last"] == 2
        assert meta["graph_type"] == graph_stub_dataset.config.graph_building.graph_type
        assert meta["cluster_counts_coarse_to_fine"][0] == 2
        assert 1 <= meta["cluster_counts_coarse_to_fine"][1] <= 4

        with np.load(output_dir / "graph_hierarchy_artifacts.npz") as npz:
            assert npz["item_ids_graph_order"].tolist() == list(range(6))
            for level in (1, 2):
                array = npz[f"coarse_to_fine_level{level}_graph_labels"]
                assert array.dtype == np.int64
                assert array.shape == (6,)

    def test_generate_indices_save_json_writes_token_codes(self, tmp_path, graph_stub_dataset, graph_stub_loader):
        trainer, _ = _make_trainer(tmp_path, save_json=True)
        trainer.fit(graph_stub_dataset, graph_stub_loader)

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
