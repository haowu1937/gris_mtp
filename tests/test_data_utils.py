import json

import numpy as np
import polars as pl
import pytest
import torch
from torch_geometric.data import Data

from gensid.data.utils import (
    EdgeWeightCollator,
    edge_w_from_subgraph,
    load_embeddings,
    load_graph_cluster_labels,
    load_train_item_mask,
)


class TestLoadEmbeddings:
    def _write_parquet(self, tmp_path, arr):
        path = str(tmp_path / "embs.parquet")
        pl.DataFrame(
            {
                "item_id": list(range(arr.shape[0])),
                "embedding": [list(row) for row in arr],
            }
        ).write_parquet(path)
        return path

    def test_loads_npy(self, tmp_path):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        path = str(tmp_path / "embs.npy")
        np.save(path, arr)
        assert np.array_equal(load_embeddings(path), arr)

    def test_loads_parquet_embedding_column(self, tmp_path):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        path = self._write_parquet(tmp_path, arr)
        out = load_embeddings(path)
        assert out.shape == (3, 4)
        assert out.dtype == np.float32
        assert np.array_equal(out, arr)

    def test_unsupported_extension_raises(self, tmp_path):
        path = str(tmp_path / "embs.txt")
        with pytest.raises(ValueError, match="Unsupported embeddings file format"):
            load_embeddings(path)


class TestLoadTrainItemMask:
    def test_basic_mask_and_skip(self, tmp_path):
        path = tmp_path / "inter.json"
        path.write_text(json.dumps({"u0": [0, 1, 2, 3], "u1": [1, 2, 4]}))
        # skip_n_last=2 -> u0:[0,1], u1:[1]
        assert load_train_item_mask(str(path), 5, skip_n_last=2).tolist() == [0, 1]
        # no skip -> {0,1,2,3,4}
        assert load_train_item_mask(str(path), 5, skip_n_last=None).tolist() == [0, 1, 2, 3, 4]

    def test_negative_item_raises(self, tmp_path):
        path = tmp_path / "inter.json"
        path.write_text(json.dumps({"u0": [0, -2, 1]}))
        with pytest.raises(ValueError, match="outside embeddings range"):
            load_train_item_mask(str(path), 3, skip_n_last=1)


class TestLoadGraphClusterLabels:
    def _write(self, tmp_path, payload):
        path = tmp_path / "labels.json"
        path.write_text(json.dumps(payload))
        return str(path)

    def test_none_path_returns_none(self):
        assert load_graph_cluster_labels(None) is None

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_graph_cluster_labels(str(tmp_path / "nope.json"))

    def test_single_vector(self, tmp_path):
        path = self._write(tmp_path, [0, 1, -1])
        assert load_graph_cluster_labels(path, num_items=3).tolist() == [0, 1, -1]

    def test_single_vector_with_levels_raises(self, tmp_path):
        path = self._write(tmp_path, [0, 1])
        with pytest.raises(ValueError, match="single label vector"):
            load_graph_cluster_labels(path, levels=1)

    def test_list_of_vectors(self, tmp_path):
        path = self._write(tmp_path, [[0, 1], [1, 0]])
        labels = load_graph_cluster_labels(path)
        assert isinstance(labels, list) and len(labels) == 2
        assert labels[0].tolist() == [0, 1]

    def test_flat_level_dict_and_selection(self, tmp_path):
        path = self._write(tmp_path, {"level_1": [0, 1, 0], "level_2": [5, 5, 6]})
        l1 = load_graph_cluster_labels(path, levels=1, num_items=3)
        assert l1.tolist() == [0, 1, 0]
        both = load_graph_cluster_labels(path, levels=[2, 1], num_items=3)
        assert len(both) == 2
        assert both[0].tolist() == [5, 5, 6]

    def test_grouped_dict_uses_label_group(self, tmp_path):
        payload = {
            "coarse_to_fine_item_labels": {"level_1": [0, 1], "level_2": [2, 2]},
            "fine_to_coarse_item_labels": {"level_1": [3, 3], "level_2": [4, 4]},
        }
        path = self._write(tmp_path, payload)
        labels = load_graph_cluster_labels(path, label_group="coarse_to_fine_item_labels")
        assert labels[0].tolist() == [0, 1]

    def test_single_named_vector(self, tmp_path):
        path = self._write(tmp_path, {"labels": [1, 2]})
        assert load_graph_cluster_labels(path).tolist() == [1, 2]

    def test_level_not_found_raises(self, tmp_path):
        path = self._write(tmp_path, {"level_1": [0, 1]})
        with pytest.raises(ValueError, match="not found"):
            load_graph_cluster_labels(path, levels="level_9")

    def test_num_items_mismatch_raises(self, tmp_path):
        path = self._write(tmp_path, {"level_1": [0, 1]})
        with pytest.raises(ValueError, match="num_items"):
            load_graph_cluster_labels(path, num_items=5)

    def test_ambiguous_label_group_raises(self, tmp_path):
        payload = {
            "a": {"level_1": [0, 1]},
            "b": {"level_1": [1, 0]},
        }
        path = self._write(tmp_path, payload)
        with pytest.raises(ValueError, match="Available label groups"):
            load_graph_cluster_labels(path)

    def test_invalid_root_type_raises(self, tmp_path):
        path = self._write(tmp_path, 42)
        with pytest.raises(ValueError, match="Unsupported labels JSON root type"):
            load_graph_cluster_labels(path)


class TestEdgeWFromSubgraph:
    def test_builds_batch_adjacency(self):
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_weight = torch.tensor([1.0, 2.0, 3.0, 4.0])
        pyg_data = type("D", (), {"edge_index": edge_index, "edge_weight": edge_weight})()
        emb_idx = torch.arange(4)
        ew = edge_w_from_subgraph(emb_idx, pyg_data)
        assert ew.shape == (4, 4)
        assert ew[0, 1] == 1.0
        assert ew[3, 0] == 4.0
        assert (ew >= 0).all()


class TestEdgeWeightCollator:
    @pytest.fixture
    def pyg_data(self):
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_weight = torch.tensor([1.0, 2.0, 3.0, 4.0])
        return Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=5)

    @pytest.fixture
    def batch(self):
        return [
            {
                "emb": torch.full((4,), float(i)),
                "index": torch.tensor(i),
                "target": torch.full((4,), float(i)),
            }
            for i in range(5)
        ]

    @pytest.mark.parametrize(
        ("num_anchors", "expected_cols"),
        [
            (8, 5),  # B=5 rows, min(B, num_anchors)=5 cols
            (2, 2),
        ],
    )
    def test_returns_edge_weights_matrix(self, pyg_data, batch, num_anchors, expected_cols):
        collator = EdgeWeightCollator(pyg_data, num_anchors=num_anchors)
        out = collator(batch)
        assert "edge_weights" in out
        assert out["edge_weights"].shape == (5, expected_cols)
        assert out["emb"].shape == (5, 4)

    def test_batched_subgraph_edges_placed(self, pyg_data, batch):
        collator = EdgeWeightCollator(pyg_data, num_anchors=8)
        ew = collator(batch)["edge_weights"]
        # edge (0,1) with weight 1.0 must be reflected after relabeling
        assert ew[0, 1] == pytest.approx(1.0)
        assert ew[1, 2] == pytest.approx(2.0)
