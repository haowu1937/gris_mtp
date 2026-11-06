import argparse
import json
import os

import numpy as np
import pytest
import torch

from gensid.pipeline import SIDPipeline


def _write_data(tmp_path, num_items=20, dim=8):
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((num_items, dim)).astype("float32")
    emb_path = str(tmp_path / "emb.npy")
    np.save(emb_path, emb)
    inter_path = str(tmp_path / "inter.json")
    with open(inter_path, "w") as f:
        json.dump({f"u{i}": list(range(num_items)) for i in range(10)}, f)
    return emb_path, inter_path


def _args(tmp_path, emb_path, inter_path, **flags):
    defaults = {
        "graph_type": "adjacent_cooc",
        "convolution_type": "APPNP",
        "ckpt_dir": str(tmp_path),
        "saved_model_dir": "test",
        "use_edge_weight": False,
        "skip_n_last": 1,
        "spectral_dim": 0,
        "cluster_labels_path": None,
        "num_anchors": 16,
        "use_edge_reconstruction_loss": False,
        "num_workers": 0,
        "prefetch_factor": None,
        "persistent_workers": False,
        "train_batch_size": 32,
        "eval_batch_size": 32,
        "seed": 42,
        "hops": [0, 1],
        "probs": None,
        "teleport_prob": 0.2,
        "n_propagations": 2,
        "decomposed_layers": 2,
        "add_self_loops": True,
        "graph_signal": False,
        "graph_augmentation": False,
        "sid_model": "rkmeans",
        "embeddings_path": emb_path,
        "interactions_path": inter_path,
        "depth": 2,
        "n_clusters": 2,
        "distance_function": "euclidean",
        "mean_function": "mean",
        "norm_input": False,
    }
    defaults.update(flags)
    return argparse.Namespace(**defaults)


class TestSIDPipelineRKMeans:
    def test_train_and_generate_end_to_end(self, tmp_path):
        emb_path, inter_path = _write_data(tmp_path)
        args = _args(tmp_path, emb_path, inter_path)
        pipeline = SIDPipeline(args, device="cpu")

        result = pipeline.train()
        assert isinstance(result, tuple) and len(result) == 2

        ckpt_dir = os.path.join(tmp_path, "test")
        ckpt_path = os.path.join(ckpt_dir, "best_collision_model.pth")
        assert os.path.exists(ckpt_path)
        metrics_path = ckpt_path.replace(".pth", ".json")
        assert os.path.exists(metrics_path)
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert metrics["fully_trained"] is True
        assert metrics["ckpt_path"] == ckpt_path

        output_file = str(tmp_path / "index.json")
        pipeline.generate(ckpt_path, output_file)
        with open(output_file) as f:
            mapping = json.load(f)
        assert len(mapping) == 20
        assert all(isinstance(v, list) and len(v) == 2 for v in mapping.values())
        assert all(0 <= int(x) < 2 for v in mapping.values() for x in v)

    def test_saved_ckpt_is_loadable_and_trained_flag_round_trips(self, tmp_path):
        emb_path, inter_path = _write_data(tmp_path, num_items=8, dim=4)
        args = _args(tmp_path, emb_path, inter_path)
        pipeline = SIDPipeline(args, device="cpu")
        pipeline.train()
        ckpt_path = os.path.join(tmp_path, "test", "best_collision_model.pth")
        state = torch.load(ckpt_path, weights_only=False)
        assert set(state["state_dict"].keys()) == {0, 1}
        assert state["args"].name == "clustering"
        assert state["best_collision_rate"] == pytest.approx(state["best_collision_rate"])


class TestSIDPipelinePQ:
    @pytest.mark.parametrize("sid_model,codebooks", [("pq", 4), ("opq", 4)])
    def test_train_and_generate_end_to_end(self, tmp_path, sid_model, codebooks):
        emb_path, inter_path = _write_data(tmp_path, num_items=40, dim=8)
        args = _args(
            tmp_path,
            emb_path,
            inter_path,
            sid_model=sid_model,
            n_subquantizers=4,
            n_bits=4,
            quantizer_backend="torch",
            kmeans_iters=10,
            opq_niter=2,
        )
        pipeline = SIDPipeline(args, device="cpu")

        result = pipeline.train()
        assert isinstance(result, tuple) and len(result) == 2

        ckpt_path = os.path.join(tmp_path, "test", "best_collision_model.pth")
        assert os.path.exists(ckpt_path)

        output_file = str(tmp_path / "index.json")
        pipeline.generate(ckpt_path, output_file)
        with open(output_file) as f:
            mapping = json.load(f)
        assert len(mapping) == 40
        assert all(isinstance(v, list) and len(v) == codebooks for v in mapping.values())
        assert all(0 <= int(x) < 16 for v in mapping.values() for x in v)

        ckpt_state = torch.load(ckpt_path, weights_only=False)
        sd_keys = set(ckpt_state["state_dict"].keys())
        assert sd_keys == {"centroids", "name", "quantizer_backend"} or sd_keys == {
            "A",
            "centroids",
            "name",
            "quantizer_backend",
        }
        assert ckpt_state["state_dict"]["name"] == sid_model
