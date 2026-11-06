import argparse
import json
import os

import numpy as np
import polars as pl
import pytest
import torch

from gensid.config.dataset_config import (
    GRQVAEFusionDatasetConfig,
    GraphFusionDatasetConfig,
    MixedDepthDatasetConfig,
    PlainDatasetConfig,
)
from gensid.config.encoder_config import APPNPEncoderConfig
from gensid.config.graph_build_config import AdjacentCoocGraphConfig
from gensid.data.base import build_data_loader, build_dataset, get_dataset_config
from gensid.data.dataset import (
    EmbDataset,
    GNNFusionEmbDataset,
    GRQVAEFusionEmbDataset,
    MixedFusionDepthDataset,
)


class _IdentityGraphEncoder:
    def __call__(self, graph_data):
        return graph_data.x


class _IdentityGraphEncoderModel:
    def __init__(self):
        self.graph_encoder = _IdentityGraphEncoder()

    def eval(self):
        return self


class _NoEncoderModel:
    def eval(self):
        return self


@pytest.fixture
def emb_files(tmp_path):
    """Write a default 5x4 float32 embedding matrix and return (path, array)."""
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((5, 4)).astype("float32")
    path = str(tmp_path / "emb.npy")
    np.save(path, arr)
    return path, arr


@pytest.fixture
def inter_file(tmp_path):
    """Write a small interaction map and return its path."""
    path = tmp_path / "inter.json"
    path.write_text(json.dumps({"u0": [0, 1, 2, 3], "u1": [1, 2, 4]}))
    return str(path)


@pytest.fixture
def appnp_encoder_config():
    return APPNPEncoderConfig(n_propagations=2, teleport_prob=0.2, decomposed_layers=2)


class TestEmbDataset:
    def _write_embeddings(self, tmp_path, num_items, dim):
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((num_items, dim)).astype("float32")
        path = str(tmp_path / "emb_custom.npy")
        np.save(path, arr)
        return path, arr

    def test_eval_mode_uses_all_items(self, emb_files):
        path, arr = emb_files
        ds = EmbDataset(PlainDatasetConfig(), path, interactions_path=None, eval_mode=True)
        assert len(ds) == 5
        assert ds.dim == 4
        sample = ds[2]
        assert set(sample.keys()) == {"emb", "index", "target"}
        assert sample["index"] == 2
        assert sample["emb"].dtype == torch.float32
        assert np.allclose(sample["emb"].numpy(), arr[2])
        assert torch.equal(sample["emb"], sample["target"])

    @pytest.mark.parametrize(
        ("skip_n_last", "expected_items"),
        [
            (2, [0, 1]),  # user0 last 2 -> [0,1]; user1 last 2 -> [1]
            (1, [0, 1, 2]),  # user0 -> [0,1,2]; user1 -> [1,2]
        ],
    )
    def test_train_mode_builds_train_mask(self, emb_files, inter_file, skip_n_last, expected_items):
        path, arr = emb_files
        ds = EmbDataset(
            PlainDatasetConfig(skip_n_last=skip_n_last),
            path,
            interactions_path=inter_file,
            eval_mode=False,
        )
        assert ds.train_item_ids.tolist() == expected_items
        assert len(ds) == len(expected_items)
        assert np.allclose(ds[0]["emb"].numpy(), arr[0])

    def test_out_of_range_item_raises(self, tmp_path):
        path, _ = self._write_embeddings(tmp_path, num_items=3, dim=4)
        interactions = tmp_path / "inter.json"
        interactions.write_text(json.dumps({"u0": [0, 7, 1]}))
        with pytest.raises(ValueError, match="outside embeddings range"):
            # skip_n_last=None -> the full sequence [0, 7, 1] is used, exposing item 7
            EmbDataset(
                PlainDatasetConfig(skip_n_last=1),
                path,
                interactions_path=str(interactions),
                eval_mode=False,
            )

    def test_parquet_embeddings_eval_mode(self, tmp_path):
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((4, 3)).astype("float32")
        parquet_path = str(tmp_path / "embs.parquet")
        pl.DataFrame({"item_id": list(range(4)), "embedding": [list(row) for row in arr]}).write_parquet(parquet_path)
        ds = EmbDataset(PlainDatasetConfig(), parquet_path, interactions_path=None, eval_mode=True)
        assert len(ds) == 4
        assert ds.dim == 3
        assert np.allclose(ds[0]["emb"].numpy(), arr[0])


class TestGNNFusionEmbDataset:
    def _make(self, emb_files, inter_file, appnp_encoder_config, **kwargs):
        emb_path, _ = emb_files
        eval_mode = kwargs.pop("eval_mode", False)
        defaults = dict(
            graph_building=AdjacentCoocGraphConfig(),
            encoder=appnp_encoder_config,
        )
        defaults.update(kwargs)
        return GNNFusionEmbDataset(
            config=GraphFusionDatasetConfig(**defaults),
            embeddings_path=emb_path,
            interactions_path=inter_file,
            eval_mode=eval_mode,
        )

    def test_train_mode_shapes(self, emb_files, inter_file, appnp_encoder_config):
        ds = self._make(emb_files, inter_file, appnp_encoder_config, eval_mode=False)
        # train items = {0,1}
        assert len(ds) == 2
        assert ds.embeddings.shape == (2, 4)
        sample = ds[0]
        assert set(sample.keys()) == {"emb", "index", "target"}
        assert sample["emb"].dtype == torch.float32

    def test_eval_mode_restores_full_matrix(self, emb_files, inter_file, appnp_encoder_config):
        ds = self._make(emb_files, inter_file, appnp_encoder_config, eval_mode=True)
        assert len(ds) == 5
        assert ds.embeddings.shape == (5, 4)
        assert ds.raw_embeddings.shape == (5, 4)

    def test_spectral_vectors_in_train_mode(self, emb_files, inter_file, appnp_encoder_config):
        ds = self._make(emb_files, inter_file, appnp_encoder_config, eval_mode=False, spectral_dim=2)
        sample = ds[0]
        assert "spectral_vectors" in sample
        assert sample["spectral_vectors"].shape == (2,)

    def test_cluster_labels_attached(self, tmp_path, emb_files, inter_file, appnp_encoder_config):
        labels = tmp_path / "labels.json"
        labels.write_text(json.dumps([0, 1, 2, 3, 4]))
        ds = self._make(
            emb_files,
            inter_file,
            appnp_encoder_config,
            eval_mode=False,
            spectral_dim=0,
            cluster_labels_path=str(labels),
        )
        sample = ds[0]
        assert "cluster_labels" in sample
        assert sample["cluster_labels"].numel() >= 1

    def test_cache_save_and_reload(self, tmp_path, emb_files, inter_file, appnp_encoder_config):
        cache = str(tmp_path / "cache" / "fused.npy")
        ds = self._make(emb_files, inter_file, appnp_encoder_config, eval_mode=True, fused_cache_path=cache)
        assert os.path.exists(cache)
        reloaded = self._make(emb_files, inter_file, appnp_encoder_config, eval_mode=True, fused_cache_path=cache)
        assert np.allclose(reloaded.embeddings, ds.embeddings)

    def test_cache_shape_mismatch_raises(self, tmp_path, emb_files, inter_file, appnp_encoder_config):
        cache_path = tmp_path / "bad.npy"
        np.save(str(cache_path), np.zeros((99, 7), dtype=np.float32))
        with pytest.raises(ValueError, match="does not match expected"):
            self._make(emb_files, inter_file, appnp_encoder_config, eval_mode=True, fused_cache_path=str(cache_path))

    def test_appnp_fused_values_exact(self, tmp_path):
        # 3 items; skip_n_last=1 keeps train prefix [0, 1], so the graph covers items {0, 1}
        # with a single undirected edge (0, 1). APPNP with K=1, alpha=0.5:
        #   h = 0.5 * (D^-1/2 (A + I) D^-1/2) x + 0.5 * x
        # A + I = [[1,1],[1,1]], D = [2,2] -> A_norm = [[0.5,0.5],[0.5,0.5]]
        # h0 = 0.5*[0.5,0.5] + 0.5*[1,0] = [0.75, 0.25]; h1 = [0.25, 0.75].
        emb = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        emb_path = tmp_path / "emb.npy"
        np.save(emb_path, emb)
        inter = tmp_path / "inter.json"
        inter.write_text(json.dumps({"u0": [0, 1, 2]}))
        ds = GNNFusionEmbDataset(
            config=GraphFusionDatasetConfig(
                graph_building=AdjacentCoocGraphConfig(),
                encoder=APPNPEncoderConfig(n_propagations=1, teleport_prob=0.5, decomposed_layers=1),
                fused_cache_path=None,
                use_edge_weight=False,
                skip_n_last=1,
                spectral_dim=0,
            ),
            embeddings_path=str(emb_path),
            interactions_path=str(inter),
            eval_mode=False,
        )
        assert ds.train_item_ids.tolist() == [0, 1]
        assert np.allclose(ds.embeddings, [[0.75, 0.25], [0.25, 0.75]], atol=1e-6)

    def test_empty_interactions_raises(self, tmp_path, emb_files):
        emb_path, _ = emb_files
        inter = tmp_path / "empty.json"
        inter.write_text(json.dumps({}))
        # empty interactions -> no train items -> empty item universe in the graph builder
        with pytest.raises(ValueError, match="all_item_ids is empty"):
            GNNFusionEmbDataset(
                config=GraphFusionDatasetConfig(
                    graph_building=AdjacentCoocGraphConfig(),
                    encoder=APPNPEncoderConfig(n_propagations=2, teleport_prob=0.2, decomposed_layers=2),
                ),
                embeddings_path=emb_path,
                interactions_path=str(inter),
            )


class TestMixedFusionDepthDataset:
    def _make(self, tmp_path, emb_files, inter_file, appnp_encoder_config, **kwargs):
        emb_path, _ = emb_files
        defaults = dict(
            hops=[0, 1],
            probs=[0.5, 0.5],
            fused_cache_root=str(tmp_path / "fused"),
            graph_building=AdjacentCoocGraphConfig(),
            encoder=appnp_encoder_config,
        )
        defaults.update(kwargs)
        return MixedFusionDepthDataset(
            config=MixedDepthDatasetConfig(**defaults),
            embeddings_path=emb_path,
            interactions_path=inter_file,
        )

    @pytest.mark.parametrize(
        ("hops", "probs", "expected_hops", "expected_probs"),
        [
            ([0, 1], [0.5, 0.5], [0, 1], [0.5, 0.5]),
            ([0, 1], None, [0, 1], [0.5, 0.5]),  # uniform default
            ([1, 1], [0.25, 0.75], [1], [1.0]),  # duplicated hops merge probabilities
        ],
    )
    def test_hops_normalization(
        self, tmp_path, emb_files, inter_file, appnp_encoder_config, hops, probs, expected_hops, expected_probs
    ):
        ds = self._make(tmp_path, emb_files, inter_file, appnp_encoder_config, hops=hops, probs=probs)
        assert ds.hops == expected_hops
        assert ds.probs == pytest.approx(expected_probs)
        assert len(ds) == len(ds.datasets[0])

    def test_datasets_built_and_getitem(self, tmp_path, emb_files, inter_file, appnp_encoder_config):
        ds = self._make(tmp_path, emb_files, inter_file, appnp_encoder_config)
        assert ds.hops == [0, 1]
        assert len(ds.datasets) == 2
        assert len(ds) > 0
        sample = ds[0]
        assert set(sample.keys()) >= {"emb", "index", "target"}

    def test_hop0_raw_hop1_fused_values(self, tmp_path, emb_files):
        # hop 0 must return the raw train embeddings, hop 1 the graph-fused ones; the
        # hop-1 fusion must match a standalone GNNFusionEmbDataset with the same config.
        emb_path, arr = emb_files
        inter = tmp_path / "inter.json"
        inter.write_text(json.dumps({"u0": [0, 1, 2]}))
        cfg = APPNPEncoderConfig(n_propagations=1, teleport_prob=0.2, decomposed_layers=2)
        kwargs = dict(
            graph_building=AdjacentCoocGraphConfig(),
            encoder=cfg,
            skip_n_last=1,
        )
        single = GNNFusionEmbDataset(
            config=GraphFusionDatasetConfig(**kwargs),
            embeddings_path=emb_path,
            interactions_path=str(inter),
            eval_mode=False,
        )
        ds = MixedFusionDepthDataset(
            config=MixedDepthDatasetConfig(
                hops=[0, 1], probs=[0.5, 0.5], fused_cache_root=str(tmp_path / "fused"), **kwargs
            ),
            embeddings_path=emb_path,
            interactions_path=str(inter),
            eval_mode=False,
        )
        assert ds.hops == [0, 1]
        assert np.allclose(ds.datasets[0].embeddings, arr[:2])
        assert np.allclose(ds.datasets[1].embeddings, single.embeddings)

    def test_nonzero_hops_share_graph_builder(self, tmp_path, appnp_encoder_config):
        emb_path, _ = self._emb_files(tmp_path)
        inter = tmp_path / "inter.json"
        inter.write_text(json.dumps({"u0": [0, 1, 2]}))
        ds = MixedFusionDepthDataset(
            config=MixedDepthDatasetConfig(
                hops=[0, 1, 2],
                probs=[0.5, 0.25, 0.25],
                fused_cache_root=str(tmp_path / "fused"),
                skip_n_last=1,
                graph_building=AdjacentCoocGraphConfig(),
                encoder=appnp_encoder_config,
            ),
            embeddings_path=emb_path,
            interactions_path=str(inter),
            eval_mode=False,
        )
        assert ds.datasets[1].graph_builder is ds.datasets[2].graph_builder
        assert ds.datasets[1].pyg_data is ds.datasets[2].pyg_data

    def _emb_files(self, tmp_path):
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((5, 4)).astype("float32")
        path = str(tmp_path / "emb_mfd.npy")
        np.save(path, arr)
        return path, arr

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"hops": []}, "hops must be non-empty"),
            ({"hops": [0, 1], "probs": [0.5]}, "probs must match"),
            ({"hops": [0], "probs": [0.0]}, r"sum\(probs\)"),
        ],
    )
    def test_invalid_config_raises(self, tmp_path, emb_files, inter_file, appnp_encoder_config, overrides, match):
        with pytest.raises(ValueError, match=match):
            self._make(tmp_path, emb_files, inter_file, appnp_encoder_config, **overrides)


class TestGRQVAEFusionEmbDataset:
    def _make(self, emb_files, inter_file, **kwargs):
        emb_path, _ = emb_files
        eval_mode = kwargs.pop("eval_mode", False)
        defaults = dict(
            graph_building=AdjacentCoocGraphConfig(),
        )
        defaults.update(kwargs)
        return GRQVAEFusionEmbDataset(
            config=GRQVAEFusionDatasetConfig(**defaults),
            embeddings_path=emb_path,
            interactions_path=inter_file,
            eval_mode=eval_mode,
        )

    def test_train_mode_serves_raw_semantic_embeddings(self, emb_files, inter_file):
        ds = self._make(emb_files, inter_file, eval_mode=False)
        # train items = {0,1}; no fusion happens during training (raw semantic emb)
        assert len(ds) == 2
        assert ds.embeddings.shape == (2, 4)
        assert np.allclose(ds.embeddings, ds.raw_embeddings)
        # graph node features are the semantic embeddings of the graph items
        assert ds.pyg_data.x.shape[0] == len(ds.item_ids)
        assert np.allclose(ds.pyg_data.x.numpy(), np.load(ds.embeddings_path)[ds.item_ids])
        sample = ds[0]
        assert set(sample.keys()) == {"emb", "index", "target"}
        assert sample["emb"].dtype == torch.float32

    def test_eval_mode_without_cache_raises(self, emb_files, inter_file):
        with pytest.raises(FileNotFoundError, match="fused embedding cache"):
            self._make(emb_files, inter_file, eval_mode=True, fused_cache_path=None)

    def test_save_fused_cache_and_eval_reload(self, tmp_path, emb_files, inter_file):
        emb_path, arr = emb_files
        cache = str(tmp_path / "graph.npy")
        train_ds = self._make(emb_files, inter_file, eval_mode=False, fused_cache_path=cache)
        model = _IdentityGraphEncoderModel()

        train_ds.save_fused_cache(model, device="cpu")
        assert os.path.exists(cache)

        # cache holds the train-item fused embeddings in train_item_ids order
        cached = np.load(cache)
        assert cached.shape == (train_ds.num_train_items, 4)
        # identity graph encoder -> fused == raw train embeddings
        assert np.allclose(cached, arr[train_ds.train_item_ids])

        # eval dataset reloads the cache and scatters it back onto the train items
        eval_ds = self._make(emb_files, inter_file, eval_mode=True, fused_cache_path=cache)
        assert len(eval_ds) == 5
        assert eval_ds.embeddings.shape == (5, 4)
        assert np.allclose(eval_ds.embeddings[train_ds.train_item_ids], cached)
        # non-train items keep the raw embeddings (identity fused == raw for all)
        assert np.allclose(eval_ds.embeddings, arr)

    def test_save_fused_cache_missing_path_raises(self, emb_files, inter_file):
        train_ds = self._make(emb_files, inter_file, eval_mode=False, fused_cache_path=None)
        with pytest.raises(ValueError, match="no fused_cache_path"):
            train_ds.save_fused_cache(_IdentityGraphEncoderModel(), device="cpu")

    def test_save_fused_cache_requires_graph_encoder(self, emb_files, inter_file, tmp_path):
        train_ds = self._make(emb_files, inter_file, eval_mode=False, fused_cache_path=str(tmp_path / "graph.npy"))
        with pytest.raises(ValueError, match="no graph_encoder"):
            train_ds.save_fused_cache(_NoEncoderModel(), device="cpu")


class TestDatasetResolver:
    """get_dataset_config / build_dataset / build_data_loader (gensid.data.base)."""

    def _args(self, tmp_path, **flags):
        defaults = dict(
            graph_type="adjacent_cooc",
            convolution_type="APPNP",
            ckpt_dir=str(tmp_path),
            use_edge_weight=False,
            skip_n_last=2,
            spectral_dim=0,
            cluster_labels_path=None,
            num_anchors=16,
            use_edge_reconstruction_loss=False,
            num_workers=0,
            prefetch_factor=None,
            persistent_workers=False,
            train_batch_size=8,
            eval_batch_size=8,
            seed=42,
            hops=[0, 1],
            probs=None,
            teleport_prob=0.2,
            n_propagations=2,
            decomposed_layers=2,
            add_self_loops=True,
            graph_signal=False,
            graph_augmentation=False,
            sid_model="rqvae",
        )
        defaults.update(flags)
        return argparse.Namespace(**defaults)

    @pytest.mark.parametrize(
        ("flags", "expected_cls"),
        [
            ({}, PlainDatasetConfig),
            ({"graph_signal": True}, GraphFusionDatasetConfig),
            ({"graph_augmentation": True}, MixedDepthDatasetConfig),
            ({"sid_model": "grqvae"}, GRQVAEFusionDatasetConfig),
        ],
    )
    def test_get_dataset_config_maps_flags(self, tmp_path, flags, expected_cls):
        cfg = get_dataset_config(self._args(tmp_path, **flags))
        assert isinstance(cfg, expected_cls)
        assert cfg.name == expected_cls().name

    def test_graph_fusion_resolver_round_trip(self, tmp_path):
        emb = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        emb_path = tmp_path / "emb.npy"
        np.save(emb_path, emb)
        inter = tmp_path / "inter.json"
        inter.write_text(json.dumps({"u0": [0, 1, 2]}))
        args = self._args(
            tmp_path, graph_signal=True, skip_n_last=1, embeddings_path=str(emb_path), interactions_path=str(inter)
        )

        cfg = get_dataset_config(args)
        ds = build_dataset(cfg, args.embeddings_path, args.interactions_path, eval_mode=False)
        loader = build_data_loader(ds, cfg, args, eval_mode=False)
        assert isinstance(ds, GNNFusionEmbDataset)
        assert ds.embeddings.shape == (2, 2)
        batch = next(iter(loader))
        assert batch["emb"].shape[0] == 2

    def test_grqvae_resolver_trains_on_neighbor_loader(self, tmp_path):
        emb_path = tmp_path / "emb.npy"
        np.save(emb_path, np.random.default_rng(0).standard_normal((3, 4)).astype("float32"))
        inter = tmp_path / "inter.json"
        inter.write_text(json.dumps({"u0": [0, 1, 2]}))
        args = self._args(tmp_path, sid_model="grqvae", embeddings_path=str(emb_path), interactions_path=str(inter))

        cfg = get_dataset_config(args)
        ds = build_dataset(cfg, args.embeddings_path, args.interactions_path, eval_mode=False)
        loader = build_data_loader(ds, cfg, args, eval_mode=False)
        assert isinstance(ds, GRQVAEFusionEmbDataset)
        assert type(loader).__name__ == "NeighborLoader"
        assert loader.dataset is not None

    def test_grqvae_resolver_eval_uses_plain_dataloader(self, tmp_path):
        emb_path = tmp_path / "emb.npy"
        np.save(emb_path, np.random.default_rng(0).standard_normal((3, 4)).astype("float32"))
        inter = tmp_path / "inter.json"
        inter.write_text(json.dumps({"u0": [0, 1, 2]}))
        # the post-training fusion cache must exist before an eval dataset can be built;
        # train items for {"u0": [0,1,2]} with skip_n_last=2 -> {0} (shape [1, 4])
        cache = str(tmp_path / "graph.npy")
        np.save(cache, np.random.default_rng(0).standard_normal((1, 4)).astype("float32"))
        args = self._args(tmp_path, sid_model="grqvae", embeddings_path=str(emb_path), interactions_path=str(inter))

        cfg = get_dataset_config(args)
        ds = build_dataset(cfg, args.embeddings_path, args.interactions_path, eval_mode=True)
        loader = build_data_loader(ds, cfg, args, eval_mode=True)
        assert isinstance(ds, GRQVAEFusionEmbDataset)
        assert type(loader).__name__ == "DataLoader"
