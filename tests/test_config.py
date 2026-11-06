import argparse

import pytest

from gensid.config.dataset_config import (
    DatasetConfig,
    GRQVAEFusionDatasetConfig,
    GraphDatasetConfig,
    GraphFusionDatasetConfig,
    MixedDepthDatasetConfig,
    PlainDatasetConfig,
)
from gensid.config.encoder_config import (
    APPNPEncoderConfig,
    EncoderConfig,
    GNNFusionEncoderConfig,
    GraphEncoderConfig,
    GraphSageEncoderConfig,
    S2GREncoderConfig,
)
from gensid.config.graph_build_config import (
    AdjacentCoocGraphConfig,
    AdjacentCoocProdGraphConfig,
    GraphBuildingConfig,
    RandomWalkGraphConfig,
    WindowedGraphConfig,
)
from gensid.config.model_config import (
    HierarchicalRKMeansModelConfig,
    RKMeansModelConfig,
    RecDMoNModelConfig,
    SpectralModelConfig,
    StackedDMoNModelConfig,
)
from gensid.config.trainer_config import (
    RecursiveClusteringTrainerConfig,
    StackedDMoNTrainerConfig,
)
from gensid.registry import REGISTRY


# ---------------------------------------------------------------------------
# Graph building configs
#
# The typed config classes (GraphBuildingConfig hierarchy) are the stable public
# configuration surface. The Registry is the single dispatch mechanism: a
# registered name resolves to a config class whose default instance reports the
# same name. The tests below pin down that contract.
# ---------------------------------------------------------------------------


class TestGraphBuildingConfig:
    @pytest.mark.parametrize(
        ("cfg_class", "expected_defaults"),
        [
            (GraphBuildingConfig, {"graph_type": "adjacent_cooc"}),
            (AdjacentCoocGraphConfig, {"graph_type": "adjacent_cooc"}),
            (
                AdjacentCoocProdGraphConfig,
                {
                    "graph_type": "adjacent_cooc_prod",
                    "seq_col": "history_item_id",
                    "filter_col": None,
                    "allowed_filter_values": None,
                    "directed": True,
                    "chronological": True,
                    "embeddings_path": None,
                    "files_per_chunk": 16,
                    "tmp_dir": None,
                },
            ),
            (
                RandomWalkGraphConfig,
                {
                    "graph_type": "rw",
                    "walk_length": 10,
                    "walks_per_item": 10,
                    "window_size": 2,
                    "self_loops": False,
                    "dedup_per_walk": False,
                },
            ),
            (
                WindowedGraphConfig,
                {
                    "graph_type": "windowed",
                    "window_size": 3,
                    "bidirectional": False,
                    "dedup_within_window": False,
                    "decay": None,
                },
            ),
        ],
    )
    def test_defaults(self, cfg_class, expected_defaults):
        cfg = cfg_class()
        for attr, expected in expected_defaults.items():
            assert getattr(cfg, attr) == expected

    @pytest.mark.parametrize(
        "cfg_class",
        [
            GraphBuildingConfig,
            AdjacentCoocGraphConfig,
            AdjacentCoocProdGraphConfig,
            RandomWalkGraphConfig,
            WindowedGraphConfig,
        ],
    )
    def test_to_kwargs_drops_graph_type_and_round_trips(self, cfg_class):
        cfg = cfg_class()
        assert "graph_type" not in cfg.to_kwargs()
        assert set(cfg.to_kwargs()) == set(cfg.__dict__) - {"graph_type"}
        assert cfg_class(**cfg.to_kwargs()) == cfg

    def test_adjacent_cooc_inherits_from_base(self):
        assert issubclass(AdjacentCoocGraphConfig, GraphBuildingConfig)

    def test_registered_configs_are_self_consistent(self):
        # The Registry is the dispatch mechanism: a registered name resolves to a
        # config class whose default instance reports the same graph_type.
        assert REGISTRY.keys("graph_building_config")
        for name in REGISTRY.keys("graph_building_config"):
            cfg_class = REGISTRY.get_graph_building_config(name)
            assert isinstance(cfg_class(), GraphBuildingConfig)
            assert cfg_class().graph_type == name

    def test_unknown_graph_type_raises(self):
        with pytest.raises(KeyError):
            REGISTRY.get_graph_building_config("bogus")


class TestEncoderConfig:
    @pytest.mark.parametrize(
        ("cfg_class", "expected_defaults"),
        [
            (EncoderConfig, {"name": None}),
            (GraphEncoderConfig, {"name": None}),
            (
                APPNPEncoderConfig,
                {
                    "name": "APPNP",
                    "teleport_prob": 0.1,
                    "n_propagations": 10,
                    "decomposed_layers": 16,
                    "add_self_loops": True,
                },
            ),
            (S2GREncoderConfig, {"name": "S2GR", "n_propagations": 10}),
            (
                GNNFusionEncoderConfig,
                {
                    "name": "GNNFusion",
                    "convolution_type": "gat",
                    "hidden_dim": 128,
                    "out_dim": 64,
                    "layers": 2,
                    "heads": 4,
                    "gated_aggr": "add",
                    "gin_eps": 0.0,
                    "gin_train_eps": False,
                },
            ),
            (
                GraphSageEncoderConfig,
                {
                    "name": "GraphSage",
                    "hidden_dim": 128,
                    "out_dim": 64,
                    "epochs": 100,
                    "lr": 1e-3,
                    "weight_decay": 1e-6,
                },
            ),
        ],
    )
    def test_defaults(self, cfg_class, expected_defaults):
        cfg = cfg_class()
        for attr, expected in expected_defaults.items():
            assert getattr(cfg, attr) == expected

    @pytest.mark.parametrize(
        "cfg_class",
        [
            EncoderConfig,
            GraphEncoderConfig,
            APPNPEncoderConfig,
            S2GREncoderConfig,
            GNNFusionEncoderConfig,
            GraphSageEncoderConfig,
        ],
    )
    def test_to_kwargs_round_trips(self, cfg_class):
        cfg = cfg_class()
        assert cfg_class(**cfg.to_kwargs()) == cfg

    def test_graph_encoder_is_pure_propagation_config(self):
        # Dataset concerns (split, graph construction, fusion) must NOT live on the
        # encoder config: they belong to DatasetConfig.
        for attr in ("graph_building", "skip_n_last"):
            assert not hasattr(GraphEncoderConfig(), attr)

    def test_dataset_concerns_live_in_dataset_configs(self):
        assert GraphFusionDatasetConfig().graph_building.graph_type == "adjacent_cooc"
        assert GraphFusionDatasetConfig().skip_n_last == 2
        assert GRQVAEFusionDatasetConfig().fused_cache_path is None
        assert MixedDepthDatasetConfig().hops == (0,)

    def test_registered_encoder_configs_are_self_consistent(self):
        # The Registry is the dispatch mechanism: each registered name must resolve
        # to a distinct config class (no silent aliases).
        assert REGISTRY.keys("encoder_config")
        resolved_names = set()
        for name in REGISTRY.keys("encoder_config"):
            cfg_class = REGISTRY.get_encoder_config(name)
            assert issubclass(cfg_class, EncoderConfig)
            resolved_names.add(cfg_class().name)
        assert len(resolved_names) == len(REGISTRY.keys("encoder_config"))

    def test_registered_encoder_configs_build_from_args(self):
        # from_args on a registered encoder config must yield the same class.
        for name in REGISTRY.keys("encoder_config"):
            cfg_class = REGISTRY.get_encoder_config(name)
            cfg = cfg_class.from_args(argparse.Namespace(teleport_prob=0.2))
            assert isinstance(cfg, cfg_class)

    def test_unknown_encoder_raises(self):
        with pytest.raises(KeyError):
            REGISTRY.get_encoder_config("bogus")


class TestDatasetConfig:
    @pytest.mark.parametrize(
        ("cfg_class", "expected_defaults"),
        [
            (DatasetConfig, {"name": None, "skip_n_last": 2}),
            (PlainDatasetConfig, {"name": "plain", "skip_n_last": 2}),
            (
                GraphFusionDatasetConfig,
                {
                    "name": "graph_fusion",
                    "skip_n_last": 2,
                    "fused_cache_path": None,
                    "use_edge_weight": False,
                    "spectral_dim": 0,
                    "cluster_labels_path": None,
                },
            ),
            (
                MixedDepthDatasetConfig,
                {
                    "name": "mixed_depth",
                    "hops": (0,),
                    "probs": None,
                    "fused_cache_root": None,
                    "return_hop": False,
                    "seed": None,
                },
            ),
            (
                GRQVAEFusionDatasetConfig,
                {"name": "grqvae_fusion", "fused_cache_path": None},
            ),
        ],
    )
    def test_defaults(self, cfg_class, expected_defaults):
        cfg = cfg_class()
        for attr, expected in expected_defaults.items():
            assert getattr(cfg, attr) == expected

    @pytest.mark.parametrize(
        "cfg_class",
        [
            DatasetConfig,
            PlainDatasetConfig,
            GraphDatasetConfig,
            GraphFusionDatasetConfig,
            MixedDepthDatasetConfig,
            GRQVAEFusionDatasetConfig,
        ],
    )
    def test_to_kwargs_round_trips(self, cfg_class):
        cfg = cfg_class()
        assert cfg_class(**cfg.to_kwargs()) == cfg

    def test_graph_configs_compose_graph_building_and_encoder(self):
        # Dataset configs delegate the *refinement* to a composed encoder config and
        # the *graph construction* to a composed graph-building config.
        assert isinstance(GraphFusionDatasetConfig().graph_building, GraphBuildingConfig)
        assert isinstance(GraphFusionDatasetConfig().encoder, EncoderConfig)
        assert isinstance(MixedDepthDatasetConfig().encoder, EncoderConfig)

    def test_registered_dataset_configs_are_self_consistent(self):
        # Contract the Registry must preserve: each registered name resolves to a
        # config class whose default instance reports the same name.
        names = REGISTRY.keys("dataset_config")
        assert set(names) == {"graph_fusion", "graph_clustering", "grqvae_fusion", "mixed_depth", "plain"}
        for name in names:
            cfg_cls = REGISTRY.get_dataset_config(name)
            assert issubclass(cfg_cls, DatasetConfig)
            assert cfg_cls().name == name

    def test_registered_datasets_pair_with_their_configs(self):
        # Each dataset class is keyed by the same name as its config, so a config
        # can drive dataset construction through the registry.
        for name in REGISTRY.keys("dataset"):
            dataset_cls = REGISTRY.get_dataset(name)
            assert REGISTRY.get_dataset_config(name)().name == dataset_cls.name == name

    def test_from_args_builds_typed_config(self):
        cfg = GraphFusionDatasetConfig.from_args(argparse.Namespace(skip_n_last=3))
        assert isinstance(cfg, GraphFusionDatasetConfig)
        assert cfg.skip_n_last == 3


# ---------------------------------------------------------------------------
# Hierarchical-graph SID method + trainer configs (recdmon / spectral / stacked_dmon)
# ---------------------------------------------------------------------------


class TestRKMeansModelConfig:
    def _args(self, **kwargs):
        base = dict(
            num_emb_list=None,
            n_clusters=256,
            depth=4,
            distance_function="euclidean",
            mean_function="mean",
            norm_kmeans_input=False,
            fit_mini_batch=False,
            mini_batch_epochs=100,
            init_batch_size=256,
            max_steps=None,
            init_mode="kmeans++",
        )
        base.update(kwargs)
        return argparse.Namespace(**base)

    def test_from_args_parses_kmeans_backend_and_kmeanspp_exact(self):
        cfg = RKMeansModelConfig.from_args(self._args(kmeans_backend="faiss", kmeanspp_exact=True))
        assert cfg.kmeans_backend == "faiss"
        assert cfg.kmeanspp_exact is True

    def test_from_args_defaults(self):
        cfg = RKMeansModelConfig.from_args(self._args())
        assert cfg.kmeans_backend == "torch"
        assert cfg.kmeanspp_exact is False

    def test_hierarchical_from_args_parses_new_fields(self):
        # regression: HierarchicalRKMeansModelConfig.from_args used to crash on
        # cls.n_clusters (default_factory field has no class attribute)
        cfg = HierarchicalRKMeansModelConfig.from_args(
            self._args(n_clusters=[32, 32], depth=2, kmeans_backend="faiss", kmeanspp_exact=True)
        )
        assert cfg.n_clusters == [32, 32]
        assert cfg.depth == 2
        assert cfg.kmeans_backend == "faiss"
        assert cfg.kmeanspp_exact is True

    def test_hierarchical_from_args_default_n_clusters_fallback(self):
        cfg = HierarchicalRKMeansModelConfig.from_args(argparse.Namespace(num_emb_list=None, depth=4))
        assert cfg.n_clusters == [256, 256, 256, 256]

    def test_kmeans_backend_defaults_on_configs(self):
        assert RKMeansModelConfig().kmeans_backend == "torch"
        assert HierarchicalRKMeansModelConfig().kmeans_backend == "torch"
        assert RKMeansModelConfig().kmeanspp_exact is False
        assert HierarchicalRKMeansModelConfig().kmeanspp_exact is False


class TestHierarchicalGraphModelConfig:
    @pytest.mark.parametrize(
        ("cfg_class", "expected_defaults"),
        [
            (
                RecDMoNModelConfig,
                {
                    "name": "recdmon",
                    "recursive_epochs": 1000,
                    "assign_hidden_dim": 256,
                    "dropout": 0.0,
                    "max_dense_nodes": 100000,
                    "dmon_balance_weight": 1.0,
                    "dmon_entropy_weight": 0.01,
                    "lr": 3e-4,
                    "weight_decay": 1e-4,
                    "recursive_factor": 6,
                    "recursive_levels": 4,
                    "grad_clip": 5.0,
                },
            ),
            (
                SpectralModelConfig,
                {
                    "name": "spectral",
                    "seed": None,
                    "recursive_factor": 6,
                    "recursive_levels": 4,
                },
            ),
            (
                StackedDMoNModelConfig,
                {
                    "name": "stacked_dmon",
                    "in_dim": 768,
                    "num_emb_list": (256, 256, 256, 256),
                    "e_dim": 32,
                    "dmon_hidden_dim": 256,
                    "assign_hidden_dim": 256,
                    "dropout": 0.0,
                    "dmon_balance_weight": 1.0,
                    "dmon_entropy_weight": 0.01,
                },
            ),
        ],
    )
    def test_defaults(self, cfg_class, expected_defaults):
        cfg = cfg_class()
        for attr, expected in expected_defaults.items():
            assert getattr(cfg, attr) == expected

    @pytest.mark.parametrize(
        "cfg_class",
        [RecDMoNModelConfig, SpectralModelConfig, StackedDMoNModelConfig],
    )
    def test_to_kwargs_drops_name_and_round_trips(self, cfg_class):
        cfg = cfg_class()
        assert "name" not in cfg.to_kwargs()
        assert set(cfg.to_kwargs()) == set(cfg.__dict__) - {"name"}
        assert cfg_class(**cfg.to_kwargs()) == cfg

    def test_from_args_builds_typed_config(self):
        cfg = RecDMoNModelConfig.from_args(argparse.Namespace(recursive_factor=3, max_dense_nodes=512))
        assert isinstance(cfg, RecDMoNModelConfig)
        assert cfg.recursive_factor == 3
        assert cfg.max_dense_nodes == 512

    def test_registered_sid_method_configs_resolve(self):
        # registry: name -> SIDMethod (sid_method group), name -> ModelConfig (config).
        for name, expected in {
            "recdmon": RecDMoNModelConfig,
            "spectral": SpectralModelConfig,
            "stacked_dmon": StackedDMoNModelConfig,
        }.items():
            assert REGISTRY.get_sid_method_config(name) is expected
            assert REGISTRY.get_sid_method(name).name == name


class TestHierarchicalGraphTrainerConfig:
    @pytest.mark.parametrize(
        ("cfg_class", "expected_defaults"),
        [
            (
                RecursiveClusteringTrainerConfig,
                {
                    "name": "recursive_clustering",
                    "seed": None,
                    "lr": 1e-3,
                    "learner": "AdamW",
                    "weight_decay": 1e-4,
                    "epochs": 20000,
                    "eval_step": 2000,
                    "compile": False,
                },
            ),
            (
                StackedDMoNTrainerConfig,
                {
                    "name": "stacked_dmon",
                    "seed": None,
                    "grad_clip": 5.0,
                    "lr": 1e-3,
                    "learner": "AdamW",
                    "weight_decay": 1e-4,
                    "epochs": 20000,
                    "eval_step": 2000,
                    "compile": False,
                },
            ),
        ],
    )
    def test_defaults(self, cfg_class, expected_defaults):
        cfg = cfg_class()
        for attr, expected in expected_defaults.items():
            assert getattr(cfg, attr) == expected

    @pytest.mark.parametrize(
        "cfg_class",
        [RecursiveClusteringTrainerConfig, StackedDMoNTrainerConfig],
    )
    def test_to_kwargs_drops_name_and_round_trips(self, cfg_class):
        cfg = cfg_class()
        assert "name" not in cfg.to_kwargs()
        assert set(cfg.to_kwargs()) == set(cfg.__dict__) - {"name"}
        assert cfg_class(**cfg.to_kwargs()) == cfg

    def test_registered_trainer_configs_resolve(self):
        for name, expected in {
            "recursive_clustering": RecursiveClusteringTrainerConfig,
            "stacked_dmon": StackedDMoNTrainerConfig,
        }.items():
            assert REGISTRY.get_trainer_config(name) is expected
            assert REGISTRY.get_trainer(name).name == name
