import json

import numpy as np
import polars as pl
import pytest
import torch

from gensid.config.graph_build_config import (
    AdjacentCoocGraphConfig,
    AdjacentCoocProdGraphConfig,
    RandomWalkGraphConfig,
    WindowedGraphConfig,
)
from gensid.data.graph import (
    InteractionGraphBuilder,
    analyze_graph_quick,
    build_adjacent_cooccurrence_graph,
    build_adjacent_cooccurrence_graph_prod_chunked,
    build_bipartite_rw_item_graph,
    build_windowed_cooccurrence_graph,
    compute_graph_spectral_vectors,
    get_graph_builder,
    polars_graph_to_pyg,
)
from gensid.registry import REGISTRY


@pytest.fixture
def interactions():
    return pl.DataFrame(
        {
            "user_id": [0, 0, 0, 1, 1, 1, 2],
            "ts": [1, 2, 3, 1, 2, 3, 1],
            "item_id": [1, 2, 3, 3, 4, 5, 6],
        }
    )


class TestAdjacentCoocGraph:
    def test_builds_adjacent_pairs_with_weights(self, interactions):
        graph = build_adjacent_cooccurrence_graph(interactions, AdjacentCoocGraphConfig())
        rows = {tuple((src, dst)): w for src, dst, w in graph.select(["src", "dst", "weight"]).iter_rows()}
        assert (1, 2) in rows
        assert (2, 3) in rows
        # (3,4) and (4,5) from user 1
        assert (3, 4) in rows
        assert (4, 5) in rows
        assert rows[(1, 2)] == 1

    def test_sorted_descending_by_weight(self, interactions):
        graph = build_adjacent_cooccurrence_graph(interactions, AdjacentCoocGraphConfig())
        weights = graph["weight"].to_list()
        assert weights == sorted(weights, reverse=True)

    def test_exact_edges_and_weights(self, interactions):
        # user 0: [1,2,3] -> (1,2),(2,3); user 1: [3,4,5] -> (3,4),(4,5); user 2: [6] -> no pair
        graph = build_adjacent_cooccurrence_graph(interactions, AdjacentCoocGraphConfig())
        edges = {(src, dst): w for src, dst, w in graph.select(["src", "dst", "weight"]).iter_rows()}
        assert edges == {(1, 2): 1, (2, 3): 1, (3, 4): 1, (4, 5): 1}

    def test_repeated_pairs_aggregate_weight(self):
        # (1,2) occurs once for user 0 and once for user 1 -> weight 2
        df = pl.DataFrame(
            {
                "user_id": [0, 0, 0, 1, 1],
                "ts": [1, 2, 3, 1, 2],
                "item_id": [1, 2, 3, 1, 2],
            }
        )
        graph = build_adjacent_cooccurrence_graph(df, AdjacentCoocGraphConfig())
        edges = {(src, dst): w for src, dst, w in graph.select(["src", "dst", "weight"]).iter_rows()}
        assert edges == {(1, 2): 2, (2, 3): 1}


class TestWindowedGraph:
    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"window_size": 2}, id="default-window"),
            pytest.param({"window_size": 2, "decay": "inverse_distance"}, id="inverse-distance-decay"),
            pytest.param({"window_size": 2, "dedup_within_window": True}, id="dedup-within-window"),
        ],
    )
    def test_runs_for_config_variants(self, interactions, kwargs):
        graph = build_windowed_cooccurrence_graph(interactions, WindowedGraphConfig(**kwargs))
        assert {"src", "dst", "weight"} == set(graph.columns)
        assert graph.height > 0

    def test_bidirectional_duplicates_edges(self, interactions):
        graph = build_windowed_cooccurrence_graph(interactions, WindowedGraphConfig(window_size=1, bidirectional=True))
        rows = {tuple(r) for r in graph.select(["src", "dst"]).iter_rows()}
        assert (1, 2) in rows and (2, 1) in rows

    def test_exact_window_size_one_edges(self, interactions):
        # one hop ahead only: (1,2),(2,3) from user 0; (3,4),(4,5) from user 1
        graph = build_windowed_cooccurrence_graph(interactions, WindowedGraphConfig(window_size=1))
        edges = {(src, dst): w for src, dst, w in graph.select(["src", "dst", "weight"]).iter_rows()}
        assert edges == {(1, 2): 1, (2, 3): 1, (3, 4): 1, (4, 5): 1}

    def test_inverse_distance_decay_weights(self, interactions):
        # window=2, decay=1/k: one-hop pairs weigh 1, two-hop pairs weigh 1/2
        graph = build_windowed_cooccurrence_graph(
            interactions, WindowedGraphConfig(window_size=2, decay="inverse_distance")
        )
        edges = {(src, dst): w for src, dst, w in graph.select(["src", "dst", "weight"]).iter_rows()}
        assert edges == {
            (1, 2): 1,
            (2, 3): 1,
            (3, 4): 1,
            (4, 5): 1,
            (1, 3): 0.5,  # user 0: two hops 1 -> 3
            (3, 5): 0.5,  # user 1: two hops 3 -> 5
        }

    def test_bidirectional_exact_edges(self, interactions):
        graph = build_windowed_cooccurrence_graph(interactions, WindowedGraphConfig(window_size=1, bidirectional=True))
        edges = {(src, dst): w for src, dst, w in graph.select(["src", "dst", "weight"]).iter_rows()}
        assert edges == {
            (1, 2): 1,
            (2, 3): 1,
            (3, 4): 1,
            (4, 5): 1,
            (2, 1): 1,
            (3, 2): 1,
            (4, 3): 1,
            (5, 4): 1,
        }

    def test_window_size_zero_raises(self, interactions):
        with pytest.raises(ValueError, match="window_size"):
            build_windowed_cooccurrence_graph(interactions, WindowedGraphConfig(window_size=0))


class TestBipartiteRWGraph:
    def test_builds_item_item_edges(self, interactions):
        graph = build_bipartite_rw_item_graph(
            interactions,
            RandomWalkGraphConfig(walk_length=3, walks_per_item=2, window_size=2),
        )
        assert set(graph.columns) == {"src", "dst", "weight"}

    def test_missing_columns_raises(self):
        df = pl.DataFrame({"user_id": [0], "item_id": [1]})
        with pytest.raises(ValueError, match="missing columns"):
            build_bipartite_rw_item_graph(df, RandomWalkGraphConfig())

    def test_empty_interactions_returns_empty_frame(self):
        df = pl.DataFrame({"user_id": [], "item_id": [], "ts": []})
        graph = build_bipartite_rw_item_graph(df, RandomWalkGraphConfig(walk_length=2, walks_per_item=1, window_size=1))
        assert graph.height == 0

    def test_self_loops_controlled_by_flag(self):
        # single user, single item: every walk is item->user->item, so only (1,1) exists
        df = pl.DataFrame({"user_id": [0], "ts": [1], "item_id": [1]})
        with_loops = build_bipartite_rw_item_graph(
            df, RandomWalkGraphConfig(walk_length=4, walks_per_item=1, window_size=1, self_loops=True)
        )
        without = build_bipartite_rw_item_graph(
            df, RandomWalkGraphConfig(walk_length=4, walks_per_item=1, window_size=1)
        )
        assert {tuple(r) for r in with_loops.select(["src", "dst"]).iter_rows()} == {(1, 1)}
        assert without.height == 0

    def test_edges_only_between_co_occurring_items(self, interactions):
        # consecutive items on a walk always share a user, so every edge must be a
        # pair of items that co-occur in some user's item set.
        graph = build_bipartite_rw_item_graph(
            interactions,
            RandomWalkGraphConfig(walk_length=6, walks_per_item=20, window_size=1),
        )
        # user 0: {1,2,3}, user 1: {3,4,5} -> item pairs that share a user, both directions
        cooccurring = {
            (a, b) for pair in [(1, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 5)] for a, b in (pair, pair[::-1])
        }
        rows = {tuple(r) for r in graph.select(["src", "dst"]).iter_rows()}
        assert rows <= cooccurring
        assert len(rows) > 0


class TestAdjacentProdGraph:
    def _write_interactions(self, tmp_path):
        df = pl.DataFrame(
            {
                "history_item_id": [[0, 1, 2], [1, 2, 3], [2, 3]],
            }
        )
        out_dir = tmp_path / "inter"
        out_dir.mkdir()
        df.write_parquet(str(out_dir / "part0.parquet"))
        df.write_parquet(str(out_dir / "part1.parquet"))
        return str(out_dir)

    def _write_emb_map(self, tmp_path):
        df = pl.DataFrame(
            {
                "item_id": [0, 1, 2, 3],
                "normalized_id": [10, 11, 12, 13],
            }
        )
        path = tmp_path / "emb_map.parquet"
        df.write_parquet(str(path))
        return str(path)

    def test_builds_graph_and_num_items(self, tmp_path):
        folder = self._write_interactions(tmp_path)
        embeddings_path = self._write_emb_map(tmp_path)
        cfg = AdjacentCoocProdGraphConfig(
            parquet_folder=folder, embeddings_path=embeddings_path, directed=True, chronological=True
        )
        graph, num_items = build_adjacent_cooccurrence_graph_prod_chunked(cfg)
        assert num_items == 4
        assert {"src", "dst", "weight"} <= set(graph.columns)
        # normalized ids are in [10,13] range
        assert graph["src"].min() >= 10 and graph["dst"].max() <= 13
        assert graph.height > 0

    def test_undirected_canonicalizes(self, tmp_path):
        folder = self._write_interactions(tmp_path)
        embeddings_path = self._write_emb_map(tmp_path)
        cfg = AdjacentCoocProdGraphConfig(
            parquet_folder=folder, embeddings_path=embeddings_path, directed=False, chronological=True
        )
        graph, _ = build_adjacent_cooccurrence_graph_prod_chunked(cfg)
        assert graph.height > 0

    def test_exact_normalized_edges(self, tmp_path):
        # single file with sequences [0,1,2], [1,2,3], [2,3] -> adjacent counts
        # (0,1):1, (1,2):2, (2,3):2, mapped through emb_map {0:10, 1:11, 2:12, 3:13}
        folder = tmp_path / "inter"
        folder.mkdir()
        pl.DataFrame({"history_item_id": [[0, 1, 2], [1, 2, 3], [2, 3]]}).write_parquet(str(folder / "part0.parquet"))
        embeddings_path = self._write_emb_map(tmp_path)
        cfg = AdjacentCoocProdGraphConfig(
            parquet_folder=str(folder), embeddings_path=embeddings_path, directed=True, chronological=True
        )
        graph, num_items = build_adjacent_cooccurrence_graph_prod_chunked(cfg)
        assert num_items == 4
        edges = {(src, dst): w for src, dst, w in graph.select(["src", "dst", "weight"]).iter_rows()}
        assert edges == {(10, 11): 1, (11, 12): 2, (12, 13): 2}

    def test_no_embeddings_path_raises(self, tmp_path):
        folder = self._write_interactions(tmp_path)
        with pytest.raises(ValueError, match="embeddings_path"):
            build_adjacent_cooccurrence_graph_prod_chunked(AdjacentCoocProdGraphConfig(parquet_folder=folder))

    def test_bad_files_per_chunk_raises(self, tmp_path):
        folder = self._write_interactions(tmp_path)
        cfg = AdjacentCoocProdGraphConfig(parquet_folder=folder, embeddings_path="x.parquet", files_per_chunk=0)
        with pytest.raises(ValueError, match="files_per_chunk"):
            build_adjacent_cooccurrence_graph_prod_chunked(cfg)

    def test_no_parquet_files_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        cfg = AdjacentCoocProdGraphConfig(parquet_folder=str(empty), embeddings_path="x.parquet")
        with pytest.raises(FileNotFoundError):
            build_adjacent_cooccurrence_graph_prod_chunked(cfg)


class TestSpectralVectors:
    def test_path_graph_returns_k_vectors(self):
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        vecs = compute_graph_spectral_vectors(edge_index, torch.ones(4), num_nodes=5, k=2)
        assert vecs.shape == (5, 2)
        assert torch.isfinite(torch.from_numpy(vecs)).all()

    def test_k_zero_returns_empty(self):
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        vecs = compute_graph_spectral_vectors(edge_index, torch.ones(2), num_nodes=3, k=0)
        assert vecs.shape == (3, 0)

    def test_small_graph_returns_zeros(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        vecs = compute_graph_spectral_vectors(edge_index, None, num_nodes=2, k=2)
        assert vecs.shape == (2, 2)
        assert (vecs == 0).all()

    def test_none_edge_weight_uses_unit_weights(self):
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        vecs = compute_graph_spectral_vectors(edge_index, None, num_nodes=4, k=1)
        assert vecs.shape == (4, 1)

    def test_path4_matches_analytic_eigenvectors(self):
        # Path P4 (deg=[1,2,2,1]); symmetric normalized Laplacian has closed-form
        # non-trivial unit eigenvectors (1,1,-1,-1)/sqrt(6) and (1,-1,-1,1)/sqrt(6),
        # i.e. [1/sqrt3, 1/sqrt6, -1/sqrt6, -1/sqrt3] and its reflection.
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        vecs = compute_graph_spectral_vectors(edge_index, torch.ones(3), num_nodes=4, k=2)
        expected = torch.tensor(
            [
                [1 / np.sqrt(3), 1 / np.sqrt(3)],
                [1 / np.sqrt(6), -1 / np.sqrt(6)],
                [-1 / np.sqrt(6), -1 / np.sqrt(6)],
                [-1 / np.sqrt(3), 1 / np.sqrt(3)],
            ],
            dtype=torch.float32,
        )
        # eigsh returns unit-norm eigenvectors; sign is arbitrary, so compare abs values
        assert np.allclose(np.abs(vecs), np.abs(expected), atol=1e-6)

    def test_returned_vectors_are_eigenvectors(self):
        # For any graph, each returned column must satisfy L_norm v = lambda v and be
        # orthonormal to the other returned columns.
        num_nodes = 15
        edge_index = torch.tensor([[i, (i + 1) % num_nodes] for i in range(num_nodes)]).T  # cycle
        chords = torch.tensor([[0, 5, 7], [8, 13, 11]])
        edge_index = torch.cat([edge_index, chords], dim=1)
        vecs = compute_graph_spectral_vectors(edge_index, torch.ones(edge_index.shape[1]), num_nodes=num_nodes, k=4)

        A = np.zeros((num_nodes, num_nodes))
        A[edge_index[0].numpy(), edge_index[1].numpy()] = 1.0
        A = 0.5 * (A + A.T)
        deg = A.sum(axis=1)
        dinv = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
        laplacian = np.eye(num_nodes) - dinv @ A @ dinv

        nonzero = [col for col in vecs.T if np.any(col)]
        assert len(nonzero) == 4
        nonzero = np.stack(nonzero, axis=1)
        assert np.allclose(nonzero.T @ nonzero, np.eye(4), atol=1e-6)  # orthonormal
        for col in nonzero.T:
            lam = col @ (laplacian @ col) / (col @ col)
            assert np.allclose(laplacian @ col, lam * col, atol=1e-5)


class TestGetGraphBuilder:
    def test_resolves_every_registered_graph_type(self):
        # Contract the Registry must keep: every graph type known to the config side
        # resolves to a callable builder. Identity is deliberately not pinned.
        assert REGISTRY.keys("graph_building_config")
        for graph_type in REGISTRY.keys("graph_building_config"):
            assert callable(get_graph_builder(graph_type))  # type: ignore[arg-type]

    def test_unknown_graph_type_raises(self):
        with pytest.raises(KeyError):
            get_graph_builder("bogus")  # type: ignore[arg-type]


class TestPolarsGraphToPyg:
    def _edge_df(self):
        return pl.DataFrame({"src": [0, 1, 2], "dst": [1, 2, 0], "weight": [1.0, 2.0, 3.0]})

    def test_num_items_node_mapping(self):
        data, node_map = polars_graph_to_pyg(self._edge_df(), num_items=4)
        assert data.num_nodes == 4
        assert node_map.height == 4
        # make_directed=False by default -> 3 directed edges become 6
        assert data.edge_index.shape == (2, 6)
        assert data.edge_weight.shape == (6,)
        assert torch.isfinite(data.x).all()

    def test_make_directed_adds_reverse_edges(self):
        data_undir, _ = polars_graph_to_pyg(self._edge_df(), make_directed=False, num_items=4)
        data_dir, _ = polars_graph_to_pyg(self._edge_df(), make_directed=True, num_items=4)
        assert data_undir.edge_index.shape[1] == 2 * data_dir.edge_index.shape[1]

    def test_exact_edges_weights_and_features(self):
        # undirected: (0,1,1),(0,2,3),(1,0,1),(1,2,2),(2,0,3),(2,1,2) sorted by (src,dst)
        data, node_map = polars_graph_to_pyg(self._edge_df(), num_items=4, make_directed=False)
        assert node_map["item_id"].to_list() == [0, 1, 2, 3]
        assert data.edge_index.tolist() == [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]
        assert data.edge_weight.tolist() == [1.0, 3.0, 1.0, 2.0, 3.0, 2.0]
        # x = log1p(out-degree): deg(0)=1+3=4, deg(1)=1+2=3, deg(2)=3+2=5, deg(3)=0
        expected_x = torch.log1p(torch.tensor([4.0, 3.0, 5.0, 0.0])).unsqueeze(-1)
        assert torch.allclose(data.x, expected_x)

    def test_parallel_edges_are_merged(self):
        df = pl.DataFrame({"src": [0, 0, 1], "dst": [1, 1, 2], "weight": [1.0, 2.0, 4.0]})
        data, _ = polars_graph_to_pyg(df, num_items=3, make_directed=True)
        # parallel (0,1) edges merge to weight 3, then sorted by (src,dst)
        assert data.edge_index.tolist() == [[0, 1], [1, 2]]
        assert data.edge_weight.tolist() == [3.0, 4.0]

    def test_all_item_ids_mapping(self):
        # edges must reference items inside the supplied universe
        df = pl.DataFrame({"src": [5, 6, 7], "dst": [6, 7, 5], "weight": [1.0, 2.0, 3.0]})
        data, node_map = polars_graph_to_pyg(df, all_item_ids=[5, 6, 7], make_directed=True)
        assert data.num_nodes == 3
        assert node_map["item_id"].to_list() == [5, 6, 7]

    def test_identity_features_are_one_hot(self):
        data, _ = polars_graph_to_pyg(self._edge_df(), add_identity_features=True, num_items=4)
        assert torch.equal(data.x, torch.eye(4))

    def test_labels_are_mapped_onto_items(self):
        labels = pl.DataFrame({"item_id": [0, 1, 2, 3], "y": [7, 8, 9, 10]})
        data, _ = polars_graph_to_pyg(self._edge_df(), labels_df=labels, num_items=4)
        assert torch.equal(data.y, torch.tensor([7, 8, 9, 10]))

    def test_missing_label_raises(self):
        labels = pl.DataFrame({"item_id": [0], "y": [7]})
        with pytest.raises(ValueError, match="no label"):
            polars_graph_to_pyg(self._edge_df(), labels_df=labels, num_items=4)

    def test_unknown_item_raises(self):
        df = pl.DataFrame({"src": [0, 99], "dst": [1, 2], "weight": [1.0, 1.0]})
        with pytest.raises(ValueError, match="not present in provided item universe"):
            polars_graph_to_pyg(df, num_items=4)

    def test_negative_item_ids_raise(self):
        df = pl.DataFrame({"src": [-1], "dst": [1], "weight": [1.0]})
        with pytest.raises(ValueError, match="non-negative"):
            polars_graph_to_pyg(df, all_item_ids=[-1, 1])

    def test_neither_num_items_nor_all_item_ids_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            polars_graph_to_pyg(self._edge_df())

    def test_num_items_must_be_positive(self):
        with pytest.raises(ValueError, match="num_items"):
            polars_graph_to_pyg(self._edge_df(), num_items=0)

    def test_empty_edges(self):
        df = pl.DataFrame({"src": [], "dst": [], "weight": []})
        data, node_map = polars_graph_to_pyg(df, num_items=2)
        assert data.edge_index.shape == (2, 0)
        assert node_map.height == 2


class TestAnalyzeGraphQuick:
    def _edge_df(self):
        return pl.DataFrame({"src": [0, 1, 2], "dst": [1, 2, 0], "weight": [1.0, 2.0, 3.0]})

    def test_undirected_stats(self):
        stats = analyze_graph_quick(self._edge_df(), directed=False)
        assert stats["num_nodes"] == 3
        assert stats["num_edges"] > 0
        assert isinstance(stats["avg_degree"], float)
        assert 0.0 <= stats["density"] <= 1.0
        assert stats["max_degree"] >= 1

    def test_missing_weight_col_defaults_to_one(self):
        df = pl.DataFrame({"src": [0, 1], "dst": [1, 2]})
        stats = analyze_graph_quick(df, directed=True)
        assert stats["num_edges"] == 2

    def test_directed_in_and_out_degree(self):
        stats = analyze_graph_quick(self._edge_df(), directed=True)
        assert stats["num_nodes"] == 3
        assert stats["num_edges"] == 3

    def test_exact_undirected_stats(self):
        # undirected triangle 0-1-2-0: 3 nodes, 3 edges, every degree 2, density 1
        stats = analyze_graph_quick(self._edge_df(), directed=False)
        assert stats["num_nodes"] == 3
        assert stats["num_edges"] == 3
        assert stats["avg_degree"] == pytest.approx(2.0)
        assert stats["median_degree"] == pytest.approx(2.0)
        assert stats["max_degree"] == 2
        assert stats["isolated_nodes"] == 0
        assert stats["density"] == pytest.approx(1.0)
        assert stats["isolated_ratio"] == pytest.approx(0.0)

    def test_exact_directed_stats(self):
        # directed cycle 0->1->2->0: 3 edges among 3 nodes, density 3/(3*2)=0.5
        stats = analyze_graph_quick(self._edge_df(), directed=True)
        assert stats["num_nodes"] == 3
        assert stats["num_edges"] == 3
        assert stats["avg_degree"] == pytest.approx(2.0)
        assert stats["density"] == pytest.approx(0.5)


class TestInteractionGraphBuilder:
    def _write_interactions(self, tmp_path):
        path = tmp_path / "inter.json"
        path.write_text(json.dumps({"u0": [0, 1, 2, 3], "u1": [1, 2, 4]}))
        return str(path)

    def test_lazy_build_populates_artifacts(self, tmp_path):
        builder = InteractionGraphBuilder(
            interactions_path=self._write_interactions(tmp_path),
            graph_building_config=AdjacentCoocGraphConfig(),
        )
        assert builder._pyg_data is None  # lazy
        data = builder.pyg_data
        assert data.edge_index.ndim == 2
        assert builder.graph_df.height > 0
        assert builder.node_map.height > 0
        assert set(builder.node_map.columns) == {"item_id", "node_id"}

    def test_make_directed(self, tmp_path):
        undir = InteractionGraphBuilder(
            interactions_path=self._write_interactions(tmp_path),
            graph_building_config=AdjacentCoocGraphConfig(make_directed=False),
        )
        dir_ = InteractionGraphBuilder(
            interactions_path=self._write_interactions(tmp_path),
            graph_building_config=AdjacentCoocGraphConfig(make_directed=True),
        )
        assert undir.pyg_data.edge_index.shape[1] == 2 * dir_.pyg_data.edge_index.shape[1]

    def test_exact_build_from_known_interactions(self, tmp_path):
        # skip last 2 per user: u0 -> [0,1], u1 -> [1]; adjacent pair (0,1);
        # all_item_ids=[0,1] -> node map {0:0, 1:1}; undirected edge (0,1)+(1,0)
        builder = InteractionGraphBuilder(
            interactions_path=self._write_interactions(tmp_path),
            graph_building_config=AdjacentCoocGraphConfig(),
        )
        data = builder.pyg_data
        assert builder.node_map["item_id"].to_list() == [0, 1]
        assert data.num_nodes == 2
        assert data.edge_index.tolist() == [[0, 1], [1, 0]]
        assert data.edge_weight.tolist() == [1.0, 1.0]
        assert data.x[:, 0].tolist() == pytest.approx([np.log1p(1.0), np.log1p(1.0)])

    def test_identity_features(self, tmp_path):
        builder = InteractionGraphBuilder(
            interactions_path=self._write_interactions(tmp_path),
            graph_building_config=AdjacentCoocGraphConfig(add_identity_features=True),
        )
        n = builder.node_map.height
        assert builder.pyg_data.x.shape == (n, n)

    def test_num_items_sets_node_count(self, tmp_path):
        # with skip_n_last=None, all_item_ids is not derived, so num_items defines the universe
        builder = InteractionGraphBuilder(
            interactions_path=self._write_interactions(tmp_path),
            graph_building_config=AdjacentCoocGraphConfig(),
            num_items=6,
            skip_n_last=None,
        )
        assert builder.pyg_data.num_nodes == 6

    def test_missing_interactions_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            InteractionGraphBuilder(
                interactions_path=str(tmp_path / "nope.json"),
                graph_building_config=AdjacentCoocGraphConfig(),
            ).pyg_data

    def test_no_interactions_path_raises(self, tmp_path):
        builder = InteractionGraphBuilder(
            interactions_path=None,
            graph_building_config=AdjacentCoocGraphConfig(),
        )
        with pytest.raises(ValueError, match="Cannot build graph"):
            _ = builder.pyg_data
