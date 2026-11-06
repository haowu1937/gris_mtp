import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import eigsh
from torch_geometric.data import Data
from tqdm import tqdm

from gensid.config.graph_build_config import (
    AdjacentCoocGraphConfig,
    AdjacentCoocProdGraphConfig,
    GraphBuildingConfig,
    GraphType,
    RandomWalkGraphConfig,
    WindowedGraphConfig,
)
from gensid.data.utils import load_train_item_mask

logger = logging.getLogger(__name__)


def compute_graph_spectral_vectors(edge_index, edge_weight, num_nodes: int, k: int) -> np.ndarray:
    """
    Compute the first k non-trivial spectral eigenvectors of the normalized graph Laplacian.

    This function constructs the symmetric normalized Laplacian matrix L = I - D^{-1/2} A D^{-1/2}
    from the given graph structure and computes its smallest k+1 eigenvalues and corresponding
    eigenvectors. The trivial eigenvector (associated with the smallest eigenvalue, typically 0
    for connected graphs) is discarded, and the next k eigenvectors are returned.

    Parameters
    ----------
    edge_index : torch.Tensor
        A tensor of shape (2, num_edges) containing the source and target node indices
        of the graph edges. Expected to be on CPU or GPU; will be moved to CPU for processing.
    edge_weight : torch.Tensor or None
        A tensor of shape (num_edges,) containing the weights of the edges. If None,
        all edges are assumed to have unit weight. Expected to be on CPU or GPU;
        will be moved to CPU for processing.
    num_nodes : int
        The total number of nodes in the graph.
    k : int
        The number of non-trivial spectral eigenvectors to compute. If k <= 0, an empty
        array of shape (num_nodes, 0) is returned.

    Returns
    -------
    np.ndarray
        A numpy array of shape (num_nodes, k) containing the first k non-trivial eigenvectors
        of the normalized Laplacian, sorted by ascending eigenvalue. Each column corresponds
        to one eigenvector. If fewer than k eigenvectors are available (e.g., due to small
        graph size or computation failure), the result is padded with zeros. In case of
        computational failure, a warning is logged and an array of zeros is returned.

    Notes
    -----
    - The adjacency matrix is symmetrized to ensure the Laplacian is symmetric.
    - The normalized Laplacian is used, which is defined as L = I - D^{-1/2} A D^{-1/2},
      where D is the degree matrix.
    - The function uses scipy.sparse.linalg.eigsh to compute the smallest eigenvalues and
      eigenvectors efficiently for sparse matrices.
    - For graphs with num_nodes <= 2, the function returns an array of zeros since there
      are no non-trivial eigenvectors.
    - If eigsh fails (e.g., due to numerical issues), a warning is logged and zeros are returned.
    """
    if k <= 0:
        return np.zeros((num_nodes, 0), dtype=np.float32)

    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    if edge_weight is None:
        val = np.ones(len(row), dtype=np.float64)
    else:
        val = edge_weight.cpu().numpy().astype(np.float64)

    A = sp.coo_matrix((val, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    A = 0.5 * (A + A.T)

    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    deg = np.maximum(deg, 1e-12)
    inv_sqrt_deg = 1.0 / np.sqrt(deg)

    D_inv_sqrt = sp.diags(inv_sqrt_deg)
    L = sp.eye(num_nodes, format="csr") - D_inv_sqrt @ A @ D_inv_sqrt

    # ask for one extra eigenvector, then drop the trivial one
    k_eff = min(k + 1, max(num_nodes - 1, 1))

    if num_nodes <= 2:
        return np.zeros((num_nodes, k), dtype=np.float32)

    try:
        evals, evecs = eigsh(L, k=k_eff, which="SM")
        order = np.argsort(evals)
        evecs = evecs[:, order]
    except Exception as e:  # noqa: BLE001
        logger.warning("eigsh failed, returning zeros for spectral vectors: %s", e)
        return np.zeros((num_nodes, k), dtype=np.float32)

    # drop the first trivial eigenvector
    evecs = evecs[:, 1 : 1 + k]

    if evecs.shape[1] < k:
        pad = np.zeros((num_nodes, k - evecs.shape[1]), dtype=np.float32)
        evecs = np.concatenate([evecs.astype(np.float32), pad], axis=1)
    else:
        evecs = evecs.astype(np.float32)

    return evecs


def build_adjacent_cooccurrence_graph(df: pl.DataFrame, config: AdjacentCoocGraphConfig) -> pl.DataFrame:
    """
    Build a directed co-occurrence graph from sequential data using adjacent pairs only.

    Input columns required:
        - user_id
        - ts
        - item_id

    Returns:
        DataFrame with columns:
        - src
        - dst
        - weight
    """
    df = df.sort(["user_id", "ts"])

    out = (
        df.with_columns(pl.col("item_id").shift(-1).over("user_id").alias("next_item"))
        .filter(pl.col("next_item").is_not_null())
        .group_by(["item_id", "next_item"])
        .agg(pl.len().alias("weight"))
        .rename(
            {
                "item_id": "src",
                "next_item": "dst",
            }
        )
        .sort("weight", descending=True)
    )

    return out


def build_adjacent_cooccurrence_graph_prod_chunked(config: AdjacentCoocProdGraphConfig) -> tuple[pl.DataFrame, int]:
    """
    Build an adjacent item-item co-occurrence graph using map-reduce.

    Map phase:
        - Read at most `files_per_chunk` interaction files.
        - Build and aggregate edges within that chunk.
        - Write partial edge counts to temporary Parquet files.

    Reduce phase:
        - Scan all partial edge files.
        - Sum weights for identical (src, dst) edges.
        - Map original item IDs to normalized IDs.

    Parameters
    ----------
    config:
        Config dataclass containing the required parameters build the graph

    """
    if config.embeddings_path is None:
        raise ValueError("embeddings_path must be provided")

    if config.files_per_chunk <= 0:
        raise ValueError("files_per_chunk must be greater than 0")

    if config.max_files is not None and config.max_files <= 0:
        raise ValueError("max_files must be greater than 0 or None")

    parquet_files = sorted(Path(config.parquet_folder).glob("*.parquet"))

    if config.max_files is not None:
        parquet_files = parquet_files[: config.max_files]

    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {config.parquet_folder!r}")

    allowed_values = None if config.allowed_filter_values is None else list(config.allowed_filter_values)

    def build_partial_graph(files: list[str]) -> pl.LazyFrame:
        cols = [config.seq_col]
        if config.filter_col is not None:
            cols.append(config.filter_col)

        lf = pl.scan_parquet(files).select(cols)

        if config.filter_col is not None:
            # row_id only needs to be unique inside the current map chunk.
            lf = lf.with_row_index("row_id").explode([config.seq_col, config.filter_col])

            if allowed_values is not None:
                lf = lf.filter(pl.col(config.filter_col).is_in(allowed_values))

            # Rebuild each filtered sequence while preserving sequence order.
            lf = (
                lf.filter(pl.col(config.seq_col).is_not_null())
                .group_by("row_id", maintain_order=True)
                .agg(pl.col(config.seq_col))
            )

        # Ignore null, empty, and one-item sequences.
        lf = lf.filter(pl.col(config.seq_col).is_not_null() & (pl.col(config.seq_col).list.len() >= 2))

        if not config.chronological:
            lf = lf.with_columns(pl.col(config.seq_col).list.reverse())

        edges = (
            lf.with_columns(
                pl.col(config.seq_col)
                .list.slice(
                    0,
                    pl.col(config.seq_col).list.len() - 1,
                )
                .alias("src"),
                pl.col(config.seq_col).list.slice(1).alias("dst"),
            )
            .select("src", "dst")
            .explode(["src", "dst"])
            .filter(pl.col("src").is_not_null() & pl.col("dst").is_not_null() & (pl.col("src") != pl.col("dst")))
        )

        if not config.directed:
            edges = edges.select(
                pl.min_horizontal("src", "dst").alias("src"),
                pl.max_horizontal("src", "dst").alias("dst"),
            )

        return edges.group_by(["src", "dst"]).agg(pl.len().cast(pl.UInt64).alias("weight"))

    # TemporaryDirectory creates a unique child directory and removes it
    # after the final graph has been collected.
    with TemporaryDirectory(
        prefix="adjacent_graph_map_",
        dir=config.tmp_dir,
    ) as work_dir:
        partial_paths: list[str] = []

        # Map phase
        for chunk_idx, start in enumerate(
            tqdm(
                range(0, len(parquet_files), config.files_per_chunk),
                desc="Building graph chunks",
                unit="chunk",
                mininterval=5,
            )
        ):
            chunk_paths = [str(path) for path in parquet_files[start : start + config.files_per_chunk]]

            partial_path = Path(work_dir) / f"partial_edges_{chunk_idx:06d}.parquet"

            build_partial_graph(chunk_paths).sink_parquet(
                str(partial_path),
                compression="zstd",
            )

            partial_paths.append(str(partial_path))

        # Reduce phase: merge duplicate edges across map outputs.
        graph = pl.scan_parquet(partial_paths).group_by(["src", "dst"]).agg(pl.col("weight").sum().alias("weight"))

        # Normalize source and destination IDs independently.
        emb_map = pl.scan_parquet(config.embeddings_path).select(["item_id", "normalized_id"])

        src_map = emb_map.select(
            pl.col("item_id").alias("src"),
            pl.col("normalized_id").alias("normalized_src"),
        )

        dst_map = emb_map.select(
            pl.col("item_id").alias("dst"),
            pl.col("normalized_id").alias("normalized_dst"),
        )

        normalized_graph = (
            graph.join(src_map, on="src", how="inner")
            .join(dst_map, on="dst", how="inner")
            .select(
                pl.col("normalized_src").alias("src"),
                pl.col("normalized_dst").alias("dst"),
                "weight",
            )
        )

        # The final result is collected before temporary partial files
        # are automatically removed.
        pl_graph = normalized_graph.collect(engine="streaming")

    num_items = pl.scan_parquet(config.embeddings_path).select(pl.len()).collect(engine="streaming").item()

    return pl_graph, num_items


# def build_adjacent_cooccurrence_graph_prod(
#     parquet_folder: str,
#     seq_col: str = "history_item_id",
#     filter_col: str | None = None,
#     allowed_filter_values: list | set | None = None,
#     directed: bool = True,
#     chronological: bool = False,
#     embeddings_path: str = None,
# ):
#     cols = [seq_col] + ([filter_col] if filter_col else [])

#     lf = pl.scan_parquet(str(Path(parquet_folder) / "*.parquet")).select(cols).with_row_index("row_id")

#     if filter_col is not None:
#         lf = lf.explode([seq_col, filter_col])

#         if allowed_filter_values is not None:
#             lf = lf.filter(pl.col(filter_col).is_in(list(allowed_filter_values)))

#         # Rebuild filtered sequence per original row.
#         lf = (
#             lf.filter(pl.col(seq_col).is_not_null())
#             .group_by("row_id", maintain_order=True)
#             .agg(pl.col(seq_col).alias(seq_col))
#         )

#     if not chronological:
#         lf = lf.with_columns(pl.col(seq_col).list.reverse())
#     lf = (
#         lf.with_columns(
#             pl.col(seq_col).list.slice(0, pl.col(seq_col).list.len() - 1).alias("src"),
#             pl.col(seq_col).list.slice(1).alias("dst"),
#         )
#         .select("src", "dst")
#         .explode(["src", "dst"])
#         .filter(pl.col("src").is_not_null() & pl.col("dst").is_not_null() & (pl.col("src") != pl.col("dst")))
#     )

#     if not directed:
#         lf = lf.select(
#             pl.min_horizontal("src", "dst").alias("src"),
#             pl.max_horizontal("src", "dst").alias("dst"),
#         )

#     graph = lf.group_by(["src", "dst"]).agg(pl.len().alias("weight"))

#     # normalize ids
#     emb_map = pl.scan_parquet(embeddings_path)
#     id_map = emb_map.select(["normalized_id", "item_id"])
#     pl_graph = graph.join(id_map, left_on="src", right_on="item_id", suffix="_src", how="inner").join(
#         id_map, left_on="dst", right_on="item_id", suffix="_dst", how="inner"
#     )
#     pl_graph = pl_graph.collect(engine="streaming")
#     pl_graph = pl_graph["normalized_id", "normalized_id_dst", "weight"].rename(
#         {"normalized_id": "src", "normalized_id_dst": "dst", "weight": "weight"}
#     )

#     num_items = emb_map.select(pl.len()).collect().item()
#     return pl_graph, num_items


def build_bipartite_rw_item_graph(df: pl.DataFrame, config: RandomWalkGraphConfig) -> pl.DataFrame:
    """
    Build an item-item graph from plain random walks on a user-item bipartite graph.
    CPU only.
    """
    from torch_cluster import random_walk

    required = {"user_id", "ts", "item_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing columns: {missing}")

    ui = df.select(["user_id", "item_id"]).unique().sort(["user_id", "item_id"])

    if ui.height == 0:
        return pl.DataFrame(
            {
                "src": pl.Series([], dtype=pl.Int64),
                "dst": pl.Series([], dtype=pl.Int64),
                "weight": pl.Series([], dtype=pl.Float64),
            }
        )

    item_map = ui.select("item_id").unique().sort("item_id").with_row_index("item_node")
    user_map = ui.select("user_id").unique().sort("user_id").with_row_index("user_node")

    num_items = item_map.height
    user_offset = num_items

    user_map = user_map.with_columns((pl.col("user_node") + user_offset).alias("user_node"))

    edges = (
        ui.join(item_map, on="item_id", how="left")
        .join(user_map, on="user_id", how="left")
        .select(["item_id", "item_node", "user_node"])
    )

    item_to_user = edges.select(
        pl.col("item_node").alias("src"),
        pl.col("user_node").alias("dst"),
    )
    user_to_item = edges.select(
        pl.col("user_node").alias("src"),
        pl.col("item_node").alias("dst"),
    )
    edge_df = pl.concat([item_to_user, user_to_item])

    edge_index = torch.tensor(
        edge_df.select(["src", "dst"]).to_numpy().T,
        dtype=torch.long,
    )

    row, col = edge_index[0].contiguous(), edge_index[1].contiguous()

    start_items = torch.arange(num_items, dtype=torch.long)
    start = start_items.repeat_interleave(config.walks_per_item)

    rw = random_walk(row, col, start=start, walk_length=config.walk_length)
    item_walks = rw[:, ::2].numpy()

    item_ids = item_map.sort("item_node")["item_id"].to_numpy()

    pair_counts = {}

    for walk in item_walks:
        seen_pairs = set()
        L = len(walk)

        for i in range(L):
            src_node = int(walk[i])
            if src_node >= num_items:
                continue
            src = int(item_ids[src_node])

            lo = max(0, i - config.window_size)
            hi = min(L, i + config.window_size + 1)

            for j in range(lo, hi):
                if i == j:
                    continue

                dst_node = int(walk[j])
                if dst_node >= num_items:
                    continue
                dst = int(item_ids[dst_node])

                if not config.self_loops and src == dst:
                    continue

                pair = (src, dst)

                if config.dedup_per_walk:
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                pair_counts[pair] = pair_counts.get(pair, 0.0) + 1.0

    if not pair_counts:
        return pl.DataFrame(
            {
                "src": pl.Series([], dtype=pl.Int64),
                "dst": pl.Series([], dtype=pl.Int64),
                "weight": pl.Series([], dtype=pl.Float64),
            }
        )

    return (
        pl.DataFrame(
            {
                "src": [k[0] for k in pair_counts],
                "dst": [k[1] for k in pair_counts],
                "weight": [float(v) for v in pair_counts.values()],
            }
        )
        .group_by(["src", "dst"])
        .agg(pl.col("weight").sum().alias("weight"))
        .sort("weight", descending=True)
    )


def build_windowed_cooccurrence_graph(df: pl.DataFrame, config: WindowedGraphConfig) -> pl.DataFrame:
    if config.window_size < 1:
        raise ValueError("window_size must be >= 1")

    df = df.sort(["user_id", "ts"])

    shifted_frames = []
    for k in range(1, config.window_size + 1):
        if config.decay == "inverse_distance":
            w_expr = pl.lit(1.0 / k).alias("pair_weight")
        else:
            w_expr = pl.lit(1.0).alias("pair_weight")

        shifted = df.with_columns(pl.col("item_id").shift(-k).over("user_id").alias("dst")).select(
            [
                pl.col("user_id"),
                pl.col("item_id").alias("src"),
                pl.col("dst"),
                w_expr,
            ]
        )
        shifted_frames.append(shifted)

    pairs = pl.concat(shifted_frames).filter(pl.col("dst").is_not_null())

    if config.dedup_within_window:
        pairs = pairs.group_by(["user_id", "src", "dst"]).agg(pl.col("pair_weight").max().alias("pair_weight"))

    if config.bidirectional:
        rev = pairs.select(
            [
                pl.col("user_id"),
                pl.col("dst").alias("src"),
                pl.col("src").alias("dst"),
                pl.col("pair_weight"),
            ]
        )
        pairs = pl.concat([pairs, rev])

    out = (
        pairs.group_by(["src", "dst"]).agg(pl.col("pair_weight").sum().alias("weight")).sort("weight", descending=True)
    )

    return out


def get_graph_builder(graph_type: GraphType):
    graph_type_builders = {
        "adjacent_cooc_prod": build_adjacent_cooccurrence_graph_prod_chunked,
        "adjacent_cooc": build_adjacent_cooccurrence_graph,
        "rw": build_bipartite_rw_item_graph,
        "windowed": build_windowed_cooccurrence_graph,
    }

    return graph_type_builders[graph_type]


def polars_graph_to_pyg(
    graph_df: pl.DataFrame,
    labels_df: pl.DataFrame | None = None,
    make_directed: bool = False,
    add_identity_features: bool = False,
    num_items: int | None = None,
    all_item_ids: list[int] | None = None,
):
    """
    Convert a Polars edge list into a PyG Data object.

    graph_df columns:
        - src
        - dst
        - weight   (optional but recommended)

    labels_df columns (optional):
        - item_id
        - y

    You must provide one of:
        - num_items: assumes valid item ids are [0, ..., num_items-1]
        - all_item_ids: explicit universe of item ids
        **PRECEDENCE**: If all_item_ids is provided, it takes precedence over num_items.
        This allows defining a specific subset of nodes (e.g., training items only)
        regardless of the global embedding matrix size.

    Returns:
        data: torch_geometric.data.Data
        node_map: Polars DataFrame with columns [item_id, node_id]
    """
    required = {"src", "dst"}
    missing = required - set(graph_df.columns)
    if missing:
        raise ValueError(f"graph_df is missing columns: {missing}")

    if "weight" not in graph_df.columns:
        graph_df = graph_df.with_columns(pl.lit(1.0).alias("weight"))

    # If both are passed, all_item_ids takes precedence to create node_map only of training items
    if all_item_ids is not None:
        if len(all_item_ids) == 0:
            raise ValueError("all_item_ids is empty")

        node_map = (
            pl.DataFrame({"item_id": all_item_ids})
            .unique()
            .sort("item_id")
            .with_row_index("node_id")
            .select(["item_id", "node_id"])
        )

        if node_map["item_id"].min() < 0:
            raise ValueError("item ids must be non-negative")
    elif num_items is not None:
        if num_items <= 0:
            raise ValueError("num_items must be positive")

        node_map = (
            pl.DataFrame({"item_id": list(range(num_items))})
            .with_columns(pl.col("item_id").alias("node_id"))
            .select(["item_id", "node_id"])
        )
    else:
        raise ValueError("Provide exactly one of num_items or all_item_ids")

    valid_ids = set(node_map["item_id"].to_list())
    edge_ids = set(graph_df["src"].to_list()) | set(graph_df["dst"].to_list())
    unknown = edge_ids - valid_ids
    if unknown:
        raise ValueError(f"graph_df contains item ids not present in provided item universe: {sorted(unknown)[:20]}")

    if all_item_ids is not None:
        src_map = node_map.rename({"item_id": "src", "node_id": "src_id"})
        dst_map = node_map.rename({"item_id": "dst", "node_id": "dst_id"})
        graph_df = graph_df.join(src_map, on="src", how="left").join(dst_map, on="dst", how="left")
    else:
        graph_df = graph_df.rename({"src": "src_id", "dst": "dst_id"})

    edges = graph_df.select(
        [
            pl.col("src_id").cast(pl.Int64),
            pl.col("dst_id").cast(pl.Int64),
            pl.col("weight").cast(pl.Float64),
        ]
    )
    num_nodes = int(node_map.height)

    if not make_directed:
        rev = edges.select(
            [
                pl.col("dst_id").alias("src_id"),
                pl.col("src_id").alias("dst_id"),
                pl.col("weight"),
            ]
        )
        edges = pl.concat([edges, rev])

    edges = edges.group_by(["src_id", "dst_id"]).agg(pl.col("weight").sum().alias("weight")).sort(["src_id", "dst_id"])

    if edges.height == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float)
    else:
        edge_index = torch.tensor(
            edges.select(["src_id", "dst_id"]).to_numpy().T,
            dtype=torch.long,
        )
        edge_weight = torch.tensor(edges["weight"].to_numpy(), dtype=torch.float)

    data = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)

    # Labels
    if labels_df is not None:
        lbl = node_map.join(labels_df, on="item_id", how="left").sort("node_id")

        if lbl["y"].null_count() > 0:
            raise ValueError("Some nodes have no label in labels_df")

        y = torch.full((num_nodes,), -1, dtype=torch.long)
        y_idx = torch.tensor(lbl["node_id"].to_numpy(), dtype=torch.long)
        y_val = torch.tensor(lbl["y"].to_numpy(), dtype=torch.long)
        y[y_idx] = y_val
        data.y = y

    # Degree-based fallback feature; isolated nodes get 0
    deg = torch.zeros(num_nodes, dtype=torch.float)
    if edge_weight.numel() > 0:
        deg.scatter_add_(0, data.edge_index[0], data.edge_weight)
    log_deg = torch.log1p(deg).unsqueeze(-1)

    if add_identity_features:
        x = torch.eye(num_nodes, dtype=torch.float)
    else:
        x = log_deg

    data.x = x
    return data, node_map


def analyze_graph_quick(graph_df: pl.DataFrame, directed: bool = False, weight_col: str = "weight"):
    if weight_col not in graph_df.columns:
        graph_df = graph_df.with_columns(pl.lit(1.0).alias(weight_col))

    if directed:
        edges = graph_df.select(["src", "dst", weight_col])
    else:
        edges = (
            graph_df.with_columns(
                [
                    pl.min_horizontal("src", "dst").alias("u"),
                    pl.max_horizontal("src", "dst").alias("v"),
                ]
            )
            .group_by(["u", "v"])
            .agg(pl.col(weight_col).sum().alias(weight_col))
            .rename({"u": "src", "v": "dst"})
        )

    nodes = pl.concat(
        [
            edges.select(pl.col("src").alias("node")),
            edges.select(pl.col("dst").alias("node")),
        ]
    ).unique()

    num_nodes = nodes.height
    num_edges = edges.height

    if directed:
        out_deg = edges.group_by("src").agg(pl.len().alias("out_degree")).rename({"src": "node"})
        in_deg = edges.group_by("dst").agg(pl.len().alias("in_degree")).rename({"dst": "node"})
        degree_df = (
            nodes.join(out_deg, on="node", how="left")
            .join(in_deg, on="node", how="left")
            .with_columns(
                [
                    pl.col("out_degree").fill_null(0),
                    pl.col("in_degree").fill_null(0),
                ]
            )
            .with_columns((pl.col("out_degree") + pl.col("in_degree")).alias("degree"))
        )
    else:
        degree_df = nodes.join(
            pl.concat(
                [
                    edges.select(pl.col("src").alias("node")),
                    edges.select(pl.col("dst").alias("node")),
                ]
            )
            .group_by("node")
            .agg(pl.len().alias("degree")),
            on="node",
            how="left",
        ).with_columns(pl.col("degree").fill_null(0))

    density = (
        (num_edges / (num_nodes * (num_nodes - 1)) if directed else (2 * num_edges) / (num_nodes * (num_nodes - 1)))
        if num_nodes > 1
        else 0.0
    )

    stats = degree_df.select(
        [
            pl.len().alias("num_nodes"),
            pl.col("degree").mean().alias("avg_degree"),
            pl.col("degree").median().alias("median_degree"),
            pl.col("degree").quantile(0.95).alias("p95_degree"),
            pl.col("degree").max().alias("max_degree"),
            (pl.col("degree") == 0).sum().alias("isolated_nodes"),
        ]
    ).to_dicts()[0]

    stats["num_edges"] = num_edges
    stats["density"] = float(density)
    stats["isolated_ratio"] = float(stats["isolated_nodes"] / num_nodes) if num_nodes else 0.0
    stats["top_degree_nodes"] = degree_df.sort("degree", descending=True).head(10)

    return stats


class InteractionGraphBuilder:
    """
    A class to build a graph from an interactions file.

    This class supports three distinct graph construction methods:
    - 'windowed': Builds edges based on co-occurrence within a fixed sliding window.
    - 'adjacent_cooc': Builds edges based on adjacent co-occurrence in the sequence.
    - 'rw': Builds edges using a random walk strategy.

    The graph building process is performed lazily; the actual construction
    occurs only upon the first access to the graph structure. This allows
    for efficient memory usage and deferred computation.

    Additionally, the resulting graph metadata can be dumped to a file
    for persistence and reloaded later to avoid recomputation.
    """

    def __init__(
        self,
        interactions_path: str,
        graph_building_config: GraphBuildingConfig,
        skip_n_last: int = 2,
        num_items: int | None = None,
    ):
        self.interactions_path = interactions_path
        self.graph_building_config = graph_building_config
        self.skip_n_last = skip_n_last
        self.num_items = num_items

        self._graph_df: pl.DataFrame | None = None
        self._node_map: pl.DataFrame | None = None
        self._pyg_data: Data | None = None

    def _ensure_built(self):
        """Internal method to trigger building if not already done."""
        if self._pyg_data is not None:
            return  # Already built or loaded

        if self.interactions_path is None:
            raise ValueError("Cannot build graph: either pass interactions_path or load from dump")

        self._build_graph_from_interactions()

    def _build_graph_from_interactions(self):
        graph_type = self.graph_building_config.graph_type
        graph_builder = get_graph_builder(graph_type)

        if not graph_type.endswith("prod"):
            df, all_item_ids = self._load_interactions(self.interactions_path, skip_n_last=self.skip_n_last)
            self._graph_df = graph_builder(df, self.graph_building_config)
        else:
            self._graph_df, self.num_items = graph_builder(self.graph_building_config)
            all_item_ids = load_train_item_mask(self.interactions_path, self.num_items, skip_n_last=self.skip_n_last)

        self._pyg_data, node_map = polars_graph_to_pyg(
            self._graph_df,
            make_directed=self.graph_building_config.make_directed,
            add_identity_features=self.graph_building_config.add_identity_features,
            num_items=self.num_items,
            all_item_ids=all_item_ids,
        )
        self._node_map = node_map.sort("node_id")

    def _load_interactions(self, path, skip_n_last=None):
        with open(os.path.join(path), "r") as f:
            inters = json.load(f)

        if skip_n_last is None:
            rows = [(user, item, pos) for user, items in inters.items() for pos, item in enumerate(items)]
        else:
            rows = [
                (user, item, pos) for user, items in inters.items() for pos, item in enumerate(items[:-skip_n_last])
            ]

        df = pd.DataFrame(rows, columns=["user_id", "item_id", "ts"]).explode("item_id").reset_index(drop=True)
        df = pl.from_pandas(df).select(["user_id", "ts", "item_id"])

        all_item_ids = None
        if skip_n_last is not None and skip_n_last > 0:
            all_item_ids = df["item_id"].unique().sort().to_numpy()

        return df, all_item_ids

    @property
    def pyg_data(self) -> Data:
        """Returns the PyG Data object. Triggers building if not already built."""
        self._ensure_built()
        return self._pyg_data

    @property
    def graph_df(self) -> pl.DataFrame:
        """Returns the Polars edge list DataFrame. Triggers building if not already built."""
        self._ensure_built()
        return self._graph_df

    @property
    def node_map(self) -> pl.DataFrame:
        """Returns the node mapping DataFrame. Triggers building if not already built."""
        self._ensure_built()
        return self._node_map

    def dump(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        meta = {
            "interactions_path": self.interactions_path,
            "graph_building_config": self.graph_building_config.to_kwargs(),
            "skip_n_last": self.skip_n_last,
            "num_items": self.num_items,
        }

        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        self.graph_df.write_parquet(os.path.join(save_dir, "graph_df.parquet"))
        self.node_map.write_parquet(os.path.join(save_dir, "node_map.parquet"))

        if getattr(self.pyg_data, "edge_weight", None) is not None:
            edge_weight = self.pyg_data.edge_weight.cpu()
        else:
            edge_weight = None

        if getattr(self.pyg_data, "x", None) is not None:
            pyg_data_x = self.pyg_data.x.cpu()
        else:
            pyg_data_x = None

        pyg_payload = {
            "edge_index": self.pyg_data.edge_index.cpu(),
            "edge_weight": edge_weight,
            "x": pyg_data_x,
            "num_nodes": int(self.pyg_data.num_nodes),
        }

        if getattr(self.pyg_data, "y", None) is not None:
            pyg_payload["y"] = self.pyg_data.y.cpu()

        torch.save(pyg_payload, os.path.join(save_dir, "pyg_data.pt"))

    @classmethod
    def from_dump(cls, load_path: str):
        """Load a previously built graph from disk."""
        from gensid.registry import REGISTRY

        meta_path = os.path.join(load_path, "meta.json")
        graph_df_path = os.path.join(load_path, "graph_df.parquet")
        node_map_path = os.path.join(load_path, "node_map.parquet")
        pyg_data_path = os.path.join(load_path, "pyg_data.pt")

        for path in [meta_path, graph_df_path, node_map_path, pyg_data_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required file: {path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        graph_type = meta["graph_building_config"]["graph_type"]
        graph_building_config = REGISTRY.get_graph_building_config(graph_type)(**meta.get("graph_building_config", {}))
        instance = cls(
            interactions_path=meta.get("interactions_path"),
            graph_building_config=graph_building_config,
            skip_n_last=meta.get("skip_n_last"),
            num_items=meta.get("num_items"),
        )

        instance._graph_df = pl.read_parquet(graph_df_path)
        instance._node_map = pl.read_parquet(node_map_path).sort("node_id")

        pyg_payload = torch.load(pyg_data_path, map_location="cpu", weights_only=False)

        instance._pyg_data = Data(
            edge_index=pyg_payload["edge_index"],
            edge_weight=pyg_payload.get("edge_weight"),
            num_nodes=pyg_payload["num_nodes"],
        )

        if pyg_payload.get("x") is not None:
            instance._pyg_data.x = pyg_payload["x"]
        if pyg_payload.get("y") is not None:
            instance._pyg_data.y = pyg_payload["y"]

        return instance
