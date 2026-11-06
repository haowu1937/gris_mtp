import json
import os
import re

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from torch.utils.data._utils.collate import default_collate
from torch_geometric.utils import subgraph


def load_embeddings(embeddings_path: str) -> np.ndarray:
    """Load an embedding matrix from a ``.npy`` file or a Parquet ``embedding`` column.

    The Parquet layout (production flows) stores one row per item with an
    ``embedding`` list column; the row order is assumed to match the 0-indexed
    item (normalized) id. ``.npy`` files are loaded directly.

    Args:
        embeddings_path (str): Path to a ``.npy`` or ``.parquet`` embeddings file.

    Returns:
        np.ndarray: Embedding matrix of shape ``[num_items, dim]``.
    """
    if str(embeddings_path).endswith(".parquet"):
        df = pl.scan_parquet(embeddings_path).collect()
        return np.array(df["embedding"].to_list(), dtype=np.float32)
    if str(embeddings_path).endswith(".npy"):
        return np.load(embeddings_path)
    raise ValueError(f"Unsupported embeddings file format: {embeddings_path!r} (expected .npy or .parquet)")


class EdgeWeightCollator:
    # TODO: Check out for errors and optimization. Why self.num_nodes is not used for num_nodes param in subgraph?
    def __init__(self, pyg_data, num_anchors=1024):
        # Keep graph on CPU for worker processes.
        self.edge_index = pyg_data.edge_index.cpu()
        self.edge_weight = pyg_data.edge_weight.cpu()
        self.num_nodes = pyg_data.num_nodes
        self.num_anchors = num_anchors

    def __call__(self, batch):
        collated = default_collate(batch)

        # For EmbDataset / APPNPEmbDataset:
        # collated = (data, emb_idx, recon_tgt, maybe spectral)
        emb_idx = collated["index"].long()

        ei, ew = subgraph(
            subset=emb_idx,
            edge_index=self.edge_index,
            edge_attr=self.edge_weight,
            relabel_nodes=True,
            num_nodes=max(emb_idx.max(), self.edge_index.max()) + 1,
        )

        B = emb_idx.size(0)
        edge_w = torch.zeros(B, min(B, self.num_anchors), dtype=torch.float32)

        if ei.numel() > 0:
            mask = ei[1] < self.num_anchors
            ei = ei[:, mask]
            ew = ew[mask]
            edge_w[ei[0], ei[1]] = ew.float()

        return {**collated, "edge_weights": edge_w}


def load_train_item_mask(path, num_items, skip_n_last=None):
    if skip_n_last is None or skip_n_last <= 0:
        return np.arange(num_items)
    with open(os.path.join(path), "r") as f:
        inters = json.load(f)

    mask = np.zeros(num_items, dtype=bool)
    for items in inters.values():
        seq = items if skip_n_last is None else items[:-skip_n_last]
        for item in seq:
            item = int(item)
            if item < 0 or item >= num_items:
                raise ValueError(f"Interaction item_id={item} is outside embeddings range [0, {num_items})")
            mask[item] = True

    return np.flatnonzero(mask).astype(np.int64)


def load_graph_cluster_labels(
    labels_path: str | None,
    levels=None,
    num_items: int | None = None,
    label_group: str = "coarse_to_fine_item_labels",
):
    """
    Load graph-cluster labels produced by hierarchical clustering scripts.

    This function normalizes several possible clustering-output formats into a
    consistent representation for RQ-VAE graph-cluster supervision.

    Supported JSON formats
    ----------------------
    1. Preferred hierarchical output:

        {
            "coarse_to_fine_item_labels": {
                "level_1": [...],
                "level_2": [...]
            },
            "fine_to_coarse_item_labels": {
                "level_1": [...],
                "level_2": [...]
            },
            ...
        }

    2. Flat level dictionary:

        {
            "level_1": [...],
            "level_2": [...]
        }

    3. Direct list of levels:

        [
            [...],   # level 1
            [...]    # level 2
        ]

    4. Single label vector:

        [...]

    Parameters
    ----------
    labels_path : str | None
        Path to the labels JSON file. If None, returns None.

    levels : str | int | list[str | int] | tuple[str | int] | None
        Which levels to load.

        - None:
            Load all levels in their natural order.
        - str:
            Load one named level, e.g. "level_1".
        - int:
            Load one 1-based level index, e.g. 1 means "level_1".
        - list/tuple:
            Load multiple levels in exactly that order.

        For RQ-VAE prefix supervision, the order matters. For example, if the
        JSON group is fine-to-coarse but you want prefix supervision
        coarse-to-fine, pass the corresponding levels explicitly.

    num_items : int | None
        If provided, every loaded label vector must have this length.

    label_group : str
        Preferred group to load when the JSON contains several groups.
        Usually one of:

            "coarse_to_fine_item_labels"
            "fine_to_coarse_item_labels"
            "coarse_to_fine_graph_labels"

        For RQ-VAE item-level supervision, prefer:

            "coarse_to_fine_item_labels"

    Returns
    -------
    None | np.ndarray | list[np.ndarray]
        - None if labels_path is None.
        - Single np.ndarray[int64] of shape [num_items] if one level is loaded.
        - List[np.ndarray[int64]] if multiple levels are loaded.

    Notes
    -----
    Returned arrays are indexed by global item id:

        labels[item_id] -> graph cluster id

    During training, slice by batch item ids:

        batch_labels = labels[emb_idx]

    For hierarchical prefix supervision:

        batch_labels_per_level = [
            level_labels[emb_idx]
            for level_labels in graph_cluster_labels
        ]

    Items with label -1 are treated as unclustered / ignored by the graph
    cluster contrastive loss.
    """
    if labels_path is None:
        return None

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing graph cluster labels file: {labels_path}")

    with open(labels_path, "r") as f:
        payload = json.load(f)

    def _is_label_vector(x):
        return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], (int, np.integer)))

    def _is_list_of_label_vectors(x):
        return isinstance(x, list) and len(x) > 0 and all(isinstance(v, list) for v in x)

    def _level_sort_key(key):
        """
        Sort level_1, level_2, ... numerically.
        Fall back to lexical order for non-standard keys.
        """
        m = re.search(r"(\d+)$", str(key))
        if m is not None:
            return int(m.group(1))
        return str(key)

    def _normalize_level_name(level):
        """
        Accept:
            1 -> "level_1"
            "1" -> "level_1"
            "level_1" -> "level_1"
        """
        if isinstance(level, int):
            return f"level_{level}"

        if isinstance(level, str):
            if level.isdigit():
                return f"level_{int(level)}"
            return level

        raise ValueError(f"Invalid level specifier: {level!r}")

    def _validate_arr(arr, level_name):
        arr = np.asarray(arr, dtype=np.int64)

        if arr.ndim != 1:
            raise ValueError(f"Labels for {level_name!r} must be 1-D, got shape {arr.shape}")

        if num_items is not None and len(arr) != int(num_items):
            raise ValueError(f"Labels for {level_name!r} have length {len(arr)}, but expected num_items={num_items}")

        return arr

    # ------------------------------------------------------------
    # Case 1: payload itself is a single vector: [...]
    # ------------------------------------------------------------
    if _is_label_vector(payload):
        if levels is not None:
            raise ValueError("levels should be None when labels JSON is a single label vector")
        return _validate_arr(payload, "labels")

    # ------------------------------------------------------------
    # Case 2: payload is a list of vectors: [[...], [...]]
    # ------------------------------------------------------------
    if _is_list_of_label_vectors(payload):
        level_dict = {f"level_{i + 1}": v for i, v in enumerate(payload)}

    # ------------------------------------------------------------
    # Case 3: payload is a dict
    # ------------------------------------------------------------
    elif isinstance(payload, dict):
        # Preferred grouped format:
        # {"coarse_to_fine_item_labels": {"level_1": [...], ...}}
        if label_group in payload:
            level_dict = payload[label_group]

        # If requested group does not exist but there is exactly one obvious
        # label group, use it.
        elif any(isinstance(v, dict) and any(str(k).startswith("level_") for k in v) for v in payload.values()):
            candidate_groups = {
                k: v
                for k, v in payload.items()
                if isinstance(v, dict) and any(str(kk).startswith("level_") for kk in v)
            }

            if len(candidate_groups) == 1:
                level_dict = next(iter(candidate_groups.values()))
            else:
                raise ValueError(
                    f"Requested label_group={label_group!r} not found. "
                    f"Available label groups: {sorted(candidate_groups.keys())}. "
                    f"Pass label_group explicitly."
                )

        # Flat level dict:
        # {"level_1": [...], "level_2": [...]}
        elif any(str(k).startswith("level_") for k in payload):
            level_dict = payload

        # Single named vector:
        # {"labels": [...]}
        elif "labels" in payload and _is_label_vector(payload["labels"]):
            if levels is not None:
                raise ValueError("levels should be None when labels JSON contains only a single 'labels' vector")
            return _validate_arr(payload["labels"], "labels")

        else:
            raise ValueError(
                "Unrecognized graph cluster labels JSON format. Expected one of: "
                "single vector, list of vectors, flat level dict, or grouped "
                "hierarchical labels dict."
            )

    else:
        raise ValueError(f"Unsupported labels JSON root type: {type(payload).__name__}")

    if not isinstance(level_dict, dict):
        raise TypeError(f"Selected label group must be a dict, got {type(level_dict).__name__}")

    available_levels = sorted(level_dict.keys(), key=_level_sort_key)

    single_level = isinstance(levels, (str, int))

    if levels is None:
        selected_levels = available_levels
    elif single_level:
        selected_levels = [_normalize_level_name(levels)]
    elif isinstance(levels, (list, tuple)):
        selected_levels = [_normalize_level_name(level) for level in levels]
    else:
        raise ValueError("`levels` must be None, str, int, or list/tuple of str/int")

    out = []

    for level in selected_levels:
        if level not in level_dict:
            raise ValueError(f"Requested graph cluster level {level!r} not found. Available levels: {available_levels}")

        out.append(_validate_arr(level_dict[level], level))

    if single_level:
        return out[0]

    return out


def edge_w_from_subgraph(emb_idx, pyg_data):
    ei, ew = subgraph(
        subset=emb_idx,
        edge_index=pyg_data.edge_index,
        edge_attr=pyg_data.edge_weight,
        relabel_nodes=True,
    )

    B = emb_idx.size(0)
    edge_w = torch.zeros(B, B, device=ei.device)

    if ei.numel() > 0:
        edge_w[ei[0], ei[1]] = ew

    return edge_w


def edge_index_to_sparse_adj(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
) -> sp.csr_matrix:
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    val = (
        np.ones(row.shape[0], dtype=np.float64) if edge_weight is None else edge_weight.cpu().numpy().astype(np.float64)
    )

    A = sp.coo_matrix((val, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    A = 0.5 * (A + A.T)
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


def to_dense_adj(edge_index, num_nodes, edge_weight, device):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    if edge_index.numel() > 0:
        if edge_weight is None:
            adj[edge_index[0], edge_index[1]] = 1.0
        else:
            adj[edge_index[0], edge_index[1]] = edge_weight.float()
    adj = 0.5 * (adj + adj.t())
    diagonal_index = torch.arange(num_nodes, dtype=torch.int, device=adj.device)
    adj[diagonal_index, diagonal_index] = 0.0
    return adj


def drop_zero_degree_nodes(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    item_ids: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, np.ndarray, np.ndarray]:
    num_nodes = x.size(0)
    deg = torch.zeros(num_nodes, dtype=torch.float32)

    if edge_index.numel() > 0:
        w = torch.ones(edge_index.size(1), dtype=torch.float32) if edge_weight is None else edge_weight.cpu().float()
        deg.scatter_add_(0, edge_index[0].cpu(), w)
        deg.scatter_add_(0, edge_index[1].cpu(), w)

    keep = deg > 0
    kept_old = keep.nonzero(as_tuple=False).view(-1)
    removed_item_ids = item_ids[(~keep).numpy()]

    if kept_old.numel() == num_nodes:
        return x, edge_index, edge_weight, item_ids, removed_item_ids

    old_to_new = torch.full((num_nodes,), -1, dtype=torch.long)
    old_to_new[kept_old] = torch.arange(kept_old.numel(), dtype=torch.long)

    edge_keep = keep[edge_index[0].cpu()] & keep[edge_index[1].cpu()]
    edge_index_f = old_to_new[edge_index[:, edge_keep].cpu()]
    edge_weight_f = None if edge_weight is None else edge_weight.cpu()[edge_keep]

    return (
        x[kept_old],
        edge_index_f,
        edge_weight_f,
        item_ids[kept_old.numpy()],
        removed_item_ids,
    )
