import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import polars as pl


# --- cluster-label -> codes (moved from generate_graph_cluster_indices.py) ---
# _is_label_vector / _is_list_of_label_vectors / _level_sort_key /
# _normalize_level_name / load_cluster_label_levels / make_contiguous_per_level /
# build_codes / collision_stats all move here verbatim (lines 49-235 of that file).


PREFIX_FORMAT = "<{prefix_letter}_{id}>"


def prefix_letter_from_position(pos: int):
    return chr(ord("a") + pos)


def format_prefix(pos: int, index: int):
    return PREFIX_FORMAT.format(prefix_letter=prefix_letter_from_position(pos), id=index)


def _is_label_vector(x: Any) -> bool:
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], int))


def _is_list_of_label_vectors(x: Any) -> bool:
    return isinstance(x, list) and len(x) > 0 and all(isinstance(v, list) for v in x)


def _level_sort_key(key: str):
    m = re.search(r"(\d+)$", str(key))
    if m is not None:
        return int(m.group(1))
    return str(key)


def _normalize_level_name(level: str | int) -> str:
    if isinstance(level, int):
        return f"level_{level}"
    if isinstance(level, str):
        if level.isdigit():
            return f"level_{int(level)}"
        return level
    raise ValueError(f"Invalid level specifier: {level!r}")


def process_cluster_label_levels(
    payload: list | dict,
    label_group: str = "coarse_to_fine_item_labels",
    levels: list[str] | None = None,
) -> tuple[list[str], list[np.ndarray]]:
    """
    Process one or more item-id indexed cluster-label arrays from a clustering JSON.

    Supports:
      1. grouped dict: {"coarse_to_fine_item_labels": {"level_1": [...], ...}}
      2. flat dict: {"level_1": [...], "level_2": [...]}
      3. list of levels: [[...], [...]]
      4. single vector: [...]
    """
    if _is_label_vector(payload):
        level_dict = {"level_1": payload}
    elif _is_list_of_label_vectors(payload):
        level_dict = {f"level_{i + 1}": v for i, v in enumerate(payload)}
    elif isinstance(payload, dict):
        if label_group in payload:
            level_dict = payload[label_group]
        elif any(str(k).startswith("level_") for k in payload):
            level_dict = payload
        else:
            candidate_groups = {
                k: v
                for k, v in payload.items()
                if isinstance(v, dict) and any(str(kk).startswith("level_") for kk in v)
            }
            if len(candidate_groups) == 1:
                level_dict = next(iter(candidate_groups.values()))
            else:
                raise ValueError(
                    f"Could not select label group. Requested {label_group!r}; "
                    f"available candidate groups: {sorted(candidate_groups.keys())}"
                )
    else:
        raise ValueError(f"Unsupported JSON root type: {type(payload).__name__}")

    if not isinstance(level_dict, dict):
        raise TypeError(f"Selected label group must be dict, got {type(level_dict).__name__}")

    available = sorted(level_dict.keys(), key=_level_sort_key)
    selected = available if levels is None else [_normalize_level_name(lvl) for lvl in levels]

    out_names: list[str] = []
    out_arrays: list[np.ndarray] = []
    for name in selected:
        if name not in level_dict:
            raise ValueError(f"Requested level {name!r} not found. Available: {available}")
        arr = np.asarray(level_dict[name], dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError(f"Level {name!r} must be 1-D, got shape {arr.shape}")
        out_names.append(name)
        out_arrays.append(arr)

    lengths = {len(a) for a in out_arrays}
    if len(lengths) != 1:
        raise ValueError(f"All levels must have the same length, got {sorted(lengths)}")

    return out_names, out_arrays


def make_contiguous_per_level(
    labels: list[np.ndarray], keep_negative: bool = True
) -> tuple[list[np.ndarray], list[dict[int, int]]]:
    """Remap arbitrary cluster ids to compact 0..K-1 ids per level."""
    remapped = []
    maps = []
    for arr in labels:
        valid = np.unique(arr[arr >= 0] if keep_negative else arr)
        mapping = {int(old): int(new) for new, old in enumerate(sorted(map(int, valid)))}
        new_arr = np.full_like(arr, -1)
        for old, new in mapping.items():
            new_arr[arr == old] = new
        if not keep_negative and np.any(arr < 0):
            # Put negative labels after valid labels if user really asked not to keep them.
            neg_values = sorted(map(int, np.unique(arr[arr < 0])))
            offset = len(mapping)
            for j, old in enumerate(neg_values):
                mapping[old] = offset + j
                new_arr[arr == old] = offset + j
        remapped.append(new_arr)
        maps.append(mapping)
    return remapped, maps


def build_codes(
    labels: list[np.ndarray],
    unclustered_strategy: str,
    append_unique_leaf: bool,
    unique_prefix: str,
    prefixes: list[str] = None,
) -> dict[str, list[str]]:
    """
    Convert level labels into token sequences.

    unclustered_strategy:
      - special: use the same special id -1 at every level.
      - unique: use item id as the code for unclustered values.
      - error: fail if any -1 is present.
    append_unique_leaf:
      If True, append a final per-path rank token to make full SIDs unique.
    """
    if len(labels) == 0:
        raise ValueError("No label levels were loaded")

    n_items = len(labels[0])
    codes: dict[str, list[str]] = {}

    for item_id in range(n_items):
        item_tokens = []
        for level_idx, arr in enumerate(labels):
            cid = int(arr[item_id])
            if cid < 0:
                if unclustered_strategy == "error":
                    raise ValueError(f"item_id={item_id} has unclustered label {cid} at level {level_idx + 1}")
                if unclustered_strategy == "unique":
                    cid = item_id
                elif unclustered_strategy == "special":
                    cid = -1
                else:
                    raise ValueError(f"Unknown unclustered_strategy={unclustered_strategy!r}")
            token = prefixes[level_idx].format(cid) if prefixes is not None else format_prefix(level_idx, cid)
            item_tokens.append(token)
        codes[str(item_id)] = item_tokens

    if append_unique_leaf:
        path_to_items: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for item_id, toks in codes.items():
            path_to_items[tuple(toks)].append(item_id)

        for toks, item_ids in path_to_items.items():
            if len(item_ids) == 1:
                # rank 0 still makes all items have same SID length.
                codes[item_ids[0]].append(unique_prefix.format(0))
            else:
                for rank, item_id in enumerate(sorted(item_ids, key=lambda x: int(x))):
                    codes[item_id].append(unique_prefix.format(rank))

    return codes


def collision_stats(codes: dict[str, list[str]]) -> dict[str, Any]:
    paths = [tuple(v) for v in codes.values()]
    c = Counter(paths)
    n = len(paths)
    n_unique = len(c)
    return {
        "num_items": n,
        "num_unique_sids": n_unique,
        "num_colliding_items": int(sum(v for v in c.values() if v > 1)),
        "collision_rate": float((n - n_unique) / n) if n else 0.0,
        "max_collision_group_size": int(max(c.values())) if c else 0,
    }


def project_labels_back_to_item_space(graph_labels: np.ndarray, item_ids: np.ndarray, num_items: int):
    """Map graph-node-ordered indices back to global item_id order."""
    full = np.full((num_items, graph_labels.shape[1]), -1, dtype=np.int64)
    full[item_ids] = graph_labels

    return full


def build_c2f_fc2_labels(graph_labels_coarse_to_fine, item_labels_coarse_to_fine):
    """Bundle coarse-to-fine and fine-to-coarse label views into one JSON dict.

    Args:
        graph_labels_coarse_to_fine (np.ndarray): Graph-node-order labels,
            coarsest level first. [num_nodes, depth]
        item_labels_coarse_to_fine (np.ndarray): Item-id-order labels,
            coarsest level first. [num_items, depth]

    Returns:
        dict: ``coarse_to_fine_item_labels``, ``fine_to_coarse_item_labels``
        and ``coarse_to_fine_graph_labels``, each ``level_<i>`` keyed.
    """
    return {
        "coarse_to_fine_item_labels": {
            f"level_{i}": labels.tolist() for i, labels in enumerate(item_labels_coarse_to_fine, start=1)
        },
        "fine_to_coarse_item_labels": {
            f"level_{i}": labels.tolist() for i, labels in enumerate(reversed(item_labels_coarse_to_fine), start=1)
        },
        "coarse_to_fine_graph_labels": {
            f"level_{i}": labels.astype(np.int64).tolist()
            for i, labels in enumerate(graph_labels_coarse_to_fine, start=1)
        },
    }


# --- parquet SID exports (folded from old generate_indices.py / generate_recdmon_indices.py) ---


def save_parquet_semantic_ids(
    indices: np.ndarray,
    output_file: str,
    embeddings_path: str | None = None,
) -> None:
    """Persist quantizer indices as the parquet SID artifact (old ``generate_indices.py --save_parquet``).

    Columns ``item_id``, ``normalized_id``, ``semantic_id``, ``raw_id``; each
    code is ``+1``-shifted so ``0`` is reserved for padding. When
    ``embeddings_path`` points to a parquet table (with ``item_id``,
    ``normalized_id``, ``raw_id``), ``item_id``/``raw_id`` come from that join;
    otherwise they fall back to the identity (normalized id == item id).
    """
    num_items, _ = indices.shape
    sid_df = pl.DataFrame(
        {
            "normalized_id": np.arange(num_items, dtype=np.int64),
            "semantic_id": [list(row + 1) for row in indices.astype(np.int64)],
        }
    )
    if embeddings_path is not None and str(embeddings_path).endswith(".parquet"):
        emb = pl.read_parquet(embeddings_path)
        result = emb.join(sid_df, on="normalized_id", how="inner").select(
            "item_id", "normalized_id", "semantic_id", "raw_id"
        )
    else:
        result = sid_df.with_columns(
            pl.col("normalized_id").alias("item_id"), pl.col("normalized_id").alias("raw_id")
        ).select("item_id", "normalized_id", "semantic_id", "raw_id")
    result.write_parquet(output_file)


def build_recdmon_semantic_ids(
    item_labels_coarse_to_fine: np.ndarray,
    output_file: str,
    all_items_index: str | None = None,
) -> None:
    """Persist clustering SIDs as the parquet artifact (old ``generate_recdmon_indices.py``).

    Columns ``item_id``, ``semantic_id``, ``raw_id`` sorted by ``item_id``.
    Known items get their per-level label ``+1`` (reserving ``0`` for padding);
    unclustered items (any level label ``< 0``) and OOV items (present in
    ``all_items_index`` but outside the embedding item space) share a per-level
    ``max(label) + 2`` prefix, and every prefix is made unique with a 1-based
    leaf rank. ``raw_id`` comes from ``all_items_index`` when given, else falls
    back to ``item_id``.
    """
    num_items, depth = item_labels_coarse_to_fine.shape
    levels = [item_labels_coarse_to_fine[:, d].astype(np.int64) for d in range(depth)]
    oov_prefix = tuple(int(np.max(level)) + 2 for level in levels)

    item_prefixes = []
    for item_id in range(num_items):
        labels = [int(level[item_id]) for level in levels]
        if any(label < 0 for label in labels):
            prefix = oov_prefix
        else:
            prefix = tuple(label + 1 for label in labels)
        item_prefixes.append((item_id, prefix))

    if all_items_index is not None:
        oov_df = pl.read_parquet(all_items_index, columns=["item_id", "raw_id"]).filter(
            ~pl.col("item_id").is_in(range(num_items))
        )
        item_prefixes.extend((int(r[0]), oov_prefix) for r in oov_df.iter_rows())

    leaf_counts = defaultdict(int)
    semantic_ids = {}
    for item_id, prefix in sorted(item_prefixes):
        leaf_counts[prefix] += 1
        semantic_ids[item_id] = [*prefix, leaf_counts[prefix]]

    df = pl.DataFrame({"item_id": list(semantic_ids.keys()), "semantic_id": list(semantic_ids.values())}).with_columns(
        pl.col("item_id").cast(pl.Int64)
    )
    if all_items_index is not None:
        df = df.join(pl.read_parquet(all_items_index, columns=["item_id", "raw_id"]), on="item_id", how="left")
    else:
        df = df.with_columns(pl.col("item_id").alias("raw_id"))
    df = df.select("item_id", "semantic_id", "raw_id").sort("item_id")
    df.write_parquet(output_file)
