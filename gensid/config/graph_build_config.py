from dataclasses import dataclass
from typing import Literal

GraphType = Literal["adjacent_cooc", "adjacent_cooc_prod", "rw", "windowed"]


@dataclass(frozen=True)
class GraphBuildingConfig:
    """Base for any interaction-graph construction config.

    Only graph-construction knobs live here; how the graph is later consumed
    (propagation, encoder, ...) is out of scope.

    Args:
        graph_type (GraphType, optional): Which graph construction strategy to use:
            "adjacent_cooc", "adjacent_cooc_prod", "rw" or "windowed".
            Defaults to "adjacent_cooc".

        make_directed (bool, optional): If True, canonicalize the built graph to
            be directed. Defaults to False.

        add_identity_features (bool, optional): If True, attach identity features
            to the graph nodes. Defaults to False.
    """

    graph_type: GraphType = "adjacent_cooc"
    make_directed: bool = False
    add_identity_features: bool = False

    def to_kwargs(self) -> dict:
        return {attr: value for attr, value in self.__dict__.items() if attr != "graph_type"}

    @classmethod
    def from_args(cls, args) -> "GraphBuildingConfig":
        return cls(**{k: getattr(args, k, v) for k, v in cls().to_kwargs().items()})


@dataclass(frozen=True)
class AdjacentCoocGraphConfig(GraphBuildingConfig):
    """Adjacent co-occurrence in the user sequence (non-prod variant).

    Adds one directed edge per adjacent pair in each user's item sequence, with
    the raw edge weight equal to the pair's occurrence count.

    Args:
        graph_type (Literal["adjacent_cooc"], optional): Graph type discriminant,
            fixed to "adjacent_cooc". Defaults to "adjacent_cooc".
    """

    graph_type: Literal["adjacent_cooc"] = "adjacent_cooc"


@dataclass(frozen=True)
class AdjacentCoocProdGraphConfig(GraphBuildingConfig):
    """Adjacent co-occurrence graph built for production-scale data.

    A map-reduce variant of ``AdjacentCoocGraphConfig`` that streams interaction
    Parquet files in chunks, aggregates partial edge counts, and maps original
    item ids to normalized ids. During graph building it yields both the edge
    list (columns: src, dst, weight) and the item count.

    Args:
        graph_type (Literal["adjacent_cooc_prod"], optional): Graph type
            discriminant, fixed to "adjacent_cooc_prod".
            Defaults to "adjacent_cooc_prod".

        parquet_folder (str, optional): Folder containing interaction Parquet
            files. Defaults to "".

        seq_col (str, optional): Column containing one item sequence per row/user.
            Defaults to "history_item_id".

        filter_col (str or None, optional): Optional list column aligned
            element-wise with ``seq_col``; when set, sequences are filtered before
            building edges. Defaults to None.

        allowed_filter_values (list, set or None, optional): When ``filter_col`` is
            set, keep sequence elements whose corresponding filter value belongs
            to this collection. Defaults to None.

        directed (bool, optional): If False, canonicalize every edge as
            (min(src, dst), max(src, dst)). Defaults to True.

        chronological (bool, optional): If False, reverse each sequence before
            constructing edges. Defaults to True.

        embeddings_path (str or None, optional): Parquet file containing ``item_id`` and
            ``normalized_id``, used to map edges to normalized ids and to count
            items. Defaults to None.

        files_per_chunk (int, optional): Maximum number of interaction files
            processed in one map batch. Defaults to 16.

        max_files (int or None, optional): Maximum total number of sorted files to
            process; None means process every file. Defaults to None.

        tmp_dir (str or None, optional): Optional parent directory for temporary
            partial edge files. The temporary child directory is deleted
            automatically. Defaults to None.
    """

    graph_type: Literal["adjacent_cooc_prod"] = "adjacent_cooc_prod"
    parquet_folder: str = ""
    seq_col: str = "history_item_id"
    filter_col: str | None = None
    allowed_filter_values: list | set | None = None
    directed: bool = True
    chronological: bool = True
    embeddings_path: str | None = None
    files_per_chunk: int = 16
    max_files: int | None = None
    tmp_dir: str | None = None


@dataclass(frozen=True)
class RandomWalkGraphConfig(GraphBuildingConfig):
    """Random-walk co-occurrence graph.

    Runs ``walks_per_item`` random walks of ``walk_length`` steps starting from
    every item through the bipartite item-user graph, then connects items that
    co-occur within a ``window_size``-radius window on the same walk.

    Args:
        graph_type (Literal["rw"], optional): Graph type discriminant, fixed to
            "rw". Defaults to "rw".

        walk_length (int, optional): Number of steps in each random walk.
            Defaults to 10.

        walks_per_item (int, optional): Number of random walks started from each
            item. Defaults to 10.

        window_size (int, optional): Radius of the co-occurrence window centered
            at each position in a walk. Defaults to 2.

        self_loops (bool, optional): If False, skip edges where src == dst.
            Defaults to False.

        dedup_per_walk (bool, optional): If True, count each (src, dst) pair only
            once per walk. Defaults to False.
    """

    graph_type: Literal["rw"] = "rw"
    walk_length: int = 10
    walks_per_item: int = 10
    window_size: int = 2
    self_loops: bool = False
    dedup_per_walk: bool = False


@dataclass(frozen=True)
class WindowedGraphConfig(GraphBuildingConfig):
    """Sliding-window co-occurrence graph.

    Builds weighted edges between each item and every item up to ``window_size``
    positions ahead of it within each (chronologically sorted) user sequence.

    Args:
        graph_type (Literal["windowed"], optional): Graph type discriminant,
            fixed to "windowed". Defaults to "windowed".

        window_size (int, optional): Number of look-ahead hops per item in a user
            sequence. Defaults to 3.

        bidirectional (bool, optional): If True, also add the reverse (dst, src)
            edge for every (src, dst) pair. Defaults to False.

        dedup_within_window (bool, optional): If True, keep a single edge per
            (user, src, dst), taking the maximum weight. Defaults to False.

        decay (str or None, optional): Weight decay over the window offsets:
            "inverse_distance" weights a k-step pair by 1/k; None uses uniform
            weight. Defaults to None.
    """

    graph_type: Literal["windowed"] = "windowed"
    window_size: int = 3
    bidirectional: bool = False
    dedup_within_window: bool = False
    decay: str | None = None  # "inverse_distance" or None
