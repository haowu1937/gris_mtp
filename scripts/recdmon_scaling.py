#!/usr/bin/env python
"""Scaling study for RecDMoN (recursive DMoN hierarchical SIDs).

Sweeps the dominant cost drivers of :class:`gensid.method.recursive_hierarchy.RecDMoN`
and reports wall-time / peak-RSS / cluster metrics at every scale:

* the number of items ``N`` (and the embedding matrix), via ``--ns``,
* the embedding size ``D``, via ``--ds``,
* method hyper-parameters, most notably the ``recursive_factor`` (the number
  of child clusters per parent, which sets ``k`` of every local DMoN split),
  plus ``--levels`` / ``--epochs`` / ``--hidden`` / ``--avg-degree``.

It runs on the device selected by ``gensid.utils.get_device`` (``--device``;
``auto`` prefers NPU, then CUDA, then CPU) and is designed to be pushed up to
``N = 100_000_000`` items.

Three modes (``--mode``):

``e2e`` (default)
    Runs the real ``RecDMoN.fit`` end-to-end on a synthetic interaction graph
    (block-structured, seeded) and times the whole recursion, breaking it down
    per hierarchy level via the fit callback. Because the *first* RecDMoN level
    clusters the whole graph with a dense DMoN split, ``N`` must fit the dense
    ``N x N`` float32 slice (``--mem-gb`` guard, default 200 GB); larger
    ``N`` are handled by the ``graph`` / ``kernel`` modes.

``kernel``
    Micro-benchmarks a *single* local DMoN split (``LocalAssignNet`` +
    ``dense_dmon_local_loss``): the per-parent cost that dominates a level is
    ``O(subset_size^2 * k)`` in time and ``O(subset_size^2)`` in memory. This is
    the term that determines whether RecDMoN can run at 100M items, so it is
    measured directly for arbitrary ``--subset-sizes`` without materializing
    ``N x N`` for the whole graph.

``graph``
    Times only the *sparse* stages (synthetic edge-list generation +
    ``edge_index_to_sparse_adj``, the same call ``RecursiveClusteringMixin``
    makes inside ``fit``). These are the only stages that are feasible at
    ``N = 100M`` (a few GBs to a few tens of GBs of RAM), so this mode shows the
    graph/adjacency budget that ``N`` alone imposes.

Results are printed as a table and optionally written to a TSV (``--out``).
Peak RSS is process-wide (``resource.getrusage``), as in ``scripts/pq_opq_scaling.py``.

Examples::

    # quick CPU sanity check (small)
    python scripts/recdmon_scaling.py --device=cpu --mode=e2e --ns 1000,5000 --ds 32 --rfs 2,4 --epochs 30

    # per-parent DMoN kernel cost (extrapolates the 100M budget)
    python scripts/recdmon_scaling.py --device=npu --mode=kernel --subset-sizes 10000,50000,100000 --ds 128 --ks 4,6

    # sparse graph + adjacency at up to 100M items (no embeddings materialized)
    python scripts/recdmon_scaling.py --mode=graph --ns 100000,1000000,100000000 --avg-degree 8
"""

import argparse
import itertools
import json
import resource
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from gensid.config.model_config import RecDMoNModelConfig
from gensid.data.utils import edge_index_to_sparse_adj
from gensid.method.recursive_hierarchy import RecDMoN
from gensid.method.utils import dense_dmon_local_loss
from gensid.nn.layers import LocalAssignNet
from gensid.utils import get_device


class _NullTqdm:
    """Drop-in for ``tqdm`` that renders nothing (``--quiet``)."""

    def __init__(self, iterable=None, **kwargs):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable) if self._iterable is not None else iter(())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _silence_tqdm() -> None:
    import importlib

    # ``gensid.method.utils`` and ``gensid.method.recursive_hierarchy`` do
    # ``from tqdm import tqdm`` at import time, so their module-global ``tqdm``
    # is a direct reference to the real class; swap that reference for a no-op.
    for module_name in ("gensid.method.utils", "gensid.method.recursive_hierarchy"):
        module = importlib.import_module(module_name)
        module.tqdm = _NullTqdm


def _peak_rss_kb() -> int:
    """Current process peak RSS in KiB (Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _sync(device: torch.device) -> None:
    """Block until queued kernels on ``device`` finish (async device timing)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize()


def _make_embeddings(num_items: int, dim: int, seed: int) -> torch.Tensor:
    """Deterministic float32 gaussian embeddings ``[num_items, dim]`` (CPU)."""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal((num_items, dim)).astype(np.float32))


def _make_graph(
    num_items: int,
    avg_degree: int,
    n_blocks: int,
    intra_frac: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize a block-structured interaction graph edge list.

    ``intra_frac`` of the ``num_items * avg_degree`` directed edges are sampled
    inside one of ``n_blocks`` contiguous item blocks (so DMoN has recoverable
    structure); the rest connect random item pairs. Returns ``(edge_index [2, E],
    edge_weight [E])`` as CPU tensors. Edge counts are capped only by memory, so
    ``num_items`` may be taken to 100M with a modest ``avg_degree``.
    """
    if num_items < 2 or avg_degree < 1:
        raise ValueError("num_items >= 2 and avg_degree >= 1 are required")
    if n_blocks < 2:
        raise ValueError("n_blocks >= 2 is required for a structured graph")

    rng = np.random.default_rng(seed)
    total_edges = int(num_items * avg_degree)
    n_intra = int(total_edges * intra_frac)
    n_inter = total_edges - n_intra
    block_size = max(1, num_items // n_blocks)

    srcs: list[np.ndarray] = []
    dsts: list[np.ndarray] = []
    if n_intra > 0:
        block = rng.integers(0, n_blocks, size=n_intra)
        off_a = rng.integers(0, block_size, size=n_intra)
        off_b = rng.integers(0, block_size, size=n_intra)
        srcs.append(block * block_size + off_a)
        dsts.append(block * block_size + off_b)
    if n_inter > 0:
        srcs.append(rng.integers(0, num_items, size=n_inter))
        dsts.append(rng.integers(0, num_items, size=n_inter))

    src = np.minimum(np.concatenate(srcs), num_items - 1).astype(np.int64)
    dst = np.minimum(np.concatenate(dsts), num_items - 1).astype(np.int64)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    weight = (1 + rng.integers(0, 10, size=src.shape[0])).astype(np.float32)

    return torch.from_numpy(np.stack([src, dst])).long(), torch.from_numpy(weight)


@dataclass
class E2EResult:
    n: int
    d: int
    recursive_factor: int
    recursive_levels: int
    recursive_epochs: int
    avg_degree: int
    n_edges: int
    adj_nnz: int
    fit_s: float
    per_level_s: str
    n_splits: int
    epochs_total: int
    avg_modularity: float
    max_subset: int
    cluster_counts: str
    rss_mb: float


class _FitStats:
    """Accumulate per-level timings / loss from ``RecDMoN.fit(callback=...)``."""

    def __init__(self):
        self.per_level_s: dict[int, float] = {}
        self.n_splits = 0
        self.epochs_total = 0
        self.modularity_sum = 0.0
        self.max_subset = 0

    def __call__(self, epoch: int, start: float, end: float, loss: float, **info) -> None:
        level = int(info.get("level", 1))
        dt = end - start
        self.per_level_s[level] = self.per_level_s.get(level, 0.0) + dt
        self.n_splits += 1
        self.epochs_total += 1
        self.max_subset = max(self.max_subset, int(info.get("subset_size", 0)))
        self.modularity_sum += float(info.get("modularity", 0.0))


class _Skipped(Exception):
    """Raised when a case cannot run (e.g. the dense root slice exceeds --mem-gb)."""


def _run_e2e_case(
    n: int,
    d: int,
    recursive_factor: int,
    levels: int,
    epochs: int,
    hidden: int,
    avg_degree: int,
    n_blocks: int,
    intra_frac: float,
    max_dense_nodes: int,
    mem_gb: float,
    device: torch.device,
    seed: int,
) -> E2EResult:
    """Fit the real ``RecDMoN`` on a synthetic graph and time it end-to-end."""
    dense_bytes = n * n * 4  # float32 dense slice of the root split (host + device)
    emb_bytes = n * d * 4
    if max(dense_bytes, emb_bytes) > mem_gb * 2**30:
        raise _Skipped(f"N={n} D={d}: dense root slice {dense_bytes / 2**30:.1f} GiB > --mem-gb {mem_gb}")

    x_graph = _make_embeddings(n, d, seed)
    edge_index, edge_weight = _make_graph(n, avg_degree, n_blocks, intra_frac, seed)
    dataset = SimpleNamespace(x_graph=x_graph, edge_index=edge_index, edge_weight=edge_weight)

    config = RecDMoNModelConfig(
        recursive_epochs=epochs,
        assign_hidden_dim=hidden,
        dropout=0.0,
        max_dense_nodes=max(n, max_dense_nodes),
        recursive_factor=recursive_factor,
        recursive_levels=levels,
        lr=3e-4,
        weight_decay=1e-4,
        grad_clip=5.0,
    )
    model = RecDMoN(config).to(device)
    stats = _FitStats()

    torch.manual_seed(seed)
    rss_before = _peak_rss_kb()
    t0 = time.perf_counter()
    model.fit(dataset, callback=stats)
    _sync(device)
    fit_s = time.perf_counter() - t0
    rss_mb = (_peak_rss_kb() - rss_before) / 1024.0

    indices = model.indices
    cluster_counts = [int(len(np.unique(indices[:, i][indices[:, i] >= 0]))) for i in range(indices.shape[1])]

    # nnz of the symmetrized CSR (0.5 * (A + A.T), diagonal zeroed) without
    # rebuilding it: each distinct unordered pair contributes two entries.
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    u = np.minimum(src, dst)
    v = np.maximum(src, dst)
    pairs = np.unique((u.astype(np.int64) << 32) | v.astype(np.int64))
    adj_nnz = 2 * int(len(pairs))

    return E2EResult(
        n=n,
        d=d,
        recursive_factor=recursive_factor,
        recursive_levels=levels,
        recursive_epochs=epochs,
        avg_degree=avg_degree,
        n_edges=int(edge_index.size(1)),
        adj_nnz=adj_nnz,
        fit_s=fit_s,
        per_level_s=json.dumps({str(k): round(v, 4) for k, v in sorted(stats.per_level_s.items())}),
        n_splits=stats.n_splits,
        epochs_total=stats.epochs_total,
        avg_modularity=stats.modularity_sum / max(1, stats.epochs_total),
        max_subset=stats.max_subset,
        cluster_counts=json.dumps(cluster_counts),
        rss_mb=rss_mb,
    )


@dataclass
class KernelResult:
    subset_size: int
    d: int
    k: int
    epochs: int
    per_epoch_s: float
    dense_mb: float
    modularity: float
    balance: float
    entropy: float
    rss_mb: float


def _run_kernel_case(
    subset_size: int,
    d: int,
    k: int,
    epochs: int,
    hidden: int,
    mem_gb: float,
    device: torch.device,
    seed: int,
) -> KernelResult:
    """Time one dense local DMoN split: the per-parent cost that dominates a level."""
    dense_bytes = subset_size * subset_size * 4
    if dense_bytes > mem_gb * 2**30:
        raise _Skipped(f"subset={subset_size}: dense slice {dense_bytes / 2**30:.1f} GiB > --mem-gb {mem_gb}")

    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.standard_normal((subset_size, d)).astype(np.float32)).to(device)

    block_size = max(1, subset_size // k)
    src = torch.arange(subset_size, device=device)
    dst = (src // block_size) * block_size + torch.randint(0, block_size, (subset_size,), device=device)
    dst = torch.clamp(dst, 0, subset_size - 1)
    adj = torch.zeros(subset_size, subset_size, dtype=torch.float32, device=device)
    adj[src, dst] = 1.0
    adj = 0.5 * (adj + adj.T)
    adj.fill_diagonal_(0.0)

    model = LocalAssignNet(d, hidden, k, 0.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    degree = adj.sum(dim=1)
    m2 = degree.sum().clamp_min(1e-12)

    def step():
        opt.zero_grad()
        logits = model(x)
        loss, metrics = dense_dmon_local_loss(
            adj=adj, logits=logits, degree=degree, m2=m2, balance_weight=1.0, entropy_weight=0.01
        )
        loss.backward()
        opt.step()
        return loss, metrics

    for _ in range(3):  # warm-up (allocations, autotuning)
        step()
    _sync(device)

    rss_before = _peak_rss_kb()
    t0 = time.perf_counter()
    loss, metrics = None, None
    for _ in range(epochs):
        loss, metrics = step()
    _sync(device)
    per_epoch_s = (time.perf_counter() - t0) / epochs
    rss_mb = (_peak_rss_kb() - rss_before) / 1024.0

    return KernelResult(
        subset_size=subset_size,
        d=d,
        k=k,
        epochs=epochs,
        per_epoch_s=per_epoch_s,
        dense_mb=dense_bytes / 2**20,
        modularity=float(metrics["modularity"].item()),  # type: ignore[index]
        balance=float(metrics["balance"].item()),  # type: ignore[index]
        entropy=float(metrics["entropy"].item()),  # type: ignore[index]
        rss_mb=rss_mb,
    )


@dataclass
class GraphResult:
    n: int
    avg_degree: int
    n_blocks: int
    n_edges: int
    adj_nnz: int
    gen_s: float
    adj_s: float
    rss_mb: float


def _run_graph_case(
    n: int,
    avg_degree: int,
    n_blocks: int,
    intra_frac: float,
    seed: int,
) -> GraphResult:
    """Time the sparse stages (edge generation + CSR adjacency) for a large ``N``."""
    rss_before = _peak_rss_kb()
    t0 = time.perf_counter()
    edge_index, edge_weight = _make_graph(n, avg_degree, n_blocks, intra_frac, seed)
    gen_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    adj = edge_index_to_sparse_adj(edge_index, edge_weight, n)
    adj_s = time.perf_counter() - t0
    rss_mb = (_peak_rss_kb() - rss_before) / 1024.0

    return GraphResult(
        n=n,
        avg_degree=avg_degree,
        n_blocks=n_blocks,
        n_edges=int(edge_index.size(1)),
        adj_nnz=int(adj.nnz),
        gen_s=gen_s,
        adj_s=adj_s,
        rss_mb=rss_mb,
    )


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def _print_e2e(results: list[E2EResult], skipped: list[str]) -> None:
    header = (
        f"{'N':>9} {'D':>4} {'rf':>2} {'lv':>2} {'ep':>3} {'deg':>3} | "
        f"{'edges':>9} {'nnz':>10} {'fit_s':>8} {'splits':>6} {'maxSub':>7} "
        f"{'avgMod':>6} {'rssMB':>7} | clusters/level"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.n:>9} {r.d:>4} {r.recursive_factor:>2} {r.recursive_levels:>2} {r.recursive_epochs:>3} "
            f"{r.avg_degree:>3} | {r.n_edges:>9} {r.adj_nnz:>10} {r.fit_s:>8.3f} {r.n_splits:>6} "
            f"{r.max_subset:>7} {r.avg_modularity:>6.4f} {r.rss_mb:>7.1f} | {r.cluster_counts}"
        )
    if skipped:
        print("\nskipped:")
        for s in skipped:
            print(f"  {s}")


def _print_kernel(results: list[KernelResult], skipped: list[str]) -> None:
    header = (
        f"{'subset':>8} {'D':>4} {'k':>3} {'ep':>4} | {'perEpoch_s':>10} {'denseMB':>8} "
        f"{'mod':>7} {'bal':>7} {'ent':>7} {'rssMB':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.subset_size:>8} {r.d:>4} {r.k:>3} {r.epochs:>4} | {r.per_epoch_s:>10.4f} {r.dense_mb:>8.0f} "
            f"{r.modularity:>7.4f} {r.balance:>7.4f} {r.entropy:>7.4f} {r.rss_mb:>7.1f}"
        )
    if skipped:
        print("\nskipped:")
        for s in skipped:
            print(f"  {s}")


def _print_graph(results: list[GraphResult]) -> None:
    header = f"{'N':>10} {'deg':>4} {'blocks':>6} | {'edges':>10} {'nnz':>10} {'gen_s':>9} {'adj_s':>9} {'rssMB':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.n:>10} {r.avg_degree:>4} {r.n_blocks:>6} | {r.n_edges:>10} {r.adj_nnz:>10} "
            f"{r.gen_s:>9.3f} {r.adj_s:>9.3f} {r.rss_mb:>8.1f}"
        )


def _write_tsv(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(str(v) for v in row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="e2e", choices=["e2e", "kernel", "graph"])
    parser.add_argument("--device", default="auto", help="torch device (auto/npu/cuda/cpu)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="optional TSV output path")
    parser.add_argument("--quiet", action="store_true", help="silence tqdm progress bars")
    parser.add_argument("--mem-gb", type=float, default=200.0, help="dense-slice memory budget for e2e/kernel cases")
    parser.add_argument(
        "--max-dense-nodes", type=int, default=1000000, help="base max_dense_nodes (auto-raised to N when N is larger)"
    )

    # e2e sweep grid
    parser.add_argument("--ns", default="1000,5000,20000,100000,1000000", help="item counts (comma list)")
    parser.add_argument("--ds", default="32,128", help="embedding dims (comma list)")
    parser.add_argument("--rfs", default="2,4,8,16", help="recursive factors (comma list)")
    parser.add_argument("--levels", type=int, default=3, help="recursive levels")
    parser.add_argument("--epochs", type=int, default=30, help="DMoN epochs per local split")
    parser.add_argument("--hidden", type=int, default=128, help="assignment MLP hidden dim")

    # graph / e2e graph params
    parser.add_argument("--avg-degree", type=int, default=10, help="target average item degree")
    parser.add_argument("--n-blocks", type=int, default=64, help="planted item blocks for graph structure")
    parser.add_argument("--intra-frac", type=float, default=0.9, help="fraction of edges inside a block")

    # kernel sweep grid
    parser.add_argument("--subset-sizes", default="1000,5000,20000", help="kernel subset sizes (comma list)")
    parser.add_argument("--ks", default="2,4", help="kernel child-cluster counts (comma list)")
    args = parser.parse_args()

    if args.quiet:
        _silence_tqdm()
    device = get_device(args.device)
    print(f"device: {device}\n")

    if args.mode == "e2e":
        grid = itertools.product(_parse_int_list(args.ns), _parse_int_list(args.ds), _parse_int_list(args.rfs))
        results: list[E2EResult] = []
        skipped: list[str] = []
        for n, d, rf in grid:
            if rf < 2:
                skipped.append(f"N={n} D={d} rf={rf}: recursive_factor must be >= 2")
                continue
            print(f"e2e: N={n} D={d} rf={rf} levels={args.levels} epochs={args.epochs}", flush=True)
            try:
                r = _run_e2e_case(
                    n,
                    d,
                    rf,
                    args.levels,
                    args.epochs,
                    args.hidden,
                    args.avg_degree,
                    args.n_blocks,
                    args.intra_frac,
                    args.max_dense_nodes,
                    args.mem_gb,
                    device,
                    args.seed + n + d,
                )
            except _Skipped as e:
                skipped.append(str(e))
                continue
            results.append(r)
        _print_e2e(results, skipped)
        if args.out and results:
            header = [f.name for f in E2EResult.__dataclass_fields__.values()]  # type: ignore[attr-defined]
            _write_tsv(args.out, header, [[getattr(r, f) for f in header] for r in results])
            print(f"wrote {args.out}")

    elif args.mode == "kernel":
        grid = itertools.product(_parse_int_list(args.subset_sizes), _parse_int_list(args.ds), _parse_int_list(args.ks))
        kresults: list[KernelResult] = []
        kskipped: list[str] = []
        for n_sub, d, k in grid:
            if k > n_sub:
                kskipped.append(f"subset={n_sub} k={k}: k > subset")
                continue
            print(f"kernel: subset={n_sub} D={d} k={k} epochs={args.epochs}", flush=True)
            try:
                r = _run_kernel_case(n_sub, d, k, args.epochs, args.hidden, args.mem_gb, device, args.seed + n_sub)
            except _Skipped as e:
                kskipped.append(str(e))
                continue
            kresults.append(r)
        _print_kernel(kresults, kskipped)
        if args.out and kresults:
            header = [f.name for f in KernelResult.__dataclass_fields__.values()]  # type: ignore[attr-defined]
            _write_tsv(args.out, header, [[getattr(r, f) for f in header] for r in kresults])
            print(f"wrote {args.out}")

    else:  # graph
        gresults = [
            _run_graph_case(n, args.avg_degree, args.n_blocks, args.intra_frac, args.seed + n)
            for n in _parse_int_list(args.ns)
        ]
        _print_graph(gresults)
        if args.out and gresults:
            header = [f.name for f in GraphResult.__dataclass_fields__.values()]  # type: ignore[attr-defined]
            _write_tsv(args.out, header, [[getattr(r, f) for f in header] for r in gresults])
            print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
