#!/usr/bin/env python
"""Scaling / parity comparison harness for torch-native vs faiss-cpu PQ and OPQ.

At every scale in the sweep (N x D x M x nbits) it verifies the torch backend

* reproduces faiss codes bit-for-bit when given the *same* trained state
  (strict parity anchor, must be exactly 1.0), and
* reaches faiss-equivalent quantization distortion end-to-end on the same data
  (within a +/-15% band), and

and reports train/encode wall-time and peak-RSS deltas per backend so the
CPU-vs-CPU and (later) CPU-vs-GPU/NPU gap can be assessed as the parameters
scale.

The torch backend runs on ``--device`` (auto = NPU > CUDA > CPU); faiss always
runs on CPU. Usage::

    python scripts/pq_opq_scaling.py --device=cpu --methods=pq,opq
    python scripts/pq_opq_scaling.py --ns 2000,50000 --ds 32,128 --ms 8,16 \
        --nbitss 4,8 --out /tmp/pq_scaling.tsv

Exit code is non-zero if any hard parity / distortion check fails.
"""

import argparse
import itertools
import resource
import sys
import time
from dataclasses import dataclass

import torch

from gensid.nn.pq import make_opq, make_pq
from gensid.utils import get_device

try:
    from gensid.nn.utils import _get_faiss
except ImportError:  # pragma: no cover - faiss optional
    _get_faiss = None


def _peak_rss_kb() -> int:
    """Current process peak RSS in KiB (Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _make_data(n: int, d: int, seed: int) -> torch.Tensor:
    """Deterministic float32 gaussian data for one (N, D) scale."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=gen, dtype=torch.float32)


def _timeit(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


@dataclass
class CaseResult:
    method: str
    n: int
    d: int
    m: int
    nbits: int
    strict_code_match: float
    e2e_code_match: float
    decode_max_err: float
    distortion_torch: float
    distortion_faiss: float
    distortion_ratio: float
    train_s_torch: float
    train_s_faiss: float
    encode_s_torch: float
    encode_s_faiss: float
    rss_mb_torch: float
    rss_mb_faiss: float

    def ok(self, tol_ratio: float = 0.15, decode_atol: float = 1e-4) -> bool:
        """All hard checks pass.

        ``strict_code_match`` (the parity anchor) must be 1.0 exactly; the
        distortion band is a quality-equivalence check -- both backends are
        stochastic and observed run-to-run spread is <10%, so a +/-15% band
        catches genuine quality gaps (a broken fit would land far outside)
        without flagging single-configuration variance in either direction.
        """
        code_ok = self.strict_code_match >= 1.0 - 1e-9
        ratio_ok = (1.0 - tol_ratio) <= self.distortion_ratio <= (1.0 + tol_ratio)
        decode_ok = self.decode_max_err <= decode_atol
        return code_ok and ratio_ok and decode_ok


def _run_case(
    method: str,
    data: torch.Tensor,
    d: int,
    m: int,
    nbits: int,
    device: torch.device,
    seed: int,
    kmeans_iters: int,
    opq_niter: int,
    encode_batch: int | None,
) -> CaseResult:
    """Run one (method, N, D, M, nbits) case and return its parity/scaling metrics."""
    n = data.shape[0]
    make = make_pq if method == "pq" else make_opq

    # 1) faiss fit: gives the reference state (centroids [+ rotation]) and e2e
    #    faiss distortion, plus faiss train time and peak-RSS delta.
    faiss_kwargs = {} if method == "pq" else {"niter": opq_niter}
    faiss_q = make("faiss", d, m, nbits, **faiss_kwargs)
    rss_before = _peak_rss_kb()
    _, train_s_faiss = _timeit(lambda: faiss_q.fit(data))
    rss_faiss = (_peak_rss_kb() - rss_before) / 1024.0
    distortion_faiss = faiss_q.distortion(data)

    # 2) strict parity anchor: inject the faiss-trained state into the torch
    #    backend; codes must match bit-for-bit and decode must agree closely.
    torch_q = make("torch", d, m, nbits, device=device, **faiss_kwargs)
    if method == "pq":
        torch_q.set_centroids(faiss_q.centroids)
    else:
        torch_q.set_state(faiss_q.A, faiss_q.centroids)
    codes_torch = torch_q.encode(data, encode_batch)
    codes_faiss = faiss_q.encode(data)
    strict_code_match = float((codes_torch.cpu() == codes_faiss).float().mean())
    recon_torch = torch_q.decode(codes_torch)
    recon_faiss = faiss_q.decode(codes_faiss)
    decode_max_err = float((recon_torch.cpu() - recon_faiss).abs().max())

    # 3) end-to-end torch fit on the same data (fresh, seeded): distortion must
    #    land in the distortion band of faiss's (see ``CaseResult.ok``); also
    #    capture train/encode time and peak-RSS delta for the scaling comparison.
    torch.manual_seed(seed)
    torch_q = make("torch", d, m, nbits, device=device, kmeans_iters=kmeans_iters, **faiss_kwargs)
    rss_before = _peak_rss_kb()
    _, train_s_torch = _timeit(lambda: torch_q.fit(data, encode_batch))
    rss_torch = (_peak_rss_kb() - rss_before) / 1024.0
    distortion_torch = torch_q.distortion(data, encode_batch)
    codes_torch_e2e = torch_q.encode(data, encode_batch)
    e2e_code_match = float((codes_torch_e2e.cpu() == codes_faiss).float().mean())
    _, encode_s_torch = _timeit(lambda: torch_q.encode(data, encode_batch))
    _, encode_s_faiss = _timeit(lambda: faiss_q.encode(data))
    distortion_ratio = distortion_torch / distortion_faiss

    return CaseResult(
        method=method,
        n=n,
        d=d,
        m=m,
        nbits=nbits,
        strict_code_match=strict_code_match,
        e2e_code_match=e2e_code_match,
        decode_max_err=decode_max_err,
        distortion_torch=distortion_torch,
        distortion_faiss=distortion_faiss,
        distortion_ratio=distortion_ratio,
        train_s_torch=train_s_torch,
        train_s_faiss=train_s_faiss,
        encode_s_torch=encode_s_torch,
        encode_s_faiss=encode_s_faiss,
        rss_mb_torch=rss_torch,
        rss_mb_faiss=rss_faiss,
    )


def _print_table(results: list[CaseResult], device: torch.device) -> None:
    header = (
        f"{'method':<4} {'N':>7} {'D':>4} {'M':>3} {'nbits':>5} | "
        f"{'strict':>6} {'e2e':>6} {'decErr':>8} {'distT':>9} {'distF':>9} {'ratio':>6} | "
        f"{'trainT':>7} {'trainF':>7} {'encT':>7} {'encF':>7} {'rssT':>7} {'rssF':>7} | ok"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        ok = "PASS" if r.ok() else "FAIL"
        print(
            f"{r.method:<4} {r.n:>7} {r.d:>4} {r.m:>3} {r.nbits:>5} | "
            f"{r.strict_code_match:>6.4f} {r.e2e_code_match:>6.4f} {r.decode_max_err:>8.2e} "
            f"{r.distortion_torch:>9.4f} {r.distortion_faiss:>9.4f} {r.distortion_ratio:>6.3f} | "
            f"{r.train_s_torch:>7.3f} {r.train_s_faiss:>7.3f} {r.encode_s_torch:>7.4f} "
            f"{r.encode_s_faiss:>7.4f} {r.rss_mb_torch:>7.1f} {r.rss_mb_faiss:>7.1f} | {ok}"
        )
    print(f"\ntorch backend device: {device}")


def _write_tsv(results: list[CaseResult], path: str) -> None:
    fields = [f.name for f in CaseResult.__dataclass_fields__.values()]  # type: ignore[attr-defined]
    with open(path, "w") as fh:
        fh.write("\t".join(fields) + "\n")
        for r in results:
            fh.write("\t".join(str(getattr(r, f)) for f in fields) + "\n")


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="torch backend device (auto/npu/cuda/cpu)")
    parser.add_argument("--methods", default="pq,opq", help="comma list: pq,opq")
    parser.add_argument("--ns", default="2000,20000", help="comma list of item counts")
    parser.add_argument("--ds", default="32,64,128", help="comma list of dims")
    parser.add_argument("--ms", default="4,8", help="comma list of sub-quantizer counts")
    parser.add_argument("--nbitss", default="4,8", help="comma list of code sizes (bits per sub-quantizer)")
    parser.add_argument(
        "--kmeans-iters", type=int, default=25, help="torch k-means iterations (faiss PQ is fixed at 25)"
    )
    parser.add_argument("--opq-niter", type=int, default=25, help="OPQ outer iterations (both backends)")
    parser.add_argument("--encode-batch", type=int, default=4096, help="torch encode batch size")
    parser.add_argument("--seed", type=int, default=0, help="torch seed for the end-to-end fit")
    parser.add_argument("--out", default=None, help="optional TSV output path")
    args = parser.parse_args()

    if _get_faiss is None:
        print("faiss is required for this comparison harness; pip install faiss-cpu", file=sys.stderr)
        return 2
    device = get_device(args.device)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    grid = itertools.product(
        _parse_int_list(args.ns), _parse_int_list(args.ds), _parse_int_list(args.ms), _parse_int_list(args.nbitss)
    )

    results: list[CaseResult] = []
    n_fail = 0
    n_skip = 0
    for n, d, m, nbits in grid:
        if d % m != 0:
            n_skip += 1
            continue
        ksub = 1 << nbits
        if n < 4 * ksub:  # faiss warns (and may degenerate) when N ~ ksub
            print(f"skip N={n} D={d} M={m} nbits={nbits}: N < 4*ksub ({4 * ksub})")
            n_skip += 1
            continue
        data = _make_data(n, d, args.seed + n + d)
        for method in methods:
            print(f"running {method}: N={n} D={d} M={m} nbits={nbits}", flush=True)
            res = _run_case(
                method, data, d, m, nbits, device, args.seed, args.kmeans_iters, args.opq_niter, args.encode_batch
            )
            results.append(res)
            if not res.ok():
                n_fail += 1

    if results:
        _print_table(results, device)
        if args.out:
            _write_tsv(results, args.out)
            print(f"wrote {args.out}")
    print(f"\ncases: {len(results)} run, {n_skip} skipped, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
