#!/usr/bin/env python3
"""gen_dataset.py — deterministic synthetic ANN datasets with brute-force ground truth.

Writes <out>/base.f32 (n x dim float32), <out>/query.f32 (nq x dim),
<out>/gt100.i32 (nq x 100 int32 row indices), <out>/gt10.i32 (nq x 10) and
<out>/meta.json (generation parameters, sha256 of every file, recall-saturation
diagnostics).

Nothing here reads or modifies /seekdb_workspace/datasets: the contest dev data
is left untouched, and every dataset gets its own directory.

Hardness intent: the lab's existing 10K normalised-Gaussian set is recall
saturated (1.0 for every candidate), which gives a parameter sweep zero
resolution.  The "cluster" profile below builds many tight, near-degenerate
neighbourhoods so that reduced graph degrees / iterations show measurable
recall loss.

Usage:
  /opt/venv/bin/python3 gen_dataset.py --out /root/param-search-data/pilot-25k-256d \
      --n 25000 --dim 256 --nq 1024 --profile cluster --clusters 500 \
      --sigma-min 0.003 --sigma-max 0.25 --seed 1314
"""
import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_base(profile: str, n: int, dim: int, clusters: int, sigma_min: float,
              sigma_max: float, rng: np.random.Generator) -> np.ndarray:
    if profile == "gauss":
        data = rng.normal(size=(n, dim)).astype(np.float32)
    elif profile == "cluster":
        k = max(1, clusters)
        centers = rng.normal(size=(k, dim)).astype(np.float32)
        # log-uniform per-cluster spread: some clusters are extremely tight, so
        # many queries have near-equal distances to several base vectors.
        sigmas = np.exp(rng.uniform(np.log(sigma_min), np.log(sigma_max), size=k)).astype(np.float32)
        assign = rng.integers(0, k, size=n)
        noise = rng.normal(size=(n, dim)).astype(np.float32)
        data = centers[assign] + noise * sigmas[assign][:, None]
        data = data.astype(np.float32)
    else:
        raise SystemExit(f"unknown profile {profile}")
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (data / norms).astype(np.float32)


def make_queries(profile: str, base: np.ndarray, nq: int, dim: int, clusters: int,
                 sigma_min: float, sigma_max: float, jitter: float, rng: np.random.Generator) -> np.ndarray:
    if jitter > 0:
        # query = a base row + jitter: easy probes, used only when explicitly asked
        idx = np.linspace(0, base.shape[0] - 1, nq, dtype=np.int64)
        q = base[idx] + rng.normal(scale=jitter, size=(nq, dim)).astype(np.float32)
        q = q.astype(np.float32)
    else:
        q = make_base(profile, nq, dim, clusters, sigma_min, sigma_max, rng)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (q / norms).astype(np.float32)


def brute_force_gt(base: np.ndarray, queries: np.ndarray, topk: int, chunk: int = 256) -> np.ndarray:
    """Exact L2 top-k by chunked GEMM + argpartition (float64 accumulation)."""
    n = base.shape[0]
    b64 = base.astype(np.float64)
    b_sq = (b64 * b64).sum(axis=1)
    gt = np.empty((queries.shape[0], topk), dtype=np.int32)
    for start in range(0, queries.shape[0], chunk):
        stop = min(start + chunk, queries.shape[0])
        q64 = queries[start:stop].astype(np.float64)
        dots = q64 @ b64.T
        dist = (q64 * q64).sum(axis=1)[:, None] + b_sq[None, :] - 2.0 * dots
        part = np.argpartition(dist, kth=topk - 1, axis=1)[:, :topk]
        cols = np.take_along_axis(dist, part, axis=1)
        order = np.argsort(cols, axis=1, kind="stable")
        gt[start:stop] = np.take_along_axis(part, order, axis=1)
    return gt


def verify_gt(base: np.ndarray, queries: np.ndarray, gt: np.ndarray, sample: int = 8) -> dict:
    """Independent float32 full-sort check on a few queries + tie/degeneracy report."""
    rng = np.random.default_rng(7)
    idx = rng.choice(queries.shape[0], size=min(sample, queries.shape[0]), replace=False)
    max_err = 0.0
    for qi in idx:
        d = ((base - queries[qi]) ** 2).sum(axis=1)
        exact = np.argsort(d, kind="stable")[: gt.shape[1]]
        # set equality (ties may reorder) + distance multiset equality
        max_err = max(max_err, float(np.abs(np.sort(d[exact]) - np.sort(d[gt[qi]])).max()))
    d0 = ((base - queries[0]) ** 2).sum(axis=1)
    top = np.sort(d0)[: gt.shape[1]]
    gaps = np.diff(top)
    return {
        "gt_selfcheck_max_distance_mismatch": max_err,
        "query0_knn_distance_min": float(top[0]),
        "query0_knn_distance_max": float(top[-1]),
        "query0_tied_at_cutoff": int((gaps == 0).sum()),
        "query0_median_gap_over_cutoff": float(np.median(gaps)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--nq", type=int, default=1024)
    ap.add_argument("--profile", default="cluster", choices=["cluster", "gauss"])
    ap.add_argument("--clusters", type=int, default=500)
    ap.add_argument("--sigma-min", type=float, default=0.003)
    ap.add_argument("--sigma-max", type=float, default=0.25)
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="if >0, queries are base rows + this noise (easy probes)")
    ap.add_argument("--seed", type=int, default=1314)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--verify-sample", type=int, default=8)
    args = ap.parse_args()

    if not sys.version_info >= (3, 8):
        raise SystemExit("python >= 3.8 required")

    out = pathlib.Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty dir {out}")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    base = make_base(args.profile, args.n, args.dim, args.clusters, args.sigma_min, args.sigma_max, rng)
    queries = make_queries(args.profile, base, args.nq, args.dim, args.clusters,
                           args.sigma_min, args.sigma_max, args.jitter, rng)
    gen_s = time.time() - t0

    t1 = time.time()
    gt = brute_force_gt(base, queries, args.topk)
    gt_s = time.time() - t1

    base.tofile(out / "base.f32")
    queries.tofile(out / "query.f32")
    gt.tofile(out / f"gt{args.topk}.i32")
    gt[:, :10].tofile(out / "gt10.i32")

    meta = {
        "dir": str(out),
        "profile": args.profile,
        "n": args.n, "dim": args.dim, "nq": args.nq, "topk": args.topk,
        "clusters": args.clusters, "sigma_min": args.sigma_min, "sigma_max": args.sigma_max,
        "jitter": args.jitter, "seed": args.seed,
        "numpy": np.__version__,
        "generation_seconds": round(gen_s, 2), "gt_seconds": round(gt_s, 2),
        "sha256": {p.name: sha256(p) for p in sorted(out.glob("*")) if p.is_file()},
        "diagnostics": verify_gt(base, queries, gt, args.verify_sample),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
