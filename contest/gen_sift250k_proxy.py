#!/usr/bin/env python3
"""Build the 250K proxy datasets used in CONTEST.md section 5.4 / 6.2.

The official SIFT250K and Hidden250K sets are not distributed, so the 250K
measurements run on two proxies derived from SIFT1M:

  --mode 128   250K base rows sampled from SIFT1M, 4096 SIFT queries.
               Replaces SIFT250K (same neighbour geometry, same dimension).
  --mode 256   128 SIFT dims + 128 dims of N(0, (4*std*beta)^2) noise on both
               base and queries. The noise adds distance variance without
               signal, which tunes the baseline recall into the band where the
               official Hidden250K sits (r@10 0.985, r@100 0.914). beta=1.0 was
               used for the submitted numbers.

Ground truth is exact L2 (float64 accumulation; float32 matmul loses too many
top-100 ties on near-duplicate clusters).

Usage:
  python3 gen_sift250k_proxy.py --mode 128 --base sift_base.fvecs \\
      --query sift_query.fvecs --out data/sift250k
  python3 gen_sift250k_proxy.py --mode 256 --beta 1.0 --base sift_base.fvecs \\
      --query sift_query.fvecs --out data/s256
"""

import argparse
import os
import time

import numpy as np

N_BASE = 250000
N_QUERY = 4096
TOPK = 100


def read_fvecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    return raw.reshape(-1, dim + 1)[:, 1:].view(np.float32)


def exact_gt(base, query, topk, chunk=128):
    b64 = base.astype(np.float64)
    bn = (b64 * b64).sum(1)
    gt = np.empty((query.shape[0], topk), dtype=np.int32)
    for s in range(0, query.shape[0], chunk):
        e = min(s + chunk, query.shape[0])
        q64 = query[s:e].astype(np.float64)
        d = bn[None, :] - 2.0 * (q64 @ b64.T) + (q64 * q64).sum(1)[:, None]
        part = np.argpartition(d, topk, axis=1)[:, :topk]
        dd = np.take_along_axis(d, part, axis=1)
        gt[s:e] = np.take_along_axis(part, np.argsort(dd, axis=1), axis=1)
        del d
    ref = np.argsort(((b64 - query[0].astype(np.float64)) ** 2).sum(1))[:topk]
    if not np.array_equal(ref, gt[0]):
        raise SystemExit("ground-truth selfcheck failed on query 0")
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="SIFT1M sift_base.fvecs")
    ap.add_argument("--query", required=True, help="SIFT1M sift_query.fvecs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("128", "256"), default="128")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261030)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base = read_fvecs(args.base)
    query = read_fvecs(args.query)
    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(base.shape[0], min(N_BASE, base.shape[0]), replace=False))
    b = np.ascontiguousarray(base[pick])
    q = query[:N_QUERY]

    if args.mode == "256":
        sig = float(b.std()) * 4.0 * args.beta
        nb = rng.standard_normal((b.shape[0], 128)).astype(np.float32) * sig
        nq = rng.standard_normal((q.shape[0], 128)).astype(np.float32) * sig
        b = np.ascontiguousarray(np.hstack([b, nb]))
        q = np.ascontiguousarray(np.hstack([q, nq]))
        tag = "b%.1f" % args.beta
        base_path = os.path.join(args.out, "base_%s.f32" % tag)
        query_path = os.path.join(args.out, "query_%s.f32" % tag)
        gt_path = os.path.join(args.out, "gt_%s.i32" % tag)
    else:
        base_path = os.path.join(args.out, "base.f32")
        query_path = os.path.join(args.out, "query.f32")
        gt_path = os.path.join(args.out, "gt100.i32")

    b.tofile(base_path)
    q.tofile(query_path)
    t0 = time.time()
    gt = exact_gt(b, q, TOPK)
    gt.tofile(gt_path)
    if args.mode == "128":
        gt[:, :10].tofile(os.path.join(args.out, "gt10.i32"))
    print("base %s query %s gt %s (%.1fs)" % (b.shape, q.shape, gt.shape, time.time() - t0))


if __name__ == "__main__":
    main()
