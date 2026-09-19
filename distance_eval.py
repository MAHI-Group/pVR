"""
Distance-level evaluation on the benchmark datasets (R3.5, R4.5, R4.6).

For every pairwise distance and dataset this reports
  5-NN accuracy under the same repeated CV as the original baselines
    (10 seeds x 5 folds, identical splits),
  leave-one-out retrieval: precision at 1 and 5, mean average precision,
  adjusted Rand index of average-linkage clustering cut at the true
    number of classes.

Distances
  kmer      L1 between k-mer frequency vectors (D_c)
  padic     D_p as implemented; checked against its tree-Wasserstein form
  padic_w1  Wasserstein-1 between k-mer distributions under the p-adic
            ultrametric p^-lcp(u,v), in closed form on the prefix tree
  wham_w1   Wasserstein-1 under the weighted Hamming metric
            sum_{i: u_i != v_i} p^-i (Reviewer 3); needs POT
  length    |L_x - L_y|
  and every distance returned by baselines.compute_all_baselines.

Usage
  pip install pot
  python distance_eval.py --out results_abl/distance_eval.json
  python distance_eval.py --out results_abl/distance_eval.json --no-wham
"""

import argparse
import json
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wilcoxon
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

from baselines import compute_all_baselines
from pvr_eff import compute_distance_matrices, repeated_cv_distance
from run_ablation import load_dataset
from run_full_eff import autodetect_configs

COMPARE = [("padic", "kmer"), ("padic_w1", "kmer"), ("wham_w1", "padic_w1"),
           ("length", "kmer")]


def prefix_marginals(F, k):
    """Length-l prefix distributions, l = 1..k, from rows of k-mer frequencies
    ordered lexicographically with the first letter most significant."""
    n = F.shape[0]
    return [F.reshape(n, 4 ** l, 4 ** (k - l)).sum(axis=2) for l in range(1, k + 1)]


def tree_w1(F, k, weights):
    """sum_l w_l ||h_l(S) - h_l(T)||_1, the W1 closed form on the prefix tree."""
    D = np.zeros((F.shape[0], F.shape[0]))
    for w, H in zip(weights, prefix_marginals(F, k)):
        if w:
            D += w * squareform(pdist(H, metric="cityblock"))
    return D


def padic_weights(k, p):
    f = [float(p) ** -j for j in range(k)] + [0.0]
    return [(f[l - 1] - f[l]) / 2.0 for l in range(1, k + 1)]


def implemented_weights(k):
    J = min(k, 3)
    T = J * (J + 1) / 2.0
    return [l / T if l <= J else 0.0 for l in range(1, k + 1)]


def weighted_hamming_cost(k, p):
    idx = np.arange(4 ** k)
    digits = np.stack([(idx // 4 ** (k - 1 - i)) % 4 for i in range(k)], axis=1)
    weights = float(p) ** -np.arange(k)
    return ((digits[:, None, :] != digits[None, :, :]) * weights).sum(axis=2)


def _wham_row(i, F, M):
    import ot
    n = F.shape[0]
    out = np.zeros(n)
    a = F[i]
    ia = np.nonzero(a)[0]
    wa = a[ia] / a[ia].sum()
    for j in range(i + 1, n):
        b = F[j]
        ib = np.nonzero(b)[0]
        out[j] = ot.emd2(wa, b[ib] / b[ib].sum(),
                         np.ascontiguousarray(M[np.ix_(ia, ib)]),
                         numItermax=1_000_000)
    return i, out


def wham_w1(F, k, p, n_jobs):
    M = weighted_hamming_cost(k, p)
    rows = Parallel(n_jobs=n_jobs)(delayed(_wham_row)(i, F, M)
                                   for i in range(F.shape[0]))
    D = np.zeros((F.shape[0], F.shape[0]))
    for i, r in rows:
        D[i, i + 1:] = r[i + 1:]
    return D + D.T


def retrieval(D, y, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    p1, p5, aps = [], [], []
    ranks = np.arange(1, n)
    for i in range(n):
        d = D[i].astype(float).copy()
        d[i] = np.inf
        order = np.lexsort((rng.random(n), d))[: n - 1]
        rel = (y[order] == y[i]).astype(float)
        p1.append(rel[0])
        p5.append(rel[:5].mean())
        n_rel = rel.sum()
        if n_rel > 0:
            aps.append((rel * np.cumsum(rel) / ranks).sum() / n_rel)
    return float(np.mean(p1)), float(np.mean(p5)), float(np.mean(aps))


def clustering_ari(D, y):
    D = np.maximum((D + D.T) / 2.0, 0.0)
    np.fill_diagonal(D, 0.0)
    pred = AgglomerativeClustering(n_clusters=len(np.unique(y)), metric="precomputed",
                                   linkage="average").fit_predict(D)
    return float(adjusted_rand_score(y, pred))


def evaluate(D, labels, y, n_seeds):
    folds = repeated_cv_distance(D, labels, k=5, n_folds=5, n_seeds=n_seeds, base_seed=42)
    p1, p5, mAP = retrieval(D, y)
    return {"acc_mean": float(np.mean(folds)), "acc_std": float(np.std(folds)),
            "folds": [float(x) for x in folds], "p_at_1": p1, "p_at_5": p5,
            "map": mAP, "ari": clustering_ari(D, y)}


def summarise(results):
    names = [n for n in results if not n.startswith("_")]
    dists = sorted(set.intersection(*[{d for d in results[n] if not d.startswith("_")}
                                      for n in names]))
    print(f"\nMeans over {len(names)} datasets")
    print(f"{'distance':14s} {'5-NN acc':>9s} {'P@1':>7s} {'P@5':>7s} {'mAP':>7s} {'ARI':>7s}")
    for d in dists:
        m = {key: np.mean([results[n][d][key] for n in names])
             for key in ("acc_mean", "p_at_1", "p_at_5", "map", "ari")}
        print(f"{d:14s} {m['acc_mean']*100:9.1f} {m['p_at_1']*100:7.1f} "
              f"{m['p_at_5']*100:7.1f} {m['map']*100:7.1f} {m['ari']:7.3f}")

    print("\nSigned-rank across datasets (first minus second)")
    for a, b in COMPARE:
        if a not in dists or b not in dists:
            continue
        for key in ("acc_mean", "map"):
            diff = np.array([results[n][a][key] - results[n][b][key] for n in names]) * 100
            try:
                p = wilcoxon(diff, zero_method="wilcox").pvalue
            except ValueError:
                p = float("nan")
            print(f"  {a:9s} vs {b:9s} {key:8s} mean {diff.mean():+6.2f}  "
                  f"wins {int((diff > 0).sum())}/{len(diff)}  "
                  f"n_eff {int((diff != 0).sum())}  p={p:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadirs", nargs="+", default=["data", "data_large"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--k-ffp", type=int, default=3)
    ap.add_argument("--k-mash", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--no-wham", action="store_true")
    ap.add_argument("--min-class-count", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    results = {"_config": vars(args)}
    for datadir in args.datadirs:
        for cfg in autodetect_configs(datadir):
            name = cfg["name"]
            if args.datasets and name not in args.datasets:
                continue
            seqs, labels = load_dataset(cfg, args.min_class_count)
            if len(seqs) < 10:
                continue
            y = LabelEncoder().fit_transform(labels)
            print(f"\n{name}: N={len(seqs)}, classes={len(set(labels))}")
            t0 = time.time()

            D_p, D_c, F = compute_distance_matrices(seqs, k=args.k, p=args.p,
                                                    n_jobs=args.n_jobs)
            gap = float(np.abs(tree_w1(F, args.k, implemented_weights(args.k)) - D_p).max())
            print(f"  max |D_p - tree form| = {gap:.2e}")
            checks = {"dp_tree_gap": gap}

            dists = {"kmer": D_c, "padic": D_p,
                     "padic_w1": tree_w1(F, args.k, padic_weights(args.k, args.p))}
            if not args.no_wham:
                t1 = time.time()
                dists["wham_w1"] = wham_w1(F, args.k, args.p, args.n_jobs)
                pos = dists["padic_w1"] > 0
                ratio = dists["wham_w1"][pos] / dists["padic_w1"][pos]
                checks.update(wham_ratio_min=float(ratio.min()),
                              wham_ratio_max=float(ratio.max()),
                              bound=args.p / (args.p - 1))
                print(f"  wham_w1 / padic_w1 in [{ratio.min():.4f}, {ratio.max():.4f}], "
                      f"bound {args.p / (args.p - 1):.4f} ({time.time() - t1:.0f}s)")
            L = np.array([len(s) for s in seqs], dtype=float)
            dists["length"] = np.abs(L[:, None] - L[None, :])
            for bname, Db in compute_all_baselines(seqs, k_ffp=args.k_ffp,
                                                   k_mash=args.k_mash).items():
                dists[bname] = np.asarray(Db, dtype=float)

            results[name] = {"_checks": checks}
            print(f"  {'distance':14s} {'5-NN acc':>9s} {'P@1':>7s} {'P@5':>7s} "
                  f"{'mAP':>7s} {'ARI':>7s}")
            for dname, D in dists.items():
                r = evaluate(D, labels, y, args.seeds)
                results[name][dname] = r
                print(f"  {dname:14s} {r['acc_mean']*100:9.1f} {r['p_at_1']*100:7.1f} "
                      f"{r['p_at_5']*100:7.1f} {r['map']*100:7.1f} {r['ari']:7.3f}")
            print(f"  ({time.time() - t0:.0f}s)")

            with open(args.out, "w") as fh:
                json.dump(results, fh, indent=2)

    summarise(results)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
