"""
Ablation and evaluation-protocol study for pVR.

Feature blocks follow the paper's notation:
  a = degree profiles on the bi-filtered graph
  b = p-adic (prefix) multiscale histograms
  c = k-mer frequency vector

Protocols:
  transductive  grid thresholds and degrees computed on the full cohort (as submitted)
  inductive     grid thresholds and degrees computed within each training fold

Betti features are excluded. They are constant across sequences, so after
per-fold standardisation they are identically zero and only inflate the
feature dimension.

Usage:
  python run_ablation.py --datadir data --out results/ablation_main.json
"""

import argparse
import json
import time
from collections import Counter, defaultdict

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from pvr_eff import (
    compute_distance_matrices,
    make_grid,
    merge_small_classes,
    multiscale_padic_features,
)
from run_full_eff import (
    autodetect_configs,
    load_dataset_auto,
    load_dataset_with_labels,
)

FEATURE_BLOCKS = {
    "a": ("degree",),
    "b": ("padic",),
    "c": ("kmer",),
    "ac": ("degree", "kmer"),
    "bc": ("padic", "kmer"),
    "abc": ("degree", "padic", "kmer"),
}

GRAPH_FREE = {"a": False, "b": True, "c": True, "ac": False, "bc": True, "abc": False}


def degree_block(Dp, Dh, grid_p, grid_h, exclude_self):
    """Neighbour fraction of each row sequence against the reference set in the columns."""
    n_ref = Dp.shape[1] - 1 if exclude_self else Dp.shape[1]
    n_ref = max(n_ref, 1)
    out = np.empty((Dp.shape[0], len(grid_p) * len(grid_h)), dtype=np.float32)
    col = 0
    for eps_p in grid_p:
        near_p = Dp <= eps_p
        for eps_h in grid_h:
            adj = near_p & (Dh <= eps_h)
            if exclude_self:
                np.fill_diagonal(adj, False)
            out[:, col] = adj.sum(axis=1) / n_ref
            col += 1
    return out


def build_fold_features(blocks, train_idx, test_idx, D_p, D_H, padic_feats,
                        freq_vectors, protocol, G_p, G_h, deg_full=None):
    tr, te = [], []
    if "degree" in blocks:
        if protocol == "transductive":
            tr.append(deg_full[train_idx])
            te.append(deg_full[test_idx])
        else:
            Dp_tt = D_p[np.ix_(train_idx, train_idx)]
            Dh_tt = D_H[np.ix_(train_idx, train_idx)]
            grid_p, grid_h = make_grid(Dp_tt, Dh_tt, G_p, G_h)
            tr.append(degree_block(Dp_tt, Dh_tt, grid_p, grid_h, exclude_self=True))
            te.append(degree_block(D_p[np.ix_(test_idx, train_idx)],
                                   D_H[np.ix_(test_idx, train_idx)],
                                   grid_p, grid_h, exclude_self=False))
    if "padic" in blocks:
        tr.append(padic_feats[train_idx])
        te.append(padic_feats[test_idx])
    if "kmer" in blocks:
        tr.append(freq_vectors[train_idx])
        te.append(freq_vectors[test_idx])
    return np.hstack(tr), np.hstack(te)


def make_classifier(method, seed, n_train, n_jobs):
    if method == "xgboost":
        return xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                 eval_metric="mlogloss", verbosity=0,
                                 random_state=seed, n_jobs=n_jobs)
    if method == "svm":
        return SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                   random_state=seed)
    if method == "5nn":
        return KNeighborsClassifier(n_neighbors=max(1, min(5, n_train - 1)),
                                    metric="euclidean", n_jobs=n_jobs)
    raise ValueError(f"Unknown method: {method}")


def macro_auc(clf, X_test, y_test, n_classes):
    if not hasattr(clf, "predict_proba"):
        return float("nan")
    try:
        proba = clf.predict_proba(X_test)
        if n_classes == 2:
            return float(roc_auc_score(y_test, proba[:, 1]))
        return float(roc_auc_score(y_test, proba, multi_class="ovr",
                                   average="macro", labels=np.arange(n_classes)))
    except ValueError:
        return float("nan")


def run_cv(features_key, protocol, labels, D_p, D_H, padic_feats, freq_vectors,
           method="xgboost", n_folds=5, n_seeds=10, base_seed=42,
           G_p=10, G_h=15, n_jobs=4):
    blocks = FEATURE_BLOCKS[features_key]
    y = LabelEncoder().fit_transform(labels)
    n_folds = max(2, min(n_folds, min(Counter(y).values())))

    deg_full = None
    if "degree" in blocks and protocol == "transductive":
        grid_p, grid_h = make_grid(D_p, D_H, G_p, G_h)
        deg_full = degree_block(D_p, D_H, grid_p, grid_h, exclude_self=True)

    records = []
    for s in range(n_seeds):
        seed = base_seed + s
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            X_train, X_test = build_fold_features(
                blocks, train_idx, test_idx, D_p, D_H, padic_feats,
                freq_vectors, protocol, G_p, G_h, deg_full=deg_full,
            )
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            fold_le = LabelEncoder()
            y_train = fold_le.fit_transform(y[train_idx])
            keep = [i for i, yt in enumerate(y[test_idx]) if yt in fold_le.classes_]
            if not keep:
                continue
            y_test = fold_le.transform(y[test_idx][keep])
            X_test = X_test[keep]

            n_classes = len(fold_le.classes_)
            clf = make_classifier(method, seed, len(y_train), n_jobs)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)

            records.append({
                "features": features_key,
                "protocol": protocol,
                "method": method,
                "seed": seed,
                "fold": fold,
                "n_train": int(len(y_train)),
                "dim": int(X_train.shape[1]),
                "accuracy": float(accuracy_score(y_test, preds)),
                "macro_f1": float(f1_score(y_test, preds, average="macro",
                                           zero_division=0)),
                "macro_auc": macro_auc(clf, X_test, y_test, n_classes),
            })
    return records


def load_dataset(cfg, min_class_count):
    if cfg["loader"] == "tsv":
        sequences, labels, _ = load_dataset_with_labels(cfg["fasta"], cfg["labels"])
    else:
        sequences, labels, _ = load_dataset_auto(cfg["fasta"], cfg.get("label_func"))
    return sequences, merge_small_classes(labels, min_count=min_class_count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--features", nargs="+", default=["c", "bc", "ac", "abc"],
                    choices=sorted(FEATURE_BLOCKS))
    ap.add_argument("--protocols", nargs="+", default=["transductive", "inductive"],
                    choices=["transductive", "inductive"])
    ap.add_argument("--method", default="xgboost", choices=["xgboost", "svm", "5nn"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--Gp", type=int, default=10)
    ap.add_argument("--Gh", type=int, default=15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--min-class-count", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    configs = autodetect_configs(args.datadir)
    if args.datasets:
        configs = [c for c in configs if c["name"] in args.datasets]
    if not configs:
        print(f"No datasets found in {args.datadir}")
        return
    print(f"Datasets: {[c['name'] for c in configs]}")

    records, summary = [], defaultdict(dict)
    for cfg in configs:
        name = cfg["name"]
        sequences, labels = load_dataset(cfg, args.min_class_count)
        if len(sequences) < 10:
            print(f"{name}: only {len(sequences)} sequences, skipping")
            continue
        print(f"\n{name}: N={len(sequences)}, classes={len(set(labels))}, "
              f"k={args.k}, p={args.p}")

        t0 = time.time()
        D_p, D_H, freq_vectors = compute_distance_matrices(
            sequences, k=args.k, p=args.p, n_jobs=args.n_jobs,
        )
        padic_feats = multiscale_padic_features(sequences, args.k, args.p)
        print(f"  distances in {time.time() - t0:.1f}s")

        for protocol in args.protocols:
            for features_key in args.features:
                if protocol == "inductive" and GRAPH_FREE[features_key]:
                    continue
                t1 = time.time()
                out = run_cv(features_key, protocol, labels, D_p, D_H,
                             padic_feats, freq_vectors, method=args.method,
                             n_folds=args.folds, n_seeds=args.seeds,
                             base_seed=args.base_seed, G_p=args.Gp, G_h=args.Gh,
                             n_jobs=args.n_jobs)
                if not out:
                    continue
                accs = [r["accuracy"] for r in out]
                aucs = [r["macro_auc"] for r in out if not np.isnan(r["macro_auc"])]
                cell = f"{protocol}:{features_key}"
                summary[name][cell] = {
                    "acc_mean": float(np.mean(accs)),
                    "acc_std": float(np.std(accs)),
                    "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
                    "dim": out[0]["dim"],
                    "n_folds": len(accs),
                }
                print(f"  {protocol:12s} {features_key:4s} "
                      f"acc={np.mean(accs)*100:.1f} +/- {np.std(accs)*100:.1f}  "
                      f"dim={out[0]['dim']:4d}  ({time.time() - t1:.0f}s)")
                for r in out:
                    r["dataset"] = name
                    r["k"] = args.k
                    r["p"] = args.p
                records.extend(out)

    with open(args.out, "w") as fh:
        json.dump({"config": vars(args), "summary": summary, "records": records},
                  fh, indent=2)
    print(f"\nWrote {len(records)} fold records to {args.out}")


if __name__ == "__main__":
    main()
