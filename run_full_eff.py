"""
Run all pVR experiments with repeated stratified cross-validation.
Each main-table cell stores 50 fold accuracies (10 seeds x 5 folds) so
mean/std and paired Wilcoxon tests can be computed downstream.
Auto-detects datasets from --datadir.

Usage:
  python run_full_eff.py --datadir data/        --outdir results/
  python run_full_eff.py --datadir data_large/  --outdir results_large/ \
                         --n_jobs 2 --skip_sensitivity

Written by: Tirtharaj Dash (assistance from Claude Opus 4.x), Jan 2026
"""

import os
import glob
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from Bio import SeqIO

from pvr_eff import (
    clean_sequence,
    compute_distance_matrices,
    make_grid,
    combined_features,
    padic_only_features,
    hamming_only_features,
    bifiltration_features,
    multiscale_padic_features,
    classify_features,
    repeated_cv_features,
    repeated_cv_distance,
    merge_small_classes,
)
from baselines import compute_all_baselines

warnings.filterwarnings("ignore")


N_SEEDS = 10
N_FOLDS = 5
BASE_SEED = 42


def load_dataset_with_labels(fasta_path, label_path):
    records = {r.id: str(r.seq) for r in SeqIO.parse(fasta_path, "fasta")}
    labels_df = pd.read_csv(label_path, sep="\t")
    sequences, labels, names = [], [], []
    for _, row in labels_df.iterrows():
        acc = row["accession"]
        matched = [v for k, v in records.items() if acc in k]
        if matched:
            seq = clean_sequence(matched[0])
            if len(seq) > 100:
                sequences.append(seq)
                labels.append(row["label"])
                names.append(acc)
    return sequences, np.array(labels), names


def assign_hrv_label(description):
    desc = description.upper()
    if "RHINOVIRUS C" in desc or "HRV-C" in desc or "HRVC" in desc:
        return "C"
    if "RHINOVIRUS B" in desc or "HRV-B" in desc or "HRVB" in desc:
        return "B"
    return "A"


def load_dataset_auto(fasta_path, label_func=None):
    records = list(SeqIO.parse(fasta_path, "fasta"))
    sequences, labels, names = [], [], []
    for r in records:
        seq = clean_sequence(str(r.seq))
        if len(seq) > 100:
            label = label_func(r.description) if label_func else "unknown"
            sequences.append(seq)
            labels.append(label)
            names.append(r.id)
    return sequences, np.array(labels), names


def autodetect_configs(datadir):
    configs = []
    for fasta in sorted(glob.glob(os.path.join(datadir, "*.fasta"))):
        name = os.path.splitext(os.path.basename(fasta))[0]
        labels_tsv = os.path.join(datadir, f"{name}_labels.tsv")
        if os.path.exists(labels_tsv):
            configs.append({"name": name, "fasta": fasta,
                            "labels": labels_tsv, "loader": "tsv"})
        elif name.startswith("hrv"):
            configs.append({"name": name, "fasta": fasta, "loader": "auto",
                            "label_func": assign_hrv_label})
    return configs


def cell(folds):
    """Pack a fold list into a JSON-friendly dict."""
    if not folds:
        return {"acc": 0.0, "std": 0.0, "folds": []}
    return {"acc": float(np.mean(folds)),
            "std": float(np.std(folds)),
            "folds": [float(x) for x in folds]}


def run_main(sequences, labels, k=4, p=5, G_p=10, G_h=15,
             n_jobs=-1, verbose=True):
    results = {"timing": {}}

    t0 = time.time()
    D_p, D_H, freq_vectors = compute_distance_matrices(
        sequences, k=k, p=p, n_jobs=n_jobs
    )
    results["timing"]["pvr_distances"] = round(time.time() - t0, 2)

    grid_p, grid_h = make_grid(D_p, D_H, G_p=G_p, G_h=G_h)

    t0 = time.time()
    feats, betti_grid = combined_features(
        sequences, D_p, D_H, grid_p, grid_h, freq_vectors,
        k=k, p=p, verbose=verbose, n_jobs=n_jobs,
    )
    results["timing"]["pvr_features"] = round(time.time() - t0, 2)

    if verbose:
        print(f"  [pVR main, repeated CV: {N_SEEDS} seeds x {N_FOLDS} folds]")
    for method in ["xgboost", "svm", "5nn"]:
        folds = repeated_cv_features(
            feats, labels, method=method,
            n_folds=N_FOLDS, n_seeds=N_SEEDS, base_seed=BASE_SEED,
        )
        results[f"pvr_{method}"] = cell(folds)
        if verbose:
            c = results[f"pvr_{method}"]
            print(f"    pvr_{method}: {c['acc']*100:.1f} +/- {c['std']*100:.1f}")

    if verbose:
        print(f"  [Baselines, repeated CV]")
    t0 = time.time()
    baseline_dists = compute_all_baselines(sequences, k_ffp=3, k_mash=7)
    results["timing"]["baselines"] = round(time.time() - t0, 2)
    for name, D in baseline_dists.items():
        folds = repeated_cv_distance(
            D, labels, k=5,
            n_folds=N_FOLDS, n_seeds=N_SEEDS, base_seed=BASE_SEED,
        )
        results[f"{name}_5nn"] = cell(folds)
        if verbose:
            c = results[f"{name}_5nn"]
            print(f"    {name}_5nn: {c['acc']*100:.1f} +/- {c['std']*100:.1f}")

    for method in ["xgboost", "svm", "5nn"]:
        folds = repeated_cv_features(
            freq_vectors, labels, method=method,
            n_folds=N_FOLDS, n_seeds=N_SEEDS, base_seed=BASE_SEED,
        )
        results[f"kmer_freq_{method}"] = cell(folds)
        if verbose:
            c = results[f"kmer_freq_{method}"]
            print(f"    kmer_freq_{method}: {c['acc']*100:.1f} +/- {c['std']*100:.1f}")

    if verbose:
        print("  [Ablations, repeated CV, XGBoost]")
    pf = padic_only_features(D_p, grid_p)
    results["abl_padic_vr"] = cell(repeated_cv_features(
        pf, labels, method="xgboost",
        n_folds=N_FOLDS, n_seeds=N_SEEDS, base_seed=BASE_SEED,
    ))

    hf = hamming_only_features(D_H, grid_h)
    results["abl_hamming_vr"] = cell(repeated_cv_features(
        hf, labels, method="xgboost",
        n_folds=N_FOLDS, n_seeds=N_SEEDS, base_seed=BASE_SEED,
    ))

    topo_feats, _, _ = bifiltration_features(
        D_p, D_H, grid_p, grid_h, verbose=False, n_jobs=n_jobs
    )
    results["abl_bifilt_topo"] = cell(repeated_cv_features(
        topo_feats, labels, method="xgboost",
        n_folds=N_FOLDS, n_seeds=N_SEEDS, base_seed=BASE_SEED,
    ))

    padic_h = multiscale_padic_features(sequences, k, p)
    results["abl_padic_hist"] = cell(repeated_cv_features(
        padic_h, labels, method="xgboost",
        n_folds=N_FOLDS, n_seeds=N_SEEDS, base_seed=BASE_SEED,
    ))

    return results, D_p, D_H, betti_grid, grid_p, grid_h


def run_sensitivity(sequences, labels, k_values=(3, 4, 5, 6),
                    n_jobs=-1, verbose=True):
    """Single-seed CV across k. Used only for the sensitivity table."""
    out = {}
    for k in k_values:
        if verbose:
            print(f"  Sensitivity: k={k}")
        try:
            D_p, D_H, fv = compute_distance_matrices(
                sequences, k=k, p=5, n_jobs=n_jobs
            )
            grid_p, grid_h = make_grid(D_p, D_H, G_p=8, G_h=10)
            feats, _, _ = bifiltration_features(
                D_p, D_H, grid_p, grid_h, verbose=False, n_jobs=n_jobs
            )
            full, _ = combined_features(
                sequences, D_p, D_H, grid_p, grid_h, fv,
                k=k, p=5, verbose=False, n_jobs=n_jobs,
            )
            acc, std, _, _, folds = classify_features(
                full, labels, method="xgboost",
                n_folds=N_FOLDS, seed=BASE_SEED,
            )
            out[f"k{k}_p5"] = {"acc": acc, "std": std, "k": k,
                              "folds": [float(x) for x in folds]}
            if verbose:
                print(f"    acc={acc*100:.1f} +/- {std*100:.1f}")
        except Exception as e:
            out[f"k{k}_p5"] = {"error": str(e), "k": k}
            print(f"    FAILED: {e}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="data")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--Gp", type=int, default=10)
    parser.add_argument("--Gh", type=int, default=15)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Restrict to a subset of dataset names.")
    parser.add_argument("--skip_sensitivity", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "all_results.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    configs = autodetect_configs(args.datadir)
    if args.datasets:
        configs = [c for c in configs if c["name"] in args.datasets]
    if not configs:
        print(f"No datasets found in {args.datadir}")
        return

    print(f"Found {len(configs)} dataset(s): "
          f"{[c['name'] for c in configs]}")

    for cfg in configs:
        name = cfg["name"]
        print(f"\n{'='*60}\nDATASET: {name}\n{'='*60}")
        if not os.path.exists(cfg["fasta"]):
            print(f"  Missing {cfg['fasta']}, skipping.")
            continue

        if cfg["loader"] == "tsv":
            sequences, labels, _ = load_dataset_with_labels(
                cfg["fasta"], cfg["labels"]
            )
        else:
            sequences, labels, _ = load_dataset_auto(
                cfg["fasta"], cfg.get("label_func")
            )
        labels = merge_small_classes(labels, min_count=3)

        if len(sequences) < 10:
            print(f"  Only {len(sequences)} sequences, skipping.")
            continue
        print(f"  N={len(sequences)}, classes={len(set(labels))}")

        ds_results = all_results.get(name, {})

        t_total = time.time()
        main_res, D_p, D_H, betti_grid, grid_p, grid_h = run_main(
            sequences, labels,
            k=args.k, p=args.p, G_p=args.Gp, G_h=args.Gh,
            n_jobs=args.n_jobs, verbose=True,
        )
        main_res["timing"]["pvr_total"] = round(time.time() - t_total, 2)
        ds_results["main"] = main_res

        if not args.skip_sensitivity:
            print("  [Sensitivity scan over k]")
            ds_results["sensitivity"] = run_sensitivity(
                sequences, labels, n_jobs=args.n_jobs, verbose=True,
            )

        np.save(os.path.join(args.outdir, f"{name}_D_p.npy"), D_p)
        np.save(os.path.join(args.outdir, f"{name}_D_H.npy"), D_H)
        np.save(os.path.join(args.outdir, f"{name}_betti_grid.npy"), betti_grid)

        all_results[name] = ds_results
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved checkpoint to {out_path}")

    print(f"\nAll done. Results in {out_path}")


if __name__ == "__main__":
    main()
