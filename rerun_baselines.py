"""
Recompute only the baseline distance-matrix cells (FFP-JS, NVM, MinHash) and
patch them into an existing all_results.json, leaving every pVR, ablation,
sensitivity, and kmer_freq cell untouched.

Use this after fixing nvm_features (normalisation) and minhash_signature
(deterministic hashing) in baselines.py, so the manuscript's baseline rows can
be refreshed without recomputing the bi-filtration.

Protocol mirrors run_full_eff.py exactly: merge_small_classes(min_count=3),
sequences > 100 nt, repeated stratified CV with N_SEEDS x N_FOLDS, 5-NN on the
precomputed distance matrix, identical seeds.

Usage:
  python rerun_baselines.py --datadir data/       --outdir results/
  python rerun_baselines.py --datadir data_large/ --outdir results_large/
  # add --dry_run to print old-vs-new without writing the JSON
"""

import os
import json
import time
import argparse
import warnings
import numpy as np

from pvr_eff import repeated_cv_distance, merge_small_classes
from baselines import (
    ffp_jensen_shannon_distance,
    nvm_distance_matrix,
    minhash_distance_matrix,
)
from run_full_eff import (
    N_SEEDS, N_FOLDS, BASE_SEED,
    autodetect_configs, load_dataset_with_labels, load_dataset_auto,
)

warnings.filterwarnings("ignore")

# Maps the result-JSON baseline keys to a builder for the distance matrix.
# Only these keys are recomputed and overwritten; everything else is preserved.
BASELINE_BUILDERS = {
    "ffp_js": lambda seqs: ffp_jensen_shannon_distance(seqs, k=3),
    "nvm": lambda seqs: nvm_distance_matrix(seqs, k=1)[0],
    "minhash": lambda seqs: minhash_distance_matrix(seqs, k=7, n_hashes=200),
}


def cell(folds):
    if not folds:
        return {"acc": 0.0, "std": 0.0, "folds": []}
    return {"acc": float(np.mean(folds)),
            "std": float(np.std(folds)),
            "folds": [float(x) for x in folds]}


def load_sequences(cfg):
    if cfg["loader"] == "tsv":
        seqs, labels, _ = load_dataset_with_labels(cfg["fasta"], cfg["labels"])
    else:
        seqs, labels, _ = load_dataset_auto(cfg["fasta"], cfg.get("label_func"))
    labels = merge_small_classes(labels, min_count=3)
    return seqs, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="data")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print old-vs-new accuracies; do not write JSON.")
    args = parser.parse_args()

    out_path = os.path.join(args.outdir, "all_results.json")
    if not os.path.exists(out_path):
        raise SystemExit(f"No existing results at {out_path}; "
                         f"run run_full_eff.py first.")
    with open(out_path) as f:
        all_results = json.load(f)

    configs = autodetect_configs(args.datadir)
    if args.datasets:
        configs = [c for c in configs if c["name"] in args.datasets]
    if not configs:
        raise SystemExit(f"No datasets found in {args.datadir}")

    for cfg in configs:
        name = cfg["name"]
        if name not in all_results:
            print(f"[skip] {name}: not in existing JSON")
            continue
        if not os.path.exists(cfg["fasta"]):
            print(f"[skip] {name}: missing {cfg['fasta']}")
            continue

        seqs, labels = load_sequences(cfg)
        if len(seqs) < 10:
            print(f"[skip] {name}: only {len(seqs)} sequences")
            continue

        main_block = all_results[name]["main"]
        print(f"\n{name}: N={len(seqs)}, classes={len(set(labels))}")

        for key, build in BASELINE_BUILDERS.items():
            t0 = time.time()
            D = build(seqs)
            folds = repeated_cv_distance(
                D, labels, k=5,
                n_folds=N_FOLDS, n_seeds=N_SEEDS, base_seed=BASE_SEED,
            )
            new = cell(folds)
            result_key = f"{key}_5nn"
            old = main_block.get(result_key, {"acc": float("nan")})
            print(f"  {result_key:14s} old={old.get('acc', float('nan'))*100:5.1f}  "
                  f"new={new['acc']*100:5.1f} +/- {new['std']*100:4.1f}  "
                  f"({time.time()-t0:.1f}s)")
            if not args.dry_run:
                main_block[result_key] = new

    if args.dry_run:
        print("\n[dry run] no files written.")
        return

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nPatched baseline cells written to {out_path}")


if __name__ == "__main__":
    main()
