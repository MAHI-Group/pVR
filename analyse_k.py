"""
Summarise the k sweep and test the rule k = round(log_4 of median length).

Reads the run_ablation.py outputs written by run_k_sweep.sh, one file per
(regime, classifier, k). For each dataset, feature set and classifier it finds
the k with the highest mean accuracy, and compares accuracy at that k with
accuracy at the rule's k and at the fixed k = 4 of the original submission.
The Mash formula with q = 1/2 gives ceil(log_4 n); both are reported.

Usage:
  python analyse_k.py --indir results_abl/ksweep
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from run_ablation import load_dataset
from run_full_eff import autodetect_configs


def median_lengths(datadirs, min_class_count):
    out = {}
    for d in datadirs:
        for cfg in autodetect_configs(d):
            seqs, _ = load_dataset(cfg, min_class_count)
            if len(seqs) >= 10:
                out[cfg["name"]] = float(np.median([len(s) for s in seqs]))
    return out


def signed_rank(diff):
    try:
        return wilcoxon(diff, zero_method="wilcox").pvalue
    except ValueError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results_abl/ksweep")
    ap.add_argument("--cells", nargs="+",
                    default=["transductive:c", "transductive:bc", "inductive:abc"])
    ap.add_argument("--fixed-k", type=int, default=4)
    ap.add_argument("--min-class-count", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    acc = defaultdict(dict)
    datadirs = set()
    for path in sorted(glob.glob(os.path.join(args.indir, "*.json"))):
        d = json.load(open(path))
        cfg = d["config"]
        datadirs.add(cfg["datadir"])
        for ds, cells in d["summary"].items():
            for cell, v in cells.items():
                acc[(cfg["method"], cell, ds)][int(cfg["k"])] = 100.0 * v["acc_mean"]

    med = median_lengths(sorted(datadirs), args.min_class_count)
    methods = sorted({m for m, _, _ in acc})
    report = {}

    for method in methods:
        for cell in args.cells:
            keys = sorted(ds for m, c, ds in acc if m == method and c == cell)
            if not keys:
                continue
            print(f"\n{method}  {cell}")
            print(f"{'dataset':20s} {'median L':>9s} {'log4 L':>7s} {'rule':>5s} "
                  f"{'best k':>7s} {'acc best':>9s} {'acc rule':>9s} "
                  f"{'acc k=' + str(args.fixed_k):>9s}")
            rows = []
            for ds in keys:
                table = acc[(method, cell, ds)]
                ks = sorted(table)
                if ds not in med or args.fixed_k not in table:
                    continue
                log4 = np.log(med[ds]) / np.log(4)
                rule = int(min(max(round(log4), ks[0]), ks[-1]))
                mash = int(min(max(np.ceil(log4), ks[0]), ks[-1]))
                best = max(ks, key=lambda k: (table[k], -k))
                row = {"dataset": ds, "median_len": med[ds], "log4": float(log4),
                       "rule_k": rule, "mash_k": mash, "best_k": best,
                       "acc_best": table[best], "acc_rule": table.get(rule, np.nan),
                       "acc_mash": table.get(mash, np.nan),
                       "acc_fixed": table[args.fixed_k], "by_k": table}
                rows.append(row)
                print(f"{ds:20s} {med[ds]:9.0f} {log4:7.2f} {rule:5d} {best:7d} "
                      f"{row['acc_best']:9.2f} {row['acc_rule']:9.2f} "
                      f"{row['acc_fixed']:9.2f}")
            if len(rows) < 3:
                continue
            best_k = np.array([r["best_k"] for r in rows])
            log4 = np.array([r["log4"] for r in rows])
            reg_rule = np.array([r["acc_best"] - r["acc_rule"] for r in rows])
            reg_mash = np.array([r["acc_best"] - r["acc_mash"] for r in rows])
            reg_fixed = np.array([r["acc_best"] - r["acc_fixed"] for r in rows])
            gain = np.array([r["acc_rule"] - r["acc_fixed"] for r in rows])
            rho, p_rho = spearmanr(best_k, log4)
            print(f"  shortfall from best k: rule {reg_rule.mean():.2f}, "
                  f"Mash (q=1/2) {reg_mash.mean():.2f}, fixed k={args.fixed_k} "
                  f"{reg_fixed.mean():.2f}")
            print(f"  rule minus fixed: mean {gain.mean():+.2f}, "
                  f"wins {int((gain > 0).sum())}/{len(gain)}, "
                  f"n_eff {int((gain != 0).sum())}, p={signed_rank(gain):.4f}")
            print(f"  Spearman(best k, log4 L) = {rho:.2f} (p={p_rho:.3f})")
            report[f"{method}|{cell}"] = {
                "rows": rows, "shortfall_rule": float(reg_rule.mean()),
                "shortfall_mash": float(reg_mash.mean()),
                "shortfall_fixed": float(reg_fixed.mean()),
                "rule_minus_fixed": float(gain.mean()),
                "p_rule_vs_fixed": float(signed_rank(gain)),
                "spearman": float(rho), "spearman_p": float(p_rho)}

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
