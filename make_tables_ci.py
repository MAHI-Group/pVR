"""
Read all_results.json (small + large) and emit LaTeX-ready Tables 3 and 4
with mean +/- std, plus paired one-sided Wilcoxon p-values for the
headline comparisons.

Usage:
  python make_tables_ci.py --small results/all_results.json \
                           --large results_large/all_results.json \
                           --out tables_ci.tex
"""

import os
import json
import argparse
import numpy as np
from scipy.stats import wilcoxon


SMALL_ORDER = ["mammalian_mito", "sars_cov2", "hrv", "influenza_ha", "hev", "ebola"]
LARGE_ORDER = ["dengue_400", "ebola_100", "hev_150", "hrv_300", "influenza_300", "sars_cov2_500"]

SMALL_COLS = ["Mam.", "CoV-2", "HRV", "Inf.", "HEV", "Ebola"]
LARGE_COLS = ["Dengue", "Ebola", "HEV", "HRV", "Inf.", "CoV-2"]

BASELINES = ["ffp_js_5nn", "nvm_5nn", "minhash_5nn", "kmer_freq_5nn"]
BASELINE_LABELS = {"ffp_js_5nn": "FFP-JS", "nvm_5nn": "NVM",
                   "minhash_5nn": "MinHash", "kmer_freq_5nn": "$k$-mer freq"}
PVR_METHODS = ["pvr_5nn", "pvr_xgboost", "pvr_svm"]


def load(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def folds_of(cell):
    """Return fold list from a result cell, or None if absent."""
    if not cell:
        return None
    return cell.get("folds")


def fmt_cell(folds, bold=False):
    if not folds:
        return "--"
    m, s = 100 * np.mean(folds), 100 * np.std(folds)
    body = f"{m:.1f} $\\pm$ {s:.1f}"
    return f"\\textbf{{{body}}}" if bold else body


def best_pvr_cell(ds_main):
    """Return (label_suffix, fold_list) for whichever pVR variant has highest mean."""
    candidates = [(m, folds_of(ds_main.get(m))) for m in PVR_METHODS]
    candidates = [(m, f) for m, f in candidates if f]
    if not candidates:
        return None, None
    m_best, f_best = max(candidates, key=lambda x: np.mean(x[1]))
    suffix = {"pvr_5nn": r"\dagger", "pvr_xgboost": r"\ddagger",
              "pvr_svm": r"\S"}[m_best]
    return suffix, f_best


def render_table(title, label, cols, ds_order, results, datasets_label):
    n = len(cols)
    head = " & ".join(cols)
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\footnotesize",
        f"\\caption{{{title}. Each cell is mean $\\pm$ std over "
        f"{50} fold accuracies (10 seeds $\\times$ 5 folds). pVR row marks "
        r"best variant: $\dagger$~5-NN, $\ddagger$~XGBoost, $\S$~SVM.}",
        f"\\label{{{label}}}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l" + "c" * n + "}",
        r"\hline",
        f"Method & {head} \\\\",
        r"\hline",
    ]

    rows = []
    for b in BASELINES:
        cells = []
        for ds in ds_order:
            cell = results.get(ds, {}).get("main", {}).get(b)
            cells.append(fmt_cell(folds_of(cell)))
        rows.append((BASELINE_LABELS[b], cells, []))

    pvr_cells, pvr_suffixes = [], []
    for ds in ds_order:
        main = results.get(ds, {}).get("main", {})
        suf, folds = best_pvr_cell(main)
        pvr_cells.append((folds, suf))
        pvr_suffixes.append(suf)

    bold_mask = [False] * len(ds_order)
    for j, ds in enumerate(ds_order):
        main = results.get(ds, {}).get("main", {})
        baseline_means = []
        for b in BASELINES:
            f = folds_of(main.get(b))
            baseline_means.append(np.mean(f) if f else -1)
        pvr_folds = pvr_cells[j][0]
        pvr_mean = np.mean(pvr_folds) if pvr_folds else -1
        if pvr_mean >= max(baseline_means):
            bold_mask[j] = True

    pvr_str = []
    for j, (folds, suf) in enumerate(pvr_cells):
        if not folds:
            pvr_str.append("--")
            continue
        body = fmt_cell(folds, bold=bold_mask[j])
        pvr_str.append(f"{body}$^{{{suf}}}$")
    rows.append((r"\texttt{pVR}", pvr_str, []))

    for label_, cells, _ in rows:
        lines.append(f"{label_} & " + " & ".join(cells) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def headline_tests(results, ds_order):
    """One-sided paired Wilcoxon: pVR (best variant) > strongest baseline."""
    out = []
    for ds in ds_order:
        main = results.get(ds, {}).get("main", {})
        suf, pvr_folds = best_pvr_cell(main)
        if not pvr_folds:
            continue
        best_b, best_b_folds = None, None
        for b in BASELINES:
            f = folds_of(main.get(b))
            if not f:
                continue
            if best_b_folds is None or np.mean(f) > np.mean(best_b_folds):
                best_b, best_b_folds = b, f
        if not best_b_folds:
            continue
        n = min(len(pvr_folds), len(best_b_folds))
        a = np.array(pvr_folds[:n])
        b_arr = np.array(best_b_folds[:n])
        diff = a - b_arr
        if np.all(diff == 0):
            p = 1.0
        else:
            try:
                _, p = wilcoxon(a, b_arr, alternative="greater")
            except Exception:
                p = float("nan")
        out.append({
            "dataset": ds,
            "best_baseline": best_b,
            "pvr_mean": float(np.mean(pvr_folds)),
            "baseline_mean": float(np.mean(best_b_folds)),
            "delta_pp": float(100 * (np.mean(pvr_folds) - np.mean(best_b_folds))),
            "p_one_sided": float(p),
            "n_paired": int(n),
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", default="results/all_results.json")
    parser.add_argument("--large", default="results_large/all_results.json")
    parser.add_argument("--out", default="tables_ci.tex")
    args = parser.parse_args()

    small = load(args.small)
    large = load(args.large)

    parts = []
    if small:
        parts.append(render_table(
            "Classification accuracy (\\%) in the low-sample regime",
            "tab:main-small", SMALL_COLS, SMALL_ORDER, small, "low-sample",
        ))
    if large:
        parts.append(render_table(
            "Classification accuracy (\\%) in the large-sample regime",
            "tab:main-large", LARGE_COLS, LARGE_ORDER, large, "large-sample",
        ))

    with open(args.out, "w") as f:
        f.write("\n\n".join(parts) + "\n")
    print(f"Wrote {args.out}")

    print("\nHeadline paired tests (one-sided Wilcoxon, pVR > best baseline):")
    for regime, res, order in [("low-sample", small, SMALL_ORDER),
                                ("large-sample", large, LARGE_ORDER)]:
        if not res:
            continue
        print(f"\n  {regime}:")
        for row in headline_tests(res, order):
            print(f"    {row['dataset']:18s}  "
                  f"pVR={row['pvr_mean']*100:5.1f}  "
                  f"vs {row['best_baseline']:14s}={row['baseline_mean']*100:5.1f}  "
                  f"Delta={row['delta_pp']:+5.1f}pp  "
                  f"p={row['p_one_sided']:.4f}  "
                  f"(n={row['n_paired']})")


if __name__ == "__main__":
    main()
