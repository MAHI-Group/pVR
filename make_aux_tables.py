"""
Regenerate Tables 7 (feature importance), 8 (runtime), and 9 (sensitivity)
from the results JSONs and (for Table 7) by retraining XGBoost on the full
combined features.

Usage:
  python make_aux_tables.py \
      --small_results results/all_results.json \
      --large_results results_large/all_results.json \
      --small_data data/ \
      --large_data data_large/ \
      --out_runtime tables_runtime.tex \
      --out_sensitivity tables_sensitivity.tex \
      --out_importance tables_importance.tex
"""

import os
import json
import argparse
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder

from pvr_eff import (
    compute_distance_matrices, make_grid, combined_features,
    merge_small_classes,
)
from run_full_eff import (
    autodetect_configs, load_dataset_with_labels, load_dataset_auto,
)


SMALL_ORDER = ["mammalian_mito", "sars_cov2", "hrv",
               "influenza_ha", "hev", "ebola"]
LARGE_ORDER = ["dengue_400", "ebola_100", "hev_150",
               "hrv_300", "influenza_300", "sars_cov2_500"]
SMALL_COLS = ["Mam.", "CoV-2", "HRV", "Inf.", "HEV", "Ebola"]
LARGE_COLS = ["Dengue", "Ebola", "HEV", "HRV", "Inf.", "CoV-2"]


def load_json(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def fmt_pct(folds):
    if not folds:
        return "--"
    return f"{100 * np.mean(folds):.1f} $\\pm$ {100 * np.std(folds):.1f}"


def write_runtime_table(small, large, out_path):
    """Table 8: pVR distance time, pVR feature time, baseline total time
    for both low- and large-sample datasets."""
    pretty = {
        "mammalian_mito": "Mammalian", "sars_cov2": "SARS-CoV-2",
        "hrv": "HRV", "influenza_ha": "Influenza",
        "hev": "HEV", "ebola": "Ebola",
        "dengue_400": "Dengue", "ebola_100": "Ebola",
        "hev_150": "HEV", "hrv_300": "HRV",
        "influenza_300": "Influenza", "sars_cov2_500": "SARS-CoV-2",
    }
    n_lookup = {
        "mammalian_mito": 30, "sars_cov2": 31, "hrv": 150,
        "influenza_ha": 59, "hev": 29, "ebola": 28,
        "dengue_400": 400, "ebola_100": 99, "hev_150": 73,
        "hrv_300": 300, "influenza_300": 300, "sars_cov2_500": 316,
    }

    def rows(results, order):
        out = []
        for ds in order:
            t = results.get(ds, {}).get("main", {}).get("timing", {})
            if not t:
                continue
            out.append(
                f"{pretty.get(ds, ds)} & {n_lookup.get(ds, '?')} & "
                f"{t.get('pvr_distances', 0):.1f} & "
                f"{t.get('pvr_features', 0):.1f} & "
                f"{t.get('baselines', 0):.1f} \\\\"
            )
        return out

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Runtime in seconds, measured on a 12-core AMD Ryzen "
        r"workstation with 64\,GB RAM. Column ``pVR dist.'' is the time "
        r"to compute $D_p$ and $D_c$. Column ``pVR feat.'' is the time "
        r"to compute the bi-filtration grid and extract features. Column "
        r"``Baselines'' is the total time to compute all four baseline "
        r"distance matrices.}",
        r"\label{tab:runtime}",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Dataset & $N$ & pVR dist. & pVR feat. & Baselines \\",
        r"\hline",
        r"\multicolumn{5}{l}{\textit{Low-sample}} \\",
    ]
    lines += rows(small, SMALL_ORDER)
    lines += [r"\hline", r"\multicolumn{5}{l}{\textit{Large-sample}} \\"]
    lines += rows(large, LARGE_ORDER)
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def write_sensitivity_table(small, large, out_path):
    """Table 9: XGBoost mean +/- std accuracy across k in {3,4,5,6}."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Sensitivity to $k$-mer size: XGBoost accuracy "
        r"(\%, mean $\pm$ std over a single 5-fold split). The prime "
        r"$p$ is invariant by Proposition~\ref{prop:prime_invariance}.}",
        r"\label{tab:sens}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"$k$ & \multicolumn{6}{c}{Low-sample datasets} \\",
        r"\cline{2-7}",
        " & " + " & ".join(SMALL_COLS) + r" \\",
        r"\hline",
    ]

    for k in [3, 4, 5, 6]:
        cells = []
        for ds in SMALL_ORDER:
            sens = small.get(ds, {}).get("sensitivity", {})
            entry = sens.get(f"k{k}_p5", {})
            folds = entry.get("folds")
            cells.append(fmt_pct(folds) if folds else "--")
        lines.append(f"{k} & " + " & ".join(cells) + r" \\")

    lines += [
        r"\hline",
        r"$k$ & \multicolumn{6}{c}{Large-sample datasets} \\",
        r"\cline{2-7}",
        " & " + " & ".join(LARGE_COLS) + r" \\",
        r"\hline",
    ]
    for k in [3, 4, 5, 6]:
        cells = []
        for ds in LARGE_ORDER:
            sens = large.get(ds, {}).get("sensitivity", {})
            entry = sens.get(f"k{k}_p5", {})
            folds = entry.get("folds")
            cells.append(fmt_pct(folds) if folds else "--")
        lines.append(f"{k} & " + " & ".join(cells) + r" \\")

    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def compute_importance(sequences, labels, k=4, p=5, G_p=10, G_h=15,
                       n_jobs=-1, seed=42):
    """Train XGBoost on the full combined feature matrix and return
    grouped feature importance (gain) as a dict with keys
    'deg', 'betti', 'hist', 'freq', summing to 100."""
    D_p, D_H, freq_vectors = compute_distance_matrices(
        sequences, k=k, p=p, n_jobs=n_jobs
    )
    grid_p, grid_h = make_grid(D_p, D_H, G_p=G_p, G_h=G_h)
    feats, _ = combined_features(
        sequences, D_p, D_H, grid_p, grid_h, freq_vectors,
        k=k, p=p, verbose=False, n_jobs=n_jobs,
    )

    Gp_act = len(grid_p)
    Gh_act = len(grid_h)
    n_deg = Gp_act * Gh_act
    n_betti = Gp_act * Gh_act * 3
    J = min(k, 3)
    n_hist = sum(p ** j for j in range(1, J + 1))
    n_freq = 4 ** k

    le = LabelEncoder()
    y = le.fit_transform(labels)
    if len(np.unique(y)) < 2:
        return {"deg": 0.0, "betti": 0.0, "hist": 0.0, "freq": 0.0}

    X = StandardScaler().fit_transform(feats)
    clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        eval_metric="mlogloss", verbosity=0, random_state=seed,
        use_label_encoder=False, n_jobs=-1,
    )
    clf.fit(X, y)
    booster = clf.get_booster()
    gain_map = booster.get_score(importance_type="gain")

    sums = {"deg": 0.0, "betti": 0.0, "hist": 0.0, "freq": 0.0}
    for fname, val in gain_map.items():
        idx = int(fname[1:])  # 'f123' -> 123
        if idx < n_deg:
            sums["deg"] += val
        elif idx < n_deg + n_betti:
            sums["betti"] += val
        elif idx < n_deg + n_betti + n_hist:
            sums["hist"] += val
        else:
            sums["freq"] += val

    total = sum(sums.values())
    if total == 0:
        return {k: 0.0 for k in sums}
    return {kk: 100.0 * vv / total for kk, vv in sums.items()}


def write_importance_table(small_results, large_results, small_data, large_data,
                           out_path, k=4, p=5, G_p=10, G_h=15, n_jobs=-1):
    """Table 7. Recompute features and retrain XGBoost on the full data
    of each dataset to extract grouped feature importance."""
    rows = []

    for cfg in autodetect_configs(small_data):
        name = cfg["name"]
        if name not in small_results:
            continue
        if cfg["loader"] == "tsv":
            seqs, labs, _ = load_dataset_with_labels(cfg["fasta"], cfg["labels"])
        else:
            seqs, labs, _ = load_dataset_auto(cfg["fasta"], cfg.get("label_func"))
        labs = merge_small_classes(labs, min_count=3)
        if len(seqs) < 10:
            continue
        print(f"  importance: {name} (N={len(seqs)})")
        imp = compute_importance(
            seqs, labs, k=k, p=p, G_p=G_p, G_h=G_h, n_jobs=n_jobs,
        )
        rows.append(("low", name, imp))

    for cfg in autodetect_configs(large_data):
        name = cfg["name"]
        if name not in large_results:
            continue
        if cfg["loader"] == "tsv":
            seqs, labs, _ = load_dataset_with_labels(cfg["fasta"], cfg["labels"])
        else:
            seqs, labs, _ = load_dataset_auto(cfg["fasta"], cfg.get("label_func"))
        labs = merge_small_classes(labs, min_count=3)
        if len(seqs) < 10:
            continue
        print(f"  importance: {name} (N={len(seqs)})")
        imp = compute_importance(
            seqs, labs, k=k, p=p, G_p=G_p, G_h=G_h, n_jobs=n_jobs,
        )
        rows.append(("large", name, imp))

    pretty = {
        "mammalian_mito": "Mammalian", "sars_cov2": "SARS-CoV-2",
        "hrv": "HRV", "influenza_ha": "Influenza",
        "hev": "HEV", "ebola": "Ebola",
        "dengue_400": "Dengue", "ebola_100": "Ebola",
        "hev_150": "HEV", "hrv_300": "HRV",
        "influenza_300": "Influenza", "sars_cov2_500": "SARS-CoV-2",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Feature group importance (\%, XGBoost gain). Deg.\ "
        r"denotes degree profiles, Hist denotes $p$-adic histograms, "
        r"and Freq denotes $k$-mer frequencies. Betti number features "
        r"contribute $0\%$ on all datasets and are omitted.}",
        r"\label{tab:importance}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Dataset & Deg. & Hist & Freq \\",
        r"\hline",
        r"\multicolumn{4}{l}{\textit{Low-sample}} \\",
    ]
    for regime, name, imp in rows:
        if regime == "low":
            lines.append(
                f"{pretty.get(name, name)} & "
                f"{imp['deg']:.1f} & {imp['hist']:.1f} & {imp['freq']:.1f} \\\\"
            )
    lines += [
        r"\hline",
        r"\multicolumn{4}{l}{\textit{Large-sample}} \\",
    ]
    for regime, name, imp in rows:
        if regime == "large":
            lines.append(
                f"{pretty.get(name, name)} & "
                f"{imp['deg']:.1f} & {imp['hist']:.1f} & {imp['freq']:.1f} \\\\"
            )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_results", default="results/all_results.json")
    parser.add_argument("--large_results", default="results_large/all_results.json")
    parser.add_argument("--small_data", default="data")
    parser.add_argument("--large_data", default="data_large")
    parser.add_argument("--out_runtime", default="table_runtime.tex")
    parser.add_argument("--out_sensitivity", default="table_sensitivity.tex")
    parser.add_argument("--out_importance", default="table_importance.tex")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--skip_importance", action="store_true",
                        help="Skip Table 7 (importance) which is the slow step.")
    args = parser.parse_args()

    small = load_json(args.small_results)
    large = load_json(args.large_results)

    write_runtime_table(small, large, args.out_runtime)
    write_sensitivity_table(small, large, args.out_sensitivity)

    if not args.skip_importance:
        write_importance_table(
            small, large, args.small_data, args.large_data,
            args.out_importance, n_jobs=args.n_jobs,
        )


if __name__ == "__main__":
    main()
