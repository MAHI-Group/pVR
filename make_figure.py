"""
Create figures for the pVR paper.

Composite Figure 1 layout (2 rows x 4 cols, plus pipeline header):
  (a) Method pipeline schematic
  (b) p-adic vs compositional distance matrices (mammalian mito)
  (c) Betti heatmap beta_0 and beta_1 (mammalian mito)
  (d) Main comparison bar chart (all datasets)
  (e) Ablation bar chart (all datasets)

Standalone manuscript figures (saved separately for direct LaTeX inclusion):
  distance_matrices_<dataset>.pdf  -- shared colour scale, two panels
  betti_heatmap_<dataset>.pdf      -- beta_0 and beta_1 only

Note: the in-memory variable D_H and the saved file <dataset>_D_H.npy
retain their historical names for backward compatibility with existing
results. In the paper this matrix is denoted D_c (compositional L_1
distance on k-mer frequency vectors). Figure labels render D_c.

Usage: python make_figure.py --resultsdir results/ --outdir figures/

Written by: Tirtharaj Dash (assistance from Claude Opus 4.x), March 2026
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "mathtext.fontset": "cm",
})


def get_acc(d, key):
    if key not in d:
        return None
    e = d[key]
    for k in ["acc", "accuracy"]:
        if k in e:
            return e[k]
    return None


def draw_pipeline(ax):
    """Panel (a): Method pipeline schematic."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    ax.set_title("(a) pVR method overview", fontsize=9, fontweight="bold",
                 loc="left", pad=4)

    boxes = [
        (0.3, 1.5, "DNA/RNA\nsequences", "#ecf0f1"),
        (2.1, 1.5, "$k$-mer\nextraction", "#d5f5e3"),
        (3.9, 2.3, "$p$-adic\nhistograms", "#d4e6f1"),
        (3.9, 0.7, "$k$-mer\nfrequencies", "#fdebd0"),
        (5.9, 2.3, "$D_p$\n($p$-adic)", "#aed6f1"),
        (5.9, 0.7, "$D_c$\n($L_1$)", "#f9e79f"),
        (7.7, 1.5, "Bi-filtered\nVR complex", "#fadbd8"),
        (9.3, 1.5, "Classify", "#d5f5e3"),
    ]

    for x, y, text, color in boxes:
        w, h = 1.4, 0.9
        if text == "Classify":
            w = 0.9
        rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                              boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor="#2c3e50",
                              linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=6.5,
                fontweight="normal")

    arrows = [
        (1.0, 1.5, 1.4, 1.5),
        (2.8, 1.8, 3.2, 2.3),
        (2.8, 1.2, 3.2, 0.7),
        (4.6, 2.3, 5.2, 2.3),
        (4.6, 0.7, 5.2, 0.7),
        (6.6, 2.0, 7.0, 1.8),
        (6.6, 1.0, 7.0, 1.2),
        (8.4, 1.5, 8.85, 1.5),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#2c3e50",
                                    lw=0.8, mutation_scale=8))

    ax.text(5.9, 1.5, "$\\cap$", ha="center", va="center", fontsize=12,
            fontweight="bold", color="#c0392b")

    ax.text(7.7, 0.15, "$\\beta_0, \\beta_1$ + degree\nprofiles as features",
            ha="center", va="center", fontsize=5.5, style="italic",
            color="#7f8c8d")


def draw_distance_matrices(ax_p, ax_h, results_dir, dataset="mammalian_mito"):
    """Panel (b): Side-by-side distance matrices on the composite figure.
    Uses a single shared colour scale across the two axes.
    The matrix loaded from <dataset>_D_H.npy is the compositional L_1
    distance D_c in the paper."""
    D_p = np.load(os.path.join(results_dir, f"{dataset}_D_p.npy"))
    D_H = np.load(os.path.join(results_dir, f"{dataset}_D_H.npy"))

    vmax = float(max(D_p.max(), D_H.max()))

    im1 = ax_p.imshow(D_p, cmap="viridis", aspect="auto", vmin=0.0, vmax=vmax)
    ax_p.set_title("$p$-adic distance $D_p$", fontsize=8)
    ax_p.set_xlabel("Sequence index", fontsize=7)
    ax_p.set_ylabel("Sequence index", fontsize=7)

    im2 = ax_h.imshow(D_H, cmap="viridis", aspect="auto", vmin=0.0, vmax=vmax)
    ax_h.set_title("$L_1$ distance $D_c$", fontsize=8)
    ax_h.set_xlabel("Sequence index", fontsize=7)

    plt.colorbar(im2, ax=[ax_p, ax_h], shrink=0.7, pad=0.02, fraction=0.04)


def draw_betti(ax0, ax1, results_dir, dataset="mammalian_mito"):
    """Panel (c): Betti heatmaps beta_0 and beta_1 on the composite figure."""
    bg = None
    for suffix in ["_betti.npy", "_betti_grid.npy"]:
        path = os.path.join(results_dir, f"{dataset}{suffix}")
        if os.path.exists(path):
            bg = np.load(path)
            break
    if bg is None or bg.ndim != 3:
        return

    im0 = ax0.imshow(bg[:, :, 0], aspect="auto", origin="lower", cmap="YlOrRd")
    ax0.set_title("$\\beta_0$ (components)", fontsize=8)
    ax0.set_xlabel("Compositional index", fontsize=7)
    ax0.set_ylabel("$p$-adic index", fontsize=7)
    plt.colorbar(im0, ax=ax0, shrink=0.7, pad=0.02)

    im1_ = ax1.imshow(bg[:, :, 1], aspect="auto", origin="lower", cmap="Blues")
    ax1.set_title("$\\beta_1$ (loops)", fontsize=8)
    ax1.set_xlabel("Compositional index", fontsize=7)
    plt.colorbar(im1_, ax=ax1, shrink=0.7, pad=0.02)


def draw_main_comparison(ax, all_results):
    """Panel (d): Main comparison bar chart."""
    methods = [
        ("ffp_js_5nn", "FFP-JS"),
        ("nvm_5nn", "NVM"),
        ("minhash_5nn", "Mash"),
        ("kmer_freq_5nn", "k-mer"),
        ("pvr_5nn", "pVR-5NN"),
        ("pvr_xgboost", "pVR-XGB"),
    ]
    datasets = list(all_results.keys())
    ds_labels = {
        "mammalian_mito": "Mammalian",
        "sars_cov2": "SARS-CoV-2",
        "hrv": "HRV",
    }
    n_m = len(methods)
    x = np.arange(len(datasets))
    width = 0.8 / n_m
    colors = ["#bdc3c7", "#95a5a6", "#7f8c8d", "#3498db", "#e74c3c", "#c0392b"]

    for i, (key, label) in enumerate(methods):
        vals = []
        for ds in datasets:
            main = all_results[ds].get("main", {})
            v = get_acc(main, key)
            vals.append(v * 100 if v else 0)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=label, color=colors[i], edgecolor="white",
               linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([ds_labels.get(d, d) for d in datasets], fontsize=7)
    ax.set_ylabel("Accuracy (%)", fontsize=8)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper center", ncol=3, fontsize=5.5, framealpha=0.9,
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.set_title("(d) Classification accuracy (5-NN / XGBoost)", fontsize=9,
                 fontweight="bold", loc="left", pad=4)


def draw_ablation(ax, all_results):
    """Panel (e): Ablation bar chart."""
    methods = [
        ("abl_padic_vr", "$p$-adic VR"),
        ("abl_hamming_vr", "Comp. VR"),
        ("abl_bifilt_topo", "Bi-filt topo"),
        ("abl_padic_hist", "$p$-adic hist"),
        ("kmer_freq_xgboost", "k-mer freq"),
        ("pvr_xgboost", "pVR (all)"),
    ]
    datasets = list(all_results.keys())
    ds_labels = {
        "mammalian_mito": "Mammalian",
        "sars_cov2": "SARS-CoV-2",
        "hrv": "HRV",
    }
    n_m = len(methods)
    x = np.arange(len(datasets))
    width = 0.8 / n_m
    colors = ["#9b59b6", "#2ecc71", "#1abc9c", "#f39c12", "#3498db", "#e74c3c"]

    for i, (key, label) in enumerate(methods):
        vals = []
        for ds in datasets:
            main = all_results[ds].get("main", {})
            v = get_acc(main, key)
            vals.append(v * 100 if v else 0)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=label, color=colors[i], edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([ds_labels.get(d, d) for d in datasets], fontsize=7)
    ax.set_ylabel("Accuracy (%)", fontsize=8)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper center", ncol=3, fontsize=5.5, framealpha=0.9,
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.set_title("(e) Ablation study (XGBoost)", fontsize=9,
                 fontweight="bold", loc="left", pad=4)


def save_distance_matrices(results_dir, dataset, outdir):
    """Standalone two-panel distance-matrix figure with shared colour scale.
    Saved as distance_matrices_<dataset>.pdf for direct inclusion in the
    manuscript."""
    Dp_path = os.path.join(results_dir, f"{dataset}_D_p.npy")
    DH_path = os.path.join(results_dir, f"{dataset}_D_H.npy")
    if not (os.path.exists(Dp_path) and os.path.exists(DH_path)):
        print(f"Skipping {dataset}: distance matrices not found.")
        return

    D_p = np.load(Dp_path)
    D_H = np.load(DH_path)
    vmax = float(max(D_p.max(), D_H.max()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))

    ax1.imshow(D_p, cmap="viridis", aspect="equal", vmin=0.0, vmax=vmax)
    ax1.set_title(r"$p$-adic distance $D_p$")
    ax1.set_xlabel("Sequence index")
    ax1.set_ylabel("Sequence index")

    im2 = ax2.imshow(D_H, cmap="viridis", aspect="equal", vmin=0.0, vmax=vmax)
    ax2.set_title(r"$L_1$ ($k$-mer composition) distance $D_c$")
    ax2.set_xlabel("Sequence index")

    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.85,
                        fraction=0.04, pad=0.03)
    cbar.set_label("Distance")
    fig.suptitle(dataset.replace("_", " "), y=0.98)

    path = os.path.join(outdir, f"distance_matrices_{dataset}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def save_betti_heatmaps(results_dir, dataset, outdir):
    """Standalone two-panel Betti heatmap figure (beta_0 and beta_1 only;
    beta_2 is empirically zero on every dataset and is omitted).
    Saved as betti_heatmap_<dataset>.pdf for direct inclusion in the
    manuscript."""
    bg = None
    for suffix in ["_betti.npy", "_betti_grid.npy"]:
        p = os.path.join(results_dir, f"{dataset}{suffix}")
        if os.path.exists(p):
            bg = np.load(p)
            break
    if bg is None or bg.ndim != 3:
        print(f"Skipping {dataset}: Betti grid not found.")
        return

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.0, 3.5))

    im0 = ax0.imshow(bg[:, :, 0], aspect="auto", origin="lower", cmap="YlOrRd")
    ax0.set_title(r"$\beta_0$ (components)")
    ax0.set_xlabel("Compositional threshold index")
    ax0.set_ylabel(r"$p$-adic threshold index")
    fig.colorbar(im0, ax=ax0, shrink=0.8)

    im1 = ax1.imshow(bg[:, :, 1], aspect="auto", origin="lower", cmap="Blues")
    ax1.set_title(r"$\beta_1$ (loops)")
    ax1.set_xlabel("Compositional threshold index")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    fig.suptitle(f"Betti numbers: {dataset.replace('_', ' ')}", y=1.0)

    path = os.path.join(outdir, f"betti_heatmap_{dataset}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def build_composite(all_results, results_dir, outdir, suffix="pdf"):
    """Composite Figure 1: pipeline + distances + betti + bars."""
    fig = plt.figure(figsize=(7.5, 8.5))
    gs = gridspec.GridSpec(3, 4, figure=fig,
                           height_ratios=[1.0, 1.2, 1.2],
                           hspace=0.45, wspace=0.45)

    ax_pipe = fig.add_subplot(gs[0, :])
    draw_pipeline(ax_pipe)

    ax_dp = fig.add_subplot(gs[1, 0])
    ax_dh = fig.add_subplot(gs[1, 1])
    draw_distance_matrices(ax_dp, ax_dh, results_dir, "mammalian_mito")
    fig.text(0.02, 0.63, "(b) Distance matrices (mammalian mito)",
             fontsize=9, fontweight="bold", va="top")

    ax_b0 = fig.add_subplot(gs[1, 2])
    ax_b1 = fig.add_subplot(gs[1, 3])
    draw_betti(ax_b0, ax_b1, results_dir, "mammalian_mito")
    fig.text(0.52, 0.63, "(c) Betti numbers (mammalian mito)",
             fontsize=9, fontweight="bold", va="top")

    ax_main = fig.add_subplot(gs[2, :2])
    draw_main_comparison(ax_main, all_results)

    ax_abl = fig.add_subplot(gs[2, 2:])
    draw_ablation(ax_abl, all_results)

    path = os.path.join(outdir, f"figure1_panel.{suffix}")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1,
                dpi=300 if suffix == "png" else None)
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultsdir", default="results")
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    results_path = os.path.join(args.resultsdir, "all_results.json")
    with open(results_path) as f:
        all_results = json.load(f)

    # Standalone manuscript figures
    for ds in ["mammalian_mito", "hrv"]:
        save_distance_matrices(args.resultsdir, ds, args.outdir)
        save_betti_heatmaps(args.resultsdir, ds, args.outdir)

    # Composite figure (PDF + PNG)
    build_composite(all_results, args.resultsdir, args.outdir, suffix="pdf")
    build_composite(all_results, args.resultsdir, args.outdir, suffix="png")


if __name__ == "__main__":
    main()
