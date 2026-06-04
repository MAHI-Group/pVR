"""
Generate figures for the pVR paper.
Usage: python plot_results.py --resultsdir results/ --outdir figures/
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})


def plot_betti_heatmaps(results_dir, dataset_name, outdir):
    for suffix in ["_betti.npy", "_betti_grid.npy"]:
        path = os.path.join(results_dir, f"{dataset_name}{suffix}")
        if os.path.exists(path):
            betti_grid = np.load(path)
            break
    else:
        print(f"No betti grid for {dataset_name}")
        return

    if betti_grid.ndim == 1:
        betti_grid = np.array(json.loads(betti_grid.tolist())) if isinstance(betti_grid.tolist(), str) else betti_grid

    if betti_grid.ndim != 3:
        print(f"Unexpected betti shape {betti_grid.shape} for {dataset_name}")
        return

    fig, axes = plt.subplots(1, min(3, betti_grid.shape[2]), figsize=(12, 3.5))
    if betti_grid.shape[2] == 1:
        axes = [axes]
    titles = [r"$\beta_0$ (components)", r"$\beta_1$ (loops)", r"$\beta_2$ (voids)"]
    cmaps = ["YlOrRd", "Blues", "Greens"]

    for dim in range(min(3, betti_grid.shape[2])):
        ax = axes[dim]
        data = betti_grid[:, :, dim]
        im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmaps[dim])
        ax.set_xlabel(r"Hamming threshold index")
        ax.set_ylabel(r"$p$-adic threshold index")
        ax.set_title(titles[dim])
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Betti numbers: {dataset_name}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(outdir, f"betti_heatmap_{dataset_name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def get_acc(result_dict, key):
    if key not in result_dict:
        return None
    entry = result_dict[key]
    for k in ["acc", "accuracy"]:
        if k in entry:
            return entry[k]
    return None


def plot_main_comparison(all_results, outdir):
    methods = [
        ("ffp_js_5nn", "FFP-JS"),
        ("nvm_5nn", "NVM"),
        ("minhash_5nn", "Mash"),
        ("kmer_freq_5nn", "k-mer freq"),
        ("pvr_5nn", "pVR (5-NN)"),
        ("pvr_xgboost", "pVR (XGB)"),
    ]
    datasets = list(all_results.keys())
    n_methods = len(methods)
    x = np.arange(len(datasets))
    width = 0.8 / n_methods
    colors = ["#bdc3c7", "#95a5a6", "#7f8c8d", "#3498db", "#e74c3c", "#c0392b"]

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(datasets)), 4.5))
    for i, (key, label) in enumerate(methods):
        vals = []
        for ds in datasets:
            main = all_results[ds].get("main", {})
            v = get_acc(main, key)
            vals.append(v * 100 if v is not None else 0)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=label, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", " ") for d in datasets], rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower right", ncol=2)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, "main_comparison.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_ablation(all_results, outdir):
    methods = [
        ("abl_padic_vr", "$p$-adic VR only"),
        ("abl_hamming_vr", "Hamming VR only"),
        ("abl_bifilt_topo", "Bi-filt topo"),
        ("abl_padic_hist", "$p$-adic hist"),
        ("kmer_freq_xgboost", "k-mer freq"),
        ("pvr_xgboost", "pVR combined"),
    ]
    datasets = list(all_results.keys())
    n_methods = len(methods)
    x = np.arange(len(datasets))
    width = 0.8 / n_methods
    colors = ["#9b59b6", "#2ecc71", "#1abc9c", "#f39c12", "#3498db", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(datasets)), 4.5))
    for i, (key, label) in enumerate(methods):
        vals = []
        for ds in datasets:
            main = all_results[ds].get("main", {})
            v = get_acc(main, key)
            vals.append(v * 100 if v is not None else 0)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=label, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", " ") for d in datasets], rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, "ablation.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_sensitivity_heatmap(all_results, outdir):
    for ds_name, ds_res in all_results.items():
        sens = ds_res.get("sensitivity", {})
        if not sens:
            continue
        ks = sorted(set(v["k"] for v in sens.values() if "k" in v))
        ps = sorted(set(v["p"] for v in sens.values() if "p" in v))
        if not ks or not ps:
            continue

        mat = np.zeros((len(ks), len(ps)))
        for i, k in enumerate(ks):
            for j, p in enumerate(ps):
                key = f"k{k}_p{p}"
                if key in sens:
                    v = get_acc(sens, key)
                    if v is not None:
                        mat[i, j] = v * 100

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                        vmin=max(0, mat.min() - 10), vmax=100)
        ax.set_xticks(range(len(ps)))
        ax.set_xticklabels(ps)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels(ks)
        ax.set_xlabel("Prime $p$")
        ax.set_ylabel("$k$-mer size")
        ax.set_title(f"pVR accuracy (%) -- {ds_name.replace('_', ' ')}")

        for i in range(len(ks)):
            for j in range(len(ps)):
                ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center",
                        fontsize=9, color="black")

        fig.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        path = os.path.join(outdir, f"sensitivity_{ds_name}.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")


def plot_distance_matrices(results_dir, dataset_name, outdir):
    D_p_path = os.path.join(results_dir, f"{dataset_name}_D_p.npy")
    D_H_path = os.path.join(results_dir, f"{dataset_name}_D_H.npy")
    if not (os.path.exists(D_p_path) and os.path.exists(D_H_path)):
        return

    D_p = np.load(D_p_path)
    D_H = np.load(D_H_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    im1 = ax1.imshow(D_p, cmap="viridis", aspect="auto")
    ax1.set_title(f"$p$-adic distance")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(D_H, cmap="viridis", aspect="auto")
    ax2.set_title(f"Hamming ($L_1$) distance")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle(dataset_name.replace("_", " "), fontsize=12)
    plt.tight_layout()
    path = os.path.join(outdir, f"distance_matrices_{dataset_name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultsdir", default="results")
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    results_path = os.path.join(args.resultsdir, "all_results.json")
    if not os.path.exists(results_path):
        print(f"Results not found at {results_path}")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    for ds_name in all_results:
        plot_betti_heatmaps(args.resultsdir, ds_name, args.outdir)
        plot_distance_matrices(args.resultsdir, ds_name, args.outdir)

    plot_main_comparison(all_results, args.outdir)
    plot_ablation(all_results, args.outdir)
    plot_sensitivity_heatmap(all_results, args.outdir)

    print(f"\nAll figures saved to {args.outdir}/")


if __name__ == "__main__":
    main()