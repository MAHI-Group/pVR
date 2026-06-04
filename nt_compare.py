"""
Compare Nucleotide Transformer v2 (zero-shot embeddings) with pVR features.

For each chosen low-sample dataset we:
  1. Compute NT v2 mean-pooled embeddings (chunked for sequences exceeding
     the 2048-token context).
  2. Compute pVR combined features using the existing pipeline.
  3. Project both to 2D with UMAP and plot side-by-side, coloured by class.
  4. Run repeated stratified CV (10 seeds x 5 folds) on the NT embeddings
     using the same protocol as pVR (XGBoost / SVM / 5-NN).
  5. Emit a LaTeX comparison table (pVR best vs NT best per dataset).

Outputs:
  results_nt/<dataset>_nt_embeddings.npy
  results_nt/<dataset>_pvr_features.npy
  results_nt/all_nt_results.json
  results_nt/table_nt_compare.tex
  figures/nt_vs_pvr_umap.{pdf,png}

Usage:
  python nt_compare.py --datasets ebola mammalian_mito influenza_ha \
                       --data_dir data/ --small_results results/all_results.json
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM
from umap import UMAP

from pvr_eff import (
    compute_distance_matrices, make_grid, combined_features,
    repeated_cv_features, merge_small_classes,
)
from run_full_eff import (
    autodetect_configs, load_dataset_with_labels, load_dataset_auto,
)


NT_MODEL = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
CHUNK_NT = 12000
MAX_TOKENS = 2048
N_SEEDS = 10
N_FOLDS = 5
BASE_SEED = 42


def load_one_dataset(name, data_dir):
    cfgs = autodetect_configs(data_dir)
    cfg = next((c for c in cfgs if c["name"] == name), None)
    if cfg is None:
        raise RuntimeError(f"Dataset '{name}' not found in {data_dir}")
    if cfg["loader"] == "tsv":
        seqs, labs, _ = load_dataset_with_labels(cfg["fasta"], cfg["labels"])
    else:
        seqs, labs, _ = load_dataset_auto(cfg["fasta"], cfg.get("label_func"))
    labs = merge_small_classes(labs, min_count=3)
    return seqs, labs


@torch.inference_mode()
def embed_nt(sequences, model, tokenizer, device,
             chunk_nt=CHUNK_NT, batch_size=4):
    """Mean-pool NT v2 hidden states. Sequences longer than chunk_nt are
    split into non-overlapping windows and the per-window pooled vectors
    are averaged into one vector per sequence."""
    final = []
    for seq in sequences:
        chunks = [seq[i:i + chunk_nt] for i in range(0, len(seq), chunk_nt)]
        if not chunks:
            chunks = [seq]
        chunk_embs = []
        for j in range(0, len(chunks), batch_size):
            batch = chunks[j:j + batch_size]
            tok = tokenizer.batch_encode_plus(
                batch, return_tensors="pt", padding="longest",
                truncation=True, max_length=MAX_TOKENS,
            )
            input_ids = tok["input_ids"].to(device)
            attn = (input_ids != tokenizer.pad_token_id).to(device)
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outs = model(input_ids, attention_mask=attn,
                                 encoder_attention_mask=attn,
                                 output_hidden_states=True)
            else:
                outs = model(input_ids, attention_mask=attn,
                             output_hidden_states=True)
            h = outs.hidden_states[-1]
            mask = attn.unsqueeze(-1).float()
            pooled = (h.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
            chunk_embs.append(pooled.cpu().numpy())
        chunk_embs = np.concatenate(chunk_embs, axis=0)
        final.append(chunk_embs.mean(axis=0))
    return np.array(final, dtype=np.float32)


def compute_pvr_features(sequences, k=4, p=5, G_p=10, G_h=15, n_jobs=-1):
    D_p, D_H, fv = compute_distance_matrices(sequences, k=k, p=p, n_jobs=n_jobs)
    grid_p, grid_h = make_grid(D_p, D_H, G_p=G_p, G_h=G_h)
    feats, _ = combined_features(
        sequences, D_p, D_H, grid_p, grid_h, fv,
        k=k, p=p, verbose=False, n_jobs=n_jobs,
    )
    return feats


def umap_2d(X, seed=42):
    n = X.shape[0]
    n_neighbors = min(15, max(3, n - 1))
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=0.3,
                   metric="cosine", random_state=seed)
    return reducer.fit_transform(X)


def plot_umap_grid(per_dataset, out_path):
    """per_dataset: list of (name, pvr_xy, nt_xy, labels)."""
    n = len(per_dataset)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (name, pvr_xy, nt_xy, labels) in enumerate(per_dataset):
        unique = sorted(set(labels))
        cmap = plt.get_cmap("tab10")
        colors = {lab: cmap(i % 10) for i, lab in enumerate(unique)}

        for row, (xy, ylabel) in enumerate([(pvr_xy, "pVR features"),
                                             (nt_xy, "NT v2 embeddings")]):
            ax = axes[row, col]
            for lab in unique:
                idx = [i for i, l in enumerate(labels) if l == lab]
                ax.scatter(xy[idx, 0], xy[idx, 1],
                           color=colors[lab], s=45,
                           edgecolor="black", linewidth=0.4,
                           label=str(lab), alpha=0.85)
            if row == 0:
                ax.set_title(name.replace("_", " "), fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            ax.legend(loc="best", fontsize=7, frameon=False,
                      handletextpad=0.2, borderpad=0.2)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_umap_single(name, pvr_xy, nt_xy, labels, out_path):
    """Side-by-side single-dataset UMAP: pVR | NT v2."""
    unique = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    colors = {lab: cmap(i % 10) for i, lab in enumerate(unique)}

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))
    for ax, xy, title in [(axes[0], pvr_xy, "pVR features"),
                          (axes[1], nt_xy, "NT v2 embeddings")]:
        for lab in unique:
            idx = [i for i, l in enumerate(labels) if l == lab]
            ax.scatter(xy[idx, 0], xy[idx, 1],
                       color=colors[lab], s=45,
                       edgecolor="black", linewidth=0.4,
                       label=str(lab), alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                   fontsize=8, frameon=False, title=name.replace("_", " "))
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

def best_variant(results_dict, prefix_or_methods):
    """Return key with highest mean fold accuracy."""
    if isinstance(prefix_or_methods, list):
        candidates = prefix_or_methods
    else:
        candidates = [k for k in results_dict
                      if k.startswith(prefix_or_methods)]
    candidates = [c for c in candidates
                  if results_dict.get(c, {}).get("folds")]
    if not candidates:
        return None
    return max(candidates,
               key=lambda c: np.mean(results_dict[c]["folds"]))


def write_compare_table(rows, out_path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Comparison of \texttt{pVR} with Nucleotide "
        r"Transformer~v2 (500M, multi-species) zero-shot embeddings on "
        r"three low-sample benchmarks. NT embeddings are mean-pooled "
        r"hidden states from the final layer; long sequences are chunked "
        r"and averaged. Both methods use repeated stratified CV "
        r"(up to $50$ folds) and the best-performing classifier "
        r"per row.}",
        r"\label{tab:nt_compare}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Dataset & \texttt{pVR} (best) & NT v2 (best probe) \\",
        r"\hline",
    ]
    pretty = {"ebola": "Ebola", "mammalian_mito": "Mammalian",
              "influenza_ha": "Influenza HA", "hev": "HEV",
              "hrv": "HRV", "sars_cov2": "SARS-CoV-2"}
    classifier_tag = {"pvr_5nn": r"$^\dagger$", "pvr_xgboost": r"$^\ddagger$",
                      "pvr_svm": r"$^\S$", "nt_5nn": r"$^\dagger$",
                      "nt_xgboost": r"$^\ddagger$", "nt_svm": r"$^\S$"}
    for r in rows:
        ds = pretty.get(r["dataset"], r["dataset"].replace("_", " "))
        if r["pvr_acc"] is None:
            pvr_cell = "--"
        else:
            tag = classifier_tag.get(r["pvr_method"], "")
            pvr_cell = f"{r['pvr_acc']:.1f} $\\pm$ {r['pvr_std']:.1f}{tag}"
        tag_nt = classifier_tag.get(r["nt_method"], "")
        nt_cell = f"{r['nt_acc']:.1f} $\\pm$ {r['nt_std']:.1f}{tag_nt}"
        lines.append(f"{ds} & {pvr_cell} & {nt_cell} \\\\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["ebola", "mammalian_mito", "influenza_ha"])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--small_results", default="results/all_results.json")
    parser.add_argument("--outdir", default="results_nt")
    parser.add_argument("--figdir", default="figures")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--cache", action="store_true",
                        help="Reuse cached .npy files if present.")
    parser.add_argument("--single", default=None,
                        help="If set, save a single-dataset side-by-side "
                             "UMAP for this dataset name instead of the grid.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {NT_MODEL} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(NT_MODEL, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        NT_MODEL, trust_remote_code=True
    ).to(device).eval()

    nt_results = {}
    per_dataset = []

    for ds in args.datasets:
        print(f"\n=== {ds} ===")
        seqs, labels = load_one_dataset(ds, args.data_dir)
        print(f"  N={len(seqs)}, classes={len(set(labels))}")

        nt_path = os.path.join(args.outdir, f"{ds}_nt_embeddings.npy")
        if args.cache and os.path.exists(nt_path):
            nt_embs = np.load(nt_path)
            print(f"  Loaded cached NT embeddings: {nt_embs.shape}")
        else:
            print("  Computing NT embeddings...")
            nt_embs = embed_nt(seqs, model, tokenizer, device)
            np.save(nt_path, nt_embs)
            print(f"  Saved NT embeddings: {nt_embs.shape}")

        pvr_path = os.path.join(args.outdir, f"{ds}_pvr_features.npy")
        if args.cache and os.path.exists(pvr_path):
            pvr_feats = np.load(pvr_path)
            print(f"  Loaded cached pVR features: {pvr_feats.shape}")
        else:
            print("  Computing pVR features...")
            pvr_feats = compute_pvr_features(seqs, n_jobs=args.n_jobs)
            np.save(pvr_path, pvr_feats)
            print(f"  Saved pVR features: {pvr_feats.shape}")

        print("  Projecting with UMAP...")
        pvr_xy = umap_2d(pvr_feats)
        nt_xy = umap_2d(nt_embs)
        per_dataset.append((ds, pvr_xy, nt_xy, labels.tolist()))

        print("  Repeated CV on NT embeddings...")
        nt_results[ds] = {}
        for method in ["xgboost", "svm", "5nn"]:
            folds = repeated_cv_features(
                nt_embs, labels, method=method,
                n_folds=N_FOLDS, n_seeds=N_SEEDS, base_seed=BASE_SEED,
            )
            nt_results[ds][f"nt_{method}"] = {
                "acc": float(np.mean(folds)) if folds else 0.0,
                "std": float(np.std(folds)) if folds else 0.0,
                "folds": [float(x) for x in folds],
            }
            c = nt_results[ds][f"nt_{method}"]
            print(f"    nt_{method}: {c['acc']*100:.1f} +/- {c['std']*100:.1f}")

    if args.single:
        match = next((d for d in per_dataset if d[0] == args.single), None)
        if match is None:
            raise RuntimeError(f"--single='{args.single}' not in computed datasets")
        name, pvr_xy, nt_xy, labels = match
        plot_umap_single(name, pvr_xy, nt_xy, labels,
                         os.path.join(args.figdir, f"nt_vs_pvr_umap_{name}.pdf"))
    else:
        plot_umap_grid(per_dataset,
                       os.path.join(args.figdir, "nt_vs_pvr_umap.pdf"))

    with open(os.path.join(args.outdir, "all_nt_results.json"), "w") as f:
        json.dump(nt_results, f, indent=2)

    pvr_data = {}
    if os.path.exists(args.small_results):
        with open(args.small_results) as f:
            pvr_data = json.load(f)

    print("\n=== Headline comparison (best classifier per row) ===")
    rows = []
    pvr_methods = ["pvr_xgboost", "pvr_svm", "pvr_5nn"]
    for ds in args.datasets:
        pvr_main = pvr_data.get(ds, {}).get("main", {})
        best_pvr = best_variant(pvr_main, pvr_methods)
        nt = nt_results[ds]
        best_nt = best_variant(nt, list(nt.keys()))

        pvr_folds = pvr_main.get(best_pvr, {}).get("folds", []) if best_pvr else []
        nt_folds = nt[best_nt]["folds"] if best_nt else []

        row = {
            "dataset": ds,
            "pvr_method": best_pvr,
            "pvr_acc": 100 * np.mean(pvr_folds) if pvr_folds else None,
            "pvr_std": 100 * np.std(pvr_folds) if pvr_folds else None,
            "nt_method": best_nt,
            "nt_acc": 100 * np.mean(nt_folds) if nt_folds else None,
            "nt_std": 100 * np.std(nt_folds) if nt_folds else None,
        }
        rows.append(row)

        pvr_str = (f"{best_pvr:<11} {row['pvr_acc']:5.1f} +/- {row['pvr_std']:.1f}"
                   if pvr_folds else "(pVR JSON missing)")
        nt_str = f"{best_nt:<11} {row['nt_acc']:5.1f} +/- {row['nt_std']:.1f}"
        print(f"  {ds:<20s} pVR: {pvr_str}    NT: {nt_str}")

    write_compare_table(rows, os.path.join(args.outdir, "table_nt_compare.tex"))


if __name__ == "__main__":
    main()
