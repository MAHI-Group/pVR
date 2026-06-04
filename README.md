# pVR: p-Adic Bi-Filtered Simplicial Complexes for Genomic Classification

## Notation

For historical reasons, the code uses `D_H` for the compositional
distance matrix (L1 distance on k-mer frequency vectors). In the
paper this is denoted `D_c`. The two refer to the same object.

## Setup

```bash
pip install -r requirements.txt
```

GUDHI may require: `conda install -c conda-forge gudhi` if pip fails.

## Step-by-step

### 1. Download data

```bash
python download_data.py --email YOUR_EMAIL@domain.com --datasets mammalian_mito sars_cov2
```

For HRV and Influenza (search-based, labels need manual curation):
```bash
python download_data.py --email YOUR_EMAIL@domain.com --datasets hrv influenza_ha
```

HRV labels are auto-assigned from FASTA descriptions. Check `data/hrv.fasta`
descriptions to verify serotype assignments. For influenza, you need to
manually create `data/influenza_ha_labels.tsv` with columns `accession` and
`label` (subtypes H1, H3, H5, etc.).

### 2. Run experiments

Full pipeline (main + sensitivity + grid resolution):
```bash
python run_experiments.py --datadir data/ --outdir results/
```

Quick run (main experiment only):
```bash
python run_experiments.py --datadir data/ --outdir results/ --experiments main
```

Custom parameters:
```bash
python run_experiments.py --datadir data/ --outdir results/ --k 7 --p 5 --Gp 12 --Gh 15
```

### 3. Generate figures

```bash
python plot_results.py --resultsdir results/ --outdir figures/
```

### 4. Fill in the paper

Results are in `results/all_results.json`. The experiment script prints a
LaTeX-ready summary table. Copy numbers into `pVR_paper.tex` placeholder tables.

## Files

- `pvr.py` -- core algorithms (encoding, distances, bi-filtration, classification)
- `download_data.py` -- NCBI data acquisition with curated accession lists
- `run_experiments.py` -- full experiment pipeline with ablations
- `plot_results.py` -- paper figure generation
- `pVR_paper.tex` -- manuscript with placeholders

## Expected runtime

- Mammalian mitochondrial (~30 sequences): ~10 minutes total
- SARS-CoV-2 (~40 sequences): ~15 minutes total
- Full pipeline with sensitivity analysis: ~2-3 hours

## Notes on GISAID

The CAKL paper uses 44 SARS-CoV-2 genomes from GISAID. Our script downloads
NCBI alternatives for the same variants. For exact replication, register at
https://gisaid.org and download their accessions. Place as
`data/sars_cov2_gisaid.fasta` with corresponding label file.
