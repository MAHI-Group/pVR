#!/bin/bash

python run_full_eff.py --datadir data/ --outdir results/ > run_small.log
python run_full_eff.py --datadir data_large/ --outdir results_large/ --n_jobs 2 > run_large.log
python make_tables_ci.py --small results/all_results.json --large results_large/all_results.json --out tables_ci.tex
