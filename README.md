# Q-BioLat: Binary Latent Protein Fitness Landscapes for QUBO-Based Optimization

Official code for the paper **Q-BioLat: Binary Latent Protein Fitness Landscapes for QUBO-Based Optimization**.

Q-BioLat studies protein fitness optimization in compact binary latent spaces. The pipeline starts from pretrained ESM sequence embeddings, constructs binary latent representations using random projection, PCA, or learned latent models, fits an internal QUBO surrogate, and applies combinatorial optimization methods such as simulated annealing, genetic algorithms, greedy hill climbing, random search, and a lightweight latent Bayesian optimization baseline. The repository also contains scripts for oracle training, decoder training, aggregation, plotting, and LaTeX export used in the paper.

![Main Figure](Main_Figure.png)

## Repository structure

```text
Q-BIOLAT-Extended-main/
├── data/
│   └── proteingym/              # GFP and AAV benchmark CSVs and sampled subsets
├── examples/                    # Data preparation, subset sampling, ESM embedding, PCA/random binarization
├── experiments/                 # Training, evaluation, optimization, aggregation, plotting, LaTeX export
├── scripts/                     # Reproducible shell pipelines for full experiment grids
├── src/
│   ├── analysis/                # Landscape diagnostics and analysis utilities
│   ├── data/                    # Dataset loading and helper functions
│   ├── models/                  # QUBO surrogate, MLP baseline, related model code
│   ├── optimization/            # SA, GA, RS, GHC, latent BO implementations
│   └── utils/                   # Metrics, retrieval, and miscellaneous utilities
├── configs/                     # Default configs for synthetic experiments
├── Main_Figure.png              # Overview figure
├── instructions.txt             # Internal run order notes
└── setup.py                     # Minimal package setup
```

In short:

- `src/` contains reusable implementations of models, optimization methods, data loaders, and utilities.
- `experiments/` contains Python entry points for training, evaluation, aggregation, plotting, and exporting paper figures/tables.
- `scripts/` contains shell scripts for running full grids and reproducing the main pipeline in order.
- `examples/` contains dataset preparation code, ESM embedding generation, subset sampling, and binary latent construction.
- `data/proteingym/` contains the full GFP and AAV CSV files together with the sampled subsets used in the paper.

## Installation

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install numpy pandas scipy scikit-learn matplotlib torch transformers xgboost requests
```

Notes:

- `torch` and `transformers` are required for ESM embeddings.
- `xgboost` is required for the external oracle benchmark.
- The provided scripts use `--device cpu` by default, but you can switch to GPU for embedding extraction if desired.

## Data

The repository already includes the ProteinGym datasets used in the paper:

```text
data/proteingym/gfp.csv
data/proteingym/aav.csv
```

It also includes the sampled subsets:

```text
gfp_1000.csv, gfp_2000.csv, gfp_5000.csv, gfp_10000.csv
aav_1000.csv, aav_2000.csv, aav_5000.csv, aav_10000.csv
```

If you want to regenerate subsets yourself, use:

```bash
python examples/make_subset_csv.py --input-csv data/proteingym/gfp.csv --output-csv data/proteingym/gfp_1000.csv --n 1000 --seed 42
python examples/make_subset_csv.py --input-csv data/proteingym/aav.csv --output-csv data/proteingym/aav_1000.csv --n 1000 --seed 42
```

The full automated pipeline scripts will regenerate these subset CSVs as needed.

## Reproducibility overview

The main experimental flow is:

```text
ProteinGym CSVs
→ sampled subsets
→ dense ESM embeddings
→ binary latent representations (random projection / PCA / AE / VAE)
→ external oracle + internal QUBO surrogate
→ combinatorial optimization
→ decoder-based sequence reconstruction and scoring
→ aggregation, plots, and LaTeX tables/figures
```

## 1. Generate ESM embeddings and random-projection binary latents

Run the GFP and AAV preprocessing pipelines:

```bash
bash scripts/run_gfp_pipeline.sh
bash scripts/run_aav_pipeline.sh
```

These scripts:

1. sample subsets of size `{1000, 2000, 5000, 10000}`;
2. compute dense ESM embeddings with `examples/build_real_peptide_dataset_esm_dense.py`;
3. create random-projection binary latents for dimensions `{8, 16, 32, 64}` using median binarization.

Representative outputs:

```text
artifacts/dense/gfp_<N>_dense.npz
artifacts/dense/aav_<N>_dense.npz
artifacts/binary/gfp_<N>_esm_binary_<D>.npz
artifacts/binary/aav_<N>_esm_binary_<D>.npz
```

## 2. Generate PCA binary latents

After dense embeddings are available, build PCA-based binary latents:

```bash
bash scripts/run_gfp_pca_pipeline.sh
bash scripts/run_aav_pca_pipeline.sh
```

These scripts project dense embeddings to dimensions `{8, 16, 32, 64}` with PCA and then binarize using per-dimension medians.

Representative outputs:

```text
artifacts/binary_pca/gfp_<N>_pca_binary_<D>.npz
artifacts/binary_pca/aav_<N>_pca_binary_<D>.npz
```

## 3. Train learned latent models (AE / VAE)

To reproduce the learned-latent experiments in the paper, train the autoencoder and variational autoencoder grids:

```bash
bash scripts/train_latent_models_grid.sh
```

This runs AE and VAE on both GFP and AAV for all sample sizes and latent dimensions.

Representative outputs:

```text
artifacts/latent_models/<dataset>_<N>_<model>_<D>.pt
artifacts/latent_models/<dataset>_<N>_<model>_<D>.json
artifacts/latent_models/<dataset>_<N>_<model>_<D>_test_latents.npz
```

## 4. Train the external sequence-level oracle

Train the oracle grid:

```bash
bash scripts/train_oracle_grid.sh
```

This fits three external sequence-level regressors on dense ESM embeddings:

- Ridge Regression
- XGBoost
- Gaussian Process Regression

Then aggregate and group the oracle results:

```bash
bash scripts/aggregate_oracle_results.sh
bash scripts/group_oracle_results.sh
```

Representative outputs:

```text
artifacts/oracle/models/
artifacts/oracle/metrics/
artifacts/oracle/preds/
artifacts/results/oracle_summary.csv
artifacts/results/oracle_grouped.csv
```

## 5. Analyze QUBO landscapes

Run the landscape diagnostics for both PCA and random-projection latents:

```bash
bash scripts/analyze_all_pca_landscape.sh
bash scripts/analyze_all_random_landscape.sh
bash scripts/group_landscape_results.sh
bash scripts/plot_landscape_diagnostics.sh
```

These scripts compute and summarize diagnostics such as smoothness, ruggedness, and spectral metrics of the learned QUBO landscapes.

Representative outputs:

```text
artifacts/results/landscape_*.json
artifacts/results/landscape_summary.csv
artifacts/figures/landscape/
```

## 6. Run combinatorial optimization baselines

Run the multiseed benchmark across both random-projection and PCA binary latents:

```bash
bash scripts/run_multiseed_grid.sh
```

This benchmark runs the following methods in binary latent space:

- Simulated Annealing (SA)
- Genetic Algorithm (GA)
- Random Search (RS)
- Greedy Hill Climbing (GHC)
- Latent Bayesian Optimization (LBO)

Each configuration is evaluated over seeds `0 1 2 3 4`.

If you want the dedicated GFP PCA multiseed script used in part of the paper workflow, run:

```bash
bash scripts/run_multiseed_grid_pca.sh
```

Then aggregate and group optimization outputs:

```bash
bash scripts/aggregate_optimization_results.sh
bash scripts/group_optimization_results.sh
```

Representative outputs:

```text
artifacts/multiseed/*.json
artifacts/results/optimization_summary.csv
artifacts/results/optimization_grouped.csv
```

## 7. Merge landscape and optimization summaries

To connect landscape diagnostics with optimization behavior:

```bash
bash scripts/merge_landscape_optimization.sh
```

Representative output:

```text
artifacts/results/merged_landscape_optimization.csv
```

## 8. Train the decoder

Train the mutation-conditioned decoder on both PCA and random-projection latents:

```bash
bash scripts/train_decoder_from_projection_latents.sh
bash scripts/aggregate_decoder_results.sh
```

A simpler wrapper is also available:

```bash
bash scripts/train_decoder.sh
```

Representative outputs:

```text
artifacts/decoder_models/*.pt
artifacts/decoder_models/*.json
artifacts/results/decoder_summary.csv
```

## 9. Decode optimized latent codes and score sequences

After optimization, decode and score optimized binary latent codes with the learned decoder and the trained oracle:

```bash
bash scripts/run_decoder_grid.sh
bash scripts/aggregate_decoded_results.sh
```

Representative outputs:

```text
artifacts/decoded_scored/*.csv
artifacts/results/decoded_summary.csv
```

## 10. Generate paper figures and tables

The repository contains scripts that aggregate results and export plots / LaTeX artifacts used in the manuscript.

### Main plotting

```bash
bash scripts/plot_final_figures.sh
bash scripts/plot_clean_figures.sh
bash scripts/plot_oracle_results.sh
bash scripts/plot_section56_results.sh
bash scripts/plot_optimization_landscape_relationships.sh
```

### Table and figure export for LaTeX

```bash
bash scripts/export_oracle_table_latex.sh
bash scripts/export_table3_latex.sh
bash scripts/export_table3_split.sh
bash scripts/export_table6_latex.sh
bash scripts/export_section56_tables.sh
bash scripts/export_main_section56_figure.sh
bash scripts/export_appendix_figure_latex.sh
```

These scripts produce grouped CSV summaries, publication figures, and LaTeX tables for the main text and appendix.

## Suggested run order

If you want to follow the main workflow in order, the shortest reproducibility path is:

```bash
# embeddings + random projection
bash scripts/run_gfp_pipeline.sh
bash scripts/run_aav_pipeline.sh

# PCA latents
bash scripts/run_gfp_pca_pipeline.sh
bash scripts/run_aav_pca_pipeline.sh

# oracle
bash scripts/train_oracle_grid.sh
bash scripts/aggregate_oracle_results.sh
bash scripts/group_oracle_results.sh

# latent model analysis (AE/VAE)
bash scripts/train_latent_models_grid.sh

# decoder
bash scripts/train_decoder_from_projection_latents.sh
bash scripts/aggregate_decoder_results.sh

# landscape analysis
bash scripts/analyze_all_pca_landscape.sh
bash scripts/analyze_all_random_landscape.sh
bash scripts/group_landscape_results.sh
bash scripts/plot_landscape_diagnostics.sh

# optimization
bash scripts/run_multiseed_grid.sh
bash scripts/aggregate_optimization_results.sh
bash scripts/group_optimization_results.sh

# merge + final figures
bash scripts/merge_landscape_optimization.sh
bash scripts/plot_final_figures.sh

# decode optimized latents
bash scripts/run_decoder_grid.sh
bash scripts/aggregate_decoded_results.sh
```

## Key experimental components reproduced by this repository

- ESM sequence embeddings
- Binary representations from random projection and PCA
- Learned latent representations from AE and VAE
- External sequence-level fitness oracle
- Internal QUBO surrogate
- Combinatorial optimization with SA, GA, RS, GHC, and LBO
- Mutation-conditioned decoder
- Landscape diagnostics, aggregation, and paper-ready figures/tables

## Citation

If you use this repository, please cite the paper.

```bibtex
@article{hy2026binary,
  title={Binary Latent Protein Fitness Landscapes for Quantum Annealing Optimization},
  author={Hy, Truong-Son},
  journal={arXiv preprint arXiv:2603.17247},
  year={2026}
}
```

## Contact

Prof. Truong-Son Hy, Ph.D.

Department of Computer Science & Heersink School of Medicine

The University of Alabama at Birmingham, United States

Email: thy@uab.edu

## License
This project is released under the MIT License.
