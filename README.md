# Replication Study (Mohs Hardness Prediction)

This project implements a replication-style ML pipeline for predicting Mohs hardness from compositional features using the Mendeley dataset `jm79zfps6b` (Version 1).

## What you get

- Data loader that finds and loads the training (naturally-occurring minerals, 622 rows) and test (artificial minerals, 52 rows) CSVs from the downloaded dataset.
- Parameterized label construction (binary and ternary bins) so you can match the paper's thresholds.
- Training/evaluation for a set of baseline classifiers (SVM/RBF variants and Random Forest variants).
- Figure-style outputs:
  - `outputs/figures/figure3_model_performance.png`
  - `outputs/figures/figure4_roc_binary_vs_ovr.png`
- Optional ANN extension via `sklearn.neural_network.MLPClassifier` (architecture and regularization search).

## Project layout

- `Module5Project.ipynb` - notebook workflow and figure replication cells.
- `src/` - reusable loaders, model builders, evaluation, plotting.
- `scripts/` - CLI runner for end-to-end replication.
- `configs/` - model/bin settings (including all nine models).
- `data/` - source CSVs.
- `outputs/` - generated figures and result CSVs.
- `docs/project-proposal.md` - proposal and replication scope notes.

## Setup

1. Download/unzip the dataset from:
   - https://data.mendeley.com/datasets/jm79zfps6b/1
2. Put the extracted folder in `data/` (recommended):
   - `data/mendeley_jm79zfps6b/`
3. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Edit `configs/default.yaml` to match the paper's exact labeling thresholds (binary/ternary).

## Run

```bash
python scripts/run_replication.py --data-dir "data/mendeley_jm79zfps6b" --config "configs/default.yaml" --run-ann true
```

Outputs are written to `outputs/results/` and `outputs/figures/`.

## Notes / matching the paper

The provided implementation is designed to be robust to small differences in CSV column names.
Once you confirm the paper's exact hardness-to-class thresholds (and which exact nine models they used), we can tighten the `configs/default.yaml` and the `src/models.py` model list to match figures exactly.

