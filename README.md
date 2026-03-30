# Replication Study (Mohs Hardness Prediction)

This workspace replicates Figures 3 and 4 from:
J. C. Garnett, "Prediction of Mohs Hardness with Machine Learning Methods Using Compositional Features."

## Primary workflow

- Open and run `Module5Project.ipynb`.
- Data files are stored in `data/`.
- Generated tables and figures are saved under `outputs/`.

## Clean project layout

- `Module5Project.ipynb` - main notebook for model runs and plotting.
- `data/` - `Mineral_Dataset_Supplementary_Info.csv` and `Artificial_Crystals_Dataset.csv`.
- `outputs/results/` - summary CSV outputs.
- `outputs/figures/` - generated figure images.
- `docs/project-proposal.md` - proposal text.
- `Module5Project_owen_remote_backup.ipynb` - preserved backup copy from remote merge.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data source

- [Mendeley dataset jm79zfps6b](https://data.mendeley.com/datasets/jm79zfps6b/1)

