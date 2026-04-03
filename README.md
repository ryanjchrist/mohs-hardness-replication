# Replication Study (Mohs Hardness Prediction)

This repository replicates **Figure 3** (model performance) and **Figure 4** (ROC) from Garnett’s chapter, and includes extensions (ANN comparison and natural-train / artificial-test evaluation). The main workflow is **`Module5Project.ipynb`**. Data live in **`data/`**; tables and figures are written to **`outputs/`**.

## Paper and DOI

J. C. Garnett, “Prediction of Mohs Hardness with Machine Learning Methods Using Compositional Features,” *ACS Symposium Series*, 2019.

- **DOI:** [10.1021/bk-2019-1326.ch002](https://doi.org/10.1021/bk-2019-1326.ch002)

## Data source and files in this repo

Public data (same source as the paper’s study):

- **Mendeley dataset:** [jm79zfps6b](https://data.mendeley.com/datasets/jm79zfps6b/1)

Files tracked under **`data/`** for class submission and reproducibility:

| File | Role |
|------|------|
| `Mineral_Dataset_Supplementary_Info.csv` | **Natural minerals** (n ≈ 622). Mohs hardness column: **`Hardness`**. |
| `Artificial_Crystals_Dataset.csv` | **Artificial / synthetic crystals** (n ≈ 52). Mohs hardness column: **`Hardness (Mohs)`**. |

The notebook uses the same 11 compositional features for both tables (see below). Rows with invalid or out-of-range hardness are skipped when building `X`, `y`.

## Features (11 inputs)

These match the **`FEATURES`** list in `Module5Project.ipynb` (Garnett-style compositional descriptors derived from elemental composition):

| Column name | Short description |
|-------------|-------------------|
| `allelectrons_Total` | Total electrons (composition aggregate) |
| `density_Total` | Total density-related aggregate |
| `allelectrons_Average` | Average electrons per atom / site |
| `val_e_Average` | Average valence-electron character |
| `atomicweight_Average` | Average atomic weight |
| `ionenergy_Average` | Average ionization energy |
| `el_neg_chi_Average` | Average electronegativity (Pauling χ) |
| `R_vdw_element_Average` | Average van der Waals radius |
| `R_cov_element_Average` | Average covalent radius |
| `zaratio_Average` | Average Z/A ratio |
| `density_Average` | Average density |

**Target (regression hardness):** read from the hardness column above; it is then binned into classification labels for each experiment (below).

## Labels (classification tasks in the notebook)

Hardness must satisfy **0.991 < h ≤ 10** for a sample to be kept (same convention as the notebook helpers).

1. **Binary (two classes)** — `binary_label_from_value`  
   - Class **0:** 0.991 < h ≤ 5.5  
   - Class **1:** 5.5 < h ≤ 10.0  

2. **Ternary (three classes)** — `ternary_label_from_value`  
   - Class **0:** 0.991 < h ≤ 4.0  
   - Class **1:** 4.0 < h ≤ 7.0  
   - Class **2:** 7.0 < h ≤ 10.0  

Some models use **one-vs-rest** or **one-vs-one** formulations on top of these label sets; see `MODEL_SPECS` in the notebook for which model uses which task.

3. **Extension (optional 9-class Mohs buckets)** — separate cell: nine ordinal buckets (including merged high-hardness region); see the notebook section “9-class Mohs hardness prediction” for exact interval definitions.

## Primary workflow

1. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Open and run **`Module5Project.ipynb`** from the repository root.

3. Outputs appear under **`outputs/results/`** (CSV) and **`outputs/figures/`** (PNG).

## Project layout

- `Module5Project.ipynb` — replication (Figures 3–4), extensions, saved outputs  
- `data/` — CSV inputs (see table above)  
- `outputs/results/` — metrics CSVs  
- `outputs/figures/` — generated figures  
- `docs/project-proposal.md` — proposal text  
- `requirements.txt` — Python dependencies  
