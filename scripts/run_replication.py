from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data_loader import load_mendeley_train_test
from src.labeling import make_binary_labels, make_ternary_labels, class_order_from_labels
from src.models import build_model_from_spec, build_base_binary_estimator_svm
from src.evaluation import (
    compute_binary_roc,
    compute_multiclass_ovr_rocs,
    evaluate_estimator,
    get_scores_for_roc,
)
from src.plots import plot_figure3, plot_figure4

def run_figure3(
    *,
    model_specs: List[Dict[str, Any]],
    X_train: np.ndarray,
    y_train_bin: np.ndarray,
    y_train_ter: np.ndarray,
    X_test: np.ndarray,
    y_test_bin: np.ndarray,
    y_test_ter: np.ndarray,
    random_seed: int,
    results_dir: Path,
) -> pd.DataFrame:
    rows = []
    for spec in model_specs:
        task = spec["task"]

        if task == "binary":
            y_tr, y_te = y_train_bin, y_test_bin
            y_type = "binary"
        else:
            # Ternary task: sometimes the "ovr" variants are implemented as
            # per-class binary classifiers (class k vs classes != k).
            if spec.get("family") == "rf" and spec.get("multiclass") == "ovr":
                pos = int(spec["positive_class"])
                y_tr = (y_train_ter == pos).astype(int)
                y_te = (y_test_ter == pos).astype(int)
                y_type = "binary"
            else:
                y_tr, y_te = y_train_ter, y_test_ter
                y_type = "ternary"

        class_ids = sorted(set(int(x) for x in np.asarray(y_tr).tolist()))
        estimator = build_model_from_spec(spec, y_type=y_type, random_seed=random_seed)
        metrics, _roc_curves = evaluate_estimator(estimator, X_train, y_tr, X_test, y_te, class_ids=class_ids)
        metrics["model_id"] = spec["name"]
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "figure3_model_metrics.csv", index=False)
    plot_figure3(df, results_dir.parent / "figures/figure3_model_performance.png")
    return df


def run_figure4_roc_binary_vs_ovr(
    *,
    X_train: np.ndarray,
    y_train_ter: np.ndarray,
    X_test: np.ndarray,
    y_test_ter: np.ndarray,
    svm_params: Dict[str, Any],
    random_seed: int,
    output_fig_path: Path,
    output_csv_path: Path,
) -> Tuple[Dict[int, Any], Dict[int, Any]]:
    class_ids = class_order_from_labels(y_test_ter)
    if len(class_ids) < 2:
        raise RuntimeError("Figure 4 requires at least 2 ternary classes.")

    # 1) Binary classifiers: train a separate binary model for each class.
    rocs_binary: Dict[int, Any] = {}
    for cid in class_ids:
        y_train_bin = (y_train_ter == cid).astype(int)
        y_test_bin = (y_test_ter == cid).astype(int)
        estimator = build_base_binary_estimator_svm(svm_params=svm_params, random_seed=random_seed)
        estimator.fit(X_train, y_train_bin)
        # Score for the positive class (label 1).
        scores = get_scores_for_roc(estimator, X_test, for_class=1)
        rocs_binary[cid] = compute_binary_roc(y_test_bin, scores)

    # 2) OVR: train a single One-vs-Rest multiclass classifier and use per-class probabilities.
    ovr_base = build_base_binary_estimator_svm(svm_params=svm_params, random_seed=random_seed)
    ovr_estimator = OneVsRestClassifier(ovr_base)
    ovr_estimator.fit(X_train, y_train_ter)
    score_matrix = get_scores_for_roc(ovr_estimator, X_test, for_class=None)  # [n_samples, n_classes]

    # Align the score matrix columns to class_ids order.
    ovr_classes = list(getattr(ovr_estimator, "classes_", class_ids))
    aligned = np.zeros((score_matrix.shape[0], len(class_ids)), dtype=float)
    for j, cid in enumerate(class_ids):
        idx = ovr_classes.index(cid)
        aligned[:, j] = score_matrix[:, idx]

    rocs_ovr = compute_multiclass_ovr_rocs(y_test_ter, aligned, class_ids=class_ids)

    # Plot and save.
    plot_figure4(class_ids, rocs_binary=rocs_binary, rocs_ovr=rocs_ovr, output_path=output_fig_path)

    rows = []
    for cid in class_ids:
        rows.append({"class_id": cid, "binary_auc": rocs_binary[cid].auc, "ovr_auc": rocs_ovr[cid].auc})
    pd.DataFrame(rows).to_csv(output_csv_path, index=False)

    return rocs_binary, rocs_ovr


def run_ann_extension(
    *,
    X_train: np.ndarray,
    y_train_ter: np.ndarray,
    X_test: np.ndarray,
    y_test_ter: np.ndarray,
    ann_cfg: Dict[str, Any],
    random_seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = ann_cfg.get("candidates", [])
    max_candidates = int(ann_cfg.get("search", {}).get("max_candidates", len(candidates)))
    candidates = candidates[:max_candidates]

    rows = []
    for i, cand in enumerate(candidates):
        hidden_layer_sizes = tuple(cand["hidden_layer_sizes"])
        alpha = float(cand.get("alpha", 1e-4))

        defaults = ann_cfg.get("mlp_defaults", {})
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            activation=defaults.get("activation", "relu"),
            solver=defaults.get("solver", "adam"),
            learning_rate=defaults.get("learning_rate", "adaptive"),
            early_stopping=bool(defaults.get("early_stopping", True)),
            max_iter=int(defaults.get("max_iter", 2500)),
            random_state=random_seed,
        )

        metrics, _ = evaluate_estimator(clf, X_train, y_train_ter, X_test, y_test_ter, class_ids=sorted(set(y_train_ter.tolist())))
        metrics["ann_candidate_idx"] = i
        metrics["hidden_layer_sizes"] = str(hidden_layer_sizes)
        metrics["alpha"] = alpha
        rows.append(metrics)

    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    df.to_csv(output_dir / "ann_model_metrics.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Folder containing extracted Mendeley CSVs.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--run-ann", type=str, default="false", help="true/false")
    args = parser.parse_args()

    cfg = load_config(args.config)
    random_seed = int(cfg.get("random_seed", 42))
    np.random.seed(random_seed)

    data_cfg = cfg["data_loader"]
    split = load_mendeley_train_test(
        args.data_dir,
        train_rows=int(data_cfg.get("train_rows", 622)),
        test_rows=int(data_cfg.get("test_rows", 51)),
    )

    # Labels
    bins_cfg = cfg["bins"]
    binary_edges = bins_cfg["binary"]
    ternary_edges = bins_cfg["ternary"]

    # binary_edges: [min, cutoff, max] => cutoff = binary_edges[1]
    y_train_bin = make_binary_labels(split.y_train, threshold=float(binary_edges[1]), positive_label=1, negative_label=0)
    y_test_bin = make_binary_labels(split.y_test, threshold=float(binary_edges[1]), positive_label=1, negative_label=0)

    # ternary_edges: [min, low_max, mid_max, max] => [low_max, mid_max]
    y_train_ter = make_ternary_labels(split.y_train, edges=[float(ternary_edges[1]), float(ternary_edges[2])], labels=[0, 1, 2])
    y_test_ter = make_ternary_labels(split.y_test, edges=[float(ternary_edges[1]), float(ternary_edges[2])], labels=[0, 1, 2])

    svm_params = cfg["figure4"]["svm"]
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure 3-style evaluation
    run_figure3(
        model_specs=cfg["models"],
        X_train=split.X_train,
        y_train_bin=y_train_bin,
        y_train_ter=y_train_ter,
        X_test=split.X_test,
        y_test_bin=y_test_bin,
        y_test_ter=y_test_ter,
        random_seed=random_seed,
        results_dir=results_dir,
    )

    # Figure 4 ROC
    run_figure4_roc_binary_vs_ovr(
        X_train=split.X_train,
        y_train_ter=y_train_ter,
        X_test=split.X_test,
        y_test_ter=y_test_ter,
        svm_params=svm_params,
        random_seed=random_seed,
        output_fig_path=figures_dir / "figure4_roc_binary_vs_ovr.png",
        output_csv_path=results_dir / "figure4_roc_auc.csv",
    )

    # ANN extension (optional)
    run_ann_flag = str(args.run_ann).lower() in {"1", "true", "yes", "y"}
    if run_ann_flag and cfg.get("ann_extension", {}).get("enabled", False):
        run_ann_extension(
            X_train=split.X_train,
            y_train_ter=y_train_ter,
            X_test=split.X_test,
            y_test_ter=y_test_ter,
            ann_cfg=cfg["ann_extension"],
            random_seed=random_seed,
            output_dir=results_dir / "ann",
        )


if __name__ == "__main__":
    main()

