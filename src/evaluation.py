from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def _get_underlying_classes(estimator) -> Optional[np.ndarray]:
    """
    Try to extract class order used by predict_proba/decision_function.
    Works for sklearn Pipelines where `classes_` lives on the final estimator.
    """
    if hasattr(estimator, "classes_"):
        return getattr(estimator, "classes_")
    # Pipeline: classes_ is typically on the last step
    if hasattr(estimator, "named_steps"):
        for _, step in reversed(list(getattr(estimator, "named_steps").items())):
            if hasattr(step, "classes_"):
                return getattr(step, "classes_")
    return None


def _safe_predict_proba(estimator, X: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(estimator, "predict_proba"):
        try:
            return estimator.predict_proba(X)
        except Exception:
            return None
    return None


def _safe_decision_function(estimator, X: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(estimator, "decision_function"):
        try:
            return estimator.decision_function(X)
        except Exception:
            return None
    return None


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def get_scores_for_roc(estimator, X: np.ndarray, *, for_class: Optional[int] = None) -> np.ndarray:
    """
    Returns ROC scores:
    - binary: scores for positive class (shape: [n_samples])
    - multiclass: if for_class is None, returns matrix shape [n_samples, n_classes]
    - multiclass one-vs-class: if for_class is set, returns scores for that class (shape: [n_samples])
    """
    proba = _safe_predict_proba(estimator, X)
    if proba is not None:
        classes_arr = _get_underlying_classes(estimator)
        classes = list(classes_arr) if classes_arr is not None else list(range(proba.shape[1]))
        if proba.ndim == 1:
            return proba
        if for_class is None:
            return proba
        idx = classes.index(for_class)
        return proba[:, idx]

    decision = _safe_decision_function(estimator, X)
    if decision is not None:
        if decision.ndim == 1:
            return decision
        if for_class is None:
            return decision
        classes_arr = _get_underlying_classes(estimator)
        classes = list(classes_arr) if classes_arr is not None else list(range(decision.shape[1]))
        idx = classes.index(for_class)
        return decision[:, idx]

    raise RuntimeError("Estimator does not provide predict_proba/decision_function for ROC.")


@dataclass(frozen=True)
class RocCurve:
    fpr: np.ndarray
    tpr: np.ndarray
    auc: float


def compute_binary_roc(y_true_binary: np.ndarray, scores: np.ndarray) -> RocCurve:
    fpr, tpr, _ = roc_curve(y_true_binary, scores)
    auc = float(np.trapz(tpr, fpr)) if len(fpr) > 1 else 0.0
    # Use roc_auc_score when possible for stability.
    try:
        auc = float(roc_auc_score(y_true_binary, scores))
    except Exception:
        pass
    return RocCurve(fpr=fpr, tpr=tpr, auc=auc)


def compute_multiclass_ovr_rocs(
    y_true_multiclass: np.ndarray,
    score_matrix: np.ndarray,
    class_ids: List[int],
) -> Dict[int, RocCurve]:
    """
    For each class, compute ROC curve for (class vs rest) using OVR scores.

    score_matrix: shape [n_samples, n_classes] corresponding to class_ids order.
    """
    out: Dict[int, RocCurve] = {}
    for i, cid in enumerate(class_ids):
        y_bin = (y_true_multiclass == cid).astype(int)
        scores_c = score_matrix[:, i]
        out[cid] = compute_binary_roc(y_bin, scores_c)
    return out


def compute_roc_auc_multiclass_ovr(
    y_true_multiclass: np.ndarray,
    score_matrix: np.ndarray,
    class_ids: List[int],
) -> float:
    y_true_onehot = np.zeros((len(y_true_multiclass), len(class_ids)), dtype=int)
    for j, cid in enumerate(class_ids):
        y_true_onehot[:, j] = (y_true_multiclass == cid).astype(int)
    try:
        return float(roc_auc_score(y_true_onehot, score_matrix, average="macro", multi_class="ovr"))
    except Exception:
        return float("nan")


def evaluate_estimator(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    class_ids: Optional[List[int]] = None,
) -> Tuple[Dict[str, float], Optional[Dict[int, RocCurve]]]:
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    metrics = compute_classification_metrics(y_test, y_pred)

    roc_curves = None
    # If multiclass, we compute OVR ROC curves when we have at least 2 classes.
    if class_ids is None:
        class_ids = sorted(set(int(x) for x in np.asarray(y_test).tolist()))

    if len(class_ids) >= 2:
        try:
            score_matrix = get_scores_for_roc(estimator, X_test, for_class=None)
            if score_matrix.ndim == 1:
                # binary: score_matrix corresponds to a single score stream (positive class).
                y_bin = (y_test == class_ids[-1]).astype(int)
                roc_curves = {class_ids[-1]: compute_binary_roc(y_bin, score_matrix)}
                metrics["roc_auc_ovr_macro"] = roc_curves[class_ids[-1]].auc
            else:
                # If this is truly binary but predict_proba returned [n_samples, 2],
                # handle it as binary ROC instead of multiclass OVR.
                if len(class_ids) == 2 and score_matrix.shape[1] == 2:
                    pos = class_ids[-1]
                    y_bin = (y_test == pos).astype(int)
                    scores_pos = get_scores_for_roc(estimator, X_test, for_class=pos)
                    roc_curves = {pos: compute_binary_roc(y_bin, scores_pos)}
                    metrics["roc_auc_ovr_macro"] = roc_curves[pos].auc
                else:
                    # Multiclass: align score_matrix columns with class_ids by estimator.classes_
                    estimator_classes = list(getattr(estimator, "classes_", class_ids))
                    # Build aligned score matrix ordered by class_ids.
                    aligned = np.zeros((score_matrix.shape[0], len(class_ids)), dtype=float)
                    for j, cid in enumerate(class_ids):
                        idx = estimator_classes.index(cid)
                        aligned[:, j] = score_matrix[:, idx]
                    roc_curves = compute_multiclass_ovr_rocs(y_test, aligned, class_ids=class_ids)
                    metrics["roc_auc_ovr_macro"] = float(np.nanmean([rc.auc for rc in roc_curves.values()]))
        except Exception:
            pass

    # Confusion matrix can be saved separately if needed.
    return metrics, roc_curves

