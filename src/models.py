from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.special import gamma as gamma_fn
from scipy.special import kv
from scipy.spatial.distance import cdist


def _resolve_gamma_value(gamma: str | float, X: np.ndarray) -> float:
    """Match sklearn's 'scale'/'auto' gamma heuristics for kernels."""
    if isinstance(gamma, str):
        if gamma == "scale":
            n_features = X.shape[1]
            x_var = float(np.var(X))
            if x_var <= 0.0:
                x_var = 1.0
            return 1.0 / (n_features * x_var)
        if gamma == "auto":
            return 1.0 / float(X.shape[1])
        raise ValueError(f"Unsupported gamma string: {gamma}")
    return float(gamma)


def _matern_kernel_matrix(X: np.ndarray, Y: np.ndarray, *, nu: float, gamma_value: float) -> np.ndarray:
    """
    Compute Matern kernel using:
      k(r) = (2^(1-nu)/Gamma(nu)) * (sqrt(2nu)*r/ell)^nu * K_nu(sqrt(2nu)*r/ell)
    where gamma_value = 1/(2*ell^2) so ell = 1/sqrt(2*gamma_value).
    """
    # Euclidean distances r = ||x - y||
    r = cdist(X, Y, metric="euclidean")

    # beta = sqrt(2*nu) * r / ell; with ell = 1/sqrt(2*gamma) => beta = 2*sqrt(nu*gamma)*r
    beta = 2.0 * np.sqrt(nu * gamma_value) * r
    factor = (2.0 ** (1.0 - nu)) / gamma_fn(nu)

    K = np.empty_like(beta, dtype=float)
    zero_mask = beta == 0
    K[zero_mask] = 1.0

    nonzero = ~zero_mask
    b = beta[nonzero]
    K[nonzero] = factor * (b**nu) * kv(nu, b)
    return K


def make_svc_rbf(*, probability: bool, c: float, gamma: str | float, random_state: int):
    # We wrap SVC in a scaler pipeline so features match the SVM assumptions.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=c,
                    gamma=gamma,
                    probability=probability,
                    random_state=random_state,
                ),
            ),
        ]
    )


class _MaternPrecomputedSVC(BaseEstimator, ClassifierMixin):
    """
    SVC-like wrapper that supports a Matern smoothness parameter `nu`
    by using `kernel='precomputed'` with an explicit Matern kernel matrix.
    """

    def __init__(
        self,
        *,
        C: float,
        nu: float,
        gamma: str | float,
        probability: bool,
        random_state: int,
    ):
        self.C = float(C)
        self.nu = float(nu)
        self.gamma = gamma
        self.probability = bool(probability)
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self._svc = None
        self._X_train = None
        self._gamma_value = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._gamma_value = _resolve_gamma_value(self.gamma, X)
        self._X_train = X
        K_train = _matern_kernel_matrix(X, X, nu=self.nu, gamma_value=self._gamma_value)
        self._svc = SVC(
            kernel="precomputed",
            C=self.C,
            probability=self.probability,
            random_state=self.random_state,
        )
        self._svc.fit(K_train, y)
        self.classes_ = self._svc.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        K = _matern_kernel_matrix(X, self._X_train, nu=self.nu, gamma_value=self._gamma_value)
        return self._svc.predict(K)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        K = _matern_kernel_matrix(X, self._X_train, nu=self.nu, gamma_value=self._gamma_value)
        return self._svc.predict_proba(K)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        K = _matern_kernel_matrix(X, self._X_train, nu=self.nu, gamma_value=self._gamma_value)
        return self._svc.decision_function(K)


def make_svc_matern(*, probability: bool, c: float, nu: float, gamma: str | float, random_state: int):
    # sklearn in this environment supports `kernel="matern"` but does not expose `nu`.
    # This wrapper implements the Matern kernel explicitly so we can match the spec.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svc",
                _MaternPrecomputedSVC(
                    C=c,
                    nu=nu,
                    gamma=gamma,
                    probability=probability,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_random_forest(*, random_state: int, n_estimators: int = 500) -> RandomForestClassifier:
    # Use class_weight="balanced" because the paper's bins can be imbalanced.
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )


def build_model_from_spec(spec: Dict[str, Any], *, y_type: str, random_seed: int):
    """
    Build an estimator matching one of the paper's 9-model specifications.

    y_type:
      - "binary": y labels are {0,1}
      - "ternary": y labels are {0,1,2}
    """
    probability = True
    family = spec["family"]
    kernel = spec.get("kernel")
    multiclass = spec.get("multiclass", "direct")

    if family == "svc":
        if kernel == "rbf":
            base = make_svc_rbf(
                probability=probability,
                c=float(spec["C"]),
                gamma=spec.get("gamma", "scale"),
                random_state=random_seed,
            )
        elif kernel == "matern":
            base = make_svc_matern(
                probability=probability,
                c=float(spec["C"]),
                nu=float(spec["nu"]),
                gamma=spec.get("gamma", "scale"),
                random_state=random_seed,
            )
        else:
            raise ValueError(f"Unsupported SVC kernel: {kernel}")

        if y_type == "binary" or multiclass == "direct":
            return base
        if multiclass == "ovo":
            return OneVsOneClassifier(base)
        if multiclass == "ovr":
            return OneVsRestClassifier(base)
        raise ValueError(f"Unsupported svc multiclass setting: {multiclass}")

    if family == "rf":
        # For RF "ovr" variants, run_replication converts ternary->binary beforehand.
        return make_random_forest(random_state=random_seed)

    raise ValueError(f"Unknown model family: {family}")


def build_base_binary_estimator_svm(*, svm_params: Dict[str, Any], random_seed: int):
    """Base estimator used when we train one-vs-rest/per-class binary models."""
    kernel = svm_params.get("kernel", "rbf")
    if kernel == "rbf":
        return make_svc_rbf(
            probability=True,
            c=float(svm_params["C"] if "C" in svm_params else svm_params["c"]),
            gamma=svm_params.get("gamma", "scale"),
            random_state=random_seed,
        )
    if kernel == "matern":
        return make_svc_matern(
            probability=True,
            c=float(svm_params["C"] if "C" in svm_params else svm_params["c"]),
            nu=float(svm_params["nu"]),
            gamma=svm_params.get("gamma", "scale"),
            random_state=random_seed,
        )
    raise ValueError(f"Unsupported kernel in svm_params: {kernel}")

