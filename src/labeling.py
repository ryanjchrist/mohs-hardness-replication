from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def make_binary_labels(
    y: np.ndarray,
    threshold: float,
    positive_label: int = 1,
    negative_label: int = 0,
) -> np.ndarray:
    """Binary hard/soft split based on Mohs hardness threshold."""
    y = np.asarray(y, dtype=float)
    return np.where(y >= threshold, positive_label, negative_label).astype(int)


def make_ternary_labels(
    y: np.ndarray,
    edges: Sequence[float],
    labels: Sequence[int] = (0, 1, 2),
) -> np.ndarray:
    """
    Map continuous hardness values to 3 ordered classes using 2 edges.

    With edges = [e1, e2]:
    - class labels[0]: y <= e1
    - class labels[1]: e1 < y <= e2
    - class labels[2]: y > e2
    """
    y = np.asarray(y, dtype=float)
    if len(edges) != 2:
        raise ValueError("ternary.edges must contain exactly 2 values: [low_max, mid_max].")
    if len(labels) != 3:
        raise ValueError("ternary.labels must contain exactly 3 class ids.")

    e1, e2 = float(edges[0]), float(edges[1])
    l0, l1, l2 = int(labels[0]), int(labels[1]), int(labels[2])
    out = np.empty_like(y, dtype=int)
    out[y <= e1] = l0
    out[(y > e1) & (y <= e2)] = l1
    out[y > e2] = l2
    return out


def class_order_from_labels(y_labels: np.ndarray) -> List[int]:
    """Return stable class order for plotting."""
    classes = sorted(set(int(x) for x in np.asarray(y_labels).tolist()))
    return classes

