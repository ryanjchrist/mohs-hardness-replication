from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import RocCurve


def plot_figure3(model_metrics: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = model_metrics.sort_values("f1_macro", ascending=False)
    plt.figure(figsize=(10, 4.5))
    plt.bar(df["model_id"], df["f1_macro"], color="#4C78A8")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("F1 Macro")
    plt.title("Figure 3: Model Performance (Test Set)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_figure4(
    class_ids: List[int],
    rocs_binary: Dict[int, RocCurve],
    rocs_ovr: Dict[int, RocCurve],
    output_path: str | Path,
    *,
    title: str = "Figure 4: ROC Curves (Binary vs OVR)",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8.5, 6))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for i, cid in enumerate(class_ids):
        cb = colors[i % len(colors)]
        rc_bin = rocs_binary[cid]
        rc_ovr = rocs_ovr[cid]
        plt.plot(rc_bin.fpr, rc_bin.tpr, color=cb, linestyle="--", linewidth=2, label=f"Class {cid} (Binary) AUC={rc_bin.auc:.3f}")
        plt.plot(rc_ovr.fpr, rc_ovr.tpr, color=cb, linestyle="-", linewidth=2, label=f"Class {cid} (OVR) AUC={rc_ovr.auc:.3f}")

    plt.plot([0, 1], [0, 1], "k:", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

