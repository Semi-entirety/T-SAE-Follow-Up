#!/usr/bin/env python
"""Plot the final three-seed, groupwise downstream results.

Four figures are produced: absolute Spatial-minus-Baseline differences and
relative improvements for ADE20K segmentation and zoom invariance. Error bars
are the standard deviation of the three paired seed differences.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "figures" / "downstream_3seed"
OUT.mkdir(parents=True, exist_ok=True)

BLOCKS = [0, 2, 4, 6, 8, 10, 11]
COLORS = {"Full": "#30343b", "G0": "#2a6fbb", "G1": "#e07a28"}


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(path)


def segmentation(relative=False):
    d = pd.read_csv(RESULTS / "segmentation_groupwise_3seed_summary.csv")
    specs = [("gap", "Same-vs-different object separation"),
             ("prec5", "Same-object neighbor precision@5")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    for ax, (metric, title) in zip(axes, specs):
        q = d[(d.task == "segment") & (d.metric == metric)]
        for group in ("Full", "G0", "G1"):
            z = q[q.group == group].sort_values("block")
            if relative:
                y = 100 * z["diff"] / z["baseline"].abs()
                err = 100 * z["diff_sd"] / z["baseline"].abs()
            else:
                y, err = z["diff"], z["diff_sd"]
            ax.errorbar(z.block, y, yerr=err, marker="o", capsize=3,
                        color=COLORS[group], label=group)
        ax.axhline(0, color="black", lw=.8)
        ax.set_title(title)
        ax.set_xlabel("DINOv2 block")
        ax.set_xticks(BLOCKS)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Relative improvement (%)" if relative else
                       "Spatial − Baseline (absolute difference)")
    axes[0].legend()
    fig.suptitle("ADE20K segmentation — mean ± SD across 3 paired seeds")
    suffix = "relative_percent" if relative else "absolute"
    finish(fig, OUT / f"segmentation_{suffix}.png")


def transformation(relative=False):
    d = pd.read_csv(RESULTS / "transformation_invariance_3seed_summary.csv")
    specs = [("scale_0.5", "Zoom-out (scale 0.5)"),
             ("scale_2.0", "Zoom-in (scale 2.0)")]
    names = {"all": "Full", "high_level": "G0", "low_level": "G1"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    for ax, (transform, title) in zip(axes, specs):
        q = d[d["transform"] == transform]
        for split in ("all", "high_level", "low_level"):
            z = q[q.split == split].sort_values("block")
            group = names[split]
            if relative:
                y = 100 * z["spatial_minus_baseline_mean"] / z["baseline_mean"].abs()
                err = 100 * z["diff_sd"] / z["baseline_mean"].abs()
            else:
                y, err = z["spatial_minus_baseline_mean"], z["diff_sd"]
            ax.errorbar(z.block, y, yerr=err, marker="o", capsize=3,
                        color=COLORS[group], label=group)
        ax.axhline(0, color="black", lw=.8)
        ax.set_title(title)
        ax.set_xlabel("DINOv2 block")
        ax.set_xticks(BLOCKS)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Relative improvement (%)" if relative else
                       "Spatial − Baseline (absolute cosine difference)")
    axes[0].legend()
    fig.suptitle("Transformation invariance — mean ± SD across 3 paired seeds")
    suffix = "relative_percent" if relative else "absolute"
    finish(fig, OUT / f"transformation_{suffix}.png")


if __name__ == "__main__":
    segmentation(False)
    segmentation(True)
    transformation(False)
    transformation(True)
