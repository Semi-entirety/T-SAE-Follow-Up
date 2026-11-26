#!/usr/bin/env python
"""Aggregate the 21 block/seed outputs from segmentation_groups_eval.py."""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.raw_root)
    seg, prop = [], []
    for run in sorted(root.glob("b*_s*")):
        if (run / "segment_summary.csv").exists():
            seg.append(pd.read_csv(run / "segment_summary.csv"))
        if (run / "propagation_summary.csv").exists():
            prop.append(pd.read_csv(run / "propagation_summary.csv"))
    if len(seg) != 21 or len(prop) != 21:
        raise RuntimeError(f"Expected 21 complete cells; found {len(seg)} segment and {len(prop)} propagation")

    records = []
    for task, df, metrics in [
        ("segment", pd.concat(seg), ["gap", "prec5"]),
        ("propagation", pd.concat(prop), ["prop_N1", "prop_N2", "prop_N4", "prop_N8"]),
    ]:
        for (block, group), q in df.groupby(["block", "group"]):
            for metric in metrics:
                sp = q[q.arm == "Spatial"].set_index("seed")[metric].sort_index()
                bl = q[q.arm == "Baseline"].set_index("seed")[metric].sort_index()
                seeds = sp.index.intersection(bl.index)
                delta = sp.loc[seeds] - bl.loc[seeds]
                _, p = ttest_rel(sp.loc[seeds], bl.loc[seeds])
                records.append(dict(
                    task=task, block=block, group=group, metric=metric, n=len(seeds),
                    spatial=sp.loc[seeds].mean(), baseline=bl.loc[seeds].mean(),
                    diff=delta.mean(), diff_sd=delta.std(ddof=1), p=p,
                    all_same_sign=bool((delta > 0).all() or (delta < 0).all()),
                ))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values(["task", "metric", "group", "block"]).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
