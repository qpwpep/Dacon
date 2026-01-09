from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


def parse_label_glob(x: str) -> Tuple[str, str]:
    # format: label=glob
    if "=" not in x:
        raise ValueError(f"--dump must be like label=glob, got: {x}")
    label, glob = x.split("=", 1)
    label = label.strip()
    glob = glob.strip()
    if not label:
        raise ValueError(f"Empty label in: {x}")
    return label, glob


def make_bin_labels(bins: List[float]) -> List[str]:
    labels = []
    for i in range(len(bins) - 1):
        labels.append(f"[{bins[i]}, {bins[i+1]})")
    return labels


def quantile(x: pd.Series, q: float) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.quantile(x.to_numpy(dtype=np.float64), q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dump",
        action="append",
        required=True,
        help="One or more: label=glob (e.g. base=outputs/base_dumps/*.csv). Can be repeated.",
    )
    ap.add_argument("--out_dir", default="outputs/error_bin_tables")
    ap.add_argument("--bins", default="0,10,20,30,40,50,60,70,80,1000000000.0")
    ap.add_argument("--bin_features_csv", default=None, help="Optional: finetune_dist_bins_by_fold.csv")
    ap.add_argument("--sat_features_csv", default=None, help="Optional: finetune_sat95_by_fold.csv")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bins = [float(x) for x in args.bins.split(",")]
    assert all(bins[i] < bins[i + 1] for i in range(len(bins) - 1)), "bins must be strictly increasing"
    labels = make_bin_labels(bins)

    # 1) load dumps
    frames = []
    for item in args.dump:
        label, glob = parse_label_glob(item)
        paths = list(Path().glob(glob)) if not glob.startswith("/") else list(Path("/").glob(glob[1:]))
        # safer glob:
        paths = list(Path().glob(glob))
        if len(paths) == 0:
            raise FileNotFoundError(f"No files for: {label}={glob}")

        for p in paths:
            df = pd.read_csv(p)
            df["method"] = label
            frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # enforce
    need_cols = ["fold", "stage", "split", "tgt_dist_m", "sat95", "err_dist_m"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in dumps. columns={list(df.columns)}")

    # (optional) keep finetune/valid only by default
    df = df[(df["stage"] == "finetune") & (df["split"] == "valid")].copy()

    # 2) dist bins from tgt_dist_m
    df["dist_bin"] = pd.cut(
        df["tgt_dist_m"].astype(float),
        bins=bins,
        right=False,
        labels=labels,
        include_lowest=True,
    ).astype(str)

    # 3) error bin table (long)
    gcols = ["method", "fold", "split", "dist_bin"]
    agg = df.groupby(gcols, observed=True).agg(
        n=("err_dist_m", "size"),
        err_mean=("err_dist_m", "mean"),
        err_p50=("err_dist_m", lambda x: quantile(x, 0.50)),
        err_p90=("err_dist_m", lambda x: quantile(x, 0.90)),
        err_p99=("err_dist_m", lambda x: quantile(x, 0.99)),
        tgt_p50=("tgt_dist_m", lambda x: quantile(x, 0.50)),
        tgt_p90=("tgt_dist_m", lambda x: quantile(x, 0.90)),
        sat95_rate=("sat95", "mean"),
    ).reset_index()

    # 4) join bin features if provided
    if args.bin_features_csv:
        feat = pd.read_csv(args.bin_features_csv)
        feat = feat[feat["split"] == "valid"].copy()
        # expected keys: fold, split, dist_bin
        keep_cols = [c for c in feat.columns if c not in ("stage",)]
        feat = feat[keep_cols]
        agg = agg.merge(feat, on=["fold", "split", "dist_bin"], how="left", suffixes=("", "_feat"))

    out_long = out_dir / "error_bins_long.csv"
    agg.to_csv(out_long, index=False)
    print("Saved:", out_long)

    # 5) wide compare if multiple methods (delta columns)
    methods = sorted(df["method"].unique().tolist())
    if len(methods) >= 2:
        pivot = agg.pivot_table(
            index=["fold", "split", "dist_bin"],
            columns="method",
            values=["err_mean", "err_p50", "err_p90", "err_p99", "n"],
            aggfunc="first",
        )
        pivot.columns = [f"{a}__{b}" for a, b in pivot.columns.to_list()]
        pivot = pivot.reset_index()

        # if exactly two methods, add delta (2nd - 1st)
        if len(methods) == 2:
            a, b = methods[0], methods[1]
            for metric in ["err_mean", "err_p50", "err_p90", "err_p99"]:
                ca = f"{metric}__{a}"
                cb = f"{metric}__{b}"
                if ca in pivot.columns and cb in pivot.columns:
                    pivot[f"{metric}__delta({b}-{a})"] = pivot[cb] - pivot[ca]

        out_wide = out_dir / "error_bins_wide.csv"
        pivot.to_csv(out_wide, index=False)
        print("Saved:", out_wide)

    # 6) sat95-only table (+ optional join)
    sat_df = df[df["sat95"] == True].copy()
    sat_agg = sat_df.groupby(["method", "fold", "split"], observed=True).agg(
        n=("err_dist_m", "size"),
        err_mean=("err_dist_m", "mean"),
        err_p50=("err_dist_m", lambda x: quantile(x, 0.50)),
        err_p90=("err_dist_m", lambda x: quantile(x, 0.90)),
        tgt_p50=("tgt_dist_m", lambda x: quantile(x, 0.50)),
        tgt_p90=("tgt_dist_m", lambda x: quantile(x, 0.90)),
    ).reset_index()

    if args.sat_features_csv:
        sat_feat = pd.read_csv(args.sat_features_csv)
        sat_feat = sat_feat[sat_feat["split"] == "valid"].copy()
        sat_agg = sat_agg.merge(sat_feat, on=["fold", "split"], how="left", suffixes=("", "_feat"))

    out_sat = out_dir / "sat95_errors_by_fold.csv"
    sat_agg.to_csv(out_sat, index=False)
    print("Saved:", out_sat)


if __name__ == "__main__":
    main()
