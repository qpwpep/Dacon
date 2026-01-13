#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eda.py
- 학습(train) 데이터를 빠르게 진단하고, 결과(표/그래프)를 outputs/eda/ 아래에 저장합니다.
- seaborn이 없거나 깨져 있어도(설치 문제) 동작하도록, 기본은 matplotlib만 사용합니다.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def basic_overview(df: pd.DataFrame, name: str = "train") -> str:
    lines = []
    lines.append(f"[{name}] shape = {df.shape[0]} rows x {df.shape[1]} cols")
    lines.append(f"[{name}] dtypes:\n{df.dtypes.value_counts().to_string()}")
    nunique = df.nunique(dropna=True).sort_values(ascending=False).head(15)
    lines.append(f"[{name}] top-15 nunique:\n{nunique.to_string()}")
    return "\n".join(lines)


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False)
    out = pd.DataFrame({"missing_ratio": miss, "missing_count": df.isna().sum()})
    return out


def plot_missingness(miss_df: pd.DataFrame, outdir: Path, topk: int = 25, title: str = "Missingness (top)") -> None:
    top = miss_df.head(topk).iloc[::-1]  # reverse for nicer barh
    plt.figure(figsize=(10, max(4, topk * 0.3)))
    plt.barh(top.index.astype(str), top["missing_ratio"].values)
    plt.title(title)
    plt.xlabel("missing ratio")
    plt.tight_layout()
    plt.savefig(outdir / "missingness_top.png", dpi=160)
    plt.close()


def target_report(df: pd.DataFrame, target: str, outdir: Path) -> str:
    vc = df[target].value_counts(dropna=False)
    pos_rate = float(df[target].mean()) if df[target].dropna().nunique() > 1 else float("nan")
    plt.figure(figsize=(5, 4))
    plt.bar(vc.index.astype(str), vc.values)
    plt.title(f"Target distribution: {target} (pos_rate={pos_rate:.3f})")
    plt.tight_layout()
    plt.savefig(outdir / "target_distribution.png", dpi=160)
    plt.close()
    return f"[target] {target} value_counts:\n{vc.to_string()}\n[target] positive_rate={pos_rate:.6f}"


def plot_numeric_distributions(df: pd.DataFrame, outdir: Path, exclude: set[str] | None = None) -> None:
    exclude = exclude or set()
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c not in exclude]
    for col in num_cols:
        s = df[col].dropna()
        plt.figure(figsize=(6, 4))
        plt.hist(s.values, bins=30)
        plt.title(f"Histogram: {col}")
        plt.tight_layout()
        plt.savefig(outdir / f"hist_{col}.png", dpi=160)
        plt.close()


def plot_categorical_top(df: pd.DataFrame, outdir: Path, target: str | None, topk: int = 15) -> None:
    cat_cols = df.select_dtypes(include=["object", "bool"]).columns
    for col in cat_cols:
        # 너무 고카디널리티(텍스트)인 컬럼은 자동 스킵
        nunique = df[col].nunique(dropna=True)
        if nunique > 50:
            continue

        vc = df[col].astype("object").fillna("<<NA>>").value_counts().head(topk)
        plt.figure(figsize=(10, max(4, len(vc) * 0.35)))
        plt.barh(vc.index.astype(str)[::-1], vc.values[::-1])
        plt.title(f"Top categories: {col} (nunique={nunique})")
        plt.tight_layout()
        plt.savefig(outdir / f"cat_top_{col}.png", dpi=160)
        plt.close()

        # 타깃이 있으면, 각 카테고리별 완료율도 함께 저장
        if target and target in df.columns:
            tmp = df[[col, target]].copy()
            tmp[col] = tmp[col].astype("object").fillna("<<NA>>")
            rate = tmp.groupby(col)[target].agg(["count", "mean"]).sort_values("count", ascending=False).head(topk)
            rate.to_csv(outdir / f"cat_target_rate_{col}.csv", encoding="utf-8-sig")


def plot_numeric_correlation(df: pd.DataFrame, outdir: Path, title: str = "Correlation (numeric only)") -> None:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return
    corr = num.corr().values

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xticks(range(num.shape[1]), num.columns, rotation=90)
    plt.yticks(range(num.shape[1]), num.columns)
    plt.tight_layout()
    plt.savefig(outdir / "corr_numeric.png", dpi=160)
    plt.close()
    num.corr().to_csv(outdir / "corr_numeric.csv", encoding="utf-8-sig")


def compare_train_test(train: pd.DataFrame, test: pd.DataFrame) -> str:
    lines = []
    train_cols = set(train.columns)
    test_cols = set(test.columns)

    only_train = sorted(train_cols - test_cols)
    only_test = sorted(test_cols - train_cols)
    common = sorted(train_cols & test_cols)

    lines.append(f"[compare] common_cols={len(common)} only_train={len(only_train)} only_test={len(only_test)}")
    if only_train:
        lines.append(f"[compare] columns only in train: {only_train[:30]}{' ...' if len(only_train) > 30 else ''}")
    if only_test:
        lines.append(f"[compare] columns only in test: {only_test[:30]}{' ...' if len(only_test) > 30 else ''}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="EDA script (saves plots/tables to outputs/eda)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python eda.py --train train.csv
              python eda.py --train data/train.csv --test data/test.csv --outdir outputs/eda
            """
        ),
    )
    ap.add_argument("--train", type=str, required=True, help="path to train.csv (must include target if available)")
    ap.add_argument("--test", type=str, default=None, help="optional path to test.csv for distribution/column checks")
    ap.add_argument("--target", type=str, default="completed", help="target column name in train")
    ap.add_argument("--id-col", type=str, default="ID", help="id column name")
    ap.add_argument("--outdir", type=str, default="outputs/eda", help="output directory")
    ap.add_argument("--topk", type=int, default=25, help="top-k categories/columns to show")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    _safe_mkdir(outdir)

    train = load_csv(args.train)
    report_lines = []
    report_lines.append(basic_overview(train, "train"))

    miss_train = missingness_report(train)
    miss_train.to_csv(outdir / "missingness_train.csv", encoding="utf-8-sig")
    plot_missingness(miss_train, outdir, topk=args.topk, title="Train missingness (top)")

    if args.target in train.columns:
        report_lines.append(target_report(train, args.target, outdir))

    # 숫자 분포(타깃/ID는 제외)
    exclude = {args.target, args.id_col}
    plot_numeric_distributions(train, outdir, exclude=exclude)

    # 카테고리 top + (가능하면) 완료율 테이블
    plot_categorical_top(train, outdir, target=(args.target if args.target in train.columns else None), topk=min(args.topk, 20))

    # numeric correlation
    plot_numeric_correlation(train, outdir)

    if args.test:
        test = load_csv(args.test)
        report_lines.append(basic_overview(test, "test"))
        report_lines.append(compare_train_test(train, test))
        miss_test = missingness_report(test)
        miss_test.to_csv(outdir / "missingness_test.csv", encoding="utf-8-sig")
        plot_missingness(miss_test, outdir, topk=args.topk, title="Test missingness (top)")

    (outdir / "eda_report.txt").write_text("\n\n".join(report_lines), encoding="utf-8")
    print(f"[OK] EDA artifacts saved to: {outdir.resolve()}")
    print(f"- report: {outdir/'eda_report.txt'}")
    print(f"- plots:  {outdir/'missingness_top.png'}, {outdir/'corr_numeric.png'}, etc.")


if __name__ == "__main__":
    main()
