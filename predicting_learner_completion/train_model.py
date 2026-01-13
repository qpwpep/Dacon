#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_model.py
- baseline.py의 "모델 학습/예측/제출" 파트를 EDA와 분리한 버전입니다.
- (권장) sklearn Pipeline + ColumnTransformer로 결측치/범주형 인코딩을 일관되게 처리합니다.
- seaborn에 의존하지 않으며, 데이터 경로를 인자로 받습니다.

Examples:
  python train_model.py --train train.csv --test test.csv --sample-sub sample_submission.csv --out submit.csv
  # uv 사용 시:
  # uv run python train_model.py --train data/train.csv --test data/test.csv --sample-sub data/sample_submission.csv --out submit.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    baseline.py의 is_major_it 생성(major_field에 'IT' 포함 여부)을 좀 더 튼튼하게 확장.
    - 원본 baseline: str.contains('IT') (대문자 IT만 잡힘)
    - 여기서는 IT/컴퓨터/SW/소프트웨어/전산 등을 포함하면 1로 처리(없으면 0)
    """
    out = df.copy()
    if "major_field" in out.columns:
        s = out["major_field"].fillna("").astype(str)
        pattern = r"(IT|컴퓨터|전산|소프트웨어|SW|S/W|Computer|CS)"
        out["is_major_it"] = s.str.contains(pattern, regex=True, case=False).astype(int)
    else:
        out["is_major_it"] = 0
    return out


def drop_high_missing(train: pd.DataFrame, test: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    miss = train.isna().mean()
    drop_cols = miss[miss > threshold].index.tolist()
    train2 = train.drop(columns=drop_cols, errors="ignore")
    test2 = test.drop(columns=drop_cols, errors="ignore")
    return train2, test2, drop_cols


def build_pipeline(X: pd.DataFrame, seed: int = 42) -> Pipeline:
    # ID는 학습에 쓰지 않는 것을 권장(대부분 누수/잡음)
    # 여기서는 호출부에서 ID를 빼고 들어온다고 가정
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )

    pipe = Pipeline([("preprocess", pre), ("model", model)])
    return pipe


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train model and create submission (EDA separated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Tips:
              - 먼저 eda.py로 데이터 상태(결측, 분포, 불균형)를 확인한 뒤 모델링을 진행하세요.
              - baseline.py처럼 특정 5개 피처만 쓰고 싶다면, --feature-cols 옵션으로 제한해도 됩니다.
            """
        ),
    )
    ap.add_argument("--train", type=str, required=True, help="path to train.csv (must include target)")
    ap.add_argument("--test", type=str, required=True, help="path to test.csv")
    ap.add_argument("--sample-sub", type=str, required=True, help="path to sample_submission.csv")
    ap.add_argument("--out", type=str, default="submit.csv", help="output submission csv path")
    ap.add_argument("--target", type=str, default="completed", help="target column name")
    ap.add_argument("--id-col", type=str, default="ID", help="id column name")
    ap.add_argument("--missing-threshold", type=float, default=0.8, help="drop columns with missing ratio > threshold (train 기준)")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--val-size", type=float, default=0.2, help="validation split ratio for quick check")
    ap.add_argument("--feature-cols", type=str, default=None,
                    help="comma-separated list of feature columns to use (optional). e.g. class1,re_registration,inflow_route,time_input,is_major_it")
    args = ap.parse_args()

    train = load_csv(args.train)
    test = load_csv(args.test)
    sub = load_csv(args.sample_sub)

    assert args.target in train.columns, f"Target column '{args.target}' not found in train"

    # feature engineering
    train = add_features(train)
    test = add_features(test)

    # drop high-missing columns based on train
    train, test, dropped = drop_high_missing(train, test, args.missing_threshold)

    # 선택 피처 제한 옵션(= baseline과 비슷하게 5개만 쓰고 싶을 때)
    if args.feature_cols:
        keep = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
        # target/id는 별도 처리
        keep = [c for c in keep if c in train.columns]
    else:
        # 기본: ID/target 제외하고 전부 사용 (원본 baseline보다 강력한 기본값)
        keep = [c for c in train.columns if c not in {args.target, args.id_col}]

    X = train[keep].copy()
    y = train[args.target].astype(int).copy()

    # 간단 검증
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=args.val_size, random_state=args.seed, stratify=y)
    pipe = build_pipeline(X_tr, seed=args.seed)
    pipe.fit(X_tr, y_tr)

    # 평가(가능하면 proba 기반 AUC)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, proba)
        pred = (proba >= 0.5).astype(int)
    else:
        pred = pipe.predict(X_va)
        auc = float("nan")

    acc = accuracy_score(y_va, pred)
    f1 = f1_score(y_va, pred)

    print(f"[val] acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")
    if dropped:
        print(f"[info] dropped {len(dropped)} high-missing columns (threshold>{args.missing_threshold}): {dropped}")

    # 전체 train으로 재학습 후 test 예측
    pipe = build_pipeline(X, seed=args.seed)
    pipe.fit(X, y)

    X_test = test[keep].copy()
    if hasattr(pipe, "predict_proba"):
        proba_test = pipe.predict_proba(X_test)[:, 1]
        pred_test = (proba_test >= 0.5).astype(int)
    else:
        pred_test = pipe.predict(X_test)

    # 제출 파일 생성
    out_path = Path(args.out)
    submission = sub.copy()
    if args.id_col in submission.columns and args.id_col in test.columns:
        # sample_submission의 ID 순서가 test와 다를 수 있음. 가능하면 merge로 맞춤.
        if submission[args.id_col].nunique() == len(submission) and test[args.id_col].nunique() == len(test):
            tmp = pd.DataFrame({args.id_col: test[args.id_col].values, args.target: pred_test})
            submission = submission.drop(columns=[args.target], errors="ignore").merge(tmp, on=args.id_col, how="left")
        else:
            submission[args.target] = pred_test
    else:
        submission[args.target] = pred_test

    submission.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] saved submission to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
