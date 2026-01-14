
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

import lightgbm as lgb


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """대회 컬럼 구성에 맞춘 최소 파생피처."""
    df = df.copy()

    # major_field 기반 파생피처 (존재할 때만)
    if "major_field" in df.columns:
        s = df["major_field"].astype("object")
        s = s.fillna("__MISSING__").astype(str)
        df["major_field"] = s
        df["is_major_it"] = s.str.contains("IT", regex=False).astype(int)

    return df


def drop_high_missing_cols(train_df: pd.DataFrame, test_df: pd.DataFrame, threshold: float):
    """train 기준 결측률이 threshold 초과인 컬럼 드랍."""
    miss = train_df.isna().mean()
    drop_cols = miss[miss > threshold].index.tolist()
    return train_df.drop(columns=drop_cols), test_df.drop(columns=drop_cols, errors="ignore"), drop_cols


def make_categorical_safe(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "completed",
):
    """
    object 컬럼을 pandas category로 변환.
    - train에서 본 카테고리 + __UNKNOWN__ + __MISSING__ 만 허용
    - test에서 train에 없던 값은 __UNKNOWN__으로 매핑
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    # 타깃/ID 제거는 밖에서 하는 걸 권장하지만, 혹시 남아있으면 방어
    for c in [target_col]:
        if c in train_df.columns:
            pass

    cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()

    categories_map = {}
    for col in cat_cols:
        tr = train_df[col].astype("object").fillna("__MISSING__").astype(str)
        # train에서 등장한 카테고리만 기반으로 categories 고정 (+ unknown token)
        cats = pd.Index(tr.unique()).append(pd.Index(["__UNKNOWN__"]))
        cats = cats.drop_duplicates()

        train_df[col] = pd.Categorical(tr, categories=cats)

        te = test_df[col].astype("object").fillna("__MISSING__").astype(str)
        te = te.where(te.isin(cats), "__UNKNOWN__")
        test_df[col] = pd.Categorical(te, categories=cats)

        categories_map[col] = cats

    return train_df, test_df, cat_cols


def main(args):
    train_path = Path(args.train_path)
    test_path = Path(args.test_path)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if args.target_col not in train.columns:
        raise ValueError(f"train에 target 컬럼({args.target_col})이 없습니다.")

    y = train[args.target_col].astype(int)
    X = train.drop(columns=[args.target_col])

    # ID 제거(있으면)
    if args.id_col in X.columns:
        X = X.drop(columns=[args.id_col])
    if args.id_col in test.columns:
        test = test.drop(columns=[args.id_col])

    # 파생피처
    X = build_features(X)
    test = build_features(test)

    # 결측률 높은 컬럼 드랍(train 기준)
    X, test, dropped = drop_high_missing_cols(X, test, threshold=args.drop_missing_threshold)
    print(f"[Info] Dropped {len(dropped)} columns by missing>{args.drop_missing_threshold}: {dropped}")

    # object -> category (unknown 안전 처리)
    X, test, cat_cols = make_categorical_safe(X, test, target_col=args.target_col)
    print(f"[Info] Categorical columns: {len(cat_cols)}")

    # LightGBM 파라미터
    # (기본은 AUC 기준, 불균형 약간 있으니 is_unbalance=True도 괜찮음)
    params = dict(
        objective="binary",
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,   # early stopping으로 best_iteration 사용
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
        n_jobs=-1,
        verbose=-1
        # 불균형 대응 (둘 중 하나만 써도 됨)
        is_unbalance=args.is_unbalance,
    )

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    oof_proba = np.zeros(len(X), dtype=float)
    best_iters = []
    aucs = []
    f1s = []

    print("\n[CV] Start")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=args.log_period),
            ],
            categorical_feature=cat_cols,  # pandas category라서 없어도 인식되지만 명시해도 OK
        )

        proba = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
        oof_proba[va_idx] = proba

        auc = roc_auc_score(y_va, proba)
        pred = (proba >= args.threshold).astype(int)
        f1 = f1_score(y_va, pred)

        aucs.append(auc)
        f1s.append(f1)
        best_iters.append(model.best_iteration_)

        print(f"[Fold {fold}] best_iter={model.best_iteration_} | AUC={auc:.5f} | F1@{args.threshold}={f1:.5f}")

    print("\n[CV] Done")
    print(f"AUC mean={np.mean(aucs):.5f} ± {np.std(aucs):.5f}")
    print(f"F1  mean={np.mean(f1s):.5f} ± {np.std(f1s):.5f}")
    best_iter_final = int(np.median(best_iters))
    print(f"Use n_estimators={best_iter_final} (median best_iteration) for final fit")

    # 최종 학습(전체 데이터)
    final_params = params.copy()
    final_params["n_estimators"] = best_iter_final
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X, y, categorical_feature=cat_cols)

    test_proba = final_model.predict_proba(test)[:, 1]
    test_pred = (test_proba >= args.threshold).astype(int)

    # 제출 파일 생성
    out_path = Path(args.out_path)
    if args.sample_submission_path and Path(args.sample_submission_path).exists():
        sub = pd.read_csv(args.sample_submission_path)
        sub[args.target_col] = test_pred
        sub.to_csv(out_path, index=False)
    else:
        # sample_submission이 없으면 최소 형태로 생성
        sub = pd.DataFrame({args.target_col: test_pred})
        sub.to_csv(out_path, index=False)

    # 확률도 같이 저장(선택)
    if args.out_proba_path:
        proba_path = Path(args.out_proba_path)
        pd.DataFrame({f"{args.target_col}_proba": test_proba}).to_csv(proba_path, index=False)
        print(f"Saved proba:", proba_path)

    print("Saved submission:", out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, default="train.csv")
    p.add_argument("--test_path", type=str, default="test.csv")
    p.add_argument("--sample_submission_path", type=str, default="sample_submission.csv")
    p.add_argument("--out_path", type=str, default="submit_lgbm.csv")
    p.add_argument("--out_proba_path", type=str, default="submit_lgbm_proba.csv")

    p.add_argument("--target_col", type=str, default="completed")
    p.add_argument("--id_col", type=str, default="ID")

    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--drop_missing_threshold", type=float, default=0.8)

    # LGBM params
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--n_estimators", type=int, default=5000)
    p.add_argument("--num_leaves", type=int, default=63)
    p.add_argument("--max_depth", type=int, default=-1)
    p.add_argument("--min_child_samples", type=int, default=20)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument("--reg_lambda", type=float, default=0.0)
    p.add_argument("--is_unbalance", action="store_true")  # 기본 False, 켜고 싶으면 옵션 주기

    # train control
    p.add_argument("--early_stopping_rounds", type=int, default=200)
    p.add_argument("--log_period", type=int, default=200)

    # submission
    p.add_argument("--threshold", type=float, default=0.5)

    args = p.parse_args()
    main(args)
