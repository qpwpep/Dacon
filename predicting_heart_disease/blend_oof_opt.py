from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ID_COL = "id"
TARGET_COL = "target"
SUBMISSION_TARGET_COL = "Heart Disease"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize blend weights from OOF predictions and export blended submission."
    )
    parser.add_argument("--oof-path", required=True)
    parser.add_argument("--test-pred-path", required=True)
    parser.add_argument("--output-submission-path", required=True)
    parser.add_argument("--output-weights-path", required=True)
    parser.add_argument("--strategy", choices=["rank", "prob", "oof_opt"], default="oof_opt")
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--id-col", default=ID_COL)
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--pred-prefix", default="pred_")
    return parser.parse_args()


def get_prediction_columns(
    oof_df: pd.DataFrame, test_df: pd.DataFrame, pred_prefix: str
) -> list[str]:
    oof_cols = [c for c in oof_df.columns if c.startswith(pred_prefix)]
    test_cols = [c for c in test_df.columns if c.startswith(pred_prefix)]
    common = sorted(set(oof_cols).intersection(test_cols))
    if not common:
        raise ValueError(
            f"No prediction columns found with prefix '{pred_prefix}'."
        )
    return common


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def rank_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].rank(method="average", pct=True).to_numpy(dtype=np.float64)


def prob_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].to_numpy(dtype=np.float64)


def coordinate_simplex_search(
    matrix: np.ndarray, y_true: np.ndarray, step: float, max_rounds: int = 500
) -> tuple[np.ndarray, float]:
    n_cols = matrix.shape[1]
    if n_cols == 1:
        pred = matrix[:, 0]
        return np.array([1.0], dtype=np.float64), auc(y_true, pred)

    weights = np.full(n_cols, 1.0 / n_cols, dtype=np.float64)
    current_auc = auc(y_true, matrix @ weights)

    for _ in range(max_rounds):
        best_auc = current_auc
        best_weights: np.ndarray | None = None

        for i in range(n_cols):
            if weights[i] < step - 1e-12:
                continue
            for j in range(n_cols):
                if i == j:
                    continue
                cand = weights.copy()
                cand[i] -= step
                cand[j] += step
                if cand[i] < -1e-12 or cand[j] > 1.0 + 1e-12:
                    continue
                cand = np.clip(cand, 0.0, 1.0)
                cand /= cand.sum()
                cand_auc = auc(y_true, matrix @ cand)
                if cand_auc > best_auc + 1e-12:
                    best_auc = cand_auc
                    best_weights = cand

        if best_weights is None:
            break
        weights = best_weights
        current_auc = best_auc

    return weights, current_auc


def equal_weight_blend(
    oof_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    columns: list[str],
    strategy: str,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    weights = np.full(len(columns), 1.0 / len(columns), dtype=np.float64)
    if strategy == "rank":
        oof_mat = rank_matrix(oof_df, columns)
        test_mat = rank_matrix(test_df, columns)
    else:
        oof_mat = prob_matrix(oof_df, columns)
        test_mat = prob_matrix(test_df, columns)

    oof_blend = oof_mat @ weights
    test_blend = test_mat @ weights
    return oof_blend, test_blend, auc(y_true, oof_blend), weights


def oof_optimized_blend(
    oof_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    columns: list[str],
    step: float,
    top_k: int,
) -> dict[str, Any]:
    individual_auc = []
    for col in columns:
        a = auc(y_true, oof_df[col].to_numpy(dtype=np.float64))
        individual_auc.append((col, a))
    individual_auc.sort(key=lambda x: x[1], reverse=True)
    selected_cols = [c for c, _ in individual_auc[: min(top_k, len(individual_auc))]]

    oof_rank = rank_matrix(oof_df, selected_cols)
    test_rank = rank_matrix(test_df, selected_cols)
    rank_w, rank_auc = coordinate_simplex_search(oof_rank, y_true, step=step)
    rank_oof_blend = oof_rank @ rank_w
    rank_test_blend = test_rank @ rank_w

    oof_prob = prob_matrix(oof_df, selected_cols)
    test_prob = prob_matrix(test_df, selected_cols)
    prob_w, prob_auc = coordinate_simplex_search(oof_prob, y_true, step=step)
    prob_oof_blend = oof_prob @ prob_w
    prob_test_blend = test_prob @ prob_w

    if rank_auc >= prob_auc:
        return {
            "selected_strategy": "rank",
            "selected_columns": selected_cols,
            "selected_weights": rank_w,
            "selected_auc": rank_auc,
            "oof_blend": rank_oof_blend,
            "test_blend": rank_test_blend,
            "comparison": {
                "rank_auc": rank_auc,
                "prob_auc": prob_auc,
            },
            "top_candidates": individual_auc[: min(top_k, len(individual_auc))],
        }

    return {
        "selected_strategy": "prob",
        "selected_columns": selected_cols,
        "selected_weights": prob_w,
        "selected_auc": prob_auc,
        "oof_blend": prob_oof_blend,
        "test_blend": prob_test_blend,
        "comparison": {
            "rank_auc": rank_auc,
            "prob_auc": prob_auc,
        },
        "top_candidates": individual_auc[: min(top_k, len(individual_auc))],
    }


def main() -> None:
    args = parse_args()
    if args.step <= 0:
        raise ValueError("--step must be > 0.")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1.")

    oof_df = pd.read_csv(args.oof_path)
    test_df = pd.read_csv(args.test_pred_path)

    if args.id_col not in oof_df.columns or args.id_col not in test_df.columns:
        raise ValueError(f"'{args.id_col}' column must exist in both inputs.")
    if args.target_col not in oof_df.columns:
        raise ValueError(f"'{args.target_col}' column must exist in OOF input.")
    if oof_df[args.target_col].isna().any():
        raise ValueError("NaN found in OOF target column.")
    if oof_df[args.id_col].duplicated().any():
        raise ValueError("Duplicate IDs found in OOF file.")
    if test_df[args.id_col].duplicated().any():
        raise ValueError("Duplicate IDs found in test prediction file.")

    pred_cols = get_prediction_columns(
        oof_df=oof_df,
        test_df=test_df,
        pred_prefix=args.pred_prefix,
    )
    if oof_df[pred_cols].isna().any().any() or test_df[pred_cols].isna().any().any():
        raise ValueError("NaN found in prediction columns.")

    y_true = oof_df[args.target_col].to_numpy(dtype=np.int32)

    if args.strategy in {"rank", "prob"}:
        oof_blend, test_blend, selected_auc, weights = equal_weight_blend(
            oof_df=oof_df,
            test_df=test_df,
            y_true=y_true,
            columns=pred_cols,
            strategy=args.strategy,
        )
        selected_columns = pred_cols
        selected_strategy = args.strategy
        comparison = {f"{args.strategy}_auc": selected_auc}
        top_candidates = []
    else:
        result = oof_optimized_blend(
            oof_df=oof_df,
            test_df=test_df,
            y_true=y_true,
            columns=pred_cols,
            step=args.step,
            top_k=args.top_k,
        )
        oof_blend = result["oof_blend"]
        test_blend = result["test_blend"]
        selected_auc = float(result["selected_auc"])
        selected_columns = list(result["selected_columns"])
        weights = np.asarray(result["selected_weights"], dtype=np.float64)
        selected_strategy = str(result["selected_strategy"])
        comparison = dict(result["comparison"])
        top_candidates = list(result["top_candidates"])

    if np.isnan(test_blend).any():
        raise ValueError("NaN found in blended test predictions.")
    if (weights < -1e-12).any():
        raise ValueError("Negative blend weights found.")

    submission_df = pd.DataFrame(
        {
            args.id_col: test_df[args.id_col],
            SUBMISSION_TARGET_COL: np.clip(test_blend, 0.0, 1.0),
        }
    )

    weights_payload = {
        "strategy_requested": args.strategy,
        "selected_strategy": selected_strategy,
        "selected_auc": selected_auc,
        "step": args.step,
        "top_k": args.top_k,
        "n_models_total": len(pred_cols),
        "selected_columns": selected_columns,
        "weights": {col: float(w) for col, w in zip(selected_columns, weights)},
        "comparison": comparison,
        "top_candidates": top_candidates,
        "oof_blend_auc_check": auc(y_true, np.asarray(oof_blend, dtype=np.float64)),
    }

    output_submission = Path(args.output_submission_path)
    output_weights = Path(args.output_weights_path)
    output_submission.parent.mkdir(parents=True, exist_ok=True)
    output_weights.parent.mkdir(parents=True, exist_ok=True)

    submission_df.to_csv(output_submission, index=False)
    with open(output_weights, "w", encoding="utf-8") as f:
        json.dump(weights_payload, f, indent=2, ensure_ascii=False)

    print("[Blend Complete]")
    print(f"strategy_requested={args.strategy}")
    print(f"selected_strategy={selected_strategy}")
    print(f"selected_auc={selected_auc:.6f}")
    print(f"output_submission={output_submission}")
    print(f"output_weights={output_weights}")


if __name__ == "__main__":
    main()
