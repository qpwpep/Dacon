from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET_COL = "Heart Disease"
ID_COL = "id"
FEATURES = [
    "Age",
    "Sex",
    "Chest pain type",
    "BP",
    "Cholesterol",
    "FBS over 120",
    "EKG results",
    "Max HR",
    "Exercise angina",
    "ST depression",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]
CAT_FEATURES = [
    "Sex",
    "Chest pain type",
    "FBS over 120",
    "EKG results",
    "Exercise angina",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]
TARGET_MAP = {"Absence": 0, "Presence": 1}

FEATURE_DTYPES: dict[str, str] = {
    ID_COL: "int32",
    "Age": "int16",
    "Sex": "int8",
    "Chest pain type": "int8",
    "BP": "int16",
    "Cholesterol": "int16",
    "FBS over 120": "int8",
    "EKG results": "int8",
    "Max HR": "int16",
    "Exercise angina": "int8",
    "ST depression": "float32",
    "Slope of ST": "int8",
    "Number of vessels fluro": "int8",
    "Thallium": "int8",
}

PRESETS: dict[str, dict[str, float | int]] = {
    "robust": {"iterations": 5000, "od_wait": 200},
    "fast": {"iterations": 1500, "od_wait": 100},
    "aggressive": {"iterations": 7000, "od_wait": 300},
}

ENSEMBLE_CONFIGS: dict[str, list[dict[str, float | str]]] = {
    "robust_set_v1": [
        {
            "config_id": "m1",
            "model_name": "M1",
            "depth": 6.0,
            "learning_rate": 0.03,
            "l2_leaf_reg": 9.0,
            "random_strength": 1.0,
            "bagging_temperature": 0.0,
        },
        {
            "config_id": "m2",
            "model_name": "M2",
            "depth": 7.0,
            "learning_rate": 0.025,
            "l2_leaf_reg": 11.0,
            "random_strength": 2.0,
            "bagging_temperature": 1.0,
        },
        {
            "config_id": "m3",
            "model_name": "M3",
            "depth": 5.0,
            "learning_rate": 0.04,
            "l2_leaf_reg": 7.0,
            "random_strength": 1.0,
            "bagging_temperature": 0.0,
        },
    ]
}

LOG_LEVEL_RANK = {"minimal": 0, "default": 1, "debug": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CatBoost CV ensembles and export standardized artifacts."
    )
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test.csv")
    parser.add_argument("--sample-sub-path", default="data/sample_submission.csv")
    parser.add_argument("--output-path", default=None)

    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-list", default="42,52,62")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--feature-mode",
        choices=["cat", "numeric", "hybrid"],
        default="cat",
    )
    parser.add_argument(
        "--model-family",
        choices=["cat_only", "cat_plus_numeric"],
        default="cat_only",
    )
    parser.add_argument("--hybrid-weight", type=float, default=0.5)
    parser.add_argument("--preset", choices=["robust", "fast", "aggressive"], default="robust")
    parser.add_argument(
        "--ensemble-config",
        choices=sorted(ENSEMBLE_CONFIGS.keys()),
        default="robust_set_v1",
    )
    parser.add_argument(
        "--blend-strategy",
        choices=["none", "rank", "prob", "oof_opt"],
        default="none",
    )

    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--l2-leaf-reg", type=float, default=None)
    parser.add_argument("--od-wait", type=int, default=None)
    parser.add_argument("--verbose-eval", type=int, default=200)

    parser.add_argument("--save-oof", choices=["true", "false"], default="true")
    parser.add_argument("--save-model-detail", choices=["true", "false"], default="true")
    parser.add_argument("--save-model-oof", choices=["true", "false"], default="true")
    parser.add_argument("--public-score-manual", type=float, default=None)
    parser.add_argument(
        "--log-level",
        choices=["minimal", "default", "debug"],
        default="minimal",
    )
    return parser.parse_args()


def parse_bool_flag(value: str) -> bool:
    return value.strip().lower() == "true"


def resolve_hyperparameters(args: argparse.Namespace) -> dict[str, float | int]:
    resolved = dict(PRESETS[args.preset])
    if args.iterations is not None:
        resolved["iterations"] = args.iterations
    if args.od_wait is not None:
        resolved["od_wait"] = args.od_wait
    return resolved


def log(message: str, log_level: str, min_level: str = "default") -> None:
    if LOG_LEVEL_RANK[log_level] >= LOG_LEVEL_RANK[min_level]:
        print(message)


def validate_required_columns(
    df: pd.DataFrame, required_cols: list[str], frame_name: str
) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {frame_name}: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )


def validate_id_uniqueness(df: pd.DataFrame, frame_name: str) -> None:
    if df[ID_COL].isna().any():
        raise ValueError(f"NaN found in {ID_COL} column of {frame_name}.")
    dup_count = int(df[ID_COL].duplicated().sum())
    if dup_count > 0:
        dup_samples = df.loc[df[ID_COL].duplicated(), ID_COL].head(10).tolist()
        raise ValueError(
            f"Duplicate {ID_COL} values found in {frame_name}: count={dup_count}, "
            f"examples={dup_samples}"
        )


def validate_sample_test_id_set(sample_sub_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    sample_ids = pd.Index(sample_sub_df[ID_COL])
    test_ids = pd.Index(test_df[ID_COL])

    missing_in_sample = test_ids.difference(sample_ids)
    extra_in_sample = sample_ids.difference(test_ids)
    if len(missing_in_sample) > 0 or len(extra_in_sample) > 0:
        raise ValueError(
            "sample_submission and test id sets do not match. "
            f"missing_in_sample={len(missing_in_sample)}, "
            f"extra_in_sample={len(extra_in_sample)}"
        )


def parse_seeds(seed: int, seed_list: str | None) -> list[int]:
    if seed_list is None or seed_list.strip() == "":
        return [seed]
    raw_items = [item.strip() for item in seed_list.split(",") if item.strip()]
    if not raw_items:
        raise ValueError("--seed-list was provided but no valid seeds were found.")

    parsed: list[int] = []
    seen: set[int] = set()
    for item in raw_items:
        parsed_seed = int(item)
        if parsed_seed not in seen:
            parsed.append(parsed_seed)
            seen.add(parsed_seed)
    return parsed


def sanitize_run_name(run_name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    sanitized = "".join(ch if ch in allowed else "_" for ch in run_name)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    sanitized = sanitized.strip("_")
    if not sanitized:
        raise ValueError("Invalid --run-name after sanitization.")
    return sanitized


def default_run_name(
    args: argparse.Namespace,
    model_configs: list[dict[str, Any]],
    seeds: list[int],
) -> str:
    cfg = model_configs[0]
    lr = str(cfg["learning_rate"]).replace(".", "p")
    l2 = str(cfg["l2_leaf_reg"]).replace(".", "p")
    return (
        f"cb_d{cfg['depth']}_lr{lr}_l2_{l2}_{args.feature_mode}_{args.model_family}"
        f"_{args.ensemble_config}_s{len(seeds)}"
    )


def map_target(target_series: pd.Series) -> pd.Series:
    mapped = target_series.map(TARGET_MAP)
    if mapped.isna().any():
        unknown_labels = sorted(
            target_series[mapped.isna()].astype(str).unique().tolist()
        )
        raise ValueError(
            "Unknown labels found in target column. "
            f"Allowed labels: {list(TARGET_MAP.keys())}. "
            f"Found unknown labels: {unknown_labels}"
        )
    return mapped.astype(np.int8)


def print_dataset_overview(
    train_df: pd.DataFrame, test_df: pd.DataFrame, log_level: str
) -> None:
    log("[Data Overview]", log_level, "default")
    log(f"train shape: {train_df.shape}", log_level, "default")
    log(f"test shape: {test_df.shape}", log_level, "default")

    if LOG_LEVEL_RANK[log_level] < LOG_LEVEL_RANK["debug"]:
        return

    print("\n[Dtypes]")
    print("train dtype counts:")
    print(train_df.dtypes.value_counts().to_string())
    print("\ntrain dtypes by column:")
    print(train_df.dtypes.to_string())

    print("\n[Missing Values - Top 20]")
    print("train:")
    print(train_df.isna().sum().sort_values(ascending=False).head(20).to_string())
    print("\ntest:")
    print(test_df.isna().sum().sort_values(ascending=False).head(20).to_string())

    print("\n[Duplicates]")
    print(f"train duplicated(all columns): {train_df.duplicated().sum()}")
    print(
        f"train duplicated(excluding {ID_COL}): "
        f"{train_df.drop(columns=[ID_COL]).duplicated().sum()}"
    )
    print(f"test duplicated(all columns): {test_df.duplicated().sum()}")
    print(
        f"test duplicated(excluding {ID_COL}): "
        f"{test_df.drop(columns=[ID_COL]).duplicated().sum()}"
    )


def resolve_model_configs(
    args: argparse.Namespace,
    hp: dict[str, float | int],
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    base_configs = ENSEMBLE_CONFIGS[args.ensemble_config]
    for cfg in base_configs:
        configs.append(
            {
                "config_id": str(cfg["config_id"]),
                "model_name": str(cfg["model_name"]),
                "depth": int(args.depth if args.depth is not None else int(cfg["depth"])),
                "learning_rate": float(
                    args.learning_rate
                    if args.learning_rate is not None
                    else float(cfg["learning_rate"])
                ),
                "l2_leaf_reg": float(
                    args.l2_leaf_reg
                    if args.l2_leaf_reg is not None
                    else float(cfg["l2_leaf_reg"])
                ),
                "random_strength": float(cfg["random_strength"]),
                "bagging_temperature": float(cfg["bagging_temperature"]),
                "iterations": int(hp["iterations"]),
                "od_wait": int(hp["od_wait"]),
            }
        )
    return configs


def build_model(model_cfg: dict[str, Any], seed: int, verbose_eval: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=int(model_cfg["iterations"]),
        learning_rate=float(model_cfg["learning_rate"]),
        depth=int(model_cfg["depth"]),
        l2_leaf_reg=float(model_cfg["l2_leaf_reg"]),
        random_strength=float(model_cfg["random_strength"]),
        bootstrap_type="Bayesian",
        bagging_temperature=float(model_cfg["bagging_temperature"]),
        random_seed=seed,
        use_best_model=True,
        od_type="Iter",
        od_wait=int(model_cfg["od_wait"]),
        allow_writing_files=False,
        verbose=verbose_eval,
    )


def resolve_submode_weights(
    model_family: str, feature_mode: str, hybrid_weight: float
) -> dict[str, float]:
    if model_family == "cat_only":
        return {"cat": 1.0}

    if feature_mode == "cat":
        return {"cat": 1.0}
    if feature_mode == "numeric":
        return {"numeric": 1.0}

    weights = {"cat": hybrid_weight, "numeric": 1.0 - hybrid_weight}
    return {k: v for k, v in weights.items() if v > 0.0}


def get_git_commit_hash() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    commit = proc.stdout.strip()
    return commit if commit else None


def load_data(
    train_path: Path, test_path: Path, sample_sub_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(
        train_path,
        usecols=[ID_COL, *FEATURES, TARGET_COL],
        dtype={**FEATURE_DTYPES, TARGET_COL: "string"},
    )
    test_df = pd.read_csv(
        test_path,
        usecols=[ID_COL, *FEATURES],
        dtype=FEATURE_DTYPES,
    )
    sample_sub_df = pd.read_csv(
        sample_sub_path,
        usecols=[ID_COL, TARGET_COL],
        dtype={ID_COL: FEATURE_DTYPES[ID_COL]},
    )
    return train_df, test_df, sample_sub_df


def build_output_paths(
    output_path_arg: str | None,
    run_name: str,
    n_splits: int,
) -> dict[str, Path]:
    if output_path_arg:
        submission_path = Path(output_path_arg)
    else:
        submission_path = Path("outputs") / f"submission_{run_name}_cv{n_splits}.csv"

    out_dir = submission_path.parent
    suffix = f"{run_name}_cv{n_splits}"
    return {
        "submission": submission_path,
        "submission_blend": out_dir / f"submission_blend_{suffix}.csv",
        "weights": out_dir / f"weights_{suffix}.json",
        "oof": out_dir / f"oof_{suffix}.csv",
        "fold_summary": out_dir / f"fold_summary_{suffix}.csv",
        "model_detail": out_dir / f"model_detail_{suffix}.csv",
        "model_oof": out_dir / f"model_oof_{suffix}.csv",
        "model_test_pred": out_dir / f"model_test_pred_{suffix}.csv",
        "feature_importance_merged": out_dir / f"feature_importance_merged_{suffix}.csv",
        "feature_importance_cat": out_dir / f"feature_importance_cat_{suffix}.csv",
        "feature_importance_numeric": out_dir / f"feature_importance_numeric_{suffix}.csv",
        "metrics": out_dir / f"metrics_{suffix}.json",
    }


def feature_importance_df(importances: list[np.ndarray], n_models: int) -> pd.DataFrame:
    matrix = np.vstack(importances).astype(np.float64)
    return pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": matrix.mean(axis=0),
            "importance_std": matrix.std(axis=0),
            "n_models": n_models,
        }
    ).sort_values(by="importance_mean", ascending=False)


def save_base_blend_artifacts(
    output_paths: dict[str, Path],
    submission_df: pd.DataFrame,
    model_weight_map: dict[str, float],
    oof_auc: float,
) -> dict[str, Any]:
    submission_df.to_csv(output_paths["submission_blend"], index=False)
    payload: dict[str, Any] = {
        "strategy": "none",
        "selected_strategy": "base_ensemble",
        "selected_auc": float(oof_auc),
        "selected_columns": sorted(model_weight_map.keys()),
        "weights": model_weight_map,
    }
    with open(output_paths["weights"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def run_blending_script(
    output_paths: dict[str, Path],
    blend_strategy: str,
    log_level: str,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("blend_oof_opt.py")),
        "--oof-path",
        str(output_paths["model_oof"]),
        "--test-pred-path",
        str(output_paths["model_test_pred"]),
        "--output-submission-path",
        str(output_paths["submission_blend"]),
        "--output-weights-path",
        str(output_paths["weights"]),
        "--strategy",
        blend_strategy,
        "--step",
        "0.02",
        "--top-k",
        "5",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "blend_oof_opt.py failed.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    if proc.stdout.strip():
        log(proc.stdout.strip(), log_level, "default")
    if proc.stderr.strip():
        log(proc.stderr.strip(), log_level, "debug")

    with open(output_paths["weights"], "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload


def main() -> None:
    args = parse_args()
    save_oof = parse_bool_flag(args.save_oof)
    save_model_detail = parse_bool_flag(args.save_model_detail)
    save_model_oof = parse_bool_flag(args.save_model_oof)

    if args.n_splits < 2:
        raise ValueError("--n-splits must be >= 2.")
    if not (0.0 <= args.hybrid_weight <= 1.0):
        raise ValueError("--hybrid-weight must be in [0, 1].")

    hp = resolve_hyperparameters(args)
    if int(hp["iterations"]) < 1:
        raise ValueError("--iterations must be >= 1.")
    if int(hp["od_wait"]) < 1:
        raise ValueError("--od-wait must be >= 1.")

    seeds = parse_seeds(seed=args.seed, seed_list=args.seed_list)
    model_configs = resolve_model_configs(args=args, hp=hp)
    submode_weights = resolve_submode_weights(
        model_family=args.model_family,
        feature_mode=args.feature_mode,
        hybrid_weight=args.hybrid_weight,
    )
    if not submode_weights:
        raise ValueError(
            "No active submode after applying --model-family/--feature-mode/--hybrid-weight."
        )

    if args.blend_strategy != "none" and not save_model_oof:
        raise ValueError("--blend-strategy requires --save-model-oof true.")

    run_name = sanitize_run_name(args.run_name or default_run_name(args, model_configs, seeds))

    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    sample_sub_path = Path(args.sample_sub_path)

    log("[Load CSV]", args.log_level, "default")
    train_df, test_df, sample_sub_df = load_data(train_path, test_path, sample_sub_path)

    validate_required_columns(
        train_df, [ID_COL, *FEATURES, TARGET_COL], frame_name="train"
    )
    validate_required_columns(test_df, [ID_COL, *FEATURES], frame_name="test")
    validate_required_columns(
        sample_sub_df, [ID_COL, TARGET_COL], frame_name="sample_submission"
    )
    validate_id_uniqueness(train_df, frame_name="train")
    validate_id_uniqueness(test_df, frame_name="test")
    validate_id_uniqueness(sample_sub_df, frame_name="sample_submission")
    validate_sample_test_id_set(sample_sub_df=sample_sub_df, test_df=test_df)

    print_dataset_overview(train_df, test_df, args.log_level)

    y = map_target(train_df[TARGET_COL])
    target_counts = train_df[TARGET_COL].value_counts(dropna=False)
    log("\n[Target Distribution]", args.log_level, "default")
    log(target_counts.to_string(), args.log_level, "default")
    log(f"positive rate (Presence): {y.mean():.6f}", args.log_level, "default")

    output_paths = build_output_paths(args.output_path, run_name, args.n_splits)
    output_dir = output_paths["submission"].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    X = train_df[FEATURES].copy()
    X_test = test_df[FEATURES].copy()

    # Build test pools once per submode to avoid repeated construction in each fold.
    test_pool_by_submode: dict[str, Pool] = {}
    for submode in submode_weights:
        if submode == "cat":
            test_pool_by_submode[submode] = Pool(X_test, cat_features=CAT_FEATURES)
        else:
            test_pool_by_submode[submode] = Pool(X_test)

    model_weight_map: dict[str, float] = {}
    total_models = len(model_configs) * len(seeds) * len(submode_weights)
    for cfg in model_configs:
        for submode, sub_weight in submode_weights.items():
            per_model_weight = sub_weight / len(seeds) / len(model_configs)
            for seed in seeds:
                col = f"pred_{cfg['config_id']}_{submode}_{seed}"
                model_weight_map[col] = per_model_weight

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof_pred = np.zeros(len(train_df), dtype=np.float32)
    test_pred = np.zeros(len(test_df), dtype=np.float32)

    need_model_pred_frames = save_model_oof or args.blend_strategy != "none"
    model_oof_preds: dict[str, np.ndarray] = {}
    model_test_preds: dict[str, np.ndarray] = {}
    if need_model_pred_frames:
        for col in model_weight_map:
            model_oof_preds[col] = np.zeros(len(train_df), dtype=np.float32)
            model_test_preds[col] = np.zeros(len(test_df), dtype=np.float64)

    fold_summary_rows: list[dict[str, Any]] = []
    model_detail_rows: list[dict[str, Any]] = []
    feature_importance_by_mode: dict[str, list[np.ndarray]] = {"cat": [], "numeric": []}
    merged_feature_importances: list[np.ndarray] = []

    log(
        (
            f"\n[CV Training] run_name={run_name}, preset={args.preset}, "
            f"n_splits={args.n_splits}, seeds={seeds}, feature_mode={args.feature_mode}, "
            f"model_family={args.model_family}, ensemble_config={args.ensemble_config}, "
            f"blend_strategy={args.blend_strategy}, hybrid_weight={args.hybrid_weight}"
        ),
        args.log_level,
        "default",
    )

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        log(f"\n--- Fold {fold_idx}/{args.n_splits} ---", args.log_level, "default")
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_valid_fold = y.iloc[valid_idx]

        fold_valid_pred = np.zeros(len(valid_idx), dtype=np.float64)
        fold_test_pred = np.zeros(len(test_df), dtype=np.float64)
        fold_best_iterations: list[int] = []
        fold_train_seconds: list[float] = []

        train_pool_by_submode: dict[str, Pool] = {}
        valid_pool_by_submode: dict[str, Pool] = {}
        for submode in submode_weights:
            if submode == "cat":
                train_pool_by_submode[submode] = Pool(
                    X_train_fold, y_train_fold, cat_features=CAT_FEATURES
                )
                valid_pool_by_submode[submode] = Pool(
                    X_valid_fold, y_valid_fold, cat_features=CAT_FEATURES
                )
            else:
                train_pool_by_submode[submode] = Pool(X_train_fold, y_train_fold)
                valid_pool_by_submode[submode] = Pool(X_valid_fold, y_valid_fold)

        for cfg in model_configs:
            for submode, sub_weight in submode_weights.items():
                per_model_weight = sub_weight / len(seeds) / len(model_configs)
                train_pool = train_pool_by_submode[submode]
                valid_pool = valid_pool_by_submode[submode]
                test_pool = test_pool_by_submode[submode]

                for seed in seeds:
                    col = f"pred_{cfg['config_id']}_{submode}_{seed}"
                    model = build_model(model_cfg=cfg, seed=seed, verbose_eval=args.verbose_eval)

                    t0 = time.perf_counter()
                    model.fit(train_pool, eval_set=valid_pool)
                    elapsed = time.perf_counter() - t0

                    valid_pred = model.predict_proba(valid_pool)[:, 1]
                    test_pred_seed = model.predict_proba(test_pool)[:, 1]
                    best_iteration = model.get_best_iteration()
                    model_auc = roc_auc_score(y_valid_fold, valid_pred)

                    fold_valid_pred += valid_pred * per_model_weight
                    fold_test_pred += test_pred_seed * per_model_weight

                    if need_model_pred_frames:
                        model_oof_preds[col][valid_idx] = valid_pred.astype(np.float32)
                        model_test_preds[col] += test_pred_seed / args.n_splits

                    fi = model.get_feature_importance()
                    feature_importance_by_mode[submode].append(fi)
                    merged_feature_importances.append(fi)

                    best_iteration_int = int(best_iteration) if best_iteration is not None else -1
                    fold_best_iterations.append(best_iteration_int)
                    fold_train_seconds.append(elapsed)
                    model_detail_rows.append(
                        {
                            "fold": fold_idx,
                            "model_name": cfg["model_name"],
                            "config_id": cfg["config_id"],
                            "seed": seed,
                            "submode": submode,
                            "weight": per_model_weight,
                            "auc": float(model_auc),
                            "best_iteration": best_iteration_int,
                            "train_seconds": elapsed,
                        }
                    )
                    log(
                        (
                            f"config={cfg['config_id']}, seed={seed}, submode={submode}, "
                            f"weight={per_model_weight:.5f}, auc={model_auc:.6f}, "
                            f"best_iteration={best_iteration_int}, train_seconds={elapsed:.3f}"
                        ),
                        args.log_level,
                        "debug",
                    )

        fold_auc = roc_auc_score(y_valid_fold, fold_valid_pred)
        fold_best_np = np.asarray(fold_best_iterations, dtype=np.float64)
        fold_time_np = np.asarray(fold_train_seconds, dtype=np.float64)
        oof_pred[valid_idx] = fold_valid_pred.astype(np.float32)
        test_pred += (fold_test_pred / args.n_splits).astype(np.float32)

        fold_summary_rows.append(
            {
                "fold": fold_idx,
                "scope": "fold_ensemble",
                "model_name": "ensemble",
                "config_id": "ensemble",
                "feature_mode": args.feature_mode,
                "model_family": args.model_family,
                "ensemble_config": args.ensemble_config,
                "seeds": ",".join(str(s) for s in seeds),
                "auc": float(fold_auc),
                "best_iteration_mean": float(fold_best_np.mean()),
                "best_iteration_median": float(np.median(fold_best_np)),
                "best_iteration_min": int(fold_best_np.min()),
                "best_iteration_max": int(fold_best_np.max()),
                "train_seconds": np.nan,
                "train_seconds_mean": np.nan,
                "train_seconds_total": np.nan,
                "n_models": int(total_models),
            }
        )
        log(f"fold_ensemble_auc={fold_auc:.6f}", args.log_level, "default")

    oof_auc = roc_auc_score(y, oof_pred.astype(np.float64))
    fold_auc_values = np.array([row["auc"] for row in fold_summary_rows], dtype=np.float64)
    fold_auc_mean = float(fold_auc_values.mean())
    fold_auc_std = float(fold_auc_values.std())

    log("\n[CV Result]", args.log_level, "minimal")
    log(f"OOF AUC: {oof_auc:.6f}", args.log_level, "minimal")
    log(f"Fold AUC mean: {fold_auc_mean:.6f}", args.log_level, "minimal")
    log(f"Fold AUC std: {fold_auc_std:.6f}", args.log_level, "minimal")

    pred_df = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: test_pred.astype(np.float32)})
    submission_df = sample_sub_df[[ID_COL]].merge(pred_df, on=ID_COL, how="left")
    if submission_df[TARGET_COL].isna().any():
        raise ValueError("NaN found in submission predictions after id merge.")
    if (
        float(submission_df[TARGET_COL].min()) < 0.0
        or float(submission_df[TARGET_COL].max()) > 1.0
    ):
        raise ValueError("Submission probabilities are out of [0, 1] range.")

    fold_summary_df = pd.DataFrame(fold_summary_rows)
    summary_rows = pd.DataFrame(
        [
            {
                "fold": "mean",
                "scope": "cv_summary",
                "model_name": "ensemble",
                "config_id": "ensemble",
                "feature_mode": args.feature_mode,
                "model_family": args.model_family,
                "ensemble_config": args.ensemble_config,
                "seeds": ",".join(str(s) for s in seeds),
                "auc": fold_auc_mean,
                "best_iteration_mean": np.nan,
                "best_iteration_median": np.nan,
                "best_iteration_min": np.nan,
                "best_iteration_max": np.nan,
                "train_seconds": np.nan,
                "train_seconds_mean": np.nan,
                "train_seconds_total": np.nan,
                "n_models": total_models,
            },
            {
                "fold": "std",
                "scope": "cv_summary",
                "model_name": "ensemble",
                "config_id": "ensemble",
                "feature_mode": args.feature_mode,
                "model_family": args.model_family,
                "ensemble_config": args.ensemble_config,
                "seeds": ",".join(str(s) for s in seeds),
                "auc": fold_auc_std,
                "best_iteration_mean": np.nan,
                "best_iteration_median": np.nan,
                "best_iteration_min": np.nan,
                "best_iteration_max": np.nan,
                "train_seconds": np.nan,
                "train_seconds_mean": np.nan,
                "train_seconds_total": np.nan,
                "n_models": total_models,
            },
            {
                "fold": "oof",
                "scope": "cv_summary",
                "model_name": "ensemble",
                "config_id": "ensemble",
                "feature_mode": args.feature_mode,
                "model_family": args.model_family,
                "ensemble_config": args.ensemble_config,
                "seeds": ",".join(str(s) for s in seeds),
                "auc": float(oof_auc),
                "best_iteration_mean": np.nan,
                "best_iteration_median": np.nan,
                "best_iteration_min": np.nan,
                "best_iteration_max": np.nan,
                "train_seconds": np.nan,
                "train_seconds_mean": np.nan,
                "train_seconds_total": np.nan,
                "n_models": total_models,
            },
        ]
    )
    fold_summary_df = pd.concat([fold_summary_df, summary_rows], ignore_index=True)
    model_detail_df = pd.DataFrame(model_detail_rows)

    fi_merged_df = feature_importance_df(
        importances=merged_feature_importances, n_models=len(merged_feature_importances)
    )
    fi_cat_df: pd.DataFrame | None = None
    fi_numeric_df: pd.DataFrame | None = None
    if feature_importance_by_mode["cat"]:
        fi_cat_df = feature_importance_df(
            importances=feature_importance_by_mode["cat"],
            n_models=len(feature_importance_by_mode["cat"]),
        )
    if feature_importance_by_mode["numeric"]:
        fi_numeric_df = feature_importance_df(
            importances=feature_importance_by_mode["numeric"],
            n_models=len(feature_importance_by_mode["numeric"]),
        )

    submission_df.to_csv(output_paths["submission"], index=False)
    if save_oof:
        oof_df = pd.DataFrame({ID_COL: train_df[ID_COL], "target": y, "oof_pred": oof_pred})
        oof_df.to_csv(output_paths["oof"], index=False)

    fold_summary_df.to_csv(output_paths["fold_summary"], index=False)
    if save_model_detail:
        model_detail_df.to_csv(output_paths["model_detail"], index=False)

    fi_merged_df.to_csv(output_paths["feature_importance_merged"], index=False)
    if fi_cat_df is not None:
        fi_cat_df.to_csv(output_paths["feature_importance_cat"], index=False)
    if fi_numeric_df is not None:
        fi_numeric_df.to_csv(output_paths["feature_importance_numeric"], index=False)

    model_oof_df: pd.DataFrame | None = None
    model_test_df: pd.DataFrame | None = None
    if need_model_pred_frames:
        pred_cols = sorted(model_oof_preds.keys())
        model_oof_df = pd.DataFrame({ID_COL: train_df[ID_COL], "target": y})
        for col in pred_cols:
            model_oof_df[col] = model_oof_preds[col]

        model_test_df = pd.DataFrame({ID_COL: test_df[ID_COL]})
        for col in pred_cols:
            model_test_df[col] = model_test_preds[col].astype(np.float32)

        if save_model_oof:
            model_oof_df.to_csv(output_paths["model_oof"], index=False)
            model_test_df.to_csv(output_paths["model_test_pred"], index=False)

    blend_payload: dict[str, Any]
    if args.blend_strategy == "none":
        blend_payload = save_base_blend_artifacts(
            output_paths=output_paths,
            submission_df=submission_df,
            model_weight_map=model_weight_map,
            oof_auc=oof_auc,
        )
    else:
        if model_oof_df is None or model_test_df is None:
            raise RuntimeError("Blending requires model OOF/test prediction frames.")
        if not save_model_oof:
            raise RuntimeError("--blend-strategy requires persisted model prediction files.")
        blend_payload = run_blending_script(
            output_paths=output_paths,
            blend_strategy=args.blend_strategy,
            log_level=args.log_level,
        )

    public_score_manual = args.public_score_manual
    oof_public_gap = None
    if public_score_manual is not None:
        oof_public_gap = float(public_score_manual - oof_auc)

    metrics_payload: dict[str, Any] = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "data": {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "sample_submission_path": str(sample_sub_path),
            "train_shape": list(train_df.shape),
            "test_shape": list(test_df.shape),
        },
        "target_distribution": {str(k): int(v) for k, v in target_counts.items()},
        "model": {
            "preset": args.preset,
            "feature_mode": args.feature_mode,
            "model_family": args.model_family,
            "ensemble_config": args.ensemble_config,
            "hybrid_weight": args.hybrid_weight,
            "feature_columns": FEATURES,
            "cat_features": CAT_FEATURES,
            "seeds": seeds,
            "iterations": int(hp["iterations"]),
            "od_wait": int(hp["od_wait"]),
            "model_configs": model_configs,
            "n_splits": args.n_splits,
            "cv_seed": args.seed,
            "log_level": args.log_level,
        },
        "blend": {
            "strategy": args.blend_strategy,
            "selected_strategy": blend_payload.get("selected_strategy"),
            "selected_auc": blend_payload.get("selected_auc"),
        },
        "save_options": {
            "save_oof": save_oof,
            "save_model_detail": save_model_detail,
            "save_model_oof": save_model_oof,
        },
        "public_score_manual": public_score_manual,
        "oof_public_gap": oof_public_gap,
        "metrics": {
            "oof_auc": float(oof_auc),
            "fold_auc_mean": fold_auc_mean,
            "fold_auc_std": fold_auc_std,
            "fold_ensemble_rows": fold_summary_rows,
        },
        "artifacts": {
            "submission": str(output_paths["submission"]),
            "submission_blend": str(output_paths["submission_blend"]),
            "weights": str(output_paths["weights"]),
            "oof": str(output_paths["oof"]) if save_oof else None,
            "fold_summary": str(output_paths["fold_summary"]),
            "model_detail": str(output_paths["model_detail"]) if save_model_detail else None,
            "model_oof": str(output_paths["model_oof"]) if save_model_oof else None,
            "model_test_pred": str(output_paths["model_test_pred"]) if save_model_oof else None,
            "feature_importance_merged": str(output_paths["feature_importance_merged"]),
            "feature_importance_cat": (
                str(output_paths["feature_importance_cat"]) if fi_cat_df is not None else None
            ),
            "feature_importance_numeric": (
                str(output_paths["feature_importance_numeric"])
                if fi_numeric_df is not None
                else None
            ),
            "metrics": str(output_paths["metrics"]),
        },
    }

    with open(output_paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    log("\n[Top Feature Importance - Merged]", args.log_level, "default")
    log(fi_merged_df.head(10).to_string(index=False), args.log_level, "default")

    log("\n[Saved Artifacts]", args.log_level, "minimal")
    for key in [
        "submission",
        "submission_blend",
        "weights",
        "fold_summary",
        "metrics",
    ]:
        log(f"{key}: {output_paths[key]}", args.log_level, "minimal")
    if save_oof:
        log(f"oof: {output_paths['oof']}", args.log_level, "minimal")
    if save_model_detail:
        log(f"model detail: {output_paths['model_detail']}", args.log_level, "minimal")
    if save_model_oof:
        log(f"model oof: {output_paths['model_oof']}", args.log_level, "minimal")
        log(f"model test pred: {output_paths['model_test_pred']}", args.log_level, "minimal")


if __name__ == "__main__":
    main()
