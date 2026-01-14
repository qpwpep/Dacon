import os
import random
import json
import re
from datetime import datetime
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

import lightgbm as lgb

import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def normalize_blank_strings_df(df: pd.DataFrame) -> pd.DataFrame:
    """공백/빈 문자열을 결측(np.nan)으로 통일.
    - object/string 컬럼에만 적용
    - '   ' 같이 공백만 있는 값도 결측으로 변환
    """
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        s = s.astype("object")
        mask = s.notna()
        if mask.any():
            stripped = s.loc[mask].astype(str).str.strip()
            stripped = stripped.mask(stripped.eq(""), np.nan)
            df.loc[mask, col] = stripped
        else:
            df[col] = s
    return df


def normalize_blank_series(s: pd.Series) -> pd.Series:
    """Series에서 공백/빈 문자열을 결측(np.nan)으로 통일 (category 변환 직전 사용)."""
    s = s.astype("object").copy()
    mask = s.notna()
    if mask.any():
        stripped = s.loc[mask].astype(str).str.strip()
        stripped = stripped.mask(stripped.eq(""), np.nan)
        s.loc[mask] = stripped
    return s


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """대회 컬럼 구성에 맞춘 최소 파생피처."""
    df = df.copy()

    # major_field 기반 파생피처 (존재할 때만)
    if "major_field" in df.columns:
        s = df["major_field"].astype("object")
        s = normalize_blank_series(s)  # 공백/빈 문자열 -> NaN
        df["major_field"] = s
        df["is_major_it"] = s.fillna("").astype(str).str.contains("IT", regex=False).astype(int)

    return df


def drop_high_missing_cols(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """train 기준 결측률이 threshold 초과인 컬럼 드랍."""
    miss = train_df.isna().mean()
    drop_cols = miss[miss > threshold].index.tolist()
    return train_df.drop(columns=drop_cols), test_df.drop(columns=drop_cols, errors="ignore"), drop_cols


def drop_constant_cols(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """train 기준 nunique<=1 인 상수 컬럼 드랍."""
    nunique = train_df.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    return train_df.drop(columns=const_cols), test_df.drop(columns=const_cols, errors="ignore"), const_cols


def infer_numeric_categorical_cols(
    train_df: pd.DataFrame,
    max_unique: int,
    include_bool: bool,
    exclude: List[str],
) -> List[str]:
    """저유니크 numeric/bool 컬럼을 category로 취급하기 위한 후보 리스트."""
    cols: List[str] = []
    for col in train_df.columns:
        if col in exclude:
            continue
        s = train_df[col]
        if pd.api.types.is_bool_dtype(s):
            if include_bool:
                cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(s):
            # NaN 제외한 유니크 수로 판단
            nuniq = s.nunique(dropna=True)
            if nuniq <= max_unique:
                cols.append(col)
    return cols


def make_categorical_safe(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, List[str]]]:
    """
    지정된 컬럼을 pandas category로 변환.
    - train에서 본 카테고리 + __UNKNOWN__ + __MISSING__ 만 허용
    - test에서 train에 없던 값은 __UNKNOWN__으로 매핑
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    categories_map: Dict[str, List[str]] = {}
    cat_cols: List[str] = []

    for col in categorical_cols:
        if col not in train_df.columns:
            continue

        tr = normalize_blank_series(train_df[col])
        tr = tr.where(tr.notna(), "__MISSING__").astype(str)
        cats = pd.Index(tr.unique()).append(pd.Index(["__UNKNOWN__", "__MISSING__"]))
        cats = cats.drop_duplicates()

        train_df[col] = pd.Categorical(tr, categories=cats)

        if col in test_df.columns:
            te = normalize_blank_series(test_df[col])
            te = te.where(te.notna(), "__MISSING__").astype(str)
            te = te.where(te.isin(cats), "__UNKNOWN__")
            test_df[col] = pd.Categorical(te, categories=cats)
        else:
            # test에 없으면, 이후 reindex에서 자동으로 채워지도록 둔다
            pass

        categories_map[col] = [str(x) for x in list(cats)]
        cat_cols.append(col)

    return train_df, test_df, cat_cols, categories_map


def maybe_init_wandb(cfg: DictConfig):
    if not cfg.wandb.enable:
        return None

    try:
        import wandb

        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
            notes=cfg.wandb.notes,
            group=cfg.wandb.group,
            job_type=cfg.wandb.job_type,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        return run
    except Exception as e:
        print(f"[Warn] wandb init failed -> continue without wandb. err={e}")
        return None


def wandb_log(run, payload: Dict[str, Any], step: Optional[int] = None) -> None:
    if run is None:
        return
    import wandb

    wandb.log(payload, step=step)



def save_yaml_in_output_dir(filename: str, payload: Dict[str, Any]) -> Path:
    """Save payload as YAML into Hydra output dir (e.g., outputs/YYYY-MM-DD/HH-MM-SS)."""
    try:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        # Fallback: current working directory (Hydra usually chdir's into output dir)
        output_dir = Path.cwd()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    cfg_obj = OmegaConf.create(payload)
    OmegaConf.save(config=cfg_obj, f=str(out_path))
    return out_path


def save_json_in_output_dir(filename: str, obj: Any) -> Path:
    """Save JSON under Hydra output_dir (or cwd fallback)."""
    try:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return out_path



def _safe_name(s: Any, max_len: int = 80) -> str:
    s = "" if s is None else str(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    if not s:
        s = "NA"
    return s[:max_len]

def make_wandb_artifact_name(cfg: DictConfig, run, suffix: str) -> str:
    """Build a reproducible artifact name that encodes project/run_name (and run id).
    Note: Avoid slashes because W&B artifact names may reject them.
    """
    proj = _safe_name(getattr(cfg.wandb, "project", "project"))
    if run is None:
        rname = "local"
        rid = "local"
    else:
        rname = _safe_name(getattr(run, "name", "run"))
        rid = _safe_name(getattr(run, "id", "id"))
    suf = _safe_name(suffix, max_len=60)
    return f"{proj}-{rname}-{rid}-{suf}"


def get_git_info(repo_dir: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir)).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo_dir)).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_dir)).decode()
        info.update({"commit": commit, "branch": branch, "is_dirty": bool(status.strip())})
    except Exception as e:
        info.update({"commit": None, "branch": None, "is_dirty": None, "error": str(e)})
    return info


def collect_run_info(cfg: DictConfig, run) -> Dict[str, Any]:
    try:
        repo_dir = Path(HydraConfig.get().runtime.cwd)
    except Exception:
        repo_dir = Path.cwd()

    wandb_info = {
        "project": str(getattr(cfg.wandb, "project", "")),
        "run_name": str(getattr(run, "name", "")) if run is not None else "local",
        "run_id": str(getattr(run, "id", "")) if run is not None else "local",
    }

    info: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "seed": int(getattr(cfg.train, "seed", 0)),
        "wandb": wandb_info,
        "python": {
            "version": sys.version.split()[0],
            "full": sys.version,
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "lightgbm_version": getattr(lgb, "__version__", None),
        "git": get_git_info(repo_dir),
    }
    return info


def make_repro_bundle_path(model_path: Path, cfg: DictConfig, run, kind: str, ts: Optional[str] = None) -> Path:
    """Unified naming for single/fold reproducible bundles."""
    run_id = _safe_name(getattr(run, "id", "local")) if run is not None else "local"
    ts = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    proj = _safe_name(getattr(cfg.wandb, "project", "project"), max_len=30)
    rname = _safe_name(getattr(run, "name", "run") if run is not None else "local", max_len=30)
    kind = _safe_name(kind, max_len=15)
    return model_path.with_name(f"{model_path.stem}_repro_bundle_{kind}_{proj}_{rname}_{run_id}_{ts}.zip")


def wandb_log_artifact(run, path: Path, name: str, art_type: str) -> None:
    if run is None:
        return
    import wandb

    art = wandb.Artifact(name=name, type=art_type)
    art.add_file(str(path))
    run.log_artifact(art)


def search_best_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    min_t: float = 0.05,
    max_t: float = 0.95,
) -> Tuple[float, float]:
    """OOF 확률(proba)에 대해 F1이 최대가 되는 threshold를 '정확하게' 탐색 + clamp.

    - grid loop(0.01 step) 대신, precision_recall_curve가 만들어주는 임계값 후보(=고유 score 경계)에서
      F1을 계산해 최댓값을 선택합니다. (pred = proba >= threshold 기준)
    - 선택된 threshold는 [min_t, max_t] 범위로 제한합니다.
      (범위 밖 후보는 아예 제외한 뒤 최적을 찾고, 범위 내 후보가 하나도 없으면 0.5를 clamp해서 사용)
    """
    # 방어적 캐스팅
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)

    if min_t > max_t:
        min_t, max_t = max_t, min_t

    # safety: nan proba는 아주 작은 값으로 처리 (있으면 비정상 케이스)
    if np.isnan(proba).any():
        proba = np.nan_to_num(proba, nan=-1.0)

    precision, recall, thresholds = precision_recall_curve(y_true, proba)

    # thresholds가 비는 경우: proba가 모두 동일하거나 데이터가 극단 케이스
    if thresholds.size == 0:
        t = float(np.clip(0.5, min_t, max_t))
        return t, float(f1_score(y_true, (proba >= t).astype(int)))

    # precision/recall은 thresholds보다 1개 길다 -> 마지막 원소(임계값 없음)는 제외
    denom = (precision[:-1] + recall[:-1] + 1e-15)
    f1s = (2.0 * precision[:-1] * recall[:-1]) / denom

    # ✅ 후보를 범위로 제한 (정확 최적 + 범위 보장)
    mask = (thresholds >= min_t) & (thresholds <= max_t)
    if np.any(mask):
        cand_t = thresholds[mask]
        cand_f1 = f1s[mask]
        best_idx = int(np.nanargmax(cand_f1))
        best_t = float(cand_t[best_idx])
        best_f1 = float(cand_f1[best_idx])
        return best_t, best_f1

    # 범위 내 threshold 후보가 하나도 없으면, 0.5를 clamp해서 사용
    t = float(np.clip(0.5, min_t, max_t))
    return t, float(f1_score(y_true, (proba >= t).astype(int)))


def _select(cfg, key: str, default=None):
    """OmegaConf.select wrapper that works even if the key doesn't exist."""
    try:
        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        return default


def run_optuna_tuning(
    cfg: DictConfig,
    X: pd.DataFrame,
    y: pd.Series,
    cat_cols: List[str],
    run=None,
) -> Dict[str, Any]:
    """Tune a few LightGBM hyperparameters with Optuna and return best overrides.

    Tuning params:
      - num_leaves
      - min_child_samples
      - reg_lambda
      - feature_fraction (mapped to colsample_bytree for LGBMClassifier)

    Metric:
      - "auc" (default) : mean AUC over folds
      - "f1"            : OOF F1 at best threshold (grid-search threshold on OOF)
    """
    enable = bool(_select(cfg, "train.optuna.enable", False))
    if not enable:
        return {}

    try:
        import optuna
    except ImportError as e:
        raise ImportError("Optuna is not installed. Run: pip install optuna") from e

    optuna_cfg = _select(cfg, "train.optuna", default={})
    n_trials = int(getattr(optuna_cfg, "n_trials", 50))
    timeout = getattr(optuna_cfg, "timeout", None)
    timeout = int(timeout) if timeout is not None else None

    metric = str(getattr(optuna_cfg, "metric", "auc")).lower()
    if metric not in ("auc", "f1"):
        raise ValueError(f"Unsupported optuna metric: {metric} (use 'auc' or 'f1')")

    n_splits = int(getattr(optuna_cfg, "n_splits", int(cfg.train.n_splits)))

    sampler_name = str(getattr(optuna_cfg, "sampler", "tpe")).lower()
    seed = int(getattr(optuna_cfg, "seed", int(cfg.train.seed)))
    if sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)

    use_pruner = bool(getattr(optuna_cfg, "use_pruner", True))
    pruner_cfg = getattr(optuna_cfg, "pruner", None)
    pruner_name = str(getattr(pruner_cfg, "name", "median")).lower() if pruner_cfg is not None else "median"
    n_warmup_steps = int(getattr(pruner_cfg, "n_warmup_steps", 10)) if pruner_cfg is not None else 10

    if not use_pruner or pruner_name == "nop":
        pruner = optuna.pruners.NopPruner()
    elif pruner_name == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        # default: median pruner
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)

    study_name = str(getattr(optuna_cfg, "study_name", "lgbm_tuning"))
    storage = getattr(optuna_cfg, "storage", None)  # e.g. "sqlite:///optuna.db"
    direction = "maximize"

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True if storage else False,
    )

    base_params = dict(OmegaConf.to_container(cfg.model, resolve=True))

    # Always keep deterministic behavior per trial (except for bagging)
    base_params["random_state"] = seed

    def objective(trial: "optuna.Trial") -> float:
        params = dict(base_params)


        # Search spaces (from YAML if provided; otherwise fall back to safe defaults)
        ss = getattr(optuna_cfg, "search_space", None)

        def _ss_int(name: str, low: int, high: int, log: bool = False) -> int:
            if ss is None or not hasattr(ss, name):
                return int(trial.suggest_int(name, low, high, log=log))
            cfg_item = getattr(ss, name)
            _low = int(getattr(cfg_item, "low", low))
            _high = int(getattr(cfg_item, "high", high))
            _log = bool(getattr(cfg_item, "log", log))
            return int(trial.suggest_int(name, _low, _high, log=_log))

        def _ss_float(name: str, low: float, high: float, log: bool = False) -> float:
            if ss is None or not hasattr(ss, name):
                return float(trial.suggest_float(name, low, high, log=log))
            cfg_item = getattr(ss, name)
            _low = float(getattr(cfg_item, "low", low))
            _high = float(getattr(cfg_item, "high", high))
            _log = bool(getattr(cfg_item, "log", log))
            return float(trial.suggest_float(name, _low, _high, log=_log))

        num_leaves = _ss_int("num_leaves", 16, 256, log=True)
        min_child_samples = _ss_int("min_child_samples", 10, 200, log=True)
        reg_lambda = _ss_float("reg_lambda", 1e-3, 100.0, log=True)
        feature_fraction = _ss_float("feature_fraction", 0.5, 1.0, log=False)

        params["num_leaves"] = int(num_leaves)
        params["min_child_samples"] = int(min_child_samples)
        params["reg_lambda"] = float(reg_lambda)
        # sklearn wrapper uses colsample_bytree; map "feature_fraction" -> "colsample_bytree"
        params["colsample_bytree"] = float(feature_fraction)

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )

        aucs: List[float] = []
        oof = np.zeros(len(X), dtype=float)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=int(cfg.train.early_stopping_rounds), verbose=False),
                ],
                categorical_feature=cat_cols if len(cat_cols) > 0 else "auto",
            )

            proba_va = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
            oof[va_idx] = proba_va
            auc = roc_auc_score(y_va, proba_va)
            aucs.append(float(auc))

            # pruning on AUC is stable even if final metric is f1
            trial.report(float(np.mean(aucs)), step=fold)
            if use_pruner and trial.should_prune():
                raise optuna.TrialPruned()

        auc_mean = float(np.mean(aucs))

        if metric == "auc":
            return auc_mean

        # metric == "f1": tune threshold on OOF proba each trial
        if bool(cfg.train.optimize_threshold.enable):
            t, f1_best = search_best_threshold(
                y_true=y.values,
                proba=oof,
                min_t=float(cfg.train.optimize_threshold.clamp_min),
                max_t=float(cfg.train.optimize_threshold.clamp_max)
            )
            trial.set_user_attr("best_threshold", float(t))
            return float(f1_best)

        # fallback: use fixed threshold
        thr = float(cfg.train.threshold)
        trial.set_user_attr("best_threshold", float(thr))
        return float(f1_score(y.values, (oof >= thr).astype(int)))

    print(f"\n[Optuna] Start tuning: n_trials={n_trials} | metric={metric} | folds={n_splits}")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best = study.best_trial
    best_params = dict(best.params)

    # rename for config override usage
    overrides = {
        "num_leaves": int(best_params["num_leaves"]),
        "min_child_samples": int(best_params["min_child_samples"]),
        "reg_lambda": float(best_params["reg_lambda"]),
        "colsample_bytree": float(best_params["feature_fraction"]),  # maps feature_fraction -> colsample_bytree
    }

    # Persist best trial params into outputs/.../best_params.yaml
    best_threshold = None
    try:
        best_threshold = best.user_attrs.get("best_threshold", None)
    except Exception:
        best_threshold = None

    best_payload = {
        "optuna": {
            "study_name": study.study_name,
            "metric": str(metric),
            "best_value": float(best.value),
            "best_trial_number": int(best.number),
            "best_threshold": float(best_threshold) if best_threshold is not None else None,
            "raw_params": dict(best.params),     # as suggested by Optuna
            "model_overrides": dict(overrides),  # ready to merge into cfg.model
        }
    }
    saved_path = save_yaml_in_output_dir("best_params.yaml", best_payload)
    print(f"[Optuna] Saved best params -> {saved_path}")

    print("\n[Optuna] Best trial")
    print("  value:", best.value)
    print("  params:", best.params)

    if run is not None:
        # single log entry for best
        wandb_log(run, {
            "optuna/best_value": float(best.value),
            "optuna/best_num_leaves": overrides["num_leaves"],
            "optuna/best_min_child_samples": overrides["min_child_samples"],
            "optuna/best_reg_lambda": overrides["reg_lambda"],
            "optuna/best_feature_fraction": overrides["colsample_bytree"],
        })

    return overrides



def cv_train(
    cfg: DictConfig,
    X: pd.DataFrame,
    y: pd.Series,
    test: pd.DataFrame,
    cat_cols: List[str],
    run,
    save_fold_models: bool = False,
    fold_model_basepath: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[float], List[float], List[Path], np.ndarray]:
    params = dict(OmegaConf.to_container(cfg.model, resolve=True))

    skf = StratifiedKFold(
        n_splits=int(cfg.train.n_splits),
        shuffle=True,
        random_state=int(cfg.train.seed),
    )

    oof_proba = np.zeros(len(X), dtype=float)
    test_proba_sum = np.zeros(len(test), dtype=float)  # fold별 test proba 누적
    feat_imp_sum = np.zeros(X.shape[1], dtype=float)   # fold별 feature importance 누적(평균용)

    best_iters: List[int] = []
    aucs: List[float] = []
    f1s: List[float] = []
    fold_model_paths: List[Path] = []

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
                lgb.early_stopping(stopping_rounds=int(cfg.train.early_stopping_rounds), verbose=False),
                lgb.log_evaluation(period=int(cfg.train.log_period)),
            ],
            categorical_feature=cat_cols if len(cat_cols) > 0 else "auto",
        )

        # OOF
        proba_va = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
        oof_proba[va_idx] = proba_va

        # Fold test proba 누적 (앙상블용)
        proba_te = model.predict_proba(test, num_iteration=model.best_iteration_)[:, 1]
        test_proba_sum += proba_te

        # Feature importance 누적(평균)
        feat_imp_sum += model.feature_importances_

        # Metrics
        auc = roc_auc_score(y_va, proba_va)
        pred_va = (proba_va >= float(cfg.train.threshold)).astype(int)
        f1 = f1_score(y_va, pred_va)

        aucs.append(float(auc))
        f1s.append(float(f1))
        best_iters.append(int(model.best_iteration_))

        print(f"[Fold {fold}] best_iter={model.best_iteration_} | AUC={auc:.5f} | F1@{cfg.train.threshold}={f1:.5f}")
        wandb_log(run, {"fold": fold, "cv/auc": auc, "cv/f1": f1, "cv/best_iter": model.best_iteration_}, step=fold)

        # (옵션) fold 모델 저장
        if save_fold_models and fold_model_basepath is not None:
            base = fold_model_basepath
            suffix = base.suffix if base.suffix else ".txt"
            fold_path = base.with_name(f"{base.stem}_fold{fold}{suffix}")
            fold_path.parent.mkdir(parents=True, exist_ok=True)
            model.booster_.save_model(str(fold_path))
            fold_model_paths.append(fold_path)

    n_splits = int(cfg.train.n_splits)
    test_proba_mean = test_proba_sum / n_splits
    feat_imp_mean = feat_imp_sum / n_splits

    return oof_proba, test_proba_mean, best_iters, aucs, f1s, fold_model_paths, feat_imp_mean



def fit_final_and_predict(
    cfg: DictConfig,
    X: pd.DataFrame,
    y: pd.Series,
    test: pd.DataFrame,
    cat_cols: List[str],
    best_iter_final: int,
) -> Tuple[lgb.LGBMClassifier, np.ndarray]:
    final_params = dict(OmegaConf.to_container(cfg.model, resolve=True))
    final_params["n_estimators"] = int(best_iter_final)
    model = lgb.LGBMClassifier(**final_params)
    model.fit(X, y, categorical_feature=cat_cols if len(cat_cols) > 0 else "auto")
    test_proba = model.predict_proba(test)[:, 1]
    return model, test_proba


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("[Config]\n" + OmegaConf.to_yaml(cfg))

    seed_everything(int(cfg.train.seed))
    run = maybe_init_wandb(cfg)

    # Save resolved config for reproducibility (included in fold model bundle under meta/)
    try:
        cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
        save_yaml_in_output_dir("config_resolved.yaml", cfg_resolved)
    except Exception as e:
        print(f"[Warn] saving resolved config failed: {e}")

    # Save run info for reproducibility (git/python/lightgbm/seed, etc.)
    try:
        run_info = collect_run_info(cfg, run)
        save_yaml_in_output_dir("run_info.yaml", run_info)
    except Exception as e:
        print(f"[Warn] saving run info failed: {e}")



    # Resolve paths (Hydra changes cwd -> use to_absolute_path for inputs)
    train_path = Path(to_absolute_path(cfg.data.train_path))
    test_path = Path(to_absolute_path(cfg.data.test_path))
    sample_submission_path = Path(to_absolute_path(cfg.data.sample_submission_path))

    if not train_path.exists():
        raise FileNotFoundError(f"train_path not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"test_path not found: {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # 공백/빈 문자열을 결측으로 통일 (문자열 컬럼)
    train = normalize_blank_strings_df(train)
    test = normalize_blank_strings_df(test)

    if cfg.data.target_col not in train.columns:
        raise ValueError(f"train에 target 컬럼({cfg.data.target_col})이 없습니다.")

    y = train[cfg.data.target_col].astype(int)
    X = train.drop(columns=[cfg.data.target_col])

    # ID 제거(있으면)
    if cfg.data.id_col in X.columns:
        X = X.drop(columns=[cfg.data.id_col])
    if cfg.data.id_col in test.columns:
        test = test.drop(columns=[cfg.data.id_col])

    # 파생피처
    X = build_features(X)
    test = build_features(test)

    # 결측률 높은 컬럼 드랍(train 기준)
    X, test, dropped_missing = drop_high_missing_cols(X, test, threshold=float(cfg.data.drop_missing_threshold))

    # 상수 컬럼 드랍(train 기준)
    dropped_const: List[str] = []
    if bool(cfg.data.drop_constant_cols):
        X, test, dropped_const = drop_constant_cols(X, test)

    # 어떤 컬럼을 category로 볼지 결정
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    if bool(cfg.data.categorical_numeric.enable):
        extra = infer_numeric_categorical_cols(
            train_df=X,
            max_unique=int(cfg.data.categorical_numeric.max_unique),
            include_bool=bool(cfg.data.categorical_numeric.include_bool),
            exclude=list(cfg.data.categorical_numeric.exclude),
        )
        # object + numeric-cat 합치기
        cat_cols = sorted(set(cat_cols).union(set(extra)))

    X, test, cat_cols, categories_map = make_categorical_safe(X, test, categorical_cols=cat_cols)

    # train/test 컬럼 정렬 및 누락 컬럼 처리
    test = test.reindex(columns=X.columns, fill_value=np.nan)


    # Save preprocessing meta for reproducibility (included in fold model bundle under meta/)
    try:
        feature_columns = list(X.columns)
        save_json_in_output_dir("feature_columns.json", feature_columns)

        # Summarize category map (avoid huge artifact)
        cat_summary: Dict[str, Dict[str, Any]] = {}
        for col, cats in categories_map.items():
            cats_list = list(cats)
            cat_summary[col] = {
                "n_categories": int(len(cats_list)),
                "sample_categories": cats_list[:30],
            }

        preprocess_payload = {
            "dropped_missing_cols": list(dropped_missing),
            "dropped_constant_cols": list(dropped_const),
            "categorical_cols": list(cat_cols),
            "n_features": int(X.shape[1]),
            "categories_map_summary": cat_summary,
        }
        save_yaml_in_output_dir("preprocess_meta.yaml", preprocess_payload)
    except Exception as e:
        print(f"[Warn] saving preprocess meta failed: {e}")

    # Dataset summary
    print(f"[Info] Train rows={len(train)} | Test rows={len(test)} | Features={X.shape[1]}")
    print(f"[Info] Pos rate={y.mean():.4f} ({int(y.sum())}/{len(y)})")
    print(f"[Info] Dropped missing>{cfg.data.drop_missing_threshold}: {len(dropped_missing)} cols -> {dropped_missing}")
    if dropped_const:
        print(f"[Info] Dropped constant cols: {len(dropped_const)} cols -> {dropped_const}")
    print(f"[Info] Categorical cols: {len(cat_cols)}")

    if run is not None:
        # Put some key info into run summary
        run.summary["data/train_rows"] = int(len(train))
        run.summary["data/test_rows"] = int(len(test))
        run.summary["data/pos_rate"] = float(y.mean())
        run.summary["data/n_features"] = int(X.shape[1])
        run.summary["data/dropped_missing_cols"] = len(dropped_missing)
        run.summary["data/dropped_constant_cols"] = len(dropped_const)
        run.summary["data/n_categorical_cols"] = len(cat_cols)


    # Optuna tuning (optional): updates cfg.model with best params before the main CV
    optuna_overrides = run_optuna_tuning(cfg, X, y, cat_cols, run)
    if optuna_overrides:
        for k, v in optuna_overrides.items():
            cfg.model[k] = v
        print(f"[Optuna] Applied best overrides to cfg.model: {optuna_overrides}")

    # CV (+ fold-ensemble prediction)
    use_fold_ensemble = bool(getattr(cfg.train, "use_fold_ensemble", True))
    save_fold_models = bool(cfg.train.save_model) and use_fold_ensemble
    fold_model_basepath = Path(cfg.train.model_path) if save_fold_models else None

    oof_proba, test_proba_cv, best_iters, aucs, f1s, fold_model_paths, cv_feat_imp = cv_train(
        cfg, X, y, test, cat_cols, run,
        save_fold_models=save_fold_models,
        fold_model_basepath=fold_model_basepath,
    )

    auc_mean, auc_std = float(np.mean(aucs)), float(np.std(aucs))
    f1_mean, f1_std = float(np.mean(f1s)), float(np.std(f1s))
    best_iter_final = int(np.median(best_iters))

    print("\n[CV] Done")
    print(f"AUC mean={auc_mean:.5f} ± {auc_std:.5f}")
    print(f"F1  mean={f1_mean:.5f} ± {f1_std:.5f}")
    print(f"Use n_estimators={best_iter_final} (median best_iteration) for final fit")

    # Optional: OOF-based threshold tuning
    threshold = float(cfg.train.threshold)
    best_f1 = float(f1_score(y.values, (oof_proba >= threshold).astype(int)))
    if bool(cfg.train.optimize_threshold.enable):
        t, f1_best = search_best_threshold(
            y_true=y.values,
            proba=oof_proba,
            min_t=float(cfg.train.optimize_threshold.clamp_min),
            max_t=float(cfg.train.optimize_threshold.clamp_max)
        )
        threshold = t
        best_f1 = f1_best
        print(f"[OOF] Best threshold={threshold:.3f} | F1={best_f1:.5f}")

    # Log aggregate metrics
    wandb_log(run, {
        "cv/auc_mean": auc_mean,
        "cv/auc_std": auc_std,
        "cv/f1_mean": f1_mean,
        "cv/f1_std": f1_std,
        "train/best_iter_final": best_iter_final,
        "oof/best_threshold": threshold,
        "oof/f1_at_best_threshold": best_f1,
    })
    # Persist run meta (threshold/metrics) for reproducibility (included in fold model bundle under meta/)
    try:
        threshold_payload = {
            "threshold": float(threshold),
            "best_f1": float(best_f1),
            "auc_mean": float(auc_mean),
            "auc_std": float(auc_std),
            "f1_mean_at_thr": float(f1_mean),
            "f1_std_at_thr": float(f1_std),
            "best_iter_final": int(best_iter_final),
            "threshold_mode": "optimized" if bool(cfg.train.optimize_threshold.enable) else "fixed",
        }
        save_yaml_in_output_dir("threshold.yaml", threshold_payload)
    except Exception as e:
        print(f"[Warn] saving threshold meta failed: {e}")



    # Save OOF
    oof_path = Path(cfg.train.oof_path)
    pd.DataFrame({"oof_proba": oof_proba, "y_true": y.values}).to_csv(oof_path, index=False)
    print("Saved OOF:", oof_path)
    if run is not None and bool(cfg.wandb.log_artifacts):
        wandb_log_artifact(run, oof_path, name=make_wandb_artifact_name(cfg, run, "oof_proba"), art_type="dataset")

    # Predict
    if use_fold_ensemble:
        test_proba = test_proba_cv
        final_model = None
        print(f"[Predict] Using fold-ensemble proba (mean over {cfg.train.n_splits} folds)")
    else:
        final_model, test_proba = fit_final_and_predict(cfg, X, y, test, cat_cols, best_iter_final)
        print("[Predict] Using single final model fit on full data")

    test_pred = (test_proba >= threshold).astype(int)

    # Submission
    out_path = Path(cfg.train.out_path)
    if sample_submission_path.exists():
        sub = pd.read_csv(sample_submission_path)
        if cfg.data.target_col not in sub.columns:
            # sample_submission에 타깃 컬럼이 없으면 추가
            sub[cfg.data.target_col] = test_pred
        else:
            sub[cfg.data.target_col] = test_pred
        sub.to_csv(out_path, index=False)
    else:
        sub = pd.DataFrame({cfg.data.target_col: test_pred})
        sub.to_csv(out_path, index=False)

    print("Saved submission:", out_path)

    # Probabilities
    if cfg.train.out_proba_path:
        proba_path = Path(cfg.train.out_proba_path)
        pd.DataFrame({f"{cfg.data.target_col}_proba": test_proba}).to_csv(proba_path, index=False)
        print("Saved proba:", proba_path)

    manifest_path: Optional[Path] = None
    fold_bundle_path: Optional[Path] = None
    model_bundle_path: Optional[Path] = None

    # Save model
    if bool(cfg.train.save_model):
        model_path = Path(cfg.train.model_path)

        if use_fold_ensemble:
            # fold 모델은 cv_train에서 이미 저장됨(save_fold_models=True일 때)
            manifest_path = model_path.with_name(f"{model_path.stem}_folds_manifest.txt")
            manifest_path.parent.mkdir(parents=True, exist_ok=True)

            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(f"use_fold_ensemble: {use_fold_ensemble}\n")
                f.write(f"n_folds: {int(cfg.train.n_splits)}\n")
                f.write(f"threshold: {float(threshold)}\n")
                f.write("fold_model_paths:\n")
                for fp in fold_model_paths:
                    f.write(f"  - {str(fp)}\n")

            print("Saved fold model manifest:", manifest_path)
            if not fold_model_paths:
                print("[Warn] fold_model_paths is empty (check save_fold_models / model_path).")

            # Bundle manifest + fold models into a single zip (for a single W&B artifact)
            try:
                import zipfile
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fold_bundle_path = make_repro_bundle_path(model_path, cfg, run, kind="fold", ts=ts)
                with zipfile.ZipFile(fold_bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    if manifest_path.exists():
                        zf.write(manifest_path, arcname=f"fold_models/{manifest_path.name}")
                    for fp in fold_model_paths:
                        fp = Path(fp)
                        if fp.exists():
                            zf.write(fp, arcname=f"fold_models/{fp.name}")
                    # Add meta files (config/best_params/threshold etc.) under meta/
                    try:
                        try:
                            output_dir = Path(HydraConfig.get().runtime.output_dir)
                        except Exception:
                            output_dir = Path.cwd()

                        meta_candidates = [
                            output_dir / "config_resolved.yaml",
                            output_dir / "best_params.yaml",
                            output_dir / "threshold.yaml",
                            output_dir / "feature_columns.json",
                            output_dir / "preprocess_meta.yaml",
                            output_dir / "run_info.yaml",
                        ]
                        for mp in meta_candidates:
                            if mp.exists() and mp.is_file():
                                zf.write(mp, arcname=f"meta/{mp.name}")

                        # Add outputs for one-shot reproducibility
                        try:
                            if out_path.exists():
                                zf.write(out_path, arcname="outputs/submission.csv")
                            if "proba_path" in locals() and (proba_path is not None) and Path(proba_path).exists():
                                zf.write(Path(proba_path), arcname="outputs/submission_proba.csv")
                            if oof_path.exists():
                                zf.write(oof_path, arcname="outputs/oof.csv")
                        except Exception as e:
                            print(f"[Warn] adding outputs to bundle failed: {e}")
                    except Exception as e:
                        print(f"[Warn] adding meta files to bundle failed: {e}")
                print("Saved fold model bundle:", fold_bundle_path)
            except Exception as e:
                print(f"[Warn] fold model bundling failed: {e}")
                fold_bundle_path = None


        else:
            booster = final_model.booster_
            model_path.parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(str(model_path))
            print("Saved model:", model_path)

            # Bundle final model + meta + outputs into a single zip for full reproducibility
            try:
                import zipfile
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_bundle_path = make_repro_bundle_path(model_path, cfg, run, kind="single", ts=ts)

                with zipfile.ZipFile(model_bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    # Model
                    if model_path.exists():
                        zf.write(model_path, arcname=f"model/{model_path.name}")

                    # Meta
                    try:
                        try:
                            output_dir = Path(HydraConfig.get().runtime.output_dir)
                        except Exception:
                            output_dir = Path.cwd()

                        meta_candidates = [
                            output_dir / "config_resolved.yaml",
                            output_dir / "best_params.yaml",
                            output_dir / "threshold.yaml",
                            output_dir / "feature_columns.json",
                            output_dir / "preprocess_meta.yaml",
                            output_dir / "run_info.yaml",
                        ]
                        for mp in meta_candidates:
                            if mp.exists() and mp.is_file():
                                zf.write(mp, arcname=f"meta/{mp.name}")
                    except Exception as e:
                        print(f"[Warn] adding meta files to model bundle failed: {e}")

                    # Outputs
                    try:
                        if out_path.exists():
                            zf.write(out_path, arcname="outputs/submission.csv")
                        if "proba_path" in locals() and (proba_path is not None) and Path(proba_path).exists():
                            zf.write(Path(proba_path), arcname="outputs/submission_proba.csv")
                        if oof_path.exists():
                            zf.write(oof_path, arcname="outputs/oof.csv")
                    except Exception as e:
                        print(f"[Warn] adding outputs to model bundle failed: {e}")

                print("Saved model bundle:", model_bundle_path)
            except Exception as e:
                print(f"[Warn] model bundling failed: {e}")
                model_bundle_path = None

    # W&B artifacts + feature importance
    if run is not None:
        if bool(cfg.wandb.log_feature_importance):
            if use_fold_ensemble:
                importances = cv_feat_imp
            else:
                importances = final_model.feature_importances_

            fi = pd.DataFrame({
                "feature": X.columns,
                "importance": importances,
            }).sort_values("importance", ascending=False)
            try:
                import wandb
                table = wandb.Table(dataframe=fi.head(200))
                wandb.log({"feature_importance_top200": table})
            except Exception as e:
                print(f"[Warn] feature importance logging failed: {e}")

        if bool(cfg.wandb.log_artifacts):
            if out_path.exists():
                wandb_log_artifact(run, out_path, name=make_wandb_artifact_name(cfg, run, "submission"), art_type="output")
            if cfg.train.out_proba_path and Path(cfg.train.out_proba_path).exists():
                wandb_log_artifact(run, Path(cfg.train.out_proba_path), name=make_wandb_artifact_name(cfg, run, "submission_proba"), art_type="output")

            # Model artifacts
            if bool(cfg.train.save_model):
                if use_fold_ensemble:
                    if fold_bundle_path is not None and fold_bundle_path.exists():
                        wandb_log_artifact(run, fold_bundle_path, name=make_wandb_artifact_name(cfg, run, "repro_bundle_fold"), art_type="model")
                    elif manifest_path is not None and manifest_path.exists():
                        wandb_log_artifact(run, manifest_path, name=make_wandb_artifact_name(cfg, run, "fold_model_manifest"), art_type="model")
                else:
                    model_path = Path(cfg.train.model_path)
                    if model_bundle_path is not None and Path(model_bundle_path).exists():
                        wandb_log_artifact(
                            run,
                            Path(model_bundle_path),
                            name=make_wandb_artifact_name(cfg, run, "repro_bundle_single"),
                            art_type="model",
                        )
                    elif model_path.exists():
                        wandb_log_artifact(run, model_path, name=make_wandb_artifact_name(cfg, run, "final_model"), art_type="model")

        run.finish()


if __name__ == "__main__":
    main()
