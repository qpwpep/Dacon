import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

import lightgbm as lgb

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


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

        tr = train_df[col].astype("object")
        tr = tr.where(tr.notna(), "__MISSING__").astype(str)
        cats = pd.Index(tr.unique()).append(pd.Index(["__UNKNOWN__"]))
        cats = cats.drop_duplicates()

        train_df[col] = pd.Categorical(tr, categories=cats)

        if col in test_df.columns:
            te = test_df[col].astype("object")
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


def wandb_log_artifact(run, path: Path, name: str, art_type: str) -> None:
    if run is None:
        return
    import wandb

    art = wandb.Artifact(name=name, type=art_type)
    art.add_file(str(path))
    run.log_artifact(art)


def search_best_threshold(y_true: np.ndarray, proba: np.ndarray, start: float, end: float, step: float) -> Tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    t = start
    while t <= end + 1e-12:
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
        t += step
    return best_t, float(best_f1)


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
            start=float(cfg.train.optimize_threshold.grid_start),
            end=float(cfg.train.optimize_threshold.grid_end),
            step=float(cfg.train.optimize_threshold.grid_step),
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

    # Save OOF
    oof_path = Path(cfg.train.oof_path)
    pd.DataFrame({"oof_proba": oof_proba, "y_true": y.values}).to_csv(oof_path, index=False)
    print("Saved OOF:", oof_path)
    if run is not None and bool(cfg.wandb.log_artifacts):
        wandb_log_artifact(run, oof_path, name="oof_proba", art_type="dataset")

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

        else:
            booster = final_model.booster_
            model_path.parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(str(model_path))
            print("Saved model:", model_path)

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
                wandb_log_artifact(run, out_path, name="submission", art_type="output")
            if cfg.train.out_proba_path and Path(cfg.train.out_proba_path).exists():
                wandb_log_artifact(run, Path(cfg.train.out_proba_path), name="submission_proba", art_type="output")
            if bool(cfg.train.save_model) and Path(cfg.train.model_path).exists():
                wandb_log_artifact(run, Path(cfg.train.model_path), name="final_model", art_type="model")

        run.finish()


if __name__ == "__main__":
    main()
