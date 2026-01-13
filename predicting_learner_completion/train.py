#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_model_hydra_wandb.py

- Hydra로 "전처리 + 샘플러 + 모델"을 전부 config 조합으로 스위칭
- 모델: conf/model/*.yaml에서 _target_만 바꿔 교체
- 전처리: conf/preprocess/*.yaml에서 step들의 _target_만 바꿔 교체
- 샘플러(SMOTE 등): conf/sampler/*.yaml에서 enable + _target_로 교체
- W&B: 설정/메트릭/아티팩트 기록

실행 예시
---------
# 기본 (preprocess=onehot_dense, sampler=none, model=random_forest)
python train_model_hydra_wandb.py

# 전처리만 교체 (희소 원핫 + 표준화) + 로지스틱
python train_model_hydra_wandb.py preprocess=standardize_onehot_sparse model=logistic_regression

# 전처리(TargetEncoder) + SMOTE + XGBoost (xgboost, imbalanced-learn, category_encoders 설치 필요)
python train_model_hydra_wandb.py preprocess=target_encoder sampler=smote model=xgboost

# 멀티런(스윕)
python train_model_hydra_wandb.py -m \
  preprocess=onehot_sparse,standardize_onehot_sparse \
  model=logistic_regression,random_forest \
  seed=1,2,3

주의
----
- RandomForest/트리 모델은 보통 sparse 입력을 직접 처리하지 못합니다.
  -> preprocess=onehot_dense(기본) 또는 preprocess=target_encoder/catboost_encoder(출력 dense) 권장
- SMOTE는 기본적으로 dense 입력을 기대합니다(특히 one-hot 희소행렬과 궁합이 안 좋음).
  -> preprocess=target_encoder/catboost_encoder 같은 dense 인코딩 + sampler=smote 추천
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline

# ---- Optional: Hydra runtime choices (selected config names)
try:
    from hydra.core.hydra_config import HydraConfig
except Exception:
    HydraConfig = None


# -------------------------
# Helper factories (Hydra _target_로 참조 가능)
# -------------------------
def make_onehot(handle_unknown: str = "ignore", sparse_output: bool = True):
    """
    sklearn OneHotEncoder 생성 (버전 호환: sparse_output / sparse)
    config에서 _target_: train_model_hydra_wandb.make_onehot 로 사용
    """
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=sparse_output)
    except TypeError:
        # older sklearn
        return OneHotEncoder(handle_unknown=handle_unknown, sparse=sparse_output)


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def add_features(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """
    간단한 feature engineering:
    - major_field 컬럼에 IT 관련 키워드가 들어가면 is_major_it=1
    """
    fe = cfg.feature_engineering
    if not fe.enable:
        return df

    major_col = fe.major_field_col
    new_col = fe.new_col
    pattern = fe.it_major_pattern

    if major_col not in df.columns:
        warnings.warn(f"[feature_engineering] column '{major_col}' not found. Skip '{new_col}'.")
        return df

    s = df[major_col].astype(str).fillna("")
    df[new_col] = s.str.contains(pattern, flags=re.IGNORECASE, regex=True).astype(int)
    return df


def drop_high_missing(
    train: pd.DataFrame,
    test: pd.DataFrame,
    threshold: float,
    protect_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    protect_cols = protect_cols or []
    ratios = train.isna().mean()
    drop_cols = [c for c, r in ratios.items() if (r > threshold and c not in protect_cols)]
    if drop_cols:
        train = train.drop(columns=drop_cols, errors="ignore")
        test = test.drop(columns=drop_cols, errors="ignore")
    return train, test, drop_cols


def parse_feature_cols(feature_cols: Optional[str]) -> Optional[List[str]]:
    if feature_cols is None:
        return None
    if isinstance(feature_cols, str):
        cols = [c.strip() for c in feature_cols.split(",") if c.strip()]
        return cols or None
    return None


def split_columns_auto(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols: List[str] = []
    num_cols: List[str] = []

    for c in X.columns:
        if (
            pd.api.types.is_bool_dtype(X[c])
            or pd.api.types.is_object_dtype(X[c])
            or pd.api.types.is_string_dtype(X[c])
            or isinstance(X[c].dtype, pd.CategoricalDtype)
        ):
            cat_cols.append(c)
        else:
            num_cols.append(c)

    return num_cols, cat_cols


def build_steps(steps_cfg) -> List[Tuple[str, Any]]:
    """
    steps_cfg 형식(예):
      steps:
        - name: imputer
          transformer:
            _target_: sklearn.impute.SimpleImputer
            strategy: median
    """
    steps: List[Tuple[str, Any]] = []
    if steps_cfg is None:
        return steps

    for i, st in enumerate(steps_cfg):
        name = st.get("name", f"step{i}")
        tr_cfg = st.get("transformer", None)
        if tr_cfg is None:
            continue
        tr = instantiate(tr_cfg)
        steps.append((name, tr))
    return steps


def build_preprocessor(X: pd.DataFrame, cfg: DictConfig) -> ColumnTransformer:
    """
    conf/preprocess/*.yaml 조합에 따라 ColumnTransformer를 구성합니다.
    - numeric.steps / categorical.steps를 각각 Pipeline로 만들고 ColumnTransformer에 넣음
    """
    num_cols, cat_cols = split_columns_auto(X)

    num_steps = build_steps(cfg.preprocess.numeric.steps)
    cat_steps = build_steps(cfg.preprocess.categorical.steps)

    num_pipe = SkPipeline(steps=num_steps) if num_steps else "passthrough"
    cat_pipe = SkPipeline(steps=cat_steps) if cat_steps else "passthrough"

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder=cfg.preprocess.remainder,
        sparse_threshold=float(cfg.preprocess.sparse_threshold),
    )
    return pre


def safe_instantiate_model(cfg: DictConfig):
    try:
        return instantiate(cfg.model)
    except ModuleNotFoundError as e:
        target = getattr(cfg.model, "_target_", "")
        if "xgboost" in target:
            raise ModuleNotFoundError(
                "xgboost가 설치되어 있지 않습니다. `pip install xgboost` 후 다시 실행하세요."
            ) from e
        if "lightgbm" in target:
            raise ModuleNotFoundError(
                "lightgbm이 설치되어 있지 않습니다. `pip install lightgbm` 후 다시 실행하세요."
            ) from e
        raise


def safe_instantiate_sampler(cfg: DictConfig, num_cols_count: int, cat_cols_count: int):
    # sampler는 preprocess 뒤에 적용됩니다.
    # - 일반 SMOTE/ADASYN/언더샘플러/SMOTEENN/SMOTETomek: 그대로 instantiate
    # - SMOTENC: categorical_features 인덱스가 필요
    #   * conf에서 categorical_features: auto 로 두면
    #     ColumnTransformer 출력이 [num | cat] (cat이 OrdinalEncoder로 1컬럼씩 유지된다는 가정)일 때
    #     categorical index를 자동으로 채웁니다.
    if not cfg.sampler.enable:
        return None
    if cfg.sampler.sampler is None:
        return None

    try:
        target = getattr(cfg.sampler.sampler, "_target_", "")
        if "SMOTENC" in str(target):
            cat_feat = getattr(cfg.sampler.sampler, "categorical_features", None)
            if cat_feat == "auto":
                cat_indices = list(range(num_cols_count, num_cols_count + cat_cols_count))
                return instantiate(cfg.sampler.sampler, categorical_features=cat_indices)
        return instantiate(cfg.sampler.sampler)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "imbalanced-learn이 설치되어 있지 않습니다. `pip install imbalanced-learn` 후 다시 실행하세요."
        ) from e


def build_pipeline(preprocess, sampler, model):
    """
    sampler가 있으면 imblearn.pipeline.Pipeline을 사용(중간에 resampling step 지원)
    없으면 sklearn Pipeline 사용
    """
    if sampler is None:
        return SkPipeline(steps=[("preprocess", preprocess), ("model", model)])

    try:
        from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "SMOTE 등을 쓰려면 imbalanced-learn이 필요합니다. `pip install imbalanced-learn`"
        ) from e

    return ImbPipeline(steps=[("preprocess", preprocess), ("sampler", sampler), ("model", model)])


def get_score_for_auc(model: Any, X: Any) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return np.asarray(s).ravel()
    return None


def maybe_init_wandb(cfg: DictConfig):
    if not cfg.wandb.enable:
        return None

    try:
        import wandb  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "wandb가 설치되어 있지 않습니다. `pip install wandb` 후 다시 실행하세요."
        ) from e

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict.pop("hydra", None)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        job_type=cfg.wandb.job_type,
        tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
        notes=cfg.wandb.notes,
        mode=cfg.wandb.mode,
        config=cfg_dict,
    )
    return run


def log_wandb_artifacts(run, cfg: DictConfig, model_path: Path, submit_path: Path):
    import wandb  # type: ignore

    if cfg.wandb.log_model and model_path.exists():
        art = wandb.Artifact(name=f"{cfg.wandb.artifact_name_prefix}-model", type="model")
        art.add_file(str(model_path))
        run.log_artifact(art)

    if cfg.wandb.log_submission and submit_path.exists():
        art = wandb.Artifact(name=f"{cfg.wandb.artifact_name_prefix}-submission", type="submission")
        art.add_file(str(submit_path))
        run.log_artifact(art)


def try_get_feature_names(pipe) -> Optional[np.ndarray]:
    try:
        pre = pipe.named_steps.get("preprocess", None)
        if pre is None:
            return None
        if hasattr(pre, "get_feature_names_out"):
            return pre.get_feature_names_out()
    except Exception:
        return None
    return None


def log_feature_importance_or_coef(run, pipe, topk: int = 30):
    if run is None:
        return
    try:
        import wandb  # type: ignore
    except Exception:
        return

    model = pipe.named_steps.get("model", None)
    if model is None:
        return

    feat_names = try_get_feature_names(pipe)
    if feat_names is None:
        return

    rows = None
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_).ravel()
        if len(imp) == len(feat_names):
            idx = np.argsort(-imp)[:topk]
            rows = [(str(feat_names[i]), float(imp[i])) for i in idx]
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        if len(coef) == len(feat_names):
            idx = np.argsort(-np.abs(coef))[:topk]
            rows = [(str(feat_names[i]), float(coef[i])) for i in idx]

    if rows:
        table = wandb.Table(columns=["feature", "value"], data=rows)
        run.log({"model/top_features": table})


def log_selected_choices(run):
    if run is None or HydraConfig is None:
        return
    try:
        choices = HydraConfig.get().runtime.choices
        # choices: {'model': 'random_forest', 'preprocess': 'onehot_dense', 'sampler': 'none', ...}
        run.log(
            {
                "choice/model": choices.get("model"),
                "choice/preprocess": choices.get("preprocess"),
                "choice/sampler": choices.get("sampler"),
            }
        )
    except Exception:
        pass


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # --- paths: Hydra run dir에서도 안정적으로 읽기 위해 절대경로화
    train_path = Path(to_absolute_path(cfg.paths.train))
    test_path = Path(to_absolute_path(cfg.paths.test))
    sample_sub_path = Path(to_absolute_path(cfg.paths.sample_sub))

    out_path = Path(cfg.paths.out)
    model_out_path = Path(cfg.paths.model_out)

    # --- load
    train = load_csv(train_path)
    test = load_csv(test_path)
    sub = load_csv(sample_sub_path)

    if cfg.target not in train.columns:
        raise ValueError(f"Target column '{cfg.target}' not found in train")

    # --- feature engineering
    train = add_features(train, cfg)
    test = add_features(test, cfg)

    # --- drop missing-heavy columns
    protect = [cfg.target, cfg.id_col]
    train, test, dropped = drop_high_missing(train, test, cfg.missing_threshold, protect_cols=protect)

    # --- select features
    feature_cols = parse_feature_cols(cfg.feature_cols)
    if feature_cols:
        keep = [c for c in feature_cols if c in train.columns and c not in {cfg.target}]
    else:
        keep = [c for c in train.columns if c not in {cfg.target, cfg.id_col}]

    X = train[keep].copy()
    y = train[cfg.target].astype(int).values

    # --- build preprocess + sampler + model from config
    # 컬럼 개수(특히 SMOTENC categorical index 자동화를 위해)
    num_cols, cat_cols = split_columns_auto(X)

    preprocess = build_preprocessor(X, cfg)
    sampler = safe_instantiate_sampler(cfg, num_cols_count=len(num_cols), cat_cols_count=len(cat_cols))
    model = safe_instantiate_model(cfg)

    pipe = build_pipeline(preprocess, sampler, model)

    # --- validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=cfg.val_size,
        random_state=cfg.seed,
        stratify=y if cfg.training.stratify else None,
    )

    run = maybe_init_wandb(cfg)
    log_selected_choices(run)

    # --- fit / eval
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_val)

    metrics: Dict[str, Any] = {
        "val/acc": float(accuracy_score(y_val, pred)),
        "val/f1": float(f1_score(y_val, pred)),
    }

    score = get_score_for_auc(pipe, X_val)
    if score is not None:
        try:
            metrics["val/auc"] = float(roc_auc_score(y_val, score))
        except Exception:
            pass

    print("[VAL]", {k: round(v, 6) for k, v in metrics.items() if isinstance(v, (float, int))})

    if run is not None:
        run.log(metrics)
        run.log(
            {
                "data/train_rows": int(len(train)),
                "data/test_rows": int(len(test)),
                "data/num_features_raw": int(len(keep)),
                "data/dropped_cols_count": int(len(dropped)),
                "pipe/has_sampler": bool(sampler is not None),
            }
        )
        if dropped:
            import wandb  # type: ignore
            table = wandb.Table(columns=["dropped_col"], data=[[c] for c in dropped])
            run.log({"data/dropped_cols": table})

        log_feature_importance_or_coef(run, pipe, topk=cfg.logging.topk_features)

    # --- fit full & predict
    pipe.fit(X, y)
    X_test = test[keep].copy()
    pred_test = pipe.predict(X_test)

    # --- submission
    submission = sub.copy()
    if cfg.id_col in submission.columns and cfg.id_col in test.columns:
        if submission[cfg.id_col].nunique() == len(submission) and test[cfg.id_col].nunique() == len(test):
            tmp = pd.DataFrame({cfg.id_col: test[cfg.id_col].values, cfg.target: pred_test})
            submission = submission.drop(columns=[cfg.target], errors="ignore").merge(tmp, on=cfg.id_col, how="left")
        else:
            submission[cfg.target] = pred_test
    else:
        submission[cfg.target] = pred_test

    submission.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] saved submission to: {out_path.resolve()}")

    # --- save model pipeline
    joblib.dump(pipe, model_out_path)
    print(f"[OK] saved model pipeline to: {model_out_path.resolve()}")

    # --- wandb artifacts
    if run is not None:
        log_wandb_artifacts(run, cfg, model_out_path, out_path)
        run.finish()


if __name__ == "__main__":
    main()
