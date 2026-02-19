# catboost_kfold.py
# CatBoost 기반 베이스라인 (K-Fold 적용 버전) - CFG 적용(상단 설정 집중형)
# + Validation 오류(틀린 샘플) 추적/저장
# + Feature Importance 집계/시각화(폴드 평균)

from __future__ import annotations

VERSION = "v2_multiselect_multihot_tokens"

import os
import re
import json
import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from catboost import CatBoostClassifier, Pool  # type: ignore
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support, accuracy_score  # type: ignore
from sklearn.model_selection import KFold, StratifiedKFold  # type: ignore


# =============================
# Centralized Experiment Config
# =============================
@dataclass(frozen=True)
class CFG:
    # -----------------
    # Repro / CV
    # -----------------
    n_splits: int = 5
    seed: int = 42
    use_stratified_if_binary: bool = True

    # -----------------
    # Data / I/O
    # -----------------
    train_path: str = "data/train.csv"
    test_path: str = "data/test.csv"
    sample_submission_path: str = "data/sample_submission.csv"
    output_path: str = "submit.csv"

    # -----------------
    # Target / Submission
    # -----------------
    target: str = "completed"
    submit_proba: bool = False
    decision_threshold: float = 0.5

    # -----------------
    # Engagement / Preference config
    # -----------------
    common_none_tokens: Tuple[str, ...] = (
        "", "없음", "없습니다", "무", ".", "아직 없음", "딱히 없음", "해당없음", "없어요", "x", "X", "none", "nan"
    )

    engagement_optional_cols: Tuple[str, ...] = (
        "major1_2",
        "class2", "class3", "class4",
        "contest_participation",
        "previous_class_3", "previous_class_4", "previous_class_5",
        "previous_class_6", "previous_class_7", "previous_class_8",
    )

    expected_domain_codes_focus: Tuple[str, ...] = ("K","J","M","R","U","C","O","P","Q","G")  # 빈도 높은 섹션 위주
    company_kw: Tuple[str, ...] = ("네이버","카카오","삼성","토스","구글","현대","LG","쿠팡","넥슨","은행")
    ic_primary_topk: int = 60
    ic_bigtech_kw: Tuple[str, ...] = ("네이버","카카오","쿠팡","토스","라인","당근","배민","우아한형제들","구글","google","meta","microsoft","amazon","apple")
    ic_finance_kw: Tuple[str, ...] = ("은행","카드","증권","보험","kb","국민","신한","하나","우리","nh","농협","삼성카드","카카오뱅크","토스뱅크")
    ic_game_kw: Tuple[str, ...] = ("넥슨","크래프톤","넷마블","엔씨","nc","펄어비스","스마일게이트")
    ic_chaebol_kw: Tuple[str, ...] = ("삼성","lg","현대","sk","롯데","한화","포스코")

    # -----------------
    # Multi-select vocab multihot
    # -----------------
    multihot_min_freq: int = 2
    multihot_token_max_len: int = 60
    major_field_topk: int = 20
    whyBDA_topk: int = 20
    cert_acq_topk: int = 30
    cert_prep_topk: int = 20
    desired_cert_tok_topk: int = 25
    ic_any_topk: int = 80

    # -----------------
    # Feature definitionincumbents_lecture_scale_reason
    # -----------------
    base_features: Tuple[str, ...] = (
    # 기본
    "class1",
    "re_registration",
    "inflow_route",
    "time_input",
    "time_input_bucket",

    # engagement (Quick win)
    "eng_filled_all",
    "eng_filled_all_ratio",
    "eng_filled_optional",
    "eng_filled_optional_ratio",
    "eng_prev_class_filled",
    "cert_acq_filled",
    "whyBDA_len",
    "what_to_gain_len",
    "interested_company_n",
    "interested_company_len",
    "ic_primary_topk",
    "ic_primary_freq",
    "ic_primary_freq_log1p",
    "ic_primary_is_rare",
    "ic_has_bigtech",
    "ic_has_finance",
    "ic_has_game",
    "ic_has_chaebol",
    "whyBDA",
    "what_to_gain",

    # 커밋먼트/상태
    "completed_semester_clean",
    "completed_semester_bucket",

    # 자격증(취득)
    "cert_count",
    "has_any_cert",
    "has_adsp",
    "has_sqld",
    "has_info_process",

    # 전공
    "major type",
    "major1_1",
    "major1_2",
    "major1_1_topk",
    "major1_1_freq",
    "major1_1_freq_log1p",
    "major1_1_is_rare",
    "major1_2_topk",
    "major1_2_freq",
    "major1_2_freq_log1p",
    "major1_2_is_rare",
    "major_data",
    "is_major_it",
    "major1_1_is_data",
    "major1_2_is_data",
    "is_double_major",

    # 이전 기수 수강(집계)
    "has_prev_class",
    "prev_class_code_nunique",
    "prev_class_gen_count",
    "prev_class_latest_gen",
    "prev_class_latest_code",

    # 희망/선호(저 cardinality는 categorical로 직접 투입)
    "school1_topk",
    "school1_freq",
    "school1_freq_log1p",
    "school1_is_rare",
    "job",
    "why_scale_ops",
    "why_benefit",
    "why_hard_alone",
    "why_satisfied_prev",
    "why_low_burden",
    "why_no_test",
    "why_want_lecture",
    "why_count",
    "gain_project",
    "gain_contest",
    "gain_analysis",
    "gain_network",
    "gain_python_sql",
    "gain_all",
    "gain_collab",
    "gain_len",
    "gain_words",
    "gain_count",
    "desired_career_path",
    "project_type",
    "hope_for_group",
    "incumbents_level",
    "incumbents_company_level",
    "incumbents_lecture",
    "incumbents_lecture_type",
    "incumbents_lecture_scale",

    # expected_domain (구조화)
    "expected_domain_code_count",
    "expected_domain_primary_code",
    "expected_domain_has_K",
    "expected_domain_has_J",
    "expected_domain_has_M",
    "expected_domain_has_R",
    "expected_domain_has_U",
    "expected_domain_has_C",
    "expected_domain_has_O",
    "expected_domain_has_P",
    "expected_domain_has_Q",
    "expected_domain_has_G",

    # desired_certificate (구조화)
    "desired_cert_count",
    "desired_has_bigdata",
    "desired_has_sqld",
    "desired_has_adsp",
    "desired_has_info_process",
    "desired_has_tableau",
    "desired_has_google_analyst",

    # 멀티선택(기존)
    "desired_job_n",
    "desired_job_except_n",
    "desired_job_has_A",
    "desired_job_has_B",
    "desired_job_has_C",
    "desired_job_has_D",
    "desired_job_has_E",
    "desired_job_has_F",
    "desired_job_has_G",
    "desired_job_has_H",
    "desired_job_has_I",
    "desired_job_has_J",
    "desired_job_except_has_A",
    "desired_job_except_has_B",
    "desired_job_except_has_C",
    "desired_job_except_has_D",
    "desired_job_except_has_E",
    "desired_job_except_has_F",
    "desired_job_except_has_G",
    "desired_job_except_has_H",

    # 텍스트(기존 유지)
    "inc_lecture_reason_len",
    "inc_lecture_reason_words",
    "inc_lecture_reason_has_job_kw",

    # 원데이 토픽
    "oneday_topic_count",
    "oneday_has_python",
    "oneday_has_sql",
    "oneday_has_crawl",
    "oneday_has_viz",
    "oneday_has_ml_dl",
    "oneday_has_de",
    "oneday_has_portfolio",
    "oneday_has_tableau",
    "oneday_has_powerbi",
    "oneday_has_ga",
    "oneday_has_contest",
    "oneday_has_finance",
    "oneday_has_cnn",
    "oneday_has_rnn_chatgpt",

    # signature rule features (FP/FN)
    "sig_fp_rereg_yes_time_le3",
    "sig_fp_rereg_yes_sem_le8",
    "sig_fp_prevclass_time_le3",
    "sig_fp_prevclass_sem_le8",
    "sig_fp_inflow_member_time_le3",
    "sig_fn_new_no_cert",
    "sig_fn_new_major2_missing",
    "sig_fn_prevnone_no_cert",
    "sig_fn_prevnone_major2_missing",
    "sig_fn_major2_missing_no_adsp",
    "eng_optional_ge2",
    "sig_fn_new_no_cert_loweng",
    "sig_fn_new_no_cert_higheng",
    "sig_fn_new_major2_missing_loweng",
    "sig_fn_new_major2_missing_higheng",
    "sig_fn_prevnone_no_cert_loweng",
    "sig_fn_prevnone_no_cert_higheng",
    "sig_fn_prevnone_major2_missing_loweng",
    "sig_fn_prevnone_major2_missing_higheng",
    "sig_fn_major2_missing_no_adsp_loweng",
    "sig_fn_major2_missing_no_adsp_higheng",
    # --- v9 additions (ensure actually used) ---
    "major_pair",
    "major_pair_freq",
    "major_pair_freq_log1p",
    "desired_job_data_n",
    "desired_job_nondata_n",
    "desired_job_is_data_only",
    "desired_job_is_nondata_only",
    "desired_job_is_mixed",
    "motivation_score",
    "motivation_geq",
    "sig_fn_new_no_cert_loweng_motivated",
    "sig_fn_prevnone_no_cert_loweng_motivated",
    "sig_fn_major2_missing_no_adsp_loweng_motivated",
    "sig_fn_major2_missing_no_adsp_loweng_has_sqld",
    "sig_fn_major2_missing_no_adsp_loweng_has_any_cert",
    "sig_fn_risk_score",
    "sig_fn_risk_score_loweng",

    )

    text_features: Tuple[str, ...] = (
        # "incumbents_lecture_scale_reason",
    )
    force_categorical: Tuple[str, ...] = (
    "class1",
    "re_registration",
    "inflow_route",
    "school1_topk",
    "job",
    "desired_career_path",
    "project_type",
    "hope_for_group",
    "incumbents_level",
    "incumbents_company_level",
    "incumbents_lecture",
    "incumbents_lecture_type",
    "incumbents_lecture_scale",
    "major type",
    "major1_1",
    "major1_2",
    "completed_semester_bucket",
    "time_input_bucket",
    "prev_class_latest_code",
    "expected_domain_primary_code",
    "whyBDA",
    "what_to_gain",
    "ic_primary_topk",
    "major_pair",
    )

    oneday_rules: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        ("python", ("python",)),
        ("sql", ("sql",)),
        ("crawl", ("크롤링", "crawling")),
        ("viz", ("시각화", "matplotlib", "seaborn")),
        ("ml_dl", ("머신러닝", "딥러닝")),
        ("de", ("데이터 엔지니어링", "파이프라인")),
        ("portfolio", ("포트폴리오",)),
        ("tableau", ("태블로", "테블로")),
        ("powerbi", ("power bi", "powerbi", "파워비", "파워비아이", "powebi")),
        ("ga", ("구글 애널리", "google analy", "google analytics")),
        ("contest", ("공모전",)),
        ("finance", ("금융",)),
        ("cnn", ("cnn",)),
        ("rnn_chatgpt", ("rnn", "chatgpt")),
    )

    # -----------------
    # Preprocessing
    # -----------------
    completed_semester_col: str = "completed_semester"
    completed_semester_max: int = 20  # 20241 같은 outlier 제거 기준

    cert_col: str = "certificate_acquisition"
    cert_none_tokens: Tuple[str, ...] = ("없음", "없습니다", "무", ".", "아직 없음", "딱히 없음")

    desired_job_col: str = "desired_job"
    desired_job_except_col: str = "desired_job_except_data"
    desired_job_codes: Tuple[str, ...] = tuple("ABCDEFGHIJ")
    desired_job_except_codes: Tuple[str, ...] = tuple("ABCDEFGH")

    auto_expand_prefixes: Tuple[str, ...] = (
        "desired_job_has_", "desired_job_except_has_",
        # train-fitted vocab multihot
        "major_field_", "whyBDA_is_",
        "cert_acq_", "cert_prep_",
        "desired_cert_tok_",
        "ic_any_",
    )


    # 텍스트 키워드(가성비 좋은 contains 피처)
    job_keywords: Tuple[str, ...] = ("취업", "이직", "포트폴리오", "프로젝트", "현업", "현직", "면접", "커리어")

    # -----------------
    # Missingness policy
    # -----------------
    drop_missing_ratio_gt: float = 0.8

    # -----------------
    # major_field -> is_major_it rule
    # -----------------
    major_col: str = "major_field"
    major_it_token: str = "IT"

    # -----------------
    # time_input policy
    # -----------------
    time_col: str = "time_input"
    time_datetime_ok_ge: float = 0.5
    time_numeric_ok_ge: float = 0.9
    time_datetime_created_cols: Tuple[str, ...] = (
        "time_input_hour",
        "time_input_dayofweek",
        "time_input_month",
        "time_input_day",
        "time_input_is_weekend",
        "time_input_unix",
    )

    # -----------------
    # school1 안정화
    # -----------------
    school1_topk: int = 30          # train 기준 상위 30개(커버리지 ~78%)
    school1_rare_leq: int = 2

    # -----------------
    # major 안정화
    # -----------------
    major_topk: int = 20            # major1_1/major1_2 상위 20개
    major_rare_threshold: int = 1   # freq <= 1이면 희소
    major_pair_sep: str = "|"

    # -----------------
    # signature rule 완화(예외 처리)
    # -----------------
    sig_eng_optional_geq: int = 2   # eng_filled_optional >= 2면 '예외/완화'로 분기
    sig_motivation_geq: int = 4      # motivation_score >= 4면 저참여(loweng)에서도 예외 신호
    sig_time_geq: float = 3           # time_input >= 3
    sig_semester_geq: float = 5       # completed_semester >= 5

    # -----------------
    # Text normalization
    # -----------------
    text_ws_regex: str = r"\s+"

    # -----------------
    # Validation error analysis
    # -----------------
    save_validation_errors: bool = True
    artifacts_dir: str = "outputs"
    errors_subdir: str = "validation_errors"
    id_col_candidates: Tuple[str, ...] = ("id", "ID", "Id", "index", "idx")
    error_extra_cols_from_raw: Tuple[str, ...] = ("major_field",)  # 원본에서 같이 보고 싶은 컬럼(있으면 포함)
    error_print_topk: int = 10  # 콘솔에 출력할 "확신이 큰 오답" 샘플 수 (fold별)
    error_save_max_rows: int = 0  # 0이면 오답 전부 저장, >0이면 오답 중 confidence 상위 N개만 저장

    # -----------------
    # Feature importance
    # -----------------
    save_feature_importance: bool = True
    fi_subdir: str = "feature_importance"
    feature_importance_type: str = "FeatureImportance"  # CatBoost get_feature_importance type
    fi_top_n_plot: int = 30


    # -----------------
    # Metrics (threshold-based, error counts)
    # -----------------
    save_fold_metrics: bool = True
    metrics_subdir: str = "metrics"

    # -----------------
    # Major normalization
    # -----------------
    major_standardize_enable: bool = True
    major_standardize_unmapped_keep: bool = True

    # -----------------
    # Feature pruning (sparse / unstable)
    # -----------------
    feature_prune_enable: bool = True
    feature_prune_zero_variance: bool = True
    feature_prune_sparse_zero_ratio_ge: float = 0.995
    feature_prune_sparse_nonzero_leq: int = 3
    feature_prune_binary_min_count: int = 3

    # -----------------
    # Class-weight grid runner
    # -----------------
    class_weight_grid_enable: bool = True
    class_weight_grid_include_auto: bool = True
    class_weight_grid: Tuple[Tuple[float, float], ...] = (
        (1.0, 1.0),
        (1.0, 1.4),
        (1.0, 1.8),
        (1.0, 2.2),
        (1.0, 2.6),
    )
    class_weight_grid_results_name: str = "class_weight_grid.json"

    # -----------------
    # CatBoost hyperparams
    # -----------------
    cat_params: Dict = field(
        default_factory=lambda: dict(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=20000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_strength=1.0,
            subsample=0.8,
            rsm=0.8,  # colsample_bylevel
            auto_class_weights="Balanced",
            od_type="Iter",
            od_wait=600,
            allow_writing_files=False,
            verbose=200,
            thread_count=-1,
            # text_features/text_processing은 CPU만 지원(텍스트 피처를 쓰면 GPU task_type 사용 불가)
            # task_type="GPU",
        )
    )


@dataclass(frozen=True)
class TimeInputStrategy:
    mode: str  # "datetime" | "numeric" | "categorical" | "drop"
    created_cols: Tuple[str, ...] = ()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resolve_id_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _make_is_major_it(df: pd.DataFrame, cfg: CFG) -> None:
    """Create binary feature is_major_it from major_col if available."""
    if cfg.major_col in df.columns:
        s = df[cfg.major_col].astype("string")
        df["is_major_it"] = (
            s.str.contains(cfg.major_it_token, case=False, regex=False, na=False).astype(np.int8)
        )
    else:
        df["is_major_it"] = np.int8(0)


def _infer_time_input_strategy(train_s: pd.Series, cfg: CFG) -> TimeInputStrategy:
    """
    Decide how to treat time_col based on training data only:
      1) datetime parse success ratio >= cfg.time_datetime_ok_ge -> derive datetime features and drop raw time_col.
      2) numeric coerce success ratio >= cfg.time_numeric_ok_ge -> numeric.
      3) else -> categorical string.
    """
    if pd.api.types.is_numeric_dtype(train_s):
        return TimeInputStrategy(mode="numeric")

    dt = pd.to_datetime(train_s, errors="coerce", infer_datetime_format=True)
    dt_ok = float(dt.notna().mean()) if len(dt) else 0.0
    if dt_ok >= cfg.time_datetime_ok_ge:
        return TimeInputStrategy(mode="datetime", created_cols=cfg.time_datetime_created_cols)

    num = pd.to_numeric(train_s, errors="coerce")
    num_ok = float(num.notna().mean()) if len(num) else 0.0
    if num_ok >= cfg.time_numeric_ok_ge:
        return TimeInputStrategy(mode="numeric")

    return TimeInputStrategy(mode="categorical")


def _apply_time_input_strategy(df: pd.DataFrame, strategy: TimeInputStrategy, cfg: CFG) -> None:
    if cfg.time_col not in df.columns:
        return

    if strategy.mode == "numeric":
        if not pd.api.types.is_numeric_dtype(df[cfg.time_col]):
            df[cfg.time_col] = pd.to_numeric(df[cfg.time_col], errors="coerce")
        return

    if strategy.mode == "categorical":
        df[cfg.time_col] = df[cfg.time_col].astype("string")
        return

    if strategy.mode == "datetime":
        dt = pd.to_datetime(df[cfg.time_col], errors="coerce", infer_datetime_format=True)

        df[cfg.time_datetime_created_cols[0]] = dt.dt.hour.astype("float32")
        df[cfg.time_datetime_created_cols[1]] = dt.dt.dayofweek.astype("float32")
        df[cfg.time_datetime_created_cols[2]] = dt.dt.month.astype("float32")
        df[cfg.time_datetime_created_cols[3]] = dt.dt.day.astype("float32")
        df[cfg.time_datetime_created_cols[4]] = (dt.dt.dayofweek >= 5).astype("float32")

        unix_ns = dt.view("int64")
        unix_sec = unix_ns.astype("float64") / 1e9
        unix_sec[dt.isna().to_numpy()] = np.nan
        df[cfg.time_datetime_created_cols[5]] = unix_sec.astype("float32")

        df.drop(columns=[cfg.time_col], inplace=True, errors="ignore")
        return

    if strategy.mode == "drop":
        df.drop(columns=[cfg.time_col], inplace=True, errors="ignore")


def _make_completed_semester_features(df: pd.DataFrame, cfg: CFG) -> None:
    col = cfg.completed_semester_col
    if col not in df.columns:
        df["completed_semester_clean"] = np.nan
        df["completed_semester_is_outlier"] = np.int8(0)
        df["completed_semester_bucket"] = "UNK"
        return

    s = pd.to_numeric(df[col], errors="coerce")
    is_out = (s < 0) | (s > cfg.completed_semester_max)
    s_clean = s.where(~is_out, np.nan)

    df["completed_semester_is_outlier"] = is_out.astype(np.int8)
    df["completed_semester_clean"] = s_clean.astype("float32")

    # 0-2 / 3-4 / 5-6 / 7-8 / 9-20
    bins = [-np.inf, 2, 4, 6, 8, cfg.completed_semester_max]
    labels = ["0-2", "3-4", "5-6", "7-8", "9-20"]
    bucket = pd.cut(s_clean, bins=bins, labels=labels, include_lowest=True)
    df["completed_semester_bucket"] = bucket.astype("string").fillna("UNK")


_SPLIT_RE = re.compile(r"\s*,\s*|\s*/\s*|\s*\|\s*|\s*;\s*|\s*·\s*|\s+및\s+")

def _normalize_none(text: str, none_tokens: Tuple[str, ...]) -> str:
    t = str(text).strip()
    if t == "" or t in none_tokens:
        return ""
    return t

# 문자열 정규화: 공백/구두점 변형을 흡수해서 '없음.' 같은 케이스도 동일 처리
_CLEAN_TOKEN_RE = re.compile(r"[^0-9a-zA-Z가-힣]+")

def _clean_token(v: object) -> str:
    t = str(v) if v is not None else ""
    t = t.strip().lower()
    t = _CLEAN_TOKEN_RE.sub("", t)  # punctuation/whitespace 제거
    return t


def _major_none_clean_set(cfg: CFG) -> set[str]:
    """Canonical none-like tokens used for major columns normalization."""
    extra = {
        "없음요",
        "없습니당",
        "해당 없음",
        "해당없음",
        "무관",
        "미정",
        "x",
        "none",
        "nan",
        "n/a",
        "na",
        "-",
    }
    toks = set(cfg.common_none_tokens) | extra
    out = {_clean_token(t) for t in toks}
    out.add("")
    return out


def _standardize_major_value(v: object, cfg: CFG) -> str:
    """Canonicalize major labels to reduce train-test schema mismatch."""
    raw = str(v) if v is not None else ""
    s = raw.strip()
    if s == "":
        return ""

    c = _clean_token(s)
    if c in _major_none_clean_set(cfg):
        return ""

    if not cfg.major_standardize_enable:
        return s

    # Remove common department suffixes, then map by keyword rules.
    c = re.sub(r"(학과|학부|전공|계열|트랙)$", "", c)
    c = c.replace("데이터사이언스", "데이터")
    c = c.replace("인공지능", "ai")

    def _has_any(text: str, kws: Tuple[str, ...]) -> bool:
        return any(k in text for k in kws)

    if _has_any(c, ("컴퓨터", "소프트웨어", "인공지능", "ai", "데이터", "통계", "산업공학", "산업경영")):
        return "IT(컴퓨터 공학 포함)"
    if _has_any(c, ("경영", "경영정보")):
        return "경영학"
    if _has_any(c, ("경제", "통상", "무역", "회계", "금융")):
        return "경제통상학"
    if _has_any(c, ("수학", "물리", "화학", "생명", "지구", "천문", "과학")):
        return "자연과학"
    if _has_any(c, ("사회", "심리", "행정", "정치", "언론", "커뮤니케이션")):
        return "사회과학"
    if _has_any(c, ("국어", "국문", "영어", "영문", "철학", "사학", "역사", "문헌", "언어")):
        return "인문학"
    if _has_any(c, ("의학", "간호", "약학", "치의", "보건", "수의")):
        return "의약학"
    if _has_any(c, ("미술", "디자인", "음악", "체육", "무용", "예술", "영상")):
        return "예체능"
    if _has_any(c, ("교육",)):
        return "교육학"
    if _has_any(c, ("공학", "기계", "전자", "전기", "화공", "신소재", "토목", "건축", "조선", "항공", "환경")):
        return "공학 (컴퓨터 공학 제외)"

    if cfg.major_standardize_unmapped_keep:
        # Keep normalized surface form to preserve potentially useful signal.
        s2 = re.sub(r"\s+", " ", s).strip()
        return s2
    return "기타"


def _normalize_major_series(s: pd.Series, cfg: CFG) -> pd.Series:
    """Normalize major text by collapsing none-like tokens to empty string."""
    raw = s.fillna("").astype("string").str.strip()
    return raw.map(lambda x: _standardize_major_value(x, cfg)).astype("string")


def _apply_major_standardization_inplace(df: pd.DataFrame, cfg: CFG) -> None:
    """Apply major normalization to raw major columns used directly by model."""
    for col in ("major1_1", "major1_2"):
        if col in df.columns:
            df[col] = _normalize_major_series(df[col], cfg)


def _make_basic_text_presence_features(df: pd.DataFrame, cfg: CFG) -> None:
    """
    base_features에 포함되어 있으나 기존 코드에서 생성되지 않던 피처를 실제로 생성:
      - cert_acq_filled
      - whyBDA_len
      - what_to_gain_len
      - interested_company_n
      - interested_company_len
    """
    # ----- certificate_acquisition filled flag -----
    cert_col = cfg.cert_col
    if cert_col in df.columns:
        x = df[cert_col].fillna("").astype("string")
        clean = x.str.lower().str.replace(cfg.text_ws_regex, "", regex=True)
        clean = clean.str.replace(_CLEAN_TOKEN_RE, "", regex=True)
        none_clean = set(_clean_token(t) for t in (cfg.cert_none_tokens + cfg.common_none_tokens))
        df["cert_acq_filled"] = (~((clean == "") | clean.isin(none_clean))).astype(np.int8)
    else:
        df["cert_acq_filled"] = np.int8(0)

    # ----- whyBDA length -----
    if "whyBDA" in df.columns:
        s = df["whyBDA"].fillna("").astype("string")
        clean = s.str.lower().str.replace(cfg.text_ws_regex, "", regex=True)
        clean = clean.str.replace(_CLEAN_TOKEN_RE, "", regex=True)
        none_clean = set(_clean_token(t) for t in cfg.common_none_tokens)
        norm = s.where(~((clean == "") | clean.isin(none_clean)), "")
        norm = norm.astype("string").str.replace(cfg.text_ws_regex, " ", regex=True).str.strip()
        df["whyBDA_len"] = norm.str.len().astype("float32")
    else:
        df["whyBDA_len"] = np.float32(0.0)

    # ----- what_to_gain length -----
    if "what_to_gain" in df.columns:
        s = df["what_to_gain"].fillna("").astype("string")
        clean = s.str.lower().str.replace(cfg.text_ws_regex, "", regex=True)
        clean = clean.str.replace(_CLEAN_TOKEN_RE, "", regex=True)
        none_clean = set(_clean_token(t) for t in cfg.common_none_tokens)
        norm = s.where(~((clean == "") | clean.isin(none_clean)), "")
        norm = norm.astype("string").str.replace(cfg.text_ws_regex, " ", regex=True).str.strip()
        df["what_to_gain_len"] = norm.str.len().astype("float32")
    else:
        df["what_to_gain_len"] = np.float32(0.0)

    # ----- interested_company token count / length -----
    col = "interested_company"
    if col in df.columns:
        s = df[col].fillna("").astype("string")
        clean = s.str.lower().str.replace(cfg.text_ws_regex, "", regex=True)
        clean = clean.str.replace(_CLEAN_TOKEN_RE, "", regex=True)
        none_clean = set(_clean_token(t) for t in cfg.common_none_tokens)

        norm = s.where(~((clean == "") | clean.isin(none_clean)), "")
        norm_ws = norm.astype("string").str.replace(cfg.text_ws_regex, " ", regex=True).str.strip()
        df["interested_company_len"] = norm_ws.str.len().astype("float32")

        def _count_companies(v: object) -> int:
            t = str(v) if v is not None else ""
            toks = [x.strip() for x in _SPLIT_RE.split(t) if x.strip()]
            uniq: set[str] = set()
            for tok in toks:
                c = _clean_token(tok)
                if c == "" or c in none_clean:
                    continue
                uniq.add(c)
            return len(uniq)

        df["interested_company_n"] = norm.map(_count_companies).astype("int16")
    else:
        df["interested_company_len"] = np.float32(0.0)
        df["interested_company_n"] = np.int16(0)




def _make_interested_company_semantic_features(df: pd.DataFrame, cfg: CFG) -> None:
    """
    interested_company를 의미 기반으로 안정화:
      - ic_primary: 첫 번째(우선순위) 회사 토큰(정규화)
      - ic_has_*: 산업/성향 키워드 플래그
    Top-k/빈도 매핑은 _fit/_apply_ic_primary_mapper에서 수행.
    """
    col = "interested_company"
    none_clean = set(_clean_token(t) for t in cfg.common_none_tokens)

    if col not in df.columns:
        df["ic_primary"] = "UNK"
        df["ic_has_bigtech"] = np.int8(0)
        df["ic_has_finance"] = np.int8(0)
        df["ic_has_game"] = np.int8(0)
        df["ic_has_chaebol"] = np.int8(0)
        return

    s = df[col].fillna("").astype("string")

    # 전체 문자열 정규화(검색용)
    clean_all = s.map(_clean_token)

    def _first_token(v: object) -> str:
        t = str(v) if v is not None else ""
        toks = [x.strip() for x in _SPLIT_RE.split(t) if x.strip()]
        for tok in toks:
            c = _clean_token(tok)
            if c == "" or c in none_clean:
                continue
            return c
        return "UNK"

    df["ic_primary"] = s.map(_first_token).astype("string")

    def _has_any_kw(series_clean: pd.Series, kws: Tuple[str, ...]) -> pd.Series:
        kw_clean = [ _clean_token(k) for k in kws ]
        # 빈 kw 제거
        kw_clean = [k for k in kw_clean if k]
        if not kw_clean:
            return pd.Series(np.zeros(len(series_clean), dtype=np.int8), index=series_clean.index)
        # OR 검색
        pat = "|".join(re.escape(k) for k in kw_clean)
        return series_clean.str.contains(pat, regex=True).fillna(False).astype(np.int8)

    df["ic_has_bigtech"] = _has_any_kw(clean_all, cfg.ic_bigtech_kw)
    df["ic_has_finance"] = _has_any_kw(clean_all, cfg.ic_finance_kw)
    df["ic_has_game"] = _has_any_kw(clean_all, cfg.ic_game_kw)
    df["ic_has_chaebol"] = _has_any_kw(clean_all, cfg.ic_chaebol_kw)


def _fit_ic_primary_mapper(train: pd.DataFrame, cfg: CFG) -> Tuple[set, Dict[str, int]]:
    if "ic_primary" not in train.columns:
        return set(), {}
    s = train["ic_primary"].fillna("UNK").astype("string")
    vc = s.value_counts()
    top = set(vc.head(cfg.ic_primary_topk).index.astype(str).tolist())
    freq = {str(k): int(v) for k, v in vc.to_dict().items()}
    return top, freq


def _apply_ic_primary_mapper(df: pd.DataFrame, top: set, freq: Dict[str, int], cfg: CFG) -> None:
    if "ic_primary" not in df.columns:
        df["ic_primary_topk"] = "UNK"
        df["ic_primary_freq"] = np.float32(0.0)
        df["ic_primary_freq_log1p"] = np.float32(0.0)
        df["ic_primary_is_rare"] = np.int8(1)
        return

    s = df["ic_primary"].fillna("UNK").astype("string")
    df["ic_primary_topk"] = s.where(s.isin(list(top)), other="OTHER").astype("string")

    f = s.map(lambda x: float(freq.get(str(x), 0))).astype("float32")
    df["ic_primary_freq"] = f
    df["ic_primary_freq_log1p"] = np.log1p(f).astype("float32")
    # 희귀: top-k 밖이거나 빈도 매우 낮음
    df["ic_primary_is_rare"] = (~s.isin(list(top))).astype(np.int8)


def _make_time_interaction_features(df: pd.DataFrame, cfg: CFG) -> None:
    """
    FP 클러스터(좋은 신호 + 낮은 time_input)를 분리하기 위한 상호작용(저카디널리티) categorical.
    """
    tb = "time_input_bucket"
    if tb not in df.columns:
        # time bucket이 없으면 생성하지 않음
        return

    def _combo(a: pd.Series, b: pd.Series, prefix: str) -> pd.Series:
        a2 = a.fillna("UNK").astype("string")
        b2 = b.fillna("UNK").astype("string")
        return (a2 + "__" + prefix + b2).astype("string")

    if "whyBDA" in df.columns:
        df["whyBDA_time_bucket"] = _combo(df["whyBDA"], df[tb], "T")
    else:
        df["whyBDA_time_bucket"] = "UNK"

    if "what_to_gain" in df.columns:
        df["what_to_gain_time_bucket"] = _combo(df["what_to_gain"], df[tb], "T")
    else:
        df["what_to_gain_time_bucket"] = "UNK"

    if "re_registration" in df.columns:
        df["rereg_time_bucket"] = _combo(df["re_registration"], df[tb], "T")
    else:
        df["rereg_time_bucket"] = "UNK"

    if "inflow_route" in df.columns:
        df["inflow_time_bucket"] = _combo(df["inflow_route"], df[tb], "T")
    else:
        df["inflow_time_bucket"] = "UNK"


def _make_certificate_features(df: pd.DataFrame, cfg: CFG) -> None:
    """
    certificate_acquisition을 (취득) vs (준비중)으로 분해.
    - 기존 호환을 위해:
        cert_count / has_any_cert / has_adsp / has_sqld / has_info_process 는 "취득" 기준으로 유지
    - 추가:
        cert_preparing_count / has_any_cert_preparing / has_*_preparing
    """
    col = cfg.cert_col
    if col not in df.columns:
        df["cert_count"] = np.int16(0)
        df["cert_preparing_count"] = np.int16(0)

        df["has_any_cert"] = np.int8(0)
        df["has_any_cert_preparing"] = np.int8(0)

        df["has_adsp"] = np.int8(0)
        df["has_adsp_preparing"] = np.int8(0)
        df["has_sqld"] = np.int8(0)
        df["has_sqld_preparing"] = np.int8(0)
        df["has_info_process"] = np.int8(0)
        df["has_info_process_preparing"] = np.int8(0)
        return

    raw = df[col].fillna("").astype("string")

    # "준비중:" 형태는 해당 문자열 전체를 준비중 리스트로 간주
    prep_prefix_re = re.compile(r"^\s*(준비중|준비|예정)\s*[:：]\s*")
    prep_token_re = re.compile(r"(준비중|준비|예정|응시|공부\s*중|계획|취득\s*예정)", re.IGNORECASE)

    none_set = set(cfg.cert_none_tokens) | {"해당없음", "없어요", "없음요"}

    def _norm_cert_name(tok: str) -> str:
        t = str(tok).strip()
        if not t:
            return ""
        # 괄호 내용 제거(대부분 준비중/응시예정 같은 보조 정보)
        t = re.sub(r"\([^\)]*\)", " ", t)
        # 준비/예정 등 보조 단어 제거
        t = re.sub(prep_token_re, " ", t)
        t = t.replace("취득", " ").replace("합격", " ")
        t = re.sub(r"[\-_/·;:：|]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _split_tokens(text: str) -> List[str]:
        t = str(text).replace("\n", " ").strip()
        if not t:
            return []
        return [x.strip() for x in _SPLIT_RE.split(t) if x.strip()]

    acquired_counts = []
    prep_counts = []
    has_adsp_acq = []
    has_sqld_acq = []
    has_info_acq = []
    has_adsp_prep = []
    has_sqld_prep = []
    has_info_prep = []

    for v in raw.tolist():
        t0 = _normalize_none(v, cfg.cert_none_tokens)
        if t0 == "":
            acquired_set: set[str] = set()
            prep_set: set[str] = set()
        else:
            prep_all = bool(prep_prefix_re.match(t0))
            t1 = prep_prefix_re.sub("", t0) if prep_all else t0

            acquired_set = set()
            prep_set = set()

            for token in _split_tokens(t1):
                token = token.strip()
                if not token or token in none_set:
                    continue

                is_prep = prep_all or bool(prep_token_re.search(token))
                name = _norm_cert_name(token)
                if not name or name in none_set:
                    continue

                if is_prep:
                    prep_set.add(name)
                else:
                    acquired_set.add(name)

        acquired_counts.append(len(acquired_set))
        prep_counts.append(len(prep_set))

        acq_low = " ".join(acquired_set).lower()
        prep_low = " ".join(prep_set).lower()

        has_adsp_acq.append(int("adsp" in acq_low))
        has_sqld_acq.append(int("sqld" in acq_low))
        has_info_acq.append(int(("정보처리기사" in acq_low) or ("정보 처리 기사" in acq_low)))

        has_adsp_prep.append(int("adsp" in prep_low))
        has_sqld_prep.append(int("sqld" in prep_low))
        has_info_prep.append(int(("정보처리기사" in prep_low) or ("정보 처리 기사" in prep_low)))

    # (취득) = 기존 피처 이름으로 유지
    df["cert_count"] = np.asarray(acquired_counts, dtype=np.int16)
    df["has_any_cert"] = (df["cert_count"] > 0).astype(np.int8)
    df["has_adsp"] = np.asarray(has_adsp_acq, dtype=np.int8)
    df["has_sqld"] = np.asarray(has_sqld_acq, dtype=np.int8)
    df["has_info_process"] = np.asarray(has_info_acq, dtype=np.int8)

    # (준비중) = 추가 피처
    df["cert_preparing_count"] = np.asarray(prep_counts, dtype=np.int16)
    df["has_any_cert_preparing"] = (df["cert_preparing_count"] > 0).astype(np.int8)
    df["has_adsp_preparing"] = np.asarray(has_adsp_prep, dtype=np.int8)
    df["has_sqld_preparing"] = np.asarray(has_sqld_prep, dtype=np.int8)
    df["has_info_process_preparing"] = np.asarray(has_info_prep, dtype=np.int8)


def _make_previous_class_agg_features(df: pd.DataFrame) -> None:
    """
    previous_class_3~8을 개별 컬럼이 아닌 집계 피처로 변환.
    생성 피처:
      - has_prev_class: 과거 수강 경험 유무(유효 코드 기준)
      - prev_class_code_nunique: 과거 수강 코드(0001~) 유니크 개수
      - prev_class_gen_count: 유효 코드가 존재하는 previous_class_* 컬럼 개수
      - prev_class_latest_gen: 가장 최근(previous_class_8 > ... > previous_class_3) 유효 코드가 나온 기수 번호
      - prev_class_latest_code: 가장 최근 기수에서의 첫 번째 코드(카테고리)
    """
    prev_cols = [c for c in df.columns if c.startswith("previous_class_")]
    if not prev_cols:
        df["has_prev_class"] = np.int8(0)
        df["prev_class_code_nunique"] = np.int16(0)
        df["prev_class_gen_count"] = np.int16(0)
        df["prev_class_latest_gen"] = np.int16(0)
        df["prev_class_latest_code"] = "NONE"
        return

    none_tokens = {"", "해당없음", "없음", "무", ".", "없습니다"}
    code_re = re.compile(r"^\s*(\d{4})\s*[:：]")

    def _gen_num(col: str) -> int:
        m = re.search(r"(\d+)$", col)
        return int(m.group(1)) if m else -1

    prev_cols_sorted = sorted(prev_cols, key=_gen_num)          # 3 -> 8
    prev_cols_sorted_desc = list(reversed(prev_cols_sorted))    # 8 -> 3

    def _extract_codes_from_cell(x) -> List[str]:
        if pd.isna(x):
            return []
        s = str(x).strip()
        if not s:
            return []
        parts = [p.strip() for p in _SPLIT_RE.split(s) if p.strip()]  # 콤마/슬래시 등 모두 분리
        codes: List[str] = []
        for p in parts:
            if p in none_tokens:
                continue
            m = code_re.match(p)
            if m:
                codes.append(m.group(1))
            else:
                m2 = re.match(r"^\s*(\d{4})\b", p)
                if m2:
                    codes.append(m2.group(1))

        # unique preserve order
        seen = set()
        out = []
        for c in codes:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    has_prev = np.zeros(len(df), dtype=np.int8)
    code_nuniq = np.zeros(len(df), dtype=np.int16)
    gen_count = np.zeros(len(df), dtype=np.int16)
    latest_gen = np.zeros(len(df), dtype=np.int16)
    latest_code = np.empty(len(df), dtype=object)
    latest_code[:] = "NONE"

    for i in range(len(df)):
        codes_set: set[str] = set()
        gen_with_code: set[int] = set()

        found_latest = False
        for c in prev_cols_sorted_desc:
            g = _gen_num(c)
            codes = _extract_codes_from_cell(df.iloc[i][c])
            if codes:
                if not found_latest:
                    latest_gen[i] = g
                    latest_code[i] = codes[0]
                    found_latest = True
                gen_with_code.add(g)
                for cd in codes:
                    codes_set.add(cd)

        if codes_set:
            has_prev[i] = 1
            code_nuniq[i] = len(codes_set)
            gen_count[i] = len(gen_with_code)

    df["has_prev_class"] = has_prev
    df["prev_class_code_nunique"] = code_nuniq
    df["prev_class_gen_count"] = gen_count
    df["prev_class_latest_gen"] = latest_gen
    df["prev_class_latest_code"] = pd.Series(latest_code, index=df.index).astype("string")


def _make_major_split_features(df: pd.DataFrame, cfg: CFG) -> None:
    def _is_data_series(s: pd.Series) -> pd.Series:
        s = s.fillna("").astype("string")
        return (
            s.str.contains("IT", case=False, regex=False, na=False)
            | s.str.contains("컴퓨터", case=False, regex=False, na=False)
            | s.str.contains("소프트웨어", case=False, regex=False, na=False)
            | s.str.contains("인공지능", case=False, regex=False, na=False)
            | s.str.contains("AI", case=False, regex=False, na=False)
            | s.str.contains("데이터", case=False, regex=False, na=False)
            | s.str.contains("통계", case=False, regex=False, na=False)
            | s.str.contains("산업", case=False, regex=False, na=False)
            | s.str.contains("산업경영", case=False, regex=False, na=False)
        )

    def _add_freq_topk(col: str, prefix: str) -> None:
        """전공 문자열을 빈도/TopK로 안정화.

        - {prefix}_freq: 빈도
        - {prefix}_freq_log1p: log1p(freq)
        - {prefix}_is_rare: 희소(<= cfg.major_rare_threshold)
        - {prefix}_topk: 상위 cfg.major_topk만 유지, 나머지 OTHER
        """
        if col not in df.columns:
            df[f"{prefix}_freq"] = np.int16(0)
            df[f"{prefix}_freq_log1p"] = np.float32(0.0)
            df[f"{prefix}_is_rare"] = np.int8(0)
            df[f"{prefix}_topk"] = pd.Series(["UNK"] * len(df), index=df.index, dtype="string")
            return

        s = _normalize_major_series(df[col], cfg)
        s = s.replace({"": "MISSING"})
        vc = s.value_counts(dropna=False)

        freq_map = vc.to_dict()
        f = s.map(freq_map).fillna(0).astype("int32")

        df[f"{prefix}_freq"] = f.astype(np.int16)
        df[f"{prefix}_freq_log1p"] = np.log1p(f.to_numpy()).astype(np.float32)
        df[f"{prefix}_is_rare"] = (f <= cfg.major_rare_threshold).astype(np.int8)

        top = set(vc.head(cfg.major_topk).index.astype(str).tolist())
        s_str = s.astype(str)
        df[f"{prefix}_topk"] = s_str.where(s_str.isin(top), "OTHER").astype("string")

    # major1_1 / major1_2 IT 플래그
    if "major1_1" in df.columns:
        df["major1_1_is_data"] = _is_data_series(_normalize_major_series(df["major1_1"], cfg)).astype(np.int8)
    else:
        df["major1_1_is_data"] = np.int8(0)

    if "major1_2" in df.columns:
        df["major1_2_is_data"] = _is_data_series(_normalize_major_series(df["major1_2"], cfg)).astype(np.int8)
    else:
        df["major1_2_is_data"] = np.int8(0)

    # major1_1 / major1_2 안정화(빈도/TopK)
    _add_freq_topk("major1_1", "major1_1")
    _add_freq_topk("major1_2", "major1_2")

    # -----------------
    # Major pair (major1_1 x major1_2) frequency features
    # -----------------
    major1_1_pair = df.get("major1_1_topk", df.get("major1_1", "UNK")).astype("string").fillna("UNK")
    major1_2_pair = df.get("major1_2_topk", df.get("major1_2", "UNK")).astype("string").fillna("UNK")
    df["major_pair"] = (major1_1_pair + cfg.major_pair_sep + major1_2_pair).astype("string").fillna("UNK")
    pair_freq = df["major_pair"].value_counts(dropna=False).to_dict()
    df["major_pair_freq"] = df["major_pair"].map(pair_freq).fillna(0).astype("float32")
    df["major_pair_freq_log1p"] = np.log1p(df["major_pair_freq"]).astype("float32")


    # 복수전공 여부
    if "major type" in df.columns:
        mt = df["major type"].fillna("").astype("string")
        df["is_double_major"] = mt.str.contains("복수|다중|이중", regex=True, na=False).astype(np.int8)
    else:
        df["is_double_major"] = np.int8(0)

    # major1_2 결측/일관성 피처
    if "major1_2" in df.columns:
        m2 = _normalize_major_series(df["major1_2"], cfg)
        df["has_major2"] = (m2 != "").astype(np.int8)
    else:
        df["has_major2"] = np.int8(0)

    # is_double_major (major type 기반)와 major1_2 존재 여부의 모순/결측을 명시적으로 피처화
    df["major2_missing_but_double_major"] = ((df["is_double_major"] == 1) & (df["has_major2"] == 0)).astype(np.int8)
    df["major2_present_but_single_major"] = ((df["is_double_major"] == 0) & (df["has_major2"] == 1)).astype(np.int8)


def _extract_codes(text: str) -> List[str]:
    """멀티선택 셀에서 코드(A,B,...)를 안정적으로 추출.

    예:
      - 'A. 데이터 분석가, B. 데이터 엔지니어'
      - 'A. ... / C. ...'
      - 'A. ...; B. ...'
    """
    if text is None:
        return []
    s = str(text).strip()
    if not s:
        return []
    if s.lower() in ("nan", "none"):
        return []

    parts = re.split(r"[\n,;/|]+", s)
    codes: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 가장 흔한 형태: 'A.'
        m = re.match(r"\s*([A-Z])\.", p)
        if m:
            codes.append(m.group(1))
            continue
        # fallback: 문자열 내부에 'A.'가 섞인 경우
        for m2 in re.finditer(r"([A-Z])\.", p):
            codes.append(m2.group(1))

    # unique preserve order
    seen = set()
    out: List[str] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _make_desired_job_features(df: pd.DataFrame, cfg: CFG) -> None:
    """desired_job / desired_job_except_data 멀티선택 피처.

    - 기존: 선택 개수(count)
    - 추가: 코드별 멀티핫 (A~J / A~H)
    """
    specs = [
        (cfg.desired_job_col, "desired_job", "desired_job_n", cfg.desired_job_codes),
        (cfg.desired_job_except_col, "desired_job_except", "desired_job_except_n", cfg.desired_job_except_codes),
    ]

    for col, prefix, out_count, codes in specs:
        if col not in df.columns:
            df[out_count] = np.int16(0)
            for code in codes:
                df[f"{prefix}_has_{code}"] = np.int8(0)
            continue

        raw = df[col].fillna("").astype("string")
        # 코드 추출 (예: 'A. ... , B. ...')
        code_lists = [_extract_codes(v) for v in raw.tolist()]

        df[out_count] = np.asarray([len(x) for x in code_lists], dtype=np.int16)

        # 멀티핫
        code_sets = [set(x) for x in code_lists]
        for code in codes:
            df[f"{prefix}_has_{code}"] = np.asarray([1 if code in s else 0 for s in code_sets], dtype=np.int8)


    # -----------------
    # Desired job composition (data-only vs mixed)
    # NOTE: codes A-D are treated as "data core" (adjust if codebook differs)
    # -----------------
    if "desired_job_n" in df.columns:
        data_codes = ["A", "B", "C", "D"]
        existing = [c for c in data_codes if f"desired_job_has_{c}" in df.columns]
        if existing:
            df["desired_job_data_n"] = df[[f"desired_job_has_{c}" for c in existing]].sum(axis=1).astype(np.int16)
        else:
            df["desired_job_data_n"] = np.int16(0)

        df["desired_job_nondata_n"] = (df["desired_job_n"].astype(np.int16) - df["desired_job_data_n"]).clip(lower=0).astype(np.int16)
        df["desired_job_is_data_only"] = ((df["desired_job_data_n"] > 0) & (df["desired_job_nondata_n"] == 0)).astype(np.int8)
        df["desired_job_is_nondata_only"] = ((df["desired_job_data_n"] == 0) & (df["desired_job_nondata_n"] > 0)).astype(np.int8)
        df["desired_job_is_mixed"] = ((df["desired_job_data_n"] > 0) & (df["desired_job_nondata_n"] > 0)).astype(np.int8)
    else:
        df["desired_job_data_n"] = np.int16(0)
        df["desired_job_nondata_n"] = np.int16(0)
        df["desired_job_is_data_only"] = np.int8(0)
        df["desired_job_is_nondata_only"] = np.int8(0)
        df["desired_job_is_mixed"] = np.int8(0)



def _make_incumbents_reason_text_stats(df: pd.DataFrame, cfg: CFG) -> None:
    col = "incumbents_lecture_scale_reason"
    if col not in df.columns:
        df["inc_lecture_reason_len"] = np.float32(0)
        df["inc_lecture_reason_words"] = np.float32(0)
        df["inc_lecture_reason_has_job_kw"] = np.int8(0)
        return

    s = df[col].fillna("").astype("string").str.replace(cfg.text_ws_regex, " ", regex=True).str.strip()
    df["inc_lecture_reason_len"] = s.str.len().astype("float32")
    df["inc_lecture_reason_words"] = s.str.split(" ").map(lambda x: len([t for t in x if t])).astype("float32")

    low = s.str.lower()
    kw_hit = np.zeros(len(s), dtype=np.int8)
    for kw in cfg.job_keywords:
        kw_hit |= low.str.contains(kw.lower(), regex=False, na=False).to_numpy().astype(np.int8)
    df["inc_lecture_reason_has_job_kw"] = kw_hit



def _make_signature_rule_features(df: pd.DataFrame, cfg: CFG) -> None:
    """Rule-mined 'signature' features to help CatBoost learn FP/FN failure modes.

    All outputs are binary (int8) or small integer scores and safe-default to 0 when source columns are missing.

    v9 fix:
      - reorder computations so that derived columns exist before being referenced
      - expose motivation_score/motivation_geq and low-engagement exception flags
    """
    n = len(df)

    def _get_str(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series([""] * n, index=df.index, dtype="string")
        return df[col].fillna("").astype("string")

    def _get_num(col: str, default: float = 0.0) -> pd.Series:
        if col not in df.columns:
            return pd.Series([default] * n, index=df.index, dtype="float32")
        return pd.to_numeric(df[col], errors="coerce").fillna(default).astype("float32")

    # axes: 재등록/이전기수 경험/시간/학기/유입/자격증/전공2 존재
    rereg = _get_str("re_registration")
    rereg_yes = rereg.str.contains("예", regex=False, na=False)
    rereg_no = rereg.str.contains("아니", regex=False, na=False)

    inflow = _get_str("inflow_route")
    inflow_member = inflow.str.contains("기존 학회원|운영진|지인 추천", regex=True, na=False)

    time_input = _get_num("time_input", default=np.nan)
    time_le3 = (time_input <= 3).fillna(False)

    sem = _get_num("completed_semester_clean", default=np.nan)
    sem_le8 = (sem <= 8).fillna(False)

    prev_cnt = _get_num("prev_class_gen_count", default=0.0)
    prev_gt0 = prev_cnt > 0

    prev_latest_code = _get_str("prev_class_latest_code").str.strip()
    prev_none = prev_latest_code.eq("NONE") | prev_latest_code.eq("")

    has_any_cert = _get_num("has_any_cert", default=0.0)
    no_cert = has_any_cert <= 0

    has_adsp = _get_num("has_adsp", default=0.0)
    no_adsp = has_adsp <= 0

    has_major2 = (_get_num("has_major2", default=0.0) > 0)
    major2_missing = ~has_major2

    # -----------------
    # Engagement split (used to "relax" overly-harsh FN rules)
    # -----------------
    eng_opt = _get_num("eng_filled_optional", default=0.0)
    eng_ge = (eng_opt >= float(cfg.sig_eng_optional_geq)).fillna(False)
    df["eng_optional_ge2"] = eng_ge.astype(np.int8)

    # -----------------
    # Motivation score (rule exceptions for low-engagement users)
    # -----------------
    mot = pd.Series(0, index=df.index, dtype=np.int16)

    time_raw = pd.to_numeric(df.get(cfg.time_col, np.nan), errors="coerce")
    sem_raw = pd.to_numeric(df.get(cfg.completed_semester_col, np.nan), errors="coerce")

    mot += (time_raw >= cfg.sig_time_geq).fillna(False).astype(np.int16)
    mot += (sem_raw >= cfg.sig_semester_geq).fillna(False).astype(np.int16)

    # fewer desired jobs (<=1) tends to be a stronger commitment signal
    if "desired_job_n" in df.columns:
        mot += (pd.to_numeric(df["desired_job_n"], errors="coerce").fillna(0) <= 1).astype(np.int16)

    # certificates: any + SQLD is a strong differentiator inside some low-eng groups
    if "has_any_cert" in df.columns:
        mot += pd.to_numeric(df["has_any_cert"], errors="coerce").fillna(0).clip(0, 1).astype(np.int16)
    if "has_sqld" in df.columns:
        mot += pd.to_numeric(df["has_sqld"], errors="coerce").fillna(0).clip(0, 1).astype(np.int16)

    # major signal
    if "is_major_it" in df.columns:
        mot += pd.to_numeric(df["is_major_it"], errors="coerce").fillna(0).clip(0, 1).astype(np.int16)

    # contest / re-registration: treat non-empty strings as participation evidence
    for _c in ["contest_participation", "contest_award", "idea_contest"]:
        if _c in df.columns:
            mot += df[_c].astype("string").str.len().fillna(0).gt(0).astype(np.int16)

    if "re_registration" in df.columns:
        mot += (
            df["re_registration"]
            .astype("string")
            .str.contains(r"(재등록|yes|y|true|1|o|예|네)", case=False, regex=True)
            .fillna(False)
            .astype(np.int16)
        )

    df["motivation_score"] = mot.clip(0, 10).astype(np.int8)
    df["motivation_geq"] = (df["motivation_score"] >= cfg.sig_motivation_geq).astype(np.int8)

    # -----------------
    # FP signatures
    # -----------------
    df["sig_fp_rereg_yes_time_le3"] = (rereg_yes & time_le3).astype(np.int8)
    df["sig_fp_rereg_yes_sem_le8"] = (rereg_yes & sem_le8).astype(np.int8)
    df["sig_fp_prevclass_time_le3"] = (prev_gt0 & time_le3).astype(np.int8)
    df["sig_fp_prevclass_sem_le8"] = (prev_gt0 & sem_le8).astype(np.int8)
    df["sig_fp_inflow_member_time_le3"] = (inflow_member & time_le3).astype(np.int8)

    # -----------------
    # FN signatures (base)
    # -----------------
    base_new_no_cert = (rereg_no & no_cert)
    df["sig_fn_new_no_cert"] = base_new_no_cert.astype(np.int8)
    df["sig_fn_new_no_cert_loweng"] = (base_new_no_cert & (~eng_ge)).astype(np.int8)
    df["sig_fn_new_no_cert_higheng"] = (base_new_no_cert & eng_ge).astype(np.int8)

    base_new_major2_missing = (rereg_no & major2_missing)
    df["sig_fn_new_major2_missing"] = base_new_major2_missing.astype(np.int8)
    df["sig_fn_new_major2_missing_loweng"] = (base_new_major2_missing & (~eng_ge)).astype(np.int8)
    df["sig_fn_new_major2_missing_higheng"] = (base_new_major2_missing & eng_ge).astype(np.int8)

    base_prevnone_no_cert = (prev_none & no_cert)
    df["sig_fn_prevnone_no_cert"] = base_prevnone_no_cert.astype(np.int8)
    df["sig_fn_prevnone_no_cert_loweng"] = (base_prevnone_no_cert & (~eng_ge)).astype(np.int8)
    df["sig_fn_prevnone_no_cert_higheng"] = (base_prevnone_no_cert & eng_ge).astype(np.int8)

    base_prevnone_major2_missing = (prev_none & major2_missing)
    df["sig_fn_prevnone_major2_missing"] = base_prevnone_major2_missing.astype(np.int8)
    df["sig_fn_prevnone_major2_missing_loweng"] = (base_prevnone_major2_missing & (~eng_ge)).astype(np.int8)
    df["sig_fn_prevnone_major2_missing_higheng"] = (base_prevnone_major2_missing & eng_ge).astype(np.int8)

    base_major2_missing_no_adsp = (major2_missing & no_adsp)
    df["sig_fn_major2_missing_no_adsp"] = base_major2_missing_no_adsp.astype(np.int8)
    df["sig_fn_major2_missing_no_adsp_loweng"] = (base_major2_missing_no_adsp & (~eng_ge)).astype(np.int8)
    df["sig_fn_major2_missing_no_adsp_higheng"] = (base_major2_missing_no_adsp & eng_ge).astype(np.int8)

    # -----------------
    # Low-engagement exception flags (완화/예외처리)
    # -----------------
    df["sig_fn_new_no_cert_loweng_motivated"] = (
        (df["sig_fn_new_no_cert_loweng"] == 1) & (df["motivation_geq"] == 1)
    ).astype(np.int8)
    df["sig_fn_prevnone_no_cert_loweng_motivated"] = (
        (df["sig_fn_prevnone_no_cert_loweng"] == 1) & (df["motivation_geq"] == 1)
    ).astype(np.int8)
    df["sig_fn_major2_missing_no_adsp_loweng_motivated"] = (
        (df["sig_fn_major2_missing_no_adsp_loweng"] == 1) & (df["motivation_geq"] == 1)
    ).astype(np.int8)

    # certificate-driven exceptions specifically for major2-missing group (empirically strong)
    if "has_sqld" in df.columns:
        df["sig_fn_major2_missing_no_adsp_loweng_has_sqld"] = (
            (df["sig_fn_major2_missing_no_adsp_loweng"] == 1) & (df["has_sqld"] == 1)
        ).astype(np.int8)
    else:
        df["sig_fn_major2_missing_no_adsp_loweng_has_sqld"] = np.int8(0)

    if "has_any_cert" in df.columns:
        df["sig_fn_major2_missing_no_adsp_loweng_has_any_cert"] = (
            (df["sig_fn_major2_missing_no_adsp_loweng"] == 1) & (df["has_any_cert"] == 1)
        ).astype(np.int8)
    else:
        df["sig_fn_major2_missing_no_adsp_loweng_has_any_cert"] = np.int8(0)

    # -----------------
    # Aggregated risk scores (soft signals)
    # -----------------
    df["sig_fn_risk_score"] = (
        df["sig_fn_new_no_cert"].astype(np.int16)
        + df["sig_fn_prevnone_no_cert"].astype(np.int16)
        + df["sig_fn_major2_missing_no_adsp"].astype(np.int16)
    ).clip(0, 10).astype(np.int8)

    df["sig_fn_risk_score_loweng"] = (
        df["sig_fn_new_no_cert_loweng"].astype(np.int16)
        + df["sig_fn_prevnone_no_cert_loweng"].astype(np.int16)
        + df["sig_fn_major2_missing_no_adsp_loweng"].astype(np.int16)
    ).clip(0, 10).astype(np.int8)



def _is_nonempty(v: object, none_tokens: Tuple[str, ...]) -> int:
    try:
        if pd.isna(v):
            return 0
    except Exception:
        pass
    s = str(v).strip()
    if s == "":
        return 0
    if s in none_tokens:
        return 0
    if s.lower() in ("nan", "none"):
        return 0
    return 1


def _make_engagement_features(df: pd.DataFrame, raw_cols: List[str], cfg: CFG, id_col: Optional[str]) -> None:
    # raw_cols: “원본 설문 컬럼 목록” (파생 feature 포함 금지)
    none_tokens = cfg.common_none_tokens
    exclude = set([cfg.target])
    if id_col is not None:
        exclude.add(id_col)

    raw_eff = [c for c in raw_cols if (c in df.columns and c not in exclude)]
    denom_all = max(1, len(raw_eff))

    # 전체 원본 컬럼 중 “실질 입력” 개수
    filled_all = np.zeros(len(df), dtype=np.int16)
    for j, c in enumerate(raw_eff):
        col = df[c]
        # vectorize 대신 안전성 우선
        filled_all += col.map(lambda x: _is_nonempty(x, none_tokens)).to_numpy(dtype=np.int16)

    df["eng_filled_all"] = filled_all
    df["eng_filled_all_ratio"] = (filled_all / float(denom_all)).astype("float32")

    # 옵션성 컬럼만
    opt_cols = [c for c in cfg.engagement_optional_cols if c in df.columns]
    denom_opt = max(1, len(opt_cols))

    if len(opt_cols) == 0:
        df["eng_filled_optional"] = np.int16(0)
        df["eng_filled_optional_ratio"] = np.float32(0.0)
        df["eng_prev_class_filled"] = np.int16(0)
        return

    filled_opt = np.zeros(len(df), dtype=np.int16)
    prev_cols = [c for c in opt_cols if c.startswith("previous_class_")]
    prev_filled = np.zeros(len(df), dtype=np.int16)

    for c in opt_cols:
        hit = df[c].map(lambda x: _is_nonempty(x, none_tokens)).to_numpy(dtype=np.int16)
        filled_opt += hit
        if c in prev_cols:
            prev_filled += hit

    df["eng_filled_optional"] = filled_opt
    df["eng_filled_optional_ratio"] = (filled_opt / float(denom_opt)).astype("float32")
    df["eng_prev_class_filled"] = prev_filled


def _make_expected_domain_features(df: pd.DataFrame, cfg: CFG) -> None:
    if "expected_domain" not in df.columns:
        df["expected_domain_code_count"] = np.int8(0)
        df["expected_domain_primary_code"] = "UNK"
        for code in cfg.expected_domain_codes_focus:
            df[f"expected_domain_has_{code}"] = np.int8(0)
        return

    s = df["expected_domain"].fillna("").astype("string")

    def _extract_codes(text: str) -> List[str]:
        toks = [x.strip() for x in _SPLIT_RE.split(str(text)) if x.strip()]
        codes = []
        for t in toks:
            m = re.match(r"^([A-Z])\s*\.", t)
            if m:
                codes.append(m.group(1))
        # unique-preserve-order
        seen = set()
        out = []
        for c in codes:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    codes_list = s.apply(_extract_codes)

    df["expected_domain_code_count"] = codes_list.apply(len).astype(np.int8)
    df["expected_domain_primary_code"] = codes_list.apply(lambda xs: xs[0] if len(xs) > 0 else "UNK").astype("string")

    for code in cfg.expected_domain_codes_focus:
        df[f"expected_domain_has_{code}"] = codes_list.apply(lambda xs: int(code in xs)).astype(np.int8)


def _make_desired_certificate_features(df: pd.DataFrame, cfg: CFG) -> None:
    # desired_certificate는 원문 cardinality가 크므로 “대표 키워드 + 개수”로 구조화
    if "desired_certificate" not in df.columns:
        df["desired_cert_count"] = np.int8(0)
        df["desired_has_bigdata"] = np.int8(0)
        df["desired_has_sqld"] = np.int8(0)
        df["desired_has_adsp"] = np.int8(0)
        df["desired_has_info_process"] = np.int8(0)
        df["desired_has_tableau"] = np.int8(0)
        df["desired_has_google_analyst"] = np.int8(0)
        return

    none_tokens = set(cfg.common_none_tokens)
    raw = df["desired_certificate"].fillna("").astype("string")

    def _clean_tokens(text: str) -> List[str]:
        t = str(text).strip()
        if t == "" or t in none_tokens:
            return []
        toks = [x.strip() for x in _SPLIT_RE.split(t) if x.strip()]
        out = []
        for x in toks:
            if x in none_tokens:
                continue
            x = re.sub(r"\([^\)]*\)", " ", x)
            x = re.sub(r"\s+", " ", x).strip()
            # 너무 긴 자유문장(노이즈) 컷
            if len(x) > 40:
                continue
            out.append(x)
        return list(dict.fromkeys(out))  # unique preserve order

    toks = raw.apply(_clean_tokens)
    df["desired_cert_count"] = toks.apply(len).astype(np.int8)

    low = raw.str.lower()
    df["desired_has_bigdata"] = low.str.contains("빅데이터", regex=False).astype(np.int8)
    df["desired_has_sqld"] = low.str.contains("sqld", regex=False).astype(np.int8)
    df["desired_has_adsp"] = low.str.contains("adsp", regex=False).astype(np.int8)
    df["desired_has_info_process"] = (low.str.contains("정보처리기사", regex=False) | low.str.contains("정보 처리 기사", regex=False)).astype(np.int8)
    df["desired_has_tableau"] = (low.str.contains("태블로", regex=False) | low.str.contains("tableau", regex=False)).astype(np.int8)
    df["desired_has_google_analyst"] = low.str.contains("구글 애널리", regex=False).astype(np.int8)


def _fit_school1_mapper(train: pd.DataFrame, cfg: CFG) -> Tuple[set, Dict[str, int]]:
    if "school1" not in train.columns:
        return set(), {}
    s = train["school1"].fillna("UNK").astype("string")
    vc = s.value_counts()
    top = set(vc.head(cfg.school1_topk).index.astype(str).tolist())
    freq = {str(k): int(v) for k, v in vc.to_dict().items()}
    return top, freq


def _apply_school1_mapper(df: pd.DataFrame, top: set, freq: Dict[str, int], cfg: CFG) -> None:
    if "school1" not in df.columns:
        df["school1_topk"] = "UNK"
        df["school1_freq"] = np.int16(0)
        df["school1_freq_log1p"] = np.float32(0.0)
        df["school1_is_rare"] = np.int8(0)
        return

    s = df["school1"].fillna("UNK").astype("string")
    s_str = s.astype(str)

    df["school1_topk"] = s_str.where(s_str.isin(top), "OTHER").astype("string")
    f = s_str.map(freq).fillna(0).astype("int16")
    df["school1_freq"] = f
    df["school1_freq_log1p"] = np.log1p(f.astype("float32"))
    df["school1_is_rare"] = (f <= cfg.school1_rare_leq).astype("int8")


def _make_onedayclass_multihot(df: pd.DataFrame, cfg: CFG) -> None:
    col = "onedayclass_topic"
    if col not in df.columns:
        for name, _ in cfg.oneday_rules:
            df[f"oneday_has_{name}"] = np.int8(0)
        df["oneday_topic_count"] = np.int16(0)
        return

    s = df[col].fillna("").astype("string")
    low = s.str.lower()

    total = np.zeros(len(df), dtype=np.int16)
    for name, kws in cfg.oneday_rules:
        hit = np.zeros(len(df), dtype=np.int8)
        for kw in kws:
            hit |= low.str.contains(kw.lower(), regex=False, na=False).to_numpy(dtype=np.int8)
        df[f"oneday_has_{name}"] = hit
        total += hit.astype(np.int16)

    df["oneday_topic_count"] = total



# ============================================================
# Train-fitted vocab multihot for selection-type columns
#   - major_field (multi-select)
#   - whyBDA (single-select -> one-hot)
#   - certificate_acquisition (acquired vs preparing -> multi-hot)
#   - desired_certificate (multi-select -> multi-hot)
#   - interested_company (multi-select -> multi-hot of top companies)
# ============================================================
def _split_multiselect_tokens(v: object) -> List[str]:
    """Split a multi-select cell into tokens, robust to commas inside parentheses.

    Delimiters (when not inside (), [], {}):
      - comma / slash / pipe / semicolon / middle dot
      - Korean '및' (surrounded by whitespace) is normalized to comma.
    """
    try:
        if pd.isna(v):
            return []
    except Exception:
        pass
    s = str(v).replace("\n", " ").strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    s = re.sub(r"\s+및\s+", ",", s)

    delims = {",", "，", "/", "|", ";", "·"}
    openers = {"(": ")", "[": "]", "{": "}"}
    closers = set(openers.values())
    depth = 0
    buf: List[str] = []
    out: List[str] = []

    for ch in s:
        if ch in openers:
            depth += 1
        elif ch in closers:
            depth = max(0, depth - 1)

        if depth == 0 and ch in delims:
            tok = "".join(buf).strip()
            if tok:
                out.append(tok)
            buf = []
        else:
            buf.append(ch)

    last = "".join(buf).strip()
    if last:
        out.append(last)

    return [t.strip() for t in out if t.strip()]


def _safe_feat_key(clean_token: str, max_len: int = 40) -> str:
    """Make a stable, reasonably short suffix for a feature column name."""
    t = str(clean_token)
    if len(t) <= max_len:
        return t
    h = hashlib.md5(t.encode("utf-8")).hexdigest()[:8]
    return f"{t[:max_len-9]}_{h}"


def _fit_token_vocab(
    train_s: pd.Series,
    *,
    cfg: CFG,
    topk: int,
    min_freq: int,
    alias_clean_map: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """Fit top-k token vocabulary on train only.

    Returns:
      vocab_clean: list of cleaned tokens kept (sorted by freq desc)
      clean_to_key: mapping cleaned_token -> safe feature key (for column names)
    """
    from collections import Counter

    alias_clean_map = alias_clean_map or {}
    none_clean = set(_clean_token(t) for t in cfg.common_none_tokens)

    c = Counter()
    for v in train_s.tolist():
        for tok in _split_multiselect_tokens(v):
            if len(tok) > cfg.multihot_token_max_len:
                continue
            ct = _clean_token(tok)
            if not ct or ct in none_clean:
                continue
            ct = alias_clean_map.get(ct, ct)
            if not ct or ct in none_clean:
                continue
            c[ct] += 1

    items = [(tok, cnt) for tok, cnt in c.items() if cnt >= int(min_freq)]
    items.sort(key=lambda x: (-x[1], x[0]))

    vocab_clean = [tok for tok, _ in items[: int(topk)]]
    clean_to_key = {tok: _safe_feat_key(tok) for tok in vocab_clean}
    return vocab_clean, clean_to_key


def _apply_token_multihot(
    df: pd.DataFrame,
    *,
    col: str,
    prefix: str,
    vocab_clean: List[str],
    clean_to_key: Dict[str, str],
    cfg: CFG,
    alias_clean_map: Optional[Dict[str, str]] = None,
) -> None:
    """Apply fitted vocab to df and create multi-hot columns + counts."""
    alias_clean_map = alias_clean_map or {}
    none_clean = set(_clean_token(t) for t in cfg.common_none_tokens)
    vocab_set = set(vocab_clean)

    n = len(df)
    token_sets: List[set[str]] = []
    for v in df.get(col, pd.Series([""] * n, index=df.index)).tolist():
        toks = _split_multiselect_tokens(v)
        sset: set[str] = set()
        for tok in toks:
            if len(tok) > cfg.multihot_token_max_len:
                continue
            ct = _clean_token(tok)
            if not ct or ct in none_clean:
                continue
            ct = alias_clean_map.get(ct, ct)
            if not ct or ct in none_clean:
                continue
            sset.add(ct)
        token_sets.append(sset)

    df[f"{prefix}nunique"] = np.asarray([len(s) for s in token_sets], dtype=np.int16)
    df[f"{prefix}oov_n"] = np.asarray([len([t for t in s if t not in vocab_set]) for s in token_sets], dtype=np.int16)

    # Multi-hot columns for vocab
    for ct in vocab_clean:
        key = clean_to_key[ct]
        out_col = f"{prefix}has_{key}"
        df[out_col] = np.asarray([1 if ct in s else 0 for s in token_sets], dtype=np.int8)


def _fit_apply_major_field_multihot(train: pd.DataFrame, test: pd.DataFrame, cfg: CFG) -> None:
    col = "major_field"
    if col not in train.columns and col not in test.columns:
        return

    # common typo aliasing
    alias = {
        _clean_token("자연고학"): _clean_token("자연과학"),
    }

    vocab, key_map = _fit_token_vocab(
        train.get(col, pd.Series([], dtype="string")),
        cfg=cfg,
        topk=cfg.major_field_topk,
        min_freq=cfg.multihot_min_freq,
        alias_clean_map=alias,
    )
    _apply_token_multihot(train, col=col, prefix="major_field_", vocab_clean=vocab, clean_to_key=key_map, cfg=cfg, alias_clean_map=alias)
    _apply_token_multihot(test,  col=col, prefix="major_field_", vocab_clean=vocab, clean_to_key=key_map, cfg=cfg, alias_clean_map=alias)


def _fit_apply_whyBDA_onehot(train: pd.DataFrame, test: pd.DataFrame, cfg: CFG) -> None:
    col = "whyBDA"
    if col not in train.columns and col not in test.columns:
        return

    # Fit categories on train (string-normalized)
    s = train[col].fillna("").astype("string").str.replace(cfg.text_ws_regex, " ", regex=True).str.strip()
    vc = s.value_counts(dropna=False)
    cats = [str(x) for x in vc.head(cfg.whyBDA_topk).index.tolist() if str(x).strip() != ""]

    # map category -> key
    cat_to_key: Dict[str, str] = {}
    for c in cats:
        ct = _clean_token(c)
        ct = ct if ct else "unk"
        cat_to_key[c] = _safe_feat_key(ct)

    def _apply(df: pd.DataFrame) -> None:
        s2 = df.get(col, pd.Series([""] * len(df), index=df.index)).fillna("").astype("string")
        s2 = s2.str.replace(cfg.text_ws_regex, " ", regex=True).str.strip()
        for c in cats:
            key = cat_to_key[c]
            df[f"whyBDA_is_{key}"] = (s2 == c).astype(np.int8)

    _apply(train)
    _apply(test)


def _parse_certificate_cell(v: object, cfg: CFG) -> Tuple[set[str], set[str]]:
    """Return (acquired_set, preparing_set) of canonical cert names."""
    try:
        if pd.isna(v):
            return set(), set()
    except Exception:
        pass
    raw = str(v).strip()
    if not raw or _clean_token(raw) in set(_clean_token(t) for t in (cfg.cert_none_tokens + cfg.common_none_tokens)):
        return set(), set()

    prep_prefix_re = re.compile(r"^\s*(준비중|준비|예정)\s*[:：]\s*")
    prep_token_re = re.compile(r"(준비중|준비|예정|응시|공부\s*중|계획|취득\s*예정)", re.IGNORECASE)

    none_set = set(cfg.cert_none_tokens) | {"해당없음", "없어요", "없음요"}

    def _norm(tok: str) -> str:
        t = str(tok).strip()
        if not t:
            return ""
        t = re.sub(r"\([^\)]*\)", " ", t)  # remove parentheses
        t = re.sub(prep_token_re, " ", t)
        t = t.replace("취득", " ").replace("합격", " ")
        t = re.sub(r"[\-_/·;:：|]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    prep_all = bool(prep_prefix_re.match(raw))
    t1 = prep_prefix_re.sub("", raw) if prep_all else raw

    acquired: set[str] = set()
    prep: set[str] = set()

    for tok in _split_multiselect_tokens(t1):
        tok = tok.strip()
        if not tok or tok in none_set:
            continue
        is_prep = prep_all or bool(prep_token_re.search(tok))
        name = _norm(tok)
        if not name or name in none_set:
            continue
        # canonicalize common english casing (adsp/sqld etc)
        if re.fullmatch(r"adsp", name.strip(), flags=re.IGNORECASE):
            name = "ADsP"
        if re.fullmatch(r"sqld", name.strip(), flags=re.IGNORECASE):
            name = "SQLD"
        if re.fullmatch(r"adp", name.strip(), flags=re.IGNORECASE):
            name = "ADP"
        if re.fullmatch(r"sqlp", name.strip(), flags=re.IGNORECASE):
            name = "SQLP"

        if is_prep:
            prep.add(name)
        else:
            acquired.add(name)

    return acquired, prep


def _fit_apply_certificate_multihot(train: pd.DataFrame, test: pd.DataFrame, cfg: CFG) -> None:
    col = cfg.cert_col
    if col not in train.columns and col not in test.columns:
        return

    from collections import Counter
    acq_c = Counter()
    prep_c = Counter()

    for v in train.get(col, pd.Series([], dtype="string")).tolist():
        acq, prep = _parse_certificate_cell(v, cfg)
        for t in acq:
            ct = _clean_token(t)
            if ct:
                acq_c[ct] += 1
        for t in prep:
            ct = _clean_token(t)
            if ct:
                prep_c[ct] += 1

    def _topk(counter: Counter, topk: int) -> Tuple[List[str], Dict[str, str]]:
        items = [(k, v) for k, v in counter.items() if v >= int(cfg.multihot_min_freq)]
        items.sort(key=lambda x: (-x[1], x[0]))
        vocab = [k for k, _ in items[: int(topk)]]
        key_map = {k: _safe_feat_key(k) for k in vocab}
        return vocab, key_map

    acq_vocab, acq_key = _topk(acq_c, cfg.cert_acq_topk)
    prep_vocab, prep_key = _topk(prep_c, cfg.cert_prep_topk)

    def _apply(df: pd.DataFrame) -> None:
        n = len(df)
        acq_sets: List[set[str]] = []
        prep_sets: List[set[str]] = []
        for v in df.get(col, pd.Series([""] * n, index=df.index)).tolist():
            acq, prep = _parse_certificate_cell(v, cfg)
            acq_sets.append(set(_clean_token(t) for t in acq if _clean_token(t)))
            prep_sets.append(set(_clean_token(t) for t in prep if _clean_token(t)))

        acq_vocab_set = set(acq_vocab)
        prep_vocab_set = set(prep_vocab)

        df["cert_acq_nunique"] = np.asarray([len(s) for s in acq_sets], dtype=np.int16)
        df["cert_acq_oov_n"] = np.asarray([len([t for t in s if t not in acq_vocab_set]) for s in acq_sets], dtype=np.int16)
        df["cert_prep_nunique"] = np.asarray([len(s) for s in prep_sets], dtype=np.int16)
        df["cert_prep_oov_n"] = np.asarray([len([t for t in s if t not in prep_vocab_set]) for s in prep_sets], dtype=np.int16)

        for ct in acq_vocab:
            df[f"cert_acq_has_{acq_key[ct]}"] = np.asarray([1 if ct in s else 0 for s in acq_sets], dtype=np.int8)
        for ct in prep_vocab:
            df[f"cert_prep_has_{prep_key[ct]}"] = np.asarray([1 if ct in s else 0 for s in prep_sets], dtype=np.int8)

    _apply(train)
    _apply(test)


def _fit_apply_desired_certificate_tok_multihot(train: pd.DataFrame, test: pd.DataFrame, cfg: CFG) -> None:
    col = "desired_certificate"
    if col not in train.columns and col not in test.columns:
        return
    vocab, key_map = _fit_token_vocab(
        train.get(col, pd.Series([], dtype="string")),
        cfg=cfg,
        topk=cfg.desired_cert_tok_topk,
        min_freq=cfg.multihot_min_freq,
    )
    _apply_token_multihot(train, col=col, prefix="desired_cert_tok_", vocab_clean=vocab, clean_to_key=key_map, cfg=cfg)
    _apply_token_multihot(test,  col=col, prefix="desired_cert_tok_", vocab_clean=vocab, clean_to_key=key_map, cfg=cfg)


def _fit_apply_interested_company_tok_multihot(train: pd.DataFrame, test: pd.DataFrame, cfg: CFG) -> None:
    col = "interested_company"
    if col not in train.columns and col not in test.columns:
        return
    vocab, key_map = _fit_token_vocab(
        train.get(col, pd.Series([], dtype="string")),
        cfg=cfg,
        topk=cfg.ic_any_topk,
        min_freq=cfg.multihot_min_freq,
    )
    _apply_token_multihot(train, col=col, prefix="ic_any_", vocab_clean=vocab, clean_to_key=key_map, cfg=cfg)
    _apply_token_multihot(test,  col=col, prefix="ic_any_", vocab_clean=vocab, clean_to_key=key_map, cfg=cfg)



def _make_whyBDA_features(df: pd.DataFrame) -> None:
    col = "whyBDA"
    if col not in df.columns:
        for k in ["scale_ops", "benefit", "hard_alone", "satisfied_prev", "low_burden", "no_test", "want_lecture"]:
            df[f"why_{k}"] = np.int8(0)
        df["why_count"] = np.int16(0)
        return

    s = df[col].fillna("").astype("string")
    low = s.str.lower()

    feats = {
        "scale_ops": ("큰 규모", "커리큘럼", "운영", "관리"),
        "benefit": ("혜택", "잡 페스티벌", "기업연계", "공모전"),
        "hard_alone": ("혼자", "어려워"),
        "satisfied_prev": ("만족", "이전 기수"),
        "low_burden": ("시간", "부담"),
        "no_test": ("코딩 테스트", "면접"),
        "want_lecture": ("현직자", "강의"),
    }

    total = np.zeros(len(df), dtype=np.int16)
    for name, kws in feats.items():
        hit = np.zeros(len(df), dtype=np.int8)
        for kw in kws:
            hit |= low.str.contains(str(kw).lower(), regex=False, na=False).to_numpy(dtype=np.int8)
        df[f"why_{name}"] = hit
        total += hit.astype(np.int16)

    df["why_count"] = total


def _make_what_to_gain_features(df: pd.DataFrame, cfg: CFG) -> None:
    col = "what_to_gain"
    if col not in df.columns:
        for k in ["project", "contest", "analysis", "network", "python_sql", "all", "collab"]:
            df[f"gain_{k}"] = np.int8(0)
        df["gain_len"] = np.float32(0.0)
        df["gain_words"] = np.float32(0.0)
        df["gain_count"] = np.int16(0)
        return

    s = df[col].fillna("").astype("string").str.replace(cfg.text_ws_regex, " ", regex=True).str.strip()
    low = s.str.lower()

    df["gain_len"] = s.str.len().astype("float32")
    df["gain_words"] = s.str.split(" ").map(lambda x: len([t for t in x if t])).astype("float32")

    feats = {
        "project": ("프로젝트",),
        "contest": ("공모전",),
        "analysis": ("분석", "데이터 분석", "역량"),
        "network": ("네트워크", "인적",),
        "python_sql": ("python", "sql"),
        "all": ("위 항목 전체", "전체"),
        "collab": ("협업", "함께",),
    }

    total = np.zeros(len(df), dtype=np.int16)
    for name, kws in feats.items():
        hit = np.zeros(len(df), dtype=np.int8)
        for kw in kws:
            hit |= low.str.contains(str(kw).lower(), regex=False, na=False).to_numpy(dtype=np.int8)
        df[f"gain_{name}"] = hit
        total += hit.astype(np.int16)

    df["gain_count"] = total




def _select_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    base_features: List[str],
    auto_prefixes: Tuple[str, ...] = (),
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    """
    Ensure required features exist; if a base feature is missing, create a safe default.
    Also supports pattern-based auto expansion (prefix match) so that generated multi-hot
    features are guaranteed to be included in training input.

    Returns X_train, y_train, X_test, final_features
    """
    final_features = list(base_features)

    # -----------------------------
    # Auto-expand: add any columns that match given prefixes (deterministic ordering)
    # -----------------------------
    if auto_prefixes:
        cand: set[str] = set()
        for df in (train, test):
            for c in df.columns:
                for p in auto_prefixes:
                    if c.startswith(p):
                        cand.add(c)
                        break
        for c in sorted(cand):
            if c not in final_features:
                final_features.append(c)

    # -----------------------------
    # Ensure existence with safe defaults
    # -----------------------------
    zero_prefixes = ("is_", "has_", "sig_") + tuple(auto_prefixes)

    for col in final_features:
        if col not in train.columns:
            train[col] = (0 if col.startswith(zero_prefixes) else np.nan)
        if col not in test.columns:
            test[col] = (0 if col.startswith(zero_prefixes) else np.nan)

    X_train = train[final_features].copy()
    y_train = train[target].copy()
    X_test = test[final_features].copy()
    return X_train, y_train, X_test, final_features


def _infer_categorical_columns(
    X_train: pd.DataFrame,
    forced: List[str] | None = None,
) -> List[str]:
    forced = forced or []
    categorical_cols = X_train.select_dtypes(include=["object", "string", "bool", "category"]).columns.tolist()
    for c in forced:
        if c in X_train.columns and c not in categorical_cols:
            categorical_cols.append(c)
    return categorical_cols


def _build_model_params(cfg: CFG) -> Dict:
    params = dict(cfg.cat_params)
    params["random_seed"] = cfg.seed
    return params


def _make_error_frame(
    X_va: pd.DataFrame,
    y_va: pd.Series,
    va_proba: np.ndarray,
    fold: int,
    cfg: CFG,
    id_series: Optional[pd.Series] = None,
    raw_extra: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a validation error DataFrame:
      - includes y_true, y_pred_proba, y_pred_label, error_type(FP/FN/OK), confidence(|p-0.5|)
      - optionally includes id column and extra raw columns (aligned by index)
      - includes validation feature values (X_va)
    """
    y_true = y_va.to_numpy()
    y_pred = (va_proba >= cfg.decision_threshold).astype(int)

    error_type = np.where((y_true == 0) & (y_pred == 1), "FP", np.where((y_true == 1) & (y_pred == 0), "FN", "OK"))
    # '확신(confidence)'은 예측한 클래스에 할당된 확률로 정의 (threshold가 0.5가 아니어도 해석 가능)
    pred_confidence = np.where(y_pred == 1, va_proba, 1.0 - va_proba)
    margin_to_threshold = np.abs(va_proba - cfg.decision_threshold)

    base = pd.DataFrame(
        {
            "fold": fold,
            "row_index": X_va.index.to_numpy(),
            "y_true": y_true,
            "y_pred_proba": va_proba,
            "y_pred": y_pred,
            "error_type": error_type,
            "confidence": pred_confidence,
            "margin_to_threshold": margin_to_threshold,
        },
        index=X_va.index,
    )

    parts = [base]

    if id_series is not None:
        parts.append(pd.DataFrame({"id": id_series.loc[X_va.index].to_numpy()}, index=X_va.index))

    if raw_extra is not None and len(cfg.error_extra_cols_from_raw) > 0:
        extra_cols = [c for c in cfg.error_extra_cols_from_raw if c in raw_extra.columns]
        if extra_cols:
            parts.append(raw_extra.loc[X_va.index, extra_cols].copy())

    parts.append(X_va.copy())
    out = pd.concat(parts, axis=1)

    # 오답만 위로 보이게 정렬(OK는 아래), 그리고 confidence 내림차순
    out["_is_wrong"] = (out["error_type"] != "OK").astype(int)
    out = out.sort_values(by=["_is_wrong", "confidence"], ascending=[False, False]).drop(columns=["_is_wrong"])
    return out


def _save_error_reports(
    err_df: pd.DataFrame,
    out_dir: str,
    name: str,
    cfg: CFG,
) -> None:
    _ensure_dir(out_dir)
    if cfg.error_save_max_rows and cfg.error_save_max_rows > 0:
        # 오답 중 confidence 상위 N만 저장
        wrong = err_df[err_df["error_type"] != "OK"].head(cfg.error_save_max_rows)
        wrong.to_csv(os.path.join(out_dir, f"{name}_wrong_top{cfg.error_save_max_rows}.csv"), index=False)
    else:
        # 오답 전부 저장
        wrong = err_df[err_df["error_type"] != "OK"]
        wrong.to_csv(os.path.join(out_dir, f"{name}_wrong.csv"), index=False)

    # 전체(오답+정답)도 같이 남기고 싶으면 아래 주석 해제
    # err_df.to_csv(os.path.join(out_dir, f"{name}_all.csv"), index=False)


def _plot_feature_importance(fi_df: pd.DataFrame, top_n: int, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"[WARN] matplotlib import failed: {e}. Skip plotting.")
        return

    sub = fi_df.head(top_n).copy()
    if sub.empty:
        print("[WARN] feature importance is empty. Skip plotting.")
        return

    # 가독성: 아래에서 위로 큰 값이 보이도록 뒤집기
    sub = sub.iloc[::-1]

    height = max(4.0, 0.35 * len(sub))
    plt.figure(figsize=(10.5, height))
    plt.barh(sub["feature"], sub["importance_mean"], xerr=(sub["importance_std"] if "importance_std" in sub.columns else None))
    plt.xlabel("Importance (mean ± std across folds)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _best_threshold_by_f1(y_true: np.ndarray, proba: np.ndarray, n_grid: int = 181):
    """
    Sweep threshold and return (best_thr, best_f1).
    n_grid=181이면 0.05~0.95를 0.005 간격으로 탐색.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=np.float64)

    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, n_grid):
        m = _binary_metrics(y_true, proba, float(thr))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = float(thr)
    return best_thr, best_f1


def _binary_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute threshold-based metrics + confusion matrix elements."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=np.float64)

    y_pred = (proba >= threshold).astype(int)

    # confusion matrix with fixed label order
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    out: Dict[str, float] = {
        "threshold": float(threshold),
        "n": float(len(y_true)),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }
    return out


def _save_json(obj: Dict, path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ============================================================
# CV-safe (fold-local) fitting helpers
#   - Fix leakage by fitting ANY distribution-dependent mapping
#     strictly on fold-train, then applying to fold-val/test.
# ============================================================

@dataclass
class MajorMaps:
    m1_freq: Dict[str, int] = field(default_factory=dict)
    m2_freq: Dict[str, int] = field(default_factory=dict)
    m1_top: set = field(default_factory=set)
    m2_top: set = field(default_factory=set)
    pair_freq: Dict[str, int] = field(default_factory=dict)


def _make_major_basic_features(df: pd.DataFrame, cfg: CFG) -> None:
    """Row-wise major features (safe outside CV): flags / consistency.
    NOTE: frequency/topk features are handled fold-locally via MajorMaps.
    """
    def _is_data_series(s: pd.Series) -> pd.Series:
        s = s.fillna("").astype("string")
        return (
            s.str.contains("IT", case=False, regex=False, na=False)
            | s.str.contains("컴퓨터", case=False, regex=False, na=False)
            | s.str.contains("소프트웨어", case=False, regex=False, na=False)
            | s.str.contains("인공지능", case=False, regex=False, na=False)
            | s.str.contains("AI", case=False, regex=False, na=False)
            | s.str.contains("데이터", case=False, regex=False, na=False)
            | s.str.contains("통계", case=False, regex=False, na=False)
            | s.str.contains("산업경영", case=False, regex=False, na=False)
            | s.str.contains("산업", case=False, regex=False, na=False)
        )

    if "major1_1" in df.columns:
        df["major1_1_is_data"] = _is_data_series(_normalize_major_series(df["major1_1"], cfg)).astype(np.int8)
    else:
        df["major1_1_is_data"] = np.int8(0)

    if "major1_2" in df.columns:
        df["major1_2_is_data"] = _is_data_series(_normalize_major_series(df["major1_2"], cfg)).astype(np.int8)
    else:
        df["major1_2_is_data"] = np.int8(0)

    # 복수전공 여부
    if "major type" in df.columns:
        mt = df["major type"].fillna("").astype("string")
        df["is_double_major"] = mt.str.contains("복수|다중|이중", regex=True, na=False).astype(np.int8)
    else:
        df["is_double_major"] = np.int8(0)

    # major1_2 결측/일관성 피처
    if "major1_2" in df.columns:
        m2 = _normalize_major_series(df["major1_2"], cfg)
        df["has_major2"] = (m2 != "").astype(np.int8)
    else:
        df["has_major2"] = np.int8(0)

    df["major2_missing_but_double_major"] = ((df["is_double_major"] == 1) & (df["has_major2"] == 0)).astype(np.int8)
    df["major2_present_but_single_major"] = ((df["is_double_major"] == 0) & (df["has_major2"] == 1)).astype(np.int8)


def _fit_major_maps(tr_df: pd.DataFrame, cfg: CFG) -> MajorMaps:
    """Fit major frequency/topk mappings on fold-train only."""
    maps = MajorMaps()

    def _fit_one(col: str) -> Tuple[Dict[str, int], set]:
        if col not in tr_df.columns:
            return {}, set()
        s = _normalize_major_series(tr_df[col], cfg)
        s = s.replace({"": "MISSING"})
        vc = s.value_counts(dropna=False)
        freq = {str(k): int(v) for k, v in vc.to_dict().items()}
        top = set(vc.head(cfg.major_topk).index.astype(str).tolist())
        return freq, top

    maps.m1_freq, maps.m1_top = _fit_one("major1_1")
    maps.m2_freq, maps.m2_top = _fit_one("major1_2")

    # pair_freq is fitted on topk-stabilized pair within fold-train
    if ("major1_1" in tr_df.columns) or ("major1_2" in tr_df.columns):
        m1 = _normalize_major_series(
            tr_df.get("major1_1", pd.Series(["UNK"] * len(tr_df), index=tr_df.index)),
            cfg,
        )
        m2 = _normalize_major_series(
            tr_df.get("major1_2", pd.Series(["UNK"] * len(tr_df), index=tr_df.index)),
            cfg,
        )
        m1 = m1.replace({"": "MISSING"}).astype(str)
        m2 = m2.replace({"": "MISSING"}).astype(str)
        m1_topk = np.where(pd.Series(m1, index=tr_df.index).isin(maps.m1_top), m1, "OTHER")
        m2_topk = np.where(pd.Series(m2, index=tr_df.index).isin(maps.m2_top), m2, "OTHER")
        pair = pd.Series(m1_topk, index=tr_df.index).astype("string") + cfg.major_pair_sep + pd.Series(m2_topk, index=tr_df.index).astype("string")
        vc_pair = pair.value_counts(dropna=False)
        maps.pair_freq = {str(k): int(v) for k, v in vc_pair.to_dict().items()}
    else:
        maps.pair_freq = {}

    return maps


def _apply_major_maps(df: pd.DataFrame, maps: MajorMaps, cfg: CFG) -> None:
    """Apply fold-train major maps to any df (fold-train/val/test)."""
    def _apply_one(col: str, prefix: str, freq: Dict[str, int], top: set) -> None:
        if col not in df.columns:
            df[f"{prefix}_freq"] = np.int16(0)
            df[f"{prefix}_freq_log1p"] = np.float32(0.0)
            df[f"{prefix}_is_rare"] = np.int8(0)
            df[f"{prefix}_topk"] = pd.Series(["UNK"] * len(df), index=df.index, dtype="string")
            return

        s = _normalize_major_series(df[col], cfg).replace({"": "MISSING"})
        s_str = s.astype(str)
        f = s_str.map(freq).fillna(0).astype("int32")

        df[f"{prefix}_freq"] = f.astype(np.int16)
        df[f"{prefix}_freq_log1p"] = np.log1p(f.to_numpy()).astype(np.float32)
        df[f"{prefix}_is_rare"] = (f <= cfg.major_rare_threshold).astype(np.int8)
        df[f"{prefix}_topk"] = s_str.where(s_str.isin(top), "OTHER").astype("string")

    _apply_one("major1_1", "major1_1", maps.m1_freq, maps.m1_top)
    _apply_one("major1_2", "major1_2", maps.m2_freq, maps.m2_top)

    # pair features
    m1_pair = df.get("major1_1_topk", df.get("major1_1", "UNK")).astype("string").fillna("UNK")
    m2_pair = df.get("major1_2_topk", df.get("major1_2", "UNK")).astype("string").fillna("UNK")
    df["major_pair"] = (m1_pair + cfg.major_pair_sep + m2_pair).astype("string").fillna("UNK")

    pf = df["major_pair"].astype(str).map(maps.pair_freq).fillna(0).astype("float32")
    df["major_pair_freq"] = pf
    df["major_pair_freq_log1p"] = np.log1p(pf).astype("float32")


def _fit_whyBDA_onehot(tr_df: pd.DataFrame, cfg: CFG) -> Tuple[List[str], Dict[str, str]]:
    col = "whyBDA"
    if col not in tr_df.columns:
        return [], {}
    s = tr_df[col].fillna("").astype("string").str.replace(cfg.text_ws_regex, " ", regex=True).str.strip()
    vc = s.value_counts(dropna=False)
    cats = [str(x) for x in vc.head(cfg.whyBDA_topk).index.tolist() if str(x).strip() != ""]
    cat_to_key: Dict[str, str] = {}
    for c in cats:
        ct = _clean_token(c) or "unk"
        cat_to_key[c] = _safe_feat_key(ct)
    return cats, cat_to_key


def _apply_whyBDA_onehot(df: pd.DataFrame, cats: List[str], cat_to_key: Dict[str, str], cfg: CFG) -> None:
    col = "whyBDA"
    s2 = df.get(col, pd.Series([""] * len(df), index=df.index)).fillna("").astype("string")
    s2 = s2.str.replace(cfg.text_ws_regex, " ", regex=True).str.strip()
    for c in cats:
        key = cat_to_key[c]
        df[f"whyBDA_is_{key}"] = (s2 == c).astype(np.int8)


@dataclass
class CertVocab:
    acq_vocab: List[str] = field(default_factory=list)
    acq_key: Dict[str, str] = field(default_factory=dict)
    prep_vocab: List[str] = field(default_factory=list)
    prep_key: Dict[str, str] = field(default_factory=dict)


def _fit_certificate_vocab(tr_df: pd.DataFrame, cfg: CFG) -> CertVocab:
    col = cfg.cert_col
    if col not in tr_df.columns:
        return CertVocab()
    from collections import Counter
    acq_c = Counter()
    prep_c = Counter()
    for v in tr_df.get(col, pd.Series([], dtype="string")).tolist():
        acq, prep = _parse_certificate_cell(v, cfg)
        for t in acq:
            ct = _clean_token(t)
            if ct:
                acq_c[ct] += 1
        for t in prep:
            ct = _clean_token(t)
            if ct:
                prep_c[ct] += 1

    def _topk(counter: Counter, topk: int) -> Tuple[List[str], Dict[str, str]]:
        items = [(k, v) for k, v in counter.items() if v >= int(cfg.multihot_min_freq)]
        items.sort(key=lambda x: (-x[1], x[0]))
        vocab = [k for k, _ in items[: int(topk)]]
        key_map = {k: _safe_feat_key(k) for k in vocab}
        return vocab, key_map

    acq_vocab, acq_key = _topk(acq_c, cfg.cert_acq_topk)
    prep_vocab, prep_key = _topk(prep_c, cfg.cert_prep_topk)
    return CertVocab(acq_vocab=acq_vocab, acq_key=acq_key, prep_vocab=prep_vocab, prep_key=prep_key)


def _apply_certificate_multihot(df: pd.DataFrame, vocab: CertVocab, cfg: CFG) -> None:
    col = cfg.cert_col
    if col not in df.columns:
        # still create aggregate features expected downstream
        df["cert_acq_nunique"] = np.int16(0)
        df["cert_acq_oov_n"] = np.int16(0)
        df["cert_prep_nunique"] = np.int16(0)
        df["cert_prep_oov_n"] = np.int16(0)
        return

    n = len(df)
    acq_sets: List[set[str]] = []
    prep_sets: List[set[str]] = []
    for v in df.get(col, pd.Series([""] * n, index=df.index)).tolist():
        acq, prep = _parse_certificate_cell(v, cfg)
        acq_sets.append(set(_clean_token(t) for t in acq if _clean_token(t)))
        prep_sets.append(set(_clean_token(t) for t in prep if _clean_token(t)))

    acq_vocab_set = set(vocab.acq_vocab)
    prep_vocab_set = set(vocab.prep_vocab)

    df["cert_acq_nunique"] = np.asarray([len(s) for s in acq_sets], dtype=np.int16)
    df["cert_acq_oov_n"] = np.asarray([len([t for t in s if t not in acq_vocab_set]) for s in acq_sets], dtype=np.int16)
    df["cert_prep_nunique"] = np.asarray([len(s) for s in prep_sets], dtype=np.int16)
    df["cert_prep_oov_n"] = np.asarray([len([t for t in s if t not in prep_vocab_set]) for s in prep_sets], dtype=np.int16)

    for ct in vocab.acq_vocab:
        df[f"cert_acq_has_{vocab.acq_key[ct]}"] = np.asarray([1 if ct in s else 0 for s in acq_sets], dtype=np.int8)
    for ct in vocab.prep_vocab:
        df[f"cert_prep_has_{vocab.prep_key[ct]}"] = np.asarray([1 if ct in s else 0 for s in prep_sets], dtype=np.int8)


def _fit_major_field_vocab(tr_df: pd.DataFrame, cfg: CFG) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    col = "major_field"
    alias = {
        _clean_token("자연고학"): _clean_token("자연과학"),
    }
    vocab, key_map = _fit_token_vocab(
        tr_df.get(col, pd.Series([], dtype="string")),
        cfg=cfg,
        topk=cfg.major_field_topk,
        min_freq=cfg.multihot_min_freq,
        alias_clean_map=alias,
    )
    return vocab, key_map, alias


def _fit_simple_token_vocab(tr_df: pd.DataFrame, col: str, cfg: CFG, topk: int) -> Tuple[List[str], Dict[str, str]]:
    vocab, key_map = _fit_token_vocab(
        tr_df.get(col, pd.Series([], dtype="string")),
        cfg=cfg,
        topk=topk,
        min_freq=cfg.multihot_min_freq,
    )
    return vocab, key_map


def _select_features_3way(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    te_df: pd.DataFrame,
    target: str,
    base_features: List[str],
    auto_prefixes: Tuple[str, ...] = (),
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    """3-way version of _select_features: ensure tr/va/te share identical columns."""
    final_features = list(base_features)

    if auto_prefixes:
        cand: set[str] = set()
        for df in (tr_df, va_df, te_df):
            for c in df.columns:
                for p in auto_prefixes:
                    if c.startswith(p):
                        cand.add(c)
                        break
        for c in sorted(cand):
            if c not in final_features:
                final_features.append(c)

    zero_prefixes = ("is_", "has_", "sig_") + tuple(auto_prefixes)

    for col in final_features:
        if col not in tr_df.columns:
            tr_df[col] = (0 if col.startswith(zero_prefixes) else np.nan)
        if col not in va_df.columns:
            va_df[col] = (0 if col.startswith(zero_prefixes) else np.nan)
        if col not in te_df.columns:
            te_df[col] = (0 if col.startswith(zero_prefixes) else np.nan)

    X_tr = tr_df[final_features].copy()
    y_tr = tr_df[target].copy()
    X_va = va_df[final_features].copy()
    y_va = va_df[target].copy()
    X_te = te_df[final_features].copy()
    return X_tr, y_tr, X_va, y_va, X_te, final_features


def _prune_sparse_features_3way(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    feature_names: List[str],
    categorical_cols: List[str],
    text_cols: List[str],
    cfg: CFG,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str], List[str], List[str]]:
    """Prune sparse/unstable features using fold-train statistics only."""
    if not cfg.feature_prune_enable:
        return X_tr, X_va, X_te, feature_names, categorical_cols, text_cols, []

    n = len(X_tr)
    drop_cols: set[str] = set()

    for c in feature_names:
        s = X_tr[c]

        if cfg.feature_prune_zero_variance and s.nunique(dropna=False) <= 1:
            drop_cols.add(c)
            continue

        if (c in categorical_cols) or (c in text_cols):
            continue

        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() == 0:
            drop_cols.add(c)
            continue

        num0 = num.fillna(0.0)
        zero_ratio = float((num0 == 0).mean())
        nonzero_cnt = int((num0 != 0).sum())
        if (zero_ratio >= cfg.feature_prune_sparse_zero_ratio_ge) and (nonzero_cnt <= cfg.feature_prune_sparse_nonzero_leq):
            drop_cols.add(c)
            continue

        # Extra guard for extremely rare binary flags.
        vals = set(np.unique(num0.to_numpy(dtype=np.float64)).tolist())
        if vals.issubset({0.0, 1.0}):
            pos = int((num0 == 1).sum())
            neg = n - pos
            if (pos < cfg.feature_prune_binary_min_count) or (neg < cfg.feature_prune_binary_min_count):
                drop_cols.add(c)

    if not drop_cols:
        return X_tr, X_va, X_te, feature_names, categorical_cols, text_cols, []

    kept = [c for c in feature_names if c not in drop_cols]
    if len(kept) == 0:
        # Safety fallback: never return empty feature matrix.
        return X_tr, X_va, X_te, feature_names, categorical_cols, text_cols, []

    X_tr = X_tr[kept].copy()
    X_va = X_va[kept].copy()
    X_te = X_te[kept].copy()
    categorical_cols = [c for c in categorical_cols if c in kept]
    text_cols = [c for c in text_cols if c in kept]
    return X_tr, X_va, X_te, kept, categorical_cols, text_cols, sorted(drop_cols)


def _build_class_weight_candidates(cfg: CFG, is_binary: bool) -> List[Tuple[str, Dict]]:
    """Build parameter candidates for class-weight grid search."""
    base_params = _build_model_params(cfg)
    if (not is_binary) or (not cfg.class_weight_grid_enable):
        return [("single", base_params)]

    cand: List[Tuple[str, Dict]] = []
    if cfg.class_weight_grid_include_auto:
        cand.append(("auto_balanced", dict(base_params)))

    for w0, w1 in cfg.class_weight_grid:
        p = dict(base_params)
        p.pop("auto_class_weights", None)
        p["class_weights"] = [float(w0), float(w1)]
        cand.append((f"cw_{w0:g}_{w1:g}", p))

    # Deduplicate by core weight settings
    uniq: List[Tuple[str, Dict]] = []
    seen: set[str] = set()
    for name, p in cand:
        if "class_weights" in p:
            sig = f"class_weights:{p['class_weights']}"
        else:
            sig = f"auto:{p.get('auto_class_weights', 'none')}"
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append((name, p))
    if not uniq:
        uniq = [("base", base_params)]
    return uniq


def _candidate_weight_desc(params: Dict) -> str:
    if "class_weights" in params:
        return f"class_weights={params['class_weights']}"
    if "auto_class_weights" in params:
        return f"auto_class_weights={params['auto_class_weights']}"
    return "unweighted"


def _read_csv_with_fallback(path: str, fallback: Optional[str] = None) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    if fallback and os.path.exists(fallback):
        return pd.read_csv(fallback)
    raise FileNotFoundError(f"CSV not found: {path} (fallback={fallback})")



def main() -> None:
    warnings.filterwarnings("ignore")
    cfg = CFG()

    # -----------------------------
    # 1) Load (원본 보관: 오류 분석용)
    # -----------------------------
    # Fallbacks for this sandbox / ad-hoc runs
    train_fallback = "/mnt/data/train.csv"
    test_fallback = "/mnt/data/test.csv"
    sub_fallback = "/mnt/data/sample_submission.csv"

    train = _read_csv_with_fallback(cfg.train_path, fallback=(train_fallback if os.path.exists(train_fallback) else None))
    test = _read_csv_with_fallback(cfg.test_path, fallback=(test_fallback if os.path.exists(test_fallback) else None))
    train_raw = train.copy()

    raw_cols = list(train.columns)
    id_col = _resolve_id_col(train_raw, cfg.id_col_candidates)
    id_series = train_raw[id_col] if id_col is not None else None

    if cfg.target not in train.columns:
        raise ValueError(f"target column '{cfg.target}' not found in train")

    # -----------------------------
    # 2) Stage-0: leakage-free feature engineering (row-wise only)
    #    - distribution-dependent ops are moved into CV loop (Stage-1)
    # -----------------------------
    train_base = train.copy()
    test_base = test.copy()

    for df in (train_base, test_base):
        _apply_major_standardization_inplace(df, cfg)
        _make_is_major_it(df, cfg)
        _make_completed_semester_features(df, cfg)
        _make_certificate_features(df, cfg)
        _make_basic_text_presence_features(df, cfg)
        _make_interested_company_semantic_features(df, cfg)

        # major basic flags (safe); freq/topk/pair are fold-fitted
        _make_major_basic_features(df, cfg)

        _make_previous_class_agg_features(df)
        _make_desired_job_features(df, cfg)
        _make_incumbents_reason_text_stats(df, cfg)
        _make_engagement_features(df, raw_cols=raw_cols, cfg=cfg, id_col=id_col)
        _make_onedayclass_multihot(df, cfg)
        _make_expected_domain_features(df, cfg)
        _make_desired_certificate_features(df, cfg)
        _make_whyBDA_features(df)
        _make_what_to_gain_features(df, cfg)
        _make_signature_rule_features(df, cfg)

    # -----------------------------
    # 3) CV splitter
    # -----------------------------
    y_full = train_base[cfg.target].copy()
    if y_full.dtype.kind not in ("i", "u", "b"):
        y_full = pd.to_numeric(y_full, errors="coerce").fillna(0).astype(int)

    is_binary = (y_full.nunique() == 2)
    splitter = (
        StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
        if (cfg.use_stratified_if_binary and is_binary)
        else KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    )

    oof_pred = np.zeros(len(train_base), dtype=np.float64)
    test_pred_folds: List[np.ndarray] = []
    fold_scores: List[float] = []
    fold_fi_dicts: List[Dict[str, float]] = []
    all_fold_errors: List[pd.DataFrame] = []
    fold_metrics_selected: Dict[int, Dict[str, float]] = {}

    errors_dir = os.path.join(cfg.artifacts_dir, cfg.errors_subdir)
    metrics_dir = os.path.join(cfg.artifacts_dir, cfg.metrics_subdir)
    fi_dir = os.path.join(cfg.artifacts_dir, cfg.fi_subdir)
    _ensure_dir(errors_dir)
    _ensure_dir(metrics_dir)
    _ensure_dir(fi_dir)

    # -----------------------------
    # 4) Prepare fold datasets (Stage-1 fold-local fit/apply)
    # -----------------------------
    MISSING_CAT = "__MISSING__"
    fold_data: List[Dict] = []
    prune_log: Dict[int, List[str]] = {}

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(np.zeros(len(train_base)), y_full), start=1):
        tr_df = train_base.iloc[tr_idx].copy()
        va_df = train_base.iloc[va_idx].copy()
        te_df = test_base.copy()

        # -------- Stage-1 (fold-local) begins --------
        # (A) major freq/topk/pair freq (fit on tr_df only)
        major_maps = _fit_major_maps(tr_df, cfg)
        for df in (tr_df, va_df, te_df):
            _apply_major_maps(df, major_maps, cfg)

        # (B) train-fitted vocab multi-hot / mappers
        # major_field multihot
        vocab_mf, key_mf, alias_mf = _fit_major_field_vocab(tr_df, cfg)
        for df in (tr_df, va_df, te_df):
            _apply_token_multihot(df, col="major_field", prefix="major_field_", vocab_clean=vocab_mf, clean_to_key=key_mf, cfg=cfg, alias_clean_map=alias_mf)

        # whyBDA onehot
        why_cats, why_key = _fit_whyBDA_onehot(tr_df, cfg)
        for df in (tr_df, va_df, te_df):
            _apply_whyBDA_onehot(df, why_cats, why_key, cfg)

        # certificate multihot
        cert_vocab = _fit_certificate_vocab(tr_df, cfg)
        for df in (tr_df, va_df, te_df):
            _apply_certificate_multihot(df, cert_vocab, cfg)

        # desired_certificate token multihot
        vocab_dc, key_dc = _fit_simple_token_vocab(tr_df, "desired_certificate", cfg, topk=cfg.desired_cert_tok_topk)
        for df in (tr_df, va_df, te_df):
            _apply_token_multihot(df, col="desired_certificate", prefix="desired_cert_tok_", vocab_clean=vocab_dc, clean_to_key=key_dc, cfg=cfg)

        # interested_company token multihot
        vocab_ic, key_ic = _fit_simple_token_vocab(tr_df, "interested_company", cfg, topk=cfg.ic_any_topk)
        for df in (tr_df, va_df, te_df):
            _apply_token_multihot(df, col="interested_company", prefix="ic_any_", vocab_clean=vocab_ic, clean_to_key=key_ic, cfg=cfg)

        # school1 topk/freq mapper
        top_school, school_freq = _fit_school1_mapper(tr_df, cfg)
        for df in (tr_df, va_df, te_df):
            _apply_school1_mapper(df, top_school, school_freq, cfg)

        # ic_primary topk/freq mapper
        top_ic_primary, ic_freq = _fit_ic_primary_mapper(tr_df, cfg)
        for df in (tr_df, va_df, te_df):
            _apply_ic_primary_mapper(df, top_ic_primary, ic_freq, cfg)

        # (C) time_input strategy fold-local
        if cfg.time_col in tr_df.columns:
            time_strategy = _infer_time_input_strategy(tr_df[cfg.time_col], cfg)
        else:
            time_strategy = TimeInputStrategy("drop")

        for df in (tr_df, va_df, te_df):
            _apply_time_input_strategy(df, time_strategy, cfg)

        # bucket only when numeric
        for df in (tr_df, va_df, te_df):
            if "time_input" in df.columns and pd.api.types.is_numeric_dtype(df["time_input"]):
                df["time_input_bucket"] = pd.cut(
                    df["time_input"],
                    bins=[-np.inf, 1, 2, 3, 4, np.inf],
                    labels=["<=1", "1-2", "2-3", "3-4", ">4"],
                    include_lowest=True,
                ).astype("string").fillna("UNK")

        for df in (tr_df, va_df, te_df):
            _make_time_interaction_features(df, cfg)

        # build feature list (same logic as original, but per-fold)
        base_features = list(cfg.base_features)
        features = [f for f in base_features if f != cfg.time_col]

        if "time_input_bucket" in tr_df.columns and "time_input_bucket" not in features:
            features.append("time_input_bucket")

        if time_strategy.mode in ("numeric", "categorical"):
            if cfg.time_col in base_features:
                insert_idx = base_features.index(cfg.time_col)
                features.insert(insert_idx, cfg.time_col)
            else:
                features.append(cfg.time_col)
        elif time_strategy.mode == "datetime":
            features.extend(list(time_strategy.created_cols))

        # (D) missing-drop fold-local (fit on tr_df only)
        protect = set(features + [cfg.target])
        missing_ratio = tr_df.drop(columns=[cfg.target], errors="ignore").isnull().mean()
        columns_to_drop = [
            c for c in missing_ratio[missing_ratio > cfg.drop_missing_ratio_gt].index.tolist()
            if c not in protect
        ]
        for df in (tr_df, va_df, te_df):
            df.drop(columns=columns_to_drop, errors="ignore", inplace=True)

        # 3-way select (ensures identical columns)
        X_tr, y_tr, X_va, y_va, X_te, feature_names = _select_features_3way(
            tr_df, va_df, te_df, cfg.target, features, auto_prefixes=cfg.auto_expand_prefixes
        )

        # Categorical vs Text split per fold
        text_cols = [c for c in cfg.text_features if c in X_tr.columns]
        categorical_cols = _infer_categorical_columns(X_tr, forced=list(cfg.force_categorical))
        categorical_cols = [c for c in categorical_cols if c not in text_cols]

        # Sparse/unstable feature pruning (fit on fold-train only)
        X_tr, X_va, X_te, feature_names, categorical_cols, text_cols, dropped_cols = _prune_sparse_features_3way(
            X_tr, X_va, X_te, feature_names, categorical_cols, text_cols, cfg
        )
        if dropped_cols:
            prune_log[fold] = dropped_cols
            print(f"[Fold {fold}] Pruned {len(dropped_cols)} sparse/unstable features.")

        for c in categorical_cols:
            for X in (X_tr, X_va, X_te):
                X[c] = X[c].astype("object")
                X[c] = X[c].where(pd.notna(X[c]), MISSING_CAT)

        for c in text_cols:
            for X in (X_tr, X_va, X_te):
                X[c] = X[c].fillna("").astype("string").str.replace(cfg.text_ws_regex, " ", regex=True).str.strip()

        cat_feature_indices = [X_tr.columns.get_loc(c) for c in categorical_cols]
        text_feature_indices = [X_tr.columns.get_loc(c) for c in text_cols]
        text_feature_indices = text_feature_indices if text_feature_indices else None

        fold_data.append(
            dict(
                fold=fold,
                va_idx=va_idx,
                X_tr=X_tr,
                y_tr=y_tr,
                X_va=X_va,
                y_va=y_va,
                X_te=X_te,
                feature_names=feature_names,
                cat_feature_indices=cat_feature_indices,
                text_feature_indices=text_feature_indices,
            )
        )

    if cfg.feature_prune_enable and prune_log:
        pruned_unique = sorted({c for cols in prune_log.values() for c in cols})
        prune_payload = {
            "n_folds_with_pruning": int(len(prune_log)),
            "fold_pruned_counts": {str(k): int(len(v)) for k, v in sorted(prune_log.items())},
            "fold_pruned_cols": {str(k): v for k, v in sorted(prune_log.items())},
            "unique_pruned_cols_count": int(len(pruned_unique)),
            "unique_pruned_cols": pruned_unique,
        }
        _save_json(prune_payload, os.path.join(metrics_dir, "feature_prune_log.json"))

    # -----------------------------
    # 5) Class-weight grid runner
    # -----------------------------
    candidates = _build_class_weight_candidates(cfg, is_binary)
    grid_rows: List[Dict] = []
    best_bundle: Optional[Dict] = None

    for cand_name, cand_params in candidates:
        print(f"[CW] Candidate '{cand_name}' -> {_candidate_weight_desc(cand_params)}")
        cand_oof = np.zeros(len(train_base), dtype=np.float64)
        cand_test_folds: List[np.ndarray] = []
        cand_fold_scores: List[float] = []
        cand_fold_fi_dicts: List[Dict[str, float]] = []
        cand_all_fold_errors: List[pd.DataFrame] = []
        cand_fold_metrics: Dict[int, Dict[str, float]] = {}

        for fd in fold_data:
            fold = int(fd["fold"])
            X_tr = fd["X_tr"]
            y_tr = fd["y_tr"]
            X_va = fd["X_va"]
            y_va = fd["y_va"]
            X_te = fd["X_te"]
            va_idx = fd["va_idx"]
            feature_names = fd["feature_names"]
            cat_feature_indices = fd["cat_feature_indices"]
            text_feature_indices = fd["text_feature_indices"]

            train_pool = Pool(X_tr, y_tr, cat_features=cat_feature_indices, text_features=text_feature_indices)
            valid_pool = Pool(X_va, y_va, cat_features=cat_feature_indices, text_features=text_feature_indices)
            test_pool = Pool(X_te, cat_features=cat_feature_indices, text_features=text_feature_indices)

            model = CatBoostClassifier(**cand_params)
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

            va_proba = model.predict_proba(valid_pool)[:, 1]
            cand_oof[va_idx] = va_proba

            te_proba = model.predict_proba(test_pool)[:, 1]
            cand_test_folds.append(te_proba.astype(np.float64))

            if is_binary:
                auc = roc_auc_score(y_va, va_proba)
                cand_fold_scores.append(float(auc))
                m = _binary_metrics(y_va.to_numpy(), va_proba, cfg.decision_threshold)
                cand_fold_metrics[fold] = m
                print(
                    f"[CW:{cand_name}][Fold {fold}/{cfg.n_splits}] "
                    f"AUC={auc:.6f} F1@{cfg.decision_threshold:.3f}={m['f1']:.4f} "
                    f"P={m['precision']:.4f} R={m['recall']:.4f}"
                )

            if cfg.save_validation_errors and is_binary:
                err_df = _make_error_frame(
                    X_va=X_va,
                    y_va=y_va,
                    va_proba=va_proba,
                    fold=fold,
                    cfg=cfg,
                    id_series=id_series,
                    raw_extra=train_raw,
                )
                cand_all_fold_errors.append(err_df)

            if cfg.save_feature_importance:
                try:
                    fi = model.get_feature_importance(train_pool, type=cfg.feature_importance_type)
                except Exception:
                    fi = model.get_feature_importance(type=cfg.feature_importance_type)
                fi = np.asarray(fi, dtype=np.float64)
                if fi.shape[0] == len(feature_names):
                    cand_fold_fi_dicts.append({name: float(val) for name, val in zip(feature_names, fi.tolist())})

        cand_auc_mean = float(np.mean(cand_fold_scores)) if cand_fold_scores else None
        cand_auc_std = float(np.std(cand_fold_scores)) if cand_fold_scores else None
        cand_thr = cfg.decision_threshold
        cand_f1 = None
        if is_binary and (not cfg.submit_proba):
            cand_thr, cand_f1 = _best_threshold_by_f1(y_full.to_numpy(), cand_oof)
            cand_thr = float(cand_thr)
        score = float(cand_f1) if (cand_f1 is not None) else (float(cand_auc_mean) if cand_auc_mean is not None else 0.0)

        row: Dict = {
            "candidate": cand_name,
            "weight_mode": _candidate_weight_desc(cand_params),
            "score": score,
            "oof_best_f1": (float(cand_f1) if cand_f1 is not None else None),
            "oof_best_threshold": float(cand_thr),
            "oof_auc_mean": cand_auc_mean,
            "oof_auc_std": cand_auc_std,
            "n_features_mean": float(np.mean([len(d.get("feature_names", [])) for d in fold_data])) if fold_data else None,
        }
        grid_rows.append(row)
        print(
            f"[CW:{cand_name}] score={score:.6f} "
            f"(best_f1={cand_f1 if cand_f1 is not None else 'NA'}, auc_mean={cand_auc_mean if cand_auc_mean is not None else 'NA'})"
        )

        if (best_bundle is None) or (score > best_bundle["score"]):
            best_bundle = {
                "candidate": cand_name,
                "params": cand_params,
                "score": score,
                "oof_pred": cand_oof,
                "test_pred_folds": cand_test_folds,
                "fold_scores": cand_fold_scores,
                "fold_fi_dicts": cand_fold_fi_dicts,
                "all_fold_errors": cand_all_fold_errors,
                "fold_metrics": cand_fold_metrics,
                "tuned_thr": float(cand_thr),
                "best_f1": (float(cand_f1) if cand_f1 is not None else None),
            }

    if best_bundle is None:
        raise RuntimeError("No class-weight candidate was evaluated.")

    # Persist grid search summary
    if is_binary and cfg.class_weight_grid_enable:
        grid_rows_sorted = sorted(grid_rows, key=lambda d: d["score"], reverse=True)
        _save_json(
            {
                "selected_candidate": best_bundle["candidate"],
                "selected_weight_mode": _candidate_weight_desc(best_bundle["params"]),
                "selected_score": float(best_bundle["score"]),
                "selected_oof_best_threshold": float(best_bundle["tuned_thr"]),
                "selected_oof_best_f1": (
                    float(best_bundle["best_f1"]) if best_bundle["best_f1"] is not None else None
                ),
                "candidates": grid_rows_sorted,
            },
            os.path.join(metrics_dir, cfg.class_weight_grid_results_name),
        )

    print(
        f"[CW] Selected '{best_bundle['candidate']}' "
        f"({_candidate_weight_desc(best_bundle['params'])}) score={best_bundle['score']:.6f}"
    )

    # Use selected candidate outputs
    oof_pred = best_bundle["oof_pred"]
    test_pred_folds = best_bundle["test_pred_folds"]
    fold_scores = best_bundle["fold_scores"]
    fold_fi_dicts = best_bundle["fold_fi_dicts"]
    all_fold_errors = best_bundle["all_fold_errors"]
    fold_metrics_selected = best_bundle["fold_metrics"]
    tuned_thr = float(best_bundle["tuned_thr"])
    best_f1 = best_bundle["best_f1"]

    if cfg.save_fold_metrics and is_binary:
        for fold, m in sorted(fold_metrics_selected.items()):
            _save_json(m, os.path.join(metrics_dir, f"fold{fold}.json"))

    # -----------------------------
    # 6) OOF summary + threshold tuning
    # -----------------------------
    if is_binary:
        mean_auc = float(np.mean(fold_scores))
        std_auc = float(np.std(fold_scores))
        print(f"OOF AUC: mean={mean_auc:.6f}, std={std_auc:.6f}")

        if cfg.save_validation_errors and len(all_fold_errors) > 0:
            all_err = pd.concat(all_fold_errors, axis=0, ignore_index=True)
            all_wrong = all_err[all_err["error_type"] != "OK"].copy()
            all_wrong = all_wrong.sort_values(by=["confidence"], ascending=False)
            all_wrong.to_csv(os.path.join(errors_dir, "folds_all_wrong.csv"), index=False)
            summary = all_wrong["error_type"].value_counts().to_dict()
            _save_json({"folds_all_wrong_counts": summary}, os.path.join(metrics_dir, "folds_all_wrong_counts.json"))

    test_proba = np.mean(np.vstack(test_pred_folds), axis=0) if len(test_pred_folds) else np.zeros(len(test_base), dtype=np.float64)
    if is_binary and (not cfg.submit_proba):
        print(f"[OOF] best_thr={tuned_thr:.3f} best_F1={best_f1:.4f} (was thr={cfg.decision_threshold:.3f})")

    # -----------------------------
    # 7) Save submission (if sample submission exists; else fallback)
    # -----------------------------
    sample_path = cfg.sample_submission_path if os.path.exists(cfg.sample_submission_path) else (sub_fallback if os.path.exists(sub_fallback) else None)
    if sample_path:
        submission = pd.read_csv(sample_path)
    else:
        # fallback: create minimal submission with id if possible
        tid_col = _resolve_id_col(test, cfg.id_col_candidates)
        if tid_col and tid_col in test.columns:
            submission = pd.DataFrame({tid_col: test[tid_col].values})
        else:
            submission = pd.DataFrame({"id": np.arange(len(test_base))})
        submission[cfg.target] = 0

    save_thr = cfg.decision_threshold
    if cfg.submit_proba:
        submission[cfg.target] = test_proba
    else:
        # Binary label submission defaults to OOF-tuned threshold to align with F1 objective.
        if is_binary:
            save_thr = tuned_thr
        submission[cfg.target] = (test_proba >= save_thr).astype(int)

    try:
        submission.to_csv(cfg.output_path, index=False)
    except Exception as e:
        print(f"[WARN] failed to save submission to {cfg.output_path}: {e}")

    if (not cfg.submit_proba) and is_binary and cfg.save_fold_metrics:
        thr_meta = {
            "configured_threshold": float(cfg.decision_threshold),
            "tuned_threshold": float(tuned_thr),
            "saved_primary_threshold": float(save_thr),
            "save_primary_path": cfg.output_path,
            "selected_candidate": best_bundle["candidate"],
            "selected_weight_mode": _candidate_weight_desc(best_bundle["params"]),
            "selected_score": float(best_bundle["score"]),
        }
        _save_json(thr_meta, os.path.join(metrics_dir, "thresholds.json"))

    if (not cfg.submit_proba) and is_binary:
        # Also persist explicit threshold variants for reproducibility.
        out_tuned = cfg.output_path.replace(".csv", f"_thr{tuned_thr:.3f}.csv")
        submission_tuned = submission.copy()
        submission_tuned[cfg.target] = (test_proba >= tuned_thr).astype(int)
        try:
            submission_tuned.to_csv(out_tuned, index=False)
        except Exception as e:
            print(f"[WARN] failed to save tuned submission to {out_tuned}: {e}")

        out_cfg = cfg.output_path.replace(".csv", f"_thr{cfg.decision_threshold:.3f}.csv")
        submission_cfg = submission.copy()
        submission_cfg[cfg.target] = (test_proba >= cfg.decision_threshold).astype(int)
        try:
            submission_cfg.to_csv(out_cfg, index=False)
        except Exception as e:
            print(f"[WARN] failed to save configured-threshold submission to {out_cfg}: {e}")

    # -----------------------------
    # 7) Save OOF wrong samples (tuned threshold) & FI summary
    # -----------------------------
    if cfg.save_validation_errors and is_binary:
        oof_pred_label = (oof_pred >= tuned_thr).astype(int)
        oof_error_type = np.where(
            (y_full.to_numpy() == 0) & (oof_pred_label == 1),
            "FP",
            np.where((y_full.to_numpy() == 1) & (oof_pred_label == 0), "FN", "OK"),
        )
        oof_conf = np.where(oof_pred_label == 1, oof_pred, 1.0 - oof_pred)
        oof_margin = np.abs(oof_pred - tuned_thr)

        oof_base = pd.DataFrame(
            {
                "fold": -1,
                "row_index": train_base.index.to_numpy(),
                "y_true": y_full.to_numpy(),
                "y_pred_proba": oof_pred,
                "y_pred": oof_pred_label,
                "error_type": oof_error_type,
                "confidence": oof_conf,
                "margin_to_threshold": oof_margin,
            },
            index=train_base.index,
        )

        parts = [oof_base]
        if id_series is not None:
            parts.append(pd.DataFrame({"id": id_series.loc[train_base.index].to_numpy()}, index=train_base.index))

        extra_cols = [c for c in cfg.error_extra_cols_from_raw if c in train_raw.columns]
        if extra_cols:
            parts.append(train_raw.loc[train_base.index, extra_cols].copy())

        # include Stage-0 engineered features for context (stable across folds)
        # NOTE: fold-local (Stage-1) features are not globally consistent, so omitted here.
        stable_cols = [c for c in cfg.base_features if c in train_base.columns and c != cfg.target]
        if stable_cols:
            parts.append(train_base.loc[train_base.index, stable_cols].copy())

        oof_err_df = pd.concat(parts, axis=1)
        oof_err_df["_is_wrong"] = (oof_err_df["error_type"] != "OK").astype(int)
        oof_err_df = oof_err_df.sort_values(by=["_is_wrong", "confidence"], ascending=[False, False]).drop(columns=["_is_wrong"])

        _save_error_reports(oof_err_df, errors_dir, "oof", cfg)
        print(f"[Saved] validation wrong samples -> {errors_dir}")

        if cfg.save_fold_metrics:
            oof_m = _binary_metrics(y_full.to_numpy(), oof_pred, tuned_thr)
            _save_json(oof_m, os.path.join(metrics_dir, "oof.json"))

    if cfg.save_feature_importance and len(fold_fi_dicts) > 0:
        # union features across folds; fill missing with 0
        all_feats: List[str] = sorted({k for d in fold_fi_dicts for k in d.keys()})
        mat = np.zeros((len(fold_fi_dicts), len(all_feats)), dtype=np.float64)
        for i, d in enumerate(fold_fi_dicts):
            for j, f in enumerate(all_feats):
                if f in d:
                    mat[i, j] = float(d[f])

        fi_mean = mat.mean(axis=0)
        fi_std = mat.std(axis=0)
        fi_df = pd.DataFrame({"feature": all_feats, "importance_mean": fi_mean, "importance_std": fi_std}).sort_values("importance_mean", ascending=False)

        fi_csv_path = os.path.join(fi_dir, "feature_importance.csv")
        fi_png_path = os.path.join(fi_dir, "feature_importance.png")
        fi_df.to_csv(fi_csv_path, index=False)
        _plot_feature_importance(fi_df, cfg.fi_top_n_plot, fi_png_path)

        print(f"[Saved] feature importance csv -> {fi_csv_path}")
        print(f"[Saved] feature importance plot -> {fi_png_path}")
    elif cfg.save_feature_importance:
        print("[WARN] No feature importance collected (fold_fi empty).")


if __name__ == "__main__":
    main()
