# catboost_kfold.py
# CatBoost 기반 베이스라인 (K-Fold 적용 버전) - CFG 적용(상단 설정 집중형)
# + Validation 오류(틀린 샘플) 추적/저장
# + Feature Importance 집계/시각화(폴드 평균)

from __future__ import annotations

VERSION = "v1_multihot_autofeatures_sigfnfix"

import os
import re
import json
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

    auto_expand_prefixes: Tuple[str, ...] = ("desired_job_has_", "desired_job_except_has_")


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

        s = df[col].fillna("").astype("string").str.strip()
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
        df["major1_1_is_data"] = _is_data_series(df["major1_1"]).astype(np.int8)
    else:
        df["major1_1_is_data"] = np.int8(0)

    if "major1_2" in df.columns:
        df["major1_2_is_data"] = _is_data_series(df["major1_2"]).astype(np.int8)
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
        m2 = df["major1_2"].fillna("").astype("string").str.strip()
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

def main() -> None:
    warnings.filterwarnings("ignore")
    cfg = CFG()

    # -----------------------------
    # 1) Load (원본 보관: 오류 분석용)
    # -----------------------------
    train = pd.read_csv(cfg.train_path)
    test = pd.read_csv(cfg.test_path)
    train_raw = train.copy()

    raw_cols = list(train.columns)
    id_col = _resolve_id_col(train_raw, cfg.id_col_candidates)
    id_series = train_raw[id_col] if id_col is not None else None

    # -----------------------------
    # 2) Feature engineering 먼저 수행
    # -----------------------------
    for df in (train, test):
        _make_is_major_it(df, cfg)
        _make_completed_semester_features(df, cfg)
        _make_certificate_features(df, cfg)
        _make_basic_text_presence_features(df, cfg)
        _make_interested_company_semantic_features(df, cfg)
        _make_major_split_features(df, cfg)
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

    top_school, school_freq = _fit_school1_mapper(train, cfg)
    _apply_school1_mapper(train, top_school, school_freq, cfg)
    _apply_school1_mapper(test,  top_school, school_freq, cfg)

    # interested_company primary top-k / freq mapper
    top_ic, ic_freq = _fit_ic_primary_mapper(train, cfg)
    _apply_ic_primary_mapper(train, top_ic, ic_freq, cfg)
    _apply_ic_primary_mapper(test,  top_ic, ic_freq, cfg)

    if cfg.time_col in train.columns:
        time_strategy = _infer_time_input_strategy(train[cfg.time_col], cfg)
    else:
        time_strategy = TimeInputStrategy("drop")

    for df in (train, test):
        _apply_time_input_strategy(df, time_strategy, cfg)
    # time_input bucket 만들기: time_input이 numeric일 때만
    for df in (train, test):
        if "time_input" in df.columns and pd.api.types.is_numeric_dtype(df["time_input"]):
            df["time_input_bucket"] = pd.cut(
                df["time_input"],
                bins=[-np.inf, 1, 2, 3, 4, np.inf],
                labels=["<=1", "1-2", "2-3", "3-4", ">4"],
                include_lowest=True,
            ).astype("string").fillna("UNK")
    # time bucket 기반 상호작용 categorical 생성
    for df in (train, test):
        _make_time_interaction_features(df, cfg)


    # time_input이 datetime 전략으로 파생 피처가 생겼다면, base_features에 반영
    base_features = list(cfg.base_features)
    features = [f for f in base_features if f != cfg.time_col]
    if "time_input_bucket" in train.columns and "time_input_bucket" not in features:
        features.append("time_input_bucket")
    if time_strategy.mode in ("numeric", "categorical"):
        if cfg.time_col in base_features:
            insert_idx = base_features.index(cfg.time_col)
            features.insert(insert_idx, cfg.time_col)
        else:
            features.append(cfg.time_col)
    elif time_strategy.mode == "datetime":
        features.extend(list(time_strategy.created_cols))

    # -----------------------------
    # 3) Drop columns with too many missings (train 기준)
    #    - 단, 우리가 사용하는 features/target은 보호
    # -----------------------------
    protect = set(features + [cfg.target])
    missing_ratio = train.drop(columns=[cfg.target], errors="ignore").isnull().mean()
    columns_to_drop = [
        c for c in missing_ratio[missing_ratio > cfg.drop_missing_ratio_gt].index.tolist() if c not in protect
    ]
    train = train.drop(columns=columns_to_drop, errors="ignore")
    test = test.drop(columns=columns_to_drop, errors="ignore")

    # -----------------------------
    # 4) Select features/target
    # -----------------------------
    X_train, y_train, X_test, features = _select_features(train, test, cfg.target, features, auto_prefixes=cfg.auto_expand_prefixes)

    # -----------------------------
    # 5) Categorical vs Text split
    # -----------------------------
    force_categorical = list(cfg.force_categorical)
    text_cols = [c for c in cfg.text_features if c in X_train.columns]
    categorical_cols = _infer_categorical_columns(X_train, forced=force_categorical)
    categorical_cols = [c for c in categorical_cols if c not in text_cols]

    MISSING_CAT = "__MISSING__"

    for c in categorical_cols:
        X_train[c] = X_train[c].astype("object")
        X_test[c]  = X_test[c].astype("object")

        X_train[c] = X_train[c].where(pd.notna(X_train[c]), MISSING_CAT)
        X_test[c]  = X_test[c].where(pd.notna(X_test[c]),  MISSING_CAT)

    for c in text_cols:
        X_train[c] = (
            X_train[c].fillna("").astype("string").str.replace(cfg.text_ws_regex, " ", regex=True).str.strip()
        )
        X_test[c] = (
            X_test[c].fillna("").astype("string").str.replace(cfg.text_ws_regex, " ", regex=True).str.strip()
        )

    cat_feature_indices = [X_train.columns.get_loc(c) for c in categorical_cols]
    text_feature_indices = [X_train.columns.get_loc(c) for c in text_cols]

    # -----------------------------
    # 6) K-Fold Training
    # -----------------------------
    if y_train.dtype.kind not in ("i", "u", "b"):
        y_train = pd.to_numeric(y_train, errors="coerce").fillna(0).astype(int)

    is_binary = (y_train.nunique() == 2)
    splitter = (
        StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
        if (cfg.use_stratified_if_binary and is_binary)
        else KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    )

    oof_pred = np.zeros(len(X_train), dtype=np.float64)
    test_pred_folds: List[np.ndarray] = []
    best_trees: List[int] = []
    fold_scores: List[float] = []

    # feature importance
    fold_fi: List[np.ndarray] = []
    feature_names = X_train.columns.tolist()

    # validation errors
    all_fold_errors: List[pd.DataFrame] = []
    errors_dir = os.path.join(cfg.artifacts_dir, cfg.errors_subdir)
    metrics_dir = os.path.join(cfg.artifacts_dir, cfg.metrics_subdir)

    params = _build_model_params(cfg)

    # test pool (fold마다 동일)
    test_pool = Pool(
        X_test,
        cat_features=cat_feature_indices,
        text_features=(text_feature_indices if text_feature_indices else None),
    )

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X_train, y_train), start=1):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        train_pool = Pool(
            X_tr,
            y_tr,
            cat_features=cat_feature_indices,
            text_features=(text_feature_indices if text_feature_indices else None),
        )
        valid_pool = Pool(
            X_va,
            y_va,
            cat_features=cat_feature_indices,
            text_features=(text_feature_indices if text_feature_indices else None),
        )

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        # Validation prediction (proba of class 1)
        va_proba = model.predict_proba(valid_pool)[:, 1]
        oof_pred[va_idx] = va_proba

        # Fold AUC
        if is_binary:
            auc = roc_auc_score(y_va, va_proba)
            fold_scores.append(float(auc))
            print(f"[Fold {fold}/{cfg.n_splits}] AUC = {auc:.6f} | trees = {model.tree_count_}")

            # Threshold-based metrics (for error inspection)
            m = _binary_metrics(y_va.to_numpy(), va_proba, cfg.decision_threshold)
            print(
                f"[Fold {fold}] thr={cfg.decision_threshold:.3f} | "
                f"F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} Acc={m['accuracy']:.4f} | "
                f"FP={int(m['fp'])} FN={int(m['fn'])}"
            )
            if cfg.save_fold_metrics:
                _save_json(m, os.path.join(metrics_dir, f"fold{fold}.json"))
        else:
            print(f"[Fold {fold}/{cfg.n_splits}] done | trees = {model.tree_count_}")

        best_trees.append(int(model.tree_count_))

        # Test prediction for ensembling
        te_proba = model.predict_proba(test_pool)[:, 1]
        test_pred_folds.append(te_proba.astype(np.float64))

        # -----------------------------
        # (A) Validation wrong samples tracking
        # -----------------------------
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
            all_fold_errors.append(err_df)

            # 저장 (fold별)
            _save_error_reports(err_df, errors_dir, f"fold{fold}", cfg)

            # 콘솔에서 "확신이 큰 오답" 일부 출력
            wrong = err_df[err_df["error_type"] != "OK"].head(cfg.error_print_topk)
            if len(wrong) > 0:
                cols_to_show = [c for c in ["fold", "row_index", "id", "y_true", "y_pred", "y_pred_proba", "error_type", "confidence", "margin_to_threshold"] if c in wrong.columns]
                print(f"[Fold {fold}] Top-{min(cfg.error_print_topk, len(wrong))} confident wrong samples:")
                print(wrong[cols_to_show].to_string(index=False))
            else:
                print(f"[Fold {fold}] No wrong samples under threshold={cfg.decision_threshold}.")

        # -----------------------------
        # (B) Feature importance
        # -----------------------------
        if cfg.save_feature_importance:
            try:
                fi = model.get_feature_importance(train_pool, type=cfg.feature_importance_type)
            except Exception:
                fi = model.get_feature_importance(type=cfg.feature_importance_type)

            fi = np.asarray(fi, dtype=np.float64)
            if fi.shape[0] == len(feature_names):
                fold_fi.append(fi)
            else:
                print(
                    f"[WARN] feature importance length mismatch: got {fi.shape[0]}, expected {len(feature_names)}. Skip this fold."
                )

    if is_binary:
        mean_auc = float(np.mean(fold_scores))
        std_auc = float(np.std(fold_scores))
        print(f"OOF AUC: mean={mean_auc:.6f}, std={std_auc:.6f}")

        # Fold별 오류를 한 파일로 모아 저장(오답만)
        if cfg.save_validation_errors and len(all_fold_errors) > 0:
            all_err = pd.concat(all_fold_errors, axis=0, ignore_index=True)
            all_wrong = all_err[all_err["error_type"] != "OK"].copy()
            all_wrong = all_wrong.sort_values(by=["confidence"], ascending=False)
            _ensure_dir(errors_dir)
            all_wrong.to_csv(os.path.join(errors_dir, "folds_all_wrong.csv"), index=False)
            # 간단 요약(counts)
            summary = all_wrong["error_type"].value_counts().to_dict()
            _save_json({"folds_all_wrong_counts": summary}, os.path.join(metrics_dir, "folds_all_wrong_counts.json"))

    # Ensemble test prediction (mean of folds)
    test_proba = np.mean(np.vstack(test_pred_folds), axis=0)

    # -----------------------------
    # (Optional) Tune threshold on OOF (before saving submissions)
    # -----------------------------
    tuned_thr = cfg.decision_threshold
    best_f1 = None
    if is_binary and (not cfg.submit_proba):
        best_thr, best_f1 = _best_threshold_by_f1(y_train.to_numpy(), oof_pred)
        tuned_thr = float(best_thr)
        print(f"[OOF] best_thr={tuned_thr:.3f} best_F1={best_f1:.4f} (was thr={cfg.decision_threshold:.3f})")

    # -----------------------------
    # 7) Save submission
    # -----------------------------
    submission = pd.read_csv(cfg.sample_submission_path)
    if cfg.submit_proba:
        submission[cfg.target] = test_proba
    else:
        submission[cfg.target] = (test_proba >= cfg.decision_threshold).astype(int)
    submission.to_csv(cfg.output_path, index=False)
    if (not cfg.submit_proba) and is_binary:
        submission2 = pd.read_csv(cfg.sample_submission_path)
        submission2[cfg.target] = (test_proba >= tuned_thr).astype(int)
        submission2.to_csv(cfg.output_path.replace(".csv", f"_thr{tuned_thr:.3f}.csv"), index=False)

    # -----------------------------
    # 8) Save OOF wrong samples & FI summary
    # -----------------------------
    if cfg.save_validation_errors and is_binary:
        # OOF 기반 전체 오답
        oof_pred_label = (oof_pred >= tuned_thr ).astype(int)
        oof_error_type = np.where(
            (y_train.to_numpy() == 0) & (oof_pred_label == 1),
            "FP",
            np.where((y_train.to_numpy() == 1) & (oof_pred_label == 0), "FN", "OK"),
        )
        oof_conf = np.where(oof_pred_label == 1, oof_pred, 1.0 - oof_pred)
        oof_margin = np.abs(oof_pred - tuned_thr )

        oof_base = pd.DataFrame(
            {
                "fold": -1,
                "row_index": X_train.index.to_numpy(),
                "y_true": y_train.to_numpy(),
                "y_pred_proba": oof_pred,
                "y_pred": oof_pred_label,
                "error_type": oof_error_type,
                "confidence": oof_conf,
                "margin_to_threshold": oof_margin,
            },
            index=X_train.index,
        )

        parts = [oof_base]
        if id_series is not None:
            parts.append(pd.DataFrame({"id": id_series.loc[X_train.index].to_numpy()}, index=X_train.index))
        extra_cols = [c for c in cfg.error_extra_cols_from_raw if c in train_raw.columns]
        if extra_cols:
            parts.append(train_raw.loc[X_train.index, extra_cols].copy())
        parts.append(X_train.copy())

        oof_err_df = pd.concat(parts, axis=1)
        oof_err_df["_is_wrong"] = (oof_err_df["error_type"] != "OK").astype(int)
        oof_err_df = oof_err_df.sort_values(by=["_is_wrong", "confidence"], ascending=[False, False]).drop(columns=["_is_wrong"])

        _save_error_reports(oof_err_df, errors_dir, "oof", cfg)
        print(f"[Saved] validation wrong samples -> {errors_dir}")

        # OOF threshold-based metrics 저장
        if cfg.save_fold_metrics:
            oof_m = _binary_metrics(y_train.to_numpy(), oof_pred, tuned_thr)
            _save_json(oof_m, os.path.join(metrics_dir, "oof.json"))

    if cfg.save_feature_importance and len(fold_fi) > 0:
        fi_dir = os.path.join(cfg.artifacts_dir, cfg.fi_subdir)
        _ensure_dir(fi_dir)

        fi_mat = np.vstack(fold_fi)
        fi_mean = fi_mat.mean(axis=0)
        fi_std = fi_mat.std(axis=0)

        fi_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": fi_mean,
                "importance_std": fi_std,
            }
        ).sort_values("importance_mean", ascending=False)

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
