import os
import random
import json
import re
import hashlib
from datetime import datetime
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np # type: ignore
import pandas as pd # type: ignore

from sklearn.model_selection import StratifiedKFold # type: ignore
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer # type: ignore

import lightgbm as lgb # type: ignore
from catboost import CatBoostClassifier # type: ignore

import hydra # type: ignore
from hydra.utils import to_absolute_path # type: ignore
from hydra.core.hydra_config import HydraConfig # type: ignore
from omegaconf import DictConfig, OmegaConf # type: ignore


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


# ---------------------------
# Multi-label (comma-separated) parsing -> multi-hot features
# ---------------------------
_BRACKET_OPEN = set(["(", "[", "{"])
_BRACKET_CLOSE = {")": "(", "]": "[", "}": "{"}

_INTERESTED_COMPANY_COL = "interested_company"
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", flags=re.UNICODE)

# expected_domain uses labels like "M. 전문, 과학 및 기술 서비스업"
# where commas can be part of a single label. We merge tokens back using
# the leading industry-code prefix pattern (e.g., "M. ").
_EXPECTED_DOMAIN_COL = "expected_domain"
_EXPECTED_DOMAIN_PREFIX_RE = re.compile(r"^[A-Z]\.\s*")


def _merge_expected_domain_tokens(tokens: List[str]) -> List[str]:
    """Merge split pieces for expected_domain.

    We first split on commas, but labels themselves may contain commas.
    Every new label starts with an industry-code prefix like "M. ".
    Any subsequent token that does NOT start with the prefix is treated as a
    continuation of the previous label and merged back with ", ".
    """
    merged: List[str] = []
    buf: Optional[str] = None

    for t in tokens:
        if _EXPECTED_DOMAIN_PREFIX_RE.match(t):
            if buf is not None:
                merged.append(buf)
            buf = t
        else:
            if buf is None:
                merged.append(t)
            else:
                buf = f"{buf}, {t}"

    if buf is not None:
        merged.append(buf)

    return merged


def _split_outside_brackets(text: str, sep: str = ",") -> List[str]:
    """Split by `sep` but ignore separators inside (), [], {}.

    Example:
      "A, B(1,2), C" -> ["A", "B(1,2)", "C"]
    """
    if text is None:
        return []
    s = str(text)
    if s.strip() == "":
        return []
    parts: List[str] = []
    buf: List[str] = []
    stack: List[str] = []
    for ch in s:
        if ch in _BRACKET_OPEN:
            stack.append(ch)
            buf.append(ch)
            continue
        if ch in _BRACKET_CLOSE:
            # pop only if matching opener exists
            if stack and stack[-1] == _BRACKET_CLOSE[ch]:
                stack.pop()
            buf.append(ch)
            continue
        if ch == sep and not stack:
            part = "".join(buf).strip()
            if part != "":
                parts.append(part)
            buf = []
            continue
        # also treat fullwidth comma as separator if sep is ','
        if sep == "," and ch == "，" and not stack:
            part = "".join(buf).strip()
            if part != "":
                parts.append(part)
            buf = []
            continue
        buf.append(ch)

    last = "".join(buf).strip()
    if last != "":
        parts.append(last)
    return parts


def _parse_multilabel_cell(v: Any, sep: str = ",", col_name: Optional[str] = None) -> List[str]:
    """Parse a single cell into token list (dedup, keep order)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    s = str(v).strip()
    if s == "":
        return []

    # previous_class_*: keep only the 4-digit class codes (e.g., "0001") to reduce dimensionality
    if col_name is not None and str(col_name).startswith("previous_class_"):
        # primary pattern: 4 digits right before a colon (handles "0001 : ...")
        codes = re.findall(r"(\d{4})(?=\s*:)", s)
        # fallback: any 4-digit code with leading zero (avoid years like 2023)
        if not codes:
            codes = re.findall(r"\b0\d{3}\b", s)

        # dedup, keep order
        cleaned_codes: List[str] = []
        seen = set()
        for c in codes:
            if c not in seen:
                cleaned_codes.append(c)
                seen.add(c)
        return cleaned_codes
    toks = _split_outside_brackets(s, sep=sep)
    # normalize whitespace
    cleaned: List[str] = []
    seen = set()
    for t in toks:
        tt = re.sub(r"\s+", " ", str(t).strip())
        if tt == "":
            continue
        if col_name == _INTERESTED_COMPANY_COL and _PUNCT_ONLY_RE.match(tt):
            continue
        if tt not in seen:
            cleaned.append(tt)
            seen.add(tt)

    # expected_domain: merge tokens back when commas are part of a label
    if col_name == _EXPECTED_DOMAIN_COL:
        cleaned = _merge_expected_domain_tokens(cleaned)

    return cleaned


def _safe_multihot_feature_name(prefix: str, token: str, used: set, max_len: int = 120) -> str:
    """Create a safe, unique column name for multi-hot features."""
    raw = f"{prefix}__{token}"
    # replace whitespace with underscore
    name = re.sub(r"\s+", "_", raw.strip())
    # keep alnum/_ and Korean; map others to underscore
    name = re.sub(r"[^0-9a-zA-Z가-힣_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    # ensure non-empty
    suffix = hashlib.md5(f"{prefix}::{token}".encode("utf-8")).hexdigest()[:8]
    if name == "":
        name = f"{prefix}__tok__{suffix}"
    # cap length & keep uniqueness
    if len(name) > max_len:
        name = name[: max_len - 11].rstrip("_") + f"__{suffix}"
    if name in used:
        name = name[: max_len - 11].rstrip("_") + f"__{suffix}"
    used.add(name)
    return name


def expand_multilabel_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: List[str],
    sep: str = ",",
    min_freq: int = 3,
    top_k: Optional[int] = 50,
    add_other: bool = True,
    add_count: bool = True,
    drop_original: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Expand comma-separated multi-label columns into (top-K) multi-hot columns.

    - Vocabulary is built ONLY from train_df (no leakage).
    - Tokens with freq < min_freq are dropped from explicit columns.
    - If top_k is not None, keep only the most frequent top_k tokens.
    - Optionally adds:
        * <col>__ml_count : number of tokens in the cell
        * <col>__ml_other : 1 if any token is outside the selected vocab
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    info: Dict[str, Any] = {}
    used_names = set(train_df.columns).union(set(test_df.columns))

    for col in cols:
        if col not in train_df.columns:
            continue

        tr_lists = train_df[col].apply(lambda v: _parse_multilabel_cell(v, sep=sep, col_name=col))
        if col in test_df.columns:
            te_lists = test_df[col].apply(lambda v: _parse_multilabel_cell(v, sep=sep, col_name=col))
        else:
            te_lists = pd.Series([[] for _ in range(len(test_df))], index=test_df.index)

        # token frequency (train only)
        exploded = tr_lists.explode()
        vc = exploded.dropna().astype(str).value_counts()
        vc = vc[vc.index.astype(str).str.len() > 0]
        # apply min_freq and top_k
        selected = vc[vc >= int(min_freq)].index.astype(str).tolist()
        if top_k is not None:
            selected = selected[: int(top_k)]

        selected_set = set(selected)

        # build column names (stable + unique)
        token_to_col: Dict[str, str] = {}
        colnames: List[str] = []
        for tok in selected:
            feat = _safe_multihot_feature_name(col, tok, used=used_names)
            token_to_col[tok] = feat
            colnames.append(feat)

        # multi-hot matrix via MultiLabelBinarizer (small data, fine)
        if selected:
            mlb = MultiLabelBinarizer(classes=selected)
            # selected_set 밖 토큰 제거 → unknown class 경고 제거
            tr_lists_sel = tr_lists.apply(lambda lst: [t for t in lst if t in selected_set])
            te_lists_sel = te_lists.apply(lambda lst: [t for t in lst if t in selected_set])

            tr_bin = mlb.fit_transform(tr_lists_sel.tolist())
            te_bin = mlb.transform(te_lists_sel.tolist())

            tr_ohe = pd.DataFrame(tr_bin, index=train_df.index, columns=colnames).astype(np.int8)
            te_ohe = pd.DataFrame(te_bin, index=test_df.index, columns=colnames).astype(np.int8)

            train_df = pd.concat([train_df.drop(columns=[col]) if drop_original else train_df, tr_ohe], axis=1)
            if col in test_df.columns and drop_original:
                test_df = test_df.drop(columns=[col])
            test_df = pd.concat([test_df, te_ohe], axis=1)
        else:
            # no selected tokens -> optionally just drop original
            if drop_original:
                train_df = train_df.drop(columns=[col])
                if col in test_df.columns:
                    test_df = test_df.drop(columns=[col])

        # optional extra features
        if add_count:
            train_df[f"{col}__ml_count"] = tr_lists.apply(len).astype(np.int16)
            test_df[f"{col}__ml_count"] = te_lists.apply(len).astype(np.int16)

        if add_other:
            def _has_other(lst: List[str]) -> int:
                if not lst:
                    return 0
                if not selected_set:
                    return 1  # everything is other if vocab is empty
                for t in lst:
                    if t not in selected_set:
                        return 1
                return 0

            train_df[f"{col}__ml_other"] = tr_lists.apply(_has_other).astype(np.int8)
            test_df[f"{col}__ml_other"] = te_lists.apply(_has_other).astype(np.int8)

        # meta
        info[col] = {
            "min_freq": int(min_freq),
            "top_k": int(top_k) if top_k is not None else None,
            "n_unique_tokens_train": int(len(vc)),
            "n_selected_tokens": int(len(selected)),
            "selected_tokens_preview": selected[:30],
            "coverage_any": float((tr_lists.apply(len) > 0).mean()),
            "coverage_selected": float((tr_lists.apply(lambda lst: any(t in selected_set for t in lst)) if selected else (tr_lists.apply(len) > 0)).mean()),
        }

    return train_df, test_df, info


def apply_rare_bucket(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_cols: List[str],
    min_freq: int = 3,
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Rare bucket 전처리: train에서 빈도 < min_freq 인 카테고리를 __RARE__로 합칩니다.

    - train의 value_counts 기준으로 rare를 정의 (데이터 누수 방지)
    - 결측은 __MISSING__으로 별도 처리 (rare로 합치지 않음)
    - test에서 train의 rare에 해당하는 값은 __RARE__로 매핑
      (그 외 train에 없던 값은 이후 make_categorical_safe에서 __UNKNOWN__ 처리)
    """
    if min_freq is None or int(min_freq) <= 1:
        return train_df, test_df, {}

    train_df = train_df.copy()
    test_df = test_df.copy()

    exclude = set(exclude_cols or [])
    info: Dict[str, Dict[str, Any]] = {}

    for col in categorical_cols:
        if col in exclude or col not in train_df.columns:
            continue

        tr = normalize_blank_series(train_df[col])
        tr = tr.where(tr.notna(), "__MISSING__").astype(str)

        vc = tr.value_counts(dropna=False)
        rare_vals = vc[vc < int(min_freq)].index.astype(str).tolist() # type: ignore
        rare_vals = [v for v in rare_vals if v != "__MISSING__"]  # 결측 토큰은 유지

        if rare_vals:
            train_df[col] = tr.where(~tr.isin(rare_vals), "__RARE__")
            if col in test_df.columns:
                te = normalize_blank_series(test_df[col])
                te = te.where(te.notna(), "__MISSING__").astype(str)
                test_df[col] = te.where(~te.isin(rare_vals), "__RARE__")
        else:
            # rare 없으면 문자열 정규화만 반영(일관성)
            train_df[col] = tr
            if col in test_df.columns:
                te = normalize_blank_series(test_df[col])
                te = te.where(te.notna(), "__MISSING__").astype(str)
                test_df[col] = te

        info[col] = {
            "min_freq": int(min_freq),
            "n_rare": int(len(rare_vals)),
            "example_rare": rare_vals[:20],
        }

    return train_df, test_df, info


# "4-1" / "4/2" / "4학년 1학기" 같은 표기를 학기수로 환산(선택)
_GRADE_TERM_RE = re.compile(r"^\s*(\d)\s*[-/]\s*([12])\s*$")
_GRADE_TERM_KO_RE = re.compile(r"^\s*(\d)\s*학년\s*([12])\s*학기\s*$")

def clean_completed_semester_series(
    s: pd.Series,
    min_val: int = 0,
    max_val: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """
    completed_semester 정제:
    - 정상 범위(min_val~max_val)의 '정수 학기수'만 남김
    - 연도/학기 형태(예: 2020.02, 20241)나 비정상 값은 NaN 처리
    - 대신 invalid flag 피처를 생성
    """
    raw = s.copy()
    invalid = pd.Series(0, index=s.index, dtype="int8")

    def _parse_one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan, 0

        # 숫자형 처리
        if isinstance(v, (int, np.integer)):
            x = float(v)
        elif isinstance(v, (float, np.floating)):
            x = float(v)
        else:
            ss = str(v).strip()
            if ss == "":
                return np.nan, 0

            # "8학기" / "8 학기" -> 8
            ss2 = re.sub(r"\s+", "", ss)
            m = re.match(r"^(\d{1,2})학기$", ss2)
            if m:
                x = float(m.group(1))
            else:
                # "4-1" 같은 표기 -> (4-1)*2+1 = 7
                m2 = _GRADE_TERM_RE.match(ss2)
                if m2:
                    g = int(m2.group(1))
                    t = int(m2.group(2))
                    # 학년이 1~6 정도인 경우만 유효 처리
                    if 1 <= g <= 6 and t in (1, 2):
                        x = float((g - 1) * 2 + t)
                    else:
                        return np.nan, 1
                else:
                    m3 = _GRADE_TERM_KO_RE.match(ss2)
                    if m3:
                        g = int(m3.group(1))
                        t = int(m3.group(2))
                        if 1 <= g <= 6 and t in (1, 2):
                            x = float((g - 1) * 2 + t)
                        else:
                            return np.nan, 1
                    else:
                        # 일반 숫자 변환 시도 ("8", "8.0" 등)
                        try:
                            x = float(ss2)
                        except Exception:
                            return np.nan, 1

        # 여기부터 “이상치 컷”
        # 1000 이상은 대부분 연도/학기 입력으로 보고 NaN
        if x >= 1000:
            return np.nan, 1

        # 정상 범위 밖이면 NaN(또는 clip로 바꿀 수도 있음)
        if x < min_val or x > max_val:
            return np.nan, 1

        # 정수에 가까우면 반올림해서 int로
        xr = int(round(x))
        if abs(x - xr) > 1e-6:
            # 8.3 같은 애매한 값은 NaN 처리
            return np.nan, 1

        return float(xr), 0

    cleaned_vals = []
    invalid_flags = []

    for v in raw.tolist():
        cv, iv = _parse_one(v)
        cleaned_vals.append(cv)
        invalid_flags.append(iv)

    cleaned = pd.Series(cleaned_vals, index=s.index, dtype="float32")
    invalid = pd.Series(invalid_flags, index=s.index, dtype="int8")

    return cleaned, invalid





def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """대회 컬럼 구성에 맞춘 최소 파생피처."""
    df = df.copy()

    # completed_semester 정제 + invalid flag
    if "completed_semester" in df.columns:
        cleaned, invalid = clean_completed_semester_series(df["completed_semester"], min_val=0, max_val=20)
        df["completed_semester"] = cleaned
        df["completed_semester_invalid"] = invalid

        # (선택) 버킷 피처: 너무 과하면 빼도 됨
        # df["completed_semester_bucket"] = pd.cut(
        #     df["completed_semester"],
        #     bins=[-0.1, 2, 4, 6, 8, 10, 20],
        #     labels=["0-2", "3-4", "5-6", "7-8", "9-10", "11+"],
        # ).astype("object")

    # major_field 기반 파생피처 (존재할 때만)
    if "major_field" in df.columns:
        s = df["major_field"].astype("object")
        s = normalize_blank_series(s)  # 공백/빈 문자열 -> NaN
        df["major_field"] = s
        df["is_major_it"] = s.fillna("").astype(str).str.contains("IT", regex=False).astype(int)

    # 응답 성실도
    base_cols = [c for c in df.columns if c not in ["completed"]]  # target 제외
    df["__filled_cnt"] = df[base_cols].notna().sum(axis=1) # type: ignore
    df["__filled_ratio"] = df["__filled_cnt"] / len(base_cols)

    # 자유서술 길이(예: incumbents_lecture_scale_reason)
    col = "incumbents_lecture_scale_reason"
    if col in df.columns:
        s = df[col].fillna("").astype(str)
        df[col + "__len"] = s.str.len()
        df[col + "__n_words"] = s.str.split().str.len()

    # previous_class 요약(코드 존재 여부/개수)
    prev_cols = [c for c in df.columns if c.startswith("previous_class_")]
    if prev_cols:
        def count_codes(row):
            txt = " ".join([str(x) for x in row if pd.notna(x)])
            return len(re.findall(r"\b\d{4}\b", txt))
        df["prev_class__code_cnt"] = df[prev_cols].apply(lambda r: count_codes(r.values), axis=1)
        df["prev_class__has_code"] = (df["prev_class__code_cnt"] > 0).astype(int)

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


def add_missing_presence_flags_and_drop(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float,
    suffix: str = "__present",
    skip_all_missing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Add presence flags for high-missing columns, then drop them.

    - Compute missing rate on train_df.
    - For columns with missing_rate > threshold:
        * add <col>{suffix} = 1 if value is present else 0
        * (optionally) skip columns that are 100% missing in train_df (flag would be constant 0)
        * drop the original high-missing columns.

    Returns:
        train_df2, test_df2, dropped_cols, added_flag_cols
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    miss = train_df.isna().mean()
    drop_cols = miss[miss > threshold].index.tolist()

    added_flags: List[str] = []
    for col in drop_cols:
        # If the column is entirely missing in train, its presence flag is constant 0 -> usually useless
        if skip_all_missing and float(miss.get(col, 0.0)) >= 1.0:
            continue

        new_col = f"{col}{suffix}"
        if new_col in train_df.columns:
            continue

        if col in train_df.columns:
            train_df[new_col] = train_df[col].notna().astype(np.int8)
        else:
            train_df[new_col] = np.int8(0)

        if col in test_df.columns:
            test_df[new_col] = test_df[col].notna().astype(np.int8)
        else:
            test_df[new_col] = np.int8(0)

        added_flags.append(new_col)

    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols, errors="ignore")
    return train_df, test_df, drop_cols, added_flags


def drop_constant_cols(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """train 기준 nunique<=1 인 상수 컬럼 드랍."""
    nunique = train_df.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist() # type: ignore
    return train_df.drop(columns=const_cols), test_df.drop(columns=const_cols, errors="ignore"), const_cols


def infer_numeric_categorical_cols(
    train_df: pd.DataFrame,
    max_unique: int,
    include_bool: bool,
    exclude: List[str],
    skip_binary_numeric: bool = True,
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
            # 0/1 같은 indicator(=멀티핫 포함)는 numeric으로 두는 게 안전함
            if skip_binary_numeric and nuniq <= 2:
                continue
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

def preprocess_fold_fit(
    cfg: DictConfig,
    X_tr: pd.DataFrame,
    X_va: Optional[pd.DataFrame] = None,
    test: Optional[pd.DataFrame] = None,
    return_meta: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], List[str], Dict[str, Any]]:
    """Fold-fit 전처리 (leakage-free).

    규칙:
      - (fit) X_tr 만 사용해 분포 기반 전처리 기준을 생성
          * multi-label vocab(top_k/min_freq)
          * drop_missing/drop_constant 기준
          * categorical_numeric 후보 결정
          * rare bucket 기준
          * category map
      - (transform) X_va/test 는 위 기준으로 변환만 수행

    반환:
      X_tr_p, X_va_p, test_p, cat_cols_p, meta

    meta는 return_meta=True일 때만 풍부하게 채웁니다(Optuna 등 반복 호출 성능 고려).
    """

    X_tr = X_tr.copy()
    X_va = None if X_va is None else X_va.copy()
    test = None if test is None else test.copy()

    # ---- 0) val/test를 한 번에 변환하기 위한 other 구성 (index 충돌 방지)
    parts = []
    va_idx = None
    te_idx = None

    if X_va is not None:
        va_idx = X_va.index
        X_va.index = X_va.index.map(lambda i: f"va__{i}")
        parts.append(X_va)
    if test is not None:
        te_idx = test.index
        test.index = test.index.map(lambda i: f"te__{i}")
        parts.append(test)

    other = pd.concat(parts, axis=0) if parts else None

    meta: Dict[str, Any] = {}

    # ---- 1) Multi-label -> multi-hot (vocab은 X_tr에서만 생성)
    multilabel_enabled = bool(_select(cfg, "data.multilabel.enable", False))
    ml_cols = list(_select(cfg, "data.multilabel.cols", [])) or []
    multilabel_info: Dict[str, Any] = {}

    if multilabel_enabled and ml_cols:
        top_k_val = _select(cfg, "data.multilabel.top_k", 50)
        top_k_val = None if top_k_val is None else int(top_k_val)

        if other is None:
            X_tr, _, multilabel_info = expand_multilabel_columns(
                train_df=X_tr,
                test_df=X_tr.iloc[0:0].copy(), # type: ignore
                cols=ml_cols,
                sep=str(_select(cfg, "data.multilabel.sep", ",")),
                min_freq=int(_select(cfg, "data.multilabel.min_freq", 3)),
                top_k=top_k_val,
                add_other=bool(_select(cfg, "data.multilabel.add_other", True)),
                add_count=bool(_select(cfg, "data.multilabel.add_count", True)),
                drop_original=bool(_select(cfg, "data.multilabel.drop_original", True)),
            )
        else:
            X_tr, other, multilabel_info = expand_multilabel_columns(
                train_df=X_tr,
                test_df=other,
                cols=ml_cols,
                sep=str(_select(cfg, "data.multilabel.sep", ",")),
                min_freq=int(_select(cfg, "data.multilabel.min_freq", 3)),
                top_k=top_k_val,
                add_other=bool(_select(cfg, "data.multilabel.add_other", True)),
                add_count=bool(_select(cfg, "data.multilabel.add_count", True)),
                drop_original=bool(_select(cfg, "data.multilabel.drop_original", True)),
            )

    # 멀티라벨 파생 컬럼 목록(나중에 categorical 후보에서 제외)
    ml_generated_cols: List[str] = []
    if multilabel_enabled and ml_cols:
        prefixes = tuple(f"{c}__" for c in ml_cols)
        ml_generated_cols = [c for c in X_tr.columns if c.startswith(prefixes)]

    # ---- 2) 결측률/상수 컬럼 drop (fit은 X_tr 기준)
    dropped_missing: List[str] = []
    dropped_const: List[str] = []

    # (개선) 고결측 컬럼은 값 자체 대신 '존재 여부'를 남기기 위해 is_present 플래그를 먼저 생성
    present_enable = bool(_select(cfg, "data.missing_present_flags.enable", True))
    present_suffix = str(_select(cfg, "data.missing_present_flags.suffix", "__present"))
    present_skip_all_missing = bool(_select(cfg, "data.missing_present_flags.skip_all_missing", True))
    present_cols: List[str] = []

    if present_enable:
        if other is None:
            X_tr, _, dropped_missing, present_cols = add_missing_presence_flags_and_drop(
                X_tr, X_tr.iloc[0:0].copy(), # type: ignore
                threshold=float(cfg.data.drop_missing_threshold),
                suffix=present_suffix,
                skip_all_missing=present_skip_all_missing,
            )
        else:
            X_tr, other, dropped_missing, present_cols = add_missing_presence_flags_and_drop(
                X_tr, other,
                threshold=float(cfg.data.drop_missing_threshold),
                suffix=present_suffix,
                skip_all_missing=present_skip_all_missing,
            )
    else:
        if other is None:
            X_tr, _, dropped_missing = drop_high_missing_cols(
                X_tr, X_tr.iloc[0:0].copy(), threshold=float(cfg.data.drop_missing_threshold) # type: ignore
            )
        else:
            X_tr, other, dropped_missing = drop_high_missing_cols(
                X_tr, other, threshold=float(cfg.data.drop_missing_threshold)
            )

    if bool(cfg.data.drop_constant_cols):
        if other is None:
            X_tr, _, dropped_const = drop_constant_cols(X_tr, X_tr.iloc[0:0].copy()) # type: ignore
        else:
            X_tr, other, dropped_const = drop_constant_cols(X_tr, other)

    # ---- 3) categorical cols 결정 (fit은 X_tr 기준)
    cat_cols = X_tr.select_dtypes(include=["object"]).columns.tolist()

    if bool(cfg.data.categorical_numeric.enable):
        exclude = list(cfg.data.categorical_numeric.exclude)

        # (옵션) 멀티라벨 파생 컬럼 제외
        if bool(_select(cfg, "data.multilabel.exclude_generated_from_categorical_numeric", True)):
            exclude = exclude + ml_generated_cols

        extra = infer_numeric_categorical_cols(
            train_df=X_tr,
            max_unique=int(cfg.data.categorical_numeric.max_unique),
            include_bool=bool(cfg.data.categorical_numeric.include_bool),
            exclude=exclude,
            skip_binary_numeric=bool(_select(cfg, "data.categorical_numeric.skip_binary_numeric", True)),
        )
        cat_cols = sorted(set(cat_cols).union(set(extra)))

    # (안전장치) 멀티라벨 파생이 cat_cols에 들어갔으면 제거
    if ml_generated_cols and bool(_select(cfg, "data.multilabel.exclude_generated_from_categorical_cols", True)):
        cat_cols = [c for c in cat_cols if c not in set(ml_generated_cols)]

    # ---- 4) Rare bucket (fit은 X_tr value_counts)
    rare_enabled = bool(_select(cfg, "data.rare_bucket.enable", False))
    rare_info: Dict[str, Dict[str, Any]] = {}

    if rare_enabled:
        min_freq = int(_select(cfg, "data.rare_bucket.min_freq", 3))
        exclude_cols = list(_select(cfg, "data.rare_bucket.exclude", [])) or []

        if other is None:
            X_tr, _, rare_info = apply_rare_bucket(
                train_df=X_tr,
                test_df=X_tr.iloc[0:0].copy(), # type: ignore
                categorical_cols=cat_cols,
                min_freq=min_freq,
                exclude_cols=exclude_cols,
            )
        else:
            X_tr, other, rare_info = apply_rare_bucket(
                train_df=X_tr,
                test_df=other,
                categorical_cols=cat_cols,
                min_freq=min_freq,
                exclude_cols=exclude_cols,
            )

    # ---- 5) category map (fit은 X_tr categories)
    categories_map: Dict[str, List[str]] = {}
    if other is None:
        X_tr, _, cat_cols, categories_map = make_categorical_safe(
            X_tr, X_tr.iloc[0:0].copy(), categorical_cols=cat_cols # type: ignore
        )
    else:
        X_tr, other, cat_cols, categories_map = make_categorical_safe(
            X_tr, other, categorical_cols=cat_cols
        )

    # ---- 6) 컬럼 정렬/정합: other는 X_tr 컬럼을 따라가게 reindex
    if other is not None:
        other = other.reindex(columns=X_tr.columns, fill_value=np.nan)

    # ---- 7) 원래 index로 복구하며 분리
    X_va_p = None
    test_p = None
    if other is not None:
        if va_idx is not None:
            X_va_p = other.loc[[f"va__{i}" for i in va_idx]].copy()
            X_va_p.index = va_idx
        if te_idx is not None:
            test_p = other.loc[[f"te__{i}" for i in te_idx]].copy()
            test_p.index = te_idx

    # ---- 8) meta (옵션)
    if return_meta:
        # Summarize category map (avoid huge artifact)
        cat_summary: Dict[str, Dict[str, Any]] = {}
        for col, cats in categories_map.items():
            cats_list = list(cats)
            cat_summary[col] = {
                "n_categories": int(len(cats_list)),
                "sample_categories": cats_list[:30],
            }

        meta = {
            "dropped_missing_cols": list(dropped_missing),
            "missing_present_flags": {
                "enable": bool(present_enable),
                "suffix": str(present_suffix),
                "skip_all_missing": bool(present_skip_all_missing),
                "flag_cols": list(present_cols),
            },
            "dropped_constant_cols": list(dropped_const),
            "categorical_cols": list(cat_cols),
            "rare_bucket": {
                "enable": bool(rare_enabled),
                "min_freq": int(_select(cfg, "data.rare_bucket.min_freq", 3)) if bool(rare_enabled) else None,
                "n_affected_cols": int(sum(1 for _c, _v in rare_info.items() if int(_v.get("n_rare", 0)) > 0)),
                "per_col_n_rare": {str(_c): int(_v.get("n_rare", 0)) for _c, _v in rare_info.items()},
            },
            "multilabel": {
                "enable": bool(multilabel_enabled and bool(ml_cols)),
                "cols": list(ml_cols) if bool(multilabel_enabled and bool(ml_cols)) else [],
                "per_col": multilabel_info,
            },
            "n_features": int(X_tr.shape[1]),
            "feature_columns": list(X_tr.columns),
            "categories_map_summary": cat_summary,
        }

    return X_tr, X_va_p, test_p, cat_cols, meta # type: ignore



def maybe_init_wandb(cfg: DictConfig):
    if not cfg.wandb.enable:
        return None

    try:
        import wandb # type: ignore

        run = wandb.init( # type: ignore
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
    import wandb # type: ignore

    wandb.log(payload, step=step) # type: ignore



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

    catboost_version = None
    try:
        if CatBoostClassifier is not None:
            import catboost  # type: ignore
            catboost_version = getattr(catboost, "__version__", None)
    except Exception:
        catboost_version = None

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
        "catboost_version": catboost_version,
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
    import wandb # type: ignore

    art = wandb.Artifact(name=name, type=art_type) # type: ignore
    art.add_file(str(path))
    run.log_artifact(art)


def search_best_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    method: str = "pr_curve",
    min_t: float = 0.05,
    max_t: float = 0.95,
    grid_start: float = 0.05,
    grid_end: float = 0.95,
    grid_step: float = 0.01,
) -> Tuple[float, float]:
    """OOF 확률(proba)에 대해 F1이 최대가 되는 threshold를 탐색 + clamp.

    method:
      - "pr_curve": precision_recall_curve가 제공하는 '후보 임계값(=고유 score 경계)'에서 정확 탐색 (기본)
      - "grid": [grid_start, grid_end] 구간을 grid_step 간격으로 전수 탐색

    공통:
      - 최종 threshold는 [min_t, max_t] 범위로 제한됩니다.
    """
    # 방어적 캐스팅
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)

    if min_t > max_t:
        min_t, max_t = max_t, min_t

    # safety: nan proba는 아주 작은 값으로 처리 (있으면 비정상 케이스)
    if np.isnan(proba).any():
        proba = np.nan_to_num(proba, nan=-1.0)

    method = (method or "pr_curve").strip().lower()

    # ---------- grid search ----------
    if method == "grid":
        start = float(max(min_t, grid_start))
        end = float(min(max_t, grid_end))

        # 1) coarse grid (더 큰 간격으로 전체 탐색)
        coarse_step = max(float(grid_step), 0.02)  # 예: 최소 0.02
        coarse_thr = np.arange(start, end + 1e-12, coarse_step, dtype=float)

        if coarse_thr.size == 0:
            t = float(np.clip(0.5, min_t, max_t))
            return t, float(f1_score(y_true, (proba >= t).astype(int)))

        y_pos = (y_true == 1)
        preds = (proba[:, None] >= coarse_thr[None, :])
        tp = np.sum(preds & y_pos[:, None], axis=0).astype(float)
        fp = np.sum(preds & (~y_pos)[:, None], axis=0).astype(float)
        fn = np.sum((~preds) & y_pos[:, None], axis=0).astype(float)
        prec = tp / (tp + fp + 1e-15)
        rec = tp / (tp + fn + 1e-15)
        f1s = 2.0 * prec * rec / (prec + rec + 1e-15)

        best_idx = int(np.nanargmax(f1s))
        best_t_coarse = float(coarse_thr[best_idx])

        # 2) fine grid (coarse 최적점 주변만 촘촘히 재탐색)
        fine_step = min(float(grid_step), 0.002)   # 예: 최대 0.002까지 촘촘히
        win = 3 * coarse_step                      # 주변 탐색 폭
        fine_start = float(max(start, best_t_coarse - win))
        fine_end   = float(min(end,   best_t_coarse + win))
        fine_thr = np.arange(fine_start, fine_end + 1e-12, fine_step, dtype=float)

        preds2 = (proba[:, None] >= fine_thr[None, :])
        tp2 = np.sum(preds2 & y_pos[:, None], axis=0).astype(float)
        fp2 = np.sum(preds2 & (~y_pos)[:, None], axis=0).astype(float)
        fn2 = np.sum((~preds2) & y_pos[:, None], axis=0).astype(float)
        prec2 = tp2 / (tp2 + fp2 + 1e-15)
        rec2 = tp2 / (tp2 + fn2 + 1e-15)
        f1s2 = 2.0 * prec2 * rec2 / (prec2 + rec2 + 1e-15)

        best_idx2 = int(np.nanargmax(f1s2))
        best_t = float(fine_thr[best_idx2])
        best_f1 = float(f1s2[best_idx2])
        return best_t, best_f1

    # ---------- pr_curve (default) ----------
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
    X_raw: pd.DataFrame,
    y: pd.Series,
    run=None,
) -> Dict[str, Any]:
    """Tune LightGBM hyperparameters with Optuna (fold-fit preprocessing, leakage-free).

    IMPORTANT:
      - 각 fold의 전처리 기준(rare bucket/vocab/category map 등)은 train-fold에서만 생성합니다.
      - 검증 fold는 해당 기준으로 transform만 수행합니다.

    Metric:
      - "auc" (default) : mean AUC over folds
      - "f1"            : OOF F1 at best threshold (OOF 기반 threshold 탐색)
    """
    enable = bool(_select(cfg, "train.optuna.enable", False))
    if not enable:
        return {}

    try:
        import optuna # type: ignore
    except ImportError as e:
        raise ImportError("Optuna is not installed. Run: pip install optuna") from e

    optuna_cfg = _select(cfg, "train.optuna", default={})
    n_trials = int(getattr(optuna_cfg, "n_trials", 50))
    timeout = getattr(optuna_cfg, "timeout", None)
    timeout = int(timeout) if timeout is not None else None

    metric = str(getattr(optuna_cfg, "metric", "auc")).lower()
    if metric not in ("auc", "f1"):
        raise ValueError(f"Unsupported optuna metric: {metric} (use 'auc' or 'f1')")

    prune_metric = str(getattr(optuna_cfg, "prune_metric", "auc")).lower()
    if prune_metric == "auto":
        prune_metric = "auc" if metric == "f1" else metric
    if prune_metric not in ("auc", "f1"):
        raise ValueError(f"Unsupported optuna prune_metric: {prune_metric} (use 'auc'|'f1'|'auto')")

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
    if n_warmup_steps > n_splits:
        n_warmup_steps = n_splits

    if not use_pruner or pruner_name == "nop":
        pruner = optuna.pruners.NopPruner()
    elif pruner_name == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)

    study_name = str(getattr(optuna_cfg, "study_name", "lgbm_tuning"))
    storage = getattr(optuna_cfg, "storage", None)  # e.g. "sqlite:///optuna.db"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True if storage else False,
    )

    base_params = dict(OmegaConf.to_container(cfg.model, resolve=True))
    base_params["random_state"] = seed

    # ---------------------------
    # (옵션) Fold별 전처리 캐시: 전처리는 trial과 무관하므로 1회만 수행해 속도를 올릴 수 있음
    # ---------------------------
    cache_preprocess = bool(_select(cfg, "train.optuna.preprocess_cache", True))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_splits = list(skf.split(X_raw, y))

    fold_cache: List[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]] = []
    if cache_preprocess:
        for tr_idx, va_idx in fold_splits:
            X_tr_raw, y_tr = X_raw.iloc[tr_idx], y.iloc[tr_idx]
            X_va_raw, y_va = X_raw.iloc[va_idx], y.iloc[va_idx]
            X_tr, X_va, _, cat_cols, _ = preprocess_fold_fit(cfg, X_tr_raw, X_va_raw, None, return_meta=False)
            fold_cache.append((X_tr, y_tr, X_va, y_va, cat_cols)) # type: ignore

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

        def _ss_float_alias(primary: str, alias: str, low: float, high: float, log: bool = False) -> float:
            if ss is not None and (not hasattr(ss, primary)) and hasattr(ss, alias):
                cfg_item = getattr(ss, alias)
                _low = float(getattr(cfg_item, "low", low))
                _high = float(getattr(cfg_item, "high", high))
                _log = bool(getattr(cfg_item, "log", log))
                return float(trial.suggest_float(primary, _low, _high, log=_log))
            return _ss_float(primary, low, high, log=log)

        def _ss_cat(name: str, choices: List[Any]) -> Any:
            if ss is None or not hasattr(ss, name):
                return trial.suggest_categorical(name, choices)
            cfg_item = getattr(ss, name)
            _choices = list(getattr(cfg_item, "choices", choices))
            if not _choices:
                _choices = choices
            return trial.suggest_categorical(name, _choices)

        learning_rate = _ss_float("learning_rate", 0.01, 0.15, log=True)
        max_depth = _ss_cat("max_depth", [-1, 4, 6, 8, 10, 12, 16])
        num_leaves = _ss_int("num_leaves", 16, 256, log=True)
        min_child_samples = _ss_int("min_child_samples", 5, 400, log=True)
        min_split_gain = _ss_float("min_split_gain", 1e-8, 1.0, log=True)
        reg_alpha = _ss_float("reg_alpha", 1e-8, 10.0, log=True)
        reg_lambda = _ss_float("reg_lambda", 1e-3, 500.0, log=True)
        subsample = _ss_float("subsample", 0.6, 1.0, log=False)
        subsample_freq = _ss_cat("subsample_freq", [0, 1, 2, 5])
        colsample_bytree = _ss_float_alias("colsample_bytree", "feature_fraction", 0.4, 1.0, log=False)
        max_bin = _ss_cat("max_bin", [127, 255, 511])
        scale_pos_weight = _ss_float("scale_pos_weight", 1.0, 4.0, log=True)

        if int(max_depth) > 0:
            max_leaves = int(2 ** int(max_depth))
            num_leaves = int(min(int(num_leaves), max_leaves))

        if float(subsample) >= 0.999:
            subsample_freq = 0
        else:
            subsample_freq = int(max(1, int(subsample_freq)))

        params.update({
            "learning_rate": float(learning_rate),
            "max_depth": int(max_depth),
            "num_leaves": int(num_leaves),
            "min_child_samples": int(min_child_samples),
            "min_split_gain": float(min_split_gain),
            "reg_alpha": float(reg_alpha),
            "reg_lambda": float(reg_lambda),
            "subsample": float(subsample),
            "subsample_freq": int(subsample_freq),
            "colsample_bytree": float(colsample_bytree),
            "max_bin": int(max_bin),
            "scale_pos_weight": float(scale_pos_weight),
        })

        aucs: List[float] = []
        oof = np.zeros(len(X_raw), dtype=float)
        seen_idx: List[int] = []

        for fold, (tr_idx, va_idx) in enumerate(fold_splits, start=1):
            if cache_preprocess:
                X_tr, y_tr, X_va, y_va, cat_cols = fold_cache[fold - 1]
            else:
                X_tr_raw, y_tr = X_raw.iloc[tr_idx], y.iloc[tr_idx]
                X_va_raw, y_va = X_raw.iloc[va_idx], y.iloc[va_idx]
                X_tr, X_va, _, cat_cols, _ = preprocess_fold_fit(cfg, X_tr_raw, X_va_raw, None, return_meta=False)

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="average_precision",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=int(cfg.train.early_stopping_rounds), verbose=False),
                ],
                categorical_feature=cat_cols if len(cat_cols) > 0 else "auto",
            )

            proba_va = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
            oof[va_idx] = proba_va
            seen_idx.extend(list(va_idx))

            auc = roc_auc_score(y_va, proba_va)
            aucs.append(float(auc))

            # ---- Pruning: report intermediate value per fold ----
            if prune_metric == "auc":
                intermediate = float(np.mean(aucs))
            else:
                try:
                    y_part = y.values[seen_idx]
                    p_part = oof[seen_idx]
                    if bool(cfg.train.optimize_threshold.enable):
                        _, f1_part = search_best_threshold(
                            y_true=y_part,
                            proba=p_part,
                            method=str(_select(cfg, "train.optimize_threshold.method", "pr_curve")),
                            min_t=float(cfg.train.optimize_threshold.clamp_min),
                            max_t=float(cfg.train.optimize_threshold.clamp_max),
                            grid_start=float(_select(cfg, "train.optimize_threshold.grid.start", _select(cfg, "train.optimize_threshold.grid_start", 0.05))),
                            grid_end=float(_select(cfg, "train.optimize_threshold.grid.end", _select(cfg, "train.optimize_threshold.grid_end", 0.95))),
                            grid_step=float(_select(cfg, "train.optimize_threshold.grid.step", _select(cfg, "train.optimize_threshold.grid_step", 0.01))),
                        )
                        intermediate = float(f1_part)
                    else:
                        thr = float(cfg.train.threshold)
                        intermediate = float(f1_score(y_part, (p_part >= thr).astype(int)))
                except Exception:
                    intermediate = float("nan")

            trial.report(intermediate, step=fold)
            if use_pruner and trial.should_prune():
                raise optuna.TrialPruned()

        auc_mean = float(np.mean(aucs))
        if metric == "auc":
            return auc_mean

        if bool(cfg.train.optimize_threshold.enable):
            t, f1_best = search_best_threshold(
                y_true=y.values,
                proba=oof,
                method=str(_select(cfg, "train.optimize_threshold.method", "pr_curve")),
                min_t=float(cfg.train.optimize_threshold.clamp_min),
                max_t=float(cfg.train.optimize_threshold.clamp_max),
                grid_start=float(_select(cfg, "train.optimize_threshold.grid.start", _select(cfg, "train.optimize_threshold.grid_start", 0.05))),
                grid_end=float(_select(cfg, "train.optimize_threshold.grid.end", _select(cfg, "train.optimize_threshold.grid_end", 0.95))),
                grid_step=float(_select(cfg, "train.optimize_threshold.grid.step", _select(cfg, "train.optimize_threshold.grid_step", 0.01))),
            )
            trial.set_user_attr("best_threshold", float(t))
            return float(f1_best)

        thr = float(cfg.train.threshold)
        trial.set_user_attr("best_threshold", float(thr))
        return float(f1_score(y.values, (oof >= thr).astype(int)))

    print(f"\n[Optuna] Start tuning: n_trials={n_trials} | metric={metric} | folds={n_splits} | foldfit_preprocess=True")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best = study.best_trial
    best_params = dict(best.params)

    overrides = {
        "learning_rate": float(best_params["learning_rate"]),
        "max_depth": int(best_params["max_depth"]),
        "num_leaves": int(best_params["num_leaves"]),
        "min_child_samples": int(best_params["min_child_samples"]),
        "min_split_gain": float(best_params["min_split_gain"]),
        "reg_alpha": float(best_params["reg_alpha"]),
        "reg_lambda": float(best_params["reg_lambda"]),
        "subsample": float(best_params["subsample"]),
        "subsample_freq": int(best_params["subsample_freq"]),
        "colsample_bytree": float(best_params["colsample_bytree"]),
        "max_bin": int(best_params["max_bin"]),
        "scale_pos_weight": float(best_params["scale_pos_weight"]),
    }

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
            "raw_params": dict(best.params),
            "model_overrides": dict(overrides),
            "foldfit_preprocess": True,
            "preprocess_cache": bool(cache_preprocess),
        }
    }
    saved_path = save_yaml_in_output_dir("best_params.yaml", best_payload)
    print(f"[Optuna] Saved best params -> {saved_path}")

    print("\n[Optuna] Best trial")
    print("  value:", best.value)
    print("  params:", best.params)

    if run is not None:
        wandb_log(run, {
            "optuna/best_value": float(best.value),
            "optuna/best_learning_rate": overrides["learning_rate"],
            "optuna/best_max_depth": overrides["max_depth"],
            "optuna/best_num_leaves": overrides["num_leaves"],
            "optuna/best_min_child_samples": overrides["min_child_samples"],
            "optuna/best_min_split_gain": overrides["min_split_gain"],
            "optuna/best_reg_alpha": overrides["reg_alpha"],
            "optuna/best_reg_lambda": overrides["reg_lambda"],
            "optuna/best_subsample": overrides["subsample"],
            "optuna/best_subsample_freq": overrides["subsample_freq"],
            "optuna/best_colsample_bytree": overrides["colsample_bytree"],
            "optuna/best_feature_fraction": overrides["colsample_bytree"],
            "optuna/best_max_bin": overrides["max_bin"],
            "optuna/foldfit_preprocess": 1,
        })

    return overrides





def cv_train_foldfit(
    cfg: DictConfig,
    X_raw: pd.DataFrame,
    y: pd.Series,
    test_raw: pd.DataFrame,
    run,
    feature_columns_ref: List[str],
    save_fold_models: bool = False,
    fold_model_basepath: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[float], List[float], List[Path], np.ndarray]:
    """CV 학습 (fold-fit 전처리로 누수 제거).

    feature_importance는 fold마다 feature space가 달라질 수 있으므로,
    feature_columns_ref(보통 full-fit 전처리 기준 컬럼)에 맞춰 정렬한 배열을 반환합니다.
    """

    params = dict(OmegaConf.to_container(cfg.model, resolve=True))

    # --- CatBoost (submodel) + Ensemble weight search config ---
    cb_enable = bool(_select(cfg, "train.catboost.enable", False))
    ens_enable = bool(_select(cfg, "train.ensemble.enable", True)) if cb_enable else False
    weight_search_enable = bool(_select(cfg, "train.ensemble.weight_search.enable", True)) if ens_enable else False

    # If user enabled CatBoost but package isn't installed -> fail fast
    if cb_enable and CatBoostClassifier is None:
        raise ImportError(
            "CatBoost is enabled (train.catboost.enable=true) but catboost is not installed. "
            "Run: pip install catboost"
        )

    skf = StratifiedKFold(
        n_splits=int(cfg.train.n_splits),
        shuffle=True,
        random_state=int(cfg.train.seed),
    )

    # Base(LGB) OOF/test proba
    oof_lgb = np.zeros(len(X_raw), dtype=float)
    test_sum_lgb = np.zeros(len(test_raw), dtype=float)

    # Sub(Cat) OOF/test proba (optional)
    oof_cb = np.zeros(len(X_raw), dtype=float) if cb_enable else None
    test_sum_cb = np.zeros(len(test_raw), dtype=float) if cb_enable else None

    # feature importance: name 기반 누적
    feat_imp_sum: Dict[str, float] = {}

    best_iters: List[int] = []
    # We'll compute fold metrics for returned model:
    # - if ensemble enabled -> blended metrics (computed after best weight is found)
    # - else -> LGB metrics
    aucs_lgb: List[float] = []
    f1s_lgb: List[float] = []
    aucs_cb: List[float] = []
    f1s_cb: List[float] = []
    fold_va_indices: List[np.ndarray] = []  # for recomputing blended fold metrics after weight search

    aucs: List[float] = []
    f1s: List[float] = []
    fold_model_paths: List[Path] = []

    print("\n[CV] Start (fold-fit preprocessing: leakage-free)")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_raw, y), start=1):
        X_tr_raw, y_tr = X_raw.iloc[tr_idx], y.iloc[tr_idx]
        X_va_raw, y_va = X_raw.iloc[va_idx], y.iloc[va_idx]
        fold_va_indices.append(np.asarray(va_idx))

        # fold-fit 전처리 (rare/vocab/category map은 train-fold에서만 생성)
        X_tr, X_va, test_fold, cat_cols, fold_meta = preprocess_fold_fit(
            cfg,
            X_tr_raw,
            X_va_raw,
            test_raw,
            return_meta=bool(save_fold_models),
        )

        # (옵션) fold별 전처리 meta 저장 (fold ensemble 재현성에 도움)
        if save_fold_models:
            try:
                fold_meta_payload = {"fold": int(fold), "preprocess": fold_meta}
                save_yaml_in_output_dir(f"fold{fold}_preprocess_meta.yaml", fold_meta_payload)
            except Exception as e:
                print(f"[Warn] saving fold preprocess meta failed (fold={fold}): {e}")

        # -------------------
        # 1) LightGBM
        # -------------------
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="binary_logloss",  # or ["binary_logloss", "auc"]
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=int(cfg.train.early_stopping_rounds),
                    verbose=False,
                    first_metric_only=True,), # early stopping은 첫 metric(logloss)로만
                lgb.log_evaluation(period=int(cfg.train.log_period)),
            ],
            categorical_feature=cat_cols if len(cat_cols) > 0 else "auto",
        )

        proba_va_lgb = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
        oof_lgb[va_idx] = proba_va_lgb

        # Fold test proba 누적 (앙상블용)
        proba_te_lgb = model.predict_proba(test_fold, num_iteration=model.best_iteration_)[:, 1]
        test_sum_lgb += proba_te_lgb

        # Feature importance 누적 (name 기반)
        try:
            names = list(X_tr.columns)
            imps = model.feature_importances_
            for n, v in zip(names, imps):
                feat_imp_sum[n] = float(feat_imp_sum.get(n, 0.0) + float(v))
        except Exception:
            pass

        # Metrics
        auc_lgb = float(roc_auc_score(y_va, proba_va_lgb))

        fixed_thr = float(cfg.train.threshold)
        f1_fixed_lgb = float(f1_score(y_va, (proba_va_lgb >= fixed_thr).astype(int)))

        # CV 로그를 "best threshold 기준 F1"로 통일
        if bool(cfg.train.optimize_threshold.enable):
            best_thr, f1_best = search_best_threshold(
                y_true=y_va.values,
                proba=proba_va_lgb,
                method=str(_select(cfg, "train.optimize_threshold.method", "pr_curve")),
                min_t=float(cfg.train.optimize_threshold.clamp_min),
                max_t=float(cfg.train.optimize_threshold.clamp_max),
                grid_start=float(_select(cfg, "train.optimize_threshold.grid.start",
                                        _select(cfg, "train.optimize_threshold.grid_start", 0.05))),
                grid_end=float(_select(cfg, "train.optimize_threshold.grid.end",
                                      _select(cfg, "train.optimize_threshold.grid_end", 0.95))),
                grid_step=float(_select(cfg, "train.optimize_threshold.grid.step",
                                       _select(cfg, "train.optimize_threshold.grid_step", 0.01))),
            )
            best_thr = float(best_thr)
            f1_lgb = float(f1_best)
        else:
            # optimize_threshold 비활성화면 고정 threshold를 best로 취급
            best_thr = float(fixed_thr)
            f1_lgb = float(f1_fixed_lgb)

        aucs_lgb.append(auc_lgb)
        f1s_lgb.append(f1_lgb)
        best_iters.append(int(model.best_iteration_))

        print(
            f"[Fold {fold}][LGB] best_iter={model.best_iteration_} | AUC={auc_lgb:.5f} | "
            f"F1_best@{best_thr:.3f}={f1_lgb:.5f} | F1_fixed@{fixed_thr:.3f}={f1_fixed_lgb:.5f}"
        )
        wandb_log(
            run,
            {
                "fold": fold,
                "cv/lgb/auc": float(auc_lgb),
                "cv/lgb/f1": float(f1_lgb),
                "cv/lgb/best_threshold": float(best_thr),
                "cv/lgb/f1_fixed": float(f1_fixed_lgb),
                "cv/lgb/threshold_fixed": float(fixed_thr),
                "cv/best_iter": int(model.best_iteration_),
            },
            step=fold,
        )

        # (옵션) fold 모델 저장
        if save_fold_models and fold_model_basepath is not None:
            base = fold_model_basepath
            suffix = base.suffix if base.suffix else ".txt"
            fold_path = base.with_name(f"{base.stem}_fold{fold}{suffix}")
            fold_path.parent.mkdir(parents=True, exist_ok=True)
            model.booster_.save_model(str(fold_path))
            fold_model_paths.append(fold_path)

        # -------------------
        # 2) CatBoost (optional)
        # -------------------
        if cb_enable:
            cb_params = dict(_select(cfg, "train.catboost.params", {}) or {})

            # Hygiene defaults (safe under Hydra/wandb)
            cb_params.setdefault("loss_function", "Logloss")
            cb_params.setdefault("eval_metric", "AUC")
            cb_params.setdefault("random_seed", int(cfg.train.seed))
            cb_params.setdefault("allow_writing_files", False)
            cb_params.setdefault("verbose", int(_select(cfg, "train.catboost.verbose", int(cfg.train.log_period))))

            # Early stopping (od_wait)
            cb_es = int(_select(cfg, "train.catboost.early_stopping_rounds", int(cfg.train.early_stopping_rounds)))
            cb_params.setdefault("od_type", "Iter")
            cb_params.setdefault("od_wait", cb_es)
            cb_params.setdefault("use_best_model", True)

            # cat feature indices
            cat_idx = []
            try:
                if cat_cols:
                    cat_idx = [int(X_tr.columns.get_loc(c)) for c in cat_cols if c in X_tr.columns] # type: ignore
            except Exception:
                cat_idx = []

            cb_model = CatBoostClassifier(**cb_params)  # type: ignore
            cb_model.fit(
                X_tr, y_tr,
                eval_set=(X_va, y_va),
                cat_features=cat_idx if len(cat_idx) > 0 else None,
            )

            proba_va_cb = cb_model.predict_proba(X_va)[:, 1]
            oof_cb[va_idx] = proba_va_cb  # type: ignore

            proba_te_cb = cb_model.predict_proba(test_fold)[:, 1]
            test_sum_cb += proba_te_cb  # type: ignore

            auc_cb = float(roc_auc_score(y_va, proba_va_cb))
            f1_fixed_cb = float(f1_score(y_va, (proba_va_cb >= fixed_thr).astype(int)))

            if bool(cfg.train.optimize_threshold.enable):
                cb_best_thr, cb_f1_best = search_best_threshold(
                    y_true=y_va.values,
                    proba=proba_va_cb,
                    method=str(_select(cfg, "train.optimize_threshold.method", "pr_curve")),
                    min_t=float(cfg.train.optimize_threshold.clamp_min),
                    max_t=float(cfg.train.optimize_threshold.clamp_max),
                    grid_start=float(_select(cfg, "train.optimize_threshold.grid.start",
                                            _select(cfg, "train.optimize_threshold.grid_start", 0.05))),
                    grid_end=float(_select(cfg, "train.optimize_threshold.grid.end",
                                        _select(cfg, "train.optimize_threshold.grid_end", 0.95))),
                    grid_step=float(_select(cfg, "train.optimize_threshold.grid.step",
                                        _select(cfg, "train.optimize_threshold.grid_step", 0.01))),
                )
                cb_best_thr = float(cb_best_thr)
                f1_cb = float(cb_f1_best)
            else:
                cb_best_thr = float(fixed_thr)
                f1_cb = float(f1_fixed_cb)

            aucs_cb.append(auc_cb)
            f1s_cb.append(f1_cb)

            print(
                f"[Fold {fold}][CB ] AUC={auc_cb:.5f} | "
                f"F1_best@{cb_best_thr:.3f}={f1_cb:.5f} | F1_fixed@{fixed_thr:.3f}={f1_fixed_cb:.5f}"
            )
            wandb_log(
                run,
                {
                    "cv/cb/auc": float(auc_cb),
                    "cv/cb/f1": float(f1_cb),
                    "cv/cb/best_threshold": float(cb_best_thr),
                    "cv/cb/f1_fixed": float(f1_fixed_cb),
                    "cv/cb/threshold_fixed": float(fixed_thr),
                },
                step=fold,
            )

            # (옵션) fold CatBoost 모델 저장
            if save_fold_models and fold_model_basepath is not None:
                base = fold_model_basepath
                cb_ext = str(_select(cfg, "train.catboost.model_ext", ".cbm"))
                cb_fold_path = base.with_name(f"{base.stem}_cb_fold{fold}{cb_ext}")
                cb_fold_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    cb_model.save_model(str(cb_fold_path))
                    fold_model_paths.append(cb_fold_path)
                except Exception as e:
                    print(f"[Warn] saving CatBoost model failed (fold={fold}): {e}")

    n_splits = int(cfg.train.n_splits)
    test_lgb_mean = test_sum_lgb / n_splits

    # Decide what to return: LGB-only vs Blended(LGB+CB)
    if ens_enable and cb_enable and (oof_cb is not None) and (test_sum_cb is not None):
        # ----- weight search on full OOF (maximize Binary F1) -----
        w_min = float(_select(cfg, "train.ensemble.weight_search.min_w_lgb", 0.0))
        w_max = float(_select(cfg, "train.ensemble.weight_search.max_w_lgb", 1.0))
        w_step = float(_select(cfg, "train.ensemble.weight_search.step", 0.01))
        w_default = float(_select(cfg, "train.ensemble.weight_search.default_w_lgb", 0.5))

        def _eval_f1_with_threshold(proba_mix: np.ndarray) -> Tuple[float, float]:
            fixed_thr = float(cfg.train.threshold)
            if bool(cfg.train.optimize_threshold.enable):
                t_best, f1_best = search_best_threshold(
                    y_true=y.values,
                    proba=proba_mix,
                    method=str(_select(cfg, "train.optimize_threshold.method", "pr_curve")),
                    min_t=float(cfg.train.optimize_threshold.clamp_min),
                    max_t=float(cfg.train.optimize_threshold.clamp_max),
                    grid_start=float(_select(cfg, "train.optimize_threshold.grid.start",
                                            _select(cfg, "train.optimize_threshold.grid_start", 0.05))),
                    grid_end=float(_select(cfg, "train.optimize_threshold.grid.end",
                                        _select(cfg, "train.optimize_threshold.grid_end", 0.95))),
                    grid_step=float(_select(cfg, "train.optimize_threshold.grid.step",
                                        _select(cfg, "train.optimize_threshold.grid_step", 0.01))),
                )
                return float(t_best), float(f1_best)
            else:
                f1_fixed = float(f1_score(y.values, (proba_mix >= fixed_thr).astype(int)))
                return float(fixed_thr), float(f1_fixed)

        best_w = w_default
        best_thr = float(cfg.train.threshold)
        best_f1 = -1.0

        if weight_search_enable:
            if w_step <= 0:
                w_step = 0.01
            if w_min > w_max:
                w_min, w_max = w_max, w_min
            grid = np.arange(w_min, w_max + 1e-12, w_step, dtype=float)
            for w_lgb in grid:
                proba_mix = (w_lgb * oof_lgb) + ((1.0 - w_lgb) * oof_cb)  # type: ignore
                t, f1v = _eval_f1_with_threshold(proba_mix)
                if f1v > best_f1:
                    best_f1 = f1v
                    best_w = float(w_lgb)
                    best_thr = float(t)
        else:
            proba_mix = (best_w * oof_lgb) + ((1.0 - best_w) * oof_cb)  # type: ignore
            best_thr, best_f1 = _eval_f1_with_threshold(proba_mix)

        # final blended outputs
        oof_proba = (best_w * oof_lgb) + ((1.0 - best_w) * oof_cb)  # type: ignore
        test_cb_mean = (test_sum_cb / n_splits)  # type: ignore
        test_proba_mean = (best_w * test_lgb_mean) + ((1.0 - best_w) * test_cb_mean)

        # recompute fold metrics with the selected weight (so main() prints the blended CV summary)
        aucs = []
        f1s = []
        for fold, va_idx in enumerate(fold_va_indices, start=1):
            y_va = y.iloc[va_idx]
            proba_va_mix = oof_proba[va_idx]
            auc_mix = float(roc_auc_score(y_va, proba_va_mix))

            fixed_thr = float(cfg.train.threshold)
            f1_fixed = float(f1_score(y_va, (proba_va_mix >= fixed_thr).astype(int)))
            if bool(cfg.train.optimize_threshold.enable):
                t_fold, f1_fold = search_best_threshold(
                    y_true=y_va.values,
                    proba=proba_va_mix,
                    method=str(_select(cfg, "train.optimize_threshold.method", "pr_curve")),
                    min_t=float(cfg.train.optimize_threshold.clamp_min),
                    max_t=float(cfg.train.optimize_threshold.clamp_max),
                    grid_start=float(_select(cfg, "train.optimize_threshold.grid.start",
                                            _select(cfg, "train.optimize_threshold.grid_start", 0.05))),
                    grid_end=float(_select(cfg, "train.optimize_threshold.grid.end",
                                        _select(cfg, "train.optimize_threshold.grid_end", 0.95))),
                    grid_step=float(_select(cfg, "train.optimize_threshold.grid.step",
                                        _select(cfg, "train.optimize_threshold.grid_step", 0.01))),
                )
                f1_mix = float(f1_fold)
                t_mix = float(t_fold)
            else:
                f1_mix = float(f1_fixed)
                t_mix = float(fixed_thr)
            aucs.append(auc_mix)
            f1s.append(f1_mix)

            wandb_log(run, {
                "cv/blend/auc": float(auc_mix),
                "cv/blend/f1": float(f1_mix),
                "cv/blend/best_threshold": float(t_mix),
                "cv/blend/w_lgb": float(best_w),
            }, step=fold)

        # Log ensemble summary
        wandb_log(run, {
            "ensemble/enable": 1,
            "ensemble/weight_search_enable": int(weight_search_enable),
            "ensemble/best_w_lgb": float(best_w),
            "ensemble/best_w_cb": float(1.0 - best_w),
            "ensemble/oof_best_threshold": float(best_thr),
            "ensemble/oof_best_f1": float(best_f1),
            "ensemble/oof_auc": float(roc_auc_score(y.values, oof_proba)),
            "cv/lgb/f1_mean": float(np.mean(f1s_lgb)) if f1s_lgb else None,
            "cv/cb/f1_mean": float(np.mean(f1s_cb)) if f1s_cb else None,
        })

    else:
        # LGB-only
        oof_proba = oof_lgb
        test_proba_mean = test_lgb_mean
        aucs = aucs_lgb
        f1s = f1s_lgb

    # feature importance를 reference column에 맞춰 정렬
    feat_imp_mean = np.array([feat_imp_sum.get(c, 0.0) / n_splits for c in feature_columns_ref], dtype=float)

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

    # (누수 방지) CV/Optuna는 raw feature에서 fold-fit 전처리를 수행
    X_raw = X.copy()
    test_raw = test.copy()

    # (메타/단일모델용) full-data 기준 전처리(기존 로직 유지)
    X_full = X.copy()
    test_full = test.copy()

    # Multi-label columns -> multi-hot (optional, train-only vocab to avoid leakage)
    multilabel_enabled = bool(_select(cfg, "data.multilabel.enable", False))
    multilabel_info: Dict[str, Any] = {}
    if multilabel_enabled:
        ml_cols = list(_select(cfg, "data.multilabel.cols", [])) or []
        if ml_cols:
            top_k_val = _select(cfg, "data.multilabel.top_k", 50)
            top_k_val = None if top_k_val is None else int(top_k_val)
            X_full, test_full, multilabel_info = expand_multilabel_columns(
                train_df=X_full,
                test_df=test_full,
                cols=ml_cols,
                sep=str(_select(cfg, "data.multilabel.sep", ",")),
                min_freq=int(_select(cfg, "data.multilabel.min_freq", 3)),
                top_k=top_k_val,
                add_other=bool(_select(cfg, "data.multilabel.add_other", True)),
                add_count=bool(_select(cfg, "data.multilabel.add_count", True)),
                drop_original=bool(_select(cfg, "data.multilabel.drop_original", True)),
            )
        else:
            multilabel_enabled = False

    # (중요) 멀티라벨로 생성된 파생 멀티핫/카운트/other 컬럼 목록(나중에 categorical 후보에서 제외)
    ml_generated_cols: List[str] = []
    if multilabel_enabled and ml_cols:
        prefixes = tuple(f"{c}__" for c in ml_cols)
        ml_generated_cols = [c for c in X_full.columns if c.startswith(prefixes)]
    # 결측률 높은 컬럼 드랍(train 기준)
    # (개선) 드랍 전에 고결측 컬럼의 '존재 여부(is_present)' 플래그를 생성해 신호를 보존
    present_enable = bool(_select(cfg, "data.missing_present_flags.enable", True))
    present_suffix = str(_select(cfg, "data.missing_present_flags.suffix", "__present"))
    present_skip_all_missing = bool(_select(cfg, "data.missing_present_flags.skip_all_missing", True))
    present_cols_full: List[str] = []

    if present_enable:
        X_full, test_full, dropped_missing, present_cols_full = add_missing_presence_flags_and_drop(
            X_full, test_full,
            threshold=float(cfg.data.drop_missing_threshold),
            suffix=present_suffix,
            skip_all_missing=present_skip_all_missing,
        )
    else:
        X_full, test_full, dropped_missing = drop_high_missing_cols(
            X_full, test_full, threshold=float(cfg.data.drop_missing_threshold)
        )

    # 상수 컬럼 드랍(train 기준)
    dropped_const: List[str] = []
    if bool(cfg.data.drop_constant_cols):
        X_full, test_full, dropped_const = drop_constant_cols(X_full, test_full)

    # 어떤 컬럼을 category로 볼지 결정
    cat_cols_full = X_full.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = list(cat_cols_full)

    if bool(cfg.data.categorical_numeric.enable):
        # 기본 exclude + (옵션) 멀티라벨 파생 컬럼 제외
        exclude = list(cfg.data.categorical_numeric.exclude)
        if bool(_select(cfg, "data.multilabel.exclude_generated_from_categorical_numeric", True)):
            exclude = exclude + ml_generated_cols

        extra = infer_numeric_categorical_cols(
            train_df=X_full,
            max_unique=int(cfg.data.categorical_numeric.max_unique),
            include_bool=bool(cfg.data.categorical_numeric.include_bool),
            exclude=exclude,
            skip_binary_numeric=bool(_select(cfg, "data.categorical_numeric.skip_binary_numeric", True)),
        )
        # object + numeric-cat 합치기
        cat_cols = sorted(set(cat_cols).union(set(extra)))

    # (안전장치) 혹시라도 cat_cols에 들어갔으면 제거
    if ml_generated_cols and bool(_select(cfg, "data.multilabel.exclude_generated_from_categorical_cols", True)):
        cat_cols = [c for c in cat_cols if c not in set(ml_generated_cols)]


    # Rare bucket (optional): merge infrequent categories into __RARE__ using train frequencies
    rare_enabled = bool(_select(cfg, "data.rare_bucket.enable", False))
    rare_info: Dict[str, Dict[str, Any]] = {}
    if rare_enabled:
        min_freq = int(_select(cfg, "data.rare_bucket.min_freq", 3))
        exclude_cols = list(_select(cfg, "data.rare_bucket.exclude", [])) or []
        X_full, test_full, rare_info = apply_rare_bucket(
            train_df=X_full,
            test_df=test_full,
            categorical_cols=cat_cols,
            min_freq=min_freq,
            exclude_cols=exclude_cols,
        )

    X_full, test_full, cat_cols, categories_map = make_categorical_safe(X_full, test_full, categorical_cols=cat_cols)

    # train/test 컬럼 정렬 및 누락 컬럼 처리
    test_full = test_full.reindex(columns=X_full.columns, fill_value=np.nan)


    # Save preprocessing meta for reproducibility (included in fold model bundle under meta/)
    try:
        feature_columns = list(X_full.columns)
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
            "missing_present_flags": {
                "enable": bool(present_enable),
                "suffix": str(present_suffix),
                "skip_all_missing": bool(present_skip_all_missing),
                "flag_cols": list(present_cols_full),
            },
            "dropped_constant_cols": list(dropped_const),
            "categorical_cols": list(cat_cols),
            "rare_bucket": {
                "enable": bool(rare_enabled),
                "min_freq": int(_select(cfg, "data.rare_bucket.min_freq", 3)) if bool(rare_enabled) else None,
                "n_affected_cols": int(sum(1 for _c, _v in rare_info.items() if int(_v.get("n_rare", 0)) > 0)),
                "per_col_n_rare": {str(_c): int(_v.get("n_rare", 0)) for _c, _v in rare_info.items()},
            },

            "multilabel": {
                "enable": bool(multilabel_enabled),
                "cols": list(_select(cfg, "data.multilabel.cols", [])) if bool(multilabel_enabled) else [],
                "per_col": multilabel_info,
            },
            "n_features": int(X_full.shape[1]),
            "categories_map_summary": cat_summary,
        }
        save_yaml_in_output_dir("preprocess_meta.yaml", preprocess_payload)
    except Exception as e:
        print(f"[Warn] saving preprocess meta failed: {e}")
    # Dataset summary
    print(f"[Info] Train rows={len(train)} | Test rows={len(test_full)} | Features={X_full.shape[1]}")
    print("[Info] CV/Optuna uses fold-fit preprocessing on raw features (leakage-free)")
    print(f"[Info] Pos rate={y.mean():.4f} ({int(y.sum())}/{len(y)})")
    print(f"[Info] Dropped missing>{cfg.data.drop_missing_threshold}: {len(dropped_missing)} cols -> {dropped_missing}")
    if present_enable and present_cols_full:
        # Keep log concise
        preview = present_cols_full if len(present_cols_full) <= 20 else present_cols_full[:20] + ['...']
        print(f"[Info] Added is_present flags: {len(present_cols_full)} cols -> {preview}")
    if dropped_const:
        print(f"[Info] Dropped constant cols: {len(dropped_const)} cols -> {dropped_const}")
    print(f"[Info] Categorical cols (full-fit): {len(cat_cols)}")

    if run is not None:
        # Put some key info into run summary
        run.summary["data/train_rows"] = int(len(train))
        run.summary["data/test_rows"] = int(len(test_full))
        run.summary["data/pos_rate"] = float(y.mean())
        run.summary["data/n_features"] = int(X_full.shape[1])
        run.summary["data/dropped_missing_cols"] = len(dropped_missing)
        run.summary["data/dropped_constant_cols"] = len(dropped_const)
        run.summary["data/n_categorical_cols"] = len(cat_cols)

        run.summary["data/multilabel_enabled"] = bool(multilabel_enabled)
        run.summary["data/multilabel_cols"] = int(len(list(_select(cfg, "data.multilabel.cols", [])) or [])) if bool(multilabel_enabled) else 0


    # Optuna tuning (optional): updates cfg.model with best params before the main CV
    optuna_overrides = run_optuna_tuning(cfg, X_raw, y, run)
    if optuna_overrides:
        for k, v in optuna_overrides.items():
            cfg.model[k] = v
        print(f"[Optuna] Applied best overrides to cfg.model: {optuna_overrides}")

    # CV (+ fold-ensemble prediction)
    use_fold_ensemble = bool(getattr(cfg.train, "use_fold_ensemble", True))
    save_fold_models = bool(cfg.train.save_model) and use_fold_ensemble
    fold_model_basepath = Path(cfg.train.model_path) if save_fold_models else None

    oof_proba, test_proba_cv, best_iters, aucs, f1s, fold_model_paths, cv_feat_imp = cv_train_foldfit(
        cfg, X_raw, y, test_raw, run, feature_columns_ref=list(X_full.columns),
        save_fold_models=save_fold_models,
        fold_model_basepath=fold_model_basepath,
    )

    auc_mean, auc_std = float(np.mean(aucs)), float(np.std(aucs))
    f1_mean, f1_std = float(np.mean(f1s)), float(np.std(f1s))
    best_iter_final = int(np.median(best_iters))

    print("\n[CV] Done")
    print(f"AUC mean={auc_mean:.5f} ± {auc_std:.5f}")
    print(f"F1(best-threshold) mean={f1_mean:.5f} ± {f1_std:.5f}")
    print(f"Use n_estimators={best_iter_final} (median best_iteration) for final fit")

    # Optional: OOF-based threshold tuning
    threshold = float(cfg.train.threshold)
    best_f1 = float(f1_score(y.values, (oof_proba >= threshold).astype(int)))
    if bool(cfg.train.optimize_threshold.enable):
        t, f1_best = search_best_threshold(
                y_true=y.values,
                proba=oof_proba,
                method=str(_select(cfg, "train.optimize_threshold.method", "pr_curve")),
                min_t=float(cfg.train.optimize_threshold.clamp_min),
                max_t=float(cfg.train.optimize_threshold.clamp_max),
                grid_start=float(_select(cfg, "train.optimize_threshold.grid.start", _select(cfg, "train.optimize_threshold.grid_start", 0.05))),
                grid_end=float(_select(cfg, "train.optimize_threshold.grid.end", _select(cfg, "train.optimize_threshold.grid_end", 0.95))),
                grid_step=float(_select(cfg, "train.optimize_threshold.grid.step", _select(cfg, "train.optimize_threshold.grid_step", 0.01))),
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
            "cv_f1_mean_bestthr": float(f1_mean),
            "cv_f1_std_bestthr": float(f1_std),
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
        final_model, test_proba = fit_final_and_predict(cfg, X_full, y, test_full, cat_cols_full, best_iter_final)
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
                        # Add fold-specific preprocessing meta (if exists)
                        try:
                            for mp in output_dir.glob("fold*_preprocess_meta.yaml"):
                                if mp.exists() and mp.is_file():
                                    zf.write(mp, arcname=f"meta/{mp.name}")
                        except Exception as e:
                            print(f"[Warn] adding fold preprocess meta files failed: {e}")

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
            booster = final_model.booster_ # type: ignore
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
                        # Add fold-specific preprocessing meta (if exists)
                        try:
                            for mp in output_dir.glob("fold*_preprocess_meta.yaml"):
                                if mp.exists() and mp.is_file():
                                    zf.write(mp, arcname=f"meta/{mp.name}")
                        except Exception as e:
                            print(f"[Warn] adding fold preprocess meta files failed: {e}")
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
                importances = final_model.feature_importances_ # type: ignore

            fi = pd.DataFrame({
                "feature": X_full.columns,
                "importance": importances,
            }).sort_values("importance", ascending=False)
            try:
                import wandb # type: ignore
                table = wandb.Table(dataframe=fi.head(200)) # type: ignore
                wandb.log({"feature_importance_top200": table}) # type: ignore
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
    main() # type: ignore
