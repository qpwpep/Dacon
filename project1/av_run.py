# -*- coding: utf-8 -*-
"""
Adversarial Validation for Dacon 2025 Power Consumption Challenge
- Files: /mnt/data/train.csv, /mnt/data/test.csv, /mnt/data/building_info.csv
- Output:
    - ./av_outputs/av_report.txt
    - ./av_outputs/feature_importance.csv
    - ./av_outputs/train_test_likeness.csv  (train 샘플별 test-유사도)
    - ./av_outputs/test_likeness.csv        (test 샘플별 test-유사도)
"""

import os
import re
import gc
import math
import warnings
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")


# ----------------------------
# 0. Paths & I/O helpers
# ----------------------------
DATA_DIR = "./data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
BINFO_PATH = os.path.join(DATA_DIR, "building_info.csv")

OUT_DIR = "./av_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------
# 1. Utility: flexible column detection
# ----------------------------
def detect_cols(df: pd.DataFrame):
    cols = df.columns.str.lower()

    # target
    target_col = None
    for c in ["answer", "target", "y", "power", "power_consumption", "electricity", "kwh"]:
        if c in cols.values:
            target_col = df.columns[cols.get_loc(c)]
            break

    # id-like (combined) -> e.g., num_date_time
    combined_id_col = None
    for c in ["num_date_time", "id", "row_id"]:
        if c in cols.values:
            combined_id_col = df.columns[cols.get_loc(c)]
            break

    # building key
    bld_col = None
    for c in ["num", "building", "building_id", "bld_id", "site_id", "건물번호"]:
        if c in cols.values:
            bld_col = df.columns[cols.get_loc(c)]
            break

    # datetime
    dt_col = None
    for c in ["date_time", "datetime", "timestamp", "time", "ds", "일시"]:
        if c in cols.values:
            dt_col = df.columns[cols.get_loc(c)]
            break

    return target_col, combined_id_col, bld_col, dt_col


def split_num_datetime(df: pd.DataFrame, combined_col: str) -> Tuple[str, str]:
    """
    Split a combined key like 'num_date_time' into (num, date_time).
    Heuristics: last '_' separates datetime; head part treated as 'num' (int).
    """
    if combined_col not in df.columns:
        return None, None

    # Try to split at the first underscore that precedes date/time-like pattern
    def splitter(x: str):
        # find last underscore
        if pd.isna(x):
            return np.nan, np.nan
        xs = str(x)
        # Common pattern: "{num}_{YYYY-MM-DD HH:MM:SS}"
        pos = xs.find("_")
        if pos == -1:
            return xs, np.nan
        left = xs[:pos]
        right = xs[pos+1:]
        return left, right

    left_right = df[combined_col].astype(str).map(splitter)
    df["__num_tmp"], df["__dt_tmp"] = zip(*left_right)

    # try to coerce num -> int
    try:
        df["__num_tmp"] = df["__num_tmp"].astype(int)
    except Exception:
        # leave as string if int conversion fails
        pass

    # choose column names that won't collide
    num_col = "num" if "num" not in df.columns else "__num"
    dt_col = "date_time" if "date_time" not in df.columns else "__date_time"

    df[num_col] = df["__num_tmp"]
    df[dt_col] = df["__dt_tmp"]

    df.drop(columns=["__num_tmp", "__dt_tmp"], inplace=True)

    return num_col, dt_col


def parse_datetime(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """Add time features: hour, dow, month, day, is_weekend."""
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df["hour"] = df[dt_col].dt.hour
    df["dow"] = df[dt_col].dt.dayofweek
    df["month"] = df[dt_col].dt.month
    df["day"] = df[dt_col].dt.day
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def merge_building_info(df: pd.DataFrame, binfo: pd.DataFrame, key: str) -> pd.DataFrame:
    bcols = binfo.columns.str.lower()
    bkey = None
    for c in [key, "num", "building", "building_id", "bld_id", "site_id"]:
        if c in bcols.values:
            bkey = binfo.columns[bcols.get_loc(c)]
            break
    if bkey is None:
        # fallback: if only 1 integer-like col, treat as key
        int_like = [c for c in binfo.columns if pd.api.types.is_integer_dtype(binfo[c])]
        bkey = int_like[0] if int_like else binfo.columns[0]

    merged = df.merge(binfo, left_on=key, right_on=bkey, how="left")
    return merged

# --- ADD: numeric-like object columns -> numeric coercion utility ---
def coerce_numeric_like(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    숫자형처럼 보이는 object 컬럼을 float로 강제 변환합니다.
    '-', '', 공백, 'nan' 등은 NaN으로 처리합니다.
    유효 숫자 비율(valid_ratio)이 임계값 이상일 때만 변환합니다.
    """
    df = df.copy()
    if cols is None:
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    else:
        obj_cols = [c for c in cols if c in df.columns and df[c].dtype == "object"]

    for c in obj_cols:
        s = df[c].astype(str)
        s = s.replace({"-": np.nan, "": np.nan, "NaN": np.nan, "nan": np.nan})
        cleaned = s.str.replace(r"[^0-9\.\-]+", "", regex=True)  # 숫자/부호/소수점만 유지 (단위/콤마 제거)
        cleaned = cleaned.replace({"": np.nan})
        num = pd.to_numeric(cleaned, errors="coerce")

        valid_ratio = np.isfinite(num).mean()
        if valid_ratio >= 0.6:
            df[c] = num
    return df

# --- ADD: normalize placeholder strings to NaN on all object cols ---
def clean_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """
    object 컬럼의 플레이스홀더/빈값을 NaN으로 정규화하고 strip 처리.
    """
    df = df.copy()
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        return df

    repl = {
        "-": np.nan, " - ": np.nan, "—": np.nan, "–": np.nan,
        "": np.nan, " ": np.nan, "NaN": np.nan, "nan": np.nan,
        "없음": np.nan, "무": np.nan, "None": np.nan, "none": np.nan
    }
    for c in obj_cols:
        s = df[c].astype(str).str.strip()
        s = s.replace(repl)
        df[c] = s
    return df


def coerce_numeric_like(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    숫자처럼 보이는 object 컬럼을 float로 강제 변환. 유효 숫자 비율이 높을 때만 변환.
    """
    df = df.copy()
    if cols is None:
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    else:
        obj_cols = [c for c in cols if c in df.columns and df[c].dtype == "object"]

    for c in obj_cols:
        s = df[c].astype(str)
        s = s.replace({"-": np.nan, "": np.nan, "NaN": np.nan, "nan": np.nan})
        cleaned = s.str.replace(r"[^0-9\.\-]+", "", regex=True)
        cleaned = cleaned.replace({"": np.nan})
        num = pd.to_numeric(cleaned, errors="coerce")
        valid_ratio = np.isfinite(num).mean()
        if valid_ratio >= 0.6:
            df[c] = num
    return df


# ----------------------------
# 2. Load data
# ----------------------------
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
binfo = pd.read_csv(BINFO_PATH)

# --- ADD: save original column sets before any parsing/merges ---
ORIG_TRAIN_COLS = set(train.columns)
ORIG_TEST_COLS  = set(test.columns)


# Detect important columns
tcol, comb_col_tr, bcol_tr, dtcol_tr = detect_cols(train)
_, comb_col_te, bcol_te, dtcol_te = detect_cols(test)

# If combined id exists, split to get num & date_time
if comb_col_tr is not None:
    num_from_comb_tr, dt_from_comb_tr = split_num_datetime(train, comb_col_tr)
    if bcol_tr is None:
        bcol_tr = num_from_comb_tr
    if dtcol_tr is None:
        dtcol_tr = dt_from_comb_tr

if comb_col_te is not None:
    num_from_comb_te, dt_from_comb_te = split_num_datetime(test, comb_col_te)
    if bcol_te is None:
        bcol_te = num_from_comb_te
    if dtcol_te is None:
        dtcol_te = dt_from_comb_te

# Finalize building key / datetime column names
bld_key = bcol_tr or bcol_te
dt_key = dtcol_tr or dtcol_te

# Defensive: ensure exist
if bld_key is None:
    # pick an integer-like column as building key
    int_cols_tr = [c for c in train.columns if pd.api.types.is_integer_dtype(train[c])]
    bld_key = int_cols_tr[0] if int_cols_tr else train.columns[0]

if dt_key is None:
    # pick a time-like column if exist; else skip time features
    candidates = [c for c in train.columns if "time" in c.lower() or "date" in c.lower()]
    dt_key = candidates[0] if candidates else None

# Add time features if possible
if dt_key is not None and dt_key in train.columns:
    train = parse_datetime(train, dt_key)
if dt_key is not None and dt_key in test.columns:
    test = parse_datetime(test, dt_key)

# Merge building_info
if bld_key is not None and bld_key in train.columns:
    train = merge_building_info(train, binfo, key=bld_key)
if bld_key is not None and bld_key in test.columns:
    test = merge_building_info(test, binfo, key=bld_key)

# --- ADD: normalize placeholders BEFORE numeric coercion ---
train = clean_placeholders(train)
test  = clean_placeholders(test)

# --- ADD: coerce numeric-like columns after parsing & merges ---
train = coerce_numeric_like(train)
test  = coerce_numeric_like(test)

# (선택) 특정 컬럼을 확실히 숫자화
for d in (train, test):
    for col in ["ESS저장용량(kWh)", "PCS용량(kW)"]:
        if col in d.columns:
            d[col] = pd.to_numeric(
                d[col].astype(str).str.replace(r"[^0-9\.\-]+", "", regex=True).replace({"": np.nan, "-": np.nan}),
                errors="coerce"
            )


# ----------------------------
# 3. Prepare AV dataset
# ----------------------------
# Remove target & obvious non-features
drop_candidates = set([
    tcol,                  # target (detect_cols로 잡힌 경우)
    "answer", "target", "y",
    "num_date_time", "id", "row_id",
])

for hard in ["풍속(m/s)", "일조(hr)"]:
    if hard in train.columns:
        drop_candidates.add(hard)

# (중요) 원시 datetime 컬럼은 파생변수로 대체하므로 드랍
if dt_key is not None:
    drop_candidates.add(dt_key)

# --- ORIG_TRAIN_COLS/ORIG_TEST_COLS 안전 확보 (앞에서 정의 안 했을 경우 대비) ---
try:
    ORIG_TRAIN_COLS
except NameError:
    ORIG_TRAIN_COLS = set(train.columns)
try:
    ORIG_TEST_COLS
except NameError:
    ORIG_TEST_COLS = set(test.columns)

# --- train 전용 + 이름이 타깃스러운 컬럼 자동 감지 ---
def looks_like_target(name: str) -> bool:
    s = str(name).lower()
    keys = [
        "target", "answer", "label", "labels", "y",
        "kwh", "power", "electric", "electricity",
        "전력", "소비", "소비량"
    ]
    return any(k in s for k in keys)

train_only_cols = [c for c in ORIG_TRAIN_COLS if c not in ORIG_TEST_COLS]
suspected_targets = [c for c in train_only_cols if looks_like_target(c)]

for hard_name in ["전력소비량(kWh)", "전력소비량", "소비전력(kWh)"]:
    if hard_name in ORIG_TRAIN_COLS:
        suspected_targets.append(hard_name)

for c in suspected_targets:
    if c not in {bld_key}:  # dt_key는 이미 drop 대상
        drop_candidates.add(c)

# --- 한쪽만 NaN이거나 한쪽만 상수인 컬럼 자동 드랍 ---
from typing import List

def cols_with_extreme_nan_or_constant(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                      nan_gap: float = 0.95) -> List[str]:
    cand = []
    common = set(train_df.columns) & set(test_df.columns)
    for c in common:
        if c in {dt_key, bld_key, tcol}:
            continue
        tr = train_df[c]
        te = test_df[c]
        tr_nan = tr.isna().mean()
        te_nan = te.isna().mean()
        if abs(tr_nan - te_nan) >= nan_gap:
            cand.append(c)
            continue
        tr_nu = tr.nunique(dropna=True)
        te_nu = te.nunique(dropna=True)
        if (tr_nu <= 1 and te_nu > 1) or (te_nu <= 1 and tr_nu > 1):
            cand.append(c)
    return cand

extreme_cols = cols_with_extreme_nan_or_constant(train, test, nan_gap=0.95)
for c in extreme_cols:
    if c not in {bld_key}:
        drop_candidates.add(c)

# --- 분포 비겹침: 분위수 기반 ---
def cols_with_disjoint_ranges(train_df: pd.DataFrame, test_df: pd.DataFrame,
                              q_lo: float = 0.01, q_hi: float = 0.99, eps: float = 1e-12) -> List[str]:
    cand = []
    common = set(train_df.columns) & set(test_df.columns)
    for c in common:
        if c in {dt_key, bld_key, tcol}:
            continue
        if not pd.api.types.is_numeric_dtype(train_df[c]) or not pd.api.types.is_numeric_dtype(test_df[c]):
            continue
        tr = train_df[c].dropna()
        te = test_df[c].dropna()
        if len(tr) == 0 or len(te) == 0:
            continue
        tr_lo, tr_hi = np.quantile(tr, [q_lo, q_hi])
        te_lo, te_hi = np.quantile(te, [q_lo, q_hi])
        if tr_hi + eps < te_lo or te_hi + eps < tr_lo:
            cand.append(c)
    return cand

disjoint_cols = cols_with_disjoint_ranges(train, test, q_lo=0.01, q_hi=0.99)
for c in disjoint_cols:
    drop_candidates.add(c)

# --- 분포 비겹침: min/max 완전 비겹침 ---
def cols_with_nonoverlap_minmax(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[str]:
    cand = []
    common = set(train_df.columns) & set(test_df.columns)
    for c in common:
        if c in {dt_key, bld_key, tcol}:
            continue
        if not pd.api.types.is_numeric_dtype(train_df[c]) or not pd.api.types.is_numeric_dtype(test_df[c]):
            continue
        tr = train_df[c].dropna().to_numpy()
        te = test_df[c].dropna().to_numpy()
        if tr.size == 0 or te.size == 0:
            continue
        tr_min, tr_max = np.min(tr), np.max(tr)
        te_min, te_max = np.min(te), np.max(te)
        if tr_max < te_min or te_max < tr_min:
            cand.append(c)
    return cand

nonoverlap_cols = cols_with_nonoverlap_minmax(train, test)
for c in nonoverlap_cols:
    drop_candidates.add(c)

# --- 분포 겹침률(히스토그램) 낮음 ---
def cols_with_low_hist_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame,
                               bins: int = 32, overlap_thresh: float = 0.02) -> List[str]:
    cand = []
    common = set(train_df.columns) & set(test_df.columns)
    for c in common:
        if c in {dt_key, bld_key, tcol}:
            continue
        if not pd.api.types.is_numeric_dtype(train_df[c]) or not pd.api.types.is_numeric_dtype(test_df[c]):
            continue
        tr = train_df[c].dropna().to_numpy()
        te = test_df[c].dropna().to_numpy()
        if tr.size == 0 or te.size == 0:
            continue
        lo = max(np.min(tr), np.min(te))
        hi = min(np.max(tr), np.max(te))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            cand.append(c)
            continue
        tr_hist, edges = np.histogram(tr, bins=bins, range=(lo, hi), density=True)
        te_hist, _     = np.histogram(te, bins=bins, range=(lo, hi), density=True)
        overlap = np.minimum(tr_hist, te_hist).sum() * (edges[1] - edges[0])
        if overlap <= overlap_thresh:
            cand.append(c)
    return cand

low_overlap_cols = cols_with_low_hist_overlap(train, test, bins=32, overlap_thresh=0.02)
for c in low_overlap_cols:
    drop_candidates.add(c)

# Keep bld_key as feature (유지), 나머지 드랍 (dt_key는 보존하지 않음)
drop_cols_train = [c for c in drop_candidates if c in train.columns and c not in {bld_key}]
drop_cols_test  = [c for c in drop_candidates if c in test.columns and c not in {bld_key}]

train_feat = train.drop(columns=drop_cols_train, errors="ignore").copy()
test_feat  = test.drop(columns=drop_cols_test, errors="ignore").copy()

# is_test 라벨
train_feat["is_test"] = 0
test_feat["is_test"] = 1

# 스키마 맞추기
common_cols = sorted(set(train_feat.columns) | set(test_feat.columns))
train_feat = train_feat.reindex(columns=common_cols)
test_feat  = test_feat.reindex(columns=common_cols)

# 결합 및 분리
df = pd.concat([train_feat, test_feat], axis=0, ignore_index=True)
y = df["is_test"].astype(int)
X = df.drop(columns=["is_test"])

if "일조(hr)" in df.columns:
    tr = df.loc[y == 0, "일조(hr)"]
    te = df.loc[y == 1, "일조(hr)"]
    print("[DEBUG] 일조(hr) train min/max/NaN:", np.nanmin(tr), np.nanmax(tr), tr.isna().mean())
    print("[DEBUG] 일조(hr) test  min/max/NaN:", np.nanmin(te), np.nanmax(te), te.isna().mean())


# --- Univariate separability pruning (AUROC & 카테고리 disjoint) ---
SEPA_AUC_THRESH = 0.995
OVERLAP_EPS     = 1e-12

def auroc_univariate(x: pd.Series, y_bin: pd.Series) -> float:
    s = pd.Series(x).astype(float)
    m = pd.Series(y_bin).astype(int)
    mask = s.notna() & m.isin([0, 1])
    s = s[mask]
    m = m[mask]
    n_pos = int((m == 1).sum())
    n_neg = int((m == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = s.rank(method="average")
    rank_sum_pos = ranks[m == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    if not np.isfinite(auc):
        return 0.5
    return max(float(auc), 1.0 - float(auc))

num_cols_initial = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols_initial = [c for c in X.columns if c not in num_cols_initial]

to_drop = set()
for c in num_cols_initial:
    auc1 = auroc_univariate(X[c], y)
    if auc1 >= SEPA_AUC_THRESH:
        to_drop.add(c)
        continue
    x0 = X.loc[y == 0, c].dropna().to_numpy()
    x1 = X.loc[y == 1, c].dropna().to_numpy()
    if x0.size and x1.size:
        if (np.nanmax(x0) + OVERLAP_EPS) < np.nanmin(x1) or (np.nanmax(x1) + OVERLAP_EPS) < np.nanmin(x0):
            to_drop.add(c)

for c in cat_cols_initial:
    s0 = set(X.loc[y == 0, c].astype(str).dropna().unique().tolist())
    s1 = set(X.loc[y == 1, c].astype(str).dropna().unique().tolist())
    if len(s0) == 0 or len(s1) == 0 or s0.isdisjoint(s1):
        to_drop.add(c)

if to_drop:
    X = X.drop(columns=list(to_drop))

# 최종 타입 분리(중요: OHE 이전)
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]

# OHE: dense 출력
ohe = OneHotEncoder(
    handle_unknown="ignore",
    min_frequency=0.01,
    sparse_output=False
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", ohe, cat_cols),
    ],
    remainder="drop",
    sparse_threshold=0.3
)

clf = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=None,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0,
    random_state=42
)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", clf)
])

# Split
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Fit & Eval
pipe.fit(X_tr, y_tr)
proba = pipe.predict_proba(X_va)[:, 1]
auc = roc_auc_score(y_va, proba)


# ----------------------------
# 4. Permutation Importance
# ----------------------------
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

N_PI = min(5000, X_va.shape[0])
y_va_series = pd.Series(y_va)
idx_pos = y_va_series[y_va_series == 1].index.to_numpy()
idx_neg = y_va_series[y_va_series == 0].index.to_numpy()

if len(idx_pos) == 0 or len(idx_neg) == 0:
    pi = None
    feat_names = []
    imp_df = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
else:
    n_each = max(1, N_PI // 2)
    rng = np.random.RandomState(42)
    take_pos = idx_pos if len(idx_pos) <= n_each else rng.choice(idx_pos, size=n_each, replace=False)
    take_neg = idx_neg if len(idx_neg) <= n_each else rng.choice(idx_neg, size=n_each, replace=False)
    sample_idx = np.concatenate([take_pos, take_neg])
    rng.shuffle(sample_idx)

    def auc_scorer(estimator, X_, y_):
        proba_ = estimator.predict_proba(X_)[:, 1]
        if np.unique(y_).size < 2:
            return 0.5
        return roc_auc_score(y_, proba_)

    pi = permutation_importance(
        pipe,
        X_va.loc[sample_idx],
        y_va.loc[sample_idx],
        n_repeats=5,
        random_state=42,
        scoring=auc_scorer,
    )

    pre = pipe.named_steps["preprocess"]
    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        num_cols_used, cat_cols_used, cat_feature_names = [], [], []
        for name, trans, cols in pre.transformers_:
            if name == "num":
                num_cols_used = list(cols)
            elif name == "cat":
                cat_cols_used = list(cols)
                try:
                    ohe = pre.named_transformers_["cat"]
                    cat_feature_names = ohe.get_feature_names_out(cat_cols_used)
                except Exception:
                    cat_feature_names = np.array([f"cat__{i}" for i in range(len(cat_cols_used))], dtype=object)
        feat_names = np.concatenate([np.array(num_cols_used, dtype=object), np.array(cat_feature_names, dtype=object)])

    n_imp = pi.importances_mean.shape[0]
    n_feat = len(feat_names)
    m = min(n_imp, n_feat)

    imp_df = pd.DataFrame(
        {
            "feature": np.array(feat_names[:m], dtype=object),
            "importance_mean": pi.importances_mean[:m],
            "importance_std": pi.importances_std[:m],
        }
    ).sort_values("importance_mean", ascending=False)

imp_df.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)


# ----------------------------
# 5. Save per-row "test-likeness"
# ----------------------------
# predict_proba on full rows
full_proba = pipe.predict_proba(X)[:, 1]
likeness = pd.DataFrame({
    "index": np.arange(len(X)),
    "is_test_label": y.values,
    "test_likeness": full_proba
})

# Split back to train/test partitions
n_train = len(train_feat)
train_like = likeness.iloc[:n_train].copy()
test_like  = likeness.iloc[n_train:].copy()

train_like.to_csv(os.path.join(OUT_DIR, "train_test_likeness.csv"), index=False)
test_like.to_csv(os.path.join(OUT_DIR, "test_likeness.csv"), index=False)

# ----------------------------
# 6. Report
# ----------------------------
report = []
report.append(f"Adversarial Validation AUC: {auc:.4f}")
report.append("")
report.append("Top 20 permuted features (by mean AUC drop):")
report.append(imp_df.head(20).to_string(index=False))

with open(os.path.join(OUT_DIR, "av_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("\n".join(report))
print(f"\nSaved:\n - {os.path.join(OUT_DIR, 'av_report.txt')}"
      f"\n - {os.path.join(OUT_DIR, 'feature_importance.csv')}"
      f"\n - {os.path.join(OUT_DIR, 'train_test_likeness.csv')}"
      f"\n - {os.path.join(OUT_DIR, 'test_likeness.csv')}")
