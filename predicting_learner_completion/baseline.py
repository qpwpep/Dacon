import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.preprocessing import TargetEncoder  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore


def main():
    # -----------------------------
    # 1) Load
    # -----------------------------
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    TARGET = "completed"
    FEATURES = ["class1", "re_registration", "inflow_route", "time_input", "is_major_it"]

    # -----------------------------
    # 2) Drop columns with too many missings (train 기준)
    #    + 안전하게 errors='ignore'
    # -----------------------------
    missing_ratio = train.drop(columns=[TARGET], errors="ignore").isnull().mean()
    columns_to_drop = missing_ratio[missing_ratio > 0.8].index.tolist()
    columns_to_drop.append("incumbents_lecture_scale_reason")

    train = train.drop(columns=columns_to_drop, errors="ignore")
    test = test.drop(columns=columns_to_drop, errors="ignore")

    # -----------------------------
    # 3) Feature engineering: is_major_it
    #    major_field가 드랍되었을 가능성도 고려
    # -----------------------------
    for df in (train, test):
        if "major_field" in df.columns:
            # na=False로 NaN 안전 처리
            df["is_major_it"] = (
                df["major_field"].astype("string").str.contains("IT", regex=True, na=False).astype(int)
            )
        else:
            df["is_major_it"] = 0

    # -----------------------------
    # 4) Select features/target
    # -----------------------------
    X_train = train[FEATURES].copy()
    y_train = train[TARGET].copy()
    X_test = test[FEATURES].copy()

    # -----------------------------
    # 5) Categorical vs Numeric split
    #    - 기본: dtype 기반
    #    - 추가: 명시적으로 범주 취급하고 싶은 컬럼은 force_categorical에 넣기
    # -----------------------------
    force_categorical = ["class1", "re_registration", "inflow_route"]

    categorical_cols = X_train.select_dtypes(include=["object", "string", "bool", "category"]).columns.tolist()
    for c in force_categorical:
        if c in X_train.columns and c not in categorical_cols:
            categorical_cols.append(c)

    numeric_cols = [c for c in FEATURES if c not in categorical_cols]

    # 범주형은 문자열로 통일(타겟 인코더가 category로 처리하게)
    for c in categorical_cols:
        X_train[c] = X_train[c].astype("string")
        X_test[c] = X_test[c].astype("string")

    # -----------------------------
    # 6) Preprocess
    #    - Cat: missing -> "__MISSING__", then TargetEncoder(cv=5 cross-fitting)
    #    - Num: median impute
    # -----------------------------
    transformers = []

    if categorical_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                # sklearn TargetEncoder는 내부적으로 cv cross-fitting을 수행해 누수를 줄임
                ("target_enc", TargetEncoder(cv=5, shuffle=True, random_state=42)),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_cols))

    if numeric_cols:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("num", num_pipe, numeric_cols))

    preprocess = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # -----------------------------
    # 7) Model
    # -----------------------------
    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    clf.fit(X_train, y_train)

    # -----------------------------
    # 8) Predict & Save submission
    # -----------------------------
    pred = clf.predict(X_test)

    submission = pd.read_csv("data/sample_submission.csv")
    submission[TARGET] = pred
    submission.to_csv("submit.csv", index=False)


if __name__ == "__main__":
    main()
