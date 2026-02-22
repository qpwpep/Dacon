from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ID_COL = "id"
TARGET_COL = "Heart Disease"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank-average blend multiple submission CSV files."
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help="Submission files to blend. Each file must contain 'id' and 'Heart Disease'.",
    )
    parser.add_argument("--output-path", default="outputs/submission_rank_blend.csv")
    parser.add_argument(
        "--weights",
        default=None,
        help="Comma-separated weights aligned with --input-files. If omitted, equal weights are used.",
    )
    return parser.parse_args()


def parse_weights(weights_arg: str | None, n_files: int) -> list[float]:
    if weights_arg is None:
        return [1.0 / n_files] * n_files

    parts = [x.strip() for x in weights_arg.split(",") if x.strip()]
    if len(parts) != n_files:
        raise ValueError(
            f"weights count mismatch: expected {n_files}, got {len(parts)}"
        )

    weights = [float(x) for x in parts]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("weights sum must be positive.")
    return [w / weight_sum for w in weights]


def validate_submission_schema(df: pd.DataFrame, file_path: Path) -> None:
    required = {ID_COL, TARGET_COL}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing required columns in {file_path}: "
            f"required={required}, actual={set(df.columns)}"
        )
    if df[ID_COL].duplicated().any():
        raise ValueError(f"Duplicate id found in {file_path}.")


def main() -> None:
    args = parse_args()
    input_paths = [Path(x) for x in args.input_files]
    weights = parse_weights(args.weights, len(input_paths))

    frames: list[pd.DataFrame] = []
    for path in input_paths:
        df = pd.read_csv(path, usecols=[ID_COL, TARGET_COL])
        validate_submission_schema(df, path)
        frames.append(df)

    base = frames[0][[ID_COL]].copy()
    id_index = pd.Index(frames[0][ID_COL])

    for i, df in enumerate(frames[1:], start=1):
        if not id_index.equals(pd.Index(df[ID_COL])):
            raise ValueError(
                f"ID order/set mismatch between first file and file index {i}: "
                f"{input_paths[i]}"
            )

    blend_score = pd.Series(0.0, index=base.index, dtype="float64")
    for w, df in zip(weights, frames):
        ranks = df[TARGET_COL].rank(method="average", pct=True)
        blend_score += w * ranks

    output_df = pd.DataFrame({ID_COL: base[ID_COL], TARGET_COL: blend_score})
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print("[Blend Complete]")
    print(f"input_files={len(input_paths)}")
    print(f"weights={weights}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
