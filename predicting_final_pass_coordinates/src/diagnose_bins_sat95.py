from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import yaml

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


NO_END_TYPES = {
    "Aerial Clearance", "Block", "Catch", "Deflection", "Error", "Foul",
    "Handball_Foul", "Hit", "Intervention", "Parry", "Take-On",
}


def unify_frame_to_ref_team(g: pd.DataFrame, ref_team_id: int, field_x: float, field_y: float) -> pd.DataFrame:
    g = g.copy()
    opp = (g["team_id"].values != ref_team_id)
    for col in ["start_x", "end_x"]:
        g.loc[opp, col] = field_x - g.loc[opp, col].astype(float)
    for col in ["start_y", "end_y"]:
        g.loc[opp, col] = field_y - g.loc[opp, col].astype(float)
    return g


def compute_kfold_game_sets(episode_game_ids: List[int], n_folds: int, fold_idx: int, seed: int) -> Tuple[set, set]:
    games = np.array(sorted(set(map(int, episode_game_ids))), dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(games)
    folds = np.array_split(games, n_folds)
    valid_games = set(map(int, folds[fold_idx].tolist()))
    train_games = set(map(int, games.tolist())) - valid_games
    return train_games, valid_games


def read_config(config_yaml: str) -> Dict[str, Any]:
    with open(config_yaml, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_train_sorted(train_csv: str) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    sort_cols = ["game_episode"] + (["action_id"] if "action_id" in df.columns else []) + ["time_seconds"]
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def build_samples(
    df_sorted: pd.DataFrame,
    *,
    field_x: float,
    field_y: float,
    dt_clip_sec: float,
    dt_norm_ref_sec: float,
    target_policy: str,
    max_tail_k: int,
) -> pd.DataFrame:
    """
    baseline 규칙에 맞춘 "샘플 단위" 테이블 생성
    columns: game_id, seq_len, dist_m, dx_norm, dy_norm, sat95, end_mask_rate, no_end_rate, dt_mean
    """
    tp = str(target_policy).strip().lower()
    assert tp in ("all_pass", "last_pass")

    rows = []
    groups = df_sorted.groupby("game_episode")
    it = groups
    if tqdm is not None:
        it = tqdm(groups, total=groups.ngroups, desc=f"Build samples ({tp}, tail={max_tail_k})")

    for game_episode, g in it:
        if max_tail_k and max_tail_k > 0:
            g = g.tail(int(max_tail_k)).reset_index(drop=True)

        type_vals = g["type_name"].fillna("None").astype(str).values
        pass_indices = np.where(type_vals == "Pass")[0]
        if len(pass_indices) == 0:
            continue

        last_is_pass = (type_vals[-1] == "Pass")

        if tp == "last_pass":
            if last_is_pass:
                pass_indices = np.array([len(type_vals) - 1], dtype=int)
            else:
                pass_indices = np.array([int(pass_indices[-1])], dtype=int)

        game_id = int(str(game_episode).split("_", 1)[0])

        # unify cache per team inside episode
        team_vals = g["team_id"].astype(int).values
        uniq_teams = list(dict.fromkeys(team_vals.tolist()))
        unified_cache = {t: unify_frame_to_ref_team(g, t, field_x, field_y) for t in uniq_teams}

        for t_idx in pass_indices.tolist():
            ref_team = int(team_vals[t_idx])
            g_ref = unified_cache[ref_team].iloc[: t_idx + 1].reset_index(drop=True)

            ex_raw = g_ref["end_x"].values.astype(np.float32)
            ey_raw = g_ref["end_y"].values.astype(np.float32)
            if np.isnan(ex_raw[-1]) or np.isnan(ey_raw[-1]):
                continue

            sx_abs = (g_ref["start_x"].values / field_x).astype(np.float32)
            sy_abs = (g_ref["start_y"].values / field_y).astype(np.float32)
            anchor_x = float(sx_abs[-1])
            anchor_y = float(sy_abs[-1])

            dx_norm = float(ex_raw[-1] / field_x) - anchor_x
            dy_norm = float(ey_raw[-1] / field_y) - anchor_y
            dist_m = float(math.sqrt((dx_norm * field_x) ** 2 + (dy_norm * field_y) ** 2))
            sat95 = (abs(dx_norm) > 0.95) or (abs(dy_norm) > 0.95)

            # end_mask stats (exclude last row)
            type_names = g_ref["type_name"].fillna("None").astype(str).values
            no_end = np.isin(type_names, list(NO_END_TYPES))
            end_ok = (~np.isnan(ex_raw)) & (~np.isnan(ey_raw))
            end_mask = (end_ok & (~no_end)).astype(np.float32)
            end_mask[-1] = 0.0

            if len(end_mask) > 1:
                end_mask_rate = float(end_mask[:-1].mean())
                no_end_rate = float(no_end[:-1].mean())
            else:
                end_mask_rate = 0.0
                no_end_rate = 0.0

            # dt mean
            t = g_ref["time_seconds"].values.astype(np.float32)
            dt = np.diff(t, prepend=t[0])
            dt = np.clip(dt, 0.0, dt_clip_sec).astype(np.float32)
            dt = (np.log1p(dt) / np.log1p(dt_norm_ref_sec)).astype(np.float32)
            dt_mean = float(dt.mean())

            rows.append({
                "game_id": game_id,
                "seq_len": int(len(g_ref)),
                "dist_m": dist_m,
                "dx_norm": dx_norm,
                "dy_norm": dy_norm,
                "sat95": bool(sat95),
                "end_mask_rate": end_mask_rate,
                "no_end_rate": no_end_rate,
                "dt_mean": dt_mean,
            })

    return pd.DataFrame(rows)


def summarize_fold_bins(
    samples: pd.DataFrame,
    games: set,
    bins: List[float],
) -> pd.DataFrame:
    sub = samples[samples["game_id"].isin(games)].copy()
    if len(sub) == 0:
        return pd.DataFrame()

    # binning
    labels = []
    for i in range(len(bins) - 1):
        labels.append(f"[{bins[i]}, {bins[i+1]})")
    sub["dist_bin"] = pd.cut(sub["dist_m"], bins=bins, right=False, labels=labels, include_lowest=True)

    total = len(sub)
    grp = sub.groupby("dist_bin", observed=True)
    out = grp.agg(
        n=("dist_m", "size"),
        frac=("dist_m", lambda x: len(x) / total),
        sat95_n=("sat95", "sum"),
        sat95_frac=("sat95", "mean"),
        dist_p50=("dist_m", lambda x: float(np.quantile(x, 0.5))),
        dist_p90=("dist_m", lambda x: float(np.quantile(x, 0.9)) if len(x) else float("nan")),
        seq_len_mean=("seq_len", "mean"),
        dt_mean=("dt_mean", "mean"),
        end_mask_rate=("end_mask_rate", "mean"),
        no_end_rate=("no_end_rate", "mean"),
    ).reset_index()

    return out


def print_fold1_specialness_from_fold_metrics(fold_metrics_csv: str):
    df = pd.read_csv(fold_metrics_csv)
    fin_valid = df[(df["stage"] == "finetune") & (df["split"] == "valid")].copy()
    fin_valid = fin_valid.sort_values("fold")

    def rank_max(col: str):
        s = fin_valid.set_index("fold")[col].astype(float)
        return int(s.idxmax()), float(s.max())

    def rank_min(col: str):
        s = fin_valid.set_index("fold")[col].astype(float)
        return int(s.idxmin()), float(s.min())

    print("\n=== Auto summary from stage_abc_fold_metrics.csv (finetune/valid) ===")
    for col in ["dist_m_p90", "dist_m_p99", "sat95_rate(|dx|>0.95 or |dy|>0.95)"]:
        mf, mv = rank_max(col)
        print(f"MAX {col}: fold={mf}, value={mv}")

    mf, mv = rank_min("no_end_rate_mean(ex_last)")
    print(f"MIN no_end_rate_mean(ex_last): fold={mf}, value={mv}")

    f1 = fin_valid[fin_valid["fold"] == 1].iloc[0]
    others = fin_valid[fin_valid["fold"] != 1]
    print("\nFold1 vs others (finetune/valid):")
    print(f"  dist_m_p90:  fold1={f1['dist_m_p90']:.3f} vs others_mean={others['dist_m_p90'].mean():.3f}")
    print(f"  dist_m_p99:  fold1={f1['dist_m_p99']:.3f} vs others_mean={others['dist_m_p99'].mean():.3f}")
    print(f"  sat95_rate:  fold1={f1['sat95_rate(|dx|>0.95 or |dy|>0.95)']:.6f} vs others_mean={others['sat95_rate(|dx|>0.95 or |dy|>0.95)'].mean():.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--config_yaml", default="config.yaml")
    ap.add_argument("--fold_metrics_csv", default="stage_abc_fold_metrics.csv")
    ap.add_argument("--out_dir", default="outputs/dist_bins_sat95")
    ap.add_argument("--bins", default="0,10,20,30,40,50,60,70,80,1e9")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) auto summary from existing fold metrics
    if Path(args.fold_metrics_csv).exists():
        print_fold1_specialness_from_fold_metrics(args.fold_metrics_csv)

    # 2) rebuild samples and compute bin/sat95 diagnostics
    cfg = read_config(args.config_yaml)
    field_x = float(cfg["data"]["field_x"])
    field_y = float(cfg["data"]["field_y"])
    dt_clip_sec = float(cfg["data"].get("dt_clip_sec", 60.0))
    dt_norm_ref_sec = float(cfg["data"].get("dt_norm_ref_sec", 60.0))

    seed = int(cfg["train"].get("seed", 42))
    n_folds = int(cfg["train"].get("n_folds", 5))
    two_stage = bool(cfg["train"].get("two_stage", True))

    # stage tail params (baseline 관례)
    pre_tail = int(cfg["train"].get("max_tail_k", cfg["data"].get("max_tail_k", 0)))
    st2 = cfg["train"].get("stage2", {}) or {}
    fin_tail = int(st2.get("max_tail_k", cfg["data"].get("max_tail_k", 0)))

    bins = [float(x) for x in args.bins.split(",")]
    # ensure increasing
    assert all(bins[i] < bins[i + 1] for i in range(len(bins) - 1)), "bins must be strictly increasing"

    df = read_train_sorted(args.train_csv)

    # split games are derived from stage1 sample game_ids (pretrain/all_pass) like baseline
    print("\nBuilding pretrain(all_pass) samples just for fold split...")
    pre_samples = build_samples(
        df, field_x=field_x, field_y=field_y,
        dt_clip_sec=dt_clip_sec, dt_norm_ref_sec=dt_norm_ref_sec,
        target_policy="all_pass",
        max_tail_k=pre_tail,
    )
    if len(pre_samples) == 0:
        raise RuntimeError("No pretrain samples built. Check train.csv columns and policy rules.")
    stage0_game_ids = pre_samples["game_id"].tolist()

    # finetune samples
    print("\nBuilding finetune(last_pass) samples...")
    fine_samples = build_samples(
        df, field_x=field_x, field_y=field_y,
        dt_clip_sec=dt_clip_sec, dt_norm_ref_sec=dt_norm_ref_sec,
        target_policy="last_pass",
        max_tail_k=fin_tail,
    )
    if len(fine_samples) == 0:
        raise RuntimeError("No finetune samples built. Check train.csv columns and policy rules.")

    # per fold bin summary for finetune
    all_rows = []
    all_sat_rows = []

    for fold_idx in range(n_folds):
        train_games, valid_games = compute_kfold_game_sets(stage0_game_ids, n_folds, fold_idx, seed)

        for split_name, games in [("train", train_games), ("valid", valid_games)]:
            tbl = summarize_fold_bins(fine_samples, games, bins)
            if len(tbl) == 0:
                continue
            tbl.insert(0, "fold", fold_idx)
            tbl.insert(1, "split", split_name)
            all_rows.append(tbl)

            # sat95-only aggregate (one row per fold/split)
            sub = fine_samples[fine_samples["game_id"].isin(games)]
            sat = sub[sub["sat95"]]
            all_sat_rows.append({
                "fold": fold_idx,
                "split": split_name,
                "n": int(len(sub)),
                "sat95_n": int(len(sat)),
                "sat95_rate": float(len(sat) / len(sub)) if len(sub) else 0.0,
                "sat95_dist_p50": float(np.quantile(sat["dist_m"], 0.5)) if len(sat) else float("nan"),
                "sat95_dist_p90": float(np.quantile(sat["dist_m"], 0.9)) if len(sat) else float("nan"),
                "sat95_seq_len_mean": float(sat["seq_len"].mean()) if len(sat) else float("nan"),
                "sat95_end_mask_rate": float(sat["end_mask_rate"].mean()) if len(sat) else float("nan"),
                "sat95_no_end_rate": float(sat["no_end_rate"].mean()) if len(sat) else float("nan"),
            })

    bins_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    sat_df = pd.DataFrame(all_sat_rows)

    bins_csv = out_dir / "finetune_dist_bins_by_fold.csv"
    sat_csv = out_dir / "finetune_sat95_by_fold.csv"
    bins_df.to_csv(bins_csv, index=False)
    sat_df.to_csv(sat_csv, index=False)

    print("\nSaved:")
    print(" -", bins_csv)
    print(" -", sat_csv)

    # quick fold1 highlight for valid split
    if len(bins_df):
        v = bins_df[bins_df["split"] == "valid"].copy()
        # tail bin >=60m
        tail_bins = v[v["dist_bin"].astype(str).str.startswith("[60.0") | v["dist_bin"].astype(str).str.startswith("[70.0") | v["dist_bin"].astype(str).str.startswith("[80.0")]
        if len(tail_bins):
            f1_tail = tail_bins[tail_bins["fold"] == 1]["frac"].sum()
            others_tail = tail_bins[tail_bins["fold"] != 1].groupby("fold")["frac"].sum().mean()
            print(f"\n[valid tail share >=60m] fold1={f1_tail:.4f} vs others_mean={others_tail:.4f}")


if __name__ == "__main__":
    main()
