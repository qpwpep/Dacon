from __future__ import annotations

import os
import copy
from pathlib import Path
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


# -------------------------
# Utils
# -------------------------
NO_END_TYPES = {
    "Aerial Clearance",
    "Block",
    "Catch",
    "Deflection",
    "Error",
    "Foul",
    "Handball_Foul",
    "Hit",
    "Intervention",
    "Parry",
    "Take-On",
}


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 재현성 우선 (약간 느려질 수 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_cfg: str) -> str:
    if device_cfg.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def euclidean_loss_meters(
    pred_norm: torch.Tensor,
    true_norm: torch.Tensor,
    field_x: float = 105.0,
    field_y: float = 68.0,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    pred_norm/true_norm: [B, 2] (정규화 좌표; absolute(0~1) 또는 delta(-1~1) 모두 OK)
    return: 평균 유클리드 거리 (미터 스케일)
    """
    dx = (pred_norm[:, 0] - true_norm[:, 0]) * field_x
    dy = (pred_norm[:, 1] - true_norm[:, 1]) * field_y
    dist = torch.sqrt(dx * dx + dy * dy + eps)
    return dist.mean()


def unify_frame_to_ref_team(
    g: pd.DataFrame,
    ref_team_id: int,
    field_x: float = 105.0,
    field_y: float = 68.0,
    team_col: str = "team_id",
) -> pd.DataFrame:
    """
    ref_team_id 관점으로 좌표 프레임 통일:
    상대팀 액션의 (start_x,start_y,end_x,end_y)를 180도 회전한다.
    x' = field_x - x, y' = field_y - y
    """
    g = g.copy()
    opp = (g[team_col].values != ref_team_id)

    # NaN이 있어도 field_x - NaN = NaN 이라 안전
    for col in ["start_x", "end_x"]:
        g.loc[opp, col] = field_x - g.loc[opp, col].astype(float) # type: ignore
    for col in ["start_y", "end_y"]:
        g.loc[opp, col] = field_y - g.loc[opp, col].astype(float) # type: ignore

    return g


# -------------------------
# Data
# -------------------------
def build_vocabs(train_csv_path: str):
    df = pd.read_csv(train_csv_path)

    def make_map(series):
        # NaN 처리 + str 통일
        vals = series.fillna("None").astype(str).unique().tolist()
        vals = sorted(vals)
        # 인덱스 규칙:
        #   0 = PAD (padding only)
        #   1 = UNK (OOV/미등록 값)
        #   2.. = 실제 카테고리 값
        return {v: i + 2 for i, v in enumerate(vals)}

    vocabs = {
        "player_id": make_map(df["player_id"]),
        "team_id": make_map(df["team_id"]),
        "type_name": make_map(df["type_name"]),
        "result_name": make_map(df["result_name"]),
    }
    sizes = {k: (len(v) + 2) for k, v in vocabs.items()}  # +2 for PAD(0), UNK(1)
    return vocabs, sizes


def build_train_episodes(
    train_csv_path: str,
    field_x: float,
    field_y: float,
    vocabs: dict,
    max_tail_k: int = 0,
    target_policy: str = "all_pass",
    dt_clip_sec: float = 60.0,
    dt_norm_ref_sec: float = 60.0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int], List[str]]:
    """
    train.csv -> 에피소드 단위로 시퀀스 생성

    입력 (row 토큰):
      num: [sx, sy, ex_filled, ey_filled, end_mask, dx, dy, dist, angle_sin, angle_cos, dt]  (정규화/스케일)
      cat: [player_id, team_id, type_name, result_name] (vocab index; PAD/UNK=0)

    중요한 점:
      - train도 test와 입력 조건을 맞추기 위해 "마지막 row의 end는 없는 것"으로 처리(end_mask=0, ex/ey=0).
      - 타깃은 마지막 row의 진짜 end_x/end_y(정규화).
    """
    df = pd.read_csv(train_csv_path)

    # 정렬은 action_id 우선이 안전 (time_seconds가 노이즈인 케이스 방지)
    sort_cols = ["game_episode"] + (["action_id"] if "action_id" in df.columns else []) + ["time_seconds"]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    episodes_num: List[np.ndarray] = []
    episodes_cat: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    sample_game_ids: List[int] = []
    sample_episode_keys: List[str] = []

    def map_idx(g: pd.DataFrame, col: str, vocab: dict) -> np.ndarray:
        vals = g[col].fillna("None").astype(str).values
        return np.asarray([vocab.get(v, 1) for v in vals], dtype=np.int64)  # default UNK=1

    desc = f"Build samples (train: {target_policy})"
    for game_episode, g in tqdm(df.groupby("game_episode"), desc=desc):
        if max_tail_k and max_tail_k > 0:
            g = g.tail(int(max_tail_k)).reset_index(drop=True)

        # 타깃 후보: Pass 계열(문자열 prefix가 "Pass")
        type_vals = g["type_name"].fillna("None").astype(str).values
        # pass_indices = np.where(np.char.startswith(type_vals.astype(str), "Pass"))[0] # type: ignore
        pass_indices = np.where(type_vals.astype(str) == "Pass")[0]  # pass만 쓰려면 이거 활성화
        if len(pass_indices) == 0:
            continue

        tp = str(target_policy).strip().lower()
        if tp not in ("all_pass", "last_pass"):
            raise ValueError(f"Invalid target_policy={target_policy!r}. Use 'all_pass' or 'last_pass'.")
        if tp == "last_pass":
            # 가능한 한 test와 맞추기: 에피소드 마지막 action이 Pass면 그 1개만 사용
            if len(type_vals) > 0 and str(type_vals[-1]) == "Pass":
                pass_indices = np.array([len(type_vals) - 1], dtype=int)
            else:
                # 마지막 action이 Pass가 아니면, 마지막 Pass 1개만 사용(백업)
                pass_indices = np.array([int(pass_indices[-1])], dtype=int)
        game_id = int(str(game_episode).split("_", 1)[0])
        episode_key = str(game_episode)

        # 같은 episode 안에서는 ref_team이 최대 2개뿐이라, team별 unify 결과를 캐시해서 재사용
        team_vals = g["team_id"].astype(int).values
        uniq_teams = list(dict.fromkeys(team_vals.tolist()))  # stable unique
        unified_cache: dict[int, pd.DataFrame] = {}
        for t_id in uniq_teams:
            unified_cache[int(t_id)] = unify_frame_to_ref_team(
                g, ref_team_id=int(t_id), field_x=field_x, field_y=field_y
            )

        for t_idx in pass_indices:
            ref_team = int(team_vals[t_idx])
            g_ref = unified_cache[ref_team].iloc[: t_idx + 1].reset_index(drop=True)

            # --- absolute normalized coords (0~1) ---
            sx_abs = (g_ref["start_x"].values / field_x).astype(np.float32)  # type: ignore
            sy_abs = (g_ref["start_y"].values / field_y).astype(np.float32)  # type: ignore

            # raw end (may be NaN)
            ex_raw = g_ref["end_x"].values.astype(np.float32)  # type: ignore
            ey_raw = g_ref["end_y"].values.astype(np.float32)  # type: ignore

            # anchor는 "타깃 Pass의 start"
            anchor_x = float(sx_abs[-1])
            anchor_y = float(sy_abs[-1])

            # 타깃: 마지막 row(=t_idx Pass)의 "실제 end"를 anchor 기준 델타로 예측
            # (입력에서 end를 가리더라도 y는 실제 end를 써야 함)
            if np.isnan(ex_raw[-1]) or np.isnan(ey_raw[-1]):
                # 드물지만 end가 없으면 학습 불가 -> 스킵
                continue
            tgt = np.asarray(
                [float(ex_raw[-1] / field_x) - anchor_x, float(ey_raw[-1] / field_y) - anchor_y],
                dtype=np.float32,
            )
            targets.append(tgt)
            sample_game_ids.append(game_id)
            sample_episode_keys.append(episode_key)

            T = len(g_ref)

            # end_mask: test와 동일 규칙 (NaN end OR NO_END_TYPES => 0), 단 타깃 row는 항상 0
            type_names = g_ref["type_name"].values
            no_end = np.isin(type_names, list(NO_END_TYPES))  # type: ignore
            end_ok = (~np.isnan(ex_raw)) & (~np.isnan(ey_raw))
            end_mask = (end_ok & (~no_end)).astype(np.float32)
            end_mask[-1] = 0.0

            # absolute end filled: end가 없으면 start로 채움 (test와 동일)
            ex_abs = (np.nan_to_num(ex_raw, nan=0.0) / field_x).astype(np.float32)  # type: ignore
            ey_abs = (np.nan_to_num(ey_raw, nan=0.0) / field_y).astype(np.float32)  # type: ignore
            ex_abs_filled = np.where(end_mask > 0, ex_abs, sx_abs).astype(np.float32)
            ey_abs_filled = np.where(end_mask > 0, ey_abs, sy_abs).astype(np.float32)

            # --- anchor-relative coords (normalized) ---
            sx_rel = (sx_abs - anchor_x).astype(np.float32)
            sy_rel = (sy_abs - anchor_y).astype(np.float32)
            ex_rel = (ex_abs_filled - anchor_x).astype(np.float32)
            ey_rel = (ey_abs_filled - anchor_y).astype(np.float32)

            # relative end filled: end가 없으면 0 (relative frame) (test와 동일)
            ex_rel_filled = np.where(end_mask > 0, ex_rel, 0.0).astype(np.float32)
            ey_rel_filled = np.where(end_mask > 0, ey_rel, 0.0).astype(np.float32)

            dx = np.where(end_mask > 0, ex_rel - sx_rel, 0.0).astype(np.float32)
            dy = np.where(end_mask > 0, ey_rel - sy_rel, 0.0).astype(np.float32)

            dx_m = (dx * field_x).astype(np.float32)
            dy_m = (dy * field_y).astype(np.float32)
            dist_m = np.sqrt(dx_m * dx_m + dy_m * dy_m).astype(np.float32)
            # dist는 입력 스케일 안정화를 위해 0~1 근처로 스케일(대각선 길이로 나눔)
            diag = float(np.sqrt(field_x * field_x + field_y * field_y))
            dist = (dist_m / diag).astype(np.float32)

            # 방향 유효성은 dist_m 기준이 더 자연스러움
            ang = np.arctan2(dy_m, dx_m).astype(np.float32)
            eps_m = 1e-3
            valid_dir = (end_mask > 0) & (dist_m > eps_m)
            angle_sin = np.where(valid_dir, np.sin(ang), 0.0).astype(np.float32)
            angle_cos = np.where(valid_dir, np.cos(ang), 0.0).astype(np.float32)

            t = g_ref["time_seconds"].values.astype(np.float32)
            dt = np.diff(t, prepend=t[0]) # type: ignore
            dt = np.clip(dt, 0.0, dt_clip_sec).astype(np.float32)
            # 0~1 근처 스케일로: 대략 60초를 1 근처로
            dt = (np.log1p(dt) / np.log1p(dt_norm_ref_sec)).astype(np.float32)

            # ---- absolute time / index features ----
            t_abs = g_ref["time_seconds"].values.astype(np.float32)
            t_abs = np.clip(t_abs, 0.0, 4000.0).astype(np.float32) # type: ignore
            t_abs_log = (np.log1p(t_abs) / np.log1p(3600.0)).astype(np.float32)

            if "period_id" in g_ref.columns:
                is_second_half = (g_ref["period_id"].values.astype(np.int64) == 2).astype(np.float32) # type: ignore
            else:
                is_second_half = np.zeros((T,), dtype=np.float32)

            if T > 1:
                idx = np.arange(T, dtype=np.float32)
                denom = float(T - 1)
                idx_norm = (idx / denom).astype(np.float32)
                steps_to_target = ((denom - idx) / denom).astype(np.float32)
            else:
                idx_norm = np.zeros((T,), dtype=np.float32)
                steps_to_target = np.zeros((T,), dtype=np.float32)

            # anchor absolute position (normalized) as additional numeric features
            anchor_x_feat = np.full((T,), anchor_x, dtype=np.float32)
            anchor_y_feat = np.full((T,), anchor_y, dtype=np.float32)

            num = np.stack(
                [
                    sx_abs,
                    sy_abs,
                    ex_abs_filled,
                    ey_abs_filled,
                    sx_rel,
                    sy_rel,
                    ex_rel_filled,
                    ey_rel_filled,
                    end_mask,
                    dx,
                    dy,
                    dist,
                    angle_sin,
                    angle_cos,
                    dt,
                    t_abs_log,
                    is_second_half,
                    idx_norm,
                    steps_to_target,
                    anchor_x_feat,
                    anchor_y_feat,
                ],
                axis=1,
            ).astype(np.float32)  # [T, 17]

            cat = np.stack(
                [
                    map_idx(g_ref, "player_id", vocabs["player_id"]),
                    map_idx(g_ref, "team_id", vocabs["team_id"]),
                    map_idx(g_ref, "type_name", vocabs["type_name"]),
                    map_idx(g_ref, "result_name", vocabs["result_name"]),
                ],
                axis=1,
            ).astype(np.int64)  # [T, 4]

            episodes_num.append(num)
            episodes_cat.append(cat)

    return episodes_num, episodes_cat, targets, sample_game_ids, sample_episode_keys


def build_test_sequence_from_path(
    episode_csv_path: str,
    field_x: float,
    field_y: float,
    vocabs: dict,
    max_tail_k: int = 0,
    dt_clip_sec: float = 60.0,
    dt_norm_ref_sec: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    test의 개별 에피소드 csv -> (num_seq, cat_seq)

    - end_x/end_y가 NaN이면 end_mask=0, ex/ey는 start로 채움(상대좌표계에선 0).
    """
    g = pd.read_csv(episode_csv_path)
    sort_cols = (["action_id"] if "action_id" in g.columns else []) + ["time_seconds"]
    g = g.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    if max_tail_k and max_tail_k > 0:
        g = g.tail(int(max_tail_k)).reset_index(drop=True)

    ref_team = int(g["team_id"].iloc[-1])
    g = unify_frame_to_ref_team(g, ref_team_id=ref_team, field_x=field_x, field_y=field_y)

    T = len(g)

        # --- absolute normalized coords (0~1) ---
    sx_abs = (g["start_x"].values / field_x).astype(np.float32)  # type: ignore
    sy_abs = (g["start_y"].values / field_y).astype(np.float32)  # type: ignore

    type_names = g["type_name"].values
    no_end = np.isin(type_names, list(NO_END_TYPES)) # type: ignore

    ex_raw = g["end_x"].values.astype(np.float32)
    ey_raw = g["end_y"].values.astype(np.float32)

    end_ok = (~np.isnan(ex_raw)) & (~np.isnan(ey_raw))

    # 핵심: NaN이 아니어도 NO_END_TYPES면 end 없다고 처리
    end_mask = (end_ok & (~no_end)).astype(np.float32)

    ex_abs = (np.nan_to_num(ex_raw, nan=0.0) / field_x).astype(np.float32) # type: ignore
    ey_abs = (np.nan_to_num(ey_raw, nan=0.0) / field_y).astype(np.float32) # type: ignore
    # NaN end는 start로 채움 (absolute frame)
    ex_abs = np.where(end_mask > 0, ex_abs, sx_abs).astype(np.float32)
    ey_abs = np.where(end_mask > 0, ey_abs, sy_abs).astype(np.float32)

    # absolute end filled (for input)
    ex_abs_filled = ex_abs
    ey_abs_filled = ey_abs

    # Anchor = 마지막 action의 start(정규화)
    anchor_x = float(sx_abs[-1])
    anchor_y = float(sy_abs[-1])

    # --- anchor-relative coords (delta frame) ---
    sx_rel = (sx_abs - anchor_x).astype(np.float32)
    sy_rel = (sy_abs - anchor_y).astype(np.float32)
    ex_rel = (ex_abs_filled - anchor_x).astype(np.float32)
    ey_rel = (ey_abs_filled - anchor_y).astype(np.float32)

    # relative end filled: end가 없으면 0 (relative frame)
    ex_rel_filled = np.where(end_mask > 0, ex_rel, 0.0).astype(np.float32)
    ey_rel_filled = np.where(end_mask > 0, ey_rel, 0.0).astype(np.float32)

    dx = np.where(end_mask > 0, ex_rel - sx_rel, 0.0).astype(np.float32)
    dy = np.where(end_mask > 0, ey_rel - sy_rel, 0.0).astype(np.float32)
    dx_m = (dx * field_x).astype(np.float32)
    dy_m = (dy * field_y).astype(np.float32)
    dist_m = np.sqrt(dx_m * dx_m + dy_m * dy_m).astype(np.float32)
    diag = float(np.sqrt(field_x * field_x + field_y * field_y))
    dist = (dist_m / diag).astype(np.float32)

    ang = np.arctan2(dy_m, dx_m).astype(np.float32)
    eps_m = 1e-3
    valid_dir = (end_mask > 0) & (dist_m > eps_m)
    angle_sin = np.where(valid_dir, np.sin(ang), 0.0).astype(np.float32)
    angle_cos = np.where(valid_dir, np.cos(ang), 0.0).astype(np.float32)

    t = g["time_seconds"].values.astype(np.float32)
    dt = np.diff(t, prepend=t[0]) # type: ignore
    dt = np.clip(dt, 0.0, dt_clip_sec).astype(np.float32)
    dt = (np.log1p(dt) / np.log1p(dt_norm_ref_sec)).astype(np.float32)

    # ---- absolute time / index features ----
    t_abs = g["time_seconds"].values.astype(np.float32)
    t_abs = np.clip(t_abs, 0.0, 4000.0).astype(np.float32) # type: ignore
    t_abs_log = (np.log1p(t_abs) / np.log1p(3600.0)).astype(np.float32)

    if "period_id" in g.columns:
        is_second_half = (g["period_id"].values.astype(np.int64) == 2).astype(np.float32) # type: ignore
    else:
        is_second_half = np.zeros((T,), dtype=np.float32)

    if T > 1:
        idx = np.arange(T, dtype=np.float32)
        denom = float(T - 1)
        idx_norm = (idx / denom).astype(np.float32)
        steps_to_target = ((denom - idx) / denom).astype(np.float32)
    else:
        idx_norm = np.zeros((T,), dtype=np.float32)
        steps_to_target = np.zeros((T,), dtype=np.float32)

    # anchor absolute position (normalized) as additional numeric features
    anchor_x_feat = np.full((T,), anchor_x, dtype=np.float32)
    anchor_y_feat = np.full((T,), anchor_y, dtype=np.float32)

    num = np.stack(
        [
            sx_abs,
            sy_abs,
            ex_abs_filled,
            ey_abs_filled,
            sx_rel,
            sy_rel,
            ex_rel_filled,
            ey_rel_filled,
            end_mask,
            dx,
            dy,
            dist,
            angle_sin,
            angle_cos,
            dt,
            t_abs_log,
            is_second_half,
            idx_norm,
            steps_to_target,
            anchor_x_feat,
            anchor_y_feat,
        ],
        axis=1,
    ).astype(np.float32)  # [T, 17]

    def map_idx(col: str, vocab: dict) -> np.ndarray:
        vals = g[col].fillna("None").astype(str).values
        return np.asarray([vocab.get(v, 1) for v in vals], dtype=np.int64)  # default UNK=1

    cat = np.stack(
        [
            map_idx("player_id", vocabs["player_id"]),
            map_idx("team_id", vocabs["team_id"]),
            map_idx("type_name", vocabs["type_name"]),
            map_idx("result_name", vocabs["result_name"]),
        ],
        axis=1,
    ).astype(np.int64)  # [T, 4]

    anchor = np.asarray([anchor_x, anchor_y], dtype=np.float32)
    return num, cat, anchor


class EpisodeDataset(Dataset):
    def __init__(self, episodes_num: List[np.ndarray], episodes_cat: List[np.ndarray], targets: List[np.ndarray]):
        self.episodes_num = episodes_num
        self.episodes_cat = episodes_cat
        self.targets = targets

    def __len__(self) -> int:
        return len(self.episodes_num)

    def __getitem__(self, idx: int):
        num = torch.tensor(self.episodes_num[idx], dtype=torch.float32)  # [T, F]
        cat = torch.tensor(self.episodes_cat[idx], dtype=torch.long)     # [T, 4]
        tgt = torch.tensor(self.targets[idx], dtype=torch.float32)       # [2]
        length = num.size(0)
        return num, cat, length, tgt

def collate_fn(batch):
    nums, cats, lengths, tgts = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)

    padded_num = pad_sequence(nums, batch_first=True)  # type: ignore # [B, T, F]
    padded_cat = pad_sequence(cats, batch_first=True)  # type: ignore # [B, T, 4] (padding=0)

    tgts = torch.stack(tgts, dim=0)  # [B, 2]
    return padded_num, padded_cat, lengths, tgts


# -------------------------
# Model
# -------------------------
class LSTMWithEmb(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        vocab_sizes: dict,
        emb_dims: dict,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        emb_dropout: float = 0.0,
    ):
        super().__init__()

        self.cat_order = ["player_id", "team_id", "type_name", "result_name"]

        self.embs = nn.ModuleDict({
            name: nn.Embedding(vocab_sizes[name], emb_dims[name], padding_idx=0)
            for name in self.cat_order
        })
        self.emb_dropout = nn.Dropout(emb_dropout)

        in_dim = numeric_dim + sum(emb_dims[name] for name in self.cat_order)

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 2)
        self.bidirectional = bidirectional

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, lengths: torch.Tensor):
        # x_num: [B,T,F], x_cat: [B,T,4]
        emb_list = []
        for i, name in enumerate(self.cat_order):
            emb = self.embs[name](x_cat[:, :, i])  # [B,T,E]
            emb_list.append(emb)

        x = torch.cat([x_num] + emb_list, dim=-1)
        x = self.emb_dropout(x)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)

        if self.bidirectional:
            # 마지막 layer의 forward/backward concat
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, 2H]
        else:
            h_last = h_n[-1]  # [B, H]

        out = torch.tanh(self.fc(h_last))  # [-1, 1] delta in normalized coord
        return out


# -------------------------
# Train/Eval
# -------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    field_x: float,
    field_y: float,
) -> float:
    model.eval()
    total = 0.0
    n = 0

    for X_num, X_cat, lengths, y in tqdm(loader, desc="Valid", leave=False):
        X_num, X_cat, lengths, y = X_num.to(device), X_cat.to(device), lengths.to(device), y.to(device)
        pred = model(X_num, X_cat, lengths)
        dist = euclidean_loss_meters(pred, y, field_x=field_x, field_y=field_y)
        bs = X_num.size(0)
        total += float(dist.item()) * bs
        n += bs

    return total / max(n, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    field_x: float,
    field_y: float,
    grad_clip: float = 0.0,
    amp: bool = False,
    log_every_steps: int = 50,
    wandb_run=None,
    epoch: int = 0,
) -> float:
    model.train()
    total = 0.0
    n = 0

    use_amp = bool(amp) and str(device).startswith("cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    for step, (X_num, X_cat, lengths, y) in enumerate(tqdm(loader, desc="Train", leave=False), start=1):
        X_num, X_cat, lengths, y = X_num.to(device), X_cat.to(device), lengths.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            pred = model(X_num, X_cat, lengths)
            loss = euclidean_loss_meters(pred, y, field_x=field_x, field_y=field_y)

        scaler.scale(loss).backward()

        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        bs = X_num.size(0)
        total += float(loss.item()) * bs
        n += bs

        if wandb_run is not None and (step % max(log_every_steps, 1) == 0):
            wandb_run.log(
                {"train/loss_m": float(loss.item()), "epoch": epoch, "step_in_epoch": step},
            )

    return total / max(n, 1)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    cfg_dict: dict,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "cfg": cfg_dict,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


# -------------------------
# Main (Hydra)
# -------------------------
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Hydra run dir 안에 현재 실험 config 저장 (재현성)
    OmegaConf.save(cfg, "hydra_config_resolved.yaml")

    if cfg.train.seed is not None:
        set_seed(int(cfg.train.seed))

    device = resolve_device(str(cfg.train.device))
    print("Using device:", device)

    # (Hydra가 cwd를 바꿀 수 있으니) 데이터 경로는 절대경로로 변환
    train_path = to_absolute_path(str(cfg.data.train_path))
    test_meta_path = to_absolute_path(str(cfg.data.test_meta_path))
    sample_sub_path = to_absolute_path(str(cfg.data.sample_submission_path))

    field_x = float(cfg.data.field_x)
    field_y = float(cfg.data.field_y)

    # dt is computed as diff(time_seconds). Clamp long gaps (e.g., stoppages) for stability.
    # If config does not define dt_clip_sec, default to 60 seconds (keeps the original dt scaling intent).
    dt_clip_sec = float(getattr(cfg.data, "dt_clip_sec", 60.0))

    # dt scaling reference (seconds). Controls dt normalization: log1p(dt)/log1p(dt_norm_ref_sec)
    dt_norm_ref_sec = float(getattr(cfg.data, "dt_norm_ref_sec", 60.0))
    if not np.isfinite(dt_norm_ref_sec) or dt_norm_ref_sec <= 0:
        raise ValueError(f"data.dt_norm_ref_sec must be a positive finite number, got {dt_norm_ref_sec!r}")

    # -----------------
    # W&B init
    # -----------------
    wandb_run = None
    if bool(cfg.wandb.enabled):
        import wandb

        # wandb.mode가 disabled면 init 자체를 안 하는 편이 더 안전하지만,
        # config에서 enabled=true 일 때만 init하도록 구성
        wandb_run = wandb.init(
            project=str(cfg.wandb.project),
            entity=None if cfg.wandb.entity in (None, "null") else str(cfg.wandb.entity),
            name=None if cfg.wandb.name in (None, "null") else str(cfg.wandb.name),
            tags=list(cfg.wandb.tags) if cfg.wandb.tags is not None else None,
            notes=None if cfg.wandb.notes in (None, "null") else str(cfg.wandb.notes),
            mode=str(cfg.wandb.mode), # type: ignore
            config=OmegaConf.to_container(cfg, resolve=True), # type: ignore
        )

    # -----------------
    # Build dataset
    # -----------------
    vocabs, vocab_sizes = build_vocabs(train_path)

    # -----------------
    # Stage configuration
    # -----------------
    target_policy = str(getattr(cfg.data, "target_policy", "all_pass")).strip()
    resume_path = str(getattr(cfg.train, "resume_path", "") or "").strip()

    two_stage = bool(getattr(cfg.train, "two_stage", False))
    stage2_cfg = cfg.train.stage2 if ("stage2" in cfg.train) else None

    # stage2에서 data.max_tail_k를 덮어쓰고 싶으면 train.stage2.max_tail_k를 사용
    def get_stage_param(stage_cfg: Any, key: str, fallback: Any) -> Any:
        try:
            if stage_cfg is not None and (key in stage_cfg) and stage_cfg[key] is not None:
                return stage_cfg[key]
        except Exception:
            pass
        return fallback

    pretrain_ckpt_name = str(getattr(cfg.output, "pretrain_ckpt_name", "pretrain.best.ckpt.pt"))
    finetune_ckpt_name = str(getattr(cfg.output, "ckpt_name", "best.ckpt.pt"))

    # 실행할 stage 목록 구성
    # - two_stage=false: data.target_policy로 1-stage 학습
    # - two_stage=true & resume_path 비어있음: pretrain(all_pass) -> finetune(last_pass)
    # - two_stage=true & resume_path 있음: pretrain을 건너뛰고 해당 ckpt로 finetune만 수행
    stages: List[Tuple[str, str, Any, str]] = []
    if two_stage:
        if resume_path:
            stages = [("finetune", "last_pass", stage2_cfg, finetune_ckpt_name)]
            print(f"[2-stage] resume_path is set -> skip pretrain, finetune from: {resume_path}")
        else:
            stages = [
                ("pretrain", "all_pass", cfg.train, pretrain_ckpt_name),
                ("finetune", "last_pass", stage2_cfg, finetune_ckpt_name),
            ]
    else:
        stages = [("train", target_policy, cfg.train, finetune_ckpt_name)]

    model: Optional[nn.Module] = None
    final_best_ckpt_path: Optional[Path] = None

    # split을 stage 간에 공유(게임 누수 방지 & 비교 공정성)
    train_games: Optional[set] = None
    valid_games: Optional[set] = None

    epoch_offset = 0

    for stage_idx, (stage_name, stage_policy, stage_cfg, stage_ckpt_name) in enumerate(stages, start=1):
        print("\n" + "=" * 80)
        print(f"[Stage {stage_idx}/{len(stages)}] {stage_name} | target_policy={stage_policy}")
        print("=" * 80)

        # stage별 hyperparams (없는 값은 train 기본값으로 fallback)
        stage_epochs = int(get_stage_param(stage_cfg, "epochs", cfg.train.epochs))
        stage_batch_size = int(get_stage_param(stage_cfg, "batch_size", cfg.train.batch_size))
        stage_lr = float(get_stage_param(stage_cfg, "lr", cfg.train.lr))
        stage_weight_decay = float(get_stage_param(stage_cfg, "weight_decay", cfg.train.weight_decay))
        stage_grad_clip = float(get_stage_param(stage_cfg, "grad_clip", cfg.train.grad_clip))
        stage_amp = bool(get_stage_param(stage_cfg, "amp", cfg.train.amp))
        stage_num_workers = int(get_stage_param(stage_cfg, "num_workers", cfg.train.num_workers))
        stage_pin_memory = bool(get_stage_param(stage_cfg, "pin_memory", cfg.train.pin_memory))
        stage_log_every = int(get_stage_param(stage_cfg, "log_every_steps", cfg.train.log_every_steps))

        stage_max_tail_k = int(get_stage_param(stage_cfg, "max_tail_k", getattr(cfg.data, "max_tail_k", 0)))

        print(
            f"stage_epochs={stage_epochs} batch_size={stage_batch_size} lr={stage_lr} "
            f"max_tail_k={stage_max_tail_k} amp={stage_amp}"
        )

        # -----------------
        # Build episodes for this stage
        # -----------------
        episodes_num, episodes_cat, targets, episode_game_ids, episode_keys = build_train_episodes(
            train_csv_path=train_path,
            field_x=field_x,
            field_y=field_y,
            vocabs=vocabs,
            max_tail_k=stage_max_tail_k,
            target_policy=stage_policy,
            dt_clip_sec=dt_clip_sec,
            dt_norm_ref_sec=dt_norm_ref_sec,
        )

        assert len(episodes_num) > 0, "No training episodes were built. Check your data & target_policy."
        assert all(np.isfinite(x).all() for x in episodes_num)
        assert np.isfinite(np.vstack(targets)).all()
        print("OK: no NaN/inf in train sequences & targets")
        print("에피소드 수:", len(episodes_num))

        # -----------------
        # Split by game_id (group split)
        # -----------------
        if train_games is None or valid_games is None:
            seed = int(cfg.train.seed) if cfg.train.seed is not None else 42
            gss = GroupShuffleSplit(n_splits=1, test_size=float(cfg.train.valid_ratio), random_state=seed)
            idx_train, idx_valid = next(gss.split(np.arange(len(episodes_num)), groups=episode_game_ids))
            train_games = set(np.array(episode_game_ids)[idx_train])
            valid_games = set(np.array(episode_game_ids)[idx_valid])
            print("overlap games:", len(train_games & valid_games))  # 반드시 0이어야 함
            print("train games:", len(train_games), "valid games:", len(valid_games))
        else:
            # stage2: stage1에서 뽑은 game split을 그대로 사용
            idx_train = np.array([i for i, gid in enumerate(episode_game_ids) if gid in train_games], dtype=int)
            idx_valid = np.array([i for i, gid in enumerate(episode_game_ids) if gid in valid_games], dtype=int)
            print("[reuse split] train episodes:", len(idx_train), "valid episodes:", len(idx_valid))
            print("overlap games:", len(train_games & valid_games))

        episodes_num_train = [episodes_num[i] for i in idx_train]
        episodes_cat_train = [episodes_cat[i] for i in idx_train]
        targets_train = [targets[i] for i in idx_train]

        episodes_num_valid = [episodes_num[i] for i in idx_valid]
        episodes_cat_valid = [episodes_cat[i] for i in idx_valid]
        targets_valid = [targets[i] for i in idx_valid]
        print("train episodes:", len(episodes_num_train), "valid episodes:", len(episodes_num_valid))

        # -----------------
        # Sanity check: last_pass should be one sample per game_episode (no duplicates)
        # -----------------
        if str(stage_policy).strip().lower() == "last_pass":
            total_all = len(episode_keys)
            uniq_all = len(set(episode_keys))
            dup_all = total_all - uniq_all
            print(f"[Check] last_pass overall: samples={total_all} unique_game_episode={uniq_all} dup={dup_all}")

            train_keys = [episode_keys[i] for i in idx_train]
            valid_keys = [episode_keys[i] for i in idx_valid]
            total_tr, uniq_tr = len(train_keys), len(set(train_keys))
            total_va, uniq_va = len(valid_keys), len(set(valid_keys))
            print(f"[Check] last_pass train subset: samples={total_tr} unique_game_episode={uniq_tr} dup={total_tr-uniq_tr}")
            print(f"[Check] last_pass valid subset: samples={total_va} unique_game_episode={uniq_va} dup={total_va-uniq_va}")

            if total_va != uniq_va:
                ctr = Counter(valid_keys)
                dups = [k for k, c in ctr.items() if c > 1]
                examples = dups[:10]
                print(f"[Check][WARN] Duplicated game_episode keys in VALID (n={len(dups)}). Examples: {examples}")

            # Stage2(=finetune)에서는 엄격하게 1-episode-1-sample을 보장하도록 assert
            if stage_name == "finetune":
                assert total_va == uniq_va, (
                    f"Stage2(valid) is not one-sample-per-episode: samples={total_va}, unique_game_episode={uniq_va}. "
                    f"Check target_policy=last_pass generation / split mapping."
                )
                assert total_tr == uniq_tr, (
                    f"Stage2(train) is not one-sample-per-episode: samples={total_tr}, unique_game_episode={uniq_tr}. "
                    f"Check target_policy=last_pass generation / split mapping."
                )

        train_loader = DataLoader(
            EpisodeDataset(episodes_num_train, episodes_cat_train, targets_train),
            batch_size=stage_batch_size,
            shuffle=True,
            num_workers=stage_num_workers,
            pin_memory=stage_pin_memory,
            collate_fn=collate_fn,
        )
        valid_loader = DataLoader(
            EpisodeDataset(episodes_num_valid, episodes_cat_valid, targets_valid),
            batch_size=stage_batch_size,
            shuffle=False,
            num_workers=stage_num_workers,
            pin_memory=stage_pin_memory,
            collate_fn=collate_fn,
        )

        # -----------------
        # Build / Load model (once)
        # -----------------
        if model is None:
            numeric_dim = int(episodes_num_train[0].shape[1])
            if "emb_dims" in cfg.model:
                emb_dims = dict(cfg.model.emb_dims)  # type: ignore
            else:
                emb_dims = {"player_id": 16, "team_id": 8, "type_name": 8, "result_name": 4}

            model = LSTMWithEmb(
                numeric_dim=numeric_dim,
                vocab_sizes=vocab_sizes,
                emb_dims=emb_dims,  # type: ignore
                hidden_dim=int(cfg.model.hidden_dim),
                num_layers=int(cfg.model.num_layers),
                dropout=float(cfg.model.dropout),
                bidirectional=bool(cfg.model.bidirectional),
                emb_dropout=float(getattr(cfg.model, "emb_dropout", 0.0)),
            ).to(device)

            if wandb_run is not None:
                n_params = sum(p.numel() for p in model.parameters())
                wandb_run.log({"model/n_params": n_params})

        assert model is not None

        # stage 시작 전: resume_path가 있으면 가중치 로드(=프리트레인 결과로 초기화 or 학습 재개)
        if resume_path and stage_name != "pretrain":
            ckpt_init = load_checkpoint(resume_path, model=model, optimizer=None)
            print(
                f"[init] Loaded weights from: {resume_path} "
                f"(epoch={ckpt_init.get('epoch')} best_metric={ckpt_init.get('best_metric')})"
            )
            # resume_path는 한 번만 쓰도록 비워둠(연속 stage에서 중복 로드 방지)
            resume_path = ""

        # stage별 optimizer는 새로 생성(파인튜닝 시 optimizer reset 권장)
        opt_name = str(get_stage_param(stage_cfg, "optimizer", getattr(cfg.train, "optimizer", "adamw"))).lower()
        if opt_name in ("adamw", "adam_w"):
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=stage_lr,
                weight_decay=stage_weight_decay,
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=stage_lr,
                weight_decay=stage_weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name!r}. Use 'adamw' or 'adam'.")

        # -----------------
        # Train loop (this stage)
        # -----------------
        best_dist = float("inf")
        best_epoch = -1
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_ckpt_path = ckpt_dir / str(stage_ckpt_name)
        stem, suffix = best_ckpt_path.stem, best_ckpt_path.suffix

        for local_epoch in range(1, stage_epochs + 1):
            global_epoch = epoch_offset + local_epoch

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                field_x=field_x,
                field_y=field_y,
                grad_clip=stage_grad_clip,
                amp=stage_amp,
                log_every_steps=stage_log_every,
                wandb_run=wandb_run,
                epoch=global_epoch,
            )

            valid_mean_dist = evaluate(
                model=model,
                loader=valid_loader,
                device=device,
                field_x=field_x,
                field_y=field_y,
            )

            print(
                f"[{stage_name}] epoch {local_epoch}/{stage_epochs} (global {global_epoch}) | "
                f"train_loss={train_loss:.4f} valid_mean_dist={valid_mean_dist:.4f}"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "stage/idx": stage_idx,
                        "stage/name": stage_name,
                        "stage/policy": stage_policy,
                        "epoch": global_epoch,
                        f"{stage_name}/train/epoch_loss_m": float(train_loss),
                        f"{stage_name}/valid/mean_dist_m": float(valid_mean_dist),
                    }
                )

            if valid_mean_dist < best_dist:
                dist = float(valid_mean_dist)
                best_dist = dist
                best_epoch = local_epoch

                ckpt_epoch_path = ckpt_dir / f"{stem}_{stage_name}_epoch{global_epoch:03d}_dist{dist:.4f}{suffix}"

                save_checkpoint(
                    path=str(ckpt_epoch_path),
                    model=model,
                    optimizer=optimizer,
                    epoch=global_epoch,
                    best_metric=dist,
                    cfg_dict=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
                )

                shutil.copy2(str(ckpt_epoch_path), str(best_ckpt_path))
                print(f" --> [{stage_name}] Best model saved! global_epoch={global_epoch}, dist={dist:.4f}")

        # stage 종료 후 best 로드(다음 stage의 시작점)
        ckpt_best = load_checkpoint(str(best_ckpt_path), model=model, optimizer=None)
        model.eval()
        print(
            f"[{stage_name}] Loaded best checkpoint: epoch={ckpt_best.get('epoch')} best_metric={ckpt_best.get('best_metric')}"
        )

        final_best_ckpt_path = best_ckpt_path
        epoch_offset += stage_epochs

    assert final_best_ckpt_path is not None, "Training finished without a checkpoint."

    # 최종 stage(best)를 inference에 사용
    best_ckpt_path = final_best_ckpt_path
# -----------------
    # Load best + inference
    # -----------------
    ckpt = load_checkpoint(str(best_ckpt_path), model=model, optimizer=None) # type: ignore

    model.eval() # type: ignore
    print(f"Loaded best checkpoint: epoch={ckpt.get('epoch')} best_metric={ckpt.get('best_metric')}")

    # test_meta + sample_submission merge 방식 유지 fileciteturn2file0L204-L207
    test_meta = pd.read_csv(test_meta_path)
    submission = pd.read_csv(sample_sub_path)
    submission = submission.merge(test_meta, on="game_episode", how="left")

    preds_x: List[float] = []
    preds_y: List[float] = []

    for _, row in tqdm(submission.iterrows(), total=len(submission), desc="Inference"):
        episode_path = to_absolute_path(str(row["path"]))  # test.csv의 path 컬럼
        num, cat, anchor = build_test_sequence_from_path(
            episode_path, field_x=field_x, field_y=field_y, vocabs=vocabs, max_tail_k=stage_max_tail_k, dt_clip_sec=dt_clip_sec, dt_norm_ref_sec=dt_norm_ref_sec,
        )

        x_num = torch.tensor(num, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T, F]
        x_cat = torch.tensor(cat, dtype=torch.long).unsqueeze(0).to(device)     # [1, T, 4]
        length = torch.tensor([num.shape[0]]).to(device)

        with torch.no_grad():
            pred = model(x_num, x_cat, length).detach().cpu().numpy()[0]  # type: ignore # [2]

        # pred is delta in normalized coords (anchor-relative)
        end_norm = pred + anchor  # absolute normalized end position
        end_norm = np.clip(end_norm, 0.0, 1.0)

        preds_x.append(float(end_norm[0] * field_x))
        preds_y.append(float(end_norm[1] * field_y))

    submission["end_x"] = preds_x
    submission["end_y"] = preds_y

    out_csv = os.path.abspath(str(cfg.output.submission_name))
    submission[["game_episode", "end_x", "end_y"]].to_csv(out_csv, index=False)  # fileciteturn2file0L240-L243
    print("Saved:", out_csv)

    if wandb_run is not None:
        # 1) best 체크포인트 artifact
        model_art = wandb.Artifact(f"model-{wandb_run.id}", type="model")
        model_art.add_file(str(best_ckpt_path))  # checkpoints/best.ckpt.pt
        wandb_run.log_artifact(model_art, aliases=["best"]).wait()

        # 2) 제출파일 artifact
        sub_art = wandb.Artifact(f"submission-{wandb_run.id}", type="submission")
        sub_art.add_file(str(out_csv))  # outputs/.../baseline_submit.csv
        wandb_run.log_artifact(sub_art, aliases=["latest"]).wait()

        wandb_run.finish()


if __name__ == "__main__":
    main()
