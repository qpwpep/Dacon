from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict

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
        g.loc[opp, col] = field_x - g.loc[opp, col].astype(float)  # type: ignore
    for col in ["start_y", "end_y"]:
        g.loc[opp, col] = field_y - g.loc[opp, col].astype(float)  # type: ignore

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
      num: [sx_abs,sy_abs,ex_abs_filled,ey_abs_filled,sx_rel,sy_rel,ex_rel_filled,ey_rel_filled,end_mask,
            dx,dy,dist,angle_sin,angle_cos,dt,t_abs_log,is_second_half,idx_norm,steps_to_target,anchor_x,anchor_y]
      cat: [player_id, team_id, type_name, result_name] (vocab index; PAD/UNK=0)

    중요한 점:
      - train도 test와 입력 조건을 맞추기 위해 "마지막 row의 end는 없는 것"으로 처리(end_mask=0, ex/ey=0).
      - 타깃은 마지막 row의 진짜 end_x/end_y(정규화)에서 anchor(start) 기준 delta로 예측.
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

        # 타깃 후보: Pass만 사용(기존 baseline과 동일)
        type_vals = g["type_name"].fillna("None").astype(str).values
        pass_indices = np.where(type_vals.astype(str) == "Pass")[0]
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
            if np.isnan(ex_raw[-1]) or np.isnan(ey_raw[-1]):
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
            diag = float(np.sqrt(field_x * field_x + field_y * field_y))
            dist = (dist_m / diag).astype(np.float32)

            ang = np.arctan2(dy_m, dx_m).astype(np.float32)
            eps_m = 1e-3
            valid_dir = (end_mask > 0) & (dist_m > eps_m)
            angle_sin = np.where(valid_dir, np.sin(ang), 0.0).astype(np.float32)
            angle_cos = np.where(valid_dir, np.cos(ang), 0.0).astype(np.float32)

            t = g_ref["time_seconds"].values.astype(np.float32)
            dt = np.diff(t, prepend=t[0])  # type: ignore
            dt = np.clip(dt, 0.0, dt_clip_sec).astype(np.float32)
            dt = (np.log1p(dt) / np.log1p(dt_norm_ref_sec)).astype(np.float32)

            # ---- absolute time / index features ----
            t_abs = g_ref["time_seconds"].values.astype(np.float32)
            t_abs = np.clip(t_abs, 0.0, 4000.0).astype(np.float32)  # type: ignore
            t_abs_log = (np.log1p(t_abs) / np.log1p(3600.0)).astype(np.float32)

            if "period_id" in g_ref.columns:
                is_second_half = (g_ref["period_id"].values.astype(np.int64) == 2).astype(np.float32)  # type: ignore
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
            ).astype(np.float32)

            cat = np.stack(
                [
                    map_idx(g_ref, "player_id", vocabs["player_id"]),
                    map_idx(g_ref, "team_id", vocabs["team_id"]),
                    map_idx(g_ref, "type_name", vocabs["type_name"]),
                    map_idx(g_ref, "result_name", vocabs["result_name"]),
                ],
                axis=1,
            ).astype(np.int64)

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
    test의 개별 에피소드 csv -> (num_seq, cat_seq, anchor_norm)

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

    sx_abs = (g["start_x"].values / field_x).astype(np.float32)  # type: ignore
    sy_abs = (g["start_y"].values / field_y).astype(np.float32)  # type: ignore

    type_names = g["type_name"].values
    no_end = np.isin(type_names, list(NO_END_TYPES))  # type: ignore

    ex_raw = g["end_x"].values.astype(np.float32)
    ey_raw = g["end_y"].values.astype(np.float32)

    end_ok = (~np.isnan(ex_raw)) & (~np.isnan(ey_raw))
    end_mask = (end_ok & (~no_end)).astype(np.float32)

    ex_abs = (np.nan_to_num(ex_raw, nan=0.0) / field_x).astype(np.float32)  # type: ignore
    ey_abs = (np.nan_to_num(ey_raw, nan=0.0) / field_y).astype(np.float32)  # type: ignore
    ex_abs = np.where(end_mask > 0, ex_abs, sx_abs).astype(np.float32)
    ey_abs = np.where(end_mask > 0, ey_abs, sy_abs).astype(np.float32)

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
    dt = np.diff(t, prepend=t[0])  # type: ignore
    dt = np.clip(dt, 0.0, dt_clip_sec).astype(np.float32)
    dt = (np.log1p(dt) / np.log1p(dt_norm_ref_sec)).astype(np.float32)

    t_abs = g["time_seconds"].values.astype(np.float32)
    t_abs = np.clip(t_abs, 0.0, 4000.0).astype(np.float32)  # type: ignore
    t_abs_log = (np.log1p(t_abs) / np.log1p(3600.0)).astype(np.float32)

    if "period_id" in g.columns:
        is_second_half = (g["period_id"].values.astype(np.int64) == 2).astype(np.float32)  # type: ignore
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
    ).astype(np.float32)

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
    ).astype(np.int64)

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
        cat = torch.tensor(self.episodes_cat[idx], dtype=torch.long)  # [T, 4]
        tgt = torch.tensor(self.targets[idx], dtype=torch.float32)  # [2]
        length = num.size(0)
        return num, cat, length, tgt


def collate_fn(batch):
    nums, cats, lengths, tgts = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)

    padded_num = pad_sequence(nums, batch_first=True)  # type: ignore
    padded_cat = pad_sequence(cats, batch_first=True)  # type: ignore

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

        self.embs = nn.ModuleDict(
            {name: nn.Embedding(vocab_sizes[name], emb_dims[name], padding_idx=0) for name in self.cat_order}
        )
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
        emb_list = []
        for i, name in enumerate(self.cat_order):
            emb = self.embs[name](x_cat[:, :, i])  # [B,T,E]
            emb_list.append(emb)

        x = torch.cat([x_num] + emb_list, dim=-1)
        x = self.emb_dropout(x)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)

        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, 2H]
        else:
            h_last = h_n[-1]  # [B, H]

        # delta in normalized coord (anchor-relative)
        out = torch.tanh(self.fc(h_last))  # [-1, 1]
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
            wandb_run.log({"train/loss_m": float(loss.item()), "epoch": epoch, "step_in_epoch": step})

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
# K-fold split (by game_id)
# -------------------------
def compute_kfold_game_sets(
    episode_game_ids: List[int],
    n_folds: int,
    fold_idx: int,
    seed: int,
) -> Tuple[set, set]:
    """
    Group K-fold split by game_id (누수 방지).
    - unique game_id를 seed로 셔플한 뒤 균등 분할
    - fold_idx의 game들이 valid, 나머지가 train
    """
    if n_folds <= 1:
        # fallback: 원래 baseline과 동일하게 GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        idx = np.arange(len(episode_game_ids))
        idx_train, idx_valid = next(gss.split(idx, groups=episode_game_ids))
        train_games = set(np.array(episode_game_ids)[idx_train])
        valid_games = set(np.array(episode_game_ids)[idx_valid])
        return train_games, valid_games

    games = np.array(sorted(set(map(int, episode_game_ids))), dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(games)

    folds = np.array_split(games, n_folds)
    valid_games = set(map(int, folds[fold_idx].tolist()))
    train_games = set(map(int, games.tolist())) - valid_games
    return train_games, valid_games


# -------------------------
# Prediction / Ensembling
# -------------------------
def predict_test_delta(
    model: nn.Module,
    submission_df: pd.DataFrame,
    vocabs: dict,
    device: str,
    field_x: float,
    field_y: float,
    max_tail_k: int,
    dt_clip_sec: float,
    dt_norm_ref_sec: float,
) -> pd.DataFrame:
    """
    반환 DF columns:
      game_episode, anchor_x, anchor_y, delta_x, delta_y  (모두 normalized)
    """
    model.eval()
    rows: List[Dict[str, float]] = []

    for _, row in tqdm(submission_df.iterrows(), total=len(submission_df), desc="Inference"):
        episode_path = to_absolute_path(str(row["path"]))
        num, cat, anchor = build_test_sequence_from_path(
            episode_path,
            field_x=field_x,
            field_y=field_y,
            vocabs=vocabs,
            max_tail_k=max_tail_k,
            dt_clip_sec=dt_clip_sec,
            dt_norm_ref_sec=dt_norm_ref_sec,
        )

        x_num = torch.tensor(num, dtype=torch.float32).unsqueeze(0).to(device)
        x_cat = torch.tensor(cat, dtype=torch.long).unsqueeze(0).to(device)
        length = torch.tensor([num.shape[0]]).to(device)

        with torch.no_grad():
            pred = model(x_num, x_cat, length).detach().cpu().numpy()[0]  # [2] delta in [-1,1]

        rows.append(
            {
                "game_episode": str(row["game_episode"]), # type: ignore
                "anchor_x": float(anchor[0]),
                "anchor_y": float(anchor[1]),
                "delta_x": float(pred[0]),
                "delta_y": float(pred[1]),
            }
        )

    return pd.DataFrame(rows)


def delta_ensemble(
    dfs: List[pd.DataFrame],
    field_x: float,
    field_y: float,
    delta_space: str = "atanh",
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    dfs: fold별 delta 예측 DF (game_episode, anchor_x/y, delta_x/y)
    delta_space:
      - "mean": tanh 이후 delta 공간에서 단순 평균
      - "atanh": tanh 이전(pre-tanh, logit-like) 공간에서 평균: atanh(delta) -> mean -> tanh
    """
    if len(dfs) == 0:
        raise ValueError("No fold prediction dfs provided.")

    base = dfs[0].copy()
    base = base.sort_values("game_episode").reset_index(drop=True)

    anchors = base[["anchor_x", "anchor_y"]].to_numpy(dtype=np.float32)
    deltas = []

    for i, df in enumerate(dfs):
        d = df.sort_values("game_episode").reset_index(drop=True)
        if not (d["game_episode"].values == base["game_episode"].values).all(): # type: ignore
            raise ValueError(f"Fold df[{i}] game_episode ordering mismatch.")
        # anchor는 fold마다 같아야 정상 (입력 동일)
        a = d[["anchor_x", "anchor_y"]].to_numpy(dtype=np.float32)
        if np.max(np.abs(a - anchors)) > 1e-4:
            # 매우 작은 오차는 허용(부동소수)
            raise ValueError(f"Fold df[{i}] anchor mismatch. Check preprocessing consistency.")
        deltas.append(d[["delta_x", "delta_y"]].to_numpy(dtype=np.float32))

    D = np.stack(deltas, axis=0)  # [K, N, 2]

    space = str(delta_space).strip().lower()
    if space in ("atanh", "logit", "pre_tanh", "pretanh"):
        D_clip = np.clip(D, -1.0 + eps, 1.0 - eps)
        Z = np.arctanh(D_clip)  # inverse tanh
        Zm = Z.mean(axis=0)
        Dm = np.tanh(Zm)
    elif space in ("mean", "tanh", "raw"):
        Dm = D.mean(axis=0)
    else:
        raise ValueError(f"Unknown delta_space={delta_space!r}. Use 'atanh' or 'mean'.")

    end_norm = anchors + Dm
    end_norm = np.clip(end_norm, 0.0, 1.0)

    out = pd.DataFrame(
        {
            "game_episode": base["game_episode"].values,
            "end_x": (end_norm[:, 0] * field_x).astype(np.float32),
            "end_y": (end_norm[:, 1] * field_y).astype(np.float32),
        }
    )
    return out


# -------------------------
# W&B helpers
# -------------------------
def init_wandb(cfg: DictConfig, name: Optional[str], group: Optional[str], job_type: Optional[str], extra_config: Optional[dict] = None):
    if not bool(cfg.wandb.enabled):
        return None
    import wandb

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    if extra_config:
        cfg_dict = dict(cfg_dict) # type: ignore
        cfg_dict.update(extra_config)

    run = wandb.init(
        project=str(cfg.wandb.project),
        entity=None if cfg.wandb.entity in (None, "null") else str(cfg.wandb.entity),
        name=name,
        tags=list(cfg.wandb.tags) if cfg.wandb.tags is not None else None,
        notes=None if cfg.wandb.notes in (None, "null") else str(cfg.wandb.notes),
        mode=str(cfg.wandb.mode),  # type: ignore
        group=group,
        job_type=job_type,
        config=cfg_dict, # type: ignore
    )
    return run


def log_artifact_if_possible(wandb_run, path: str, art_type: str, name: str, aliases: Optional[List[str]] = None, metadata: Optional[dict] = None):
    if wandb_run is None:
        return
    import wandb
    art = wandb.Artifact(name, type=art_type, metadata=metadata or {})
    art.add_file(path)
    wandb_run.log_artifact(art, aliases=aliases or []).wait()


# -------------------------
# Fold runner
# -------------------------
def run_one_fold(
    cfg: DictConfig,
    fold_idx: int,
    vocabs: dict,
    vocab_sizes: dict,
    base_group: Optional[str],
) -> Dict[str, str]:
    """
    Returns dict:
      - best_ckpt_path
      - fold_pred_delta_csv
      - fold_submit_csv
    """
    # fold마다 seed를 조금씩 다르게 (원하면 cfg에서 fold_seed_offset을 추가해도 됨)
    base_seed = int(cfg.train.seed) if cfg.train.seed is not None else 42
    set_seed(base_seed + int(fold_idx))

    device = resolve_device(str(cfg.train.device))
    train_path = to_absolute_path(str(cfg.data.train_path))
    test_meta_path = to_absolute_path(str(cfg.data.test_meta_path))
    sample_sub_path = to_absolute_path(str(cfg.data.sample_submission_path))

    field_x = float(cfg.data.field_x)
    field_y = float(cfg.data.field_y)

    dt_clip_sec = float(getattr(cfg.data, "dt_clip_sec", 60.0))
    dt_norm_ref_sec = float(getattr(cfg.data, "dt_norm_ref_sec", 60.0))
    if not np.isfinite(dt_norm_ref_sec) or dt_norm_ref_sec <= 0:
        raise ValueError(f"data.dt_norm_ref_sec must be positive finite, got {dt_norm_ref_sec!r}")

    # stage configuration (baseline 유지)
    target_policy = str(getattr(cfg.data, "target_policy", "all_pass")).strip()
    resume_path = str(getattr(cfg.train, "resume_path", "") or "").strip()
    two_stage = bool(getattr(cfg.train, "two_stage", False))
    stage2_cfg = cfg.train.stage2 if ("stage2" in cfg.train) else None

    def get_stage_param(stage_cfg: Any, key: str, fallback: Any) -> Any:
        try:
            if stage_cfg is not None and (key in stage_cfg) and stage_cfg[key] is not None:
                return stage_cfg[key]
        except Exception:
            pass
        return fallback

    pretrain_ckpt_name = str(getattr(cfg.output, "pretrain_ckpt_name", "pretrain.best.ckpt.pt"))
    finetune_ckpt_name = str(getattr(cfg.output, "ckpt_name", "best.ckpt.pt"))

    stages: List[Tuple[str, str, Any, str]] = []
    if two_stage:
        if resume_path:
            stages = [("finetune", "last_pass", stage2_cfg, finetune_ckpt_name)]
            print(f"[fold {fold_idx}] [2-stage] resume_path is set -> skip pretrain, finetune from: {resume_path}")
        else:
            stages = [
                ("pretrain", "all_pass", cfg.train, pretrain_ckpt_name),
                ("finetune", "last_pass", stage2_cfg, finetune_ckpt_name),
            ]
    else:
        stages = [("train", target_policy, cfg.train, finetune_ckpt_name)]

    # W&B run for this fold
    base_name = None if cfg.wandb.name in (None, "null") else str(cfg.wandb.name)
    run_name = None if base_name is None else f"{base_name}-fold{fold_idx}"
    wandb_run = init_wandb(cfg, name=run_name, group=base_group, job_type="fold", extra_config={"fold": fold_idx})

    model: Optional[nn.Module] = None
    final_best_ckpt_path: Optional[Path] = None

    train_games: Optional[set] = None
    valid_games: Optional[set] = None
    epoch_offset = 0

    n_folds = int(getattr(cfg.train, "n_folds", 1))

    for stage_idx, (stage_name, stage_policy, stage_cfg, stage_ckpt_name) in enumerate(stages, start=1):
        print("\n" + "=" * 80)
        print(f"[fold {fold_idx}] [Stage {stage_idx}/{len(stages)}] {stage_name} | target_policy={stage_policy}")
        print("=" * 80)

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

        # Build episodes for this stage
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

        assert len(episodes_num) > 0, "No training episodes were built. Check data & target_policy."
        assert all(np.isfinite(x).all() for x in episodes_num)
        assert np.isfinite(np.vstack(targets)).all()
        print(f"[fold {fold_idx}] OK: no NaN/inf in sequences & targets | episodes={len(episodes_num)}")

        # Split by game_id (K-fold)
        if train_games is None or valid_games is None:
            train_games, valid_games = compute_kfold_game_sets(
                episode_game_ids=episode_game_ids,
                n_folds=n_folds,
                fold_idx=fold_idx,
                seed=base_seed,
            )
            print(f"[fold {fold_idx}] overlap games:", len(train_games & valid_games))
            print(f"[fold {fold_idx}] train games:", len(train_games), "valid games:", len(valid_games))
        else:
            print(f"[fold {fold_idx}] [reuse split] train games:", len(train_games), "valid games:", len(valid_games))

        idx_train = np.array([i for i, gid in enumerate(episode_game_ids) if gid in train_games], dtype=int)
        idx_valid = np.array([i for i, gid in enumerate(episode_game_ids) if gid in valid_games], dtype=int)

        episodes_num_train = [episodes_num[i] for i in idx_train]
        episodes_cat_train = [episodes_cat[i] for i in idx_train]
        targets_train = [targets[i] for i in idx_train]

        episodes_num_valid = [episodes_num[i] for i in idx_valid]
        episodes_cat_valid = [episodes_cat[i] for i in idx_valid]
        targets_valid = [targets[i] for i in idx_valid]
        print(f"[fold {fold_idx}] train episodes:", len(episodes_num_train), "valid episodes:", len(episodes_num_valid))

        # Sanity check: last_pass duplicates
        if str(stage_policy).strip().lower() == "last_pass":
            total_all = len(episode_keys)
            uniq_all = len(set(episode_keys))
            dup_all = total_all - uniq_all
            print(f"[fold {fold_idx}] [Check] last_pass overall: samples={total_all} unique_game_episode={uniq_all} dup={dup_all}")

            train_keys = [episode_keys[i] for i in idx_train]
            valid_keys = [episode_keys[i] for i in idx_valid]
            total_tr, uniq_tr = len(train_keys), len(set(train_keys))
            total_va, uniq_va = len(valid_keys), len(set(valid_keys))
            print(f"[fold {fold_idx}] [Check] last_pass train subset: samples={total_tr} unique_game_episode={uniq_tr} dup={total_tr-uniq_tr}")
            print(f"[fold {fold_idx}] [Check] last_pass valid subset: samples={total_va} unique_game_episode={uniq_va} dup={total_va-uniq_va}")

            if total_va != uniq_va:
                ctr = Counter(valid_keys)
                dups = [k for k, c in ctr.items() if c > 1]
                print(f"[fold {fold_idx}] [WARN] duplicated game_episode in VALID (n={len(dups)}). examples={dups[:10]}")

            if stage_name == "finetune":
                assert total_va == uniq_va, "Stage2(valid) not one-sample-per-episode."
                assert total_tr == uniq_tr, "Stage2(train) not one-sample-per-episode."

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

        # Build / Load model (once)
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

        # stage 시작 전: resume_path가 있으면 가중치 로드
        if resume_path and stage_name != "pretrain":
            ckpt_init = load_checkpoint(resume_path, model=model, optimizer=None)
            print(f"[fold {fold_idx}] [init] Loaded weights from: {resume_path} (epoch={ckpt_init.get('epoch')})")
            resume_path = ""

        # optimizer per stage
        opt_name = str(get_stage_param(stage_cfg, "optimizer", getattr(cfg.train, "optimizer", "adamw"))).lower()
        if opt_name in ("adamw", "adam_w"):
            optimizer = torch.optim.AdamW(model.parameters(), lr=stage_lr, weight_decay=stage_weight_decay)
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=stage_lr, weight_decay=stage_weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name!r}. Use 'adamw' or 'adam'.")

        # Train loop (this stage)
        best_dist = float("inf")
        ckpt_dir = Path("checkpoints") / f"fold_{fold_idx}"
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
                f"[fold {fold_idx}] [{stage_name}] epoch {local_epoch}/{stage_epochs} (global {global_epoch}) | "
                f"train_loss={train_loss:.4f} valid_mean_dist={valid_mean_dist:.4f}"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "fold": fold_idx,
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
                print(f"[fold {fold_idx}] --> [{stage_name}] Best model saved! global_epoch={global_epoch}, dist={dist:.4f}")

        # stage 종료 후 best 로드
        ckpt_best = load_checkpoint(str(best_ckpt_path), model=model, optimizer=None)
        model.eval()
        print(f"[fold {fold_idx}] [{stage_name}] Loaded best checkpoint: epoch={ckpt_best.get('epoch')} best_metric={ckpt_best.get('best_metric')}")

        final_best_ckpt_path = best_ckpt_path
        epoch_offset += stage_epochs

    assert final_best_ckpt_path is not None, "Training finished without a checkpoint."
    best_ckpt_path = final_best_ckpt_path

    # fold inference
    ckpt = load_checkpoint(str(best_ckpt_path), model=model, optimizer=None)  # type: ignore
    model.eval()  # type: ignore
    print(f"[fold {fold_idx}] Loaded best checkpoint: epoch={ckpt.get('epoch')} best_metric={ckpt.get('best_metric')}")

    test_meta = pd.read_csv(test_meta_path)
    submission = pd.read_csv(sample_sub_path)
    submission = submission.merge(test_meta, on="game_episode", how="left")

    fold_pred_df = predict_test_delta(
        model=model,  # type: ignore
        submission_df=submission,
        vocabs=vocabs,
        device=device,
        field_x=field_x,
        field_y=field_y,
        max_tail_k=int(getattr(cfg.data, "max_tail_k", 0) if (not two_stage) else int(getattr(cfg.train.stage2, "max_tail_k", getattr(cfg.data, "max_tail_k", 0)))),
        dt_clip_sec=dt_clip_sec,
        dt_norm_ref_sec=dt_norm_ref_sec,
    )

    pred_dir = Path("fold_predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    fold_pred_path = pred_dir / f"pred_delta_fold{fold_idx}.csv"
    fold_pred_df.to_csv(str(fold_pred_path), index=False)

    # also create per-fold submission (for debugging)
    fold_submit_df = delta_ensemble([fold_pred_df], field_x=field_x, field_y=field_y, delta_space="mean")
    out_submit_path = Path(f"fold{fold_idx}_" + str(cfg.output.submission_name))
    fold_submit_df.to_csv(str(out_submit_path), index=False)

    print(f"[fold {fold_idx}] Saved fold delta preds:", os.path.abspath(str(fold_pred_path)))
    print(f"[fold {fold_idx}] Saved fold submission:", os.path.abspath(str(out_submit_path)))

    # W&B artifacts
    if wandb_run is not None:
        log_artifact_if_possible(
            wandb_run,
            path=str(best_ckpt_path),
            art_type="model",
            name=f"model-fold{fold_idx}-{wandb_run.id}",
            aliases=[f"fold{fold_idx}", "best"],
            metadata={"fold": fold_idx, "best_metric": float(ckpt.get("best_metric", np.nan))},
        )
        log_artifact_if_possible(
            wandb_run,
            path=str(fold_pred_path),
            art_type="fold_pred",
            name=f"pred-delta-fold{fold_idx}-{wandb_run.id}",
            aliases=[f"fold{fold_idx}", "latest"],
            metadata={"fold": fold_idx},
        )
        log_artifact_if_possible(
            wandb_run,
            path=str(out_submit_path),
            art_type="submission",
            name=f"submission-fold{fold_idx}-{wandb_run.id}",
            aliases=[f"fold{fold_idx}", "latest"],
            metadata={"fold": fold_idx},
        )
        wandb_run.finish()

    return {
        "best_ckpt_path": os.path.abspath(str(best_ckpt_path)),
        "fold_pred_delta_csv": os.path.abspath(str(fold_pred_path)),
        "fold_submit_csv": os.path.abspath(str(out_submit_path)),
    }


# -------------------------
# Main (Hydra)
# -------------------------
@hydra.main(version_base=None, config_path="conf", config_name="config_kfold_v3")
def main(cfg: DictConfig) -> None:
    # Hydra run dir 안에 현재 실험 config 저장 (재현성)
    OmegaConf.save(cfg, "hydra_config_resolved.yaml")

    # (Hydra가 cwd를 바꿀 수 있으니) 데이터 경로는 절대경로로 변환 가능한 형태로 유지
    train_path = to_absolute_path(str(cfg.data.train_path))

    # Build vocab once
    vocabs, vocab_sizes = build_vocabs(train_path)

    n_folds = int(getattr(cfg.train, "n_folds", 1))
    fold = int(getattr(cfg.train, "fold", -1))
    do_ensemble = bool(getattr(cfg.train, "ensemble", False))
    delta_space = str(getattr(cfg.train, "ensemble_delta_space", "atanh")).strip().lower()

    # W&B group name (한 run 내부에서 fold들을 묶기)
    group_cfg = getattr(cfg.wandb, "group", None)
    if group_cfg in (None, "null", ""):
        base_group = f"kfold-{Path(os.getcwd()).name}"
    else:
        base_group = str(group_cfg)

    if fold == -1:
        folds_to_run = list(range(n_folds))
    else:
        if fold < 0 or fold >= n_folds:
            raise ValueError(f"train.fold must be in [0, {n_folds-1}] or -1, got {fold}")
        folds_to_run = [fold]

    results = []
    pred_paths = []
    for fidx in folds_to_run:
        res = run_one_fold(cfg=cfg, fold_idx=fidx, vocabs=vocabs, vocab_sizes=vocab_sizes, base_group=base_group)
        results.append(res)
        pred_paths.append(res["fold_pred_delta_csv"])

    # Ensemble: all folds only
    if do_ensemble and (fold == -1) and n_folds > 1:
        field_x = float(cfg.data.field_x)
        field_y = float(cfg.data.field_y)

        dfs = [pd.read_csv(p) for p in pred_paths]
        ens_df = delta_ensemble(dfs, field_x=field_x, field_y=field_y, delta_space=delta_space)

        ens_name = str(getattr(cfg.output, "ensemble_submission_name", "ensemble_submit.csv"))
        ens_path = os.path.abspath(ens_name)
        ens_df.to_csv(ens_path, index=False)
        print("Saved ensemble submission:", ens_path)

        # W&B ensemble artifact logging (separate run)
        wandb_run = init_wandb(
            cfg,
            name=(None if cfg.wandb.name in (None, "null") else f"{str(cfg.wandb.name)}-ensemble"),
            group=base_group,
            job_type="ensemble",
            extra_config={"delta_space": delta_space, "n_folds": n_folds},
        )
        if wandb_run is not None:
            # bundle all fold pred files into one artifact for traceability
            import wandb
            bundle = wandb.Artifact(f"fold-preds-{wandb_run.id}", type="fold_preds", metadata={"n_folds": n_folds})
            for p in pred_paths:
                bundle.add_file(p)
            wandb_run.log_artifact(bundle, aliases=["latest"]).wait()

            log_artifact_if_possible(
                wandb_run,
                path=ens_path,
                art_type="submission",
                name=f"submission-ensemble-{wandb_run.id}",
                aliases=["ensemble", "latest"],
                metadata={"delta_space": delta_space, "n_folds": n_folds},
            )
            wandb_run.finish()

    print("Done.")


if __name__ == "__main__":
    main()
