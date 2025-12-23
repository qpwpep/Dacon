from __future__ import annotations

import os
import copy
from pathlib import Path
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional

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
    pred_norm/true_norm: [B, 2] (0~1로 정규화된 좌표)
    return: 평균 유클리드 거리 (미터 스케일)
    """
    dx = (pred_norm[:, 0] - true_norm[:, 0]) * field_x
    dy = (pred_norm[:, 1] - true_norm[:, 1]) * field_y
    dist = torch.sqrt(dx * dx + dy * dy + eps)
    return dist.mean()


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
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]:
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
    episode_game_ids: List[int] = []

    def map_idx(g: pd.DataFrame, col: str, vocab: dict) -> np.ndarray:
        vals = g[col].fillna("None").astype(str).values
        return np.asarray([vocab.get(v, 1) for v in vals], dtype=np.int64)  # default UNK=1

    for game_episode, g in tqdm(df.groupby("game_episode"), desc="Build episodes (train)"):
        if max_tail_k and max_tail_k > 0:
            g = g.tail(int(max_tail_k)).reset_index(drop=True)

        game_id = int(str(game_episode).split("_", 1)[0])
        episode_game_ids.append(game_id)

        # --- absolute normalized coords (0~1) ---
        sx_abs = (g["start_x"].values / field_x).astype(np.float32)  # type: ignore
        sy_abs = (g["start_y"].values / field_y).astype(np.float32)  # type: ignore
        ex_abs = (g["end_x"].values / field_x).astype(np.float32)    # type: ignore
        ey_abs = (g["end_y"].values / field_y).astype(np.float32)    # type: ignore

        # Anchor = 마지막 action(=예측 대상 Pass)의 start 좌표 (정규화)
        anchor_x = float(sx_abs[-1])
        anchor_y = float(sy_abs[-1])

        # --- anchor-relative coords (delta coordinate frame) ---
        sx = (sx_abs - anchor_x).astype(np.float32)
        sy = (sy_abs - anchor_y).astype(np.float32)
        ex = (ex_abs - anchor_x).astype(np.float32)
        ey = (ey_abs - anchor_y).astype(np.float32)

        # 타깃: 마지막 row의 end를 anchor 기준 델타로 예측
        tgt = np.asarray([float(ex[-1]), float(ey[-1])], dtype=np.float32)
        targets.append(tgt)

        T = len(g)

        # 입력은 test와 동일하게: 마지막 end는 비워둠
        end_mask = np.ones((T,), dtype=np.float32)
        end_mask[-1] = 0.0

        ex_filled = ex.copy()
        ey_filled = ey.copy()
        # 마지막 row는 end가 없다고 가정 -> (start와 동일) => 상대좌표계에선 (0,0)
        ex_filled[-1] = 0.0
        ey_filled[-1] = 0.0

        dx = np.where(end_mask > 0, ex - sx, 0.0).astype(np.float32)
        dy = np.where(end_mask > 0, ey - sy, 0.0).astype(np.float32)
        dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)

        ang = np.arctan2(dy, dx).astype(np.float32)
        eps = 1e-6
        valid_dir = (end_mask > 0) & (dist > eps)
        angle_sin = np.where(valid_dir, np.sin(ang), 0.0).astype(np.float32)
        angle_cos = np.where(valid_dir, np.cos(ang), 0.0).astype(np.float32)

        t = g["time_seconds"].values.astype(np.float32)
        dt = np.diff(t, prepend=t[0]) # type: ignore
        dt = np.clip(dt, 0.0, None).astype(np.float32)
        # 0~1 근처 스케일로: 대략 60초를 1 근처로
        dt = (np.log1p(dt) / np.log1p(60.0)).astype(np.float32)

        # anchor absolute position (normalized) as additional numeric features
        anchor_x_feat = np.full((T,), anchor_x, dtype=np.float32)
        anchor_y_feat = np.full((T,), anchor_y, dtype=np.float32)

        num = np.stack(
            [sx, sy, ex_filled, ey_filled, end_mask, dx, dy, dist, angle_sin, angle_cos, dt, anchor_x_feat, anchor_y_feat],
            axis=1,
        ).astype(np.float32)  # [T, 11]

        cat = np.stack(
            [
                map_idx(g, "player_id", vocabs["player_id"]),
                map_idx(g, "team_id", vocabs["team_id"]),
                map_idx(g, "type_name", vocabs["type_name"]),
                map_idx(g, "result_name", vocabs["result_name"]),
            ],
            axis=1,
        ).astype(np.int64)  # [T, 4]

        episodes_num.append(num)
        episodes_cat.append(cat)

    return episodes_num, episodes_cat, targets, episode_game_ids


def build_test_sequence_from_path(
    episode_csv_path: str,
    field_x: float,
    field_y: float,
    vocabs: dict,
    max_tail_k: int = 0,
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

    # --- absolute normalized coords (0~1) ---
    sx_abs = (g["start_x"].values / field_x).astype(np.float32)  # type: ignore
    sy_abs = (g["start_y"].values / field_y).astype(np.float32)  # type: ignore

    ex_raw = g["end_x"].values.astype(np.float32)  # type: ignore
    ey_raw = g["end_y"].values.astype(np.float32)  # type: ignore

    end_ok = (~np.isnan(ex_raw)) & (~np.isnan(ey_raw))
    end_mask = end_ok.astype(np.float32)

    ex_abs = (np.nan_to_num(ex_raw, nan=0.0) / field_x).astype(np.float32) # type: ignore
    ey_abs = (np.nan_to_num(ey_raw, nan=0.0) / field_y).astype(np.float32) # type: ignore
    ex_abs = np.where(end_mask > 0, ex_abs, sx_abs)
    ey_abs = np.where(end_mask > 0, ey_abs, sy_abs)

    # Anchor = 마지막 action의 start(정규화)
    anchor_x = float(sx_abs[-1])
    anchor_y = float(sy_abs[-1])

    # --- anchor-relative coords ---
    sx = (sx_abs - anchor_x).astype(np.float32)
    sy = (sy_abs - anchor_y).astype(np.float32)
    ex = (ex_abs - anchor_x).astype(np.float32)
    ey = (ey_abs - anchor_y).astype(np.float32)

    dx = np.where(end_mask > 0, ex - sx, 0.0).astype(np.float32)
    dy = np.where(end_mask > 0, ey - sy, 0.0).astype(np.float32)
    dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)

    ang = np.arctan2(dy, dx).astype(np.float32)
    eps = 1e-6
    valid_dir = (end_mask > 0) & (dist > eps)
    angle_sin = np.where(valid_dir, np.sin(ang), 0.0).astype(np.float32)
    angle_cos = np.where(valid_dir, np.cos(ang), 0.0).astype(np.float32)

    t = g["time_seconds"].values.astype(np.float32)
    dt = np.diff(t, prepend=t[0]) # type: ignore
    dt = np.clip(dt, 0.0, None).astype(np.float32)
    dt = (np.log1p(dt) / np.log1p(60.0)).astype(np.float32)

    T = len(g)
    # anchor absolute position (normalized) as additional numeric features
    anchor_x_feat = np.full((T,), anchor_x, dtype=np.float32)
    anchor_y_feat = np.full((T,), anchor_y, dtype=np.float32)

    num = np.stack(
        [sx, sy, ex, ey, end_mask, dx, dy, dist, angle_sin, angle_cos, dt, anchor_x_feat, anchor_y_feat],
        axis=1,
    ).astype(np.float32)  # [T, 13]

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
    max_tail_k = int(cfg.data.max_tail_k) if "max_tail_k" in cfg.data else 0

    episodes_num, episodes_cat, targets, episode_game_ids = build_train_episodes(
        train_csv_path=train_path, field_x=field_x, field_y=field_y, vocabs=vocabs, max_tail_k=max_tail_k
        )
    print("에피소드 수:", len(episodes_num))

    seed = int(cfg.train.seed) if cfg.train.seed is not None else 42
    gss = GroupShuffleSplit(n_splits=1, test_size=float(cfg.train.valid_ratio), random_state=seed)
    idx_train, idx_valid = next(gss.split(np.arange(len(episodes_num)), groups=episode_game_ids))
    train_games = set(np.array(episode_game_ids)[idx_train])
    valid_games = set(np.array(episode_game_ids)[idx_valid])
    print("overlap games:", len(train_games & valid_games))  # 반드시 0이어야 함
    print("train games:", len(train_games), "valid games:", len(valid_games))

    episodes_num_train = [episodes_num[i] for i in idx_train]
    episodes_cat_train = [episodes_cat[i] for i in idx_train]
    targets_train = [targets[i] for i in idx_train]

    episodes_num_valid = [episodes_num[i] for i in idx_valid]
    episodes_cat_valid = [episodes_cat[i] for i in idx_valid]
    targets_valid = [targets[i] for i in idx_valid]
    print("train episodes:", len(episodes_num_train), "valid episodes:", len(episodes_num_valid))

    train_loader = DataLoader(
        EpisodeDataset(episodes_num_train, episodes_cat_train, targets_train),
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=int(cfg.train.num_workers),
        pin_memory=bool(cfg.train.pin_memory),
    )
    valid_loader = DataLoader(
        EpisodeDataset(episodes_num_valid, episodes_cat_valid, targets_valid),
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=int(cfg.train.num_workers),
        pin_memory=bool(cfg.train.pin_memory),
    )

    # -----------------
    # Model / Optim
    # -----------------
    numeric_dim = int(episodes_num_train[0].shape[1])  # 11
    # emb_dims는 config(model.emb_dims)에서 가져오고, 없으면 기본값 사용
    if "emb_dims" in cfg.model and cfg.model.emb_dims is not None:
        emb_dims = OmegaConf.to_container(cfg.model.emb_dims, resolve=True)  # type: ignore
    else:
        emb_dims = {"player_id": 16, "team_id": 8, "type_name": 8, "result_name": 4}

    model = LSTMWithEmb(
        numeric_dim=numeric_dim,
        vocab_sizes=vocab_sizes,
        emb_dims=emb_dims, # type: ignore
        hidden_dim=int(cfg.model.hidden_dim),
        num_layers=int(cfg.model.num_layers),
        dropout=float(cfg.model.dropout),
        bidirectional=bool(cfg.model.bidirectional),
        emb_dropout=float(cfg.model.get("emb_dropout", 0.0)),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    if wandb_run is not None:
        # 파라미터 수 같은 메타 정보 기록
        n_params = sum(p.numel() for p in model.parameters())
        wandb_run.log({"model/n_params": n_params})

    # -----------------
    # Train loop
    # -----------------
    best_dist = float("inf")
    best_epoch = -1
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = ckpt_dir / str(cfg.output.ckpt_name)  # cfg.output.ckpt_name 사용
    stem, suffix = best_ckpt_path.stem, best_ckpt_path.suffix

    for epoch in range(1, int(cfg.train.epochs) + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            field_x=field_x,
            field_y=field_y,
            grad_clip=float(cfg.train.grad_clip),
            amp=bool(cfg.train.amp),
            log_every_steps=int(cfg.train.log_every_steps),
            wandb_run=wandb_run,
            epoch=epoch,
        )

        valid_mean_dist = evaluate(
            model=model,
            loader=valid_loader,
            device=device,
            field_x=field_x,
            field_y=field_y,
        )

        print(
            f"[Epoch {epoch}] "
            f"train_loss(m)={train_loss:.4f} | "
            f"valid_mean_dist(m)={valid_mean_dist:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss_m": float(train_loss),
                    "valid/mean_dist_m": float(valid_mean_dist),
                    "best/mean_dist_m": float(min(best_dist, valid_mean_dist)),
                }
            )

        if valid_mean_dist < best_dist:
            dist = float(valid_mean_dist)
            best_dist = dist
            best_epoch = epoch

            ckpt_epoch_path = ckpt_dir / f"{stem}_epoch{epoch:03d}_dist{dist:.4f}{suffix}"

            # 1) epoch별 불변 파일로 저장
            save_checkpoint(
                path=str(ckpt_epoch_path),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=dist,
                cfg_dict=OmegaConf.to_container(cfg, resolve=True), # type: ignore
            )

            # 2) best 파일은 복사본으로만 갱신 (로드 편의)
            shutil.copy2(str(ckpt_epoch_path), str(best_ckpt_path))

            print(f" --> Best model saved! epoch={epoch}, dist={dist:.4f}")


    # -----------------
    # Load best + inference
    # -----------------
    ckpt = load_checkpoint(str(best_ckpt_path), model=model, optimizer=None)

    model.eval()
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
            episode_path, field_x=field_x, field_y=field_y, vocabs=vocabs, max_tail_k=max_tail_k
        )

        x_num = torch.tensor(num, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T, F]
        x_cat = torch.tensor(cat, dtype=torch.long).unsqueeze(0).to(device)     # [1, T, 4]
        length = torch.tensor([num.shape[0]]).to(device)

        with torch.no_grad():
            pred = model(x_num, x_cat, length).detach().cpu().numpy()[0]  # [2]

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
