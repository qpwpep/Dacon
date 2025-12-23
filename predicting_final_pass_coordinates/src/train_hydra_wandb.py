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
def build_train_episodes(
    train_csv_path: str,
    field_x: float,
    field_y: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    train.csv를 읽어서
      - 입력 시퀀스: start(x,y), end(x,y), start(x,y), end(x,y), ...
        (단, 마지막 row의 end는 타깃으로 쓰기 때문에 입력에 포함하지 않음)
      - 타깃: 마지막 row의 end_x, end_y (정규화)
    를 에피소드 단위로 만들어 리스트로 반환.
    """
    df = pd.read_csv(train_csv_path)
    # baseline에서처럼 episode/time 정렬 (시퀀스 순서를 보장) fileciteturn2file0L26-L27
    sort_cols = ["game_episode", "time_seconds"] + (["action_id"] if "action_id" in df.columns else [])
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    episodes: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    episode_game_ids: List[int] = []

    for game_episode, g in tqdm(df.groupby("game_episode"), desc="Build episodes (train)"):
        # game_id 파싱 (game_episode = "{game_id}_{episode_id}")
        game_id = int(str(game_episode).split("_", 1)[0])
        episode_game_ids.append(game_id)

        # 정규화된 좌표 준비 fileciteturn2file0L37-L41
        sx = (g["start_x"].values / field_x).astype(np.float32) # type: ignore
        sy = (g["start_y"].values / field_y).astype(np.float32) # type: ignore
        ex = (g["end_x"].values / field_x).astype(np.float32) # type: ignore
        ey = (g["end_y"].values / field_y).astype(np.float32) # type: ignore

        coords: List[List[float]] = []
        for i in range(len(g)):
            coords.append([float(sx[i]), float(sy[i])])  # start는 항상 포함 fileciteturn2file0L45-L46
            if i < len(g) - 1:
                coords.append([float(ex[i]), float(ey[i])])  # 마지막 end는 입력 제외 fileciteturn2file0L47-L49

        seq = np.asarray(coords, dtype=np.float32)  # [T, 2] fileciteturn2file0L51-L52
        tgt = np.asarray([float(ex[-1]), float(ey[-1])], dtype=np.float32)

        episodes.append(seq)
        targets.append(tgt)

    return episodes, targets, episode_game_ids


def build_test_sequence_from_path(
    episode_csv_path: str,
    field_x: float,
    field_y: float,
) -> np.ndarray:
    """
    test의 개별 에피소드 csv를 읽어서 입력 시퀀스를 생성.
    test에서는 마지막 row의 end_x/end_y가 NaN인 경우가 많아 baseline은
    마지막 row 이전까지만 end를 넣는 방식으로 처리함 fileciteturn2file0L223-L225
    """
    g = pd.read_csv(episode_csv_path)
    sort_cols = ["time_seconds"] + (["action_id"] if "action_id" in g.columns else [])
    g = g.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    sx = (g["start_x"].values / field_x).astype(np.float32) # type: ignore
    sy = (g["start_y"].values / field_y).astype(np.float32) # type: ignore

    # end는 마지막 row를 제외하고만 사용 (마지막이 NaN인 케이스 대비) fileciteturn2file0L223-L225
    ex = (g["end_x"].values / field_x).astype(np.float32) # type: ignore
    ey = (g["end_y"].values / field_y).astype(np.float32) # type: ignore

    coords: List[List[float]] = []
    n = len(g)
    for i in range(n):
        coords.append([float(sx[i]), float(sy[i])])
        if i < n - 1:
            coords.append([float(ex[i]), float(ey[i])])

    return np.asarray(coords, dtype=np.float32)


class EpisodeDataset(Dataset):
    def __init__(self, episodes: List[np.ndarray], targets: List[np.ndarray]):
        self.episodes = episodes
        self.targets = targets

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int):
        seq = torch.tensor(self.episodes[idx])  # [T, 2] fileciteturn2file0L68-L72
        tgt = torch.tensor(self.targets[idx])   # [2]
        length = seq.size(0)
        return seq, length, tgt


def collate_fn(batch):
    seqs, lengths, tgts = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)  # type: ignore # [B, T, 2] fileciteturn2file0L74-L79
    tgts = torch.stack(tgts, dim=0)
    return padded, lengths, tgts


# -------------------------
# Model
# -------------------------
class LSTMBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 2)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # 마지막 layer의 hidden state
        h_last = h_n[-1]
        out = self.fc(h_last)
        out = torch.sigmoid(out).clamp(0.0, 1.0)  # baseline과 동일하게 0~1로 클램프 fileciteturn2file0L134-L137
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

    for X, lengths, y in tqdm(loader, desc="Valid", leave=False):
        X, lengths, y = X.to(device), lengths.to(device), y.to(device)
        pred = model(X, lengths)
        dist = euclidean_loss_meters(pred, y, field_x=field_x, field_y=field_y)
        bs = X.size(0)
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

    for step, (X, lengths, y) in enumerate(tqdm(loader, desc="Train", leave=False), start=1):
        X, lengths, y = X.to(device), lengths.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            pred = model(X, lengths)
            loss = euclidean_loss_meters(pred, y, field_x=field_x, field_y=field_y)

        scaler.scale(loss).backward()

        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        bs = X.size(0)
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
    episodes, targets, episode_game_ids = build_train_episodes(
        train_path, field_x=field_x, field_y=field_y
        )
    print("에피소드 수:", len(episodes))

    seed = int(cfg.train.seed) if cfg.train.seed is not None else 42
    gss = GroupShuffleSplit(n_splits=1, test_size=float(cfg.train.valid_ratio), random_state=seed)
    idx_train, idx_valid = next(gss.split(np.arange(len(episodes)), groups=episode_game_ids))
    train_games = set(np.array(episode_game_ids)[idx_train])
    valid_games = set(np.array(episode_game_ids)[idx_valid])
    print("overlap games:", len(train_games & valid_games))  # 반드시 0이어야 함
    print("train games:", len(train_games), "valid games:", len(valid_games))

    episodes_train = [episodes[i] for i in idx_train]
    targets_train = [targets[i] for i in idx_train]
    episodes_valid = [episodes[i] for i in idx_valid]
    targets_valid = [targets[i] for i in idx_valid]
    print("train episodes:", len(episodes_train), "valid episodes:", len(episodes_valid))

    train_loader = DataLoader(
        EpisodeDataset(episodes_train, targets_train),
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=int(cfg.train.num_workers),
        pin_memory=bool(cfg.train.pin_memory),
    )
    valid_loader = DataLoader(
        EpisodeDataset(episodes_valid, targets_valid),
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=int(cfg.train.num_workers),
        pin_memory=bool(cfg.train.pin_memory),
    )

    # -----------------
    # Model / Optim
    # -----------------
    model = LSTMBaseline(
        input_dim=int(cfg.model.input_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_layers=int(cfg.model.num_layers),
        dropout=float(cfg.model.dropout),
        bidirectional=bool(cfg.model.bidirectional),
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
        seq = build_test_sequence_from_path(episode_path, field_x=field_x, field_y=field_y)

        x = torch.tensor(seq).unsqueeze(0).to(device)  # [1, T, 2] fileciteturn2file0L229-L230
        length = torch.tensor([seq.shape[0]]).to(device)

        with torch.no_grad():
            pred = model(x, length).detach().cpu().numpy()[0]  # [2] fileciteturn2file0L232-L234

        preds_x.append(float(pred[0] * field_x))
        preds_y.append(float(pred[1] * field_y))

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
