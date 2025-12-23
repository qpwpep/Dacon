import copy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader


TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"
SUBMISSION = "./data/sample_submission.csv"

BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
HIDDEN_DIM = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


df = pd.read_csv("./data/train.csv")
df = df.sort_values(["game_episode", "time_seconds"]).reset_index(drop=True)

episodes = []
targets = []

for _, g in tqdm(df.groupby("game_episode")):
    g = g.reset_index(drop=True)
    if len(g) < 2:
        continue

    # 정규화된 좌표 준비
    sx = g["start_x"].values / 105.0 # type: ignore
    sy = g["start_y"].values / 68.0 # type: ignore
    ex = g["end_x"].values   / 105.0 # type: ignore
    ey = g["end_y"].values   / 68.0 # type: ignore

    coords = []
    for i in range(len(g)):
        # 항상 start는 들어감
        coords.append([sx[i], sy[i]])
        # 마지막 행 이전까지만 end를 넣음 (마지막 end는 타깃이므로)
        if i < len(g) - 1:
            coords.append([ex[i], ey[i]])

    seq = np.array(coords, dtype="float32")        # [T, 2]
    target = np.array([ex[-1], ey[-1]], dtype="float32")  # 마지막 행 end_x, end_y

    episodes.append(seq)
    targets.append(target)

print("에피소드 수 : ", len(episodes))


class EpisodeDataset(Dataset):
    def __init__(self, episodes, targets):
        self.episodes = episodes
        self.targets = targets

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        seq = torch.tensor(self.episodes[idx])   # [T, 2]
        tgt = torch.tensor(self.targets[idx])    # [2]
        length = seq.size(0)
        return seq, length, tgt

def collate_fn(batch):
    seqs, lengths, tgts = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)  # type: ignore # [B, T, 2]
    tgts = torch.stack(tgts, dim=0)                # [B, 2]
    return padded, lengths, tgts

# 에피소드 단위 train / valid split
idx_train, idx_valid = train_test_split(
    np.arange(len(episodes)), test_size=0.2, random_state=42
)

episodes_train = [episodes[i] for i in idx_train]
targets_train  = [targets[i]  for i in idx_train]
episodes_valid = [episodes[i] for i in idx_valid]
targets_valid  = [targets[i]  for i in idx_valid]

train_loader = DataLoader(
    EpisodeDataset(episodes_train, targets_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

valid_loader = DataLoader(
    EpisodeDataset(episodes_valid, targets_valid),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)

print("train episodes:", len(episodes_train), "valid episodes:", len(episodes_valid))


def euclidean_loss_meters(pred_norm, true_norm, field_x=105.0, field_y=68.0, eps=1e-9):
    """Mean Euclidean distance in meters (same scale as evaluation)."""
    dx = (pred_norm[:, 0] - true_norm[:, 0]) * field_x
    dy = (pred_norm[:, 1] - true_norm[:, 1]) * field_y
    dist = torch.sqrt(dx * dx + dy * dy + eps)
    return dist.mean()


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 2)  # (x_norm, y_norm)

    def forward(self, x, lengths):
        # x: [B, T, 2], lengths: [B]
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]      # [B, H] 마지막 layer의 hidden state
        out = self.fc(h_last) # [B, 2]
        out = torch.sigmoid(out)
        out = out.clamp(0.0, 1.0)
        return out

model = LSTMBaseline(input_dim=2, hidden_dim=HIDDEN_DIM).to(DEVICE)
criterion = euclidean_loss_meters
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


best_dist = float("inf")
best_model_state = None

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    total_loss = 0.0

    for X, lengths, y in tqdm(train_loader):
        X, lengths, y = X.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        pred = model(X, lengths)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    train_loss = total_loss / len(train_loader.dataset) # type: ignore

    # --- Valid: 평균 유클리드 거리 ---
    model.eval()
    dists = []

    with torch.no_grad():
        for X, lengths, y in tqdm(valid_loader):
            X, lengths, y = X.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
            pred = model(X, lengths)

            pred_np = pred.cpu().numpy()
            true_np = y.cpu().numpy()

            pred_x = pred_np[:, 0] * 105.0
            pred_y = pred_np[:, 1] * 68.0
            true_x = true_np[:, 0] * 105.0
            true_y = true_np[:, 1] * 68.0

            dist = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
            dists.append(dist)

    mean_dist = np.concatenate(dists).mean()  # 평균 유클리드 거리

    print(
        f"[Epoch {epoch}] "
        f"train_loss={train_loss:.4f} | "
        f"valid_mean_dist={mean_dist:.4f}"
    )

    # ----- BEST MODEL 업데이트 -----
    if mean_dist < best_dist:
        best_dist = mean_dist
        best_model_state = copy.deepcopy(model.state_dict())
        print(f" --> Best model updated! (dist={best_dist:.4f})")


# Best Model Load
model.load_state_dict(best_model_state) # type: ignore
model.eval()

test_meta = pd.read_csv("./data/test.csv")
submission = pd.read_csv("./data/sample_submission.csv")

submission = submission.merge(test_meta, on="game_episode", how="left")

preds_x, preds_y = [], []

for _, row in tqdm(submission.iterrows(), total=len(submission)):
    g = pd.read_csv(row["path"]).reset_index(drop=True)
    # 정규화된 좌표 준비
    sx = g["start_x"].values / 105.0 # type: ignore
    sy = g["start_y"].values / 68.0 # type: ignore
    ex = g["end_x"].values / 105.0 # type: ignore
    ey = g["end_y"].values / 68.0 # type: ignore

    coords = []
    for i in range(len(g)):
        # start는 항상 존재하므로 그대로 사용
        coords.append([sx[i], sy[i]])
        # 마지막 행은 end_x가 NaN이므로 자동으로 제외됨
        if i < len(g) - 1:
            coords.append([ex[i], ey[i]])

    seq = np.array(coords, dtype="float32")  # [T, 2]

    x = torch.tensor(seq).unsqueeze(0).to(DEVICE)      # [1, T, 2]
    length = torch.tensor([seq.shape[0]]).to(DEVICE)   # [1]

    with torch.no_grad():
        pred = model(x, length).cpu().numpy()[0]       # [2], 정규화 좌표

    preds_x.append(pred[0] * 105.0)
    preds_y.append(pred[1] * 68.0)
print("Inference Done.")


submission["end_x"] = preds_x
submission["end_y"] = preds_y
submission[["game_episode", "end_x", "end_y"]].to_csv("./baseline_submit.csv", index=False)
print("Saved: baseline_submit.csv")
