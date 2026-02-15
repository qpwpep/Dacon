# Predicting Final Pass Coordinates

LSTM baseline for predicting the final pass destination (`end_x`, `end_y`) from event sequences in football episodes.

Main entrypoint: `src/baseline.py`

## 1. Task

- Input: event sequence for each `game_episode`
- Output: final pass destination in field coordinates (`end_x`, `end_y`)
- Field size: `105m x 68m`
- Training objective: mean Euclidean distance in meters

## 2. Project Structure

```text
.
|-- src/
|   |-- baseline.py            # main training/inference/ensemble pipeline
|   |-- conf/
|   |   `-- config.yaml        # Hydra config
|   `-- eda.ipynb              # EDA notebook
|-- baseline.py                # legacy simple baseline script
|-- pyproject.toml
`-- README.md
```

## 3. EDA Snapshot (`src/eda.ipynb`)

Notebook outputs show:

- Train shape: `(356721, 15)`
- Test shape: `(2414, 3)`
- Unique `type_name`: `26`
- Unique `team_id`: `12`
- `start_x` mean `47.26`, `start_y` mean `34.15`
- `end_x` mean `51.04`, `end_y` mean `34.13`
- Coordinate range: `[0, 105] x [0, 68]`

## 4. Data Pipeline (`src/baseline.py`)

### 4.1 Sequence construction

- Group by `game_episode`
- Stable sort by `action_id` (if present) and `time_seconds`
- Build variable-length episode sequences

### 4.2 Team-frame normalization

`unify_frame_to_ref_team` normalizes directions by rotating opponent actions:

- `x' = field_x - x`
- `y' = field_y - y`

This makes sequences team-centric around the target pass team.

### 4.3 Target policy

Two supported sample policies:

- `all_pass`: use all pass events in each episode as training targets
- `last_pass`: use only one pass target per episode (test-like setting)

Default config uses 2-stage training:

- Stage 1 (pretrain): `all_pass`
- Stage 2 (finetune): `last_pass`

### 4.4 Features per timestep

The model consumes `21 numeric + 4 categorical` features.

Numeric features:

1. `sx_abs`, `sy_abs`
2. `ex_abs_filled`, `ey_abs_filled`
3. `sx_rel`, `sy_rel`
4. `ex_rel_filled`, `ey_rel_filled`
5. `end_mask`
6. `dx`, `dy`, `dist`
7. `angle_sin`, `angle_cos`
8. `dt` (clipped and log-normalized)
9. `t_abs_log`
10. `is_second_half`
11. `idx_norm`, `steps_to_target`
12. `anchor_x`, `anchor_y`

Categorical features:

1. `player_id`
2. `team_id`
3. `type_name`
4. `result_name`

Index convention for categorical vocab:

- `0`: PAD
- `1`: UNK
- `2+`: observed categories

### 4.5 End-point handling

For actions with no valid end position (`NaN` or in `NO_END_TYPES`):

- `end_mask = 0`
- absolute end is replaced with start
- relative end is replaced with `0`

### 4.6 Training target

Model target is anchor-relative delta, not absolute coordinate:

- `anchor = (last_pass_start_x_norm, last_pass_start_y_norm)`
- `delta_x = end_x_norm - anchor_x`
- `delta_y = end_y_norm - anchor_y`

Model output is constrained by `tanh` to `[-1, 1]`.

## 5. Model (`LSTMWithEmb`)

Architecture:

- Embedding layers for 4 categorical columns
- Concatenate embeddings with numeric features
- Packed LSTM over variable-length sequences
- Final hidden state -> linear head -> 2D delta regression

Default config (`src/conf/config.yaml`):

- `hidden_dim=128`
- `num_layers=2`
- `dropout=0.1`
- `bidirectional=true`
- Embedding dims: `player_id=16`, `team_id=8`, `type_name=8`, `result_name=4`

## 6. Training Strategy

### 6.1 Group K-fold split

- Split by `game_id` to avoid leakage
- Default: `n_folds=5`
- `fold=-1` runs all folds

### 6.2 Stage-wise optimization

- Different hyperparameters can be assigned per stage
- AMP (`torch.amp`) and gradient clipping are supported
- Best checkpoint selected by validation mean distance

### 6.3 Episode cache

Stage-wise episode packs are cached to speed up reruns:

- Cache path default: `cache/episodes_kfold_v3`
- Metadata checks include format version, feature dimensions, vocab signature, and preprocessing signatures (`NO_END_TYPES`, dt settings, field size, etc.)

## 7. Inference and Ensemble

### 7.1 Fold inference

For each fold:

- Predict normalized deltas for each test episode
- Save `game_episode`, `anchor_x`, `anchor_y`, `delta_x`, `delta_y`
- Output file: `fold_predictions/pred_delta_fold{k}.csv`

### 7.2 Delta-space ensemble

`delta_ensemble` supports:

- `mean`: average in delta space
- `atanh` (default): average in pre-tanh space (`atanh -> mean -> tanh`)
- optional trimmed mean with `ensemble_trim_k`

Final submission is reconstructed as:

- `end_norm = clip(anchor + ensembled_delta, 0, 1)`
- `end_x = end_norm_x * field_x`
- `end_y = end_norm_y * field_y`

## 8. Config Highlights (`src/conf/config.yaml`)

Main groups:

- `data`: file paths, field size, target policy, time normalization
- `model`: LSTM and embedding settings
- `train`: seed, optimizer, folds, 2-stage setup, ensemble options
- `output`: checkpoint names, submission names, cache options
- `wandb`: experiment tracking options

Default behavior in this repo:

- 2-stage training enabled (`train.two_stage=true`)
- 5-fold training enabled (`train.n_folds=5`, `train.fold=-1`)
- Fold ensemble enabled (`train.ensemble=true`)

## 9. How To Run

### 9.1 Install

```bash
uv sync
```

or

```bash
pip install -e .
```

### 9.2 Expected data layout

```text
data/
|-- train.csv
|-- test.csv
`-- sample_submission.csv
```

### 9.3 Run full default pipeline (5 folds + ensemble)

```bash
python src/baseline.py
```

### 9.4 Run a single fold

```bash
python src/baseline.py train.fold=0 train.ensemble=false
```

### 9.5 Disable W&B

```bash
python src/baseline.py wandb.enabled=false
```

## 10. Output Artifacts

Inside each Hydra run directory:

- `hydra_config_resolved.yaml`
- `checkpoints/fold_{k}/*.pt`
- `fold_predictions/pred_delta_fold{k}.csv`
- `fold{k}_baseline_submit.csv`
- `ensemble_submit.csv` (when all folds + ensemble run)

## 11. Portfolio Talking Points

This implementation demonstrates:

- Test-condition-aware 2-stage training (`all_pass -> last_pass`)
- Leakage-safe game-level cross validation
- Structured feature engineering for variable-length sports event sequences
- Reproducible experiment setup with Hydra, deterministic seed handling, and W&B artifacts
- Efficient reruns with validated stage-level preprocessing cache
