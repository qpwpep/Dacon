# Hydra + W&B LightGBM Baseline

이 폴더는 기존 `baseline.py`(argparse 기반)를 **Hydra 설정 기반**으로 바꾸고,
실험 로그/아티팩트를 **Weights & Biases(wandb)** 로 기록할 수 있게 만든 버전입니다.

## 설치
```bash
pip install hydra-core omegaconf wandb lightgbm scikit-learn pandas numpy
```

## 실행
> Hydra는 실행 시 작업 디렉토리를 `outputs/...`로 바꿉니다.  
> `train.csv/test.csv/sample_submission.csv`는 실행한 원본 폴더 기준으로 상대경로가 해석됩니다.

```bash
python train_hydra.py
```

## 설정 오버라이드 예시 (Hydra)
```bash
# learning rate / num_leaves 바꿔서 실행
python train_hydra.py model.learning_rate=0.03 model.num_leaves=127

# 멀티런 (그리드 서치)
python train_hydra.py -m model.learning_rate=0.03,0.05 model.num_leaves=63,127
```

## W&B 온라인 동기화
기본은 `wandb.mode=offline` 입니다. 온라인으로 올리려면:
```bash
python train_hydra.py wandb.mode=online wandb.project=YOUR_PROJECT
```

## 출력물
각 실행(run)마다 아래 파일이 `outputs/YYYY-MM-DD/HH-MM-SS/` 아래에 생성됩니다.
- `submit_lgbm.csv` : 예측 라벨 제출 파일
- `submit_lgbm_proba.csv` : 예측 확률
- `oof_proba.csv` : CV OOF 확률/정답 (threshold 최적화에 사용)
- `final_model.txt` : LightGBM 모델 텍스트 저장

Hydra 설정 파일은 `.hydra/` 폴더에 자동 저장됩니다.
