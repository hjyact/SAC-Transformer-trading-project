# SAC-Transformer Trading

단일 종목을 대상으로 학습하는 Soft Actor-Critic와 Transformer 결합 트레이딩 에이전트입니다.

실행 경로는 하나입니다.

- 환경: `TradingEnv`
- 모델: Transformer-SAC (`GTrXL + PatchTST + RevIN`)
- 검증: 시간 순서 train/test split, train 통계만 eval 정규화에 주입
- 비용: 수수료와 슬리피지를 환경 보상과 체결에 반영
- 리스크: Sharpe/Sortino/MDD/CVaR 계열 보상과 Buy & Hold 비교

다종목 학습, 앙상블, MLP/Transformer 백본 선택 CLI는 제거했습니다.

## 근거

현재 구조는 아래 관행을 기준으로 맞췄습니다.

- FinRL 계열 금융 RL: 거래비용, 유동성/리스크 제약을 모델 평가에 포함
- 시계열/Kaggle 관행: 랜덤 K-fold 대신 시간 순서 보존, walk-forward 또는 holdout 검증, 미래 데이터 누수 방지
- PatchTST/RevIN 계열: 비정상 시계열에 Transformer 패치 인코딩과 인스턴스 정규화 적용
- SAC 계열: twin critic, entropy tuning, replay buffer, N-step/PER, DroQ, action smoothness로 학습 안정화

## 설치

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

이미 `.venv`가 있으면 설치 명령만 다시 실행하면 됩니다.

```bash
.venv/bin/python -m pip install -r requirements.txt
```

시스템 파이썬(`/usr/bin/python`)으로 실행하면 `ModuleNotFoundError: No module named 'torch'`가 날 수 있습니다. 이 프로젝트는 `.venv/bin/python`으로 실행합니다.

## 빠른 시작

먼저 PyTorch, CUDA, CPU worker 설정을 확인합니다.

```bash
.venv/bin/python scripts/gpu_smoke_test.py
```

정상이라면 RTX GPU 환경에서는 대략 아래 항목이 출력됩니다.

```text
device=cuda
cuda_available=True
gpu_name=NVIDIA GeForce RTX 5070
cuda_matmul_ok=(2048, 2048)
```

빠른 학습 경로 smoke test를 실행합니다.

```bash
.venv/bin/python main.py --mode test --no-plot --device auto
```

GPU를 반드시 쓰고 싶으면 `--device cuda`를 지정합니다.

```bash
SAC_CPU_WORKERS=8 .venv/bin/python main.py --mode test --no-plot --device cuda
```

현재 구조는 모델/텐서 연산은 GPU, 피처 생성과 환경/리플레이 버퍼 처리는 CPU를 쓰는 방식입니다. `SAC_CPU_WORKERS`는 OMP/MKL/OpenBLAS/PyTorch CPU thread 수를 맞추며, 기본값은 사용 가능한 CPU 수 기준 최대 8입니다.

## 자주 쓰는 실행

합성 데이터로 빠르게 전체 경로를 점검합니다.

```bash
.venv/bin/python main.py --mode test --no-plot --device auto
```

## AAPL 추천 실행

먼저 짧은 smoke test로 환경과 모델 경로를 확인합니다.

```bash
.venv/bin/python main.py --mode test --no-plot --device auto
```

AAPL 일봉 기준 추천 학습 명령입니다.

```bash
SAC_CPU_WORKERS=8 .venv/bin/python main.py \
  --mode train \
  --ticker AAPL \
  --start 2018-01-01 \
  --end 2024-12-31 \
  --interval 1d \
  --steps 200000 \
  --reward mixed \
  --best-metric sharpe \
  --device auto
```

Yahoo 다운로드가 일시적으로 실패할 때만 fallback을 허용합니다.

```bash
SAC_CPU_WORKERS=8 .venv/bin/python main.py \
  --mode train \
  --ticker AAPL \
  --start 2018-01-01 \
  --end 2024-12-31 \
  --interval 1d \
  --steps 200000 \
  --reward mixed \
  --best-metric sharpe \
  --device auto \
  --allow-synthetic-fallback
```

학습된 `best_sac` 체크포인트를 같은 기간 설정으로 평가합니다.

```bash
.venv/bin/python main.py \
  --mode eval \
  --ticker AAPL \
  --start 2018-01-01 \
  --end 2024-12-31 \
  --interval 1d \
  --load best_sac \
  --device auto
```

더 엄격하게 확인하려면 walk-forward를 실행합니다.

```bash
SAC_CPU_WORKERS=8 .venv/bin/python main.py \
  --mode walkforward \
  --ticker AAPL \
  --start 2018-01-01 \
  --end 2024-12-31 \
  --interval 1d \
  --steps 200000 \
  --wf-splits 5 \
  --wf-mode anchored \
  --device auto
```

## 학습

```bash
SAC_CPU_WORKERS=8 .venv/bin/python main.py \
  --mode train \
  --ticker AAPL \
  --start 2018-01-01 \
  --end 2024-12-31 \
  --steps 200000 \
  --device auto
```

## 평가

```bash
.venv/bin/python main.py \
  --mode eval \
  --ticker AAPL \
  --start 2018-01-01 \
  --end 2024-12-31 \
  --load best_sac \
  --device auto
```

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---:|---|
| `--ticker` | None | yfinance 종목. 없으면 합성 데이터 |
| `--interval` | `1d` | yfinance interval |
| `--test-ratio` | `0.2` | 마지막 구간 평가 비율 |
| `--steps` | `200000` | 학습 step |
| `--window-size` | `30` | 관측 윈도우 |
| `--episode-length` | `126` | 에피소드 길이 |
| `--reward` | `mixed` | `pnl`, `sharpe`, `sortino`, `mixed`, `dsr`, `dsr_cvar` |
| `--best-metric` | `sharpe` | `sharpe`, `alpha_vs_bh`, `calmar` |
| `--eval-episodes` | `5` | 평가 에피소드 수 |
| `--device` | `auto` | `auto`, `cuda`, `cpu` |
| `--cpu-workers` | 자동 | CPU thread/worker 수. 환경변수 `SAC_CPU_WORKERS`로도 설정 |
| `--no-plot` | off | 결과 이미지 저장 생략 |

Transformer 크기는 필요할 때만 조정합니다.

```bash
.venv/bin/python main.py --mode train --ticker AAPL --trans-d-model 96 --trans-heads 4 --trans-layers 3 --device auto
```

## 파일 구조

```text
main.py                    CLI와 단일 학습/평가 진입점
config.py                  환경, SAC, 피처, 학습 설정
env/trading_env.py         단일 종목 트레이딩 환경
networks/transformer_nets.py
agent/sac_agent.py         Transformer-SAC 에이전트
agent/replay_buffer.py     Uniform/PER replay buffer
training/trainer.py        학습 루프와 주기 평가
training/walk_forward.py   별도 walk-forward 검증
evaluation/evaluator.py    백테스트 지표와 그림
utils/features.py          기술적/매크로 피처
```
