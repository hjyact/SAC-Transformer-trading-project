"""
config.py — SAC 트레이딩 시스템 전역 설정

참고 이론:
  - SAC (Haarnoja et al., 2018): 엔트로피 최대화 + Off-policy Actor-Critic
  - Kelly Criterion: 최적 포지션 사이징
  - Markowitz MPT: 리스크-수익 트레이드오프
  - Temporal Difference: TD(λ) 기반 가치 추정
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
import torch

ROOT  = Path(__file__).parent
CKPT_DIR   = ROOT / "checkpoints";  CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR    = ROOT / "logs";         LOG_DIR.mkdir(exist_ok=True)
RESULT_DIR = ROOT / "results";      RESULT_DIR.mkdir(exist_ok=True)
MPL_CACHE_DIR = ROOT / ".cache" / "matplotlib"; MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

def _available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


def _default_cpu_workers() -> int:
    return max(1, min(8, _available_cpu_count()))


CPU_WORKERS = int(os.getenv("SAC_CPU_WORKERS", str(_default_cpu_workers())))
DEVICE = os.getenv("SAC_DEVICE", "auto").lower()
if DEVICE == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def configure_runtime(device: str | None = None, cpu_workers: int | None = None) -> dict:
    """Configure Linux-friendly CPU threading and GPU backend settings."""
    global DEVICE, CPU_WORKERS

    if cpu_workers is not None:
        CPU_WORKERS = max(1, int(cpu_workers))
    else:
        CPU_WORKERS = int(os.getenv("SAC_CPU_WORKERS", str(CPU_WORKERS)))

    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, str(CPU_WORKERS))

    torch.set_num_threads(CPU_WORKERS)
    try:
        torch.set_num_interop_threads(max(1, min(4, CPU_WORKERS)))
    except RuntimeError:
        pass

    requested = (device or os.getenv("SAC_DEVICE", "auto")).lower()
    if requested == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested == "cuda" and not torch.cuda.is_available():
        DEVICE = "cpu"
    elif requested in {"cuda", "cpu"}:
        DEVICE = requested
    else:
        raise ValueError(f"지원하지 않는 device: {requested}")

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    return {
        "device": DEVICE,
        "cpu_workers": CPU_WORKERS,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


RUNTIME_INFO = configure_runtime()


# ── 환경 설정 ──────────────────────────────────────────
@dataclass
class EnvConfig:
    # 관측 윈도우 (과거 몇 봉을 state로 볼 것인가)
    window_size: int        = 30

    # 포트폴리오
    initial_capital: float  = 1_000_000.0   # 초기 자본 (원)
    commission: float       = 0.00015        # 편도 수수료
    slippage: float         = 0.0001         # 슬리피지
    max_position: float     = 1.0            # 최대 포지션 (자본의 100%)

    # 보상 설계
    reward_type: str        = "sharpe"       # "pnl" | "sharpe" | "sortino" | "mixed"
    # reward_scaling=0.1 권장 (Q값/π_loss 폭주 방지):
    #   sharpe reward 는 -5~+5 clip 인데 step 평균 ≈0.3 → 252 step 누적 ≈75 →
    #   Q값이 너무 커져 SAC entropy term (α·log_p) 이 묻히고 정책이 Q-greedy 로 빠짐.
    #   0.1 배 스케일이면 Q ≈ 5~15 정도로 entropy/policy balance 회복.
    reward_scaling: float   = 0.1
    risk_penalty: float     = 0.1            # 과도한 리스크 패널티 계수
    drawdown_penalty: float = 0.5            # MDD 패널티

    # 에피소드
    # 126 (반년) 권장 — episode 다양성 ↑, eval random-start 와 결합 시 OOS 추정 안정.
    episode_length: int     = 126
    use_random_start: bool  = True           # 랜덤 시작점 (과적합 방지)
    # Eval 시점에도 랜덤 시작점으로 N 회 평가 → OOS 통계 신뢰성 ↑ (시간 OOS 유지).
    # eval_episodes (train_cfg) 만큼 다양한 시작점에서 평가하고 평균/표본 통계 산출.
    eval_random_start: bool = True

    # 행동 변화 패널티 (Mysore CAPS 2021 의 env-level 변형)
    # 0.5~1.0 권장 — actor loss 의 λ_T 가 Q 값에 묻히는 문제 우회.
    # 과매매 (현실에서 매일 매매) 를 reward 수준에서 직접 억제.
    action_change_penalty: float = 0.5

    # ── Differential Sharpe Ratio (Moody & Saffell 2001)
    # 매 step 의 Sharpe 증분을 보상으로 — risk-adjusted return 직접 최적화
    # reward_type="dsr" 또는 "dsr_cvar" 일 때 사용
    dsr_eta: float          = 0.01      # 지수 가중 이동평균 적응 속도

    # ── CVaR Risk Penalty (Coache & Jaimungal 2021)
    # 하위 5% 분위수 평균 패널티 — 꼬리위험 직접 회피
    cvar_alpha: float       = 0.05      # 5% 분위수
    cvar_window: int        = 50        # rolling 윈도우
    cvar_lambda: float      = 1.0       # 패널티 가중치

    # ── Train-time Data Augmentation
    # 학습 모드 (mode="train") 의 _get_obs() / reset() 에서만 적용.
    # eval/실거래 경로는 영향 X. 모두 ON 이 기본 (--no-* 플래그로 끄기).
    #
    #  ① Observation Jittering (Iwana 2021, Bishop 1995):
    #     표준화된 obs window 에 작은 가우시안 노이즈 추가.
    #  ② Magnitude Warping (Iwana 2021, Wen et al. IJCAI 2021):
    #     에피소드 단위로 cubic-spline 곡선 (≈1±σ) 샘플링 → window 에 곱셈.
    #     같은 시점이라도 매번 약간 다른 진폭으로 학습 → 일반화.
    #  ③ Domain Randomization (Tobin 2017 IROS, Peng 2018 ICRA):
    #     매 에피소드 commission/slippage 를 베이스 ± domain_rand_pct 범위에서
    #     랜덤 추출 → 거래비용에 robust 한 정책 학습.
    use_obs_jitter: bool      = True
    obs_jitter_sigma: float   = 0.01    # 표준화된 obs (~N(0,1)) 기준 std

    use_magnitude_warp: bool  = True
    mag_warp_sigma: float     = 0.05    # 곡선 진폭 (1 ± σ)
    mag_warp_knots: int       = 4       # cubic-spline knot 수

    use_domain_rand: bool     = True
    domain_rand_pct: float    = 0.3     # commission/slippage 변동 비율 (±30%)


# ── SAC 하이퍼파라미터 ─────────────────────────────────
@dataclass
class SACConfig:
    # 네트워크
    hidden_dims: list       = field(default_factory=lambda: [256, 256])
    activation: str         = "relu"         # "relu" | "tanh" | "elu"

    # 학습률
    actor_lr: float         = 3e-4
    critic_lr: float        = 3e-4
    alpha_lr: float         = 3e-4           # 엔트로피 온도 자동 조정
    # L2 regularization (AdamW). 0 = vanilla Adam 과 동일.
    # 작은 데이터셋에서 critic 가중치 발산 억제.
    # Loshchilov & Hutter, ICLR 2019 — Decoupled Weight Decay Regularization.
    weight_decay: float     = 1e-4

    # SAC 핵심 파라미터
    # gamma=0.97 권장 — 0.99 는 Q ≈ E[r]/(1-γ) = 100·E[r] 까지 누적 →
    # 음수 r 영역에서 Q 폭주 (π_loss=+99 의 원인). 0.97 은 효율적 horizon ≈ 33 step
    # (≈ 1.5개월) — 단기 트레이딩 의사결정에 더 적합 (Andrychowicz 2021).
    gamma: float            = 0.97
    tau: float              = 0.005          # Soft target update 계수
    alpha: float            = 0.2            # 초기 엔트로피 온도
    auto_alpha: bool        = True           # 자동 엔트로피 조정 (SAC-v2)
    target_entropy: float   = -1.0           # 목표 엔트로피 (-action_dim)

    # 리플레이 버퍼
    buffer_size: int        = 100_000
    batch_size: int         = 256
    min_replay_size: int    = 1_000          # 학습 시작 전 워밍업

    # 학습
    gradient_steps: int     = 1             # 환경 스텝당 gradient 업데이트 횟수
    target_update_interval: int = 1         # target network 업데이트 주기
    grad_clip: float        = 1.0           # gradient clipping

    # ── PER (Schaul 2015) + N-step (Sutton 1988)
    use_per: bool           = True          # Prioritized Experience Replay
    per_alpha: float        = 0.6           # priority exponent
    per_beta_start: float   = 0.4           # IS weight 어닐링 시작
    per_beta_end: float     = 1.0           # IS weight 어닐링 종료
    per_beta_anneal_steps: int = 200_000    # β 어닐링 구간 (total_timesteps 와 매칭)
    per_eps: float          = 1e-6          # |δ|=0 회피용 작은 양수
    n_step: int             = 3             # N-step return (bias-variance trade-off)

    # ── 보상 정규화 (Welford 1962)
    # ⚠ 주의: normalize_reward=True 와 env_cfg.reward_scaling 은 충돌함.
    #   Welford 가 reward 를 σ=1 분포로 다시 표준화 → env 의 reward_scaling
    #   효과가 무력화되고 Q 가 다시 폭주 (관측: π_loss = +99).
    #   기본 False — env 의 reward_scaling 만으로 스케일 조정.
    normalize_reward: bool  = False
    reward_norm_center: bool= False         # True: (r-μ)/σ, False: r/σ
    reward_norm_clip: float = 10.0          # 정규화 후 클리핑
    # ── DroQ (Hiraoka et al., ICLR 2022)
    #    Q 네트워크에 작은 dropout + 높은 UTD ratio 로 Q 과대추정 억제
    use_droq: bool          = True
    critic_dropout: float   = 0.05          # critic 만 (actor 는 결정론 유지)
    utd_ratio: int          = 5             # env step 당 critic 업데이트 횟수 (G)
                                            # DroQ 논문 권장 20, 트레이딩에선 3-5 권장

    # ── LAP — Loss-Adjusted Prioritization (Fujimoto et al., NeurIPS 2020)
    #    PER 의 priority 폭주 방지: priority = max(|δ|, λ_min) + Huber 손실
    use_lap: bool           = True
    lap_min_priority: float = 1.0           # priority 하한 (정규화)
    huber_delta: float      = 1.0           # Huber loss 의 transition point

    # ── CAPS — Action Policy Smoothness (Mysore et al., NeurIPS 2021)
    #    L_T : ‖π(s_t) - π(s_{t+1})‖  → 연속 시점 행동 부드럽게 (과매매 억제)
    #    L_S : ‖π(s) - π(s + ε)‖     → 상태 섭동에 대한 강건성
    # λ_T=0.2 권장 — 0.05 는 Q 값에 묻혀 효과 없음. env-level action_change_penalty
    # 와 병행 (env 는 reward 즉답, actor-level 은 정책 부드러움).
    caps_lambda_t: float    = 0.2
    caps_lambda_s: float    = 0.0           # S4RL state_aug 으로 대체 (sac_cfg.use_state_aug)
    caps_spatial_sigma: float = 0.05

    # ── Primacy Bias Reset (Nikishin ICML 2022 / BBF Schwarzer NeurIPS 2023
    #    / Shrink&Perturb Ash NeurIPS 2020)
    reset_interval: int     = 0             # 0 = off, 예: 50_000 (gradient 업데이트 기준)
    reset_mode: str         = "shrink_perturb"  # "head" | "full_critic" | "shrink_perturb"
    reset_actor: bool       = True          # BBF: actor 도 함께 리셋
    reset_optimizer: bool   = True          # optimizer state 초기화
    reset_shrink_factor: float = 0.5        # Shrink&Perturb: θ ← 0.5·θ + noise
    reset_perturb_sigma: float = 0.02       # 추가 노이즈 std

    # ── S4RL — State Augmentation (Sinha & Garg, CoRL 2022)
    # Critic 학습 시 batch obs/next_obs 에 작은 가우시안 노이즈 추가.
    # 논문 §4: S4RL-N (Gaussian) 가 가장 단순하면서 가장 강건한 variant.
    # CAPS-spatial 과의 차이:
    #   - CAPS spatial λ_S : actor 만, 한 batch 내 clean+perturb 두 forward
    #   - S4RL state-aug  : critic 입력 자체를 한번 perturb (target/online 동일)
    # 두 기법은 보완 가능. 본 시스템은 S4RL 을 기본 ON, CAPS_S 는 0 으로 비활성.
    use_state_aug: bool         = True
    state_aug_sigma: float      = 0.01      # 표준화된 obs 기준 std

    # ── Transformer Actor/Critic (networks/transformer_nets.py)
    # 단일 모델: 시계열 transformer 기반 SAC. 참고 논문:
    #   • GTrXL (Parisotto 2020 ICML) — online RL 안정화 (pre-LN + gated residual)
    #   • PatchTST (Nie 2023 ICLR)   — 시계열 토큰화 + channel-independent
    #   • RevIN (Kim 2022 ICLR)      — instance-wise normalization (비정상성 강건)
    use_transformer: bool         = True
    trans_d_model: int            = 64       # 모델 차원
    trans_n_heads: int            = 4        # multi-head attention
    trans_n_layers: int           = 2        # encoder layer 수
    trans_dropout: float          = 0.1      # attention/FFN dropout
    trans_use_revin: bool         = True     # RevIN (분포 shift 강건성)
    trans_use_gtrxl_gate: bool    = True     # GTrXL gated residual (identity-init)
    trans_share_critic_encoder: bool = True  # q1/q2 encoder 공유 (메모리 절약)

    # ── BC Regularization (Fujimoto & Gu, NeurIPS 2021 — TD3+BC)
    # Actor loss 에 ‖π(s) - a_buffer‖² 항 추가 → 학습 데이터 분포 벗어나지 않음
    # 과적합 / OOS 발산 억제. Offline RL 의 핵심 트릭.
    use_bc_regularization: bool = False     # 기본 OFF (호환성)
    bc_lambda: float            = 2.5       # 논문 권장 λ = 2.5
    bc_norm_q: bool             = True      # Q값으로 λ 스케일링 (논문 §3.2)


# ── 피처 설정 ──────────────────────────────────────────
@dataclass
class FeatureConfig:
    # 기술적 지표 파라미터
    rsi_period: int         = 14
    macd_fast: int          = 12
    macd_slow: int          = 26
    macd_signal: int        = 9
    bb_period: int          = 20
    atr_period: int         = 14
    vol_windows: list       = field(default_factory=lambda: [5, 10, 20])
    ma_windows: list        = field(default_factory=lambda: [5, 10, 20, 60])

    # 추가 이론 기반 피처
    use_hurst: bool         = True    # Hurst 지수 (장기기억 / 평균회귀 판별)
    use_microstructure: bool= True    # 시장 미시구조 (bid-ask spread proxy)
    use_regime: bool        = True    # HMM 기반 레짐 피처 (변동성 레짐)

    # ── Fractional Differentiation (López de Prado 2018 Ch.5)
    # log_ret (1차 차분) 은 메모리 완전 소실 → 분수차분 d∈(0,1) 로 정상성+메모리 양립
    # threshold=1e-4 → d=0.4 윈도우 ~280봉 (1년), d=0.5 윈도우 ~200봉 (0.8년)
    use_frac_diff: bool          = True
    frac_diff_d_values: list     = field(default_factory=lambda: [0.4, 0.5])
    frac_diff_threshold: float   = 1e-4

    # ── Macro Features (Welch & Goyal 2008)
    # VIX, 10Y/3M 금리, yield curve slope. 시장 레짐 정보 추가.
    use_macro: bool              = True


# ── 학습 설정 ──────────────────────────────────────────
@dataclass
class TrainConfig:
    total_timesteps: int    = 200_000
    eval_interval: int      = 5_000
    eval_episodes: int      = 5      # 다양한 시작점에서 5회 평가 (env_cfg.eval_random_start=True 가정)
    save_interval: int      = 10_000
    log_interval: int       = 1_000
    seed: int               = 42

    # ── Best 모델 선택 기준
    # "sharpe"      : eval Sharpe (B&H 무시) — 강세장에서도 의미 있음 (위험조정)
    # "alpha_vs_bh" : α vs B&H — 시장 대비 초과수익 (강세장에선 SAC 가 헷지·현금·숏을
    #                  섞으면 구조적으로 못 이김 → 함정 주의)
    # "calmar"      : Calmar = total_return / |MDD| (수익/위험 비율)
    #
    # 기본 sharpe — 합성/실제 데이터 모두에서 모델의 *학습 자체* 진단에 적합.
    # 실거래 의사결정이 목적이면 학습 끝난 후 alpha_vs_bh / calmar 도 함께 확인.
    best_metric: str        = "sharpe"
    # Best 갱신의 최소 마진. metric < best_min_margin 이면 best 갱신 안 함.
    # sharpe 기준 0.0 = 음수 Sharpe 모델 차단 (random 보다 못한 모델 방지).
    # alpha_vs_bh 로 바꾸면 0.0 = B&H 못 이기는 모델 차단.
    best_min_margin: float  = 0.0

    # ── Early Stopping
    # eval 가 best 갱신 안 되면 patience 후 학습 자동 중단.
    # patience=10 → eval_interval×10 = 50k step 동안 갱신 없으면 중단.
    early_stop_patience: int = 10


env_cfg     = EnvConfig()
sac_cfg     = SACConfig()
feat_cfg    = FeatureConfig()
train_cfg   = TrainConfig()
