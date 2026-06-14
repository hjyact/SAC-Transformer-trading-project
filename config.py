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
    # 60봉 (3개월) — 중기 추세 인식. 30봉은 국지적 노이즈에 취약.
    window_size: int        = 60

    # 포트폴리오
    initial_capital: float  = 1_000_000.0   # 초기 자본 (원)
    commission: float       = 0.00015        # 편도 수수료
    slippage: float         = 0.0001         # 슬리피지
    max_position: float     = 1.0            # 레버리지 없음 — 안정적 학습 우선

    # 보상 설계
    # mixed: alpha vs B&H 직접 보상 + Sharpe + inaction 패널티 — 시장 추종 강제
    # reward_scaling=0.1: mixed 모드에서 Sharpe 항 스케일
    # (pnl+15 조합은 현금 보유 Sharpe≈0이 되어 포지션 회피를 유발함)
    reward_type: str        = "mixed"
    reward_scaling: float   = 0.1
    risk_penalty: float     = 0.1
    drawdown_penalty: float = 0.3

    # 에피소드: 126봉(6개월) — 추세 학습에 충분한 기간, credit assignment 향상
    episode_length: int     = 126
    use_random_start: bool  = True
    # eval_random_start=True: 다양한 시작점에서 평균 → OOS 통계 신뢰성 ↑
    eval_random_start: bool = True

    # v9: 환경 레벨 거래 억제 추가 (gross alpha +0.22%에서 비용 절감으로 net positive 목표)
    action_change_penalty: float = 0.1

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
    actor_lr: float         = 1e-4          # v9: 느린 수렴 → 더 오랜 학습 기회
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
    gamma: float            = 0.99           # 긴 할인 horizon(≈100 step ≈5개월) → 추세 학습
    tau: float              = 0.005          # Soft target update 계수
    alpha: float            = 0.2            # 초기 엔트로피 온도
    auto_alpha: bool        = True           # 자동 엔트로피 조정 (SAC-v2)
    target_entropy: float   = -3.0          # 더 낮춰서 α온도 상승 방지 (결정론적 정책 허용)

    # 리플레이 버퍼
    buffer_size: int        = 300_000        # 100k→300k: 더 긴 역사 학습
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
    per_beta_anneal_steps: int = 500_000    # total_timesteps 와 매칭
    per_eps: float          = 1e-6
    n_step: int             = 3             # 5는 비정상 시계열에서 노이즈 증폭 확인됨

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
    critic_dropout: float   = 0.05
    utd_ratio: int          = 2             # 5→2: FPS 개선 + 학습 안정화 (과도한 업데이트 제거)

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
    caps_lambda_t: float    = 0.3           # v9: 약간 강화 (거래 횟수 감소 목표)
    caps_lambda_s: float    = 0.0           # S4RL state_aug 으로 대체 (sac_cfg.use_state_aug)
    caps_spatial_sigma: float = 0.05

    # ── Primacy Bias Reset (Nikishin ICML 2022 / BBF Schwarzer NeurIPS 2023
    #    / Shrink&Perturb Ash NeurIPS 2020)
    reset_interval: int     = 0             # 0=off: 매번 리셋이 eval 성능을 되돌리는 패턴 확인 → 비활성화
    reset_mode: str         = "shrink_perturb"
    reset_actor: bool       = True
    reset_optimizer: bool   = True
    reset_shrink_factor: float = 0.8
    reset_perturb_sigma: float = 0.02

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
    trans_d_model: int            = 128      # v8: 대형 모델 복원 (v2 기준 최고 alpha 달성)
    trans_n_heads: int            = 8        # 8 heads: 더 다양한 어텐션 패턴
    trans_n_layers: int           = 3        # 3 layers: 깊은 계층 패턴
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

    # ── Momentum Features (Jegadeesh & Titman 1993)
    # ROC 다중 주기: 1M(21), 3M(63), 6M(126), 12M(252) 수익률 모멘텀.
    # 52주 고가/저가 근접도, RSI slope. 가장 검증된 알파 팩터.
    use_momentum: bool           = True
    mom_windows: list            = field(default_factory=lambda: [21, 63, 126, 252])

    # ── Macro Features (Welch & Goyal 2008)
    # VIX, 10Y/3M 금리, yield curve slope. 시장 레짐 정보 추가.
    use_macro: bool              = True


# ── 학습 설정 ──────────────────────────────────────────
@dataclass
class TrainConfig:
    total_timesteps: int    = 500_000
    eval_interval: int      = 5_000
    # eval_random_start=False이면 동일 구간을 반복 → 3회로 충분
    eval_episodes: int      = 3
    save_interval: int      = 10_000
    log_interval: int       = 1_000
    seed: int               = 42

    best_metric: str        = "alpha_vs_bh" # B&H 초과 수익 최적화 (시장 추종 아님)
    # -0.05: B&H 5% 이내 모델도 저장 → 초기 학습 중 best 공백 방지
    best_min_margin: float  = -0.02

    # 0 = early stopping 비활성화
    early_stop_patience: int = 0


env_cfg     = EnvConfig()
sac_cfg     = SACConfig()
feat_cfg    = FeatureConfig()
train_cfg   = TrainConfig()
