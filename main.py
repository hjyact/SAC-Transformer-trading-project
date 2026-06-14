"""
main.py — SAC 트레이딩 시스템 진입점

사용법:
    python main.py                              # 기본 실행 (합성 데이터)
    python main.py --ticker AAPL --mode train   # 실제 데이터 학습
    python main.py --mode eval --load best_sac  # 저장 모델 평가
    python main.py --reward mixed --steps 500000

참고 이론 요약:
  ┌─────────────────────────────────────────────────────────┐
  │ SAC (Haarnoja 2018)    : 엔트로피 최대화 Off-policy RL  │
  │ Twin Critics (TD3)     : Q값 과대추정 방지              │
  │ Squashed Gaussian      : 연속 행동 경계 처리            │
  │ Auto-α (SAC v2)        : 탐험-활용 자동 균형            │
  │ Sharpe 보상            : 위험 조정 수익 최적화          │
  │ Kelly Criterion        : 포지션 사이징 이론             │
  │ Hurst Exponent         : 시장 레짐 감지                 │
  │ Garman-Klass Vol       : OHLC 기반 효율적 변동성 추정   │
  │ Amihud Illiquidity     : 유동성 리스크 피처             │
  │ Roll Spread            : 시장 미시구조 피처             │
  └─────────────────────────────────────────────────────────┘
"""

import argparse
import logging
import os

from config import configure_runtime

# Apply default thread/env settings before importing numpy/pandas.
configure_runtime()

import numpy as np
import pandas as pd
import torch

from config import (
    env_cfg, sac_cfg, feat_cfg, train_cfg, DEVICE, CPU_WORKERS,
)
from utils.features import build_all_features, download_macro_features, merge_macro_features
from utils.plot_style import apply_korean_font
from env.trading_env import TradingEnv
from agent.sac_agent import SACAgent
from agent.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from training.trainer import SACTrainer
from evaluation.evaluator import run_evaluation, plot_training_curves, plot_backtest

# matplotlib 한글 폰트 일괄 설정 (마이너스 깨짐 포함)
apply_korean_font()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── 데이터 준비 ────────────────────────────────────────

def prepare_data(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    데이터 로드 → 피처 생성 → 훈련/테스트 분리.
    시간 순서 유지, look-ahead bias 없음.
    """
    if args.use_synthetic or args.ticker is None:
        if args.ticker is None and not args.use_synthetic:
            logger.info("ticker 미지정: 합성 데이터 생성 중...")
        else:
            logger.info("합성 데이터 생성 중...")
        price_df = _make_synthetic_data(n=2000, seed=42)
    else:
        try:
            import yfinance as yf
            logger.info(f"데이터 다운로드: {args.ticker}")
            raw = yf.download(args.ticker, start=args.start, end=args.end,
                              interval=args.interval, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            price_df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if price_df.empty:
                raise ValueError("downloaded data is empty")
        except Exception as e:
            if not getattr(args, "allow_synthetic_fallback", False):
                raise RuntimeError(
                    f"Yahoo 데이터 다운로드 실패: ticker={args.ticker}, "
                    f"start={args.start}, end={args.end}, interval={args.interval}. "
                    "--allow-synthetic-fallback 를 주면 실패 시 합성 데이터로 대체합니다."
                ) from e
            logger.warning(f"다운로드 실패 ({e}), 합성 데이터 사용")
            price_df = _make_synthetic_data(n=2000, seed=42)

    logger.info(f"원본 데이터: {len(price_df)}행")

    # Macro features 병합 (use_macro=True 일 때, Welch & Goyal 2008)
    if getattr(feat_cfg, "use_macro", False) and not args.use_synthetic and args.ticker:
        try:
            macro_df = download_macro_features(args.start, args.end, args.interval)
            if macro_df is not None:
                price_df = merge_macro_features(price_df, macro_df)
                logger.info(f"Macro features 병합: {[c.replace('_macro_', '') for c in price_df.columns if c.startswith('_macro_')]}")
        except Exception as e:
            logger.warning(f"Macro features 로드 실패 ({e}) — 진행")

    # 피처 엔지니어링
    logger.info("피처 엔지니어링 중...")
    feat_df = build_all_features(price_df, feat_cfg)

    # 유효 행 필터링
    common = feat_df.index.intersection(price_df.index)
    feat_df  = feat_df.loc[common].dropna()
    price_df = price_df.loc[feat_df.index]

    logger.info(f"유효 데이터: {len(feat_df)}행, 피처: {feat_df.shape[1]}개")

    # 훈련/테스트 분리 (시간 순서 유지)
    split = int(len(feat_df) * (1 - args.test_ratio))
    train_feat  = feat_df.iloc[:split]
    train_price = price_df.iloc[:split]
    test_feat   = feat_df.iloc[split:]
    test_price  = price_df.iloc[split:]

    logger.info(f"훈련: {len(train_feat)}행 | 테스트: {len(test_feat)}행")

    return (train_feat, train_price), (test_feat, test_price)


def _make_synthetic_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    GBM + 레짐 전환 합성 데이터.
    실제 주가의 변동성 클러스터링 및 추세 변화를 모방.
    """
    np.random.seed(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")

    # GARCH-like 변동성 클러스터링
    vol = np.zeros(n)
    vol[0] = 0.01
    for i in range(1, n):
        shock = np.random.randn()
        vol[i] = np.sqrt(0.00001 + 0.09 * (vol[i-1]*shock)**2 + 0.90 * vol[i-1]**2)

    ret = np.random.randn(n) * vol
    # 추세 레짐 (bull/bear)
    regime = np.sin(np.linspace(0, 4*np.pi, n)) * 0.0002
    ret   += regime

    close = 100 * np.exp(ret.cumsum())
    hi_lo_spread = np.abs(np.random.randn(n)) * vol * close
    open_  = close * np.exp(-ret * np.random.uniform(0.3, 0.7, n))
    high   = close + hi_lo_spread * 0.5
    low    = close - hi_lo_spread * 0.5
    volume = np.random.lognormal(15, 1, n)

    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)


# ── 환경 및 에이전트 구축 ──────────────────────────────

def _apply_args_to_configs(args) -> None:
    """CLI 옵션을 env_cfg / sac_cfg 에 반영."""
    global DEVICE, CPU_WORKERS

    if hasattr(args, "cpu_workers") or hasattr(args, "device"):
        import config as _config
        _config.configure_runtime(
            device=getattr(args, "device", None),
            cpu_workers=getattr(args, "cpu_workers", None),
        )
        DEVICE = _config.DEVICE
        CPU_WORKERS = _config.CPU_WORKERS

    env_cfg.reward_type     = args.reward
    env_cfg.episode_length  = args.episode_length
    env_cfg.window_size     = args.window_size
    if hasattr(args, "action_change_penalty"):
        env_cfg.action_change_penalty = float(args.action_change_penalty)
    if hasattr(args, "reward_scaling"):
        env_cfg.reward_scaling = float(args.reward_scaling)
    if hasattr(args, "eval_random_start"):
        env_cfg.eval_random_start = bool(args.eval_random_start)

    sac_cfg.use_transformer = True
    if hasattr(args, "use_per"):
        sac_cfg.use_per = bool(args.use_per)
    if hasattr(args, "n_step"):
        sac_cfg.n_step  = int(args.n_step)
    if hasattr(args, "critic_dropout"):
        sac_cfg.critic_dropout = float(args.critic_dropout)
    if hasattr(args, "weight_decay"):
        sac_cfg.weight_decay = float(args.weight_decay)
    if hasattr(args, "gamma"):
        sac_cfg.gamma = float(args.gamma)
    if hasattr(args, "normalize_reward"):
        sac_cfg.normalize_reward = bool(args.normalize_reward)
    if hasattr(args, "use_droq"):
        sac_cfg.use_droq = bool(args.use_droq)
    if hasattr(args, "utd_ratio"):
        sac_cfg.utd_ratio = int(args.utd_ratio)
    if hasattr(args, "use_lap"):
        sac_cfg.use_lap = bool(args.use_lap)
    if hasattr(args, "caps_lambda_t"):
        sac_cfg.caps_lambda_t = float(args.caps_lambda_t)
    if hasattr(args, "reset_interval"):
        sac_cfg.reset_interval = int(args.reset_interval)
    if hasattr(args, "reset_mode"):
        sac_cfg.reset_mode = str(args.reset_mode)
    if hasattr(args, "reset_actor"):
        sac_cfg.reset_actor = bool(args.reset_actor)

    # ── DSR / CVaR (env_cfg)
    if hasattr(args, "dsr_eta"):
        env_cfg.dsr_eta = float(args.dsr_eta)
    if hasattr(args, "cvar_alpha"):
        env_cfg.cvar_alpha = float(args.cvar_alpha)
    if hasattr(args, "cvar_window"):
        env_cfg.cvar_window = int(args.cvar_window)
    if hasattr(args, "cvar_lambda"):
        env_cfg.cvar_lambda = float(args.cvar_lambda)

    # ── BC Regularization (sac_cfg)
    if hasattr(args, "use_bc_regularization"):
        sac_cfg.use_bc_regularization = bool(args.use_bc_regularization)
    if hasattr(args, "bc_lambda"):
        sac_cfg.bc_lambda = float(args.bc_lambda)

    # ── Feature flags (feat_cfg)
    if hasattr(args, "use_macro"):
        feat_cfg.use_macro = bool(args.use_macro)
    if hasattr(args, "use_frac_diff"):
        feat_cfg.use_frac_diff = bool(args.use_frac_diff)
    if hasattr(args, "use_momentum"):
        feat_cfg.use_momentum = bool(args.use_momentum)

    # ── Train-time Data Augmentation (env_cfg)
    if hasattr(args, "use_obs_jitter"):
        env_cfg.use_obs_jitter = bool(args.use_obs_jitter)
    if hasattr(args, "obs_jitter_sigma"):
        env_cfg.obs_jitter_sigma = float(args.obs_jitter_sigma)
    if hasattr(args, "use_magnitude_warp"):
        env_cfg.use_magnitude_warp = bool(args.use_magnitude_warp)
    if hasattr(args, "mag_warp_sigma"):
        env_cfg.mag_warp_sigma = float(args.mag_warp_sigma)
    if hasattr(args, "use_domain_rand"):
        env_cfg.use_domain_rand = bool(args.use_domain_rand)
    if hasattr(args, "domain_rand_pct"):
        env_cfg.domain_rand_pct = float(args.domain_rand_pct)

    # ── S4RL State Augmentation (sac_cfg)
    if hasattr(args, "use_state_aug"):
        sac_cfg.use_state_aug = bool(args.use_state_aug)
    if hasattr(args, "state_aug_sigma"):
        sac_cfg.state_aug_sigma = float(args.state_aug_sigma)

    # ── Transformer Actor/Critic: 단일 모델로 고정
    sac_cfg.use_transformer = True
    if hasattr(args, "trans_d_model"):
        sac_cfg.trans_d_model = int(args.trans_d_model)
    if hasattr(args, "trans_n_heads"):
        sac_cfg.trans_n_heads = int(args.trans_n_heads)
    if hasattr(args, "trans_n_layers"):
        sac_cfg.trans_n_layers = int(args.trans_n_layers)
    if hasattr(args, "trans_dropout"):
        sac_cfg.trans_dropout = float(args.trans_dropout)
    if hasattr(args, "trans_use_revin"):
        sac_cfg.trans_use_revin = bool(args.trans_use_revin)
    if hasattr(args, "trans_use_gtrxl_gate"):
        sac_cfg.trans_use_gtrxl_gate = bool(args.trans_use_gtrxl_gate)

    # ── Train cfg (best metric, early stop, eval)
    if hasattr(args, "best_metric"):
        train_cfg.best_metric = str(args.best_metric)
    if hasattr(args, "best_min_margin"):
        train_cfg.best_min_margin = float(args.best_min_margin)
    if hasattr(args, "early_stop_patience"):
        train_cfg.early_stop_patience = int(args.early_stop_patience)
    if hasattr(args, "eval_episodes"):
        train_cfg.eval_episodes = int(args.eval_episodes)


def _create_agent_and_buffer(
    obs_dim: int,
    action_dim: int,
    obs_meta: dict | None = None,
) -> tuple[SACAgent, object]:
    """SAC agent + replay buffer 생성. _apply_args_to_configs() 호출 후 사용.

    obs_meta: transformer 사용 시 필수. {"window":W, "n_feat":F, "port_dim":P}.
    """
    agent = SACAgent(obs_dim, action_dim, sac_cfg, DEVICE, obs_meta=obs_meta)

    if sac_cfg.use_per:
        buffer = PrioritizedReplayBuffer(
            obs_dim, action_dim, sac_cfg.buffer_size,
            n_step           = sac_cfg.n_step,
            gamma            = sac_cfg.gamma,
            alpha            = sac_cfg.per_alpha,
            beta_start       = sac_cfg.per_beta_start,
            beta_end         = sac_cfg.per_beta_end,
            beta_anneal_steps= sac_cfg.per_beta_anneal_steps,
            eps              = sac_cfg.per_eps,
            use_lap          = sac_cfg.use_lap,
            lap_min_priority = sac_cfg.lap_min_priority,
        )
        per_kind = "PER+LAP" if sac_cfg.use_lap else "PER"
        logger.info(
            f"버퍼: {per_kind} (α={sac_cfg.per_alpha}, β: "
            f"{sac_cfg.per_beta_start}→{sac_cfg.per_beta_end} / "
            f"{sac_cfg.per_beta_anneal_steps:,} steps), n-step={sac_cfg.n_step}"
        )
    else:
        buffer = ReplayBuffer(obs_dim, action_dim, sac_cfg.buffer_size)
        logger.info("버퍼: Uniform Replay (PER 비활성)")

    logger.info(
        f"런타임: device={DEVICE} | cpu_workers={CPU_WORKERS} | "
        f"torch_threads={torch.get_num_threads()} | cuda={torch.cuda.is_available()}"
    )
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info(
        f"기법: DroQ={sac_cfg.use_droq}(p={sac_cfg.critic_dropout}, UTD={sac_cfg.utd_ratio}) | "
        f"LAP={sac_cfg.use_lap} | "
        f"CAPS(λ_t={sac_cfg.caps_lambda_t}, λ_s={sac_cfg.caps_lambda_s}) | "
        f"Reset={sac_cfg.reset_interval}"
    )
    # Train-time data augmentation (overfit 방지 — Iwana 2021, Tobin 2017, Sinha 2022)
    logger.info(
        f"증강(train-only): "
        f"S4RL={getattr(sac_cfg, 'use_state_aug', False)}"
        f"(σ={getattr(sac_cfg, 'state_aug_sigma', 0):.3f}) | "
        f"Jitter={getattr(env_cfg, 'use_obs_jitter', False)}"
        f"(σ={getattr(env_cfg, 'obs_jitter_sigma', 0):.3f}) | "
        f"MagWarp={getattr(env_cfg, 'use_magnitude_warp', False)}"
        f"(σ={getattr(env_cfg, 'mag_warp_sigma', 0):.2f}) | "
        f"DomainRand={getattr(env_cfg, 'use_domain_rand', False)}"
        f"(±{getattr(env_cfg, 'domain_rand_pct', 0):.0%})"
    )
    # 보상/평가 진단 (Welford ↔ reward_scaling 충돌, best_metric 함정 회피)
    logger.info(
        f"보상/평가: "
        f"reward_scaling={getattr(env_cfg, 'reward_scaling', 1.0):.3f} | "
        f"normalize_reward={getattr(sac_cfg, 'normalize_reward', False)} | "
        f"γ={sac_cfg.gamma:.3f} (eff horizon ≈ {int(1/(1-sac_cfg.gamma))} step) | "
        f"best_metric={getattr(train_cfg, 'best_metric', 'sharpe')}"
        f"(margin={getattr(train_cfg, 'best_min_margin', 0):.2f})"
    )
    n_p = sum(p.numel() for p in agent.actor.parameters()) + \
          sum(p.numel() for p in agent.critic.parameters())
    logger.info(
        f"백본: Transformer (GTrXL+PatchTST+RevIN) — "
        f"d_model={sac_cfg.trans_d_model}, heads={sac_cfg.trans_n_heads}, "
        f"layers={sac_cfg.trans_n_layers}, dropout={sac_cfg.trans_dropout}, "
        f"RevIN={sac_cfg.trans_use_revin}, GTrXL-gate={sac_cfg.trans_use_gtrxl_gate} | "
        f"총 파라미터(actor+critic): {n_p:,}"
    )

    return agent, buffer


def build_env_and_agent(
    train_data: tuple,
    test_data: tuple,
    args,
) -> tuple[TradingEnv, TradingEnv, SACAgent, ReplayBuffer]:
    """단일 종목 학습용 환경 + 에이전트 구축."""
    train_feat, train_price = train_data
    test_feat,  test_price  = test_data

    _apply_args_to_configs(args)

    train_env = TradingEnv(train_feat, train_price, env_cfg, mode="train")
    # eval 환경에 train 통계 주입 → test 셋 mean/std 사용으로 인한 look-ahead bias 차단
    eval_env  = TradingEnv(
        test_feat, test_price, env_cfg, mode="eval",
        feat_mean=train_env._feat_mean.copy(),
        feat_std =train_env._feat_std.copy(),
    )

    obs_dim    = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    logger.info(f"관측 차원: {obs_dim} | 행동 차원: {action_dim} | Device: {DEVICE}")

    obs_meta = {
        "window":   train_env.cfg.window_size,
        "n_feat":   train_env.n_feat,
        "port_dim": 4,    # compute_portfolio_features 출력 차원
    }
    agent, buffer = _create_agent_and_buffer(obs_dim, action_dim, obs_meta=obs_meta)
    return train_env, eval_env, agent, buffer


# ── 학습 ──────────────────────────────────────────────

def _final_eval_and_plot(agent, eval_env, args, default_name: str = "SAC_Best"):
    """
    학습 후 / eval 모드의 최종 평가 + plot.
    단일 종목 평가 → 1장의 backtest 그림.
    """
    result = run_evaluation(agent, eval_env, n_episodes=1, name=default_name)
    if not args.no_plot:
        plot_backtest(result, benchmark=result.get("benchmark"),
                      name=default_name, save=True)
    return result


def run_training(args):
    train_data, test_data = prepare_data(args)
    train_env, eval_env, agent, buffer = build_env_and_agent(train_data, test_data, args)

    train_cfg.total_timesteps = args.steps
    trainer = SACTrainer(train_env, eval_env, agent, buffer, train_cfg)

    # ── 체크포인트에서 이어 학습 (--resume)
    if getattr(args, "resume", False):
        try:
            agent.load(args.load)
            logger.info(f"✅ 체크포인트 이어서 학습: {args.load}.pt")
            # 가중치 학습됨 → 워밍업 단축 (배치 1개만 채우면 OK)
            agent.cfg.min_replay_size = max(agent.cfg.batch_size, 500)
            logger.info(f"   워밍업 단축: min_replay_size={agent.cfg.min_replay_size}")

            # 첫 평가로 best_sac 기준점 설정 (resume 후 첫 eval 이 무조건 best 되는 사고 방지)
            logger.info("   기준 성능 측정 중...")
            init_eval = trainer._evaluate()
            best_metric = getattr(train_cfg, "best_metric", "sharpe")
            # _evaluate()는 alpha_vs_bh 키가 없고 total_return+bh_return만 반환.
            # _compute_best_metric()으로 올바른 기준점 계산 (sharpe/alpha_vs_bh/calmar 일관 처리)
            trainer._best_eval_score = trainer._compute_best_metric(init_eval, best_metric)
            logger.info(
                f"   기준 {best_metric}={trainer._best_eval_score:.4f} | "
                f"TotalRet={init_eval['total_return']:+.2%} | "
                f"BH={init_eval['bh_return']:+.2%} | "
                f"MDD={init_eval['mdd']:+.2%}"
            )
        except FileNotFoundError:
            logger.warning(f"⚠ 체크포인트 없음: checkpoints/{args.load}.pt — 처음부터 학습")

    logger.info(f"\n{'='*60}")
    logger.info("SAC 학습 시작" + (" (이어서)" if getattr(args, "resume", False) else ""))
    logger.info(f"{'='*60}")

    eval_history = trainer.train()
    trainer.save_results()

    if not args.no_plot and eval_history:
        plot_training_curves(eval_history, save=True)
        logger.info("학습 곡선 저장 완료")

    # 최고 모델 로드 후 최종 평가
    logger.info("\n최고 모델 로드 후 최종 평가...")
    try:
        agent.load("best_sac")
    except FileNotFoundError:
        logger.info("(저장된 최고 모델 없음, 현재 모델로 평가)")
    except RuntimeError as e:
        # 백본/차원 mismatch — 이전 학습의 체크포인트와 호환 안 됨
        logger.warning(f"⚠ best_sac 로드 실패 — 현재 학습된 모델로 평가합니다.\n{e}")

    return _final_eval_and_plot(agent, eval_env, args, default_name="SAC_Best")


# ── 평가만 실행 ────────────────────────────────────────

def run_eval_only(args):
    # eval 모드에서는 사용자가 지정한 전체 기간을 테스트셋으로 사용하는 경우가 많음.
    # --test-ratio 를 1.0으로 강제하여 전체를 eval_env 에 할당.
    original_test_ratio = args.test_ratio
    args.test_ratio = 1.0
    try:
        train_data, test_data = prepare_data(args)
    finally:
        args.test_ratio = original_test_ratio

    # train_data 가 비어있으면 (test_ratio=1.0) test_data 를 정규화 통계용으로 사용
    if len(train_data[0]) == 0:
        train_data = test_data

    _, eval_env, agent, _ = build_env_and_agent(train_data, test_data, args)

    logger.info(f"모델 로드: {args.load}")
    agent.load(args.load)

    return _final_eval_and_plot(agent, eval_env, args, default_name=args.load)


# ── 빠른 테스트 ────────────────────────────────────────

def run_quick_test(args):
    """설치 확인 및 단기 동작 테스트 (1000 스텝)."""
    logger.info("빠른 테스트 모드 (1000 스텝)")
    args.steps         = 1000
    args.use_synthetic = True
    train_cfg.eval_interval  = 500
    train_cfg.log_interval   = 200
    train_cfg.save_interval  = 1000
    sac_cfg.min_replay_size  = 100
    sac_cfg.buffer_size      = 5000
    sac_cfg.batch_size       = 64
    return run_training(args)


# ── CLI ───────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SAC 트레이딩 시스템")
    p.add_argument("--mode",      choices=["train","eval","test","walkforward"],
                   default="test")
    p.add_argument("--ticker",    default=None,        help="단일 종목 (None=합성 데이터)")
    p.add_argument("--start",     default="2018-01-01")
    p.add_argument("--end",       default="2024-12-31")
    p.add_argument("--interval",  default="1d")
    p.add_argument("--steps",     type=int, default=300_000)
    p.add_argument("--reward",    choices=["pnl","sharpe","sortino","mixed","dsr","dsr_cvar"],
                   default="mixed",
                   help="mixed=alpha vs B&H 직접 보상 (기본), dsr=Differential Sharpe")
    p.add_argument("--episode-length", type=int, default=126, dest="episode_length",
                   help="에피소드 길이 (126=6개월 기본). 추세 학습에 충분한 기간 확보.")
    p.add_argument("--window-size",    type=int, default=60,  dest="window_size")
    p.add_argument("--test-ratio",     type=float, default=0.2, dest="test_ratio")
    p.add_argument("--load",      default="best_sac",  help="로드할 체크포인트 이름 (확장자 .pt 제외)")
    p.add_argument("--resume",    action="store_true", default=False,
                   help="--load 의 체크포인트에서 이어 학습 (가중치만 불러옴, 버퍼는 재워밍업)")
    p.add_argument("--use-synthetic",  action="store_true", dest="use_synthetic")
    p.add_argument("--allow-synthetic-fallback", action="store_true",
                   dest="allow_synthetic_fallback",
                   help="Yahoo 다운로드 실패 시 합성 데이터로 대체 허용")
    p.add_argument("--no-plot",        action="store_true", dest="no_plot")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default=os.getenv("SAC_DEVICE", "auto"),
                   help="실행 장치 선택 (auto=cuda 가능 시 GPU, 기본 auto)")
    p.add_argument("--cpu-workers", type=int, default=int(os.getenv("SAC_CPU_WORKERS", str(CPU_WORKERS))),
                   help="CPU 병렬 worker/thread 수 (OMP/MKL/OpenBLAS/torch threads)")
    # Walk-forward 옵션 (training/walk_forward.py 참조)
    p.add_argument("--wf-splits",   type=int,   default=5,   dest="wf_splits",
                   help="WF 분할 수 (PBO 원하면 6+, López de Prado 2018)")
    p.add_argument("--wf-mode",     choices=["anchored", "rolling"], default="anchored",
                   dest="wf_mode",
                   help="anchored=확장 윈도우 / rolling=고정 슬라이딩 (Pardo 1992)")
    p.add_argument("--wf-rolling-size", type=int, default=None, dest="wf_rolling_size",
                   help="rolling mode 의 학습 윈도우 봉 수 (None=1 fold size)")
    p.add_argument("--wf-embargo",  type=float, default=0.01, dest="wf_embargo",
                   help="train/val 사이 embargo 비율 (López de Prado 2018, 0.01 권장)")
    p.add_argument("--wf-seeds",    type=int,   default=1,   dest="wf_seeds",
                   help="폴드당 시드 수 (Henderson 2018 — RL 재현성, 3+ 권장)")
    p.add_argument("--wf-from-scratch", action="store_true", default=False,
                   dest="wf_from_scratch",
                   help="매 폴드 가중치 초기화 (default: warm start)")
    p.add_argument("--wf-pbo",      action="store_true", default=False, dest="wf_pbo",
                   help="PBO (Probability of Backtest Overfitting) 계산 — n_splits>=6 권장")

    # ── PER / N-step / Reward Norm (런타임 토글)
    p.add_argument("--per",           dest="use_per", action="store_true",  default=True,
                   help="Prioritized Experience Replay 사용 (기본 ON)")
    p.add_argument("--no-per",        dest="use_per", action="store_false",
                   help="PER 비활성 (균일 샘플링)")
    p.add_argument("--n-step",        dest="n_step", type=int, default=3,
                   help="N-step return (1=기본 SAC, 3~5 권장; variance vs bias)")
    p.add_argument("--reward-scaling", dest="reward_scaling", type=float, default=0.1,
                   help="reward 스케일 (mixed 모드 기본 0.1)")
    p.add_argument("--gamma", dest="gamma", type=float, default=0.99,
                   help="할인율 (기본 0.99, eff horizon ≈ 100 step. 추세 추종 필수.)")
    p.add_argument("--normalize-reward", dest="normalize_reward",
                   action="store_true", default=False,
                   help="Welford 온라인 보상 정규화 (기본 OFF — reward_scaling 과 충돌)")
    p.add_argument("--no-normalize-reward", dest="normalize_reward",
                   action="store_false")

    # ── 최신 SAC 기법 (DroQ / LAP / CAPS / Reset)
    p.add_argument("--droq",     dest="use_droq", action="store_true",  default=True,
                   help="DroQ: critic dropout + 높은 UTD (Hiraoka 2022 / 기본 ON)")
    p.add_argument("--no-droq",  dest="use_droq", action="store_false")
    p.add_argument("--utd",      dest="utd_ratio", type=int, default=2,
                   help="UTD ratio G (DroQ: critic 업데이트 횟수 / env step, 기본 2)")
    p.add_argument("--critic-dropout", dest="critic_dropout", type=float, default=0.05,
                   help="DroQ critic dropout 확률 (기본 0.05)")
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-4,
                   help="AdamW weight decay (Loshchilov 2019, 기본 1e-4)")
    p.add_argument("--lap",      dest="use_lap", action="store_true",  default=True,
                   help="LAP: Huber + max(|δ|,λ) priority (Fujimoto 2020 / 기본 ON)")
    p.add_argument("--no-lap",   dest="use_lap", action="store_false")
    p.add_argument("--caps-t",   dest="caps_lambda_t", type=float, default=0.3,
                   help="CAPS temporal smoothness λ_T (Mysore 2021, 기본 0.2 — 과매매 억제)")
    p.add_argument("--reset-interval", dest="reset_interval", type=int, default=0,
                   help="Primacy bias reset 주기 (gradient updates 기준, 0=off 기본)")
    p.add_argument("--reset-mode", dest="reset_mode",
                   choices=["head", "full_critic", "shrink_perturb"],
                   default="shrink_perturb",
                   help="Reset 방식: head=마지막층만 / full_critic=전체 재초기화 / shrink_perturb=θ←α·θ+noise (Ash 2020)")
    p.add_argument("--reset-actor", dest="reset_actor", action="store_true", default=True,
                   help="BBF: actor 도 함께 리셋 (기본 ON)")
    p.add_argument("--no-reset-actor", dest="reset_actor", action="store_false",
                   help="Critic 만 리셋 (Nikishin v1)")
    p.add_argument("--action-change-penalty", dest="action_change_penalty",
                   type=float, default=0.1,
                   help="env reward 에 -λ·|Δaction| 추가 (v9: 0.1로 거래 억제)")

    # ── DSR + CVaR 보상 파라미터 (Moody-Saffell 2001, Coache-Jaimungal 2021)
    p.add_argument("--dsr-eta",     dest="dsr_eta", type=float, default=0.01,
                   help="DSR EWMA 적응 속도 (Moody-Saffell 2001)")
    p.add_argument("--cvar-alpha",  dest="cvar_alpha", type=float, default=0.05,
                   help="CVaR 분위수 (기본 0.05 = 5퍼센트)")
    p.add_argument("--cvar-window", dest="cvar_window", type=int, default=50,
                   help="CVaR rolling 윈도우")
    p.add_argument("--cvar-lambda", dest="cvar_lambda", type=float, default=1.0,
                   help="CVaR 패널티 가중치")

    # ── BC Regularization (Fujimoto-Gu 2021 TD3+BC)
    p.add_argument("--use-bc-reg", dest="use_bc_regularization",
                   action="store_true", default=False,
                   help="Actor loss 에 BC 정규화 추가 (TD3+BC, Fujimoto 2021)")
    p.add_argument("--bc-lambda",  dest="bc_lambda", type=float, default=2.5,
                   help="BC 정규화 가중치 (논문 권장 2.5)")

    # ── Macro features (Welch & Goyal 2008)
    p.add_argument("--use-macro",  dest="use_macro", action="store_true", default=True,
                   help="VIX/금리/yield curve macro 피처 추가 (기본 ON)")
    p.add_argument("--no-macro",   dest="use_macro", action="store_false")

    # ── Fractional Differentiation (López de Prado 2018)
    p.add_argument("--use-frac-diff", dest="use_frac_diff",
                   action="store_true", default=True,
                   help="분수차분 피처 (기본 ON)")
    p.add_argument("--no-frac-diff",  dest="use_frac_diff", action="store_false")

    # ── Momentum Features (Jegadeesh & Titman 1993)
    p.add_argument("--use-momentum", dest="use_momentum",
                   action="store_true", default=True,
                   help="다중 주기 모멘텀 피처: ROC 1/3/6/12M, 52주 고가/저가 (기본 ON)")
    p.add_argument("--no-momentum",  dest="use_momentum", action="store_false")

    # ── Train-time Data Augmentation (Iwana 2021, Wen 2021, Tobin 2017, Sinha 2022)
    # 모든 항목은 train 모드에서만 적용 — eval/실거래 경로 무영향.
    p.add_argument("--obs-jitter", dest="use_obs_jitter",
                   action="store_true", default=True,
                   help="관측 가우시안 jittering (Iwana 2021 / 기본 ON)")
    p.add_argument("--no-obs-jitter", dest="use_obs_jitter", action="store_false")
    p.add_argument("--obs-jitter-sigma", dest="obs_jitter_sigma",
                   type=float, default=0.01,
                   help="jitter 노이즈 std (표준화된 obs 기준, 권장 0.005~0.02)")

    p.add_argument("--magnitude-warp", dest="use_magnitude_warp",
                   action="store_true", default=True,
                   help="cubic-spline magnitude warping (Iwana 2021 / 기본 ON)")
    p.add_argument("--no-magnitude-warp", dest="use_magnitude_warp", action="store_false")
    p.add_argument("--mag-warp-sigma", dest="mag_warp_sigma",
                   type=float, default=0.05,
                   help="warp 곡선 진폭 σ (≈1±σ 곡선)")

    p.add_argument("--domain-rand", dest="use_domain_rand",
                   action="store_true", default=True,
                   help="commission/slippage 도메인 랜덤화 (Tobin 2017 / 기본 ON)")
    p.add_argument("--no-domain-rand", dest="use_domain_rand", action="store_false")
    p.add_argument("--domain-rand-pct", dest="domain_rand_pct",
                   type=float, default=0.3,
                   help="commission/slippage 변동 비율 (±N, 권장 0.2~0.5)")

    p.add_argument("--state-aug", dest="use_state_aug",
                   action="store_true", default=True,
                   help="S4RL critic state augmentation (Sinha 2022 / 기본 ON)")
    p.add_argument("--no-state-aug", dest="use_state_aug", action="store_false")
    p.add_argument("--state-aug-sigma", dest="state_aug_sigma",
                   type=float, default=0.01,
                   help="S4RL state noise std")

    # ── Transformer Actor/Critic (Parisotto 2020 / Nie 2023 / Kim 2022)
    # 단일 모델로 고정: GTrXL + PatchTST + RevIN 기반 Transformer-SAC.
    p.add_argument("--trans-d-model", dest="trans_d_model", type=int, default=128,
                   help="Transformer 모델 차원 (기본 128)")
    p.add_argument("--trans-heads", dest="trans_n_heads", type=int, default=8,
                   help="Multi-head attention head 수 (기본 8)")
    p.add_argument("--trans-layers", dest="trans_n_layers", type=int, default=3,
                   help="Transformer encoder layer 수 (기본 3)")
    p.add_argument("--trans-dropout", dest="trans_dropout", type=float, default=0.1,
                   help="Transformer dropout (기본 0.1)")
    p.add_argument("--trans-no-revin", dest="trans_use_revin",
                   action="store_false", default=True,
                   help="RevIN 비활성 (기본 ON, Kim 2022)")
    p.add_argument("--trans-no-gate", dest="trans_use_gtrxl_gate",
                   action="store_false", default=True,
                   help="GTrXL gating 비활성 (기본 ON, Parisotto 2020)")

    # ── Best 모델 선택 기준 + Early stopping
    p.add_argument("--best-metric", dest="best_metric",
                   choices=["sharpe", "alpha_vs_bh", "calmar"], default="alpha_vs_bh",
                   help="best_sac 갱신 기준 (alpha_vs_bh: B&H 초과수익 최적화)")
    p.add_argument("--best-min-margin", dest="best_min_margin", type=float, default=-0.02,
                   help="best 갱신 최소 마진 (기본 -0.02: B&H 2pct 이내 모델도 저장)")
    p.add_argument("--early-stop-patience", dest="early_stop_patience", type=int, default=0,
                   help="N회 연속 best 못 갱신 시 학습 중단 (0=off 기본)")
    p.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=5,
                   help="평가 시 N회 평균 (기본 5)")
    p.add_argument("--eval-random-start", dest="eval_random_start",
                   action="store_true", default=True,
                   help="eval 시 random start 활성 (기본 ON — 다양한 구간 평균)")
    p.add_argument("--no-eval-random-start", dest="eval_random_start", action="store_false")

    return p.parse_args()


def run_walkforward(args):
    """
    Walk-Forward 재학습 실행.

    데이터 사용:
        --start ~ --end 의 전체 데이터를 walk-forward 분할.
        (--test-ratio 는 무시 — 진정한 TRUE OOS 검증은 별도 --mode eval 로)

    적용 이론:
        Pardo (1992) anchored/rolling, López de Prado (2018) embargo,
        Henderson (2018) multi-seed, Bailey (2017) PBO.
    """
    from training.walk_forward import WalkForwardTrainer

    # WF 는 전체 데이터를 자체 분할. test_ratio 임시 0 으로 prepare_data 호출
    original_test_ratio = args.test_ratio
    args.test_ratio = 0.0
    try:
        train_data, _ = prepare_data(args)
    finally:
        args.test_ratio = original_test_ratio
    full_feat, full_price = train_data

    # env_cfg / sac_cfg 에 CLI 옵션 반영 (메인 학습 경로와 동일)
    _apply_args_to_configs(args)

    wf = WalkForwardTrainer(
        n_splits           = args.wf_splits,
        n_steps_per_fold   = args.steps // args.wf_splits,
        mode               = args.wf_mode,
        rolling_train_size = args.wf_rolling_size,
        embargo_pct        = args.wf_embargo,
        retrain_from_scratch = args.wf_from_scratch,
        n_seeds            = args.wf_seeds,
        compute_pbo        = args.wf_pbo,
    )
    summary = wf.run(full_feat, full_price, env_cfg, sac_cfg)
    return summary


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "test":
        run_quick_test(args)
    elif args.mode == "train":
        run_training(args)
    elif args.mode == "eval":
        run_eval_only(args)
    elif args.mode == "walkforward":
        run_walkforward(args)
