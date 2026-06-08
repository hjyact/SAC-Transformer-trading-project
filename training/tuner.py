"""
training/tuner.py — Optuna 기반 하이퍼파라미터 자동 탐색

SAC의 주요 하이퍼파라미터를 베이지안 최적화로 탐색합니다.

탐색 대상:
  - 학습률 (actor_lr, critic_lr, alpha_lr)
  - 네트워크 크기 (hidden_dims, n_layers)
  - SAC 파라미터 (gamma, tau, target_entropy)
  - 보상 설계 파라미터 (reward_scaling, drawdown_penalty)
  - 환경 파라미터 (window_size, reward_type)

사용법:
    python -m training.tuner --trials 30 --timeout 3600
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from copy import deepcopy
from dataclasses import replace

logger = logging.getLogger(__name__)


def make_synthetic_data(n: int = 1500, seed: int = 42) -> tuple:
    """튜닝용 합성 데이터 생성."""
    np.random.seed(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    vol   = np.zeros(n); vol[0] = 0.01
    for i in range(1, n):
        s = np.random.randn()
        vol[i] = np.sqrt(0.00001 + 0.09*(vol[i-1]*s)**2 + 0.90*vol[i-1]**2)
    ret   = np.random.randn(n)*vol + np.sin(np.linspace(0, 4*np.pi, n))*0.0002
    close = 100 * np.exp(ret.cumsum())
    df = pd.DataFrame({
        "Open":   close * np.exp(-ret*0.5),
        "High":   close * (1 + np.abs(np.random.randn(n)) * vol),
        "Low":    close * (1 - np.abs(np.random.randn(n)) * vol),
        "Close":  close,
        "Volume": np.random.lognormal(15, 1, n),
    }, index=dates)
    return df


def objective(trial, price_df: pd.DataFrame) -> float:
    """
    Optuna objective: 검증셋 Sharpe ratio 최대화.
    각 trial은 독립적인 환경/에이전트로 단기 학습 후 평가.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("pip install optuna")

    from config import EnvConfig, SACConfig, TrainConfig, FeatureConfig
    from utils.features import build_all_features
    from env.trading_env import TradingEnv
    from agent.sac_agent import SACAgent
    from agent.replay_buffer import ReplayBuffer

    # ── 탐색 공간 정의 ─────────────────────────────────
    reward_type    = trial.suggest_categorical("reward_type",    ["sharpe", "sortino", "mixed"])
    window_size    = trial.suggest_int("window_size",            10, 60, step=10)
    reward_scaling = trial.suggest_float("reward_scaling",       0.5, 5.0, log=True)
    drawdown_pen   = trial.suggest_float("drawdown_penalty",     0.1, 2.0)
    risk_pen       = trial.suggest_float("risk_penalty",         0.05, 0.5)

    actor_lr  = trial.suggest_float("actor_lr",  1e-5, 1e-3, log=True)
    critic_lr = trial.suggest_float("critic_lr", 1e-5, 1e-3, log=True)
    gamma     = trial.suggest_float("gamma",     0.95, 0.999)
    tau       = trial.suggest_float("tau",       0.001, 0.02)
    n_layers  = trial.suggest_int("n_layers",    1, 3)
    hidden    = trial.suggest_categorical("hidden_size", [128, 256, 512])
    batch_sz  = trial.suggest_categorical("batch_size",  [128, 256, 512])

    # ── 피처 & 데이터 준비 ─────────────────────────────
    feat_cfg = FeatureConfig()
    feat_df  = build_all_features(price_df.copy(), feat_cfg)
    common   = feat_df.index.intersection(price_df.index)
    feat_df  = feat_df.loc[common]
    price_df2= price_df.loc[common]

    split      = int(len(feat_df) * 0.75)
    train_feat = feat_df.iloc[:split]
    train_px   = price_df2.iloc[:split]
    val_feat   = feat_df.iloc[split:]
    val_px     = price_df2.iloc[split:]

    # ── 환경 설정 ──────────────────────────────────────
    env_cfg = EnvConfig(
        window_size     = window_size,
        reward_type     = reward_type,
        reward_scaling  = reward_scaling,
        drawdown_penalty= drawdown_pen,
        risk_penalty    = risk_pen,
        episode_length  = 126,   # 튜닝 시 짧게
    )
    train_env = TradingEnv(train_feat, train_px, env_cfg, "train")
    # val 환경에 train 통계 주입 → tune 시 OOS 통계 누수 차단
    val_env   = TradingEnv(
        val_feat, val_px, env_cfg, "eval",
        feat_mean=train_env._feat_mean.copy(),
        feat_std =train_env._feat_std.copy(),
    )

    # ── SAC 설정 ───────────────────────────────────────
    sac_cfg = SACConfig(
        actor_lr    = actor_lr,
        critic_lr   = critic_lr,
        gamma       = gamma,
        tau         = tau,
        hidden_dims = [hidden] * n_layers,
        batch_size  = batch_sz,
        buffer_size = 20_000,
        min_replay_size = 200,
    )

    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]
    obs_meta = {
        "window":   train_env.cfg.window_size,
        "n_feat":   train_env.n_feat,
        "port_dim": 4,
    }
    agent   = SACAgent(obs_dim, act_dim, sac_cfg, obs_meta=obs_meta)
    buffer  = ReplayBuffer(obs_dim, act_dim, sac_cfg.buffer_size)

    # ── 단기 학습 (trial당 10,000 스텝) ───────────────
    N_STEPS = 10_000
    obs, _  = train_env.reset()
    for t in range(N_STEPS):
        if t < sac_cfg.min_replay_size:
            action = train_env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, r, term, trunc, _ = train_env.step(action)
        buffer.add(obs, action, r, next_obs, float(term))
        obs = next_obs
        if term or trunc:
            obs, _ = train_env.reset()

        if t >= sac_cfg.min_replay_size and buffer.is_ready:
            agent.train_step(buffer)

    # ── 검증 평가 ──────────────────────────────────────
    step_rets = []
    obs, _ = val_env.reset()
    done   = False
    while not done:
        action = agent.select_action(obs, deterministic=True)
        obs, _, term, trunc, info = val_env.step(action)
        step_rets.append(info["step_ret"])
        done = term or trunc

    rets   = np.array(step_rets)
    if len(rets) < 10:
        return -999.0

    sharpe = (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252)
    return float(sharpe)


def run_tuning(n_trials: int = 30, timeout: int = 3600) -> dict:
    """
    Optuna 탐색 실행.

    Parameters
    ----------
    n_trials : 시도 횟수
    timeout  : 최대 실행 시간 (초)
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("optuna 미설치. pip install optuna")
        return {}

    price_df = make_synthetic_data(1500, seed=42)

    study = optuna.create_study(
        direction="maximize",
        study_name="sac_trader",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    study.optimize(
        lambda trial: objective(trial, price_df),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=1,
        show_progress_bar=True,
    )

    best = study.best_params
    logger.info(f"\n최적 하이퍼파라미터 (Sharpe={study.best_value:.4f}):")
    for k, v in best.items():
        logger.info(f"  {k:25s}: {v}")

    # 결과 저장
    from config import RESULT_DIR
    import json
    out = RESULT_DIR / "best_hyperparams.json"
    with open(out, "w") as f:
        json.dump({"best_params": best, "best_sharpe": study.best_value}, f, indent=2)
    logger.info(f"저장: {out}")

    return best


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--trials",  type=int, default=30)
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    best = run_tuning(args.trials, args.timeout)
    print("\n=== 최적 파라미터 ===")
    for k, v in best.items():
        print(f"  {k}: {v}")
