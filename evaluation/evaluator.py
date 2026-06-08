"""
evaluation/evaluator.py — 백테스트 성과 분석 및 시각화

참고 이론:
  - Sharpe Ratio (Sharpe, 1966)
  - Sortino Ratio (Sortino & van der Meer, 1991)
  - Calmar Ratio: CAGR / |MDD|
  - Maximum Drawdown
  - Kelly Criterion: 최적 배팅 비율 = μ/σ²
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Ensure project root is importable when this module is run directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import RESULT_DIR
from utils.plot_style import apply_korean_font

logger = logging.getLogger(__name__)

# 한글 폰트 + 마이너스 부호 깨짐 차단 (Windows: Malgun Gothic)
apply_korean_font()

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size": 10,
})


# ── 성과 지표 계산 ─────────────────────────────────────

def compute_metrics(
    equity_curve: pd.Series,
    step_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.03,  # 연 3% 무위험 수익률
) -> Dict[str, float]:
    """
    전체 성과 지표를 계산합니다.

    Parameters
    ----------
    equity_curve    : 시간별 포트폴리오 가치
    step_returns    : 스텝별 수익률 (없으면 equity_curve에서 계산)
    risk_free_rate  : 연간 무위험 수익률
    """
    if step_returns is None:
        step_returns = equity_curve.pct_change().dropna().values

    rets = np.array(step_returns)
    n    = len(rets)

    # 기본 통계
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_days = (equity_curve.index[-1] - equity_curve.index[0]).days if hasattr(equity_curve.index, 'days') else n
    years  = max(n / 252, 1e-9)

    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    excess   = rets - daily_rf

    # 수익률 통계
    mean_ret = rets.mean()
    std_ret  = rets.std() + 1e-9

    # Sharpe Ratio (연율화)
    sharpe = (excess.mean() / (excess.std() + 1e-9)) * np.sqrt(252)

    # Sortino Ratio (하방 위험만)
    downside = excess[excess < 0]
    downside_std = downside.std() + 1e-9 if len(downside) > 0 else std_ret
    sortino = (excess.mean() / downside_std) * np.sqrt(252)

    # CAGR
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1

    # MDD
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / rolling_max
    mdd         = drawdown.min()

    # MDD 기간
    end_mdd   = drawdown.idxmin()
    start_mdd = equity_curve[:end_mdd].idxmax()
    mdd_days  = (end_mdd - start_mdd).days if hasattr(end_mdd, 'days') else 0

    # Calmar Ratio
    calmar = cagr / (abs(mdd) + 1e-9)

    # 승률
    win_rate = (rets > 0).mean()

    # Profit Factor
    wins  = rets[rets > 0].sum()
    loses = abs(rets[rets < 0].sum())
    profit_factor = wins / (loses + 1e-9)

    # 연간 변동성
    ann_vol = std_ret * np.sqrt(252)

    # Kelly Criterion: f* = μ/σ² (연율화 기준으로 변환 후 클리핑)
    # 실전에서는 Half-Kelly (f*/2) 사용 권장
    kelly_raw   = (mean_ret * 252) / (ann_vol ** 2 + 1e-9)
    kelly       = float(np.clip(kelly_raw, 0.0, 2.0))    # 0~2 범위로 클리핑
    half_kelly  = kelly / 2.0

    return {
        "total_return":   total_return,
        "cagr":           cagr,
        "ann_vol":        ann_vol,
        "sharpe":         sharpe,
        "sortino":        sortino,
        "calmar":         calmar,
        "mdd":            mdd,
        "mdd_days":       mdd_days,
        "win_rate":       win_rate,
        "profit_factor":  profit_factor,
        "kelly":          kelly,
        "half_kelly":     half_kelly,
        "n_periods":      n,
    }


# ── 전체 평가 실행 ─────────────────────────────────────

def run_evaluation(
    agent,
    eval_env,
    n_episodes: int = 1,
    deterministic: bool = True,
    name: str = "SAC",
) -> Dict:
    """에이전트를 환경에서 실행하여 전체 성과를 평가합니다."""
    all_capitals   = []
    all_positions  = []
    all_step_rets  = []
    all_actions    = []

    for ep in range(n_episodes):
        obs, _   = eval_env.reset()
        done     = False
        ep_caps  = [eval_env.cfg.initial_capital]
        ep_pos   = []
        ep_rets  = []
        ep_acts  = []

        while not done:
            action = agent.select_action(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

            ep_caps.append(info["capital"])
            ep_pos.append(info["position"])
            ep_rets.append(info["step_ret"])
            ep_acts.append(float(action[0]))

        all_capitals.append(ep_caps)
        all_positions.append(ep_pos)
        all_step_rets.extend(ep_rets)
        all_actions.extend(ep_acts)

    # 첫 에피소드 기준으로 equity curve 구성
    equity = pd.Series(
        all_capitals[0],
        name="equity",
    )

    step_rets = np.array(all_step_rets)
    metrics   = compute_metrics(equity, step_rets)

    # ── Buy & Hold 벤치마크 계산 (동일 기간, 항상 100% 매수)
    benchmark = None
    try:
        start_i = eval_env._start
        end_i   = min(eval_env._step_idx, len(eval_env.prices) - 1)
        bh_prices = eval_env.prices["Close"].iloc[start_i:end_i + 1].reset_index(drop=True)
        if len(bh_prices) > 1:
            init_cap = eval_env.cfg.initial_capital
            benchmark = (bh_prices / bh_prices.iloc[0]) * init_cap
            benchmark.name = "buy_and_hold"

            # 벤치마크 대비 초과수익 / 정보비율
            bh_rets   = benchmark.pct_change().dropna().values
            min_len   = min(len(step_rets), len(bh_rets))
            excess    = step_rets[:min_len] - bh_rets[:min_len]
            ann_alpha = float(excess.mean() * 252)
            ir        = float(excess.mean() / (excess.std() + 1e-9) * np.sqrt(252))
            bh_total  = float((benchmark.iloc[-1] / benchmark.iloc[0]) - 1)
            metrics["benchmark_total_return"] = bh_total
            metrics["alpha_vs_bh"]            = metrics["total_return"] - bh_total
            metrics["annualized_alpha"]       = ann_alpha
            metrics["information_ratio"]      = ir
    except Exception as e:
        logger.warning(f"Buy & Hold 벤치마크 계산 실패: {e}")

    logger.info(f"\n{'='*55}")
    logger.info(f"평가 결과 — {name}")
    logger.info(f"{'='*55}")
    logger.info(f"  총 수익률   : {metrics['total_return']:>+10.2%}")
    logger.info(f"  CAGR        : {metrics['cagr']:>+10.2%}")
    logger.info(f"  Sharpe      : {metrics['sharpe']:>10.3f}")
    logger.info(f"  Sortino     : {metrics['sortino']:>10.3f}")
    logger.info(f"  Calmar      : {metrics['calmar']:>10.3f}")
    logger.info(f"  MDD         : {metrics['mdd']:>+10.2%}")
    logger.info(f"  승률        : {metrics['win_rate']:>10.2%}")
    logger.info(f"  Profit Factor: {metrics['profit_factor']:>9.2f}")
    logger.info(f"  Kelly       : {metrics['kelly']:>10.4f}")
    if benchmark is not None:
        logger.info(f"  ─────────────────────────────────")
        logger.info(f"  Buy & Hold  : {metrics['benchmark_total_return']:>+10.2%}")
        logger.info(f"  α vs B&H    : {metrics['alpha_vs_bh']:>+10.2%}  ({metrics['annualized_alpha']:+.2%}/yr)")
        logger.info(f"  Info Ratio  : {metrics['information_ratio']:>10.3f}")

    return {
        "metrics":   metrics,
        "equity":    equity,
        "positions": all_positions[0] if all_positions else [],
        "returns":   step_rets,
        "actions":   all_actions,
        "benchmark": benchmark,
    }


# ── 시각화 ────────────────────────────────────────────

def plot_training_curves(eval_history: List[Dict], save: bool = True) -> plt.Figure:
    """학습 곡선 (Sharpe, Return, MDD, Alpha)."""
    steps   = [h["step"] for h in eval_history]
    sharpes = [h["sharpe"] for h in eval_history]
    returns = [h["total_return"] for h in eval_history]
    mdds    = [h["mdd"] for h in eval_history]
    trades  = [h["mean_trades"] for h in eval_history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("SAC 학습 곡선", fontsize=14, fontweight="bold")

    axes[0,0].plot(steps, sharpes, color="#4C72B0", linewidth=1.5)
    axes[0,0].axhline(1.0, color="green", linestyle="--", alpha=0.5, label="Sharpe=1")
    axes[0,0].set_title("Sharpe Ratio (Rolling)")
    axes[0,0].set_ylabel("Sharpe")
    axes[0,0].legend()

    axes[0,1].plot(steps, [r*100 for r in returns], color="#DD8452", linewidth=1.5)
    axes[0,1].axhline(0, color="gray", linestyle=":", alpha=0.4)
    axes[0,1].set_title("총 수익률 (%)")
    axes[0,1].set_ylabel("Return (%)")

    axes[1,0].plot(steps, [m*100 for m in mdds], color="#C44E52", linewidth=1.5)
    axes[1,0].fill_between(steps, [m*100 for m in mdds], 0, color="#C44E52", alpha=0.2)
    axes[1,0].set_title("Maximum Drawdown (%)")
    axes[1,0].set_ylabel("MDD (%)")

    axes[1,1].plot(steps, trades, color="#55A868", linewidth=1.5)
    axes[1,1].set_title("거래 횟수 / 에피소드")
    axes[1,1].set_ylabel("# Trades")

    for ax in axes.flat:
        ax.set_xlabel("Timestep")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(RESULT_DIR / "training_curves.png", bbox_inches="tight")
    return fig


def plot_backtest(eval_result: Dict, benchmark: Optional[pd.Series] = None,
                  name: str = "SAC", save: bool = True) -> plt.Figure:
    """백테스트 결과 시각화 (자산곡선, 포지션, 분포)."""
    equity    = eval_result["equity"]
    positions = eval_result["positions"]
    rets      = eval_result["returns"]
    metrics   = eval_result["metrics"]

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # ① 자산 곡선
    ax1 = fig.add_subplot(gs[0, :])
    norm = equity / equity.iloc[0] * 100
    ax1.plot(norm.index if hasattr(norm, 'index') else range(len(norm)),
             norm.values, label=name, color="#4C72B0", linewidth=2)
    if benchmark is not None:
        bench_norm = benchmark / benchmark.iloc[0] * 100
        ax1.plot(bench_norm.index if hasattr(bench_norm, 'index') else range(len(bench_norm)),
                 bench_norm.values, label="Buy & Hold", color="gray",
                 linewidth=1.2, linestyle="--", alpha=0.7)
    ax1.set_title(f"{name} — 자산 곡선 (기준=100)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ② Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max * 100
    ax2.fill_between(range(len(dd)), dd.values, 0, color="#C44E52", alpha=0.5)
    ax2.set_title("Drawdown (%)")
    ax2.grid(alpha=0.3)

    # ③ 포지션
    ax3 = fig.add_subplot(gs[1, 1])
    if positions:
        ax3.plot(positions, color="#55A868", linewidth=0.8, alpha=0.8)
        ax3.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax3.fill_between(range(len(positions)), positions, 0,
                          where=[p > 0 for p in positions], color="green", alpha=0.2)
        ax3.fill_between(range(len(positions)), positions, 0,
                          where=[p < 0 for p in positions], color="red", alpha=0.2)
    ax3.set_title("포지션 히스토리")
    ax3.set_ylim(-1.1, 1.1)
    ax3.grid(alpha=0.3)

    # ④ 수익률 분포
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(rets * 100, bins=50, color="#4C72B0", alpha=0.75, edgecolor="white")
    ax4.axvline(0, color="red", linestyle="--", alpha=0.7)
    ax4.axvline(rets.mean()*100, color="orange", linestyle="--",
                label=f"Mean: {rets.mean()*100:.3f}%")
    ax4.set_title("수익률 분포")
    ax4.set_xlabel("Return (%)")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # ⑤ 성과 테이블
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")
    table_data = [
        ["지표",                 "값"],
        ["총 수익률",            f"{metrics['total_return']:+.2%}"],
        ["CAGR",                 f"{metrics['cagr']:+.2%}"],
        ["Sharpe Ratio",         f"{metrics['sharpe']:.3f}"],
        ["Sortino Ratio",        f"{metrics['sortino']:.3f}"],
        ["Calmar Ratio",         f"{metrics['calmar']:.3f}"],
        ["MDD",                  f"{metrics['mdd']:.2%}"],
        ["승률",                 f"{metrics['win_rate']:.2%}"],
        ["Profit Factor",        f"{metrics['profit_factor']:.2f}"],
        ["Kelly Fraction",       f"{min(metrics['kelly'], 1.0):.3f}"],
    ]
    tbl = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    ax5.set_title("성과 요약")

    fig.suptitle(f"{name} SAC 트레이딩 백테스트", fontsize=14, fontweight="bold")

    if save:
        fig.savefig(RESULT_DIR / f"{name}_backtest.png", bbox_inches="tight")
        logger.info(f"저장: {RESULT_DIR / f'{name}_backtest.png'}")
    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("evaluation.evaluator direct-run test")
    sample_equity = pd.Series([100.0, 101.0, 100.5, 102.0])
    sample_returns = np.array([0.01, -0.004950, 0.014925])
    metrics = compute_metrics(sample_equity, sample_returns)
    for key, value in metrics.items():
        print(f"{key}: {value}")
