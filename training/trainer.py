"""
training/trainer.py — SAC 학습 루프

학습 전략:
  - Warm-up: 초기 N 스텝은 랜덤 행동으로 버퍼 채움
  - Off-policy: 환경 1 스텝 → gradient 1회 (gradient_steps 설정 가능)
  - Periodic Eval: eval_interval마다 결정론적 정책으로 평가
  - Early Stopping: eval Sharpe 기준
  - Curriculum: 쉬운 에피소드 → 어려운 에피소드 (선택적)
"""

import numpy as np
import torch
import logging
import time
from pathlib import Path
from collections import deque
from typing import Optional, Dict, List

from config import TrainConfig, train_cfg, RESULT_DIR
from agent.sac_agent import SACAgent
from agent.replay_buffer import ReplayBuffer
from env.trading_env import TradingEnv
from utils.reward_normalizer import RewardNormalizer

logger = logging.getLogger(__name__)


class SACTrainer:
    """
    SAC 학습 매니저.
    """

    def __init__(
        self,
        train_env: TradingEnv,
        eval_env:  TradingEnv,
        agent: SACAgent,
        buffer: ReplayBuffer,
        cfg: TrainConfig = train_cfg,
    ):
        self.train_env = train_env
        self.eval_env  = eval_env
        self.agent     = agent
        self.buffer    = buffer
        self.cfg       = cfg

        # 로그
        self._ep_returns: deque = deque(maxlen=50)
        self._ep_lengths: deque = deque(maxlen=50)
        self._loss_history: List[Dict] = []
        self._eval_history: List[Dict] = []

        self._best_eval_score = -np.inf
        self._timestep        = 0
        self._episode         = 0
        self._no_improve_count = 0    # early stopping 카운터
        self._stop_training    = False

        # 보상 정규화 (Welford). SAC config 의 normalize_reward 로 on/off.
        if getattr(agent.cfg, "normalize_reward", False):
            kwargs = dict(
                center=getattr(agent.cfg, "reward_norm_center", False),
                clip  =getattr(agent.cfg, "reward_norm_clip", 10.0),
            )
            self._reward_normalizer = RewardNormalizer(**kwargs)
        else:
            self._reward_normalizer = None

    # ── 메인 학습 루프 ─────────────────────────────────

    def train(self) -> List[Dict]:
        """
        전체 학습 루프.

        Returns
        -------
        eval_history : 에포크별 평가 결과 리스트
        """
        logger.info(f"SAC 학습 시작 | device={self.agent.device} | "
                    f"total_steps={self.cfg.total_timesteps:,}")

        obs, _ = self.train_env.reset(seed=self.cfg.seed)
        ep_ret = 0.0
        ep_len = 0
        t_start = time.time()

        for t in range(1, self.cfg.total_timesteps + 1):
            self._timestep = t

            # ── 행동 선택
            if t < self.agent.cfg.min_replay_size:
                # 워밍업: 랜덤 탐험
                action = self.train_env.action_space.sample()
            else:
                action = self.agent.select_action(obs, deterministic=False)

            # ── 환경 스텝
            next_obs, reward, terminated, truncated, info = self.train_env.step(action)
            done = terminated or truncated

            # 보상 정규화 (학습용 신호만 — info["step_ret"] 등 실제 성과는 손대지 않음).
            stored_reward = reward
            if self._reward_normalizer is not None:
                stored_reward = self._reward_normalizer.update_and_normalize(reward)

            # PER 의 β 어닐링용 글로벌 스텝 동기화
            if hasattr(self.buffer, "set_step"):
                self.buffer.set_step(t)

            # 버퍼 저장 (terminated ≠ truncated 구분: time-limit 은 done=False)
            self.buffer.add(obs, action, stored_reward, next_obs, float(terminated))

            obs     = next_obs
            ep_ret += reward
            ep_len += 1

            # ── 에피소드 종료
            if done:
                # n-step 잔여 누적분 flush
                if hasattr(self.buffer, "end_of_episode"):
                    self.buffer.end_of_episode()

                self._ep_returns.append(ep_ret)
                self._ep_lengths.append(ep_len)
                self._episode += 1
                obs, _ = self.train_env.reset()
                ep_ret = 0.0
                ep_len = 0

            # ── Gradient 업데이트
            if t >= self.agent.cfg.min_replay_size and self.buffer.is_ready:
                for _ in range(self.agent.cfg.gradient_steps):
                    losses = self.agent.train_step(self.buffer)
                    self._loss_history.append({**losses, "step": t})

            # ── 로깅
            if t % self.cfg.log_interval == 0 and self._ep_returns:
                elapsed = time.time() - t_start
                fps     = t / elapsed
                mean_ret = np.mean(self._ep_returns)
                mean_len = np.mean(self._ep_lengths)
                recent_losses = self._loss_history[-10:] if self._loss_history else [{}]
                mean_critic = np.mean([l.get("critic_loss", 0) for l in recent_losses])
                mean_actor  = np.mean([l.get("actor_loss", 0)  for l in recent_losses])

                # 부가 진단 (PER β, 보상 정규화 통계)
                extras = ""
                if getattr(self.buffer, "is_per", False):
                    extras += f" | β={self.buffer.beta:.3f}"
                if self._reward_normalizer is not None:
                    extras += (f" | r̂μ={self._reward_normalizer.mean:+.3f}"
                               f" σ={self._reward_normalizer.std:.3f}")

                logger.info(
                    f"[{t:7,d}/{self.cfg.total_timesteps:,}] "
                    f"Ep={self._episode:4d} | "
                    f"Ret={mean_ret:+7.4f} | "
                    f"Len={mean_len:5.0f} | "
                    f"α={self.agent.alpha:.4f} | "
                    f"Q_loss={mean_critic:.4f} | "
                    f"π_loss={mean_actor:.4f} | "
                    f"FPS={fps:.0f}"
                    f"{extras}"
                )

            # ── 평가
            if t % self.cfg.eval_interval == 0:
                eval_result = self._evaluate()
                eval_result["step"] = t
                self._eval_history.append(eval_result)

                bh_ret = eval_result.get("bh_return", 0.0)
                alpha  = eval_result["total_return"] - bh_ret
                logger.info(
                    f"  ── EVAL ── "
                    f"TotalRet={eval_result['total_return']:+.2%} | "
                    f"B&H={bh_ret:+.2%} | "
                    f"α={alpha:+.2%} | "
                    f"Sharpe={eval_result['sharpe']:.3f} | "
                    f"MDD={eval_result['mdd']:.2%} | "
                    f"WinRate={eval_result.get('win_rate', 0):.1%} | "
                    f"Trades={eval_result['mean_trades']:.0f}"
                )

                # 최고 모델 저장 — best_metric 에 따라 다른 기준
                metric_name = getattr(self.cfg, "best_metric", "sharpe")
                score = self._compute_best_metric(eval_result, metric_name)

                # 최소 마진 게이트: score < best_min_margin 이면 갱신 안 함.
                # 예: best_metric="alpha_vs_bh", best_min_margin=0.0 일 때
                # B&H 못 이기는 (α<0) 모델은 "best" 가 되지 않음 → 음수 영역
                # 미세 개선만 잡는 함정 방지.
                margin = float(getattr(self.cfg, "best_min_margin", -np.inf))

                if score > self._best_eval_score and score >= margin:
                    self._best_eval_score = score
                    self._no_improve_count = 0
                    self.agent.save("best_sac")
                    logger.info(f"  ✅ 최고 모델 저장 ({metric_name}={score:+.4f})")
                else:
                    self._no_improve_count += 1
                    if score > self._best_eval_score and score < margin:
                        logger.info(
                            f"  ⚠ best 후보({metric_name}={score:+.4f})지만 "
                            f"min_margin={margin:+.3f} 미달 — 갱신 안 함"
                        )

                # Early stopping
                patience = int(getattr(self.cfg, "early_stop_patience", 0))
                if patience > 0 and self._no_improve_count >= patience:
                    logger.info(
                        f"  ⏹ Early stop: {patience}회 연속 best 갱신 실패 "
                        f"→ step {t:,} 에서 학습 중단"
                    )
                    self._stop_training = True
                    break

            # ── 주기적 저장
            if t % self.cfg.save_interval == 0:
                self.agent.save(f"sac_step_{t}")

            # Early stopping triggered
            if self._stop_training:
                break

        metric_name = getattr(self.cfg, "best_metric", "sharpe")
        logger.info(
            f"\n학습 완료 ({self._timestep:,}/{self.cfg.total_timesteps:,} step) | "
            f"최고 {metric_name}: {self._best_eval_score:+.4f}"
        )
        return self._eval_history

    # ── Best Metric 계산 ──────────────────────────────

    def _compute_best_metric(self, eval_result: Dict, name: str) -> float:
        """
        Best 모델 선택 기준.

        - "sharpe"      : 그대로 eval Sharpe
        - "alpha_vs_bh" : agent total_return - B&H total_return
                          (B&H 못 이기는 모델은 best 안 됨)
        - "calmar"      : total_return / |MDD| — 위험 조정 수익률
        """
        if name == "alpha_vs_bh":
            # eval_result["bh_return"] 가 있으면 우선 사용 (eval_random_start 와 일치).
            # 없으면 (legacy) 현재 env 상태에서 즉시 계산.
            try:
                if "bh_return" in eval_result:
                    bh_return = float(eval_result["bh_return"])
                else:
                    bh_return = self._compute_bh_return()
                return float(eval_result.get("total_return", 0) - bh_return)
            except Exception:
                return float(eval_result.get("sharpe", -np.inf))
        elif name == "calmar":
            ret = eval_result.get("total_return", 0)
            mdd = abs(eval_result.get("mdd", 1e-9))
            return float(ret / (mdd + 1e-9))
        else:    # default: sharpe
            return float(eval_result.get("sharpe", -np.inf))

    def _compute_bh_return(self) -> float:
        """평가 환경의 Buy & Hold total_return (현재 env._start ~ env._step_idx 기준)."""
        env = self.eval_env
        try:
            start_i = env._start
            end_i   = min(env._step_idx, len(env.prices) - 1)
            prices  = env.prices["Close"].iloc[start_i:end_i + 1].values
            if len(prices) < 2:
                return 0.0
            return float(prices[-1] / prices[0] - 1)
        except Exception:
            return 0.0

    def _episode_bh_return(self) -> float:
        """방금 끝난 에피소드의 B&H — eval_random_start 일관성용 별칭."""
        return self._compute_bh_return()

    # ── 평가 ──────────────────────────────────────────

    def _evaluate(self) -> Dict:
        """
        결정론적 정책으로 평가.

        단일 종목 환경에서 eval_episodes 만큼 결정론적 평가를 수행한다.

        Sharpe 는 실제 수익률(step_ret) 기반으로 일관되게 계산.
        """
        all_step_rets = []
        ep_capitals   = []
        ep_mdds       = []
        ep_trades     = []
        ep_bh_returns: List[float] = []   # eval_random_start 와 일치하는 per-episode B&H

        episode_plan = [None for _ in range(self.cfg.eval_episodes)]

        for _ in episode_plan:
            obs, _ = self.eval_env.reset()
            done   = False
            step_rets = []

            while not done:
                action = self.agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                step_rets.append(info["step_ret"])
                done = terminated or truncated

            all_step_rets.extend(step_rets)
            ep_capitals.append(info["capital"])
            ep_mdds.append(info["mdd"])
            ep_trades.append(info["trade_count"])

            # 이 에피소드 구간의 B&H total return — random-start 와 정확히 일치
            ep_bh_returns.append(self._episode_bh_return())

        # 에피소드 평균 총 수익률
        init = self.eval_env.cfg.initial_capital
        mean_capital = np.mean(ep_capitals)
        total_return = (mean_capital - init) / init

        # Sharpe: 전 에피소드 수익률 시계열 기반 (연율화)
        rets   = np.array(all_step_rets)
        sharpe = (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252) if len(rets) > 1 else 0.0

        # Sortino 추가
        downside    = rets[rets < 0]
        down_std    = downside.std() + 1e-8 if len(downside) > 0 else rets.std() + 1e-8
        sortino     = (rets.mean() / down_std) * np.sqrt(252) if len(rets) > 1 else 0.0

        # B&H 평균 (eval_random_start 활성 시 episode 별로 다른 시작점이므로 평균)
        mean_bh = float(np.mean(ep_bh_returns)) if ep_bh_returns else 0.0

        result = {
            "mean_return":  float(rets.mean()),   # 일평균 수익률
            "total_return": float(total_return),
            "sharpe":       float(sharpe),
            "sortino":      float(sortino),
            "mdd":          float(np.mean(ep_mdds)),
            "mean_trades":  float(np.mean(ep_trades)),
            "win_rate":     float((rets > 0).mean()),
            "bh_return":    mean_bh,
        }
        return result

    # ── 결과 저장 ──────────────────────────────────────

    def save_results(self):
        import json
        path = RESULT_DIR / "training_log.json"
        with open(path, "w") as f:
            json.dump({
                "eval_history": self._eval_history,
                "loss_history": self._loss_history[-1000:],  # 최근 1000개만
            }, f, indent=2)
        logger.info(f"학습 결과 저장: {path}")
