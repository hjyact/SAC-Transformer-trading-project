"""
agent/ensemble.py — 다중 SAC 에이전트 앙상블

서로 다른 시드/설정으로 학습한 N개 에이전트를 앙상블하여
개별 에이전트보다 안정적인 정책을 생성합니다.

앙상블 방법:
  1. Mean Voting:   각 에이전트 행동의 단순 평균
  2. Confidence:    행동 분산이 낮을수록 가중치 부여
  3. Rank-based:    검증 성능 기반 가중 평균

참고: Ensemble Methods in Machine Learning (Dietterich, 2000)
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Dict
import logging

from config import SACConfig, sac_cfg, DEVICE, CKPT_DIR

logger = logging.getLogger(__name__)


class SACEnsemble:
    """
    다중 SAC 에이전트 앙상블.

    사용 예:
        ensemble = SACEnsemble(obs_dim, action_dim, n_agents=5)
        ensemble.train_all(train_env, eval_env)
        action = ensemble.select_action(obs)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int = 5,
        base_cfg: SACConfig = sac_cfg,
        device: str = DEVICE,
    ):
        from agent.sac_agent import SACAgent

        self.n_agents   = n_agents
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.device     = device

        # 각 에이전트마다 학습률/hidden 약간 다르게 (다양성 확보)
        self.agents: List[SACAgent] = []
        self.weights: List[float]   = [1.0 / n_agents] * n_agents
        self._sharpes: List[float]  = [0.0] * n_agents

        for i in range(n_agents):
            cfg_i = SACConfig(
                actor_lr    = base_cfg.actor_lr  * (0.5 + i * 0.25),
                critic_lr   = base_cfg.critic_lr * (0.5 + i * 0.25),
                hidden_dims = base_cfg.hidden_dims,
                gamma       = base_cfg.gamma,
                tau         = base_cfg.tau,
                auto_alpha  = base_cfg.auto_alpha,
                buffer_size = base_cfg.buffer_size,
                batch_size  = base_cfg.batch_size,
                min_replay_size = base_cfg.min_replay_size,
            )
            agent = SACAgent(obs_dim, action_dim, cfg_i, device)
            self.agents.append(agent)

        logger.info(f"앙상블 초기화: {n_agents}개 에이전트")

    # ── 행동 선택 ──────────────────────────────────────

    def select_action(
        self,
        obs: np.ndarray,
        method: str = "confidence",
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        앙상블 행동 선택.

        method:
          "mean"       : 단순 평균
          "confidence" : 행동 표준편차 기반 가중 평균 (일관된 에이전트 우선)
          "weighted"   : 검증 Sharpe 기반 가중 평균
        """
        actions = np.array([
            agent.select_action(obs, deterministic=deterministic)
            for agent in self.agents
        ])  # shape: (n_agents, action_dim)

        if method == "mean":
            return actions.mean(axis=0)

        elif method == "confidence":
            # 표준편차가 작을수록 (에이전트들이 동의할수록) 신뢰도 높음
            std   = actions.std(axis=0).mean() + 1e-8
            confs = np.array([
                1.0 / (np.abs(a - actions.mean(axis=0)).mean() + 1e-8)
                for a in actions
            ])
            weights = confs / confs.sum()
            return (actions * weights[:, None]).sum(axis=0)

        elif method == "weighted":
            w = np.array(self.weights)
            w = w / w.sum()
            return (actions * w[:, None]).sum(axis=0)

        return actions.mean(axis=0)

    def update_weights(self, sharpes: List[float]):
        """검증 Sharpe 기반으로 앙상블 가중치 업데이트."""
        self._sharpes = sharpes
        arr = np.array(sharpes)
        arr = arr - arr.min() + 1e-8   # 음수 없앰
        self.weights = (arr / arr.sum()).tolist()
        for i, (w, s) in enumerate(zip(self.weights, sharpes)):
            logger.info(f"  Agent {i}: Sharpe={s:.4f}, weight={w:.4f}")

    # ── 앙상블 학습 ────────────────────────────────────

    def train_all(
        self,
        train_env,
        eval_env,
        n_steps: int = 100_000,
        seed_offset: int = 0,
    ) -> List[float]:
        """
        모든 에이전트를 독립적으로 학습 후 검증 Sharpe 반환.
        """
        from agent.replay_buffer import ReplayBuffer

        sharpes = []
        for i, agent in enumerate(self.agents):
            logger.info(f"\n앙상블 에이전트 {i+1}/{self.n_agents} 학습 중...")
            np.random.seed(seed_offset + i)
            torch.manual_seed(seed_offset + i)

            buffer = ReplayBuffer(self.obs_dim, self.action_dim, agent.cfg.buffer_size)
            sharpe = _train_single(agent, buffer, train_env, eval_env, n_steps)
            sharpes.append(sharpe)
            agent.save(f"ensemble_agent_{i}")
            logger.info(f"  Agent {i} 완료 | Sharpe={sharpe:.4f}")

        self.update_weights(sharpes)
        return sharpes

    def save_all(self):
        for i, agent in enumerate(self.agents):
            agent.save(f"ensemble_agent_{i}")

    def load_all(self):
        from agent.sac_agent import SACAgent
        for i, agent in enumerate(self.agents):
            try:
                agent.load(f"ensemble_agent_{i}")
            except FileNotFoundError:
                logger.warning(f"앙상블 에이전트 {i} 체크포인트 없음")


def _train_single(agent, buffer, train_env, eval_env, n_steps: int) -> float:
    """단일 에이전트 학습 후 Sharpe 반환."""
    obs, _ = train_env.reset()
    for t in range(n_steps):
        if t < agent.cfg.min_replay_size:
            action = train_env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, r, term, trunc, _ = train_env.step(action)
        buffer.add(obs, action, r, next_obs, float(term))
        obs = next_obs
        if term or trunc:
            obs, _ = train_env.reset()

        if t >= agent.cfg.min_replay_size and buffer.is_ready:
            agent.train_step(buffer)

    # 검증 Sharpe
    step_rets = []
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action = agent.select_action(obs, deterministic=True)
        obs, _, term, trunc, info = eval_env.step(action)
        step_rets.append(info["step_ret"])
        done = term or trunc

    rets = np.array(step_rets)
    return float((rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252)) if len(rets) > 1 else 0.0
