"""
utils/signal_generator.py — 실전 신호 생성기

학습된 SAC 모델을 실제 데이터에 적용하여 매매 신호를 생성합니다.

출력:
  - 목표 포지션: [-1.0, 1.0]
  - 행동 신뢰도: 표준편차 (낮을수록 확신)
  - Kelly 기반 포지션 사이징
  - 신호 필터링 (최소 신뢰도 threshold)

⚠️ 실전 사용 시 주의:
  - 본 신호는 교육/연구 목적이며 투자 조언이 아닙니다
  - 실전 적용 전 충분한 검증 및 리스크 관리 필수
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    학습된 SAC 에이전트로 실시간 매매 신호를 생성합니다.

    신호 생성 파이프라인:
      1. 최신 OHLCV 수집
      2. 피처 계산 (window_size 봉 필요)
      3. 관측 벡터 구성 (피처 + 포트폴리오 상태)
      4. 에이전트 추론 (확률론적 N회 → 평균/분산)
      5. Kelly 기반 포지션 사이징
      6. 신호 필터링 및 출력
    """

    def __init__(
        self,
        agent,
        env_cfg,
        feat_cfg,
        train_feature_stats: Optional[Dict] = None,   # 훈련 데이터 mean/std
        n_mc_samples: int = 20,   # Monte Carlo 샘플 수 (신뢰도 추정)
        min_confidence: float = 0.3,   # 신호 발생 최소 신뢰도 (행동 std)
        use_half_kelly: bool = True,   # Half-Kelly 포지션 사이징
    ):
        self.agent      = agent
        self.env_cfg    = env_cfg
        self.feat_cfg   = feat_cfg
        self.stats      = train_feature_stats  # {mean: arr, std: arr}
        self.n_mc       = n_mc_samples
        self.min_conf   = min_confidence
        self.half_kelly = use_half_kelly

        # 포트폴리오 상태 추적
        self.current_position: float = 0.0
        self.entry_price:      float = 0.0
        self.holding_steps:    int   = 0
        self.capital:          float = env_cfg.initial_capital
        self.peak_capital:     float = env_cfg.initial_capital

    def generate(
        self,
        recent_df: pd.DataFrame,
        current_capital: Optional[float] = None,
    ) -> Dict:
        """
        최신 OHLCV 데이터로 신호를 생성합니다.

        Parameters
        ----------
        recent_df      : 최근 window_size + buffer 봉의 OHLCV
        current_capital: 현재 자본 (None이면 내부 추적값 사용)

        Returns
        -------
        dict:
            position    : 목표 포지션 [-1, 1]
            raw_action  : 에이전트 원시 출력
            confidence  : 신호 신뢰도 (1 - 행동 표준편차)
            kelly_size  : Kelly 기준 권장 포지션
            signal      : "BUY" | "SELL" | "HOLD" | "FLAT"
            details     : 상세 정보
        """
        from utils.features import build_all_features, compute_portfolio_features

        if current_capital is not None:
            self.capital = current_capital

        # 피처 계산
        feat_df = build_all_features(recent_df.copy(), self.feat_cfg)
        feat_df = feat_df.ffill().bfill()

        if len(feat_df) < self.env_cfg.window_size:
            logger.warning(f"데이터 부족: {len(feat_df)} < {self.env_cfg.window_size}")
            return self._null_signal("데이터 부족")

        # 마지막 window_size 봉 피처
        window = feat_df.iloc[-self.env_cfg.window_size:].values.astype(np.float32)

        # 정규화
        if self.stats:
            mean = self.stats["mean"]
            std  = self.stats["std"]
        else:
            mean = window.mean(axis=0)
            std  = window.std(axis=0) + 1e-8
        window_norm = np.clip((window - mean) / std, -10.0, 10.0)
        window_norm = np.nan_to_num(window_norm, nan=0.0)

        # 포트폴리오 상태
        cur_price = float(recent_df["Close"].iloc[-1])
        unrealized = (cur_price - self.entry_price) / (self.entry_price + 1e-9) \
                     if self.entry_price > 0 else 0.0
        cash_ratio = max(1.0 - abs(self.current_position), 0.0)

        port_feat = np.array([
            self.current_position,
            np.tanh(unrealized * 10),
            cash_ratio,
            min(self.holding_steps / 252, 1.0),
        ], dtype=np.float32)

        obs = np.concatenate([window_norm.flatten(), port_feat])

        # Monte Carlo 추론 (확률론적 정책으로 N회 샘플링)
        actions = []
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.agent.device)
        with torch.no_grad():
            for _ in range(self.n_mc):
                action, _ = self.agent.actor(obs_t)
                actions.append(action.cpu().numpy().flatten())

        actions    = np.array(actions)   # (n_mc, action_dim)
        mean_action = actions.mean(axis=0)
        std_action  = actions.std(axis=0)

        raw_position = float(mean_action[0])
        action_std   = float(std_action[0])

        # 신뢰도: 행동 표준편차가 낮을수록 높음 (0~1)
        confidence = float(np.exp(-action_std * 3))   # exp 감쇠

        # Kelly 포지션 사이징
        kelly_pos = self._kelly_sizing(raw_position, recent_df)

        # 최종 포지션 결정
        if confidence < self.min_conf:
            final_position = 0.0
            signal_type    = "FLAT"
        else:
            final_position = kelly_pos if self.half_kelly else raw_position
            final_position = np.clip(final_position, -self.env_cfg.max_position,
                                                       self.env_cfg.max_position)
            delta = final_position - self.current_position
            if abs(delta) < 0.05:
                signal_type = "HOLD"
            elif delta > 0:
                signal_type = "BUY"
            else:
                signal_type = "SELL"

        # 상태 업데이트
        self._update_state(final_position, cur_price)

        result = {
            "position":    round(final_position, 4),
            "raw_action":  round(raw_position, 4),
            "confidence":  round(confidence, 4),
            "action_std":  round(action_std, 4),
            "kelly_size":  round(kelly_pos, 4),
            "signal":      signal_type,
            "details": {
                "current_price":    cur_price,
                "current_capital":  self.capital,
                "unrealized_pnl":   round(unrealized, 4),
                "holding_steps":    self.holding_steps,
                "mc_samples":       self.n_mc,
                "action_mean_all":  [round(float(a), 4) for a in actions.mean(axis=0)],
            },
        }

        logger.info(
            f"신호: {signal_type:4s} | "
            f"포지션={final_position:+.3f} | "
            f"신뢰도={confidence:.3f} | "
            f"Kelly={kelly_pos:+.3f} | "
            f"가격={cur_price:.2f}"
        )
        return result

    def _kelly_sizing(self, raw_position: float, recent_df: pd.DataFrame) -> float:
        """
        Kelly Criterion 기반 포지션 사이징.
        최근 수익률 통계로 이론적 최적 베팅 비율 계산.
        Half-Kelly 적용 (실전 권장).
        """
        rets = np.log(recent_df["Close"] / recent_df["Close"].shift(1)).dropna().values[-60:]
        if len(rets) < 10:
            return raw_position * 0.5

        mu  = rets.mean() * 252
        sig = rets.std() * np.sqrt(252)

        # Kelly: f* = μ/σ²
        kelly_raw = mu / (sig**2 + 1e-9)
        kelly_capped = float(np.clip(kelly_raw, -1.0, 1.0))

        if self.half_kelly:
            kelly_capped *= 0.5

        # 에이전트 신호 방향과 Kelly 크기를 결합
        direction = np.sign(raw_position) if abs(raw_position) > 0.1 else 0
        return float(np.clip(direction * abs(kelly_capped), -1.0, 1.0))

    def _update_state(self, new_position: float, cur_price: float):
        if abs(new_position - self.current_position) > 0.05:
            if new_position != 0:
                self.entry_price = cur_price
            self.holding_steps = 0
        else:
            self.holding_steps += 1
        self.current_position = new_position

    def _null_signal(self, reason: str) -> Dict:
        return {
            "position": 0.0, "raw_action": 0.0,
            "confidence": 0.0, "action_std": 1.0,
            "kelly_size": 0.0, "signal": "FLAT",
            "details": {"reason": reason},
        }

    def reset(self):
        """포트폴리오 상태 초기화."""
        self.current_position = 0.0
        self.entry_price      = 0.0
        self.holding_steps    = 0
        self.capital          = self.env_cfg.initial_capital
        self.peak_capital     = self.env_cfg.initial_capital


def extract_feature_stats(feat_df: pd.DataFrame) -> Dict:
    """
    훈련 데이터에서 피처 통계를 추출합니다.
    실전 정규화에 사용 (look-ahead bias 방지).
    """
    return {
        "mean": feat_df.mean(axis=0).values.astype(np.float32),
        "std":  (feat_df.std(axis=0) + 1e-8).values.astype(np.float32),
    }
