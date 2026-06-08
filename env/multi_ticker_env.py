"""
env/multi_ticker_env.py — 다종목 트레이딩 환경 (Multi-Ticker Trading Environment)

설계:
  - 종목당 TradingEnv 1개를 보유 (총 N개)
  - 에피소드 = 단일 종목 (가격 점프 0, n-step return 종목 경계 안 넘음)
  - train 모드: reset() 시 종목을 균일 무작위 선택 → catastrophic forgetting 방지
  - eval  모드: reset(options={"ticker": "SPY"}) 로 명시 지정, 또는 라운드로빈

이론적 배경:
  - 도메인 랜덤화 (Tobin et al., 2017 IROS) — 학습 분포를 인위적으로 확장하면
    OOD 일반화가 좋아진다. 종목 랜덤 샘플링은 자산 차원 도메인 랜덤화.
  - Multi-task RL (Yu et al., 2020 NeurIPS, PCGrad) — 다중 태스크에서 균일
    샘플링이 단순하면서도 강력. 가중치는 gradient conflict 측정 후 결정.
  - Catastrophic forgetting (Kirkpatrick et al., 2017 PNAS) — 순차 학습 시
    이전 태스크 망각. 균일 인터리빙으로 자연 회피.

부작용 방지:
  1. obs/action space 일관성 검증 (모든 종목 동일 차원 필수)
  2. step() 의 info 에 "ticker" 키 추가 — 디버깅 + 종목별 보상 정규화에 사용
  3. reset 의 seed 는 자신의 np_random 만 갱신 (하위 env 의 reset 시드와 분리)
  4. 평가용 일관 순회를 위한 rotate_tickers() 헬퍼 제공
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from config import EnvConfig, env_cfg
from env.trading_env import TradingEnv

logger = logging.getLogger(__name__)


class MultiTickerEnv(gym.Env):
    """
    여러 TradingEnv 를 래핑하여 한 에피소드 = 한 종목 단위로 학습.

    Parameters
    ----------
    ticker_data : Dict[ticker_name, (feat_df, price_df)]
        각 종목의 피처/가격 DataFrame.
    cfg         : EnvConfig
        모든 종목이 공유하는 환경 설정 (window/episode/reward 등).
    mode        : "train" | "eval"
        train  → reset 시 종목 무작위 선택, 랜덤 시작점
        eval   → reset 시 options["ticker"] 또는 순차 선택, 처음부터
    sample_weights : Optional[Dict[ticker, float]]
        train 모드 종목 샘플링 가중치. None 이면 균일.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        ticker_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
        cfg: EnvConfig = env_cfg,
        mode: str = "train",
        sample_weights: Optional[Dict[str, float]] = None,
        feat_stats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        super().__init__()
        if not ticker_data:
            raise ValueError("ticker_data 가 비어 있습니다")

        self.cfg = cfg
        self.mode = mode

        # 종목별 TradingEnv 생성
        # feat_stats 가 주어지면 (eval 환경에 train 통계 주입 등) 종목별 stats 사용 →
        # test 셋 mean/std 사용으로 인한 look-ahead bias 차단.
        self.envs: Dict[str, TradingEnv] = {}
        for tk, (feat_df, price_df) in ticker_data.items():
            stats = feat_stats.get(tk) if feat_stats else None
            if stats is not None:
                mean_arr, std_arr = stats
                self.envs[tk] = TradingEnv(
                    feat_df, price_df, cfg, mode=mode,
                    feat_mean=mean_arr, feat_std=std_arr,
                )
            else:
                self.envs[tk] = TradingEnv(feat_df, price_df, cfg, mode=mode)

        self.tickers: List[str] = list(self.envs.keys())

        # obs/action space 일관성 검증 (모든 종목 동일 차원 필수)
        ref_tk = self.tickers[0]
        ref_obs_space = self.envs[ref_tk].observation_space
        ref_act_space = self.envs[ref_tk].action_space
        for tk in self.tickers[1:]:
            if self.envs[tk].observation_space.shape != ref_obs_space.shape:
                raise RuntimeError(
                    f"[{tk}] obs_space {self.envs[tk].observation_space.shape} "
                    f"!= ref {ref_obs_space.shape}. 피처 차원 불일치"
                )
            if self.envs[tk].action_space.shape != ref_act_space.shape:
                raise RuntimeError(
                    f"[{tk}] action_space 불일치"
                )

        self.observation_space = ref_obs_space
        self.action_space      = ref_act_space

        # 샘플링 가중치 정규화
        if sample_weights is None:
            self._sample_probs = np.ones(len(self.tickers)) / len(self.tickers)
        else:
            w = np.array([sample_weights.get(tk, 1.0) for tk in self.tickers], dtype=np.float64)
            if (w <= 0).any():
                raise ValueError("sample_weights 는 모두 양수여야 합니다")
            self._sample_probs = w / w.sum()

        # 현재 활성 종목 / 라운드로빈 인덱스 (eval 용)
        self._current_ticker: str = self.tickers[0]
        self._rr_idx: int = 0

        # 통계 (디버깅)
        self._episode_count: Dict[str, int] = {tk: 0 for tk in self.tickers}

        logger.info(
            f"MultiTickerEnv 초기화: {len(self.tickers)} tickers, mode={mode}, "
            f"obs_dim={ref_obs_space.shape[0]}, action_dim={ref_act_space.shape[0]}"
        )

    # ── 활성 환경 프록시 ─────────────────────────────────

    @property
    def current_env(self) -> TradingEnv:
        return self.envs[self._current_ticker]

    @property
    def current_ticker(self) -> str:
        return self._current_ticker

    # evaluator.py 등 외부 코드가 TradingEnv 의 내부 상태에 직접 접근하는 경우를 위해
    # 활성 종목의 속성을 프록시. 새 속성 필요시 여기 추가.
    @property
    def _start(self) -> int:
        return self.current_env._start

    @property
    def _step_idx(self) -> int:
        return self.current_env._step_idx

    @property
    def prices(self):
        return self.current_env.prices

    # ── 리셋 ──────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        에피소드 시작 시 종목 선택.

        options 키:
          "ticker"        : 특정 종목 강제 지정 (eval 에 유용)
          "rotate"        : True 면 라운드로빈으로 다음 종목 (eval 일관 순회)

        options 가 없는 경우:
          train 모드: 균일(또는 가중) 무작위 선택
          eval  모드: self._current_ticker 유지 (호출 전 set_ticker() 권장)
        """
        super().reset(seed=seed)

        opts = options or {}
        if "ticker" in opts:
            tk = opts["ticker"]
            if tk not in self.envs:
                raise KeyError(f"unknown ticker '{tk}'. 등록된 종목: {self.tickers}")
            self._current_ticker = tk
        elif opts.get("rotate", False):
            self._current_ticker = self.tickers[self._rr_idx % len(self.tickers)]
            self._rr_idx += 1
        elif self.mode == "train":
            # 균일(또는 가중) 무작위 선택
            idx = int(self.np_random.choice(len(self.tickers), p=self._sample_probs))
            self._current_ticker = self.tickers[idx]
        # else (eval, no options): self._current_ticker 유지

        self._episode_count[self._current_ticker] += 1

        # 하위 env 의 reset 에는 자체 seed 만 전달 (선택적)
        obs, info = self.current_env.reset(seed=seed)
        info["ticker"] = self._current_ticker
        return obs, info

    def set_ticker(self, ticker: str) -> None:
        """다음 reset() 호출 시 사용할 종목을 명시적으로 지정.

        run_evaluation() 등 options 인자를 못 받는 외부 코드에서
        종목별 평가를 강제할 때 사용한다.
        """
        if ticker not in self.envs:
            raise KeyError(f"unknown ticker '{ticker}'. 등록된 종목: {self.tickers}")
        self._current_ticker = ticker

    # ── 스텝 ──────────────────────────────────────────

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        # 종목별 reward 정규화 / 디버깅에 활용
        info["ticker"] = self._current_ticker
        return obs, reward, terminated, truncated, info

    # ── 유틸 ──────────────────────────────────────────

    def episode_count_summary(self) -> Dict[str, int]:
        """학습 중 종목별 누적 에피소드 수 (균일성 모니터링용)."""
        return dict(self._episode_count)

    def reset_rotation(self) -> None:
        """라운드로빈 인덱스 초기화 (eval 1바퀴 시작 전)."""
        self._rr_idx = 0

    def get_feat_stats(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        종목별 (feat_mean, feat_std) 사본 반환.
        train MultiTickerEnv 의 통계를 eval MultiTickerEnv 에 주입해
        eval 셋 통계 사용으로 인한 look-ahead bias 를 차단할 때 사용.
        """
        return {
            tk: (env._feat_mean.copy(), env._feat_std.copy())
            for tk, env in self.envs.items()
        }
