"""
env/trading_env.py — Gymnasium 트레이딩 환경

설계 원칙:
  - 연속 행동 공간 (포지션 비율 [-1, 1])
      -1: 최대 공매도, 0: 현금, +1: 최대 매수
      → SAC의 연속 행동 공간 활용 극대화
      → 포지션 조정 = action - current_position (거래 발생 시에만 수수료)

  - 보상 설계 (Reward Shaping):
      1. PnL 보상: 단순 수익률
      2. Sharpe 보상: 위험 조정 수익 (rolling Sharpe)
      3. Sortino 보상: 하방 위험만 패널티
      4. Mixed: 위 조합 + MDD 패널티 + 거래비용 명시적 반영

  - 관측 공간:
      [window_size × n_features] (기술적 지표 히스토리)
      + [4] 포트폴리오 상태 (position, unrealized_pnl, cash_ratio, holding_time)

  - Look-ahead Bias 방지:
      step()에서는 현재까지의 데이터만 사용
      다음 종가는 action 적용 후에만 접근
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings("ignore")

from config import EnvConfig, env_cfg
from utils.features import compute_portfolio_features


class TradingEnv(gym.Env):
    """
    연속 행동 공간 SAC 트레이딩 환경.

    Observation: (window_size × n_features + 4,) flat vector
    Action:      (1,) ∈ [-1, 1]  →  목표 포지션 비율
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        feature_df: pd.DataFrame,
        price_df: pd.DataFrame,
        cfg: EnvConfig = env_cfg,
        mode: str = "train",   # "train" | "eval"
        feat_mean: Optional[np.ndarray] = None,
        feat_std:  Optional[np.ndarray] = None,
    ):
        """
        Parameters
        ----------
        feature_df : build_all_features()로 만든 피처 DataFrame (NaN 제거 완료)
        price_df   : 원본 OHLCV (체결가 계산용)
        cfg        : EnvConfig
        mode       : train → 랜덤 시작점, eval → 처음부터
        feat_mean, feat_std :
            외부 주입 정규화 통계 (shape (n_feat,)). eval 환경에 train 통계를
            주입하면 test 셋 mean/std 사용으로 인한 look-ahead bias 를 차단.
            None 이면 자체 데이터로 계산 (train 환경의 기본 동작).
        """
        super().__init__()
        self.cfg  = cfg
        self.mode = mode

        # 데이터 정렬
        common_idx   = feature_df.index.intersection(price_df.index)
        # forward-fill로 NaN 처리 (초기 윈도우 기간만 NaN 존재)
        feat_filled  = feature_df.loc[common_idx].ffill().bfill()
        self.features = feat_filled.values.astype(np.float32)
        self.prices   = price_df.loc[common_idx].reset_index(drop=True)
        self.n_steps  = len(self.features)
        self.n_feat   = self.features.shape[1]

        # 피처 정규화 통계
        # 외부 주입 우선 (eval 환경에 train 통계 전달 → look-ahead 차단)
        if feat_mean is not None and feat_std is not None:
            self._feat_mean = np.asarray(feat_mean, dtype=np.float32)
            self._feat_std  = np.asarray(feat_std,  dtype=np.float32) + 1e-8
        else:
            self._feat_mean = np.nanmean(self.features, axis=0).astype(np.float32)
            self._feat_std  = np.nanstd(self.features, axis=0).astype(np.float32) + 1e-8

        # 공간 정의
        obs_dim = cfg.window_size * self.n_feat + 4   # 4 = 포트폴리오 상태
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        # 연속 행동: [-1, 1] 목표 포지션 (0=현금, 1=풀매수, -1=풀공매도)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # 내부 상태
        self._reset_state()

        # 보상 계산용 히스토리 버퍼
        self._ret_history: list = []

    # ── 리셋 ──────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._reset_state()

        # 에피소드 시작점
        # train : 항상 random start (cfg.use_random_start 기본 True)
        # eval  : cfg.eval_random_start=True 이면 random start (시간 OOS 유지하면서
        #         test 셋 내 다양한 시작점으로 평균 → OOS 통계 신뢰성 ↑).
        min_start = self.cfg.window_size
        is_train_random = (self.mode == "train" and self.cfg.use_random_start)
        is_eval_random  = (self.mode == "eval"  and
                           getattr(self.cfg, "eval_random_start", False))
        if is_train_random or is_eval_random:
            max_start = max(min_start + 1, self.n_steps - self.cfg.episode_length - 1)
            self._start = int(self.np_random.integers(min_start, max_start))
        else:
            self._start = min_start

        self._step_idx = self._start
        self._end      = min(self._start + self.cfg.episode_length, self.n_steps - 1)

        # ── Train-time data augmentation 준비 (eval/실거래 경로 영향 X)
        self._setup_episode_augmentation()

        obs = self._get_obs()
        return obs, {}

    def _setup_episode_augmentation(self) -> None:
        """
        에피소드 단위 데이터 증강 상태 초기화.

        ① Domain Randomization (Tobin 2017 IROS, Peng 2018 ICRA)
            commission/slippage 를 base ± domain_rand_pct 범위로 랜덤화.
            정책이 거래비용 변동에 둔감해지도록 유도.

        ② Magnitude Warping (Iwana 2021, Wen 2021)
            window 가 보는 구간 전체 길이의 cubic-spline 곡선을 미리 샘플링
            후 _get_obs() 가 슬라이딩하면서 사용. 에피소드 내 일관 (시점간
            보간 연속성 유지) 하면서 매 에피소드 다른 진폭 패턴.
        """
        # 기본값 (eval / 비활성 시)
        self._eff_commission = self.cfg.commission
        self._eff_slippage   = self.cfg.slippage
        self._warp_curve     = None
        self._warp_offset    = 0

        if self.mode != "train":
            return

        # ① Domain randomization
        if getattr(self.cfg, "use_domain_rand", False):
            pct = float(self.cfg.domain_rand_pct)
            self._eff_commission = float(
                self.cfg.commission * (1.0 + self.np_random.uniform(-pct, pct))
            )
            self._eff_slippage = float(
                self.cfg.slippage * (1.0 + self.np_random.uniform(-pct, pct))
            )

        # ② Magnitude warping
        if getattr(self.cfg, "use_magnitude_warp", False):
            from utils.augmentation import magnitude_warp_matrix
            self._warp_offset = self._start - self.cfg.window_size
            warp_len = (self._end - self._warp_offset) + 2  # +1 margin
            self._warp_curve = magnitude_warp_matrix(
                length     = max(warp_len, self.cfg.window_size + 1),
                n_features = self.n_feat,
                sigma      = float(self.cfg.mag_warp_sigma),
                n_knots    = int(self.cfg.mag_warp_knots),
                rng        = self.np_random,
            )

    def _reset_state(self):
        self._start       = self.cfg.window_size
        self._step_idx    = self._start
        self._end         = self.n_steps - 1
        self.capital      = self.cfg.initial_capital
        self.position     = 0.0      # [-1, 1] 현재 포지션
        self.prev_action  = 0.0      # 직전 step 의 행동 (action-change penalty 용)
        self.cash         = self.cfg.initial_capital
        self.holdings     = 0.0      # 보유 주식 금액
        self.entry_price  = 0.0
        self.holding_steps = 0
        self.peak_capital = self.cfg.initial_capital
        self._ret_history = []
        self._trade_count = 0

        # DSR 상태 (Moody & Saffell 2001 §3.2):
        #   A_t : 수익률 EWMA 1차 모멘트
        #   B_t : 수익률 EWMA 2차 모멘트
        #   D_t = (B_{t-1}·ΔA - 0.5·A_{t-1}·ΔB) / (B_{t-1} - A_{t-1}²)^1.5
        self._dsr_A = 0.0
        self._dsr_B = 0.0

        # Train-time augmentation state — reset() 에서 _setup_episode_augmentation()
        # 이 덮어쓰지만, step() 호출이 reset() 전에 일어나도 안전하도록 초기화.
        self._eff_commission = self.cfg.commission
        self._eff_slippage   = self.cfg.slippage
        self._warp_curve     = None
        self._warp_offset    = 0

    # ── 스텝 ──────────────────────────────────────────

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        action: ndarray shape (1,), 목표 포지션 비율 ∈ [-1, 1]

        순서:
          1. 목표 포지션 결정
          2. 현재→목표 포지션 조정 (수수료 부과)
          3. 다음 봉 종가로 PnL 계산
          4. 보상 계산
          5. 다음 관측 반환
        """
        target_pos = float(np.clip(action[0], -1.0, 1.0))
        target_pos = np.clip(target_pos, -self.cfg.max_position, self.cfg.max_position)

        # 현재 시점 가격 (체결가 = 현재 봉 종가 + 슬리피지)
        # train 모드는 _eff_slippage / _eff_commission 사용 (domain randomization).
        cur_close  = float(self.prices.iloc[self._step_idx]["Close"])
        exec_price = cur_close * (1 + np.sign(target_pos - self.position) * self._eff_slippage)

        # 포지션 변화량
        pos_delta  = target_pos - self.position
        trade_cost = abs(pos_delta) * self.capital * self._eff_commission

        # 포지션 업데이트
        self.position = target_pos
        self.cash     = self.capital * (1 - target_pos) - trade_cost
        self.holdings = self.capital * target_pos

        if abs(pos_delta) > 0.01:
            self._trade_count += 1
            if target_pos != 0:
                self.entry_price = exec_price
            self.holding_steps = 0
        else:
            self.holding_steps += 1

        # 다음 봉 종가로 PnL
        self._step_idx += 1
        next_close = float(self.prices.iloc[self._step_idx]["Close"])
        price_ret  = (next_close - cur_close) / (cur_close + 1e-9)

        # 포지션 수익 (롱: 상승 이익, 숏: 하락 이익)
        position_pnl = self.position * price_ret * self.capital
        self.capital  = max(self.capital + position_pnl - trade_cost, 1.0)
        self.holdings = self.capital * abs(self.position)

        # 최고점 업데이트 (MDD 계산용)
        self.peak_capital = max(self.peak_capital, self.capital)

        # 보상 계산
        # step_ret: 현재 자본 대비 수익률 (복리 기준, 단위 정규화)
        step_ret = position_pnl / (self.capital + 1e-9)
        self._ret_history.append(step_ret)

        # 행동 변화량: 현재 step 의 입력 action 과 직전 입력의 차
        action_delta = abs(target_pos - self.prev_action)
        reward = self._compute_reward(step_ret, trade_cost, action_delta)
        self.prev_action = target_pos

        # 종료 조건
        terminated = self.capital <= self.cfg.initial_capital * 0.5  # 50% 손실
        truncated  = self._step_idx >= self._end

        obs  = self._get_obs()
        info = self._get_info(step_ret, trade_cost, next_close)

        return obs, reward, terminated, truncated, info

    # ── 보상 함수 ──────────────────────────────────────

    def _compute_reward(self, step_ret: float, trade_cost: float,
                        action_delta: float = 0.0) -> float:
        """
        보상 설계 원칙:
          - 모든 보상은 [-5, +5] 범위 내로 클리핑
          - step_ret은 이미 현재 자본 대비 수익률 (소수점 단위)
          - Sharpe 계산: 최근 20 스텝 롤링 (너무 길면 gradient signal 약해짐)
          - 거래비용 패널티: 실제 cost를 자본 대비 비율로 정규화

        action_change_penalty (Mysore CAPS 2021 의 env-level 변형):
          - actor loss 에 넣은 λ_T 가 Q값(~150)에 묻히는 문제를 우회
          - 환경 보상에 직접 -λ·|Δaction| 차감
          - λ ≈ 0.5~1.0 권장 (보상 스케일이 [-5,5] 이므로)
        """
        cfg = self.cfg

        # 행동 변화 패널티 (모든 reward_type 에 공통 적용)
        a_pen = getattr(cfg, "action_change_penalty", 0.0) * float(action_delta)

        # 항상 수행: 거래비용 패널티 (포지션 변동 시에만 유의미)
        cost_norm = trade_cost / (self.capital + 1e-9)  # 자본 대비 비율

        if cfg.reward_type == "pnl":
            reward = (step_ret - cost_norm) * cfg.reward_scaling - a_pen
            return float(np.clip(reward, -5.0, 5.0))

        rets = np.array(self._ret_history[-20:])
        if len(rets) < 5:
            return float(np.clip(step_ret * cfg.reward_scaling - a_pen, -5.0, 5.0))

        mean_ret = rets.mean()
        std_ret  = rets.std() + 1e-8

        if cfg.reward_type == "sharpe":
            sharpe = mean_ret / std_ret
            reward = sharpe * cfg.reward_scaling - cost_norm * 10.0 - a_pen
            return float(np.clip(reward, -5.0, 5.0))

        elif cfg.reward_type == "sortino":
            downside = rets[rets < 0]
            down_std = downside.std() + 1e-8 if len(downside) > 0 else std_ret
            sortino  = mean_ret / down_std
            reward   = sortino * cfg.reward_scaling - cost_norm * 10.0 - a_pen
            return float(np.clip(reward, -5.0, 5.0))

        elif cfg.reward_type == "mixed":
            sharpe = mean_ret / std_ret
            mdd_ratio = (self.capital - self.peak_capital) / (self.peak_capital + 1e-9)
            mdd_pen   = cfg.drawdown_penalty * min(mdd_ratio, 0.0)
            cost_pen  = -cfg.risk_penalty * cost_norm * 100.0
            leverage_pen = -0.02 * max(0.0, abs(self.position) - 0.8) ** 2

            reward = (
                sharpe * cfg.reward_scaling
                + mdd_pen
                + cost_pen
                + leverage_pen
                - a_pen
            )
            return float(np.clip(reward, -5.0, 5.0))

        elif cfg.reward_type in ("dsr", "dsr_cvar"):
            # ── Differential Sharpe Ratio (Moody & Saffell 2001) ──
            # 매 step 의 Sharpe 증분. EWMA 로 1, 2차 모멘트 추적:
            #   ΔA = R_t - A_{t-1}
            #   ΔB = R_t² - B_{t-1}
            #   D_t = (B_{t-1}·ΔA - 0.5·A_{t-1}·ΔB) / (B_{t-1} - A_{t-1}²)^1.5
            #   A_t = A_{t-1} + η·ΔA
            #   B_t = B_{t-1} + η·ΔB
            R = step_ret
            eta = float(getattr(cfg, "dsr_eta", 0.01))
            A_prev = self._dsr_A
            B_prev = self._dsr_B

            dA = R - A_prev
            dB = R * R - B_prev

            denom_sq = B_prev - A_prev * A_prev
            if denom_sq <= 1e-9:
                # 초기화 직후 또는 분산 0 — DSR 정의 X, step_ret 으로 대체
                dsr = R
            else:
                denom = np.power(denom_sq, 1.5)
                dsr = (B_prev * dA - 0.5 * A_prev * dB) / (denom + 1e-9)

            # 상태 업데이트
            self._dsr_A = A_prev + eta * dA
            self._dsr_B = B_prev + eta * dB

            # DSR 자체는 보통 매우 작은 값 (10^-3 ~ 10^-1) → 100x 스케일
            reward = float(dsr) * 100.0 - cost_norm * 10.0 - a_pen

            # CVaR 5% 패널티 (Coache & Jaimungal 2021)
            if cfg.reward_type == "dsr_cvar":
                cvar_window = int(getattr(cfg, "cvar_window", 50))
                cvar_alpha  = float(getattr(cfg, "cvar_alpha", 0.05))
                cvar_lambda = float(getattr(cfg, "cvar_lambda", 1.0))

                window_rets = np.array(self._ret_history[-cvar_window:])
                if len(window_rets) >= 10:
                    # CVaR_α = -E[R | R ≤ VaR_α]   (값이 클수록 꼬리 손실 큼)
                    var_thr = np.quantile(window_rets, cvar_alpha)
                    tail = window_rets[window_rets <= var_thr]
                    if len(tail) > 0:
                        cvar = -float(tail.mean())    # 양수 = 꼬리 손실
                        reward -= cvar_lambda * cvar * 50.0

            return float(np.clip(reward, -5.0, 5.0))

        return float(np.clip(step_ret * cfg.reward_scaling - a_pen, -5.0, 5.0))

    # ── 관측 ──────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        [window_size × n_feat] 피처 히스토리 + 포트폴리오 상태 4개
        → flat vector

        학습 모드에선 (Iwana 2021, Wen 2021):
          - Magnitude warping  : window_norm 에 cubic-spline 곡선 곱셈
          - Jittering          : window_norm 에 가우시안 노이즈 가산
        포트폴리오 상태 4개는 실제 상태 정보이므로 손대지 않는다.
        """
        start = self._step_idx - self.cfg.window_size
        end   = self._step_idx

        window = self.features[start:end]  # (window_size, n_feat)

        # 표준화
        window_norm = (window - self._feat_mean) / self._feat_std

        # ── Train-time augmentation (eval/실거래는 그대로 통과)
        if self.mode == "train":
            # Magnitude warping (multiplicative on standardized window)
            if self._warp_curve is not None:
                rs = start - self._warp_offset
                re = rs + self.cfg.window_size
                if 0 <= rs and re <= self._warp_curve.shape[0]:
                    window_norm = window_norm * self._warp_curve[rs:re]

            # Jittering (additive Gaussian noise)
            if getattr(self.cfg, "use_obs_jitter", False):
                sigma = float(self.cfg.obs_jitter_sigma)
                if sigma > 0.0:
                    noise = self.np_random.normal(
                        0.0, sigma, size=window_norm.shape
                    ).astype(np.float32)
                    window_norm = window_norm + noise

        window_norm = np.clip(window_norm, -10.0, 10.0)
        window_norm = np.nan_to_num(window_norm, nan=0.0)

        # 포트폴리오 상태
        cur_close        = float(self.prices.iloc[self._step_idx]["Close"])
        unrealized_pnl   = (cur_close - self.entry_price) / (self.entry_price + 1e-9) if self.entry_price > 0 else 0.0
        cash_ratio       = max(self.cash, 0) / (self.capital + 1e-9)

        port_feat = compute_portfolio_features(
            position=self.position,
            unrealized_pnl_pct=unrealized_pnl,
            cash_ratio=cash_ratio,
            holding_steps=self.holding_steps,
        )

        obs = np.concatenate([window_norm.flatten(), port_feat])
        return obs.astype(np.float32)

    def _get_info(self, step_ret, trade_cost, cur_price) -> Dict:
        total_ret = (self.capital - self.cfg.initial_capital) / self.cfg.initial_capital
        mdd = (self.capital - self.peak_capital) / (self.peak_capital + 1e-9)
        return {
            "capital":    self.capital,
            "position":   self.position,
            "step_ret":   step_ret,
            "total_ret":  total_ret,
            "mdd":        mdd,
            "trade_cost": trade_cost,
            "trade_count": self._trade_count,
            "cur_price":  cur_price,
        }

    # ── 렌더링 ─────────────────────────────────────────

    def render(self):
        total_ret = (self.capital - self.cfg.initial_capital) / self.cfg.initial_capital
        mdd = (self.capital - self.peak_capital) / (self.peak_capital + 1e-9)
        print(
            f"Step {self._step_idx:4d} | "
            f"Capital: {self.capital:>12,.0f} | "
            f"Ret: {total_ret:>+7.2%} | "
            f"Pos: {self.position:>+5.2f} | "
            f"MDD: {mdd:>7.2%} | "
            f"Trades: {self._trade_count}"
        )
