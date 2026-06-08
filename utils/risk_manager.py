"""
utils/risk_manager.py — 실전 리스크 관리

이론적 근거:
  - Kelly Criterion (Kelly, 1956): 최적 배팅 비율
  - Value at Risk (VaR) / CVaR: 꼬리 리스크 측정
  - Regime Detection: 변동성 레짐에 따른 포지션 조정
  - Drawdown Control: MDD 기반 동적 포지션 축소

구성:
  RiskManager.check(signal, portfolio) → 조정된 포지션
  VaRCalculator.compute(returns) → VaR, CVaR
  RegimeFilter.get_regime(price_df) → "bull" | "bear" | "neutral"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)


# ── 리스크 파라미터 ────────────────────────────────────

@dataclass
class RiskConfig:
    max_position:      float = 1.0      # 최대 포지션 크기
    max_drawdown_stop: float = 0.15     # 15% MDD 도달 시 포지션 강제 축소
    var_confidence:    float = 0.95     # VaR 신뢰수준
    var_lookback:      int   = 60       # VaR 계산 윈도우
    regime_filter:     bool  = True     # 레짐 필터 사용 여부
    vol_target:        float = 0.15     # 목표 연간 변동성 (변동성 타기팅)
    use_vol_targeting: bool  = True     # 변동성 타기팅 사용


# ── 메인 리스크 매니저 ─────────────────────────────────

class RiskManager:
    """
    SAC 에이전트 신호를 받아 리스크 조정 포지션을 출력합니다.

    체계:
      1. MDD 기반 동적 포지션 축소 (Drawdown Control)
      2. 변동성 타기팅 (목표 변동성 달성을 위한 스케일링)
      3. 레짐 필터 (Bear 시장에서 롱 포지션 제한)
      4. VaR 한도 (단일 포지션 최대 손실 제한)
    """

    def __init__(self, cfg: RiskConfig = None):
        self.cfg = cfg or RiskConfig()
        self._portfolio_history: list = []
        self._peak_value: float = 1.0

    def adjust(
        self,
        raw_position: float,
        portfolio_value: float,
        price_df: pd.DataFrame,
    ) -> dict:
        """
        SAC 원시 신호를 리스크 조정합니다.

        Parameters
        ----------
        raw_position    : SAC 에이전트 출력 [-1, 1]
        portfolio_value : 현재 포트폴리오 가치
        price_df        : 최근 OHLCV (VaR, 레짐 계산용)

        Returns
        -------
        dict: adjusted_position, risk_factors, warnings
        """
        self._portfolio_history.append(portfolio_value)
        self._peak_value = max(self._peak_value, portfolio_value)

        warnings = []
        multiplier = 1.0

        # ① MDD 기반 포지션 축소
        current_dd = (portfolio_value - self._peak_value) / self._peak_value
        dd_mult = self._drawdown_multiplier(current_dd)
        if dd_mult < 1.0:
            warnings.append(f"MDD 제어: {current_dd:.1%} → 포지션 {dd_mult:.2f}x")
        multiplier *= dd_mult

        rets = np.log(price_df["Close"] / price_df["Close"].shift(1)).dropna()

        # ② 변동성 타기팅
        if self.cfg.use_vol_targeting and len(rets) >= 20:
            vol_mult = self._vol_target_multiplier(rets.values)
            multiplier *= vol_mult

        # ③ 레짐 필터
        regime = "neutral"
        if self.cfg.regime_filter and len(price_df) >= 60:
            regime = RegimeFilter.detect(price_df)
            regime_mult = {"bull": 1.0, "neutral": 0.7, "bear": 0.3}[regime]
            if raw_position > 0 and regime == "bear":
                warnings.append(f"Bear 레짐: 롱 포지션 {regime_mult:.1f}x 축소")
                multiplier *= regime_mult

        # ④ VaR 한도
        if len(rets) >= self.cfg.var_lookback:
            var, cvar = VaRCalculator.compute(
                rets.values[-self.cfg.var_lookback:],
                self.cfg.var_confidence,
            )
            # CVaR 기반 최대 포지션 = 목표손실(5%) / CVaR
            max_pos_var = min(0.05 / (abs(cvar) + 1e-6), self.cfg.max_position)
            if abs(raw_position) * multiplier > max_pos_var:
                warnings.append(f"VaR 한도: CVaR={cvar:.2%} → 최대 포지션 {max_pos_var:.2f}")
                multiplier = min(multiplier, max_pos_var / (abs(raw_position) + 1e-8))

        # 최종 포지션
        adjusted = float(np.clip(raw_position * multiplier,
                                 -self.cfg.max_position,
                                  self.cfg.max_position))

        return {
            "adjusted_position": adjusted,
            "raw_position":      raw_position,
            "multiplier":        round(multiplier, 4),
            "current_dd":        round(current_dd, 4),
            "regime":            regime,
            "warnings":          warnings,
        }

    def _drawdown_multiplier(self, dd: float) -> float:
        """
        Drawdown에 따른 선형 포지션 축소.
        dd=0% → 1.0x, dd=max_dd_stop → 0.1x
        """
        if dd >= 0:
            return 1.0
        ratio = abs(dd) / self.cfg.max_drawdown_stop
        return float(np.clip(1.0 - 0.9 * ratio, 0.1, 1.0))

    def _vol_target_multiplier(self, rets: np.ndarray) -> float:
        """
        변동성 타기팅 (Volatility Targeting).
        현재 변동성 / 목표 변동성 = 포지션 스케일
        목표보다 변동성 높으면 포지션 축소, 낮으면 확대.
        """
        realized_vol = rets[-20:].std() * np.sqrt(252)
        if realized_vol < 1e-6:
            return 1.0
        target_mult = self.cfg.vol_target / realized_vol
        return float(np.clip(target_mult, 0.2, 2.0))

    def reset(self):
        self._portfolio_history = []
        self._peak_value = 1.0


# ── VaR / CVaR 계산기 ─────────────────────────────────

class VaRCalculator:
    """
    Historical Simulation 방식 VaR / CVaR 계산.

    VaR(α): α 신뢰수준에서 최대 손실
    CVaR(α): VaR를 초과하는 손실의 기댓값 (Expected Shortfall)

    CVaR이 VaR보다 우월한 이유:
      - 꼬리 리스크(tail risk)를 직접 측정
      - Subadditivity 성질 (포트폴리오 분산 효과 반영)
    """

    @staticmethod
    def compute(
        returns: np.ndarray,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """
        Parameters
        ----------
        returns    : 수익률 시계열
        confidence : 신뢰수준 (0.95 = 95%)

        Returns
        -------
        (VaR, CVaR) : 둘 다 음수 (손실)
        """
        sorted_rets = np.sort(returns)
        var_idx     = int((1 - confidence) * len(sorted_rets))
        var         = float(sorted_rets[var_idx])
        cvar        = float(sorted_rets[:var_idx].mean()) if var_idx > 0 else var

        return var, cvar

    @staticmethod
    def report(returns: np.ndarray, confidence: float = 0.95) -> str:
        var, cvar = VaRCalculator.compute(returns, confidence)
        return (
            f"VaR({confidence:.0%}): {var:.3%} | "
            f"CVaR({confidence:.0%}): {cvar:.3%}"
        )


# ── 레짐 감지기 ───────────────────────────────────────

class RegimeFilter:
    """
    변동성 및 추세 기반 시장 레짐 감지.

    레짐:
      "bull"    : 저변동성 + 상승 추세
      "bear"    : 고변동성 + 하락 추세
      "neutral" : 그 외

    방법:
      - 단기 vs 장기 이동평균 크로스 (추세)
      - 실현 변동성 vs 장기 평균 변동성 (레짐)
    """

    @staticmethod
    def detect(price_df: pd.DataFrame) -> Literal["bull", "bear", "neutral"]:
        close = price_df["Close"]
        rets  = np.log(close / close.shift(1)).dropna()

        # 추세 신호
        ma20  = close.rolling(20).mean().iloc[-1]
        ma60  = close.rolling(60).mean().iloc[-1]
        trend = "up" if ma20 > ma60 else "down"

        # 변동성 레짐
        recent_vol = rets.iloc[-20:].std() * np.sqrt(252)
        long_vol   = rets.std() * np.sqrt(252)
        vol_regime = "high" if recent_vol > long_vol * 1.3 else "low"

        if trend == "up" and vol_regime == "low":
            return "bull"
        elif trend == "down" and vol_regime == "high":
            return "bear"
        else:
            return "neutral"

    @staticmethod
    def get_history(price_df: pd.DataFrame, window: int = 5) -> pd.Series:
        """롤링 레짐 히스토리."""
        regimes = []
        for i in range(window, len(price_df)):
            sub = price_df.iloc[max(0, i-60):i]
            regimes.append(RegimeFilter.detect(sub))
        return pd.Series(regimes, index=price_df.index[window:])
