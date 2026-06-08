"""
utils/reward_normalizer.py — Welford 온라인 보상 정규화

목적:
  SAC critic은 보상 분포의 스케일에 매우 민감하다. Sharpe 기반 보상은
  std → 0 구간에서 값이 폭증하기 때문에, 클리핑만으로는 critic이
  Q값을 일관되게 학습하기 어렵다. 학습 도중 누적 통계를 갱신하며
  보상을 정규화하면 actor_loss 발산을 안정적으로 억제할 수 있다.

알고리즘:
  Welford (1962) 의 numerically stable online variance:
      n      ← n + 1
      δ      ← x - μ
      μ     += δ / n
      M₂    += δ · (x - μ)
      σ²     = M₂ / (n - 1)

  → 정규화: r̂ = clip( (r - μ) / σ , -clip, +clip )
     (실전에서는 평균-중심화 없이 std로 나누기만 쓰는 변형도 흔함:
      OpenAI baselines / Engstrom et al., 2020)

참고:
  - Engstrom et al. "Implementation Matters in Deep Policy Gradients" (2020)
  - Welford "Note on a method for calculating corrected sums of
    squares and products" (1962)
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class RewardNormalizerState:
    n: int       = 0
    mean: float  = 0.0
    m2: float    = 0.0


class RewardNormalizer:
    """
    Welford 온라인 평균/분산 추정 + 옵션 기반 정규화.

    옵션:
      center : True  → (r - μ) / σ   (배치-norm 풍, 평균 0 강제)
               False → r / σ          (분산만 정규화 — PPO 구현에서 흔함)
      clip   : 정규화 후 양극 클리핑 (∞ 발산 방지)
    """

    def __init__(
        self,
        center: bool = False,
        clip: float = 10.0,
        epsilon: float = 1e-8,
    ):
        self.center  = center
        self.clip    = clip
        self.epsilon = epsilon
        self._state  = RewardNormalizerState()
        self._announced = False

    # ── 통계 갱신 ─────────────────────────────────────

    def update(self, reward: float) -> None:
        s = self._state
        s.n += 1
        delta = reward - s.mean
        s.mean += delta / s.n
        s.m2  += delta * (reward - s.mean)

    def update_batch(self, rewards) -> None:
        for r in np.asarray(rewards).flatten():
            self.update(float(r))

    # ── 정규화 ───────────────────────────────────────

    def normalize(self, reward: float) -> float:
        if not self._announced:
            print("Running reward normalization using Welford's online algorithm")
            self._announced = True

        s = self._state
        if s.n < 2:
            return float(reward)

        var = s.m2 / (s.n - 1)
        std = math.sqrt(max(var, 0.0)) + self.epsilon

        r = (reward - s.mean) / std if self.center else reward / std
        if self.clip is not None:
            r = max(-self.clip, min(self.clip, r))
        return float(r)

    def update_and_normalize(self, reward: float, ticker: str = None) -> float:
        """
        Parameters
        ----------
        ticker : 단일 종목 normalizer 는 무시 (PerTickerRewardNormalizer 와 호환 시그니처).
        """
        self.update(reward)
        return self.normalize(reward)

    # ── 조회 ─────────────────────────────────────────

    @property
    def mean(self) -> float:
        return self._state.mean

    @property
    def std(self) -> float:
        s = self._state
        if s.n < 2:
            return 1.0
        return math.sqrt(max(s.m2 / (s.n - 1), 0.0))

    @property
    def count(self) -> int:
        return self._state.n

    # ── 직렬화 ───────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "n":     self._state.n,
            "mean":  self._state.mean,
            "m2":    self._state.m2,
            "center": self.center,
            "clip":  self.clip,
        }

    def load_state_dict(self, sd: dict) -> None:
        self._state.n    = int(sd.get("n", 0))
        self._state.mean = float(sd.get("mean", 0.0))
        self._state.m2   = float(sd.get("m2", 0.0))
        if "center" in sd: self.center = bool(sd["center"])
        if "clip"   in sd: self.clip   = float(sd["clip"])


# ──────────────────────────────────────────────────────
# 종목별 Reward Normalizer (Multi-Ticker)
# ──────────────────────────────────────────────────────

class PerTickerRewardNormalizer:
    """
    종목별 독립 Welford 정규화기.

    동기:
      자산군이 다른 종목들(SPY vs TLT vs GLD)을 함께 학습할 때, 한 글로벌
      통계로 정규화하면 저변동성 종목(TLT)의 학습 신호가 고변동성 종목(SPY)에
      묻혀버린다. 종목별로 (μ, σ) 를 따로 유지하면 모든 종목의 reward 가
      유사한 스케일로 critic 에 전달된다.

    참고:
      - Engstrom et al. "Implementation Matters in Deep Policy Gradients" (ICLR 2020)
        reward scaling 이 RL 성능에 미치는 영향을 정량화. multi-task / multi-asset
        설정에서 task-wise 정규화의 필요성을 시사.

    호환 인터페이스:
      RewardNormalizer 와 동일하게 update_and_normalize(reward, ticker) 시그니처를
      따르므로, 트레이너 코드는 어느 쪽을 받든 동일하게 호출 가능.
    """

    DEFAULT_KEY = "_default"

    def __init__(
        self,
        center: bool = False,
        clip: float = 10.0,
        epsilon: float = 1e-8,
    ):
        self._defaults = dict(center=center, clip=clip, epsilon=epsilon)
        self._normalizers: dict = {}
        # 호환용: 단일 normalizer 처럼 mean/std 를 묻는 코드가 깨지지 않도록 평균값 반환
        self._announced = False

    # ── 갱신 + 정규화 ─────────────────────────────────

    def _get_or_create(self, ticker: str) -> RewardNormalizer:
        key = ticker if ticker is not None else self.DEFAULT_KEY
        if key not in self._normalizers:
            self._normalizers[key] = RewardNormalizer(**self._defaults)
        return self._normalizers[key]

    def update(self, reward: float, ticker: str = None) -> None:
        self._get_or_create(ticker).update(reward)

    def normalize(self, reward: float, ticker: str = None) -> float:
        return self._get_or_create(ticker).normalize(reward)

    def update_and_normalize(self, reward: float, ticker: str = None) -> float:
        if not self._announced:
            print("Running per-ticker reward normalization (Welford, dict-keyed)")
            self._announced = True
        return self._get_or_create(ticker).update_and_normalize(reward, ticker)

    # ── 호환 조회 (전체 평균) ───────────────────────────

    @property
    def mean(self) -> float:
        if not self._normalizers:
            return 0.0
        return float(np.mean([n.mean for n in self._normalizers.values()]))

    @property
    def std(self) -> float:
        if not self._normalizers:
            return 1.0
        vals = [n.std for n in self._normalizers.values() if n.count >= 2]
        return float(np.mean(vals)) if vals else 1.0

    @property
    def count(self) -> int:
        return sum(n.count for n in self._normalizers.values())

    # ── 종목별 통계 (모니터링) ──────────────────────────

    def per_ticker_stats(self) -> dict:
        """{ticker: (mean, std, count)} — 학습 로그에 출력."""
        return {
            tk: (n.mean, n.std, n.count)
            for tk, n in self._normalizers.items()
        }

    # ── 직렬화 ───────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "defaults":    self._defaults,
            "normalizers": {tk: n.state_dict() for tk, n in self._normalizers.items()},
        }

    def load_state_dict(self, sd: dict) -> None:
        self._defaults = sd.get("defaults", self._defaults)
        self._normalizers = {}
        for tk, n_sd in sd.get("normalizers", {}).items():
            n = RewardNormalizer(**self._defaults)
            n.load_state_dict(n_sd)
            self._normalizers[tk] = n
