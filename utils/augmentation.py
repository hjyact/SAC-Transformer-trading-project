"""
utils/augmentation.py — 학습 전용 시계열 / 상태 증강 유틸

참고 이론:
  ① Time Series Data Augmentation Survey
      - Iwana & Uchida (2021) "An empirical survey of data augmentation
        for time series classification with neural networks." PLOS ONE.
      - Wen et al. (2021) "Time Series Data Augmentation for Deep Learning:
        A Survey." IJCAI.
      → 검증된 두 변형: Jittering (가우시안 노이즈) + Magnitude warping
        (cubic-spline 부드러운 진폭 변형). 의료/금융 시계열에서 가장 자주
        효과 입증.

  ② S4RL (Sinha & Garg, CoRL 2022)
      "S4RL: Surprisingly Simple Self-Supervision for Offline RL"
      → critic 입력 state 에 작은 가우시안 노이즈 → OOD 강건성. SAC 에서
        가장 단순하면서 가장 강건한 variant (논문 §4).

  ③ Domain Randomization
      - Tobin et al. (2017 IROS) "Domain Randomization for Transferring
        Deep Neural Networks from Simulation to the Real World."
      - Peng et al. (2018 ICRA) "Sim-to-Real Transfer of Robotic Control
        with Dynamics Randomization."
      → 환경 파라미터 (마찰, 마스 …) 매 에피소드 randomize → 정책의
        파라미터 불변성. 금융 RL: commission/slippage 변동에 적용.

본 모듈은 모두 학습 시점에만 호출되며 평가/실거래 경로에서는 사용되지
않는다 (TradingEnv.mode == "train" 일 때만 활성).
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ── Magnitude Warping ────────────────────────────────────

def magnitude_warp_curve(
    length: int,
    sigma: float = 0.05,
    n_knots: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Cubic-spline 으로 부드럽게 보간된 진폭 곡선 (length,).

    knot 위치에 (1 + ε_k), ε_k ~ N(0, σ²) 인 K+2 개 random scale 을 두고
    cubic spline 으로 보간. K 가 클수록 더 잦은 굴곡.

    Returns
    -------
    curve : np.ndarray shape (length,), dtype float32
        평균 ≈ 1, 표준편차 ≈ σ 인 부드러운 곡선.

    참고
    ----
    Iwana & Uchida 2021 §3.3; Wen et al. IJCAI 2021 §4.3 (Magnitude domain).
    """
    if rng is None:
        rng = np.random.default_rng()

    k = max(2, int(n_knots))
    knot_x = np.linspace(0, length - 1, k + 2)
    knot_y = 1.0 + rng.normal(0.0, sigma, size=k + 2)

    # scipy 가 있으면 cubic spline, 없으면 선형 보간
    try:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(knot_x, knot_y)
        curve = cs(np.arange(length, dtype=np.float64))
    except ImportError:
        curve = np.interp(np.arange(length), knot_x, knot_y)

    return curve.astype(np.float32)


def magnitude_warp_matrix(
    length: int,
    n_features: int,
    sigma: float = 0.05,
    n_knots: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    피처별 독립 magnitude warp 곡선 (length, n_features).

    피처별 독립 곡선 → spurious cross-feature correlation 학습 약화.
    피처 동기화가 필요하면 같은 곡선을 모든 피처에 적용해도 됨 (현재는 독립).
    """
    if rng is None:
        rng = np.random.default_rng()

    out = np.empty((length, n_features), dtype=np.float32)
    for f in range(n_features):
        out[:, f] = magnitude_warp_curve(
            length, sigma=sigma, n_knots=n_knots, rng=rng,
        )
    return out


# ── Jittering ────────────────────────────────────────────

def jitter(
    x: np.ndarray,
    sigma: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    가우시안 노이즈 추가 (additive). 표준화된 입력 기준 σ ∈ [0.005, 0.02] 권장.

    Iwana & Uchida 2021 §3.1 — 가장 단순하면서도 강력한 시계열 증강.
    Bishop (1995) "Training with noise is equivalent to Tikhonov regularization."
    → 입력에 노이즈 = weight decay 와 등가 (소규모 σ 한정).
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=x.shape).astype(x.dtype)
    return x + noise
