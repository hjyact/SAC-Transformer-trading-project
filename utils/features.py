"""
utils/features.py — 이론 기반 피처 엔지니어링

참고 이론:
  - Hurst Exponent (R/S Analysis): 시계열 장기기억 측정
      H > 0.5 → 추세 추종 (trending)
      H < 0.5 → 평균 회귀 (mean-reverting)
      H ≈ 0.5 → 랜덤 워크 (efficient market)

  - Market Microstructure (Roll, 1984):
      Bid-ask spread proxy = 2√(-Cov(ΔP_t, ΔP_{t-1}))

  - Realized Volatility (Andersen & Bollerslev, 1998):
      고빈도 수익률의 제곱합으로 변동성 추정

  - Garman-Klass Volatility (1980):
      OHLC 데이터 활용 변동성 추정 (종가만 쓸 때보다 효율적)

  - Amihud Illiquidity (2002):
      |수익률| / 거래대금 → 유동성 프록시
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


def build_all_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    OHLCV → 전체 피처 DataFrame 반환.
    모든 피처는 look-ahead bias 없이 현재까지의 데이터만 사용.
    """
    f = df.copy()

    f = _price_features(f)
    f = _technical_indicators(f, cfg)
    f = _volatility_features(f, cfg)
    f = _volume_features(f, cfg)
    f = _microstructure_features(f)

    if cfg.use_hurst:
        f = _hurst_features(f)

    if getattr(cfg, "use_momentum", False):
        f = _momentum_features(f, cfg)

    if getattr(cfg, "use_regime", False):
        f = _regime_features(f)

    # Fractional Differentiation (López de Prado 2018 Ch.5)
    # log_ret 의 메모리 손실 보완 — 정상성+메모리 양립
    if getattr(cfg, "use_frac_diff", False):
        f = _frac_diff_features(f, cfg)

    # Macro features (Welch & Goyal 2008) — 거시 변수 정규화
    # (병합된 _macro_* 컬럼이 있을 때만 작동)
    if getattr(cfg, "use_macro", False):
        f = _macro_features(f)

    # 원본 OHLCV 제거 (스케일 문제 방지)
    f.drop(columns=["Open", "High", "Low", "Close", "Volume"], inplace=True)

    # 무한값 처리
    f.replace([np.inf, -np.inf], np.nan, inplace=True)

    return f


# ── 가격 수익률 피처 ───────────────────────────────────

def _price_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]

    df["log_ret"]    = np.log(c / c.shift(1))
    df["log_ret_2"]  = np.log(c / c.shift(2))
    df["log_ret_5"]  = np.log(c / c.shift(5))
    df["log_ret_10"] = np.log(c / c.shift(10))
    df["log_ret_20"] = np.log(c / c.shift(20))

    # 고가-저가 범위
    df["hl_ratio"] = np.log(df["High"] / df["Low"])

    # 갭
    df["gap"] = np.log(df["Open"] / c.shift(1))

    # 종가 위치 (Low~High 내)
    hl = df["High"] - df["Low"] + 1e-9
    df["close_pos"] = (c - df["Low"]) / hl

    # 캔들 방향
    df["candle_dir"] = np.sign(c - df["Open"])

    return df


# ── 기술적 지표 ────────────────────────────────────────

def _technical_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    c = df["Close"]

    # RSI
    df["rsi"] = _rsi(c, cfg.rsi_period)
    df["rsi_norm"] = df["rsi"] / 100.0 - 0.5   # [-0.5, 0.5] 정규화

    # MACD
    ema_f = c.ewm(span=cfg.macd_fast,   adjust=False).mean()
    ema_s = c.ewm(span=cfg.macd_slow,   adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=cfg.macd_signal, adjust=False).mean()
    df["macd_hist_norm"] = (macd - sig) / (c + 1e-9)

    # Bollinger %B
    sma = c.rolling(cfg.bb_period).mean()
    std = c.rolling(cfg.bb_period).std()
    df["bb_pct_b"] = (c - (sma - 2*std)) / (4*std + 1e-9)
    df["bb_width"]  = 4 * std / (sma + 1e-9)

    # 이동평균 대비 위치 (정규화)
    for w in cfg.ma_windows:
        ma = c.rolling(w).mean()
        df[f"price_ma_{w}"] = (c - ma) / (ma + 1e-9)

    # Stochastic
    lo = df["Low"].rolling(14).min()
    hi = df["High"].rolling(14).max()
    df["stoch_k"] = (c - lo) / (hi - lo + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ADX (추세 강도)
    df["adx"] = _adx(df, 14)

    # CCI (Commodity Channel Index)
    tp  = (df["High"] + df["Low"] + c) / 3
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * mad + 1e-9)
    df["cci_norm"] = df["cci"].clip(-3, 3) / 3.0

    return df


def _rsi(series, period):
    delta   = series.diff()
    gain    = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    loss    = (-delta).clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


def _adx(df, period):
    """Average Directional Index (추세 강도 지표)"""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    up   = high - high.shift(1)
    down = low.shift(1) - low
    dm_p = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    dm_m = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    atr  = tr.ewm(span=period, adjust=False).mean()
    di_p = 100 * dm_p.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
    di_m = 100 * dm_m.ewm(span=period, adjust=False).mean() / (atr + 1e-9)

    dx   = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
    adx  = dx.ewm(span=period, adjust=False).mean()
    return adx / 100.0  # [0,1]


# ── 변동성 피처 ────────────────────────────────────────

def _volatility_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    c = df["Close"]
    ret = np.log(c / c.shift(1))

    # 롤링 실현 변동성 (연율화)
    for w in cfg.vol_windows:
        df[f"rvol_{w}"] = ret.rolling(w).std() * np.sqrt(252)

    # Garman-Klass 변동성 (OHLC 활용, 더 효율적)
    # σ²_GK = 0.5*(ln(H/L))² - (2ln2-1)*(ln(C/O))²
    df["gk_vol"] = np.sqrt(
        0.5 * np.log(df["High"] / df["Low"]).pow(2).rolling(20).mean()
        - (2*np.log(2)-1) * np.log(c / df["Open"]).pow(2).rolling(20).mean()
    ) * np.sqrt(252)

    # 변동성 레짐 (현재 변동성 / 장기 평균)
    long_vol = ret.rolling(60).std() * np.sqrt(252)
    df["vol_regime"] = df["rvol_20"] / (long_vol + 1e-9)

    # ATR 정규화
    prev_c = c.shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_c).abs(),
        (df["Low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=cfg.atr_period, adjust=False).mean()
    df["atr_norm"] = atr / (c + 1e-9)

    return df


# ── 거래량 피처 ────────────────────────────────────────

def _volume_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    vol = df["Volume"]
    c   = df["Close"]

    for w in cfg.vol_windows:
        df[f"vol_ratio_{w}"] = vol / (vol.rolling(w).mean() + 1e-9)

    # OBV 모멘텀
    direction = np.sign(c.diff())
    obv       = (vol * direction).cumsum()
    df["obv_mom"] = obv / (obv.rolling(20).std() + 1e-9)

    # Amihud Illiquidity (2002)
    # ILLIQ = |r_t| / (Volume_t × Price_t)  → 유동성 부족 = 값 클수록 비유동적
    dollar_vol = vol * c
    df["amihud"] = np.log(
        np.abs(df["log_ret"]) / (dollar_vol + 1e-9) * 1e9 + 1e-9
    )

    # VWAP 대비 위치
    vwap = (c * vol).rolling(20).sum() / (vol.rolling(20).sum() + 1e-9)
    df["vwap_dev"] = (c - vwap) / (vwap + 1e-9)

    return df


# ── 시장 미시구조 피처 ─────────────────────────────────

def _microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll (1984) Spread Estimator:
        spread = 2 * sqrt(max(-Cov(ΔP_t, ΔP_{t-1}), 0))
    가격 충격과 유동성의 대리 지표로 활용.
    """
    ret = df["log_ret"]

    # 자기 공분산 (시차 1)
    cov = ret.rolling(20).apply(
        lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 5 else 0,
        raw=True,
    )
    df["roll_spread"] = 2 * np.sqrt(np.maximum(-cov, 0))

    # 가격 반전 지표 (단기 평균회귀 신호)
    df["price_reversal_5"]  = -df["log_ret_5"]   # 단기 반전
    df["price_reversal_20"] = -df["log_ret_20"]  # 중기 반전

    # 시간 피처 (sin/cos 인코딩)
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.to_datetime(idx)
        except Exception:
            idx = pd.date_range("2000-01-01", periods=len(df), freq="B")

    df["dow_sin"]   = np.sin(2 * np.pi * idx.dayofweek / 5)
    df["dow_cos"]   = np.cos(2 * np.pi * idx.dayofweek / 5)
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)

    return df


# ── Hurst 지수 ─────────────────────────────────────────

def _hurst_features(df: pd.DataFrame, min_window: int = 40) -> pd.DataFrame:
    """
    R/S Analysis (Hurst, 1951):
        H = log(R/S) / log(n)

    롤링 윈도우로 시간에 따른 레짐 변화 감지.
    H > 0.5: 추세 지속 → 모멘텀 전략 유리
    H < 0.5: 평균회귀 → 역추세 전략 유리
    """
    ret = df["log_ret"].fillna(0).values
    n   = len(ret)
    hurst_vals = np.full(n, np.nan)

    window = 60
    for i in range(window, n):
        hurst_vals[i] = _compute_hurst(ret[max(0, i-window):i])

    df["hurst"] = hurst_vals
    df["hurst_centered"] = df["hurst"] - 0.5   # 0 중심 (음수=평균회귀, 양수=추세)

    return df


def _compute_hurst(ts: np.ndarray) -> float:
    """R/S 분석으로 Hurst 지수 계산."""
    if len(ts) < 20:
        return 0.5
    try:
        ts    = ts - ts.mean()
        cs    = np.cumsum(ts)
        R     = cs.max() - cs.min()
        S     = ts.std()
        if S < 1e-10:
            return 0.5
        return np.log(R / S) / np.log(len(ts))
    except Exception:
        return 0.5


# ── Fractional Differentiation (López de Prado 2018 Ch.5) ──────────

def _frac_diff_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Fixed-Window Fractional Difference (FFD) 가중치 생성.

    이론 (López de Prado 2018 §5.5):
        (1 - L)^d = Σ_{k=0..∞} C(d, k) · (-L)^k
        where C(d, k) = d·(d-1)·...·(d-k+1) / k!

        Iterative: w_0 = 1, w_k = -w_{k-1} · (d - k + 1) / k

    Fixed-window (FFD) 의 장점 (expanding window 대비):
        ① 모든 시점에서 동일 가중치 → 정상성 일관
        ② 메모리 사용 일정 (k 가 threshold 이하로 작아지면 자름)
        ③ 평균 수렴 안정 (look-ahead bias 0)

    Parameters
    ----------
    d : float ∈ (0, 1)
        분수차분 차수. 0.5 ≈ 메모리 절반 유지, 1.0 = 일반 1차 차분 (메모리 0).
        SPY 같은 주가는 d ≈ 0.3~0.5 가 정상성+메모리 양립 sweet spot.
    threshold : float
        |w_k| < threshold 이면 자름. 작을수록 윈도우 길어짐.
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
        if k > 10_000:    # 안전장치: 너무 작은 d 가 들어와도 무한 루프 방지
            logger_local_warn = True
            break
    # 시간순 컨볼루션 위해 역순 반환 (가장 오래된 가중치 → 가장 최근)
    return np.array(w[::-1], dtype=np.float64)


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5,
) -> pd.Series:
    """
    Fixed-Window 분수차분 적용 (López de Prado 2018 Ch.5).

    원본 시계열의 "메모리" 를 보존하면서 정상성 (stationarity) 확보.
    log_ret (1차 차분) 은 메모리 완전 소실 → 학습 시그널 약함.
    분수차분 d=0.4 는 정상성과 메모리 모두 70% 이상 유지.

    Returns
    -------
    pd.Series : 동일 길이, 앞쪽 window-1 개 행은 NaN (warmup)

    참고:
      ADF 검정으로 정상성 확인 후 최소 d 선택이 정석.
      여기선 d=0.3, 0.5 두 값을 함께 제공 — 모델이 두 메모리 스케일을 모두 봄.
    """
    weights = _frac_diff_weights_ffd(d, threshold)
    width = len(weights) - 1
    n = len(series)

    if n <= width:
        return pd.Series(np.full(n, np.nan), index=series.index)

    vals = series.values.astype(np.float64)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(width, n):
        window = vals[i - width: i + 1]
        if np.any(np.isnan(window)):
            continue
        out[i] = np.dot(weights, window)

    return pd.Series(out, index=series.index, name=series.name)


# ── Macro Features (Welch & Goyal 2008) ────────────────────────────

def download_macro_features(
    start: str,
    end:   str,
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    거시 변수 다운로드 — 시장 레짐 정보 (Welch & Goyal 2008).

    티커:
      ^VIX  : 변동성 지수 (시장 공포 — 강한 평균회귀 신호)
      ^TNX  : 10년 미국 국채 금리 (장기 금리 사이클)
      ^IRX  : 3개월 미국 국채 금리 (단기 금리)
      yield_slope = TNX - IRX : 수익률 곡선 기울기 (경기 침체 선행 지표)

    참고:
      - Welch, I. & Goyal, A. (2008) *A Comprehensive Look at the Empirical
        Performance of Equity Premium Prediction*. Review of Financial Studies.
      - Estrella, A. & Mishkin, F. (1998) *Predicting U.S. Recessions:
        Financial Variables as Leading Indicators*. RES.

    Returns
    -------
    DataFrame indexed by date with columns:
      [vix, vix_log, vix_change, tnx, irx, yield_slope]
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    tickers = ["^VIX", "^TNX", "^IRX"]
    macro_data = {}

    for tk in tickers:
        try:
            df = yf.download(tk, start=start, end=end, interval=interval,
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                continue
            macro_data[tk] = df["Close"]
        except Exception:
            continue

    if not macro_data:
        return None

    out = pd.DataFrame(index=macro_data[list(macro_data.keys())[0]].index)

    if "^VIX" in macro_data:
        vix = macro_data["^VIX"]
        out["vix"]        = vix
        out["vix_log"]    = np.log(vix.clip(lower=1.0))
        out["vix_change"] = vix.pct_change(5)    # 5일 변화율

    if "^TNX" in macro_data:
        out["tnx"] = macro_data["^TNX"]

    if "^IRX" in macro_data:
        out["irx"] = macro_data["^IRX"]

    if "tnx" in out.columns and "irx" in out.columns:
        out["yield_slope"] = out["tnx"] - out["irx"]
        # 수익률 곡선 inversion 지표 (< 0 = 침체 신호)
        out["yield_inverted"] = (out["yield_slope"] < 0).astype(float)

    return out


def merge_macro_features(
    asset_df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Asset OHLCV df 에 macro 피처 병합. 시간 정렬, 결측 forward-fill.
    """
    if macro_df is None or macro_df.empty:
        return asset_df

    # asset index 기준 정렬 + ffill (macro 가 매일 안 나올 수 있음)
    aligned = macro_df.reindex(asset_df.index).ffill().bfill()
    # macro 피처를 OHLCV 옆에 붙임 — build_all_features 의 _macro_features() 가 처리
    asset_df = asset_df.copy()
    for col in aligned.columns:
        asset_df[f"_macro_{col}"] = aligned[col]
    return asset_df


def _macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    병합된 macro 컬럼 (_macro_*) 을 피처로 변환.

    원본 absolute 값은 비정상 → 변환 후 사용:
      - vix_log     : 그대로 (이미 log)
      - vix_change  : 그대로 (이미 변화율)
      - yield_slope : 그대로 (이미 차분)
      - 나머지 raw : z-score (rolling 60)
    """
    macro_cols = [c for c in df.columns if c.startswith("_macro_")]
    if not macro_cols:
        return df

    for col in macro_cols:
        key = col.replace("_macro_", "")
        # 이미 정상화된 시계열은 그대로 — z-score 만
        if key in ("vix_log", "vix_change", "yield_slope", "yield_inverted"):
            mu = df[col].rolling(60, min_periods=10).mean()
            sd = df[col].rolling(60, min_periods=10).std() + 1e-8
            df[f"macro_{key}"] = (df[col] - mu) / sd
        else:
            # raw level (vix, tnx, irx) → 변화율로 변환
            ret = df[col].pct_change(5).fillna(0)
            mu = ret.rolling(60, min_periods=10).mean()
            sd = ret.rolling(60, min_periods=10).std() + 1e-8
            df[f"macro_{key}_chg"] = (ret - mu) / sd

    # 원본 _macro_ 컬럼 제거 (변환된 것만 남김)
    df.drop(columns=macro_cols, inplace=True)
    return df


def _frac_diff_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    분수차분 피처 추가 (LDP 2018 Ch.5).

    log(Close) 에 대해 두 가지 d 값으로 차분 — 다중 메모리 스케일 캡처:
      d=0.3 : 메모리 풍부, 약한 정상성 (장기 패턴)
      d=0.5 : 메모리 절반, 강한 정상성 (중기 패턴)

    표준 스케일링 (rolling z-score) 으로 학습 안정성 확보.
    """
    log_price = np.log(df["Close"].replace(0, np.nan)).ffill()

    threshold = getattr(cfg, "frac_diff_threshold", 1e-5)
    d_values = getattr(cfg, "frac_diff_d_values", [0.3, 0.5])

    for d in d_values:
        fd_raw = frac_diff_ffd(log_price, d=float(d), threshold=threshold)
        # rolling z-score (60봉 윈도우) — feat 스케일 안정화
        roll_mean = fd_raw.rolling(60, min_periods=10).mean()
        roll_std  = fd_raw.rolling(60, min_periods=10).std() + 1e-8
        df[f"fracdiff_close_d{int(d*10):02d}"] = (fd_raw - roll_mean) / roll_std

    return df


# ── 포트폴리오 상태 피처 ───────────────────────────────

# ── 모멘텀 피처 (Jegadeesh & Titman 1993) ────────────────

def _momentum_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    다중 주기 가격 모멘텀 피처.

    이론:
      - Cross-sectional Momentum (Jegadeesh & Titman 1993):
          12개월 과거 수익률이 높은 종목이 다음 달도 좋음 (모멘텀 효과)
      - 52주 고가 근접도 (George & Hwang 2004):
          52주 고가 근처에 있을수록 상승 가능성 ↑ (앵커링 효과)
      - 다중 스케일 캡처 (1M/3M/6M/12M) → 단기~장기 모멘텀 동시 반영

    구현:
      - 각 기간 ROC 를 rolling z-score 정규화 → 분포 일관성 확보
      - 52주 고가/저가 대비 현재 위치 (거리)
      - RSI 기울기 (5일 RSI 변화) → 모멘텀 가속도
      - 거래량 모멘텀 (OBV 기울기) → 가격모멘텀 확인
    """
    c = df["Close"]
    mom_windows = getattr(cfg, "mom_windows", [21, 63, 126, 252])

    for w in mom_windows:
        roc = c.pct_change(w)
        mu = roc.rolling(252, min_periods=60).mean()
        sd = roc.rolling(252, min_periods=60).std() + 1e-8
        df[f"mom_{w}"] = ((roc - mu) / sd).clip(-5, 5)

    # 52주 고가/저가 근접도 (George & Hwang 2004)
    roll252 = max(252, c.count() // 4)
    df["dist_52w_high"] = (c / c.rolling(252, min_periods=60).max() - 1.0).clip(-1, 0)
    df["dist_52w_low"]  = (c / c.rolling(252, min_periods=60).min() - 1.0).clip(0, 5)

    # 모멘텀 가속도: 단기 수익률 - 중기 수익률 (추세 전환 신호)
    ret = np.log(c / c.shift(1))
    df["mom_accel"] = ret.rolling(10).mean() - ret.rolling(60).mean()

    # RSI 기울기 (5일 변화)
    if "rsi" in df.columns:
        df["rsi_slope"] = (df["rsi"].diff(5) / 5.0 / 100.0).clip(-0.1, 0.1)

    # OBV 모멘텀 기울기 (거래량 추세 방향)
    if "obv_mom" in df.columns:
        df["obv_slope"] = df["obv_mom"].diff(10).clip(-5, 5)

    return df


# ── 레짐 피처 ────────────────────────────────────────────

def _regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    시장 레짐 피처 (추세 + 변동성 레짐).

    이론:
      - 추세 레짐: MA50 vs MA200 크로스오버 (Golden/Death Cross)
          장기 상승 추세에서 모멘텀 전략 유효성 ↑
      - 변동성 레짐: 단기 vs 장기 변동성 비율
          고변동성 구간 = 레짐 전환 가능성 ↑ → 포지션 축소 필요
      - 추세 강도: MA 기울기 정규화
          완만한 기울기 = 추세 지속, 급격한 기울기 = 과열/과매도
    """
    c = df["Close"]
    ret = np.log(c / c.shift(1))

    ma50  = c.rolling(50,  min_periods=20).mean()
    ma200 = c.rolling(200, min_periods=60).mean()

    # 추세 방향 플래그 (0/1)
    df["above_ma50"]   = (c > ma50).astype(np.float32)
    df["above_ma200"]  = (c > ma200).astype(np.float32)
    df["golden_cross"] = (ma50 > ma200).astype(np.float32)

    # MA 기울기 (정규화) — 추세 강도
    ma50_slope  = ma50.pct_change(5).fillna(0)
    ma200_slope = ma200.pct_change(20).fillna(0)
    for name, s in [("ma50_slope", ma50_slope), ("ma200_slope", ma200_slope)]:
        mu = s.rolling(120, min_periods=20).mean()
        sd = s.rolling(120, min_periods=20).std() + 1e-8
        df[name] = ((s - mu) / sd).clip(-4, 4)

    # 변동성 레짐 (단기 vs 장기 비율) — 이미 vol_regime 있지만 더 긴 기준 추가
    short_vol = ret.rolling(10).std()  * np.sqrt(252)
    long_vol  = ret.rolling(120).std() * np.sqrt(252)
    df["vol_regime_120"] = (short_vol / (long_vol + 1e-9)).clip(0, 5)

    # 가격 위치 채널 (52주 내 상대 위치 [0,1])
    h252 = c.rolling(252, min_periods=60).max()
    l252 = c.rolling(252, min_periods=60).min()
    df["price_channel_pos"] = ((c - l252) / (h252 - l252 + 1e-9)).clip(0, 1)

    return df


# ── 포트폴리오 상태 피처 ───────────────────────────────────

def compute_portfolio_features(
    position: float,
    unrealized_pnl_pct: float,
    cash_ratio: float,
    holding_steps: int,
    max_holding: int = 252,
) -> np.ndarray:
    """
    현재 포트폴리오 상태를 정규화된 피처로 변환.
    환경(env)의 step()에서 호출됩니다.
    """
    return np.array([
        position,                              # [-1, 1] 이미 정규화
        np.tanh(unrealized_pnl_pct * 10),     # 수익률 tanh 압축
        cash_ratio,                            # [0, 1]
        holding_steps / max_holding,           # 보유 기간 비율
    ], dtype=np.float32)
