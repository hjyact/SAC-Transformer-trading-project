"""
utils/multi_data.py — 다종목 데이터 로더 (Multi-Ticker Data Pipeline)

목적:
  단일 종목 학습의 데이터 부족(과적합) 문제를 자산군 다각화로 보완.
  종목별 독립 피처 정규화 → cross-ticker look-ahead bias 회피
  (López de Prado 2018, Ch.3 — 학습/평가 데이터 분리 원칙).

설계 원칙:
  1. 종목별 독립 피처 생성. 정규화 통계는 각 TradingEnv 가 자체 슬라이스에서 계산.
  2. 공통 시간 인덱스로 모든 종목 정렬 → 거래일/거래시간 불일치 종목 자동 배제.
  3. train/test 분리는 모든 종목에 동일한 시간 기준 적용 (시간 OOS 일관성).
  4. parquet 캐싱으로 재다운로드 회피. 캐시 키 = ticker+interval+start+end.
  5. 데이터 품질 검증: 최소 봉 수, NaN 비율, 영 거래량 봉 비율.

참고:
  - Liu et al. (2020) FinRL — 다종목 환경 데이터 파이프라인 표준 참고
  - Yang et al. (2020) Ensemble DRL trading — 30개 Dow 종목 동시 학습
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from config import ROOT, feat_cfg
from utils.features import build_all_features

logger = logging.getLogger(__name__)

CACHE_DIR = ROOT / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ── 데이터 컨테이너 ─────────────────────────────────────

@dataclass
class TickerSplit:
    """종목 1개의 train/test 분리 결과."""
    ticker:     str
    train_feat: pd.DataFrame
    train_price: pd.DataFrame
    test_feat:  pd.DataFrame
    test_price: pd.DataFrame

    @property
    def n_features(self) -> int:
        return self.train_feat.shape[1]

    @property
    def n_train(self) -> int:
        return len(self.train_feat)

    @property
    def n_test(self) -> int:
        return len(self.test_feat)


@dataclass
class MultiTickerData:
    """전체 다종목 데이터셋."""
    splits:     Dict[str, TickerSplit] = field(default_factory=dict)
    tickers:    List[str]              = field(default_factory=list)
    n_features: int                    = 0
    start:      Optional[pd.Timestamp] = None
    end:        Optional[pd.Timestamp] = None
    interval:   str                    = "1d"

    def summary(self) -> str:
        lines = [
            f"MultiTickerData ({len(self.tickers)} tickers, "
            f"{self.n_features} features, interval={self.interval})",
            f"  Period: {self.start.date() if self.start else '?'} ~ "
            f"{self.end.date() if self.end else '?'}",
        ]
        for tk in self.tickers:
            s = self.splits[tk]
            lines.append(
                f"  [{tk:6s}] train={s.n_train:5d}  test={s.n_test:5d}"
            )
        return "\n".join(lines)


# ── 단일 종목 다운로드 + 캐시 ────────────────────────────

def _cache_path(ticker: str, interval: str, start: str, end: str) -> Path:
    return CACHE_DIR / f"{ticker}_{interval}_{start}_{end}.parquet"


def _download_single(
    ticker: str,
    start:  str,
    end:    str,
    interval: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """단일 종목 OHLCV 다운로드 (parquet 캐시 우선, 실패 시 그냥 진행)."""
    cache = _cache_path(ticker, interval, start, end)
    if use_cache and cache.exists():
        try:
            logger.info(f"  [{ticker}] 캐시 로드: {cache.name}")
            return pd.read_parquet(cache)
        except Exception as e:
            logger.warning(f"  [{ticker}] 캐시 읽기 실패 ({e}) — 재다운로드")

    import yfinance as yf
    logger.info(f"  [{ticker}] yfinance 다운로드: {start}~{end} {interval}")
    raw = yf.download(
        ticker, start=start, end=end, interval=interval, progress=False,
        auto_adjust=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if raw.empty:
        raise ValueError(f"yfinance returned empty data for {ticker}")
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # 캐시 저장 (실패해도 다운로드는 성공이므로 무시)
    if use_cache:
        try:
            df.to_parquet(cache)
        except Exception as e:
            logger.warning(
                f"  [{ticker}] 캐시 저장 실패 ({e}) — pyarrow 미설치? "
                f"`pip install pyarrow` 권장. 다운로드 자체는 정상"
            )
    return df


# ── 품질 검증 ────────────────────────────────────────────

def _validate_quality(
    ticker: str,
    price_df: pd.DataFrame,
    feat_df:  pd.DataFrame,
    min_bars: int,
    max_nan_ratio: float,
    max_zero_vol_ratio: float,
) -> Optional[str]:
    """품질 미달이면 사유 문자열, OK 면 None."""
    if len(price_df) < min_bars:
        return f"{ticker}: 봉 수 {len(price_df)} < {min_bars}"
    nan_ratio = feat_df.isna().any(axis=1).mean()
    if nan_ratio > max_nan_ratio:
        return f"{ticker}: NaN 비율 {nan_ratio:.1%} > {max_nan_ratio:.1%}"
    zero_vol_ratio = (price_df["Volume"] <= 0).mean()
    if zero_vol_ratio > max_zero_vol_ratio:
        return f"{ticker}: 영거래량 비율 {zero_vol_ratio:.1%} > {max_zero_vol_ratio:.1%}"
    return None


# ── 메인 로더 ────────────────────────────────────────────

def load_multi_ticker(
    tickers:   List[str],
    start:     str,
    end:       str,
    interval:  str          = "1d",
    test_ratio: float       = 0.2,
    min_bars:  int          = 300,
    max_nan_ratio: float    = 0.05,
    max_zero_vol_ratio: float = 0.10,
    use_cache: bool         = True,
    align_common_index: bool = True,
) -> MultiTickerData:
    """
    여러 종목 다운로드 → 종목별 피처 생성 → 공통 시간 인덱스 정렬 → train/test 분리.

    Parameters
    ----------
    tickers           : 종목 리스트
    start, end        : ISO 날짜 문자열
    interval          : "1d" / "1h" / "30m" / ...
    test_ratio        : 마지막 N% 를 평가용으로 분리 (모든 종목 동일 시점)
    min_bars          : 종목별 최소 봉 수 (미달 시 제외)
    max_nan_ratio     : 피처 NaN 비율 상한
    max_zero_vol_ratio: 영 거래량 비율 상한
    use_cache         : parquet 캐시 사용
    align_common_index: True 면 모든 종목의 공통 시간 인덱스만 사용
                        (False 면 종목별 자체 인덱스 유지)

    Returns
    -------
    MultiTickerData : 종목별 TickerSplit 컨테이너

    Notes
    -----
    피처 정규화는 여기서 하지 않음. 각 TradingEnv 가 자체 슬라이스에서
    self._feat_mean/std 를 계산하므로 cross-ticker leakage 없음.
    """
    if not tickers:
        raise ValueError("tickers 리스트가 비어 있습니다")

    logger.info(f"다종목 데이터 로드: {tickers} ({interval}, {start}~{end})")

    # 1) 종목별 다운로드
    prices: Dict[str, pd.DataFrame] = {}
    skipped: List[str] = []
    for tk in tickers:
        try:
            df = _download_single(tk, start, end, interval, use_cache)
            prices[tk] = df
        except Exception as e:
            logger.warning(f"  [{tk}] 다운로드 실패 ({e}) — 제외")
            skipped.append(tk)
            continue

    if not prices:
        raise RuntimeError("모든 종목 다운로드 실패")

    # 2) 공통 시간 인덱스 (선택)
    if align_common_index and len(prices) >= 2:
        common = None
        for tk, df in prices.items():
            common = df.index if common is None else common.intersection(df.index)
        common = common.sort_values()
        if len(common) < min_bars:
            raise RuntimeError(
                f"공통 인덱스 부족: {len(common)} < {min_bars}. "
                f"종목별 거래일/거래시간이 다를 수 있음. "
                f"align_common_index=False 로 우회 가능"
            )
        for tk in list(prices.keys()):
            prices[tk] = prices[tk].loc[common]
        logger.info(f"  공통 인덱스: {len(common)}행 ({common[0].date()}~{common[-1].date()})")

    # 3) 종목별 피처 생성 + 품질 검증
    feats: Dict[str, pd.DataFrame] = {}
    for tk, p_df in list(prices.items()):
        f_df = build_all_features(p_df, feat_cfg)
        # 공통 행만 (피처 dropna 와 가격 정렬)
        valid_idx = f_df.dropna().index.intersection(p_df.index)
        f_df = f_df.loc[valid_idx]
        p_df_aligned = p_df.loc[valid_idx]

        err = _validate_quality(
            tk, p_df_aligned, f_df, min_bars, max_nan_ratio, max_zero_vol_ratio,
        )
        if err is not None:
            logger.warning(f"  품질 미달 → 제외: {err}")
            skipped.append(tk)
            prices.pop(tk, None)
            continue

        feats[tk] = f_df
        prices[tk] = p_df_aligned

    if not feats:
        raise RuntimeError("품질 통과 종목 없음")

    # 4) 피처 컬럼 일관성 검증 (build_all_features 가 동일 cfg 면 동일 컬럼 보장)
    ref_cols = list(feats[next(iter(feats))].columns)
    for tk, f in feats.items():
        if list(f.columns) != ref_cols:
            raise RuntimeError(
                f"[{tk}] 피처 컬럼 불일치. ref={ref_cols[:3]}... vs {list(f.columns)[:3]}..."
            )
    n_feat = len(ref_cols)

    # 5) train/test 분리 (시간 기준 동일하게)
    splits: Dict[str, TickerSplit] = {}
    for tk in feats:
        n = len(feats[tk])
        cut = int(n * (1 - test_ratio))
        splits[tk] = TickerSplit(
            ticker      = tk,
            train_feat  = feats[tk].iloc[:cut],
            train_price = prices[tk].iloc[:cut],
            test_feat   = feats[tk].iloc[cut:],
            test_price  = prices[tk].iloc[cut:],
        )

    # 6) 결과 패킹
    first_tk = next(iter(splits))
    data = MultiTickerData(
        splits     = splits,
        tickers    = list(splits.keys()),
        n_features = n_feat,
        start      = splits[first_tk].train_price.index[0],
        end        = splits[first_tk].test_price.index[-1] if splits[first_tk].n_test > 0
                     else splits[first_tk].train_price.index[-1],
        interval   = interval,
    )

    logger.info(data.summary())
    if skipped:
        logger.info(f"  제외된 종목: {skipped}")

    return data


# ── 종목별 train/test 슬라이스 추출 헬퍼 ────────────────

def to_env_data(
    data: MultiTickerData,
) -> Tuple[Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
           Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]]:
    """
    MultiTickerData → MultiTickerEnv 가 받는 dict 형태로 변환.

    Returns
    -------
    train_dict : {ticker: (train_feat_df, train_price_df)}
    eval_dict  : {ticker: (test_feat_df,  test_price_df)}
    """
    train_dict = {
        tk: (s.train_feat, s.train_price) for tk, s in data.splits.items()
    }
    eval_dict = {
        tk: (s.test_feat, s.test_price) for tk, s in data.splits.items()
        if s.n_test > 0
    }
    return train_dict, eval_dict
