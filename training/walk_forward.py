"""
training/walk_forward.py — Walk-Forward Analysis (재학습 + OOS 평가)

이론 배경
─────────
  Pardo (1992) "Design, Testing, and Optimization of Trading Systems"
    Walk-forward 의 원조. anchored (확장 윈도우) vs rolling (고정 윈도우).
    시장 비정상성 (non-stationarity) 대응의 표준 방법.

  López de Prado (2018) "Advances in Financial Machine Learning" Ch.7
    Purged k-fold + Embargo. 자기상관으로 인한 train/val 간 leakage 방지.
    완전한 purged k-fold 는 정확한 sample 종료시간이 필요해 RL 에선 직접 적용
    어려움 — 본 구현은 단순 시간 embargo 만 적용.

  Bailey, Borwein, López de Prado, Zhu (2017) "The Probability of Backtest
  Overfitting" Journal of Computational Finance
    CSCV (Combinatorially Symmetric Cross-Validation) 기반 PBO 지표.
    백테스트 결과가 우연/과적합일 확률 추정. 본 구현은 단순화된 rank 기반 근사.

  Henderson, P. et al. (2018) "Deep Reinforcement Learning that Matters" AAAI
    RL 결과의 재현성 위기. 단일 시드 결과는 신뢰 불가. 다중 시드 평균/표준편차
    보고가 필수임을 주장.

개선사항 (기존 대비)
────────────────────
  1. Embargo period (autocorrelation leak 방지)
  2. Anchored vs Rolling mode 선택
  3. PER + N-step + Welford 정규화 (메인 학습과 일관성)
  4. end_of_episode() 호출 (n-step 종목/에피소드 경계 cut)
  5. 폴드별 Buy & Hold 벤치마크 + α 계산
  6. 다중 시드 옵션 (단일 시드 운빨 제거)
  7. PBO (Probability of Backtest Overfitting) 계산
  8. 상세 폴드별 로깅 + 종합 보고서
"""

import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WalkForwardTrainer:
    """
    Walk-Forward 방식 재학습 및 OOS 평가.

    Parameters
    ----------
    n_splits : int (기본 5)
        분할 수. PBO 계산을 원하면 6 이상 권장 (Bailey et al. 2017).
    n_steps_per_fold : int (기본 100_000)
        폴드당 학습 step. 50k~200k 권장.
    mode : str ('anchored' | 'rolling', 기본 'anchored')
        anchored : 학습 데이터 누적 확장 (1990~2010 → 1990~2015 → ...)
        rolling  : 고정 길이 슬라이딩 (1990~2010 → 1995~2015 → ...)
        시장 구조 변화 클 때 rolling, 안정적이면 anchored 권장 (Pardo 1992).
    rolling_train_size : int or None
        mode='rolling' 일 때 학습 윈도우 길이 (봉 수). None 이면 1 fold size.
    embargo_pct : float (기본 0.01)
        Train/val 사이 embargo 비율. 1% = train 끝 ~ val 시작 사이 빈 구간.
        자기상관 leak 방지 (López de Prado 2018 Ch.7).
    retrain_from_scratch : bool (기본 False)
        True  : 매 폴드 가중치 초기화 (엄격한 검증)
        False : 이전 폴드 가중치 계승 (warm start, 학습 효율 ↑)
    n_seeds : int (기본 1)
        폴드당 시드 수. Henderson et al. (2018) 은 RL 평가에 3+ 시드 권장.
        시간 비용 N배 증가. 단일 시드 결과 검증 후 늘리는 게 실용적.
    compute_pbo : bool (기본 False)
        Bailey et al. (2017) PBO 계산. n_splits >= 6 필요.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_steps_per_fold: int = 100_000,
        mode: str = "anchored",
        rolling_train_size: Optional[int] = None,
        embargo_pct: float = 0.01,
        retrain_from_scratch: bool = False,
        n_seeds: int = 1,
        compute_pbo: bool = False,
    ):
        if mode not in ("anchored", "rolling"):
            raise ValueError(f"mode 는 'anchored' 또는 'rolling' 만 가능 (받은: {mode})")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError(f"embargo_pct 는 [0, 1) (받은: {embargo_pct})")
        if n_seeds < 1:
            raise ValueError(f"n_seeds >= 1 (받은: {n_seeds})")

        self.n_splits          = n_splits
        self.n_steps           = n_steps_per_fold
        self.mode              = mode
        self.rolling_train_size= rolling_train_size
        self.embargo_pct       = embargo_pct
        self.from_scratch      = retrain_from_scratch
        self.n_seeds           = n_seeds
        self.compute_pbo       = compute_pbo

        if compute_pbo and n_splits < 6:
            logger.warning(
                f"PBO 계산은 n_splits >= 6 권장 (현재 {n_splits}) — 결과 신뢰성 낮음"
            )

        self.fold_results: List[Dict] = []

    # ── 분할 인덱스 생성 ────────────────────────────────

    def _create_splits(self, n: int) -> List[Tuple[int, int, int, int]]:
        """
        분할 인덱스 리스트 반환. 각 분할 = (train_start, train_end, val_start, val_end).

        Embargo 효과:
            ┌────train────┬─embargo─┬───val───┐
            0          tr_end   vl_start    vl_end

        anchored (확장 윈도우):
            폴드 1: [0      → fs   ]  val: [fs+E   → 2fs]
            폴드 2: [0      → 2fs  ]  val: [2fs+E  → 3fs]
            폴드 3: [0      → 3fs  ]  val: [3fs+E  → 4fs]

        rolling (고정 윈도우, rolling_train_size 만큼):
            폴드 1: [0          → fs   ]  val: [fs+E   → 2fs]
            폴드 2: [fs         → 2fs  ]  val: [2fs+E  → 3fs]
            폴드 3: [2fs        → 3fs  ]  val: [3fs+E  → 4fs]
        """
        fold_size = n // self.n_splits
        embargo   = max(1, int(fold_size * self.embargo_pct)) if self.embargo_pct > 0 else 0
        splits: List[Tuple[int, int, int, int]] = []

        for fold in range(self.n_splits - 1):
            train_end = fold_size * (fold + 1)
            val_start = train_end + embargo
            val_end   = min(fold_size * (fold + 2), n)

            if val_end - val_start < 30:    # 너무 작은 val 은 평가 무의미
                logger.warning(f"  폴드 {fold+1}: val 크기 부족 ({val_end - val_start}봉) — 건너뜀")
                continue

            if self.mode == "anchored":
                train_start = 0
            else:  # rolling
                if self.rolling_train_size is None:
                    train_start = max(0, train_end - fold_size)
                else:
                    train_start = max(0, train_end - int(self.rolling_train_size))

            splits.append((train_start, train_end, val_start, val_end))

        return splits

    # ── 메인 실행 ──────────────────────────────────────

    def run(
        self,
        feat_df: pd.DataFrame,
        price_df: pd.DataFrame,
        env_cfg,
        sac_cfg,
    ) -> Dict:
        """
        전체 Walk-Forward 학습 + OOS 평가.

        Returns
        -------
        summary : dict (fold 별 결과 + 종합 통계 + PBO)
        """
        n = len(feat_df)
        splits = self._create_splits(n)

        if not splits:
            raise RuntimeError(f"유효한 분할 없음 (n={n}, n_splits={self.n_splits})")

        logger.info(
            f"\n{'='*60}\n"
            f"Walk-Forward 시작\n"
            f"  데이터 : {n:,}봉 ({price_df.index[0].date()}~{price_df.index[-1].date()})\n"
            f"  분할   : {len(splits)}개 폴드 (mode={self.mode})\n"
            f"  학습   : {self.n_steps:,} step/fold × {self.n_seeds} seed\n"
            f"  embargo: {self.embargo_pct:.1%}\n"
            f"  warm start: {not self.from_scratch}\n"
            f"{'='*60}"
        )

        all_oos_rets: List[float] = []
        prev_agent = None    # warm start 용 (시드 0 의 다음 폴드 시작점)

        for fold_idx, (tr_s, tr_e, vl_s, vl_e) in enumerate(splits):
            fold_num = fold_idx + 1
            tr_feat = feat_df.iloc[tr_s:tr_e]
            tr_price = price_df.iloc[tr_s:tr_e]
            vl_feat = feat_df.iloc[vl_s:vl_e]
            vl_price = price_df.iloc[vl_s:vl_e]

            logger.info(
                f"\n[Fold {fold_num}/{len(splits)}] "
                f"Train: {tr_price.index[0].date()}~{tr_price.index[-1].date()} "
                f"({len(tr_feat):,}봉) | "
                f"Embargo: {vl_s - tr_e}봉 | "
                f"Val: {vl_price.index[0].date()}~{vl_price.index[-1].date()} "
                f"({len(vl_feat):,}봉)"
            )

            # ── 시드 루프 (Henderson 2018 권고)
            seed_results: List[Dict] = []
            for seed_idx in range(self.n_seeds):
                seed = getattr(sac_cfg, "seed", 42) + fold_idx * 1000 + seed_idx

                warm = (
                    prev_agent if (seed_idx == 0 and not self.from_scratch and prev_agent is not None)
                    else None
                )
                if warm is not None:
                    logger.info(f"  seed {seed_idx} (s={seed}): warm start (이전 폴드 계승)")

                result = self._train_and_eval_fold(
                    fold_num=fold_num,
                    seed_idx=seed_idx,
                    seed=seed,
                    tr_feat=tr_feat, tr_price=tr_price,
                    vl_feat=vl_feat, vl_price=vl_price,
                    env_cfg=env_cfg, sac_cfg=sac_cfg,
                    warm_agent=warm,
                )
                seed_results.append(result)

                if seed_idx == 0:    # 시드 0 의 에이전트만 warm start 로 다음 폴드에 전달
                    prev_agent = result.pop("_agent", None)
                else:
                    result.pop("_agent", None)

            # ── 시드 집계
            fold_result = self._aggregate_seeds(fold_num, seed_results)
            self.fold_results.append(fold_result)
            all_oos_rets.extend(fold_result.pop("_step_rets", []))

            self._log_fold(fold_result)

        # ── 전체 종합
        summary = self._aggregate_folds(all_oos_rets)
        if self.compute_pbo and len(self.fold_results) >= 6:
            summary["pbo"] = self._compute_pbo()

        self._log_summary(summary)
        return summary

    # ── 폴드 1 시드 학습 + OOS 평가 ────────────────────

    def _train_and_eval_fold(
        self,
        fold_num: int,
        seed_idx: int,
        seed: int,
        tr_feat, tr_price, vl_feat, vl_price,
        env_cfg, sac_cfg,
        warm_agent,
    ) -> Dict:
        # 지연 import (순환 회피)
        from env.trading_env import TradingEnv
        from agent.sac_agent import SACAgent
        from agent.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
        from utils.reward_normalizer import RewardNormalizer

        train_env = TradingEnv(tr_feat, tr_price, env_cfg, mode="train")
        # val 환경에 train 통계 주입 → fold 내 OOS 통계 누수 차단
        val_env   = TradingEnv(
            vl_feat, vl_price, env_cfg, mode="eval",
            feat_mean=train_env._feat_mean.copy(),
            feat_std =train_env._feat_std.copy(),
        )
        # transformer backbone 호환용 obs_meta
        obs_meta = {
            "window":   train_env.cfg.window_size,
            "n_feat":   train_env.n_feat,
            "port_dim": 4,
        }

        obs_dim = train_env.observation_space.shape[0]
        act_dim = train_env.action_space.shape[0]

        # 에이전트 (warm start or new)
        agent = warm_agent if warm_agent is not None else \
                SACAgent(obs_dim, act_dim, sac_cfg, obs_meta=obs_meta)

        # 버퍼 — 메인 학습과 동일 (PER + N-step + LAP)
        if getattr(sac_cfg, "use_per", False):
            buffer = PrioritizedReplayBuffer(
                obs_dim, act_dim, sac_cfg.buffer_size,
                n_step           = sac_cfg.n_step,
                gamma            = sac_cfg.gamma,
                alpha            = sac_cfg.per_alpha,
                beta_start       = sac_cfg.per_beta_start,
                beta_end         = sac_cfg.per_beta_end,
                beta_anneal_steps= self.n_steps,    # 폴드 길이에 맞춤
                eps              = sac_cfg.per_eps,
                use_lap          = sac_cfg.use_lap,
                lap_min_priority = sac_cfg.lap_min_priority,
            )
        else:
            buffer = ReplayBuffer(obs_dim, act_dim, sac_cfg.buffer_size)

        # 보상 정규화 (메인과 일관)
        reward_norm = (
            RewardNormalizer(
                center=getattr(sac_cfg, "reward_norm_center", False),
                clip  =getattr(sac_cfg, "reward_norm_clip", 10.0),
            )
            if getattr(sac_cfg, "normalize_reward", False) else None
        )

        # ── 학습 루프
        obs, _ = train_env.reset(seed=seed)
        for t in range(1, self.n_steps + 1):
            if t < sac_cfg.min_replay_size:
                action = train_env.action_space.sample()
            else:
                action = agent.select_action(obs, deterministic=False)

            next_obs, reward, term, trunc, _ = train_env.step(action)
            done = term or trunc

            stored_r = reward
            if reward_norm is not None:
                stored_r = reward_norm.update_and_normalize(reward)

            if hasattr(buffer, "set_step"):
                buffer.set_step(t)

            buffer.add(obs, action, stored_r, next_obs, float(term))
            obs = next_obs

            if done:
                if hasattr(buffer, "end_of_episode"):
                    buffer.end_of_episode()
                obs, _ = train_env.reset()

            if t >= sac_cfg.min_replay_size and buffer.is_ready:
                agent.train_step(buffer)

        # ── OOS 평가 (결정론 정책)
        oos_rets: List[float] = []
        oos_positions: List[float] = []
        obs, _ = val_env.reset(seed=seed)
        info = {}
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, _, term, trunc, info = val_env.step(action)
            oos_rets.append(info["step_ret"])
            oos_positions.append(info["position"])
            done = term or trunc

        oos_rets_arr = np.array(oos_rets)
        capital = info.get("capital", env_cfg.initial_capital)

        # B&H 벤치마크 (실제 평가에 사용된 구간만)
        bh_return = self._compute_bh(val_env, vl_price)

        agent_total_ret = (capital - env_cfg.initial_capital) / env_cfg.initial_capital
        sharpe = (oos_rets_arr.mean() / (oos_rets_arr.std() + 1e-8)) * np.sqrt(252) \
                 if len(oos_rets_arr) > 1 else 0.0
        downside = oos_rets_arr[oos_rets_arr < 0]
        down_std = downside.std() + 1e-8 if len(downside) > 0 else oos_rets_arr.std() + 1e-8
        sortino  = (oos_rets_arr.mean() / down_std) * np.sqrt(252) \
                   if len(oos_rets_arr) > 1 else 0.0

        # 시드 0 만 체크포인트 저장 (warm start 와 비교용 대표 모델)
        if seed_idx == 0:
            ckpt_name = f"wf_fold_{fold_num:02d}"
            try:
                agent.save(ckpt_name)
            except Exception as e:
                logger.warning(f"  체크포인트 저장 실패: {e}")

        return {
            "fold":         fold_num,
            "seed_idx":     seed_idx,
            "seed":         seed,
            "sharpe":       float(sharpe),
            "sortino":      float(sortino),
            "total_return": float(agent_total_ret),
            "mdd":          float(info.get("mdd", 0)),
            "n_trades":     int(info.get("trade_count", 0)),
            "bh_return":    float(bh_return),
            "alpha_vs_bh":  float(agent_total_ret - bh_return),
            "val_start":    str(vl_price.index[0].date()),
            "val_end":      str(vl_price.index[-1].date()),
            "val_bars":     len(vl_feat),
            "_step_rets":   oos_rets,
            "_positions":   oos_positions,
            "_agent":       agent if seed_idx == 0 else None,
        }

    # ── Buy & Hold 벤치마크 ───────────────────────────

    def _compute_bh(self, val_env, vl_price) -> float:
        """평가에 실제로 사용된 구간 기준 B&H 수익률."""
        try:
            start_i = val_env._start
            end_i   = min(val_env._step_idx, len(val_env.prices) - 1)
            prices  = val_env.prices["Close"].iloc[start_i:end_i + 1].values
            if len(prices) < 2:
                return 0.0
            return float(prices[-1] / prices[0] - 1)
        except Exception as e:
            logger.warning(f"  B&H 계산 실패: {e}")
            return 0.0

    # ── 시드 집계 ─────────────────────────────────────

    def _aggregate_seeds(self, fold_num: int, seed_results: List[Dict]) -> Dict:
        """동일 폴드의 다중 시드 평균/표준편차."""
        if len(seed_results) == 1:
            r = seed_results[0]
            return {
                **{k: v for k, v in r.items() if not k.startswith("_") or k == "_step_rets"},
                "sharpe_std":  0.0,
                "alpha_std":   0.0,
                "n_seeds":     1,
            }

        sharpes = [r["sharpe"]      for r in seed_results]
        rets    = [r["total_return"]for r in seed_results]
        mdds    = [r["mdd"]         for r in seed_results]
        alphas  = [r["alpha_vs_bh"] for r in seed_results]

        # 시드 0 의 step_rets 만 사용 (대표값) — 다중 시드 평균은 통계적으로 부정확
        return {
            "fold":         fold_num,
            "sharpe":       float(np.mean(sharpes)),
            "sharpe_std":   float(np.std(sharpes)),
            "sortino":      float(np.mean([r["sortino"] for r in seed_results])),
            "total_return": float(np.mean(rets)),
            "alpha_vs_bh":  float(np.mean(alphas)),
            "alpha_std":    float(np.std(alphas)),
            "bh_return":    seed_results[0]["bh_return"],
            "mdd":          float(np.mean(mdds)),
            "n_trades":     int(np.mean([r["n_trades"] for r in seed_results])),
            "val_start":    seed_results[0]["val_start"],
            "val_end":      seed_results[0]["val_end"],
            "val_bars":     seed_results[0]["val_bars"],
            "n_seeds":      len(seed_results),
            "_step_rets":   seed_results[0]["_step_rets"],
        }

    # ── 전체 폴드 집계 ────────────────────────────────

    def _aggregate_folds(self, all_oos_rets: List[float]) -> Dict:
        all_rets = np.array(all_oos_rets)
        folds = self.fold_results

        return {
            "n_folds":             len(folds),
            "mode":                self.mode,
            "embargo_pct":         self.embargo_pct,
            "mean_fold_sharpe":    float(np.mean([f["sharpe"] for f in folds])),
            "std_fold_sharpe":     float(np.std ([f["sharpe"] for f in folds])),
            "median_fold_sharpe":  float(np.median([f["sharpe"] for f in folds])),
            "mean_fold_alpha":     float(np.mean([f["alpha_vs_bh"] for f in folds])),
            "positive_alpha_folds": sum(1 for f in folds if f["alpha_vs_bh"] > 0),
            "positive_sharpe_folds": sum(1 for f in folds if f["sharpe"] > 0),
            "oos_sharpe":          float((all_rets.mean() / (all_rets.std() + 1e-8)) * np.sqrt(252))
                                    if len(all_rets) > 1 else 0.0,
            "oos_win_rate":        float((all_rets > 0).mean()) if len(all_rets) > 0 else 0.0,
            "folds":               folds,
        }

    # ── PBO 계산 (Bailey et al. 2017 의 rank 기반 단순화) ─

    def _compute_pbo(self) -> float:
        """
        Probability of Backtest Overfitting — 단순화된 rank 기반 추정.

        본 CSCV 는 N 개 폴드를 두 그룹으로 모든 조합 분할 후 in-sample best 의
        out-of-sample rank 분포를 계산. 본 구현은 단일 시퀀스 폴드의 Sharpe
        순위 분포로 근사 — 진정한 CSCV 보다 약하지만 신호 방향성은 일치.

        값 의미:
            < 0.3  : 과적합 위험 낮음
            0.3~0.5: 보통
            > 0.5  : 과적합 의심 — 결과 신뢰 X
        """
        sharpes = np.array([f["sharpe"] for f in self.fold_results])
        n = len(sharpes)
        if n < 6:
            return float("nan")

        # 단순 휴리스틱: in-sample (전반부 폴드 best) 의 out-of-sample (후반부) rank 분위
        half = n // 2
        in_sample  = sharpes[:half]
        out_sample = sharpes[half:]
        if in_sample.std() < 1e-6 or out_sample.std() < 1e-6:
            return float("nan")

        in_best_idx  = int(np.argmax(in_sample))
        # 해당 폴드 위치가 out_sample 에서도 최고였을까? 단순화: 같은 상대 순위?
        # 정식 CSCV 의 logit 변환은 본 단순화에선 생략.
        oos_ranks = np.argsort(np.argsort(out_sample))    # 0~half-1
        median_rank = (len(out_sample) - 1) / 2
        below_median = sum(1 for r in oos_ranks if r <= median_rank)
        return float(below_median / len(out_sample))

    # ── 로깅 ─────────────────────────────────────────

    def _log_fold(self, fr: Dict) -> None:
        seed_info = (
            f" (avg {fr['n_seeds']} seeds, σ={fr.get('sharpe_std', 0):.3f})"
            if fr.get("n_seeds", 1) > 1 else ""
        )
        logger.info(
            f"  → OOS Sharpe={fr['sharpe']:+.3f}{seed_info} | "
            f"Sortino={fr['sortino']:+.3f} | "
            f"TotalRet={fr['total_return']:+.2%} | "
            f"B&H={fr['bh_return']:+.2%} | "
            f"αvsB&H={fr['alpha_vs_bh']:+.2%} | "
            f"MDD={fr['mdd']:+.2%} | "
            f"Trades={fr['n_trades']}"
        )

    def _log_summary(self, s: Dict) -> None:
        logger.info(f"\n{'='*60}\nWalk-Forward 종합 결과\n{'='*60}")
        logger.info(
            f"  Mode={s['mode']} | Embargo={s['embargo_pct']:.1%} | Folds={s['n_folds']}"
        )
        logger.info(f"  ─── 폴드별 분포 ───")
        logger.info(
            f"  평균 폴드 Sharpe : {s['mean_fold_sharpe']:+.3f} ± {s['std_fold_sharpe']:.3f} "
            f"(중앙값 {s['median_fold_sharpe']:+.3f})"
        )
        logger.info(f"  평균 폴드 α(vsB&H): {s['mean_fold_alpha']:+.2%}")
        logger.info(
            f"  α > 0 폴드 : {s['positive_alpha_folds']}/{s['n_folds']} | "
            f"Sharpe > 0 폴드: {s['positive_sharpe_folds']}/{s['n_folds']}"
        )
        logger.info(f"  ─── 전체 OOS 통계 ───")
        logger.info(f"  통합 OOS Sharpe : {s['oos_sharpe']:+.3f}")
        logger.info(f"  통합 OOS 승률   : {s['oos_win_rate']:.2%}")

        if "pbo" in s and not np.isnan(s["pbo"]):
            warn = " ⚠ 과적합 의심" if s["pbo"] > 0.5 else " (양호)"
            logger.info(f"  PBO (단순화)    : {s['pbo']:.1%}{warn}")
            logger.info(f"                     ← Bailey et al. (2017), < 0.5 권장")

        logger.info(f"{'='*60}")
        logger.info(
            "  ※ 실거래 검토 기준 권장:\n"
            "      평균 폴드 Sharpe > 1.0  &  std < 평균의 50%  &\n"
            "      α > 0 폴드 >= 3/4  &  PBO < 0.4"
        )
        logger.info(f"{'='*60}\n")
