"""
agent/replay_buffer.py — 경험 리플레이 버퍼 (Uniform + PER + N-step)

Off-policy 학습의 핵심.

제공 클래스:
  ReplayBuffer              — 균일 샘플링 (Mnih 2015)
  PrioritizedReplayBuffer   — TD-error 기반 우선 샘플링 (Schaul 2015) + N-step
  NStepWrapper              — 임의 버퍼에 N-step return 누적 기능 추가

참고:
  - Mnih et al. "Human-level control through deep RL" (2015)
  - Schaul et al. "Prioritized Experience Replay" (2015)
  - Hessel et al. "Rainbow: Combining Improvements in Deep RL" (2018)

PER 핵심:
    p_i  = (|δ_i| + ε)^α          ── 우선순위
    P(i) = p_i / Σ p_j             ── 샘플링 확률
    w_i  = (N · P(i))^(-β)         ── IS 보정 가중치 (Critic 손실에 곱)
    β    : 0 → 1 (학습 진행에 따라 어닐링)
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional, Dict


# ──────────────────────────────────────────────────────
# Uniform Replay (기존 동작 유지)
# ──────────────────────────────────────────────────────

class ReplayBuffer:
    """
    균일 샘플링 경험 리플레이 버퍼.
    """

    def __init__(self, obs_dim: int, action_dim: int, capacity: int = 100_000):
        self.capacity   = capacity
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self._ptr       = 0
        self._size      = 0
        self.is_per     = False

        self.obs     = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_obs= np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.dones   = np.zeros((capacity, 1),          dtype=np.float32)
        # 1-step gamma 효과: target = r + γ · Q(s', ·)
        self.gammas  = np.ones((capacity, 1),           dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        gamma_eff: float = None,
    ):
        self.obs[self._ptr]      = obs
        self.actions[self._ptr]  = action
        self.rewards[self._ptr]  = reward
        self.next_obs[self._ptr] = next_obs
        self.dones[self._ptr]    = float(done)
        if gamma_eff is not None:
            self.gammas[self._ptr] = gamma_eff

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def end_of_episode(self) -> None:
        """에피소드 경계 신호. Uniform buffer 는 n-step 누적이 없어 no-op."""
        pass

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        return self._size >= 256


# ──────────────────────────────────────────────────────
# Sum-Tree (PER 의 효율적 자료구조)
# ──────────────────────────────────────────────────────

class _SumTree:
    """
    Schaul (2015) §3.3 의 sum-tree.
    리프 = priority, 내부 노드 = 자식 합.
    sample(s): O(log N), update(i, p): O(log N).
    """

    def __init__(self, capacity: int):
        # 다음 2의 거듭제곱으로 패딩하면 인덱싱이 단순해진다
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity, dtype=np.float64)
        self._max_p   = 1.0   # 새 transition 의 초기 priority

    def total(self) -> float:
        return float(self.tree[1])

    @property
    def max_priority(self) -> float:
        return self._max_p

    def update(self, leaf_idx: int, priority: float):
        """leaf_idx ∈ [0, capacity) — 외부 인덱스."""
        idx = leaf_idx + self.capacity
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        # 부모 전파
        idx //= 2
        while idx >= 1:
            self.tree[idx] += change
            idx //= 2
        if priority > self._max_p:
            self._max_p = priority

    def get(self, s: float) -> Tuple[int, float]:
        """
        누적 s 에 해당하는 리프 (외부 인덱스) 와 priority 반환.
        s ∈ [0, total()].
        """
        idx = 1
        while idx < self.capacity:
            left  = 2 * idx
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s  -= self.tree[left]
                idx = right
        leaf_idx = idx - self.capacity
        return leaf_idx, float(self.tree[idx])


# ──────────────────────────────────────────────────────
# Prioritized Experience Replay + N-step
# ──────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Schaul (2015) Prioritized Experience Replay + N-step return.

    저장 구조:
      (s_t, a_t,  R_t^{(n)},  s_{t+n}, done_{t+n}, γ^n_eff)
      R_t^{(n)} = Σ_{k=0..n-1} γ^k · r_{t+k}     ── 에피소드 종료 시 cut
      γ^n_eff   = γ^k_terminated  (n-step 완주 시 γ^n)

    사용 (`add`):
      add(s, a, r, s', done) 호출만 하면 내부 deque 가 n-step transition 으로
      변환해 push. 에피소드 경계는 done 으로 식별.

    참고:
      Sutton (1988) TD(n)  → bias-variance trade-off:
        n 이 작을수록 bias 큼 (bootstrap 강함),
        n 이 클수록 variance 큼 (Monte-Carlo 풍).
        일반적으로 3 ~ 5 가 안정적.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int = 100_000,
        n_step: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 200_000,
        eps: float = 1e-6,
        use_lap: bool = False,
        lap_min_priority: float = 1.0,
    ):
        self.capacity    = capacity
        self.obs_dim     = obs_dim
        self.action_dim  = action_dim
        self.n_step      = max(1, int(n_step))
        self.gamma       = float(gamma)
        self.alpha       = float(alpha)
        self.beta_start  = float(beta_start)
        self.beta_end    = float(beta_end)
        self.beta_anneal_steps = int(beta_anneal_steps)
        self.eps         = float(eps)
        self.is_per      = True
        # LAP — Loss-Adjusted Prioritization (Fujimoto et al., 2020)
        #   priority = max(|δ|, λ_min)  (α 지수승 없음)
        #   외부에서 critic 도 Huber loss 사용해야 등가 보정 성립
        self.use_lap = bool(use_lap)
        self.lap_min_priority = float(lap_min_priority)

        self._ptr   = 0
        self._size  = 0
        self._step  = 0   # β 어닐링용 글로벌 카운터 (외부에서 set_step 호출)

        # 본 저장소
        self.obs      = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.actions  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),          dtype=np.float32)
        self.gammas   = np.full ((capacity, 1), gamma,   dtype=np.float32)  # γ^n_eff

        # Sum-Tree
        self.tree = _SumTree(capacity)

        # N-step 누적 deque
        self._nstep_buf: deque = deque(maxlen=self.n_step)

    # ── β 어닐링 ─────────────────────────────────────

    def set_step(self, global_step: int):
        self._step = int(global_step)

    @property
    def beta(self) -> float:
        t = min(self._step, self.beta_anneal_steps)
        frac = t / max(self.beta_anneal_steps, 1)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    # ── 저장 ─────────────────────────────────────────

    def add(self, obs, action, reward, next_obs, done, gamma_eff=None):
        """
        single-step transition 을 받아 n-step 으로 누적 후 push.
        """
        self._nstep_buf.append((obs, action, float(reward), next_obs, float(done)))

        # n-step 가득 차면 첫 transition 으로 변환
        if len(self._nstep_buf) >= self.n_step:
            self._push_nstep()

        # 에피소드 종료: 큐에 남은 transition 들도 모두 flush
        if done:
            while self._nstep_buf:
                self._push_nstep()

    def end_of_episode(self) -> None:
        """
        시간제한 truncation 등 done=0 으로 저장됐지만 실제로 에피소드가 끝난 경우
        남은 n-step 누적분을 강제 flush.

        SAC value bootstrap 보존:
          저장된 transition 의 done 플래그는 건드리지 않는다 (이미 0).
          target_q = r + γ^n_eff · (1 - 0) · Q(next_s, ...) 그대로 작동.
          n-step 의 next_s 는 truncation 직전 next_obs (이미 add 시점에 캡처됨)
          이므로 같은 에피소드 내 유효 관측.

        다종목 학습 필수:
          다음 에피소드가 다른 종목이면, deque 잔여분이 다음 add 와 섞이며
          old-ticker s0 + (old+new) reward + new-ticker next_s 로 변형되어
          치명적인 cross-ticker leakage 발생. 이 메서드 호출로 차단.
        """
        while self._nstep_buf:
            self._push_nstep()

    def _push_nstep(self):
        """deque 의 첫 transition 을 n-step 보상으로 누적해 본 저장소에 기록."""
        if not self._nstep_buf:
            return

        s0, a0, _, _, _ = self._nstep_buf[0]
        R     = 0.0
        gamma = 1.0
        done_n  = 0.0
        next_s  = self._nstep_buf[-1][3]
        gamma_eff = 1.0

        for (_, _, r, ns, d) in self._nstep_buf:
            R += gamma * r
            gamma_eff = gamma * self.gamma
            next_s = ns
            if d:
                done_n = 1.0
                gamma  = 0.0   # 종료 후 보상 없음
                break
            gamma *= self.gamma

        # 본 저장소에 기록 + sum-tree priority = max_priority (새 데이터 우선)
        i = self._ptr
        self.obs[i]      = s0
        self.actions[i]  = a0
        self.rewards[i]  = R
        self.next_obs[i] = next_s
        self.dones[i]    = done_n
        self.gammas[i]   = gamma_eff

        # 새 transition 의 초기 priority — LAP 모드는 지수승 없음
        if self.use_lap:
            init_p = max(self.tree.max_priority, self.lap_min_priority)
        else:
            init_p = self.tree.max_priority ** self.alpha
        self.tree.update(i, init_p)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

        self._nstep_buf.popleft()

    # ── 샘플링 ────────────────────────────────────────

    def sample(self, batch_size: int):
        """
        Returns
        -------
        (obs, actions, rewards, next_obs, dones, gammas, is_weights, indices)
        - rewards : n-step 누적 보상
        - gammas  : γ^n_eff (target 의 bootstrap 계수)
        - is_weights : IS 보정 가중치, shape (B, 1), 최대 1 로 정규화
        - indices : sum-tree 의 leaf index, update_priorities 에서 사용
        """
        assert self._size > 0, "buffer is empty"
        total = self.tree.total()
        if total <= 0:
            # 초기화 직후 등 priority 가 모두 0 인 경우 → uniform fallback
            idx = np.random.randint(0, self._size, size=batch_size)
            iw  = np.ones((batch_size, 1), dtype=np.float32)
            return (self.obs[idx], self.actions[idx], self.rewards[idx],
                    self.next_obs[idx], self.dones[idx], self.gammas[idx],
                    iw, idx)

        segment   = total / batch_size
        indices   = np.zeros(batch_size, dtype=np.int64)
        priorities= np.zeros(batch_size, dtype=np.float64)

        # 계층화 샘플링: 구간을 batch_size 등분 후 각 구간에서 1개씩
        for k in range(batch_size):
            lo, hi = segment * k, segment * (k + 1)
            s = np.random.uniform(lo, hi)
            leaf, p = self.tree.get(s)
            indices[k]   = leaf
            priorities[k]= p

        # IS weight: w_i = (N · P(i))^(-β),  P(i) = p_i / total
        probs = priorities / max(total, 1e-12)
        beta  = self.beta
        weights = (self._size * np.maximum(probs, 1e-12)) ** (-beta)
        weights /= weights.max() + 1e-12        # 학습 안정화: max → 1
        iw = weights.astype(np.float32).reshape(-1, 1)

        return (self.obs[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_obs[indices],
                self.dones[indices],
                self.gammas[indices],
                iw,
                indices)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        TD-error 기반 priority 갱신.

        모드:
          - 일반 PER (Schaul 2015): p_i = (|δ_i| + ε)^α
          - LAP    (Fujimoto 2020): p_i = max(|δ_i|, λ_min)
            · 지수승 없음 (α는 미사용)
            · 손실 함수가 Huber 여야 IS-bias 가 자동 보정됨
        """
        abs_td = np.abs(td_errors).reshape(-1)
        if self.use_lap:
            for i, e in zip(indices, abs_td):
                p = float(max(e, self.lap_min_priority))
                self.tree.update(int(i), p)
        else:
            for i, e in zip(indices, abs_td + self.eps):
                p = float(e ** self.alpha)
                self.tree.update(int(i), p)

    # ── Dunder ───────────────────────────────────────

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        return self._size >= 256
