"""
networks/sac_nets.py — SAC 신경망 아키텍처

참고 이론:
  - SAC Actor: Squashed Gaussian Policy (Haarnoja et al., 2018)
      π(a|s) = tanh(μ + ε·σ), ε ~ N(0,I)
      → log_prob 계산 시 tanh Jacobian 보정 필수
      → 엔트로피 최대화로 자연스러운 탐험

  - Twin Critics (TD3 아이디어 차용):
      min(Q₁, Q₂) 사용으로 과대추정 방지 (Fujimoto et al., 2018)

  - Layer Normalization:
      금융 데이터의 비정상성(non-stationarity)에 강건

  - Residual Connections (선택적):
      깊은 네트워크에서 gradient flow 개선
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, List

LOG_STD_MIN = -20
LOG_STD_MAX =  2
EPSILON     = 1e-6


# ── 공통 MLP 빌더 ─────────────────────────────────────

def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: List[int],
    activation: str = "relu",
    output_activation: bool = False,
    use_layer_norm: bool = True,
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    Layer Normalization을 포함한 MLP 구축.
    금융 데이터의 분포 변화(non-stationarity)에 대응.

    dropout > 0 일 때 각 은닉층 활성화 직후에 Dropout 삽입.
    DroQ (Hiraoka et al., 2022): critic 에 작은 dropout 을 켜 두면
    M개 critic 평균과 유사한 효과를 작은 비용으로 얻을 수 있다.
    Dropout 은 학습/타겟 양쪽에서 활성 상태로 두는 것이 핵심.
    """
    act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[activation]
    layers = []
    dims   = [in_dim] + hidden_dims

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if use_layer_norm:
            layers.append(nn.LayerNorm(dims[i+1]))
        layers.append(act_fn())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

    layers.append(nn.Linear(dims[-1], out_dim))
    if output_activation:
        layers.append(act_fn())

    return nn.Sequential(*layers)


# ── Actor (Squashed Gaussian Policy) ─────────────────

class GaussianActor(nn.Module):
    """
    SAC의 확률론적 정책 네트워크.

    출력: μ, log_σ → 가우시안 샘플링 → tanh squash
    tanh squash로 행동을 [-1, 1] 범위로 강제 (포지션 한계)

    log_prob 보정:
        log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u_i))
        (tanh Jacobian에 의한 보정, 수치 안정성을 위해 log-space 계산)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int], activation: str = "relu"):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_dims[-1], hidden_dims[:-1],
                             activation=activation, use_layer_norm=True)
        self.mu_head      = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

        # 가중치 초기화 (작은 초기 출력으로 안정적인 학습 시작)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        action   : tanh squashed 행동 ∈ (-1, 1)
        log_prob : 행동의 로그 확률 (엔트로피 계산에 사용)
        """
        x       = self.net(obs)
        mu      = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()

        dist = Normal(mu, std)
        u    = dist.rsample()   # reparameterization trick (gradient 통과)

        action   = torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, obs: torch.Tensor) -> np.ndarray:
        """추론 전용 (gradient 불필요)."""
        with torch.no_grad():
            action, _ = self.forward(obs)
        return action.cpu().numpy()

    def get_deterministic_action(self, obs: torch.Tensor) -> np.ndarray:
        """평가 전용: 확률적 탐험 없이 결정론적 행동."""
        with torch.no_grad():
            x  = self.net(obs)
            mu = self.mu_head(x)
        return torch.tanh(mu).cpu().numpy()

    def deterministic_mu(self, obs: torch.Tensor) -> torch.Tensor:
        """
        결정론적 tanh-squashed 평균 행동 (gradient 통과).
        CAPS smoothness loss 계산에 사용 — actor 학습 신호로 흘려야 하므로
        get_deterministic_action 과 달리 no_grad 가 아니다.
        """
        x  = self.net(obs)
        mu = self.mu_head(x)
        return torch.tanh(mu)


# ── Critic (Twin Q-Networks) ──────────────────────────

class TwinCritic(nn.Module):
    """
    Twin Q-Network (TD3 / SAC-v2 표준).

    두 개의 독립적인 Q-네트워크를 학습하고
    min(Q₁, Q₂)를 target으로 사용하여 과대추정(overestimation) 방지.

    Q(s, a): 상태-행동 쌍의 가치 추정
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int],
                 activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        in_dim = obs_dim + action_dim

        self.q1 = build_mlp(in_dim, 1, hidden_dims, activation=activation,
                            use_layer_norm=True, dropout=dropout)
        self.q2 = build_mlp(in_dim, 1, hidden_dims, activation=activation,
                            use_layer_norm=True, dropout=dropout)

        self._init_weights()

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """두 Q값 모두 반환 (학습 시 사용)."""
        x  = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """min(Q₁, Q₂) 반환 (target 계산 시 사용)."""
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)

    def _init_weights(self):
        for net in [self.q1, self.q2]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.zeros_(m.bias)
