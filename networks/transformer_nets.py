"""
networks/transformer_nets.py — Transformer 기반 SAC Actor / TwinCritic

설계 (참고 논문):

  ① GTrXL (Parisotto et al., ICML 2020)
     "Stabilizing Transformers for Reinforcement Learning"
     - Online RL 에서 표준 transformer 가 발산하는 두 가지 원인:
         a) Post-LN residual : `x = LN(x + Sublayer(x))` → 학습 초기 분산 폭주
         b) Identity-deviation : sublayer 출력이 곧장 residual 에 합쳐져 정책 변동 큼
     - 해결책 두 개:
         a) Identity-map reordering (pre-LN): `x = x + Sublayer(LN(x))`
         b) GRU-style gating: `(1-z)·x + z·tanh(Wg[r·x, y])`, z 의 bias 를
            큰 양수로 초기화 → 학습 초기 z≈0 → output ≈ x (identity).
            정책이 transformer 잡음 없이 안정 출발.

  ② PatchTST (Nie et al., ICLR 2023)
     "A Time Series is Worth 64 Words"
     - 시계열 (T,F) 를 patch 로 자르고 channel-independent 처리.
     - 본 코드는 window=30 으로 patch 작음 → patch 대신 step-level 토큰
       (T=window 토큰, 각 토큰 d_model 차원) 사용.
     - 학습 가능한 positional embedding.

  ③ RevIN (Kim et al., ICLR 2022)
     "Reversible Instance Normalization for Distribution Shift"
     - 각 sample 시계열별 mean/std 정규화 (LayerNorm 과 달리 시간축 기준).
     - 트레이딩처럼 비정상성(non-stationary) 강한 환경에서 distribution shift 강건.
     - 본 코드는 encoder 만 쓰므로 forecasting 의 denorm 단계는 생략.

  ④ Decision Transformer (Chen et al., NeurIPS 2021)
     영감 — RL 시퀀스 모델링 자체는 sequence-conditional 정책 학습이 가능함을 입증.
     본 구현은 SAC 의 actor/critic 을 transformer encoder 로 갈아끼우는 방식
     (DT 처럼 sequence-conditional 행동 생성은 아님).

구조:
  obs = [window×n_feat features │ portfolio_state(4)]  ← env 가 만든 flat vector
   ├─ split → time-series block (B, window, n_feat) + port_feat (B, 4)
   ├─ TimeSeriesEncoder(time-series block) → (B, d_model)
   │     RevIN → Linear → +PosEmbed → TransformerEncoder(pre-LN) → GTrXLGate → last token
   └─ concat with port_feat → MLP head → [μ, logσ] or Q
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX =  2
EPSILON     = 1e-6


# ── RevIN: Reversible Instance Normalization (Kim et al., ICLR 2022) ────────

class RevIN(nn.Module):
    """
    Per-sample, per-feature instance normalization across the time dim.

    forward(x):  x ∈ (B, T, F) → normalized (B, T, F).

    트레이딩 환경의 강한 비정상성(distribution shift) 하에서 LayerNorm/BatchNorm 보다
    더 안정. forecasting 의 denorm 단계는 SAC encoder 에선 불필요해 생략.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps          = eps
        self.affine       = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) — normalize along T per (B, F)
        mean = x.mean(dim=1, keepdim=True)
        var  = x.var(dim=1, keepdim=True, unbiased=False)
        x    = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x


# ── GTrXL Gated Residual (Parisotto et al., ICML 2020) ───────────────────────

class GTrXLGate(nn.Module):
    """
    GRU-style gated residual (Parisotto 2020 §3.2, Eq. 8-10).

        r = σ(W_r [x, y])
        z = σ(W_z [x, y] - b_g)   ← b_g 큰 양수면 z≈0 → output≈x (identity-init)
        h = tanh(W_g [r·x, y])
        output = (1 - z) · x + z · h

    학습 초기에 encoder 가 identity 처럼 작동 → 정책이 transformer 잡음 없이 출발 →
    online RL 에서 발산 위험 크게 감소 (논문 §4.2).
    """
    def __init__(self, d_model: int, bg: float = 2.0):
        super().__init__()
        self.W_r = nn.Linear(2 * d_model, d_model, bias=False)
        self.W_z = nn.Linear(2 * d_model, d_model, bias=True)
        self.W_g = nn.Linear(2 * d_model, d_model, bias=False)
        # 핵심: W_z bias 를 -b_g 로 두어 학습 초기에 z≈sigmoid(-2)≈0.12
        # → output ≈ 0.88·x + 0.12·h ≈ identity
        nn.init.constant_(self.W_z.bias, -float(bg))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: identity residual,  y: sublayer output (e.g. transformer encoded)
        xy   = torch.cat([x, y], dim=-1)
        r    = torch.sigmoid(self.W_r(xy))
        z    = torch.sigmoid(self.W_z(xy))
        rx_y = torch.cat([r * x, y], dim=-1)
        h    = torch.tanh(self.W_g(rx_y))
        return (1.0 - z) * x + z * h


# ── Positional Embedding ─────────────────────────────────────────────────────

class LearnedPositionalEmbedding(nn.Module):
    """학습 가능한 positional embedding (PatchTST 표준)."""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1), :]


# ── TimeSeriesEncoder: PatchTST + GTrXL + RevIN ──────────────────────────────

class TimeSeriesEncoder(nn.Module):
    """
    Time-series (B, window, n_feat) → embedding (B, d_model).

    pipeline:
        x   (B, T, F)
        ├─ RevIN                         (Kim 2022 — distribution shift)
        ├─ Linear(F → d_model)           (PatchTST: input projection)
        ├─ + LearnedPositionalEmbedding
        ├─ TransformerEncoder(pre-LN)    (Parisotto 2020 §3.1)
        ├─ GTrXLGate(input_proj, encoded)(Parisotto 2020 §3.2 — identity-init)
        ├─ LayerNorm
        └─ last-token pool → (B, d_model)
    """
    def __init__(
        self,
        n_feat:      int,
        window:      int,
        d_model:     int  = 64,
        n_heads:     int  = 4,
        n_layers:    int  = 2,
        dropout:     float= 0.1,
        use_revin:   bool = True,
        use_gating:  bool = True,
        ff_mult:     int  = 4,
    ):
        super().__init__()
        self.window  = window
        self.n_feat  = n_feat
        self.d_model = d_model

        self.revin = RevIN(n_feat) if use_revin else None

        self.input_proj = nn.Linear(n_feat, d_model)
        self.pos_embed  = LearnedPositionalEmbedding(window, d_model)
        self.in_drop    = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # pre-LN (GTrXL §3.1 identity-map reordering)
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * ff_mult,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.use_gating = use_gating
        if use_gating:
            self.out_gate = GTrXLGate(d_model, bg=2.0)

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        if self.revin is not None:
            x = self.revin(x)

        h = self.input_proj(x)              # (B, T, D)
        h = self.pos_embed(h)
        h = self.in_drop(h)

        enc = self.encoder(h)               # (B, T, D)

        if self.use_gating:
            # identity-init gating — 학습 초기에 enc ≈ h (input proj) 로 시작
            enc = self.out_gate(h, enc)

        enc = self.final_norm(enc)

        # trading 에선 최근 정보가 가장 중요 → last-token pool (auto-regressive style)
        # 평균 pool 보다 작은 시그널이 안 묻힘 (DT/GTrXL 도 동일 방식).
        return enc[:, -1, :]                # (B, D)


# ── Transformer Actor (Squashed Gaussian Policy) ─────────────────────────────

class TransformerGaussianActor(nn.Module):
    """
    SAC 의 stochastic actor — MLP 대신 TimeSeriesEncoder + portfolio MLP head.

    obs 입력 형식 (env 와 일관):
        obs = [window×n_feat features │ portfolio_state(port_dim)]   shape (B, obs_dim)

    내부에서:
        features → (B, window, n_feat) → TimeSeriesEncoder → (B, d_model)
        port_feat (B, port_dim) 와 concat → MLP head → μ, log_σ
    """
    def __init__(
        self,
        window:      int,
        n_feat:      int,
        port_dim:    int,
        action_dim:  int,
        d_model:     int  = 64,
        n_heads:     int  = 4,
        n_layers:    int  = 2,
        dropout:     float= 0.1,
        use_revin:   bool = True,
        use_gating:  bool = True,
    ):
        super().__init__()
        self.window   = window
        self.n_feat   = n_feat
        self.port_dim = port_dim
        self._feat_dim = window * n_feat

        self.encoder = TimeSeriesEncoder(
            n_feat, window, d_model, n_heads, n_layers,
            dropout, use_revin, use_gating,
        )

        head_in = d_model + port_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.mu_head      = nn.Linear(d_model, action_dim)
        self.log_std_head = nn.Linear(d_model, action_dim)

        # 작은 초기 출력으로 학습 초반 발산 방지 (SAC 표준)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.zeros_(self.log_std_head.bias)

    def _split_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = obs[:, : self._feat_dim].reshape(-1, self.window, self.n_feat)
        port = obs[:, self._feat_dim:]
        return feat, port

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        feat, port = self._split_obs(obs)
        enc = self.encoder(feat)            # (B, d_model)
        x   = torch.cat([enc, port], dim=-1)
        return self.head(x)                 # (B, d_model)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z       = self._encode(obs)
        mu      = self.mu_head(z)
        log_std = self.log_std_head(z).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()

        dist     = Normal(mu, std)
        u        = dist.rsample()
        action   = torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def get_action(self, obs: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            action, _ = self.forward(obs)
        return action.cpu().numpy()

    def get_deterministic_action(self, obs: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            z  = self._encode(obs)
            mu = self.mu_head(z)
        return torch.tanh(mu).cpu().numpy()

    def deterministic_mu(self, obs: torch.Tensor) -> torch.Tensor:
        """CAPS / BC loss 용 — gradient 흐름 유지."""
        z  = self._encode(obs)
        mu = self.mu_head(z)
        return torch.tanh(mu)


# ── Transformer Twin Critic ─────────────────────────────────────────────────

class TransformerTwinCritic(nn.Module):
    """
    Twin Q-Network — encoder 공유 + 두 개의 Q head.

    설계 결정 — encoder 공유 vs 분리:
      • SAC TwinCritic 표준은 두 Q 가 완전 분리 (학습 데이터 동일, 초기화 다름).
      • 본 구현은 encoder 공유 + head 분리 → 메모리 절반, 표현 학습 빠름.
        두 Q head 의 gradient 가 평균되어 encoder 학습 → 분리 대비 OOD 강건성
        약간 손해 가능. 그러나 DroQ dropout 이 동일 효과 일부 보강 (Hiraoka 2022).
      • 메모리 여유 있고 더 보수적으로 가고 싶으면 share_encoder=False 로 변경.

    obs 형식: TransformerGaussianActor 와 동일.
    """
    def __init__(
        self,
        window:        int,
        n_feat:        int,
        port_dim:      int,
        action_dim:    int,
        d_model:       int  = 64,
        n_heads:       int  = 4,
        n_layers:      int  = 2,
        dropout:       float= 0.1,
        use_revin:     bool = True,
        use_gating:    bool = True,
        share_encoder: bool = True,
    ):
        super().__init__()
        self.window     = window
        self.n_feat     = n_feat
        self.port_dim   = port_dim
        self._feat_dim  = window * n_feat
        self._share     = share_encoder

        enc_kwargs = dict(
            n_feat=n_feat, window=window, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers,
            dropout=dropout, use_revin=use_revin, use_gating=use_gating,
        )

        if share_encoder:
            self.encoder = TimeSeriesEncoder(**enc_kwargs)
        else:
            self.encoder1 = TimeSeriesEncoder(**enc_kwargs)
            self.encoder2 = TimeSeriesEncoder(**enc_kwargs)

        head_in    = d_model + port_dim + action_dim
        self.q1_head = self._build_q_head(head_in, d_model, dropout)
        self.q2_head = self._build_q_head(head_in, d_model, dropout)

        # critic 마지막 layer 작은 초기화 — Q 값 폭주 억제
        for m in (self.q1_head, self.q2_head):
            last_linear = [mm for mm in m.modules() if isinstance(mm, nn.Linear)][-1]
            nn.init.orthogonal_(last_linear.weight, gain=1.0)
            nn.init.zeros_(last_linear.bias)

    @staticmethod
    def _build_q_head(in_dim: int, d_model: int, dropout: float) -> nn.Sequential:
        layers = [
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers += [
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(d_model, 1))
        return nn.Sequential(*layers)

    def _encode(self, obs: torch.Tensor, which: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = obs[:, : self._feat_dim].reshape(-1, self.window, self.n_feat)
        port = obs[:, self._feat_dim:]
        if self._share:
            enc = self.encoder(feat)
        else:
            enc = self.encoder1(feat) if which == 0 else self.encoder2(feat)
        return enc, port

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._share:
            enc, port = self._encode(obs, which=0)
            xa = torch.cat([enc, port, action], dim=-1)
            return self.q1_head(xa), self.q2_head(xa)
        else:
            enc1, port = self._encode(obs, which=0)
            enc2, _    = self._encode(obs, which=1)
            xa1 = torch.cat([enc1, port, action], dim=-1)
            xa2 = torch.cat([enc2, port, action], dim=-1)
            return self.q1_head(xa1), self.q2_head(xa2)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)
