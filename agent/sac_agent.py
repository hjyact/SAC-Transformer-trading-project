"""
agent/sac_agent.py — SAC (Soft Actor-Critic) 에이전트

알고리즘: Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy
           Deep Reinforcement Learning with a Stochastic Actor" (2018)
           + SAC v2: "Soft Actor-Critic Algorithms and Applications" (2019)

핵심 수식:
  ① Critic 손실 (Bellman backup + 엔트로피):
        y = r + γ(1-d) · [min_Q(s',ã') - α·log π(ã'|s')]
        L_Q = E[(Q(s,a) - y)²]   (Twin Q 각각)

  ② Actor 손실 (정책 개선):
        L_π = E[α·log π(a|s) - Q(s,a)]
        → 엔트로피를 최대화하면서 Q값도 최대화

  ③ 온도(α) 자동 조정 (SAC v2):
        L_α = E[-α · (log π(a|s) + H̄)]
        H̄: 목표 엔트로피 (보통 -action_dim)
        → α가 자동으로 탐험-활용 균형 조절

  ④ Soft target update:
        θ_target ← τ·θ + (1-τ)·θ_target
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from config import SACConfig, sac_cfg, DEVICE, CKPT_DIR
from networks.sac_nets import GaussianActor, TwinCritic
from agent.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class SACAgent:
    """
    Soft Actor-Critic 에이전트.

    외부 인터페이스:
        select_action(obs, deterministic)  → 행동 선택
        train_step(buffer)                 → 1회 gradient 업데이트
        save(name) / load(name)            → 체크포인트
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cfg: SACConfig = sac_cfg,
        device: str = DEVICE,
        obs_meta: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        obs_meta : Optional[dict]
            transformer 사용 시 필수. {"window":int, "n_feat":int, "port_dim":int}.
            env 의 (window_size, n_features, portfolio_state_dim) 정보.
            obs 가 [window×n_feat 피처 | port_dim 포트폴리오] 형식이라는 가정.
        """
        self.cfg        = cfg
        self.device     = device
        self.action_dim = action_dim
        self._total_updates = 0
        self.use_transformer = bool(getattr(cfg, "use_transformer", False))
        self.obs_meta = obs_meta

        # ── 네트워크 초기화
        # DroQ: critic 에만 작은 dropout. Actor 는 결정론적 추론 시 일관성을 위해 dropout 없음.
        critic_dropout = float(getattr(cfg, "critic_dropout", 0.0)) \
                         if getattr(cfg, "use_droq", False) else 0.0

        if self.use_transformer:
            self.actor, self.critic, self.critic_target = self._build_transformer_nets(
                obs_dim, action_dim, critic_dropout,
            )
        else:
            self.actor = GaussianActor(
                obs_dim, action_dim, cfg.hidden_dims, cfg.activation
            ).to(device)

            self.critic = TwinCritic(
                obs_dim, action_dim, cfg.hidden_dims, cfg.activation,
                dropout=critic_dropout,
            ).to(device)

            self.critic_target = TwinCritic(
                obs_dim, action_dim, cfg.hidden_dims, cfg.activation,
                dropout=critic_dropout,
            ).to(device)
        # DroQ: target critic 의 dropout 은 target_q 계산 시에도 활성 상태 유지 (eval() 호출 X).
        # 단, target 은 requires_grad=False 로 잠가 둠 (soft update 만 받음).

        # target 네트워크는 critic과 동일하게 초기화 후 고정 (soft update만)
        self._hard_update(self.critic_target, self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ── 옵티마이저
        # AdamW (Loshchilov & Hutter, ICLR 2019) — Decoupled Weight Decay.
        # cfg.weight_decay=0 이면 vanilla Adam 과 등가, >0 이면 L2 정규화 추가.
        # 작은 데이터셋 + 작은 시장 신호 환경에서 critic 가중치 발산 억제.
        wd = float(getattr(cfg, "weight_decay", 0.0))
        self.actor_opt  = torch.optim.AdamW(
            self.actor.parameters(),  lr=cfg.actor_lr,  weight_decay=wd,
        )
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(), lr=cfg.critic_lr, weight_decay=wd,
        )

        # ── 엔트로피 온도 α (SAC v2 자동 조정)
        if cfg.auto_alpha:
            self.log_alpha   = torch.tensor(
                np.log(cfg.alpha), dtype=torch.float32,
                device=device, requires_grad=True
            )
            self.alpha_opt   = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
            self.target_entropy = cfg.target_entropy if cfg.target_entropy != -1.0 \
                                  else -float(action_dim)
        else:
            self.log_alpha  = torch.tensor(np.log(cfg.alpha), device=device)

        # 로그
        self._loss_log = {"critic": [], "actor": [], "alpha": [], "alpha_val": []}

    # ── Transformer backbone 빌더 ─────────────────────

    def _build_transformer_nets(self, obs_dim: int, action_dim: int, critic_dropout: float):
        """
        Transformer Actor / TwinCritic / target TwinCritic 생성.

        GTrXL + PatchTST + RevIN 결합 (networks/transformer_nets.py 참조).
        obs_meta 가 없거나 obs_dim 과 일치하지 않으면 검증 실패 → 오류.
        """
        from networks.transformer_nets import (
            TransformerGaussianActor, TransformerTwinCritic,
        )
        if self.obs_meta is None:
            raise ValueError(
                "use_transformer=True 일 때 SACAgent(..., obs_meta={"
                "'window':W, 'n_feat':F, 'port_dim':P}) 필수. "
                "main.py 가 train_env 로부터 obs_meta 를 전달해야 합니다."
            )
        W = int(self.obs_meta["window"])
        F_ = int(self.obs_meta["n_feat"])
        P = int(self.obs_meta["port_dim"])
        expected = W * F_ + P
        if expected != obs_dim:
            raise ValueError(
                f"obs_meta 와 obs_dim 불일치: W*F+P={expected} vs obs_dim={obs_dim}. "
                f"env 의 window_size/n_feat/port_dim 을 다시 확인하세요."
            )

        cfg = self.cfg
        common = dict(
            window      = W,
            n_feat      = F_,
            port_dim    = P,
            action_dim  = action_dim,
            d_model     = int(getattr(cfg, "trans_d_model",  64)),
            n_heads     = int(getattr(cfg, "trans_n_heads",  4)),
            n_layers    = int(getattr(cfg, "trans_n_layers", 2)),
            use_revin   = bool(getattr(cfg, "trans_use_revin", True)),
            use_gating  = bool(getattr(cfg, "trans_use_gtrxl_gate", True)),
        )
        trans_dropout = float(getattr(cfg, "trans_dropout", 0.1))

        actor = TransformerGaussianActor(
            **common, dropout=trans_dropout,
        ).to(self.device)

        critic_kwargs = dict(
            **common,
            dropout       = critic_dropout,
            share_encoder = bool(getattr(cfg, "trans_share_critic_encoder", True)),
        )
        critic        = TransformerTwinCritic(**critic_kwargs).to(self.device)
        critic_target = TransformerTwinCritic(**critic_kwargs).to(self.device)
        return actor, critic, critic_target

    # ── 행동 선택 ──────────────────────────────────────

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """
        obs: (obs_dim,) numpy array
        deterministic: True → 평가 시 결정론적 행동 (탐험 없음)
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        if deterministic:
            action = self.actor.get_deterministic_action(obs_t)
        else:
            action = self.actor.get_action(obs_t)

        return action.flatten()

    # ── 학습 스텝 ──────────────────────────────────────

    def train_step(self, buffer) -> Dict[str, float]:
        """
        리플레이 버퍼에서 배치를 샘플링하고 SAC 업데이트 수행.

        지원 기능:
          • Uniform Replay / PER (IS 가중) / PER+LAP (Huber + max-priority)
          • N-step return (γ^n_eff 이미 버퍼에서 누적)
          • DroQ: UTD ratio G ── critic 만 G 회, actor / α 는 1 회 (논문 표준)
          • CAPS: actor 손실에 temporal/spatial smoothness 항 추가
          • Primacy Bias Reset: cfg.reset_interval 마다 critic head 재초기화
        """
        G = max(1, int(getattr(self.cfg, "utd_ratio", 1)) if getattr(self.cfg, "use_droq", False) else 1)

        last_critic_loss = 0.0
        last_obs = last_next_obs = None  # 마지막 배치를 actor 학습에 재사용
        last_buf_actions = None           # BC Reg 용 buffer actions

        for g in range(G):
            obs, actions, rewards, next_obs, dones, gammas, is_w, indices = \
                self._sample_batch(buffer)

            alpha = self.log_alpha.exp().detach()
            critic_loss, td_errors = self._update_critic(
                obs, actions, rewards, next_obs, dones, gammas, alpha, is_w,
            )
            last_critic_loss = critic_loss

            if indices is not None:
                buffer.update_priorities(indices, td_errors)

            # Soft target update — DroQ 는 매 critic 업데이트마다 수행 (논문 권장)
            self._total_updates += 1
            if self._total_updates % self.cfg.target_update_interval == 0:
                self._soft_update(self.critic_target, self.critic, self.cfg.tau)

            last_obs = obs
            last_next_obs = next_obs
            last_buf_actions = actions

        # ── Actor (CAPS + BC Reg 포함) — UTD 와 무관하게 1회만
        alpha = self.log_alpha.exp().detach()
        actor_loss = self._update_actor(last_obs, last_next_obs, alpha,
                                         buffer_actions=last_buf_actions)

        # ── α 자동 조정 — 1회만
        alpha_loss = 0.0
        if self.cfg.auto_alpha:
            alpha_loss = self._update_alpha(last_obs)

        # ── Primacy Bias Reset
        reset_iv = int(getattr(self.cfg, "reset_interval", 0))
        if reset_iv > 0 and self._total_updates > 0 and self._total_updates % reset_iv == 0:
            self._reset_critic_head()

        return {
            "critic_loss": last_critic_loss,
            "actor_loss":  actor_loss,
            "alpha_loss":  alpha_loss,
            "alpha":       float(self.log_alpha.exp().item()),
        }

    # ── 배치 샘플링 헬퍼 ──────────────────────────────

    def _sample_batch(self, buffer):
        """Uniform / PER 양쪽을 동일 인터페이스로 반환."""
        if getattr(buffer, "is_per", False):
            obs_n, act_n, rew_n, nxt_n, don_n, gam_n, iw_n, indices = buffer.sample(self.cfg.batch_size)
            obs      = torch.FloatTensor(obs_n).to(self.device)
            actions  = torch.FloatTensor(act_n).to(self.device)
            rewards  = torch.FloatTensor(rew_n).to(self.device)
            next_obs = torch.FloatTensor(nxt_n).to(self.device)
            dones    = torch.FloatTensor(don_n).to(self.device)
            gammas   = torch.FloatTensor(gam_n).to(self.device)
            is_w     = torch.FloatTensor(iw_n).to(self.device)
        else:
            batch = buffer.sample(self.cfg.batch_size)
            obs, actions, rewards, next_obs, dones = [
                torch.FloatTensor(b).to(self.device) for b in batch
            ]
            gammas   = torch.full_like(rewards, self.cfg.gamma)
            is_w     = None
            indices  = None
        return obs, actions, rewards, next_obs, dones, gammas, is_w, indices

    # ── ① Critic 손실 ─────────────────────────────────

    def _update_critic(self, obs, actions, rewards, next_obs, dones, gammas, alpha, is_w):
        """
        Soft Bellman backup with entropy regularization + N-step + IS weighting:
            y = R_n + γ^n_eff · (1-d) · [min_Q(s_{t+n}, ã) - α·log π(ã|s_{t+n})]

        손실:
          • 기본 PER (Schaul 2015) → 가중 MSE
          • LAP   (Fujimoto 2020)  → 가중 Huber  (priority 폭주 보정)

        S4RL state augmentation (Sinha & Garg, CoRL 2022 §4):
          critic 입력 obs / next_obs 에 작은 가우시안 노이즈 → OOD 강건성.
          actor / α update 에는 적용하지 않음 (clean obs 로 정책 평가).

        rewards  : R_n (n-step 누적 보상)
        gammas   : γ^n_eff (에피소드 종료 시 0 으로 cut 된 effective discount)
        is_w     : (B,1) IS weight, PER 일 때만 적용

        Returns
        -------
        loss_value : float
        td_errors  : np.ndarray (B,) — priority 갱신용
        """
        # ── S4RL: critic 입력에만 노이즈 (Sinha & Garg 2022)
        if getattr(self.cfg, "use_state_aug", False):
            sigma = float(getattr(self.cfg, "state_aug_sigma", 0.0))
            if sigma > 0.0:
                obs      = obs      + torch.randn_like(obs)      * sigma
                next_obs = next_obs + torch.randn_like(next_obs) * sigma

        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_obs)
            # DroQ: target critic 의 dropout 도 활성 (train mode) — 별도 처리 불필요
            q_next = self.critic_target.q_min(next_obs, next_action)
            target_q = rewards + gammas * (1.0 - dones) * (
                q_next - alpha * next_log_prob
            )

        q1, q2 = self.critic(obs, actions)
        td1 = q1 - target_q
        td2 = q2 - target_q

        use_lap   = bool(getattr(self.cfg, "use_lap", False))
        huber_d   = float(getattr(self.cfg, "huber_delta", 1.0))

        if use_lap:
            # Huber loss (element-wise) — LAP 등가 보정
            loss1 = F.huber_loss(q1, target_q, reduction="none", delta=huber_d)
            loss2 = F.huber_loss(q2, target_q, reduction="none", delta=huber_d)
        else:
            loss1 = td1 ** 2
            loss2 = td2 ** 2

        if is_w is not None:
            critic_loss = (is_w * loss1).mean() + (is_w * loss2).mean()
        else:
            critic_loss = loss1.mean() + loss2.mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.critic_opt.step()

        # priority 는 두 critic 중 보수적인 쪽 (max) 의 |TD|
        with torch.no_grad():
            td_for_prio = torch.max(td1.abs(), td2.abs()).squeeze(-1).detach().cpu().numpy()

        return float(critic_loss.item()), td_for_prio

    # ── ② Actor 손실 ──────────────────────────────────

    def _update_actor(self, obs, next_obs, alpha, buffer_actions=None) -> float:
        """
        정책 개선 (CAPS + BC Regularization 포함):
            L_π = E[α·log π(a|s) - min_Q(s,a)]
                + λ_T · ‖μ(s_t) - μ(s_{t+1})‖²      ── CAPS temporal
                + λ_S · ‖μ(s)   - μ(s + ε)‖²       ── CAPS spatial
                + λ_BC · ‖μ(s)  - a_buffer‖²         ── TD3+BC (Fujimoto 2021)

        CAPS (Mysore et al., NeurIPS 2021):
          - 시점 간 행동 변화 정규화 → 과매매 억제.

        BC Regularization (Fujimoto & Gu, NeurIPS 2021 — TD3+BC):
          - 학습 데이터의 행동 분포에 정책을 묶음
          - OOD action 추출 방지 → OOS 일반화 향상
          - λ_BC = α / (E[|Q|]) · 원래_λ_BC  (논문 §3.2 normalization)

        Critic 은 고정 (actor gradient 만).
        """
        action, log_prob = self.actor(obs)
        q_val = self.critic.q_min(obs, action)

        actor_loss = (alpha * log_prob - q_val).mean()

        lam_t = float(getattr(self.cfg, "caps_lambda_t", 0.0))
        lam_s = float(getattr(self.cfg, "caps_lambda_s", 0.0))
        sigma = float(getattr(self.cfg, "caps_spatial_sigma", 0.05))

        if lam_t > 0.0 and next_obs is not None:
            mu_t  = self.actor.deterministic_mu(obs)
            mu_t1 = self.actor.deterministic_mu(next_obs)
            caps_t_loss = F.mse_loss(mu_t, mu_t1)
            actor_loss = actor_loss + lam_t * caps_t_loss

        if lam_s > 0.0:
            mu_clean = self.actor.deterministic_mu(obs)
            obs_noisy = obs + torch.randn_like(obs) * sigma
            mu_noisy = self.actor.deterministic_mu(obs_noisy)
            caps_s_loss = F.mse_loss(mu_clean, mu_noisy)
            actor_loss = actor_loss + lam_s * caps_s_loss

        # ── TD3+BC: Behavioral Cloning Regularization (Fujimoto-Gu 2021)
        use_bc = bool(getattr(self.cfg, "use_bc_regularization", False))
        if use_bc and buffer_actions is not None:
            lam_bc   = float(getattr(self.cfg, "bc_lambda", 2.5))
            norm_q   = bool(getattr(self.cfg, "bc_norm_q", True))

            mu_pred = self.actor.deterministic_mu(obs)
            bc_loss = F.mse_loss(mu_pred, buffer_actions)

            if norm_q:
                # 논문 §3.2: λ' = λ / E[|Q|]  ← 두 항의 scale 균일화
                q_scale = q_val.abs().mean().detach() + 1e-6
                effective_lam = lam_bc / q_scale.item()
            else:
                effective_lam = lam_bc

            actor_loss = actor_loss + effective_lam * bc_loss

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.actor_opt.step()

        return float(actor_loss.item())

    # ── ③ α 자동 조정 ─────────────────────────────────

    def _update_alpha(self, obs) -> float:
        """
        엔트로피 온도 자동 조정 (SAC v2):
            L_α = E[-α · (log π(a|s) + H̄)]
        현재 정책 엔트로피가 목표보다 낮으면 α 증가 (더 탐험)
        목표보다 높으면 α 감소 (더 집중)
        """
        with torch.no_grad():
            _, log_prob = self.actor(obs)

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy)).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        return float(alpha_loss.item())

    # ── Soft / Hard Update ─────────────────────────────

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """θ_target ← τ·θ + (1-τ)·θ_target"""
        for t_p, s_p in zip(target.parameters(), source.parameters()):
            t_p.data.copy_(tau * s_p.data + (1 - tau) * t_p.data)

    def _hard_update(self, target: nn.Module, source: nn.Module):
        target.load_state_dict(source.state_dict())

    # ── Primacy Bias Reset (Nikishin ICML 2022 / BBF Schwarzer NeurIPS 2023) ─

    def _reset_critic_head(self):
        """
        주기적 모듈 리셋. 학습 초반의 잘못된 표현이 끝까지 영향을 미치는
        primacy bias 를 끊는다.

        모드 (cfg.reset_mode):
          "head"           : critic 의 마지막 Linear 만 재초기화 (Nikishin 2022 v1)
          "full_critic"    : critic 전체 재초기화
          "shrink_perturb" : θ ← α·θ + ε·N(0, std)   (Ash 2020) — 지식 일부 보존
                             가장 안전한 디폴트, full reset 보다 회복 빠름

        cfg.reset_actor=True 면 위 모드를 actor 에도 동일하게 적용.
        cfg.reset_optimizer=True 면 optimizer state 도 초기화.

        BBF (Schwarzer 2023) 의 핵심 권고: actor 와 critic 을 함께 리셋해야
        actor 가 stale critic 위에 잘못 학습되어 있던 패턴까지 끊긴다.
        Critic 만 리셋 시 actor 가 곧바로 같은 잘못된 정책으로 critic 을 재훈련.
        """
        mode    = str(getattr(self.cfg, "reset_mode", "shrink_perturb"))
        shrink  = float(getattr(self.cfg, "reset_shrink_factor", 0.5))
        noise_s = float(getattr(self.cfg, "reset_perturb_sigma", 0.02))
        do_actor = bool(getattr(self.cfg, "reset_actor", True))
        do_opt   = bool(getattr(self.cfg, "reset_optimizer", True))

        def _reinit_last_linear(seq: nn.Sequential):
            for m in reversed(list(seq.modules())):
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=float(np.sqrt(2)))
                    nn.init.zeros_(m.bias)
                    return

        def _reinit_all_linear(module: nn.Module):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=float(np.sqrt(2)))
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        def _shrink_perturb(module: nn.Module):
            """θ ← shrink · θ + noise_std · N(0, 1)  (Ash 2020)"""
            with torch.no_grad():
                for p in module.parameters():
                    if p.requires_grad:
                        p.mul_(shrink).add_(noise_s * torch.randn_like(p))

        # ── Critic 적용 — MLP / Transformer 백본 양쪽 호환
        if self.use_transformer:
            critic_q_modules = (self.critic.q1_head, self.critic.q2_head)
        else:
            critic_q_modules = (self.critic.q1, self.critic.q2)

        for net in critic_q_modules:
            if mode == "head":
                _reinit_last_linear(net)
            elif mode == "full_critic":
                _reinit_all_linear(net)
            else:  # "shrink_perturb"
                _shrink_perturb(net)
        # transformer + shrink_perturb 면 encoder 도 함께 perturb (한 번 더)
        if self.use_transformer and mode == "shrink_perturb":
            if getattr(self.critic, "_share", False):
                _shrink_perturb(self.critic.encoder)
            else:
                _shrink_perturb(self.critic.encoder1)
                _shrink_perturb(self.critic.encoder2)
        self._hard_update(self.critic_target, self.critic)

        # ── Actor 적용 (BBF 권고)
        if do_actor:
            if mode == "head":
                # actor 의 mu_head + log_std_head 만 재초기화
                nn.init.orthogonal_(self.actor.mu_head.weight, gain=0.01)
                nn.init.zeros_(self.actor.mu_head.bias)
                nn.init.orthogonal_(self.actor.log_std_head.weight, gain=0.01)
                nn.init.zeros_(self.actor.log_std_head.bias)
            elif mode == "full_critic":
                # 이름이 critic-only 처럼 보여도 actor 도 동일 강도로 처리
                _reinit_all_linear(self.actor)
                nn.init.orthogonal_(self.actor.mu_head.weight, gain=0.01)
                nn.init.zeros_(self.actor.mu_head.bias)
            else:  # shrink_perturb
                _shrink_perturb(self.actor)

        # ── Optimizer 재생성 (AdamW moment 초기화 → 학습률 자유)
        if do_opt:
            wd = float(getattr(self.cfg, "weight_decay", 0.0))
            self.critic_opt = torch.optim.AdamW(
                self.critic.parameters(), lr=self.cfg.critic_lr, weight_decay=wd,
            )
            if do_actor:
                self.actor_opt = torch.optim.AdamW(
                    self.actor.parameters(), lr=self.cfg.actor_lr, weight_decay=wd,
                )

        logger.info(
            f"[Primacy Reset] mode={mode} | "
            f"critic={'✓'} actor={'✓' if do_actor else '✗'} opt={'✓' if do_opt else '✗'} | "
            f"update={self._total_updates}"
            + (f" | shrink={shrink} σ={noise_s}" if mode == "shrink_perturb" else "")
        )

    # ── 저장 / 로드 ────────────────────────────────────

    @staticmethod
    def peek_checkpoint(name: str = "sac_agent") -> Dict:
        """체크포인트의 metadata 만 빠르게 확인 (use_transformer, obs_meta 등)."""
        path = CKPT_DIR / f"{name}.pt"
        ckpt = torch.load(path, map_location="cpu")
        return {
            "use_transformer": bool(ckpt.get("use_transformer", False)),
            "obs_meta":        ckpt.get("obs_meta"),
            "total_updates":   ckpt.get("total_updates", 0),
        }

    def save(self, name: str = "sac_agent") -> Path:
        path = CKPT_DIR / f"{name}.pt"
        torch.save({
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt":     self.actor_opt.state_dict(),
            "critic_opt":    self.critic_opt.state_dict(),
            "log_alpha":     self.log_alpha.detach().cpu(),
            "total_updates": self._total_updates,
            # 백본 종류 + 차원 metadata — load 시 mismatch 조기 발견
            "use_transformer": self.use_transformer,
            "obs_meta":        self.obs_meta,
        }, path)
        logger.info(f"체크포인트 저장: {path}")
        return path

    def load(self, name: str = "sac_agent") -> "SACAgent":
        path = CKPT_DIR / f"{name}.pt"
        ckpt = torch.load(path, map_location=self.device)

        # ── 백본 호환성 사전 검증 (이전 MLP 체크포인트를 transformer 로 로드 시
        # state_dict mismatch 의 거대한 traceback 대신 친절한 안내).
        saved_use_transformer = bool(ckpt.get("use_transformer", False))
        if saved_use_transformer != self.use_transformer:
            cur  = "Transformer" if self.use_transformer else "MLP"
            prev = "Transformer" if saved_use_transformer else "MLP"
            raise RuntimeError(
                f"백본 불일치: 체크포인트 '{name}.pt' 는 {prev} 로 저장됨, "
                f"현재 agent 는 {cur} 로 초기화됨.\n"
                f"현재 실행 경로는 Transformer-SAC 단일 모델만 사용합니다. "
                f"Transformer 체크포인트로 다시 학습하거나 같은 구조의 체크포인트를 지정하세요."
            )

        try:
            self.actor.load_state_dict(ckpt["actor"])
            self.critic.load_state_dict(ckpt["critic"])
            self.critic_target.load_state_dict(ckpt["critic_target"])
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
        except RuntimeError as e:
            raise RuntimeError(
                f"state_dict 로드 실패: {name}.pt — 같은 백본이지만 차원/구조가 다를 수 있음.\n"
                f"  체크포인트 obs_meta={ckpt.get('obs_meta')} vs 현재 obs_meta={self.obs_meta}\n"
                f"원인: hidden_dims / d_model / window_size / feature 수 변경.\n"
                f"원래 오류: {e}"
            )

        self.log_alpha   = ckpt["log_alpha"].to(self.device).requires_grad_(self.cfg.auto_alpha)
        self._total_updates = ckpt.get("total_updates", 0)

        logger.info(f"체크포인트 로드: {path}")
        return self

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())
