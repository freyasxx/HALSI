# A new PPO class that supports skill-level updates, outputs latent z and duration
# Skill-level PPO with diffusion-compatible buffer

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from halsi.rl.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from halsi.rl.utils.mpi_tools import mpi_avg, proc_id, num_procs
from halsi.utils.general_utils import AttrDict
from halsi.rl.agents.ppo_core import discount_cumsum


class SkillPPOBuffer:
    def __init__(self, obs_dim, n_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.n_buf = np.zeros((size, n_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.logp_duration_buf = np.zeros(size, dtype=np.float32)
        self.duration_idx_buf = np.zeros(size, dtype=np.int32)

        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.gamma, self.lam = gamma, lam

    def store(self, obs, n, reward, val, logp, logp_duration, duration_idx):
        ## Store (s₀, z, R) for each skill rollout
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.n_buf[self.ptr] = n
        self.rew_buf[self.ptr] = reward
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.logp_duration_buf[self.ptr] = logp_duration
        self.duration_idx_buf[self.ptr] = duration_idx

        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Finish current path, calculate advantage and return values
        Handle various possible types of last_val
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        path_len = self.ptr - self.path_start_idx
        
        # print(f"Finishing path: {self.path_start_idx} to {self.ptr}")
        
        if self.path_start_idx == self.ptr:
            # Empty path, just return
            self.path_start_idx = self.ptr
            return
        
        # Handle various possible types of last_val
        if isinstance(last_val, torch.Tensor):
            if last_val.numel() == 1:  # Single element
                last_val_scalar = float(last_val.item())
            else:  # Multiple elements, take the first
                last_val_scalar = float(last_val[0].item())
        elif isinstance(last_val, np.ndarray):
            if last_val.size == 1:  # Single element
                last_val_scalar = float(last_val.item())
            else:  # Multiple elements, take the first
                last_val_scalar = float(last_val.flat[0])
        else:
            # Assume scalar
            last_val_scalar = float(last_val)
        
        # print(f"Converted last_val to scalar: {last_val_scalar}")
        
        # Calculate advantage and return for each skill separately
        for i in range(self.path_start_idx, self.ptr):
            # Ensure reward and value are scalars
            if isinstance(self.rew_buf[i], (np.ndarray, torch.Tensor)):
                if hasattr(self.rew_buf[i], 'numel') and self.rew_buf[i].numel() == 1:
                    reward = float(self.rew_buf[i].item())
                elif hasattr(self.rew_buf[i], 'size') and self.rew_buf[i].size == 1:
                    reward = float(self.rew_buf[i].item())
                else:
                    reward = float(self.rew_buf[i].flat[0] if hasattr(self.rew_buf[i], 'flat') else self.rew_buf[i][0])
            else:
                reward = float(self.rew_buf[i])
                
            if isinstance(self.val_buf[i], (np.ndarray, torch.Tensor)):
                if hasattr(self.val_buf[i], 'numel') and self.val_buf[i].numel() == 1:
                    value = float(self.val_buf[i].item())
                elif hasattr(self.val_buf[i], 'size') and self.val_buf[i].size == 1:
                    value = float(self.val_buf[i].item())
                else:
                    value = float(self.val_buf[i].flat[0] if hasattr(self.val_buf[i], 'flat') else self.val_buf[i][0])
            else:
                value = float(self.val_buf[i])
                
            # Calculate advantage and return
            self.adv_buf[i] = reward + self.gamma * last_val_scalar - value
            self.ret_buf[i] = reward + self.gamma * last_val_scalar
        
        # Update path start index
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)

        data = dict(obs=self.obs_buf, n=self.n_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf,
                    logp_duration=self.logp_duration_buf,
                    duration_idx=self.duration_idx_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32 if k != 'duration_idx' else torch.long)
            for k, v in data.items()}


class DurationActorCritic(nn.Module):
    def __init__(self, obs_dim, n_dim, max_duration=10, hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]), activation(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), activation(),
        )
        self.mu = nn.Linear(hidden_sizes[-1], n_dim)
        self.log_std = nn.Parameter(torch.ones(n_dim) * -0.5)
        self.duration_head = nn.Linear(hidden_sizes[-1], max_duration)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)

    def step(self, obs):
        with torch.no_grad():
            h = self.encoder(obs)
            mu = self.mu(h)
            std = torch.exp(self.log_std).clamp(0.2, 1.0)
            eps = torch.randn_like(std)
            n = mu + eps * std
            logp = -0.5 * (((eps) ** 2) + 2 * torch.log(std) + np.log(2 * np.pi)).sum(dim=-1)
            duration_logits = self.duration_head(h)
            v = self.value_head(h).squeeze(-1)
        return n, duration_logits, v, logp, mu, std

    def v(self, obs):
        h = self.encoder(obs)
        return self.value_head(h).squeeze(-1)


class PPOWithDuration:
    def __init__(self, max_duration, hidden_sizes=(64,64),
                 steps_per_epoch=4000, epochs=50, gamma=0.99, lam=0.97, clip_ratio=0.2,
                 pi_lr=1e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80,save_freq=10,
                 target_kl=0.01, max_ep_len=1000, seed=0, obs_dim=25, act_dim=4, n_dim=16):
        self.steps_per_epoch = steps_per_epoch
        setup_pytorch_for_mpi()
        seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.ac = DurationActorCritic(obs_dim, n_dim, max_duration, hidden_sizes)
        sync_params(self.ac)

        local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.buf = SkillPPOBuffer(obs_dim, n_dim, local_steps_per_epoch, gamma, lam)

        self.pi_optimizer = Adam(list(self.ac.mu.parameters()) + list(self.ac.duration_head.parameters()) + [self.ac.log_std], lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.value_head.parameters(), lr=vf_lr)

        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        self.max_ep_len = max_ep_len
        self.epochs = epochs
        self.save_freq = save_freq
        self.max_duration = max_duration

    def update(self):
        data = self.buf.get()
        obs, n, adv, ret, logp_old = data['obs'], data['n'], data['adv'], data['ret'], data['logp']
        logp_old_duration = data['logp_duration']
        duration_idx = data['duration_idx']

        # ---------- Policy update ----------
        for i in range(self.train_pi_iters):
            h = self.ac.encoder(obs)

            # ----- Policy loss for n -----
            mu = self.ac.mu(h)
            std = torch.exp(self.ac.log_std).clamp(0.2, 1.0)
            dist = torch.distributions.Normal(mu, std)
            logp = dist.log_prob(n).sum(dim=-1)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            entropy_n = dist.entropy().mean()
            loss_pi_n = -(torch.min(ratio * adv, clip_adv)).mean()

            # ----- Policy loss for duration -----
            duration_logits = self.ac.duration_head(h)
            duration_dist = torch.distributions.Categorical(logits=duration_logits)
            logp_d = duration_dist.log_prob(duration_idx)
            ratio_d = torch.exp(logp_d - logp_old_duration)
            clip_adv_d = torch.clamp(ratio_d, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            loss_pi_duration = -(torch.min(ratio_d * adv, clip_adv_d)).mean()
            
            ## Add duration_penalty
            duration_penalty = (duration_idx.float() + 1) / self.max_duration 

            # ----- Combine losses -----
            #  0.1 is the weight for duration_penalty, currently optimal
            loss_pi_total = loss_pi_n + loss_pi_duration - 0.1 * duration_penalty.mean()
            kl = (logp_old - logp).mean().item()

            self.pi_optimizer.zero_grad()
            loss_pi_total.backward()

            # ---- Sync gradients: policy network parts ----
            for m in [self.ac.encoder, self.ac.mu, self.ac.duration_head]:
                mpi_avg_grads(m)

            # ---- Sync log_std parameter gradients ----
            if self.ac.log_std.grad is not None:
                avg_log_std_grad = mpi_avg(self.ac.log_std.grad)
                self.ac.log_std.grad.copy_(torch.as_tensor(avg_log_std_grad, dtype=self.ac.log_std.grad.dtype))

            self.pi_optimizer.step()

            if kl > 1.5 * self.target_kl:
                break

        # ---------- Value function update ----------
        for _ in range(self.train_v_iters):
            v = self.ac.v(obs)
            loss_v = ((v - ret) ** 2).mean()

            self.vf_optimizer.zero_grad()
            loss_v.backward()

            # ---- Sync gradients: value network parts ----
            for m in [self.ac.encoder, self.ac.value_head]:
                mpi_avg_grads(m)

            self.vf_optimizer.step()

        return AttrDict(
            LossPi=loss_pi_total.item(),
            LossV=loss_v.item(),
            KL=kl,
            Entropy=entropy_n.item(),
        )

