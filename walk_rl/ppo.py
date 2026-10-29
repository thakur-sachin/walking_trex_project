"""Compact PPO with an asymmetric actor-critic, written for GPU-vectorized envs.

Why in-repo instead of rsl_rl/SB3: no new dependency in the isaac6 env, and the
deployable artifact is simply (obs normalizer + small MLP) so it can be
exported as plain NumPy weights for the Raspberry Pi 4 (no torch needed).

- Actor sees only deployable observations; critic additionally sees privileged
  sim state (base velocity, foot contacts, servo torques), used in training only.
- State-independent learned log-std; env clips actions to [-1, 1].
- Fixed (linearly decayed) learning rate with a per-minibatch KL guard, GAE, clipped
  objective, time-out bootstrapping via V(s_t). An rsl_rl-style adaptive LR was tried
  and removed: at small action std it pins the LR at its floor (docs/WALK_TASK.md).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    def __init__(self, dim: int, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot
        self.var = m2 / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / (self.var + 1e-8).sqrt()).clamp(-self.clip, self.clip)


def mlp(inp: int, hidden: list[int], out: int, out_gain: float) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, np.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.ELU()]
        d = h
    last = nn.Linear(d, out)
    nn.init.orthogonal_(last.weight, out_gain)
    nn.init.zeros_(last.bias)
    return nn.Sequential(*layers, last)


class ActorCritic(nn.Module):
    def __init__(self, actor_dim: int, critic_dim: int, act_dim: int, actor_hidden, critic_hidden, init_log_std: float):
        super().__init__()
        self.actor = mlp(actor_dim, list(actor_hidden), act_dim, 0.01)
        self.critic = mlp(critic_dim, list(critic_hidden), 1, 1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))
        self.actor_norm = RunningNorm(actor_dim)
        self.critic_norm = RunningNorm(critic_dim)

    def dist(self, actor_obs: torch.Tensor) -> torch.distributions.Normal:
        mu = self.actor(self.actor_norm(actor_obs))
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.critic_norm(critic_obs)).squeeze(-1)

    @torch.no_grad()
    def act_deterministic(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.actor_norm(actor_obs))


def export_numpy_policy(ac: ActorCritic, path: Path, meta: dict) -> None:
    """Plain-NumPy export of the deployable actor (normalizer + MLP) for the Pi."""
    arrays = {"norm_mean": ac.actor_norm.mean.cpu().numpy(), "norm_var": ac.actor_norm.var.cpu().numpy(),
              "norm_clip": np.array(ac.actor_norm.clip)}
    k = 0
    for m in ac.actor:
        if isinstance(m, nn.Linear):
            arrays[f"w{k}"] = m.weight.detach().cpu().numpy()
            arrays[f"b{k}"] = m.bias.detach().cpu().numpy()
            k += 1
    arrays["num_layers"] = np.array(k)
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez(path, **arrays)


class PPO:
    def __init__(self, env, cfg, log_dir: Path, device: str):
        self.env, self.cfg, self.device = env, cfg, device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ac = ActorCritic(env.num_actor_obs, env.num_critic_obs, env.num_actions,
                              cfg.actor_hidden, cfg.critic_hidden, cfg.init_log_std).to(device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr)
        self.lr = cfg.lr
        self.iteration = 0

    def learn(self, iterations: int, save_every: int = 50, extra_meta: dict | None = None) -> None:
        env, cfg, ac, dev = self.env, self.cfg, self.ac, self.device
        N, T = env.num_envs, cfg.rollout_steps
        Da, Dc, A = env.num_actor_obs, env.num_critic_obs, env.num_actions
        buf = {
            "oa": torch.zeros(T, N, Da, device=dev), "oc": torch.zeros(T, N, Dc, device=dev),
            "act": torch.zeros(T, N, A, device=dev), "logp": torch.zeros(T, N, device=dev),
            "val": torch.zeros(T, N, device=dev), "rew": torch.zeros(T, N, device=dev),
            "done": torch.zeros(T, N, device=dev), "mu": torch.zeros(T, N, A, device=dev),
        }
        obs = env.reset()
        log_path = self.log_dir / "train_log.jsonl"
        t_start = time.time()
        start_it = self.iteration
        end_it = start_it + iterations
        self.best_ema, self.best_iter = -1e9, -1
        ema, ema_n = None, 0
        decay_to = getattr(cfg, "lr_decay_to", None)
        while self.iteration < end_it:
            if decay_to is not None:
                frac = (self.iteration - start_it) / max(iterations, 1)
                self.lr = cfg.lr + (decay_to - cfg.lr) * frac  # linear decay over this call
                for g in self.opt.param_groups:
                    g["lr"] = self.lr
            t0 = time.time()
            harness_alpha = self._harness_schedule()
            if hasattr(env, "set_harness"):
                env.set_harness(harness_alpha)
            push_scale = self._ramp_schedule(getattr(env.cfg, "push", None))
            if hasattr(env, "set_push_scale"):
                env.set_push_scale(push_scale)
            rew_acc, n_rew = 0.0, 0
            for t in range(T):
                with torch.no_grad():
                    # normalizer is FROZEN during the rollout and updated from the
                    # batch after the policy update: updating it mid-rollout made
                    # the actor see differently-normalized inputs at update time,
                    # producing a spurious KL.
                    d = ac.dist(obs["actor"])
                    act = d.sample()
                    logp = d.log_prob(act).sum(-1)
                    val = ac.value(obs["critic"])
                buf["oa"][t], buf["oc"][t] = obs["actor"], obs["critic"]
                buf["act"][t], buf["logp"][t], buf["val"][t], buf["mu"][t] = act, logp, val, d.mean
                obs, rew, done, extras = env.step(act)
                # time-out bootstrapping (value of the pre-step state as proxy)
                rew = rew + cfg.gamma * val * extras["time_outs"].float()
                buf["rew"][t], buf["done"][t] = rew, done.float()
                rew_acc += rew.mean().item()
                n_rew += 1
            t_collect = time.time() - t0

            with torch.no_grad():
                last_val = ac.value(obs["critic"])
                adv = torch.zeros(T, N, device=dev)
                gae = torch.zeros(N, device=dev)
                for t in reversed(range(T)):
                    nxt = last_val if t == T - 1 else buf["val"][t + 1]
                    nonterm = 1.0 - buf["done"][t]
                    delta = buf["rew"][t] + cfg.gamma * nxt * nonterm - buf["val"][t]
                    gae = delta + cfg.gamma * cfg.lam * nonterm * gae
                    adv[t] = gae
                ret = adv + buf["val"]

            stats = self._update(buf, adv, ret)
            with torch.no_grad():
                ac.actor_norm.update(buf["oa"].reshape(-1, Da))
                ac.critic_norm.update(buf["oc"].reshape(-1, Dc))
            self.iteration += 1
            ep = env.pop_episode_stats()
            rec = {"iter": self.iteration, "lr": self.lr, "mean_step_reward": rew_acc / n_rew,
                   "samples": self.iteration * N * T, "t_collect": t_collect,
                   "t_total": time.time() - t0, "elapsed_min": (time.time() - t_start) / 60,
                   "action_std": ac.log_std.exp().mean().item(), "harness_alpha": harness_alpha, "push_scale": push_scale, **stats, **ep}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if self.iteration % 5 == 0 or self.iteration == 1:
                print(f"[ppo] it {self.iteration:5d} | rew/step {rec['mean_step_reward']:+.3f} | "
                      f"ep_len {ep.get('episode_length_s', float('nan')):5.2f}s | dist {ep.get('distance_x_m', float('nan')):+.2f}m | "
                      f"term {ep.get('terminated_frac', float('nan')):.2f} | kl {stats['kl']:.4f} | lr {self.lr:.1e} | "
                      f"std {rec['action_std']:.2f} | {rec['t_total']:.1f}s/it", flush=True)
            # smoothed reward -> keep the best policy: v3's best weights (~it 750) were
            # overwritten by a later collapse because only the latest was kept.
            ema = rec["mean_step_reward"] if ema is None else 0.8 * ema + 0.2 * rec["mean_step_reward"]
            # Only compare policies trained under the final conditions: the harness inflates
            # reward and pushes deflate it, so v8c's "best" was a harness-assisted policy (it 250).
            curricula_done = harness_alpha == 0.0 and push_scale >= self._ramp_schedule_final()
            ema_n = ema_n + 1 if curricula_done else 0
            if not curricula_done:
                ema = None
            # 20 iterations of smoothing under final conditions before a policy can be "best"
            if ema_n >= 20 and self.iteration - start_it >= 30 and ema > self.best_ema:
                self.best_ema, self.best_iter = ema, self.iteration
                self.save(extra_meta, tag="best")
            if self.iteration % save_every == 0:
                self.save(extra_meta)
                self.save(extra_meta, tag=f"ckpt_{self.iteration:04d}")
        self.save(extra_meta)
        if self.best_iter >= 0:  # best tracking only starts after a 30-iteration warm-up
            print(f"[ppo] done. best smoothed reward {self.best_ema:+.3f} at iteration {self.best_iter} "
                  f"(saved as best.pt / best_numpy.npz)", flush=True)
        else:
            print("[ppo] done.", flush=True)

    def _harness_schedule(self) -> float:
        """Assist strength for this iteration: 1 until start_iter, linear to 0 at end_iter."""
        h = getattr(self.env.cfg, "harness", None)
        if h is None or not h.enabled:
            return 0.0
        it = self.iteration
        if it < h.start_iter:
            return 1.0
        if it >= h.end_iter:
            return 0.0
        return 1.0 - (it - h.start_iter) / float(h.end_iter - h.start_iter)

    def _ramp_schedule_final(self) -> float:
        c = getattr(self.env.cfg, "push", None)
        return 1.0 if (c is not None and c.enabled) else 0.0

    def _ramp_schedule(self, c) -> float:
        """0 until c.start_iter, linear to 1 at c.end_iter (push disturbances: start after the
        harness has gone, so the policy first learns to walk unaided)."""
        if c is None or not c.enabled:
            return 0.0
        it = self.iteration
        if it < c.start_iter:
            return 0.0
        if it >= c.end_iter:
            return 1.0
        return (it - c.start_iter) / float(c.end_iter - c.start_iter)

    def _update(self, buf, adv, ret) -> dict:
        cfg, ac = self.cfg, self.ac
        T, N = adv.shape
        flat = lambda x: x.reshape(T * N, *x.shape[2:])
        oa, oc, act, logp_old, val_old = flat(buf["oa"]), flat(buf["oc"]), flat(buf["act"]), flat(buf["logp"]), flat(buf["val"])
        mu_old, adv_f, ret_f = flat(buf["mu"]), flat(adv), flat(ret)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        n = T * N
        mb = n // cfg.minibatches
        sums = {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0, "kl": 0.0}
        old_std = ac.log_std.exp().detach().clone()
        count = 0
        stop = False
        for _ in range(cfg.epochs):
            if stop:
                break
            epoch_kl = []
            perm = torch.randperm(n, device=self.device)
            for i in range(cfg.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                d = ac.dist(oa[idx])
                logp = d.log_prob(act[idx]).sum(-1)
                ratio = (logp - logp_old[idx]).exp()
                pi_loss = -torch.min(ratio * adv_f[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv_f[idx]).mean()
                v = ac.value(oc[idx])
                v_clip = val_old[idx] + (v - val_old[idx]).clamp(-cfg.clip, cfg.clip)
                v_loss = torch.max((v - ret_f[idx]) ** 2, (v_clip - ret_f[idx]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pi_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent
                with torch.no_grad():
                    # KL(old || new) for diagonal Gaussians, old = policy that collected the data
                    kl = (torch.log(d.stddev / old_std) + (old_std ** 2 + (mu_old[idx] - d.mean) ** 2) / (2 * d.stddev ** 2) - 0.5).sum(-1).mean()
                if kl.item() > 2.0 * cfg.desired_kl:
                    # v3 collapsed when a whole epoch at lr=1e-3 overshot (KL 0.24, 12x target):
                    # check per minibatch, BEFORE stepping, and stop updating this iteration.
                    stop = True
                    break
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                self.opt.step()
                sums["loss_pi"] += pi_loss.item(); sums["loss_v"] += v_loss.item()
                sums["entropy"] += ent.item(); sums["kl"] += kl.item()
                epoch_kl.append(kl.item())
                count += 1
            # fixed LR: bound the update by stopping epochs early once the policy has moved enough
            if epoch_kl and float(np.mean(epoch_kl)) > 1.5 * cfg.desired_kl:
                break
        sums["epochs_run"] = float(count) / cfg.minibatches
        n = max(count, 1)
        return {k: (v / n if k != "epochs_run" else v) for k, v in sums.items()}

    def save(self, extra_meta: dict | None = None, tag: str | None = None) -> None:
        """tag=None -> checkpoint.pt (latest); 'best' -> best.pt; 'ckpt_0750' -> checkpoints/ckpt_0750.pt"""
        meta = {"iteration": self.iteration, **(extra_meta or {})}
        if tag is None:
            ck_path, np_path = self.log_dir / "checkpoint.pt", self.log_dir / "policy_numpy.npz"
        elif tag == "best":
            ck_path, np_path = self.log_dir / "best.pt", self.log_dir / "best_numpy.npz"
        else:
            d = self.log_dir / "checkpoints"
            d.mkdir(exist_ok=True)
            ck_path, np_path = d / f"{tag}.pt", d / f"{tag}_numpy.npz"
        torch.save({"model": self.ac.state_dict(), "opt": self.opt.state_dict(), "lr": self.lr, "meta": meta}, ck_path)
        export_numpy_policy(self.ac, np_path, meta)

    def load(self, path: Path, reset_lr: bool = False) -> None:
        ck = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.lr = self.cfg.lr if reset_lr else ck["lr"]
        for g in self.opt.param_groups:
            g["lr"] = self.lr
        self.iteration = ck["meta"]["iteration"]
