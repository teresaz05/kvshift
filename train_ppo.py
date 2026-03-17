"""implements ppo"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical



class PPOPolicy(nn.Module):
    def __init__(self, a: int, b: int, c: int) -> None:
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(a, b), nn.ReLU(), nn.Linear(b, c))
        self.v = nn.Sequential(nn.Linear(a, b), nn.ReLU(), nn.Linear(b, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pi(x), self.v(x).squeeze(-1)


def nxt(s: list[str]) -> list[int]:
    d: dict[str, int] = {}
    out = [10**9] * len(s)
    for i in range(len(s) - 1, -1, -1):
        out[i] = d.get(s[i], 10**9)
        d[s[i]] = i
    return out


def build(df: pd.DataFrame, b: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    rs = []
    for _, g in df.groupby("trace_id"):
        g = g.sort_values("step").reset_index(drop=True)
        s = g["token"].tolist()
        p = g["phase"].tolist()
        z = g["gold"].astype(int).tolist()
        nu = nxt(s)
        cache = []
        last = {}
        cnt = defaultdict(int)
        for i, w in enumerate(s):
            cnt[w] += 1
            if w in cache:
                last[w] = i
                continue
            if len(cache) < b:
                cache.append(w)
                last[w] = i
                continue
            cands = sorted(cache, key=lambda q: last.get(q, -1))[:k]
            xx = []
            rr = []
            for q in cands:
                idx = max(m for m in range(i) if s[m] == q)
                age = i - last.get(q, i)
                freq = cnt[q]
                gap = (nu[idx] - i) if nu[idx] < 10**9 else 10**9
                xx.extend([
                    age / max(1, b),
                    min(freq, 10) / 10.0,
                    float(z[idx]),
                    float(p[idx] == "retrieval"),
                    float(p[idx] == "local"),
                    float(p[idx] == "summary"),
                ])
                rr.append(0.0 if gap >= 64 else -(1.0 / (1.0 + gap)))
            while len(cands) < k:
                xx.extend([0.0] * 6)
                rr.append(0.0)
                cands.append("<PAD>")
            xs.append(xx)
            rs.append(rr)
            dead = cands[int(np.argmax(rr))] if any(v != 0 for v in rr) else cands[0]
            if dead in cache:
                cache.remove(dead)
            cache.append(w)
            last[w] = i
    return np.asarray(xs, dtype=np.float32), np.asarray(rs, dtype=np.float32)


def gae(r: np.ndarray, v: np.ndarray, lam: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros_like(r)
    q = np.zeros_like(r)
    u = 0.0
    w = 0.0
    for i in range(len(r) - 1, -1, -1):
        d = r[i] + gamma * w - v[i]
        u = d + gamma * lam * u
        a[i] = u
        q[i] = u + v[i]
        w = v[i]
    return a, q


def eval_ppo(m: PPOPolicy, x: np.ndarray, r: np.ndarray) -> dict:
    with torch.no_grad():
        lg, _ = m(torch.from_numpy(x))
        a = lg.argmax(dim=1).numpy()
    got = r[np.arange(len(r)), a]
    return {"mean_reward": float(got.mean()), "weighted_hit_proxy": float(1.0 + got.mean())}


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "ppo.pt")
    a.add_argument("--cache", type=int, default=16)
    a.add_argument("--k", type=int, default=8)
    a.add_argument("--h", type=int, default=64)
    a.add_argument("--total_timesteps", type=int, default=20000)
    a.add_argument("--n_steps", type=int, default=256)
    a.add_argument("--epochs", type=int, default=6)
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--gamma", type=float, default=0.9995)
    a.add_argument("--lam", type=float, default=0.95)
    a.add_argument("--clip", type=float, default=0.2)
    r = a.parse_args()
    d = pd.read_csv(r.data)
    x, rr = build(d, r.cache, r.k)
    m = PPOPolicy(6 * r.k, r.h, r.k)
    o = torch.optim.Adam(m.parameters(), lr=r.lr)
    hist = []
    t = 0
    while t < r.total_timesteps:
        idx = np.random.choice(len(x), size=min(r.n_steps, len(x)), replace=False)
        xb = torch.from_numpy(x[idx])
        rb = rr[idx]
        with torch.no_grad():
            lg, v = m(xb)
            dist = Categorical(logits=lg)
            ab = dist.sample()
            lp0 = dist.log_prob(ab)
            rew = torch.from_numpy(rb[np.arange(len(rb)), ab.numpy()].astype(np.float32))
        adv, ret = gae(rew.numpy(), v.numpy(), r.lam, r.gamma)
        adv = torch.from_numpy(((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32))
        ret = torch.from_numpy(ret.astype(np.float32))
        for _ in range(r.epochs):
            lg2, v2 = m(xb)
            dist2 = Categorical(logits=lg2)
            lp = dist2.log_prob(ab)
            ratio = torch.exp(lp - lp0)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1.0 - r.clip, 1.0 + r.clip) * adv
            pl = -torch.minimum(s1, s2).mean()
            vl = ((v2 - ret) ** 2).mean()
            el = -dist2.entropy().mean()
            loss = pl + 0.5 * vl + 0.02 * el
            o.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            o.step()
        t += len(idx)
        hist.append({"steps": t, "mean_rollout_reward": float(rew.mean().item()), "value": float(v.mean().item())})
    res = eval_ppo(m, x, rr)
    r.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": m.state_dict(), "k": r.k, "cache": r.cache, "h": r.h}, r.out)
    Path(str(r.out) + ".train.json").write_text(json.dumps(hist, indent=2))
    Path(str(r.out) + ".eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
