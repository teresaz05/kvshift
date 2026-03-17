"""implements bc"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset



class BehaviorCloningNet(nn.Module):
    def __init__(self, a: int, b: int, c: int) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.Linear(a, b), nn.ReLU(), nn.Linear(b, c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


def nxt(s: list[str]) -> list[int]:
    d: dict[str, int] = {}
    out = [10**9] * len(s)
    for i in range(len(s) - 1, -1, -1):
        out[i] = d.get(s[i], 10**9)
        d[s[i]] = i
    return out


def build(df: pd.DataFrame, b: int, k: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    xs = []
    ys = []
    meta = []
    for tid, g in df.groupby("trace_id"):
        g = g.sort_values("step").reset_index(drop=True)
        s = g["token"].tolist()
        p = g["phase"].tolist()
        t = g["task"].tolist()
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
            yy = []
            for j, q in enumerate(cands):
                idx = max(m for m in range(i) if s[m] == q)
                age = i - last.get(q, i)
                freq = cnt[q]
                yy.append((nu[idx] - i) if nu[idx] < 10**9 else 10**9)
                xx.extend([
                    age / max(1, b),
                    min(freq, 10) / 10.0,
                    float(z[idx]),
                    float(p[idx] == "retrieval"),
                    float(p[idx] == "local"),
                    float(p[idx] == "summary"),
                ])
            while len(cands) < k:
                cands.append("<PAD>")
                yy.append(-1)
                xx.extend([0.0] * 6)
            good = [m for m, v in enumerate(yy) if v >= 0]
            if not good:
                continue
            y = max(good, key=lambda m: yy[m])
            xs.append(xx)
            ys.append(y)
            meta.append({"trace_id": tid, "step": i, "phase": t[i], "teacher": "restricted_oracle"})
            dead = cands[y]
            if dead in cache:
                cache.remove(dead)
            cache.append(w)
            last[w] = i
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), meta


def eval_bc(m: BehaviorCloningNet, x: np.ndarray, y: np.ndarray) -> dict:
    with torch.no_grad():
        p = m(torch.from_numpy(x)).argmax(dim=1).numpy()
    acc = float((p == y).mean()) if len(y) else 0.0
    return {"teacher_match": acc, "n_examples": int(len(y))}


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "bc.pt")
    a.add_argument("--cache", type=int, default=16)
    a.add_argument("--k", type=int, default=8)
    a.add_argument("--h", type=int, default=64)
    a.add_argument("--epochs", type=int, default=12)
    a.add_argument("--lr", type=float, default=3e-4)
    r = a.parse_args()
    d = pd.read_csv(r.data)
    x, y, meta = build(d, r.cache, r.k)
    m = BehaviorCloningNet(6 * r.k, r.h, r.k)
    o = torch.optim.Adam(m.parameters(), lr=r.lr)
    c = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    hist = []
    for e in range(r.epochs):
        s = 0.0
        n = 0
        for u, v in dl:
            o.zero_grad()
            q = m(u)
            loss = c(q, v)
            loss.backward()
            o.step()
            s += float(loss.item()) * len(u)
            n += len(u)
        hist.append({"epoch": e, "loss": s / max(1, n)})
    res = eval_bc(m, x, y)
    r.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": m.state_dict(), "k": r.k, "cache": r.cache, "h": r.h}, r.out)
    Path(str(r.out) + ".train.json").write_text(json.dumps(hist, indent=2))
    Path(str(r.out) + ".labels.json").write_text(json.dumps(meta[:200], indent=2))
    Path(str(r.out) + ".eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
