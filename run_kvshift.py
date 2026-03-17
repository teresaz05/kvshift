"""kvshift implementation"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn



class PhaseDetector(nn.Module):
    def __init__(self, a: int, b: int) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.Linear(a, b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


class PhaseExpert(nn.Module):
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


def seq(df: pd.DataFrame, b: int, k: int) -> list[dict]:
    rows = []
    for tid, g in df.groupby("trace_id"):
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
            xs = []
            pos = []
            for q in cands:
                idx = max(m for m in range(i) if s[m] == q)
                age = i - last.get(q, i)
                freq = cnt[q]
                gap = (nu[idx] - i) if nu[idx] < 10**9 else 10**9
                xs.extend([
                    age / max(1, b),
                    min(freq, 10) / 10.0,
                    float(z[idx]),
                    float(p[idx] == "retrieval"),
                    float(p[idx] == "local"),
                    float(p[idx] == "summary"),
                ])
                pos.append((idx, gap))
            while len(cands) < k:
                xs.extend([0.0] * 6)
                pos.append((-1, 10**9))
                cands.append("<PAD>")
            rows.append({
                "trace_id": tid,
                "step": i,
                "phase": p[i],
                "x": np.asarray(xs, dtype=np.float32),
                "g": np.asarray([np.mean(xs), np.std(xs), float(p[i] == 'retrieval'), float(p[i] == 'local'), float(p[i] == 'summary')], dtype=np.float32),
                "pos": pos,
            })
            cache.remove(cands[0])
            cache.append(w)
            last[w] = i
    return rows


def pick(row: dict, mode: str) -> int:
    pos = row["pos"]
    if mode == "lru":
        good = [i for i, (idx, _) in enumerate(pos) if idx >= 0]
        return good[0] if good else 0
    if mode == "lfu":
        xs = row["x"].reshape(-1, 6)
        return int(np.argmin(xs[:, 1]))
    if mode == "window":
        xs = row["x"].reshape(-1, 6)
        return int(np.argmin(0.6 * xs[:, 0] + 0.4 * xs[:, 2]))
    good = [i for i, (_, gap) in enumerate(pos) if gap < 10**9]
    if not good:
        return 0
    return max(good, key=lambda i: pos[i][1])


def teach(rows: list[dict]) -> dict[str, str]:
    modes = ["lru", "lfu", "window", "oracle"]
    out = {}
    for ph in ["retrieval", "local", "summary"]:
        best = None
        best_v = -1.0
        sub = [r for r in rows if r["phase"] == ph]
        for m in modes:
            ok = 0.0
            for r in sub:
                a = pick(r, m)
                idx, gap = r["pos"][a]
                ok += 0.0 if gap < 64 else 1.0
            v = ok / max(1, len(sub))
            if v > best_v:
                best_v = v
                best = m
        out[ph] = str(best)
    return out


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "kvshift.pt")
    a.add_argument("--cache", type=int, default=16)
    a.add_argument("--k", type=int, default=8)
    a.add_argument("--h", type=int, default=64)
    a.add_argument("--epochs", type=int, default=12)
    a.add_argument("--lr", type=float, default=3e-4)
    r = a.parse_args()
    d = pd.read_csv(r.data)
    rows = seq(d, r.cache, r.k)
    tm = teach(rows)
    det = PhaseDetector(5, 3)
    exp = [PhaseExpert(6 * r.k, r.h, r.k) for _ in range(3)]
    od = torch.optim.Adam(det.parameters(), lr=r.lr)
    oe = torch.optim.Adam([p for m in exp for p in m.parameters()], lr=r.lr)
    ce = nn.CrossEntropyLoss()
    ph_map = {"retrieval": 0, "local": 1, "summary": 2}
    tx = torch.from_numpy(np.stack([r["x"] for r in rows]).astype(np.float32))
    tg = torch.from_numpy(np.stack([r["g"] for r in rows]).astype(np.float32))
    tp = torch.from_numpy(np.asarray([ph_map.get(r["phase"], 0) for r in rows], dtype=np.int64))
    ty = []
    for r0 in rows:
        ph = r0["phase"]
        ty.append(pick(r0, tm.get(ph, "oracle")))
    ty = torch.from_numpy(np.asarray(ty, dtype=np.int64))
    hist = []
    for e in range(r.epochs):
        od.zero_grad()
        dl = ce(det(tg), tp)
        dl.backward()
        od.step()
        oe.zero_grad()
        q = det(tg).argmax(dim=1)
        loss = 0.0
        for i in range(3):
            m = q == i
            if int(m.sum()) == 0:
                continue
            loss = loss + ce(exp[i](tx[m]), ty[m])
        loss.backward()
        oe.step()
        hist.append({"epoch": e, "detector_loss": float(dl.item()), "expert_loss": float(loss.item())})
    with torch.no_grad():
        q = det(tg).argmax(dim=1)
        a1 = []
        for i in range(len(rows)):
            a1.append(int(exp[int(q[i])](tx[i:i+1]).argmax(dim=1).item()))
        a1 = np.asarray(a1)
    score = 0.0
    for i, r0 in enumerate(rows):
        _, gap = r0["pos"][a1[i]]
        score += 0.0 if gap < 64 else 1.0
    res = {
        "weighted_hit_proxy": float(score / max(1, len(rows))),
        "detector_acc": float((q.numpy() == tp.numpy()).mean()),
        "teachers": tm,
        "n_examples": int(len(rows)),
    }
    r.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"detector": det.state_dict(), "experts": [m.state_dict() for m in exp], "k": r.k, "cache": r.cache, "h": r.h, "teachers": tm}, r.out)
    Path(str(r.out) + ".train.json").write_text(json.dumps(hist, indent=2))
    Path(str(r.out) + ".eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
