"""largely an evaluation file: contains workload screening, static baselines, learned-methods comparison"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_bc import BehaviorCloningNet
from train_ppo import PPOPolicy
from run_kvshift import PhaseDetector, PhaseExpert


def nxt(s: list[str]) -> list[int]:
    d: dict[str, int] = {}
    out = [10**9] * len(s)
    for i in range(len(s) - 1, -1, -1):
        out[i] = d.get(s[i], 10**9)
        d[s[i]] = i
    return out


def ex(df: pd.DataFrame, b: int, k: int) -> list[dict]:
    out = []
    for tid, g in df.groupby("trace_id"):
        g = g.sort_values("step").reset_index(drop=True)
        s = g["token"].tolist()
        p = g["phase"].tolist()
        task = g["task"].tolist()
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
            info = []
            for q in cands:
                idx = max(m for m in range(i) if s[m] == q)
                age = i - last.get(q, i)
                freq = cnt[q]
                gap = (nu[idx] - i) if nu[idx] < 10**9 else 10**9
                row = [
                    age / max(1, b),
                    min(freq, 10) / 10.0,
                    float(z[idx]),
                    float(p[idx] == "retrieval"),
                    float(p[idx] == "local"),
                    float(p[idx] == "summary"),
                ]
                xx.extend(row)
                info.append({"gap": gap, "gold": z[idx], "age": age, "freq": freq})
            while len(cands) < k:
                cands.append("<PAD>")
                xx.extend([0.0] * 6)
                info.append({"gap": 10**9, "gold": 0, "age": 0, "freq": 0})
            out.append({
                "trace_id": tid,
                "step": i,
                "shift": int(g.loc[i, "shift"]),
                "phase": p[i],
                "task": task[i],
                "x": np.asarray(xx, dtype=np.float32),
                "g": np.asarray([np.mean(xx), np.std(xx), float(p[i] == 'retrieval'), float(p[i] == 'local'), float(p[i] == 'summary')], dtype=np.float32),
                "info": info,
            })
            dead = cands[0]
            if dead in cache:
                cache.remove(dead)
            cache.append(w)
            last[w] = i
    return out


def row_metric(r: dict, a: int) -> dict:
    v = r["info"][a]
    bad = bool(v["gold"] == 1 and v["gap"] < 64)
    soft = bool(v["gap"] < 64)
    return {
        "trace_id": r["trace_id"],
        "step": r["step"],
        "shift": r["shift"],
        "phase": r["phase"],
        "task": r["task"],
        "gap": float(v["gap"] if v["gap"] < 10**9 else 1e6),
        "reward": float(0.0 if v["gap"] >= 64 else -(1.0 / (1.0 + v["gap"]))),
        "hit": float(0.0 if soft else 1.0),
        "weighted_hit": float(0.0 if bad else 0.7 if soft else 1.0),
    }


def pick_base(r: dict, mode: str) -> int:
    vals = r["info"]
    if mode == "LRU":
        return int(np.argmax([v["age"] for v in vals]))
    if mode == "LFU":
        return int(np.argmin([v["freq"] for v in vals]))
    if mode == "Window":
        return int(np.argmax([0.7 * v["age"] - 2.0 * v["gold"] for v in vals]))
    if mode == "Bandit-UCB":
        return int(np.argmax([0.4 * v["age"] - 0.6 * v["freq"] - 2.0 * v["gold"] for v in vals]))
    return int(np.argmax([v["gap"] for v in vals]))


def metrics(rows: list[dict], alpha: float = 0.95, h: int = 64, sustain: int = 4) -> dict:
    a = []
    t = []
    c = []
    for _, g in pd.DataFrame(rows).groupby("trace_id"):
        g = g.reset_index(drop=True)
        shifts = list(g.index[g["shift"] == 1])
        for s in shifts:
            pre = g.iloc[max(0, s - h):s]
            post = g.iloc[s + 1:s + 1 + h]
            if post.empty:
                continue
            mp = float(pre["weighted_hit"].mean()) if not pre.empty else float(post["weighted_hit"].mean())
            mt = alpha * mp
            vals = post["weighted_hit"].to_numpy()
            area = float(np.maximum(0.0, mt - vals).sum() / max(1e-8, h * max(mt, 1e-8)))
            tr = float(h)
            for i in range(max(0, len(vals) - sustain + 1)):
                if np.all(vals[i:i+sustain] >= mt):
                    tr = float(i + 1)
                    break
            a.append(area)
            t.append(tr)
            c.append(float(tr >= h))
    if not a:
        return {"aurc_norm_fair": 1.0, "t_rec_fair": float(h), "censored_frac": 1.0, "weighted_hit_rate": float(pd.DataFrame(rows)["weighted_hit"].mean()) if rows else 0.0}
    return {
        "aurc_norm_fair": float(np.mean(a)),
        "t_rec_fair": float(np.mean(t)),
        "censored_frac": float(np.mean(c)),
        "weighted_hit_rate": float(pd.DataFrame(rows)["weighted_hit"].mean()),
    }


def bc_rows(egs: list[dict], model: Path, k: int) -> list[dict]:
    ck = torch.load(model, map_location="cpu", weights_only=True)
    m = BehaviorCloningNet(6 * k, ck["h"], k)
    m.load_state_dict(ck["state_dict"])
    x = torch.from_numpy(np.stack([r["x"] for r in egs]).astype(np.float32))
    with torch.no_grad():
        a = m(x).argmax(dim=1).numpy()
    return [row_metric(r, int(a[i])) for i, r in enumerate(egs)]


def ppo_rows(egs: list[dict], model: Path, k: int) -> list[dict]:
    ck = torch.load(model, map_location="cpu", weights_only=True)
    m = PPOPolicy(6 * k, ck["h"], k)
    m.load_state_dict(ck["state_dict"])
    x = torch.from_numpy(np.stack([r["x"] for r in egs]).astype(np.float32))
    with torch.no_grad():
        a = m(x)[0].argmax(dim=1).numpy()
    return [row_metric(r, int(a[i])) for i, r in enumerate(egs)]


def kv_rows(egs: list[dict], model: Path, k: int) -> list[dict]:
    ck = torch.load(model, map_location="cpu", weights_only=True)
    d = PhaseDetector(5, 3)
    d.load_state_dict(ck["detector"])
    exs = [PhaseExpert(6 * k, ck["h"], k) for _ in range(3)]
    for m, s in zip(exs, ck["experts"]):
        m.load_state_dict(s)
    gx = torch.from_numpy(np.stack([r["g"] for r in egs]).astype(np.float32))
    xx = torch.from_numpy(np.stack([r["x"] for r in egs]).astype(np.float32))
    with torch.no_grad():
        q = d(gx).argmax(dim=1).numpy()
    out = []
    for i, r in enumerate(egs):
        with torch.no_grad():
            a = int(exs[int(q[i])](xx[i:i+1]).argmax(dim=1).item())
        out.append(row_metric(r, a))
    return out


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("mode", choices=["screen", "baselines", "compare", "gold"])
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--cache", type=int, default=16)
    a.add_argument("--k", type=int, default=8)
    a.add_argument("--bc_model", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "bc.pt")
    a.add_argument("--ppo_model", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "ppo.pt")
    a.add_argument("--kv_model", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "kvshift.pt")
    r = a.parse_args()
    p = r.data or make_data.q(Path(__file__).resolve().parent)
    df = pd.read_csv(p)
    egs = ex(df, r.cache, r.k)
    r.out.parent.mkdir(parents=True, exist_ok=True)
    if r.mode == "screen":
        best = metrics([row_metric(x, pick_base(x, "LFU")) for x in egs])
        oracle = metrics([row_metric(x, pick_base(x, "Oracle-Gated")) for x in egs])
        z = {
            "oracle_headroom": float((best["aurc_norm_fair"] - oracle["aurc_norm_fair"]) / max(1e-8, best["aurc_norm_fair"])),
            "censoring": float(best["censored_frac"]),
            "screened_in": bool(((best["aurc_norm_fair"] - oracle["aurc_norm_fair"]) / max(1e-8, best["aurc_norm_fair"])) >= 0.08 and best["censored_frac"] <= 0.70),
        }
        r.out.write_text(json.dumps(z, indent=2))
        print(json.dumps(z, indent=2))
    elif r.mode == "baselines":
        rows = []
        for m in ["LRU", "LFU", "Window", "Bandit-UCB", "Oracle-Gated"]:
            x = metrics([row_metric(v, pick_base(v, m)) for v in egs])
            x["method"] = m
            rows.append(x)
        out = pd.DataFrame(rows)[["method", "aurc_norm_fair", "t_rec_fair", "censored_frac", "weighted_hit_rate"]]
        out.to_csv(r.out, index=False)
        print(out.to_string(index=False))
    elif r.mode == "compare":
        rows = []
        for m, fn in [("BC", bc_rows), ("PPO", ppo_rows), ("KVShift", kv_rows)]:
            x = metrics(fn(egs, {"BC": r.bc_model, "PPO": r.ppo_model, "KVShift": r.kv_model}[m], r.k))
            x["method"] = m
            rows.append(x)
        out = pd.DataFrame(rows)[["method", "aurc_norm_fair", "t_rec_fair", "censored_frac", "weighted_hit_rate"]]
        out.to_csv(r.out, index=False)
        print(out.to_string(index=False))
    else:
        z = {
            "needle_token_accuracy": float(df[df["gold"] == 1].shape[0] / max(1, df.shape[0])),
            "gold_token_count": int(df["gold"].sum()),
            "all_token_count": int(df.shape[0]),
            "task_breakdown": {k: int(v) for k, v in df.groupby("task")["gold"].sum().to_dict().items()},
        }
        r.out.write_text(json.dumps(z, indent=2))
        print(json.dumps(z, indent=2))


if __name__ == "__main__":
    main()
