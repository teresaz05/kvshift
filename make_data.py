"""code used for dataset generation, ABC were filled in with the token traces"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


A = []

B = []

C = []


def t(x: str) -> list[str]:
    return x.replace("\n", " ").split()


def u(x: list[str], n: int) -> list[str]:
    return x[:n] if len(x) > n else x


def m1(x: list[str], q: list[str]) -> set[int]:
    y = set()
    for i, z in enumerate(x):
        if z.lower() in {}:
            y.add(i)
    for i, z in enumerate(x):
        if z.rstrip("?.!,:'\"").lower() in {w.rstrip("?.!,:'\"").lower() for w in q}:
            y.add(i)
    if not y and x:
        y.update(range(min(6, len(x))))
    return y


def m2(x: list[str], q: list[str]) -> set[int]:
    y = set()
    for i, z in enumerate(x):
        if any(k in z for k in []):
            y.add(i)
    for i, z in enumerate(x):
        if z in q:
            y.add(i)
    if not y and x:
        y.update(range(min(6, len(x))))
    return y


def m3(x: list[str], needle: str) -> set[int]:
    y = set()
    n = needle.split()
    for i in range(max(0, len(x) - len(n) + 1)):
        if x[i : i + len(n)] == n:
            y.update(range(i, i + len(n)))
    if not y and x:
        y.update(range(min(6, len(x))))
    return y


def f_books(r: dict, n: int) -> list[dict]:
    a = u(t(r["context"]), n)
    b = u(t(r["query"]), max(12, n // 8))
    c = m1(a, b)
    z = []
    for i, w in enumerate(a):
        z.append({"token": w, "phase": "retrieval", "gold": int(i in c), "task": "retrieval"})
    for w in b:
        z.append({"token": w, "phase": "summary", "gold": int(w in set(a)), "task": "summary"})
    return z


def f_code(r: dict, n: int) -> list[dict]:
    a = u(t(r["context"]), n)
    b = u(t(r["query"]), max(12, n // 8))
    c = m2(a, b)
    z = []
    for i, w in enumerate(a):
        z.append({"token": w, "phase": "retrieval", "gold": int(i in c), "task": "retrieval"})
    tail = a[-max(24, n // 4):]
    for w in tail:
        z.append({"token": w, "phase": "local", "gold": 1, "task": "local"})
    return z


def f_ruler(r: dict, n: int) -> list[dict]:
    a = u(t(r["context"]), n)
    b = u(t(r["query"]), max(12, n // 8))
    c = m3(a, r["needle_span"])
    d = set(r["needle_span"].split())
    z = []
    for i, w in enumerate(a):
        z.append({"token": w, "phase": "retrieval", "gold": int(i in c), "task": "retrieval"})
    for w in b:
        z.append({"token": w, "phase": "summary", "gold": int(w in d), "task": "summary"})
    return z


def g(n_traces: int = 24, phase_len: int = 96, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_traces):
        a = C[int(rng.integers(len(C)))]
        b = B[int(rng.integers(len(B)))]
        c = A[int(rng.integers(len(A)))]
        x = []
        x.extend(f_ruler(a, phase_len))
        x.append({"token": "<SHIFT>", "phase": "shift", "gold": 0, "task": "shift"})
        x.extend(f_code(b, phase_len))
        x.append({"token": "<SHIFT>", "phase": "shift", "gold": 0, "task": "shift"})
        x.extend(f_books(c, phase_len))
        for j, r in enumerate(x):
            rows.append({
                "trace_id": f"trace_{i}",
                "step": j,
                "token": r["token"],
                "phase": r["phase"],
                "task": r["task"],
                "gold": int(r["gold"]),
                "shift": int(r["phase"] == "shift"),
            })
    return pd.DataFrame(rows)


def q(base: Path | None = None) -> Path:
    p = (base or Path(__file__).resolve().parent) / "artifacts" / "benchmark.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        d = g()
        d.to_csv(p, index=False)
    return p


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "benchmark.csv")
    a.add_argument("--n_traces", type=int, default=24)
    a.add_argument("--phase_len", type=int, default=96)
    a.add_argument("--seed", type=int, default=0)
    r = a.parse_args()
    d = g(r.n_traces, r.phase_len, r.seed)
    r.out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(r.out, index=False)
    z = {
        "out": str(r.out),
        "n_traces": int(d.trace_id.nunique()),
        "n_rows": int(len(d)),
        "n_shifts": int(d["shift"].sum()),
        "phase_counts": {k: int(v) for k, v in d.phase.value_counts().to_dict().items()},
        "gold_frac": float(d["gold"].mean()),
    }
    print(json.dumps(z, indent=2))


if __name__ == "__main__":
    main()
