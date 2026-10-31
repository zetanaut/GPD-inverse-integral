"""Resumable numerical-domain contractions from exact validated xi families."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import torch
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NLO = ROOT / "GPD_NLO_completion_package"
FG = ROOT / "completion_runs/NLO_final_geometry"
sys.path.insert(0, str(FG / "source"))
sys.path.insert(0, str(ROOT / "completion_runs/NLO_feasible_domain"))
import geometry_validation as gv
from domain import LABELS, PAIRS, spin2_map
from src.ordered_evolution import coupling


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Preserve completed contraction: " + str(path))
    tmp = path.with_suffix(path.suffix + ".partial")
    torch.save(value, tmp)
    tmp.replace(path)


def ordered_map(k0, k1, q2):
    alpha, end, _ = coupling(q2)
    n = len(k0)
    if end == 0:
        return torch.tensor(np.eye(n), dtype=torch.float64), 0
    def rhs(t, vector):
        matrix = vector.reshape(n, n)
        a = 2 * float(alpha(t))
        return (a * (k0 @ matrix) + a * a * (k1 @ matrix)).ravel()
    sol = solve_ivp(rhs, (0., end), np.eye(n).ravel(), method="DOP853",
                    rtol=1e-13, atol=1e-15)
    if not sol.success or not np.isfinite(sol.y).all():
        raise RuntimeError("Ordered matrix evolution failed at " + str(q2))
    return torch.tensor(sol.y[:, -1].reshape(n, n).copy(), dtype=torch.float64), sol.nfev


def build(xi):
    token = hashlib.sha256(float(xi).hex().encode()).hexdigest()[:16]
    folder = HERE / "families" / token
    state = json.loads((folder / "PIPELINE.json").read_text())
    if state["status"] != "FAMILY_VALIDATED_WITH_CONTINUOUS_DIAGNOSTICS":
        raise RuntimeError("Family validation incomplete")
    if state["xi_hex"] != float(xi).hex():
        raise ValueError("Exact xi mismatch")
    q2s = state["Q2"]
    dirs = state["directories"]
    binding = {"xi_hex": state["xi_hex"], "Q2_hex": [float(q).hex() for q in q2s],
               "builder_sha256": sha(__file__),
               "operator_sources": {d: {"KEY_sha256": sha(NLO / d / "KEY.json"),
                                        "operators_sha256": sha(NLO / d / "operators.npz")}
                                    for d in dirs.values()}}
    cache = folder / "CONTRACTIONS.pt"
    if cache.exists():
        prior = torch.load(cache, weights_only=True)
        if prior["binding"] != binding:
            raise RuntimeError("Completed contraction binding changed")
        return {"path": str(cache), "sha256": sha(cache), "cached": True}
    start = time.monotonic(); matrices = {}; bases = {}
    for label in LABELS:
        directory = dirs[label]
        basis, k0, k1, _ = gv.pf.op(directory)
        bases[label] = basis
        partial = folder / f"CONTRACTION_{label}.pt"
        if partial.exists():
            item = torch.load(partial, weights_only=True)
            if item["binding"] != binding or item["label"] != label:
                raise RuntimeError("Partial contraction binding changed")
            matrices[label] = item["data"]
            continue
        nq = len(basis.qids); n = len(k0)
        x, w = basis.outputs(96)
        M = np.zeros((3, n))
        M[0, :nq] = (w * x) @ basis.basis(x, "odd")
        M[1, 2 * nq:] = w @ basis.basis(x, "even")
        M[2, nq:2 * nq] = (w * x) @ basis.basis(x, "odd")
        evolutions = {}; nfevs = {}
        for q2 in q2s:
            evolutions[str(q2)], nfevs[str(q2)] = ordered_map(k0, k1, q2)
        hard = {}; delta = {}
        for q2 in q2s:
            h192 = gv.pf.hard(basis, q2, 192)
            h384 = gv.pf.hard(basis, q2, 384)
            hard[str(q2)] = torch.tensor(np.stack((h192.real, h192.imag)), dtype=torch.float64)
            delta[str(q2)] = torch.tensor(np.stack(((h384-h192).real, (h384-h192).imag)), dtype=torch.float64)
        data = {"nodes": torch.tensor(basis.nodes, dtype=torch.float64),
                "qids": torch.tensor(basis.qids), "gids": torch.tensor(basis.gids),
                "U": evolutions, "nq": nq, "M": torch.tensor(M, dtype=torch.float64),
                "hard": hard, "hard_delta": delta}
        save(partial, {"binding": binding, "label": label, "data": data, "nfev": nfevs})
        matrices[label] = data
        print("CONTRACTION", label, "seconds", time.monotonic()-start, flush=True)
    pairs = []
    for left, right, weight in PAIRS:
        a, b = bases[left], bases[right]
        edges = np.unique(np.r_[a.edges, b.edges])
        z, w = np.polynomial.legendre.leggauss(128)
        x = np.concatenate([(lo+hi)/2+(hi-lo)*z/2 for lo, hi in zip(edges[:-1], edges[1:])])
        ww = np.concatenate([(hi-lo)*w/2 for lo, hi in zip(edges[:-1], edges[1:])])
        pairs.append({"left": left, "right": right, "weight": weight,
                      "w": torch.tensor(ww, dtype=torch.float64),
                      "lq": torch.tensor(a.basis(x, "odd"), dtype=torch.float64),
                      "lg": torch.tensor(a.basis(x, "even"), dtype=torch.float64),
                      "rq": torch.tensor(b.basis(x, "odd"), dtype=torch.float64),
                      "rg": torch.tensor(b.basis(x, "even"), dtype=torch.float64)})
    spin2 = {str(q): torch.tensor(spin2_map(q), dtype=torch.float64) for q in q2s}
    save(cache, {"binding": binding, "point": {"xi": xi, "data": matrices, "pairs": pairs},
                 "spin2": spin2, "seconds": time.monotonic()-start})
    return {"path": str(cache), "sha256": sha(cache), "cached": False,
            "seconds": time.monotonic()-start}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.xi), indent=2), flush=True)
