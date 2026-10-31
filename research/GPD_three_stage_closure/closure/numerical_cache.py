"""Exact-Q2 numerical contractions built from existing source-keyed generators.

The existing family point/basis comparisons are reused verbatim. Only ordered
evolution and hard rows at a missing scale are added. No generator is rebuilt.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "GPD_ReH_experimental_run"))
import build_experimental_contractions as original


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def family_path(xi):
    token = hashlib.sha256(float(xi).hex().encode()).hexdigest()[:16]
    path = ROOT / "GPD_ReH_experimental_run/families" / token
    state = json.loads((path / "PIPELINE.json").read_text())
    if state["xi_hex"] != float(xi).hex() or state["status"] != "FAMILY_VALIDATED_WITH_CONTINUOUS_DIAGNOSTICS":
        raise ValueError("Exact xi generator binding failed")
    return path, state


def extension_path(xi, q2):
    key = float(xi).hex() + "|" + float(q2).hex()
    return HERE / "numerical_contractions" / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".pt")


def build_one(xi, q2):
    xi, q2 = float(xi), float(q2)
    family, state = family_path(xi)
    base_path = family / "CONTRACTIONS.pt"
    base = torch.load(base_path, weights_only=True)
    if base["binding"]["xi_hex"] != xi.hex():
        raise ValueError("Base contraction coordinate mismatch")
    if str(q2) in base["spin2"]:
        return {"status": "existing", "xi_hex": xi.hex(), "Q2_hex": q2.hex(), "path": str(base_path)}
    out = extension_path(xi, q2)
    binding = {"xi_hex": xi.hex(), "Q2_hex": q2.hex(),
               "base_sha256": digest(base_path),
               "builder_sha256": digest(ROOT / "GPD_ReH_experimental_run/build_experimental_contractions.py"),
               "operator_sources": base["binding"]["operator_sources"]}
    if out.exists():
        data = torch.load(out, weights_only=True)
        if data["binding"] != binding:
            raise RuntimeError("Cached extension binding changed")
        return {"status": "cached", "xi_hex": xi.hex(), "Q2_hex": q2.hex(), "path": str(out), "sha256": digest(out)}
    additions = {}
    for label in base["point"]["data"]:
        directory = state["directories"][label]
        source = base["binding"]["operator_sources"][directory]
        for filename, key in (("KEY.json", "KEY_sha256"), ("operators.npz", "operators_sha256")):
            if digest(ROOT / "GPD_NLO_completion_package" / directory / filename) != source[key]:
                raise ValueError("Operator source changed: " + directory + "/" + filename)
        basis, k0, k1, _ = original.gv.pf.op(directory)
        old = base["point"]["data"][label]
        if len(basis.nodes) != len(old["nodes"]) or not np.array_equal(basis.nodes, old["nodes"].numpy()):
            raise ValueError("Source basis differs from cached point")
        U, nfev = original.ordered_map(k0, k1, q2)
        h192 = original.gv.pf.hard(basis, q2, 192)
        h384 = original.gv.pf.hard(basis, q2, 384)
        additions[label] = {"U": U,
                            "hard": torch.tensor(np.stack((h192.real, h192.imag)), dtype=torch.float64),
                            "hard_delta": torch.tensor(np.stack(((h384-h192).real, (h384-h192).imag)), dtype=torch.float64),
                            "nfev": nfev}
    payload = {"binding": binding, "additions": additions,
               "spin2": torch.tensor(original.spin2_map(q2), dtype=torch.float64)}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pt.partial")
    torch.save(payload, tmp)
    os.replace(tmp, out)
    return {"status": "built", "xi_hex": xi.hex(), "Q2_hex": q2.hex(), "path": str(out), "sha256": digest(out)}


def load_pair(xi, q2):
    xi, q2 = float(xi), float(q2)
    family, _ = family_path(xi)
    base = torch.load(family / "CONTRACTIONS.pt", weights_only=True)
    if str(q2) in base["spin2"]:
        return base["point"], base["spin2"][str(q2)]
    path = extension_path(xi, q2)
    if not path.exists():
        raise FileNotFoundError(f"Missing exact numerical extension {xi.hex()}|{q2.hex()}")
    ext = torch.load(path, weights_only=True)
    if ext["binding"]["xi_hex"] != xi.hex() or ext["binding"]["Q2_hex"] != q2.hex() or ext["binding"]["base_sha256"] != digest(family / "CONTRACTIONS.pt"):
        raise ValueError("Numerical extension binding mismatch")
    point = dict(base["point"])
    point["data"] = {}
    for label, data in base["point"]["data"].items():
        extra = ext["additions"][label]
        item = dict(data)
        for key in ("U", "hard", "hard_delta"):
            item[key] = dict(data[key])
            item[key][str(q2)] = extra[key]
        point["data"][label] = item
    return point, ext["spin2"]
