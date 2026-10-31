"""Export exact validated weak/reference adjoint rows for one completed family."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FG = ROOT / "completion_runs/NLO_final_geometry"
NLO = ROOT / "GPD_NLO_completion_package"
sys.path.insert(0, str(FG / "source"))
import geometry_validation as gv
from gpd_reh.bank import CHANNELS


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def export(xi):
    token = hashlib.sha256(float(xi).hex().encode()).hexdigest()[:16]
    folder = HERE / "families" / token
    state = json.loads((folder / "PIPELINE.json").read_text())
    if state["status"] != "FAMILY_VALIDATED_WITH_CONTINUOUS_DIAGNOSTICS":
        raise RuntimeError("Family diagnostics not complete")
    if state["xi_hex"] != float(xi).hex():
        raise ValueError("Family xi changed")
    records = []
    arrays = {}
    for i, q2 in enumerate(state["Q2"]):
        validations = [r for r in state["validation"] if r["Q2"] == q2]
        if {v["field"] for v in validations} != {"smooth", "Dterm_added"}:
            raise RuntimeError("Both fixed-field validations are required")
        for v in validations:
            if sha(folder / v["file"]) != v["sha256"]:
                raise RuntimeError("Validation changed")
        for method, label in (("weak10", "nominal"), ("strong10", "reference")):
            directory = state["directories"][method]
            b, _, _, key = gv.pf.op(directory)
            if float(b.xi).hex() != state["xi_hex"]:
                raise ValueError("Operator xi changed")
            dense = gv.pf.cached_row(directory, q2)
            if dense.shape != (2, 3, len(b.nodes)) or not np.isfinite(dense).all():
                raise ValueError("Invalid adjoint row")
            prefix = f"r{i}_{method}"
            arrays[prefix + "_x"] = b.nodes.copy()
            arrays[prefix + "_weights"] = dense.transpose(0, 2, 1).copy()
            records.append({
                "method": method, "xi": xi, "Q2": q2, "prefix": prefix,
                "directory": directory,
                "operator_key_sha256": sha(NLO / directory / "KEY.json"),
                "operator_sha256": sha(NLO / directory / "operators.npz"),
                "smooth_validation_sha256": next(v["sha256"] for v in validations if v["field"] == "smooth"),
                "Dterm_validation_sha256": next(v["sha256"] for v in validations if v["field"] == "Dterm_added"),
                "evolution_direction": "backward" if q2 < 2.0 else "forward" if q2 > 2.0 else "identity",
                "hard_order": 192, "adjoint_rtol": 1e-12, "adjoint_atol": 1e-14,
                "basis": key["basis"],
            })
    arrays["metadata_json"] = np.array(json.dumps({
        "physics": "fixed nf3 MSbar NLO", "Q02": 2.0,
        "alpha0": 0.35691594851514724, "running": "two-loop",
        "input_channels": CHANNELS,
        "output_components": ["Re_internal", "Im_internal"],
        "source": {
            "ordered_evolution_sha256": sha(NLO / "src/ordered_evolution.py"),
            "forward_sha256": sha(NLO / "candidate/forward.py"),
            "hard_sha256": sha(NLO / "frozen/python/nlo_hard.py"),
            "geometry_validation_sha256": sha(FG / "source/geometry_validation.py"),
        },
        "rows": records,
    }, sort_keys=True))
    dest = folder / "ACCEPTED_EXPERIMENTAL_ROWS.npz"
    if dest.exists():
        with np.load(dest, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise RuntimeError("Completed row export changed")
    else:
        np.savez_compressed(dest, **arrays)
    from import_validated_rows import import_rows
    imported = import_rows(dest, HERE / "banks")
    return {"family_xi": xi, "pairs": len(state["Q2"]), "export_sha256": sha(dest),
            "bank": imported}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.xi), indent=2), flush=True)
