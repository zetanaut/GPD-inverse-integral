"""Diagnostics-first review of at most four saved experimental central candidates."""
from __future__ import annotations

import csv
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ["GPD_REVIEW_ROOT"] = str(ROOT)
sys.path[:0] = [str(ROOT / "completion_runs/NLO_final_geometry/source"),
                str(ROOT / "GPD_direct_error_calibration"),
                str(ROOT / "GPD_NLO_constrained_solver_review"),
                str(ROOT / "GPD_accuracy_cost_study")]
import geometry_validation as gv
import fit_nlo as accepted
from diagnose_fit import diagnose
from experimental_optimizer import SolverProblem, sha, atomic
from experimental_domain import ExperimentalDomain


def checkpoint_with_contract(original, destination, summary):
    original = Path(original)
    cp = torch.load(original, weights_only=True, map_location="cpu")
    contract = {k: summary[k] for k in ("source_sha256", "t_mean", "t_std_population")}
    cp["experiment_contract"] = contract
    if destination.exists():
        old = torch.load(destination, weights_only=True, map_location="cpu")
        if old["experiment_contract"] != contract or not torch.equal(old["flat"], cp["flat"]):
            raise RuntimeError("Completed central checkpoint changed")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(".partial.pt")
        torch.save(cp, tmp); tmp.replace(destination)
    return cp


def full_forward(problem, checkpoint, out):
    out.mkdir(parents=True, exist_ok=True)
    with (HERE / "prepared/observations.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    family_cache = {}
    records = []
    for row in rows:
        xi, q2, t = float(row["xi"]), float(row["Q2_GeV2"]), float(row["t_GeV2"])
        token = hashlib.sha256(xi.hex().encode()).hexdigest()[:16]
        if token not in family_cache:
            family_cache[token] = json.loads((HERE / "families" / token / "PIPELINE.json").read_text())
        family = family_cache[token]
        if family["xi_hex"] != xi.hex() or q2 not in family["Q2"]:
            raise ValueError("Forward diagnostic coordinate mismatch")
        destination = out / (row["row_id"].replace(":", "_") + ".json")
        binding = {"checkpoint_sha256": sha(checkpoint), "row_id": row["row_id"],
                   "xi_hex": xi.hex(), "Q2_hex": q2.hex(), "t_hex": t.hex(),
                   "family_pipeline_sha256": sha(HERE / "families" / token / "PIPELINE.json")}
        if destination.exists():
            saved = json.loads(destination.read_text())
            if saved["binding"] != binding:
                raise RuntimeError("Changed saved full-forward diagnostic")
            result = saved["result"]
        else:
            class AtT:
                def __call__(self, x, skewness):
                    return problem.model(x, skewness, skewness.new_full(skewness.shape, t))
            point = {"xi": xi, "directories": family["directories"]}
            result = gv.pf.validate_point(point, q2, "experimental_central", AtT())
            atomic(destination, {"binding": binding, "result": result})
        budgets = np.asarray(result["CFF_estimated_total"], dtype=float)
        limits = np.asarray(result["CFF_limits"], dtype=float)
        fields = np.asarray([v["estimated_total"] for v in result["fields"].values()], dtype=float)
        valid = bool(np.isfinite(budgets).all() and np.isfinite(limits).all() and
                     np.isfinite(fields).all() and np.all(limits > 0) and np.all(budgets >= 0))
        records.append({"row_id": row["row_id"], "xi": xi, "Q2": q2, "t": t,
                        "file": destination.name, "file_sha256": sha(destination),
                        "finite": valid, "legacy_guides_pass": result["pass"],
                        "CFF_estimated_total": budgets.tolist(),
                        "CFF_limits": limits.tolist(),
                        "field_estimated_total": fields.tolist(),
                        "warnings": {k: v.get("independent_hard", {}).get("warnings", [])
                                     for k, v in result["checks"].items() if k in ("weak10", "strong10")}})
        print("FULL_FORWARD", row["row_id"], len(records), len(rows), flush=True)
    all_budgets = np.asarray([r["CFF_estimated_total"] for r in records])
    all_limits = np.asarray([r["CFF_limits"] for r in records])
    all_fields = np.asarray([r["field_estimated_total"] for r in records])
    summary = {"rows": len(records), "finite": all(r["finite"] for r in records),
               "max_CFF_budget_over_legacy_limit": float(np.max(all_budgets/all_limits)),
               "max_field_estimated_total": float(np.max(all_fields)),
               "legacy_guides_pass_count": sum(r["legacy_guides_pass"] for r in records),
               "warning_rows": [r["row_id"] for r in records if any(r["warnings"].values())],
               "records": records}
    atomic(out.parent / "FULL_FORWARD_SUMMARY.json", summary)
    return summary


def derivative_and_kkt(problem, flat):
    domain = ExperimentalDomain(HERE / "prepared", HERE / "families")
    oracle = accepted.Oracle(problem, domain, flat.numpy().copy())
    values = oracle.values(flat.numpy())
    constraints = values["c"]
    working = np.flatnonzero(constraints >= -.01)
    gradient, residual_jacobian, A = oracle.derivatives(flat.numpy(), working)
    kkt = accepted.old.kkt(gradient, constraints, A, working, oracle.names)
    return {"J": values["J"], "maximum_constraint": float(np.max(constraints)),
            "n_constraints": len(constraints), "working_indices": working.tolist(),
            "raw_gradient_L2": float(np.linalg.norm(gradient)), "KKT": kkt,
            "residual_jacobian_shape": list(residual_jacobian.shape),
            "oracle_counts": oracle.counts, "oracle_timers": oracle.timers}


def run():
    start = time.monotonic()
    optimizer = HERE / "results/central_optimization"
    roster = json.loads((optimizer / "CANDIDATE_ROSTER.json").read_text())
    if roster != sorted(roster, key=lambda r: (r["J"], r["seed"], r["stage"])):
        raise ValueError("Optimizer candidate ranking changed")
    summary = json.loads((HERE / "prepared/INPUT_SUMMARY.json").read_text())
    results = []
    for index, row in enumerate(roster[:4]):
        out = HERE / "results/central_diagnostics" / f"candidate{index}"
        out.mkdir(parents=True, exist_ok=True)
        checkpoint = out / "CENTRAL_CHECKPOINT.pt"
        cp = checkpoint_with_contract(optimizer / f"ranked/candidate{index}.pt", checkpoint, summary)
        if (cp["seed"], cp["stage"]) != (row["seed"], row["stage"]):
            raise ValueError("Candidate checkpoint/roster mismatch")
        problem = SolverProblem(row["seed"])
        problem.set_model(cp["flat"])
        local = diagnose(HERE / "prepared", HERE / "banks", checkpoint, out / "two_map_diagnostics")
        full = full_forward(problem, checkpoint, out / "full_forward")
        derivative = derivative_and_kkt(problem, cp["flat"])
        atomic(out / "DERIVATIVE_KKT.json", derivative)
        flags = []
        if full["max_CFF_budget_over_legacy_limit"] > 1:
            flags.append("CFF_budget_exceeds_legacy_guide")
        if full["max_field_estimated_total"] > 1e-3:
            flags.append("field_budget_exceeds_legacy_guide")
        if local["moment_RMS_max_at192"] > 1e-3:
            flags.append("moment_RMS_exceeds_legacy_guide")
        if derivative["maximum_constraint"] > 0:
            flags.append("nonlinear_constraint_excess")
        if full["warning_rows"]:
            flags.append("independent_hard_warnings")
        finite = bool(local["finite"] and full["finite"] and np.isfinite(derivative["J"]))
        record = {"rank": index, "seed": row["seed"], "stage": row["stage"],
                  "J": row["J"], "finite": finite, "quality_flags": flags,
                  "checkpoint": str(checkpoint.relative_to(HERE)),
                  "checkpoint_sha256": sha(checkpoint),
                  "two_map_result": local,
                  "full_forward_summary": str((out / "FULL_FORWARD_SUMMARY.json").relative_to(HERE)),
                  "derivative_KKT": str((out / "DERIVATIVE_KKT.json").relative_to(HERE))}
        results.append(record)
        atomic(HERE / "results/CANDIDATE_DIAGNOSTICS_PROGRESS.json", results)
        if finite:
            break
    final = {"status": "FINITE_CENTRAL_APPROXIMATION" if results and results[-1]["finite"] else "ABSTAIN_NONFINITE",
             "selected": results[-1] if results and results[-1]["finite"] else None,
             "candidates_checked": results,
             "seconds_diagnostics": time.monotonic()-start,
             "selection": "Lowest J/seed/stage fully finite candidate among at most four; quality flags retained"}
    atomic(HERE / "results/CENTRAL_DIAGNOSTICS.json", final)
    return final


if __name__ == "__main__":
    print(json.dumps(run(), indent=2), flush=True)
