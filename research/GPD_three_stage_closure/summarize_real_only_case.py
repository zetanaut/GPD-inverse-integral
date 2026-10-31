"""Per-case post-selection science and resource numbers for the real-only bridge."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from closure.bank import RowBank
from closure.problem import GroupedForward
from closure.solver_binding import problem_and_domain
from closure.truths import Truth
from run_real_only_case import ALLOWED, atomic_json


def truth_targets(truth):
    z, w = np.polynomial.legendre.leggauss(384)
    values = []
    for lo, hi, mode in ((.2, .4, "bin"), (0., 1., "xmoment")):
        x = (lo+hi)/2+(hi-lo)*z/2
        weight = w*(hi-lo)/2
        h = truth.values(x, .275, -.363)
        charge = (2/9)*(h[:,0]+h[:,1])
        values.append(float(np.sum(charge*weight*(5. if mode == "bin" else x))))
    return values


def count_warnings(out, validation):
    total = 0
    for row in validation.get("full_operator", {}).get("coordinate_selected_rows", []):
        path = out / row["file"]
        detail = json.loads(path.read_text())
        for item in detail.get("checks", {}).values():
            hard = item.get("independent_hard", {})
            total += len(hard.get("warnings", []))
    return total


def main(task_id):
    if task_id not in ALLOWED:
        raise ValueError("Task is not authorized")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    out = HERE / "runs" / task_id
    result = json.loads((out / "RESULT.json").read_text())
    score = json.loads((out / "SCORES.json").read_text())
    task = score["task"]
    ck = torch.load(out / result["scored_checkpoint"], weights_only=True, map_location="cpu")
    flat = ck["flat"].detach().double()
    p, _ = problem_and_domain(task_id)
    model = p.set_model(flat)
    bank = RowBank(HERE / "inputs/accepted_experimental_banks", HERE / "scale_rows")
    truth = Truth(task["truth"])
    with torch.no_grad():
        nom = GroupedForward(p.geometry, bank, "nominal")(model).numpy()
        ref = GroupedForward(p.geometry, bank, "reference")(model).numpy()
        components = p.components(flat).numpy()
    observed = p.y[:,0].numpy()
    sigma = p.sigma[:,0].numpy()
    mass = p.mass.numpy()
    re_data_RMS = float(np.sqrt(np.sum(mass*((nom[:,0]-observed)/sigma)**2)))
    numerical = nom[:,0]-ref[:,0]
    target = bank.field_cff(p.geometry, truth, "reference")
    clean_Re_RMS = float(np.sqrt(np.mean((ref[:,0]-target[:,0])**2)))
    validated = result["selected"] if result["selected"] is not None else next(
        (x for x in result["validations"] if x["source_checkpoint"]["path"] ==
         result["scored_checkpoint"]), result["validations"][0])
    target_pred = validated["target_384"]
    target_true = truth_targets(truth)
    field = score["field"]
    optimization = result["optimization"]
    peak = max(result["peak_RSS_bytes"], optimization["peak_RSS_bytes"])
    total = (out / "RESULT.json").stat().st_mtime - (out / "RUN_LOCK.json").stat().st_mtime
    moment = validated.get("moments", {}).get("quadrature_RMS", {}).get("384")
    row = {
        "task_id": task_id, "seed": task["seed"], "input_type": "noisy" if task["branch"] == "exp_Re_noisy" else "exact",
        "observed_sha256": json.loads((out / "RUN_LOCK.json").read_text())["observed_sha256"],
        "initial_state_sha256": optimization["fresh_initial_sha256"],
        "selected_checkpoint_sha256": result["scored_checkpoint_sha256"],
        "stage": ck["stage"], "selection_status": result["status"],
        "J_data": float(components[0]), "J_poly": float(components[1]), "J_total": float(sum(components)),
        "Re_data_RMS_in_quoted_error_units": re_data_RMS,
        "Re_clean_truth_RMS": clean_Re_RMS,
        "charge_full_relative_L2": field["charge_relative_L2"]["all"],
        "charge_ERBL_relative_L2": field["charge_relative_L2"]["ERBL"],
        "charge_DGLAP_relative_L2": field["charge_relative_L2"]["DGLAP"],
        "charge_max_slice_relative_L2": field["charge_max_slice_relative_L2"],
        "charge_absolute_RMS": field["charge_absolute_RMS"]["all"],
        "bin_error": target_pred[0]-target_true[0],
        "xmoment_error": target_pred[1]-target_true[1],
        "independent_moment_RMS": moment,
        "numerical_Re_max_abs": float(np.max(abs(numerical))),
        "numerical_Re_sigma_units": float(np.max(abs(numerical)/sigma)),
        "integration_warning_count": count_warnings(out, validated),
        "integration_warning_scope": "three coordinate-selected full-operator post-fit rows",
        "optimizer_termination": optimization["solver_status"],
        "solver_seconds": optimization["stage_times"]["solver_seconds"],
        "total_seconds": total, "peak_memory_bytes": peak,
        "fit_Re_truth_relative_L2": score["CFF"]["experimental"]["truth_error"]["relative_L2"][0],
        "fit_Im_truth_relative_L2_unfitted": score["CFF"]["experimental"]["truth_error"]["relative_L2"][1],
        "holdout_Re_truth_relative_L2": score["CFF"]["surface_check"]["truth_error"]["relative_L2"][0],
        "holdout_Im_truth_relative_L2_unfitted": score["CFF"]["surface_check"]["truth_error"]["relative_L2"][1],
        "target_predicted": target_pred, "target_truth": target_true,
        "candidate_outcomes": [{"stage": x["stage"], "J": x["J"], "pass": x["pass"],
                                "constraint_max": x["constraint_max"],
                                "moments_pass": x["moments"]["pass"],
                                "full_operator_pass": x["full_operator"]["pass"]}
                               for x in result["validations"]],
        "stage_times_seconds": optimization["stage_times"],
        "oracle_timers_seconds": optimization["oracle_timers"],
        "binding_stats": optimization["binding_stats"],
        "linear_algebra_seconds_overlaps_subproblem": optimization["linear_algebra_seconds"],
        "not_a_calibrated_uncertainty_interval": True,
    }
    if abs(row["J_data"] - re_data_RMS**2) > 1e-9*max(1,abs(row["J_data"])):
        raise RuntimeError("Reported weighted Re RMS and objective data term differ")
    atomic_json(out / "CASE_SUMMARY.json", row)
    print(json.dumps({k:row[k] for k in ("task_id","selection_status","stage","J_total",
                                        "charge_full_relative_L2","solver_seconds","peak_memory_bytes")}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=ALLOWED, required=True)
    main(parser.parse_args().task)
