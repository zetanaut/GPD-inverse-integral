"""One locked, fresh Z1/L0/seed31 finite-budget optimization on this Mac."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from closure.io import sha256
from closure.solver_binding import accepted, adam400, isolated_solver, oracle_at, problem_and_domain, restored_start

TASK = "ideal_L0_Z1_s31"
OUT = HERE / "runs" / TASK


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_torch(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def checkpoint(problem, flat, stage, path):
    with torch.no_grad():
        state = {name: value.detach().clone() for name, value in problem.parameters(flat).items()}
        J = float(problem.objective(flat))
    atomic_torch(path, {"flat": flat.detach().clone(), "state_dict": state,
                        "seed": 31, "stage": stage, "J": J})
    return {"stage": stage, "J": J, "path": str(path.relative_to(OUT)), "sha256": sha256(path)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # Verify the original execution lock and exact arrays before starting.
    from verify_portable import main as verify
    verify()
    source_lock = HERE / "readiness/EXECUTION_LOCK.json"
    lock = json.loads(source_lock.read_text())
    if lock["case_order"][0] != TASK:
        raise RuntimeError("First registered task changed")
    observed = HERE / "prepared/observed/ideal_L0_Z1.npz"
    numerical = HERE / "numerical_contractions/TASK_ideal_L0_Z1_s31.json"
    binding = {"task": TASK, "source_lock_sha256": sha256(source_lock),
               "study_sha256": sha256(HERE / "prepared/STUDY.json"),
               "observed_sha256": sha256(observed),
               "accepted_map_index_sha256": sha256(HERE / "inputs/accepted_experimental_banks/index.json"),
               "new_map_index_sha256": sha256(HERE / "scale_rows/index.json"),
               "numerical_index_sha256": sha256(numerical),
               "runner_sha256": sha256(__file__), "platform": platform.platform(),
               "python": sys.version, "torch": torch.__version__, "numpy": np.__version__}
    run_lock = OUT / "RUN_LOCK.json"
    if run_lock.exists():
        if json.loads(run_lock.read_text()) != binding:
            raise RuntimeError("Existing coarse-case binding changed")
    else:
        atomic_json(run_lock, binding)
    if (OUT / "OPTIMIZER_COMPLETE.json").exists():
        print(json.dumps({"status": "OPTIMIZER_ALREADY_COMPLETE", "path": str(OUT)}), flush=True)
        return
    p, d = problem_and_domain(TASK)
    original_initial_sha = hashlib.sha256(p.x0.numpy().tobytes()).hexdigest()
    expected = json.loads((HERE / "readiness/LOCAL_BINDING_CHECK.json").read_text())["seed31_initial_sha256"]
    if original_initial_sha != expected or not torch.equal(p.x0, problem_and_domain(TASK)[0].x0):
        raise RuntimeError("Prescribed fresh seed31 initialization changed")
    started = time.monotonic()
    stage_times = {}
    if (OUT / "ADAM400.pt").exists():
        ck = torch.load(OUT / "ADAM400.pt", weights_only=True)
        if ck["seed"] != 31 or ck["stage"] != "Adam400":
            raise RuntimeError("Fresh Adam checkpoint mismatch")
        adam_state = ck["flat"].detach()
        stage_times["Adam400_seconds"] = json.loads((OUT / "ADAM400.json").read_text())["seconds"]
    else:
        begin = time.monotonic()
        adam_state = adam400(p)
        stage_times["Adam400_seconds"] = time.monotonic() - begin
        rec = checkpoint(p, adam_state, "Adam400", OUT / "ADAM400.pt")
        atomic_json(OUT / "ADAM400.json", {"checkpoint": rec, "seconds": stage_times["Adam400_seconds"],
                                             "initial_sha256": original_initial_sha,
                                             "not_the_prior_timing_checkpoint": True})
        print(json.dumps({"stage": "Adam400", "J": rec["J"], "seconds": stage_times["Adam400_seconds"]}), flush=True)
    if (OUT / "STAGE0.pt").exists():
        ck = torch.load(OUT / "STAGE0.pt", weights_only=True)
        stage0 = ck["flat"].detach()
        stage_times["restoration_seconds"] = json.loads((OUT / "RESTORATION.json").read_text())["seconds"]
    else:
        begin = time.monotonic()
        choice, details = restored_start(p, d, adam_state)
        stage_times["restoration_seconds"] = time.monotonic() - begin
        if choice is None:
            atomic_json(OUT / "RESTORATION_FAILURE.json", {"details": details, "seconds": stage_times["restoration_seconds"]})
            raise RuntimeError("No feasible original-rule start")
        value, name, stage0 = choice
        rec = checkpoint(p, stage0, 0, OUT / "STAGE0.pt")
        atomic_json(OUT / "RESTORATION.json", {"selected": name, "selected_J": value,
                                                  "candidates": details, "checkpoint": rec,
                                                  "seconds": stage_times["restoration_seconds"]})
        print(json.dumps({"stage": "restoration", "selected": name, "J": value,
                          "seconds": stage_times["restoration_seconds"]}), flush=True)
    oracle = oracle_at(p, d, stage0)
    solver, oracle, stats, settings = isolated_solver(oracle)
    if settings["max_iterations"] != 60 or settings["initial_radius"] != .01:
        raise RuntimeError("Original finite budget changed")
    if (OUT / "ITERATIONS.jsonl").exists():
        raise RuntimeError("Interrupted solver history retained; no silent restart")
    history_path = OUT / "ITERATIONS.jsonl"
    solver_begin = time.monotonic()
    def record(a, rec, trial):
        iteration = rec["iteration"] + 1
        if iteration in (20, 40, 60):
            checkpoint(p, torch.as_tensor(a, dtype=torch.float64), iteration, OUT / f"STAGE{iteration}.pt")
        line = {"iteration": iteration, "J": rec["J"], "accepted": rec.get("accepted"),
                "radius_end": rec.get("radius_end"), "trials": len(rec.get("trials", [])),
                "seconds_since_solver_start": time.monotonic() - solver_begin,
                "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        with history_path.open("a") as stream:
            stream.write(json.dumps(line, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(line), flush=True)
    last, diag = solver.solve(oracle, accepted.old.kkt, record, settings)
    stage_times["solver_seconds"] = time.monotonic() - solver_begin
    final_stage = len(diag["history"])
    if not (OUT / f"STAGE{final_stage}.pt").exists():
        checkpoint(p, torch.as_tensor(last, dtype=torch.float64), final_stage, OUT / f"STAGE{final_stage}.pt")
    atomic_json(OUT / "SOLVER_DIAGNOSTICS.json", diag)
    candidates = []
    for stage in (0, 20, 40, 60, final_stage):
        path = OUT / f"STAGE{stage}.pt"
        if path.exists() and stage not in [c["stage"] for c in candidates]:
            ck = torch.load(path, weights_only=True)
            candidates.append({"stage": stage, "J": ck["J"], "path": str(path.relative_to(OUT)),
                               "sha256": sha256(path)})
    candidates.sort(key=lambda item: (item["J"], 31, item["stage"]))
    complete = {"status": "FINITE_BUDGET_OPTIMIZER_COMPLETE_AWAITING_DIAGNOSTICS",
                "task": TASK, "fresh_initial_sha256": original_initial_sha,
                "chosen_start": json.loads((OUT / "RESTORATION.json").read_text())["selected"],
                "solver_status": diag["status"], "solver_iterations": final_stage,
                "candidates_by_supplied_J": candidates, "stage_times": stage_times,
                "wall_seconds_since_process_start": time.monotonic() - started,
                "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "oracle_timers": diag["oracle_timers"], "binding_stats": stats,
                "linear_algebra_seconds": diag["linear_algebra_seconds"],
                "oracle_counts": diag["oracle_counts"], "solver_counts": diag["counts"]}
    atomic_json(OUT / "OPTIMIZER_COMPLETE.json", complete)
    print(json.dumps({"status": complete["status"], "iterations": final_stage,
                      "solver_seconds": stage_times["solver_seconds"],
                      "peak_RSS_bytes": complete["peak_RSS_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
