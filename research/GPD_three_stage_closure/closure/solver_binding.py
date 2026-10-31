"""Isolated binding of the accepted finite-budget solver to ClosureProblem.

No truth module, evaluation-only target, or timing checkpoint is imported.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
os.environ["GPD_REVIEW_ROOT"] = str(ROOT)
sys.path[:0] = [str(ROOT / "GPD_direct_error_calibration"),
                str(ROOT / "GPD_NLO_constrained_solver_review"),
                str(ROOT / "GPD_accuracy_cost_study")]

import fit_nlo as accepted
from .bank import RowBank
from .problem import ClosureProblem
from .numerical_domain import ClosureDomain


def task_by_id(task_id):
    study = json.loads((HERE / "prepared/STUDY.json").read_text())
    found = [item for item in study["core_tasks"] if item["id"] == task_id]
    if len(found) != 1:
        raise ValueError("Unknown or duplicate registered task")
    return study, found[0]


def problem_and_domain(task_id):
    study, task = task_by_id(task_id)
    bank = RowBank(HERE / "inputs/accepted_experimental_banks", HERE / "scale_rows")
    path = HERE / "prepared/observed" / f"{task['branch']}_{task['truth']}.npz"
    p = ClosureProblem(path, bank, study, seed=task["seed"])
    d = ClosureDomain(p, study)
    return p, d


def isolated_solver(oracle):
    """Patch only the old step adapter's moment derivative to the 13-t geometry."""
    path = ROOT / "GPD_NLO_constrained_solver_review/bind_solver.py"
    spec = importlib.util.spec_from_file_location("closure_isolated_bind_solver", path)
    binding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binding)
    binding.residual_and_jacobian = lambda p, d, flat: d.mom.residual_and_jacobian(p, flat)
    solver_path = ROOT / "completion_runs/NLO_constrained_solver/solver.py"
    spec = importlib.util.spec_from_file_location("closure_isolated_outer_solver", solver_path)
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
    oracle, stats = binding.bind(solver, oracle)
    settings = dict(solver.SETTINGS, max_iterations=60, max_evaluations=12000,
                    optimization_seconds=float("inf"), initial_radius=.01)
    return solver, oracle, stats, settings


def oracle_at(problem, domain, flat):
    return accepted.Oracle(problem, domain, flat.detach().numpy())


def adam400(problem):
    """Original fresh seed and optimizer settings; caller owns checkpointing."""
    x = problem.x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=.003, betas=(.9, .999), eps=1e-8, weight_decay=0)
    for _ in range(400):
        opt.zero_grad(set_to_none=True)
        loss = problem.objective(x)
        if not torch.isfinite(loss):
            raise FloatingPointError("Adam nonfinite")
        loss.backward()
        opt.step()
    return x.detach()


def restored_start(problem, domain, adam_state):
    """Same random/Adam comparison and head restoration as the accepted runner."""
    options = []
    details = []
    for name, state in (("random", problem.x0), ("Adam400", adam_state)):
        restored, info = accepted.restore_head(problem, domain, state)
        details.append({"name": name, **info})
        if restored is not None:
            with torch.no_grad():
                options.append((float(problem.objective(restored)), name, restored))
    if not options:
        return None, details
    options.sort(key=lambda value: (value[0], value[1]))
    return options[0], details
