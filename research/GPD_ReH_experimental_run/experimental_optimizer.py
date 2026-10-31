"""Bind the tested ReH/t problem to the accepted two-start finite-budget loop."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ["GPD_REVIEW_ROOT"] = str(ROOT)
sys.path[:0] = [str(ROOT / "GPD_direct_error_calibration"),
                str(ROOT / "GPD_NLO_constrained_solver_review"),
                str(ROOT / "GPD_accuracy_cost_study")]

from gpd_reh.problem import ExperimentalProblem
from gpd_reh.bank import preflight, read_index
from experimental_domain import ExperimentalDomain
from study_core import training_prefix
import fit_nlo as accepted


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


class SolverProblem(ExperimentalProblem):
    def __init__(self, seed):
        super().__init__(HERE / "prepared", HERE / "banks", seed=seed)
        self.n_scalar = len(self.y)
        if self.n_scalar != 160:
            raise ValueError("Wrong experimental data denominator")

    def components(self, flat):
        residual = self.residual(flat)
        return torch.stack((residual[:self.n_scalar].square().sum(),
                            residual[self.n_scalar:].square().sum()))

    def residual_jacobian(self, flat):
        return self.jacobian(flat)


def source_binding():
    names = [
        "GPD_ReH_experimental_run/experimental_optimizer.py",
        "GPD_ReH_experimental_run/experimental_domain.py",
        "GPD_ReH_experimental_run/gpd_reh/problem.py",
        "GPD_ReH_experimental_run/gpd_reh/model.py",
        "GPD_ReH_experimental_run/gpd_reh/moments.py",
        "GPD_ReH_experimental_run/gpd_reh/bank.py",
        "GPD_direct_error_calibration/fit_nlo.py",
        "GPD_NLO_constrained_solver_review/bind_solver.py",
        "GPD_NLO_constrained_solver_review/residual_subproblem.py",
        "completion_runs/NLO_constrained_solver/solver.py",
        "GPD_accuracy_cost_study/study_core.py",
    ]
    return {name: sha(ROOT / name) for name in names}


def freeze():
    mean_t = json.loads((HERE / "results/MEAN_T_MODEL_CHECK.json").read_text())
    if (mean_t["mean_t_reference_max_abs"] > 1e-14 or mean_t["t_change_L2"] <= 0 or
            mean_t["parameter_count"] != 3867 or not mean_t["all_parameters_trainable"]):
        raise RuntimeError("Mean-t reproduction or active t conditioning failed")
    pre = preflight(HERE / "prepared", HERE / "banks")
    if pre["status"] != "READY_FOR_CENTRAL_FIT" or pre["n_exact_pairs_present"] != 102:
        raise RuntimeError("Exact experimental NLO rows incomplete")
    index = read_index(HERE / "banks")
    family = json.loads((HERE / "results/EXPERIMENTAL_BATCH.json").read_text())
    if family["status"] != "ALL_FAMILIES_COMPLETE":
        raise RuntimeError("Numerical family construction incomplete")
    source = json.loads((HERE / "CENTRAL_CONTRACT.json").read_text())
    output = {
        "source_hash": source["input"]["sha256"],
        "t_mean": source["network"]["t_mean"],
        "t_std_population": source["network"]["t_std_population"],
        "bank_index_sha256": sha(HERE / "banks/index.json"),
        "bank_entries": len(index["entries"]),
        "contractions": {r["xi_hex"]: sha(HERE / r["contractions"])
                         for r in family["families"]},
        "sources": source_binding(),
        "mean_t_model_check_sha256": sha(HERE / "results/MEAN_T_MODEL_CHECK.json"),
        "Adam_steps": 400,
        "learning_rate": .003,
        "trust_region_iterations": 60,
        "seeds": [31, 43],
        "candidate_stages": [0, 20, 40, 60],
        "ranking": ["J", "seed", "stage"],
        "max_diagnostics_candidates": 4,
    }
    path = HERE / "results/OPTIMIZER_LOCK.json"
    if path.exists():
        if json.loads(path.read_text()) != output:
            raise RuntimeError("Frozen experimental optimizer/configuration changed")
    else:
        atomic(path, output)
    return output


def preflight_derivatives():
    problem = SolverProblem(31)
    flat = problem.x0.detach().clone()
    r = problem.residual(flat)
    J = float(problem.objective(flat))
    independent = float(problem.independent_objective(flat))
    if not np.isfinite(J) or abs(J-float(r.square().sum())) > 1e-10 * max(1., abs(J)):
        raise RuntimeError("J != squared residual norm")
    if abs(J-independent) > 1e-8 * max(1., abs(J)):
        raise RuntimeError("Independent objective disagrees")
    x = flat.requires_grad_(True)
    gradient = torch.autograd.grad(problem.objective(x), x)[0].detach()
    jac = problem.jacobian(flat)
    reconstructed = 2 * jac.T @ r
    error = float((gradient-reconstructed).abs().max())
    if error > 1e-6 * max(1., float(gradient.abs().max())):
        raise RuntimeError("Original-coordinate gradient/Jacobian disagreement")
    record = {"J": J, "independent_J": independent,
              "residual_count": int(r.numel()), "parameter_count": int(flat.numel()),
              "gradient_jacobian_max_abs": error}
    atomic(HERE / "results/OPTIMIZER_DERIVATIVE_CHECK.json", record)
    return record


def run():
    lock = freeze()
    out = HERE / "results/central_optimization"
    out.mkdir(exist_ok=True)
    terminal = out / "OPTIMIZATION_COMPLETE.json"
    if terminal.exists():
        record = json.loads(terminal.read_text())
        if record["lock_sha256"] != sha(HERE / "results/OPTIMIZER_LOCK.json"):
            raise RuntimeError("Completed optimization binding changed")
        return record
    if (out / "OPTIMIZATION_STARTED.json").exists():
        raise RuntimeError("Interrupted central optimization preserved for review")
    checks = preflight_derivatives()
    bind_path = ROOT / "GPD_NLO_constrained_solver_review/bind_solver.py"
    spec = importlib.util.spec_from_file_location("experimental_isolated_bind_solver", bind_path)
    binding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binding)
    binding.residual_and_jacobian = lambda problem, domain, flat: domain.mom.residual_and_jacobian(problem, flat)
    with np.load(HERE / "prepared/observations.npz", allow_pickle=False) as z:
        y = z["y"].copy()
        covariance = np.diag(z["sigma"].copy()**2)
    def problem_factory(values, cov, seed):
        if not np.array_equal(values, y) or not np.array_equal(cov, covariance):
            raise ValueError("Experimental observations/covariance changed")
        return SolverProblem(seed)
    def domain_factory():
        return ExperimentalDomain(HERE / "prepared", HERE / "families")
    fn, old_ast_sha = training_prefix(accepted.run, {
        "Problem": problem_factory, "Domain": domain_factory, "bind_solver": binding,
    })
    atomic(out / "OPTIMIZATION_STARTED.json", {"lock_sha256": sha(HERE / "results/OPTIMIZER_LOCK.json"),
                                               "old_optimizer_AST_sha256": old_ast_sha,
                                               "derivative_preflight": checks})
    result = fn(y, covariance, out)
    result["lock_sha256"] = sha(HERE / "results/OPTIMIZER_LOCK.json")
    result["old_optimizer_AST_sha256"] = old_ast_sha
    result["optimizer_contract"] = lock
    atomic(terminal, result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2), flush=True)
