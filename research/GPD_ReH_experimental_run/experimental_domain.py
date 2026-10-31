"""Actual-t numerical constraints and fixed-t predictive moments for the old solver."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.func import jacrev

from gpd_reh.moments import FixedTMoments
from gpd_reh.polynomiality_base import design

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "GPD_NLO_resolution_review/pilot"))
import common


class TDependentMoments:
    """The accepted disjoint-xi predictive norm, averaged across seven fixed t."""
    def __init__(self, t_nodes, order=192):
        fit_xi = np.linspace(.02, .98, 41)
        _, independent_xi = common.helpers.moment_grids()
        independent_xi = np.asarray(independent_xi)
        self.fit = FixedTMoments(t_nodes, xi_nodes=fit_xi, order=order)
        self.independent = FixedTMoments(t_nodes, xi_nodes=independent_xi, order=order)
        self.predict = []
        for _, spin, _ in self.fit.spec:
            pfit = design(fit_xi, spin)
            self.predict.append(torch.tensor(design(independent_xi, spin) @ np.linalg.pinv(pfit),
                                             dtype=torch.float64))
        self.denominator = 1e-3 * math.sqrt(len(t_nodes) * len(independent_xi) * len(self.fit.spec))

    def residual(self, problem, flat):
        model = lambda x, xi, t: problem.field(flat, x, xi, t)
        values = []
        for t in self.fit.t_nodes:
            mf = self.fit.raw_moments(model, t)
            md = self.independent.raw_moments(model, t)
            values.append(torch.stack([(md[:, i] - matrix @ mf[:, i]) / self.fit.scales[i]
                                       for i, matrix in enumerate(self.predict)], dim=1))
        return torch.cat(values).flatten() / self.denominator

    def value(self, problem, flat):
        return torch.linalg.vector_norm(self.residual(problem, flat))

    def residual_and_jacobian(self, problem, flat):
        """Differentiate integrated moment slices, avoiding a whole-field AD graph."""
        channel = torch.tensor([s[0] for s in self.fit.spec], dtype=torch.long)
        moments = []
        derivatives = []
        for t in self.fit.t_nodes:
            parts = []
            jac_parts = []
            for grid in (self.fit, self.independent):
                mrows = []
                jrows = []
                for j in range(len(grid.xi)):
                    weights = grid.xw[j]
                    xx = grid.x[j:j+1]
                    xi = grid.xi[j:j+1]
                    tt = t.reshape(1)
                    def local(theta):
                        h = problem.field(theta, xx, xi, tt)[0]
                        return (h[channel] * weights).sum(-1)
                    with torch.no_grad():
                        mrows.append(local(flat).detach())
                    jrows.append(jacrev(local)(flat.detach()).detach())
                parts.append(torch.stack(mrows))
                jac_parts.append(torch.stack(jrows))
            mf, md = parts
            jf, jd = jac_parts
            moments.append(torch.stack([(md[:, i] - matrix @ mf[:, i]) / self.fit.scales[i]
                                        for i, matrix in enumerate(self.predict)], dim=1))
            derivatives.append(torch.stack([(jd[:, i] - matrix @ jf[:, i]) / self.fit.scales[i]
                                            for i, matrix in enumerate(self.predict)], dim=1))
        e = torch.cat(moments).flatten() / self.denominator
        E = torch.cat(derivatives).reshape(-1, flat.numel()) / self.denominator
        return e.numpy(), E.numpy()


class ExperimentalDomain:
    """One predictive norm and eleven existing numerical inequalities per ReH row."""
    def __init__(self, prepared, families, row_ids=None, margin=.95, order=192):
        prepared = Path(prepared)
        with (prepared / "observations.csv").open(newline="") as f:
            observations = list(csv.DictReader(f))
        if row_ids is not None:
            wanted = set(row_ids)
            observations = [r for r in observations if r["row_id"] in wanted]
            if len(observations) != len(wanted):
                raise ValueError("Requested domain rows missing")
        summary = json.loads((prepared / "INPUT_SUMMARY.json").read_text())
        self.mom = TDependentMoments(summary["moment_t_nodes"], order=order)
        self.margin = float(margin)
        self.entries = []
        self.names = ["moment_predictive_RMS_80_by_7_over_1e-3"]
        family_cache = {}
        for row in observations:
            xi = float(row["xi"])
            q2 = float(row["Q2_GeV2"])
            token = hashlib.sha256(xi.hex().encode()).hexdigest()[:16]
            if token not in family_cache:
                path = Path(families) / token / "CONTRACTIONS.pt"
                family_cache[token] = torch.load(path, weights_only=True)
                if family_cache[token]["binding"]["xi_hex"] != xi.hex():
                    raise ValueError("Numerical contraction xi mismatch")
            group = family_cache[token]
            if str(q2) not in group["spin2"]:
                raise ValueError("Numerical contraction Q2 missing")
            self.entries.append((row["row_id"], xi, q2, float(row["t_GeV2"]),
                                 group["point"], group["spin2"][str(q2)]))
            prefix = row["row_id"]
            self.names += [prefix + "_CFF_" + s for s in ("Re", "Im")]
            self.names += [prefix + "_field_" + s for s in ("Sigma", "T", "Hg")]
            self.names += [prefix + "_transport_" + m + "_" + s
                           for m in ("weak10", "strong10") for s in ("Sigma", "Hg", "T")]

    def constraints(self, problem, flat):
        values = [(self.mom.value(problem, flat) - 1).reshape(1)]
        for _, xi, q2, t, point, spin2 in self.entries:
            key = str(q2)
            ds = point["data"]
            starts = {}
            xi_tensor = flat.new_tensor([xi])
            t_tensor = flat.new_tensor([t])
            for label, data in ds.items():
                field = problem.field(flat, data["nodes"][None], xi_tensor, t_tensor)[0]
                starts[label] = torch.cat((field[0, data["qids"]],
                                           field[1, data["qids"]],
                                           field[2, data["gids"]]))
            evolved = {label: data["U"][key] @ starts[label] for label, data in ds.items()}
            cff = {label: data["hard"][key] @ evolved[label] for label, data in ds.items()}
            reserve = 1e-6 * max(1., np.log(q2 / 2.) / np.log(2.3 / 2.))
            cff_budget = flat.new_full((2,), reserve)
            field_budget = flat.new_full((3,), reserve)
            for pair in point["pairs"]:
                left, right = pair["left"], pair["right"]
                nl, nr = ds[left]["nq"], ds[right]["nq"]
                vl, vr = evolved[left], evolved[right]
                fl = torch.stack((pair["lq"] @ vl[:nl], pair["lq"] @ vl[nl:2*nl],
                                  pair["lg"] @ vl[2*nl:]))
                fr = torch.stack((pair["rq"] @ vr[:nr], pair["rq"] @ vr[nr:2*nr],
                                  pair["rg"] @ vr[2*nr:]))
                err = torch.linalg.vector_norm((fl-fr)*torch.sqrt(pair["w"]), dim=1)
                den = torch.linalg.vector_norm(fr*torch.sqrt(pair["w"]), dim=1).clamp_min(1e-15)
                field_budget = field_budget + pair["weight"] * err / den
                cff_budget = cff_budget + pair["weight"] * torch.abs(cff[left]-cff[right])
            for method in ("weak10", "strong10"):
                cff_budget = cff_budget + 2*torch.abs(ds[method]["hard_delta"][key] @ evolved[method])
            limit = 1e-4 * torch.abs(cff["strong10"]).clamp_min(1.)
            values.extend((cff_budget/(self.margin*limit)-1,
                           field_budget/(self.margin*1e-3)-1))
            for method in ("weak10", "strong10"):
                data = ds[method]
                reference = spin2 @ (data["M"] @ starts[method])
                actual = data["M"] @ evolved[method]
                values.append(torch.abs(actual-reference)/
                              (self.margin*1e-4*torch.abs(reference).clamp_min(1e-15))-1)
        return torch.cat(values)
