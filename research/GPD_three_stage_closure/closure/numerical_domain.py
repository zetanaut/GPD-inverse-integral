"""Original numerical inequalities at the exact closure case geometry."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "GPD_ReH_experimental_run"))
from experimental_domain import ExperimentalDomain, TDependentMoments
from .numerical_cache import load_pair


class ClosureDomain(ExperimentalDomain):
    def __init__(self, problem, study, margin=.95, order=192):
        self.mom = TDependentMoments(study["polynomiality"]["t_nodes"], order=order)
        self.margin = float(margin)
        self.entries = []
        self.names = ["moment_predictive_RMS_80_by_13_over_1e-3"]
        pairs = {}
        for row_id, row in zip(problem.row_ids, problem.geometry):
            xi, q2, t = row["xi"], row["Q2_GeV2"], row["t_GeV2"]
            key = (xi.hex(), q2.hex())
            if key not in pairs:
                pairs[key] = load_pair(xi, q2)
            point, spin2 = pairs[key]
            self.entries.append((row_id, xi, q2, t, point, spin2))
            self.names += [row_id + "_CFF_" + s for s in ("Re", "Im")]
            self.names += [row_id + "_field_" + s for s in ("Sigma", "T", "Hg")]
            self.names += [row_id + "_transport_" + m + "_" + s
                           for m in ("weak10", "strong10") for s in ("Sigma", "Hg", "T")]
