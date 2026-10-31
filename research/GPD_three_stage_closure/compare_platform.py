"""Compare the destination's fresh initial objective/derivative to the Mac lock."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from closure.solver_binding import problem_and_domain


def main():
    original = json.loads((HERE / "readiness/LOCAL_BINDING_CHECK.json").read_text())
    p, d = problem_and_domain("ideal_L0_Z1_s31")
    x = p.x0.detach().requires_grad_(True)
    value = p.objective(x)
    grad = torch.autograd.grad(value, x)[0]
    rng = np.random.default_rng(20261007)
    direction = torch.tensor(rng.normal(size=3867), dtype=torch.float64)
    direction /= direction.norm()
    derivative = float(grad @ direction)
    J = float(value.detach())
    differences = {"J_abs": abs(J - original["J"]),
                   "directional_AD_abs": abs(derivative - original["directional_AD"])}
    accepted = differences["J_abs"] <= 1e-8 * max(1., abs(original["J"])) and \
               differences["directional_AD_abs"] <= 1e-7 * max(1., abs(original["directional_AD"]))
    result = {"status": "MATCH" if accepted else "PLATFORM_DIFFERENCE_REQUIRES_REVIEW",
              "J": J, "directional_AD": derivative, "mac_reference": {
                  "J": original["J"], "directional_AD": original["directional_AD"]},
              "differences": differences, "platform": platform.platform(),
              "python": sys.version, "torch": torch.__version__, "numpy": np.__version__}
    path = HERE / "readiness/DESTINATION_COMPARISON.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    if not accepted:
        raise RuntimeError("Destination objective/derivative differs; review saved comparison")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
