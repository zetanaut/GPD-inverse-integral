"""Sequential cache-resuming exact-family construction with a measured resource gate."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

HERE = Path(__file__).resolve().parent
FIRST_XI = 0.20700060350030175
FAMILIES = json.loads((HERE / "prepared/kernel_families.json").read_text())["families"]
STATE = HERE / "results/EXPERIMENTAL_BATCH.json"
GATE = HERE / "results/MEASURED_RESOURCE_PLAN.json"


def atomic(path, data):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def family_dir(xi):
    token = hashlib.sha256(float(xi).hex().encode()).hexdigest()[:16]
    return HERE / "families" / token


def measured_gate():
    if GATE.exists():
        prior = json.loads(GATE.read_text())
        if prior["status"] != "PASS":
            raise RuntimeError("Preserved resource blocker")
        if "representative_peak_child_RSS_bytes_macos" not in prior:
            first = json.loads((family_dir(FIRST_XI) / "PIPELINE.json").read_text())
            prior["representative_peak_child_RSS_bytes_macos"] = max(
                step["peak_child_RSS_bytes_macos"] for step in first["steps"])
            prior["peak_RSS_scope"] = "Maximum measured child RSS in representative pipeline; independently timed base did not expose peak RSS"
            atomic(GATE, prior)
        return prior
    first = json.loads((family_dir(FIRST_XI) / "PIPELINE.json").read_text())
    if first["status"] != "FAMILY_VALIDATED_WITH_CONTINUOUS_DIAGNOSTICS":
        raise RuntimeError("Representative family must finish first")
    contraction = family_dir(FIRST_XI) / "CONTRACTIONS.pt"
    if not contraction.exists():
        raise RuntimeError("Representative numerical contraction must finish first")
    import torch
    contract_record = torch.load(contraction, weights_only=True)
    used = first["operator_bytes"] + sum(p.stat().st_size for p in family_dir(FIRST_XI).rglob("*") if p.is_file())
    wall = sum(s["seconds"] for s in first["steps"]) + contract_record["seconds"]
    free = shutil.disk_usage(HERE).free
    reserve = 10 * 1024**3
    remaining = len(FAMILIES) - 1
    projected = 2 * used * remaining + reserve
    record = {
        "representative_xi": FIRST_XI,
        "representative_wall_seconds": wall,
        "representative_operator_bytes": used,
        "representative_peak_child_RSS_bytes_macos": max(
            step["peak_child_RSS_bytes_macos"] for step in first["steps"]),
        "peak_RSS_scope": "Maximum measured child RSS in representative pipeline; independently timed base did not expose peak RSS",
        "families_total": len(FAMILIES),
        "remaining_families": remaining,
        "sequential": True,
        "estimated_remaining_seconds_at_representative_rate": wall * remaining,
        "free_bytes_at_gate": free,
        "storage_projection": "2 * representative operator bytes * remaining families + 10 GiB reserve",
        "required_free_bytes": projected,
        "status": "PASS" if free > projected else "BLOCKED",
    }
    atomic(GATE, record)
    if record["status"] != "PASS":
        raise RuntimeError("Measured storage plan exceeds free space")
    return record


def main():
    measured_gate()
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "status": "RUNNING", "families": [], "started_unix": time.time(),
    }
    completed = {r["xi_hex"] for r in state["families"] if r["status"] == "COMPLETE"}
    for family in FAMILIES:
        xi = float(family["xi"])
        if float(xi).hex() in completed:
            continue
        if any(r["xi_hex"] == float(xi).hex() and r["status"] == "FAILED" for r in state["families"]):
            raise RuntimeError("Preserved failed family requires review")
        folder = family_dir(xi)
        record = next((r for r in state["families"] if r["xi_hex"] == float(xi).hex()), None)
        if record is None:
            record = {"xi": xi, "xi_hex": float(xi).hex(), "status": "RUNNING"}
            state["families"].append(record)
        atomic(STATE, state)
        pipeline = folder / "PIPELINE.json"
        if not pipeline.exists() or json.loads(pipeline.read_text())["status"] != "FAMILY_VALIDATED_WITH_CONTINUOUS_DIAGNOSTICS":
            log = folder / "BATCH_DRIVER.log"
            folder.mkdir(parents=True, exist_ok=True)
            with log.open("a") as out:
                rc = subprocess.run([sys.executable, str(HERE / "prepare_experimental_family.py"),
                                     "--xi", str(xi)], cwd=HERE.parent,
                                    stdout=out, stderr=subprocess.STDOUT).returncode
            if rc:
                record.update(status="FAILED", returncode=rc, log=str(log.relative_to(HERE)))
                state["status"] = "BLOCKED"; atomic(STATE, state)
                raise RuntimeError("Family construction stopped: " + float(xi).hex())
        log = folder / "EXPORT.log"
        with log.open("a") as out:
            rc = subprocess.run([sys.executable, str(HERE / "export_experimental_rows.py"),
                                 "--xi", str(xi)], cwd=HERE,
                                stdout=out, stderr=subprocess.STDOUT).returncode
        if rc:
            record.update(status="FAILED", returncode=rc, log=str(log.relative_to(HERE)))
            state["status"] = "BLOCKED"; atomic(STATE, state)
            raise RuntimeError("Family row export stopped: " + float(xi).hex())
        log = folder / "CONTRACTION.log"
        if not (folder / "CONTRACTIONS.pt").exists():
            with log.open("a") as out:
                rc = subprocess.run([sys.executable, str(HERE / "build_experimental_contractions.py"),
                                     "--xi", str(xi)], cwd=HERE.parent,
                                    stdout=out, stderr=subprocess.STDOUT).returncode
            if rc:
                record.update(status="FAILED", returncode=rc, log=str(log.relative_to(HERE)))
                state["status"] = "BLOCKED"; atomic(STATE, state)
                raise RuntimeError("Family numerical contractions stopped: " + float(xi).hex())
        record.update(status="COMPLETE", pairs=len(family["Q2_GeV2"]),
                      family_pipeline=str(pipeline.relative_to(HERE)),
                      row_export=str((folder / "ACCEPTED_EXPERIMENTAL_ROWS.npz").relative_to(HERE)),
                      contractions=str((folder / "CONTRACTIONS.pt").relative_to(HERE)))
        atomic(STATE, state)
        print(json.dumps({"family_complete": float(xi).hex(),
                          "completed": sum(r["status"] == "COMPLETE" for r in state["families"]),
                          "total": len(FAMILIES)}), flush=True)
    state["status"] = "ALL_FAMILIES_COMPLETE"
    atomic(STATE, state)


if __name__ == "__main__":
    main()
