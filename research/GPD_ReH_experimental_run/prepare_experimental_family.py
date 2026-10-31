"""Sequential, resumable construction for one exact experimental xi family."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NLO = ROOT / "GPD_NLO_completion_package"
FG = ROOT / "completion_runs/NLO_final_geometry"
sys.path.insert(0, str(FG / "source"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def path_from_log(output):
    found = [Path(line[9:]) for line in output.splitlines() if line.startswith("COMPLETE ")]
    if len(found) != 1 or not (found[0] / "RESULT.json").exists():
        raise RuntimeError("No complete operator result in stage log")
    return found[0]


def build(xi, q2_values, adopt_base=False):
    ident = hashlib.sha256(float(xi).hex().encode()).hexdigest()[:16]
    out = HERE / "families" / ident
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "PIPELINE.json"
    state = json.loads(marker.read_text()) if marker.exists() else {
        "xi": xi, "xi_hex": float(xi).hex(), "Q2": q2_values,
        "status": "IN_PROGRESS", "steps": [], "validation": [],
    }
    if state["xi_hex"] != float(xi).hex() or state["Q2"] != q2_values:
        raise ValueError("Existing family plan changed")

    def save():
        atomic(marker, state)

    def run(name, script, args, cap=1200, adopt_log=None):
        prior = next((s for s in state["steps"] if s["name"] == name), None)
        if prior:
            if prior["returncode"] != 0:
                raise RuntimeError("Preserved failed stage requires review: " + name)
            log = out / prior["log"]
            if sha(log) != prior["log_sha256"]:
                raise RuntimeError("Completed stage log changed: " + name)
            return log.read_text()
        log = out / (name + ".log")
        if log.exists() and not adopt_log:
            # A power cut can leave a completed child log before PIPELINE.json
            # receives its atomic step record. Recover that result when its
            # source-keyed operator archive is already complete.
            text = log.read_text()
            completed = None
            if name in ("base", "inner72", "output48"):
                try:
                    completed = path_from_log(text)
                except RuntimeError:
                    pass
            elif not name.startswith("source_"):
                lines = text.strip().splitlines()
                if lines:
                    candidate = Path(lines[-1])
                    if (candidate / "operators.npz").exists() and (candidate / "RESULT.json").exists():
                        completed = candidate
            if completed is not None:
                key = json.loads((completed / "KEY.json").read_text())
                if float(key["basis"]["xi"]).hex() != float(xi).hex():
                    raise ValueError("Recovered operator has wrong exact xi")
                state["steps"].append({"name": name, "command": ["recovered completed pre-interruption log"],
                                       "log": log.name, "log_sha256": sha(log), "returncode": 0,
                                       "seconds": json.loads((completed / "RESULT.json").read_text()).get("seconds", 0),
                                       "peak_child_RSS_bytes_macos": 0})
                save()
                return text
            attempts = state.setdefault("interrupted_attempts", [])
            prior_path = out / f"{name}.interrupted.{len(attempts) + 1}.log"
            log.rename(prior_path)
            attempts.append({"name": name, "log": prior_path.name, "sha256": sha(prior_path),
                             "reason": "No complete stage record after interrupted execution; source-keyed row caches retained"})
            save()
        if adopt_log:
            text = Path(adopt_log).read_text()
            completed = path_from_log(text)
            key = json.loads((completed / "KEY.json").read_text())
            if float(key["basis"]["xi"]).hex() != float(xi).hex():
                raise ValueError("Adopted base has wrong exact xi")
            shutil.copy2(adopt_log, log)
            elapsed = json.loads((completed / "RESULT.json").read_text())["seconds"]
            command = ["adopted independently measured first-family base"]
        else:
            command = [sys.executable, str(script), *map(str, args)]
            start = time.monotonic()
            with log.open("x") as f:
                try:
                    rc = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                        timeout=cap + 120).returncode
                except subprocess.TimeoutExpired:
                    rc = "RESOURCE_TIMEOUT"
            elapsed = time.monotonic() - start
            if rc != 0:
                state["steps"].append({"name": name, "command": command, "log": log.name,
                                       "log_sha256": sha(log), "returncode": rc, "seconds": elapsed})
                state["status"] = "BLOCKED"; save()
                raise RuntimeError("Stage failed; original log preserved: " + name)
            text = log.read_text()
        entry = {"name": name, "command": command, "log": log.name,
                 "log_sha256": sha(log), "returncode": 0, "seconds": elapsed,
                 "peak_child_RSS_bytes_macos": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
        state["steps"].append(entry); save()
        return text

    def rel(path):
        return str(path.relative_to(NLO))

    def derived(name, script, parent, degree):
        output = run(name, NLO / "candidate" / script, [parent, "--degree", degree], 120)
        path = Path(output.strip().splitlines()[-1])
        if not (path / "operators.npz").exists():
            raise RuntimeError("Derived operator missing: " + name)
        return path

    def corrected(name, parent):
        output = run(name, NLO / "candidate/correct_qg_rows.py", [parent], 120)
        path = Path(output.strip().splitlines()[-1])
        if not (path / "operators.npz").exists():
            raise RuntimeError("Corrected operator missing: " + name)
        return path

    args = ["--xi", xi, "--degree", 10, "--layout", "log3", "--q", 48,
            "--outer", 18, "--cap", 1200, "--bridge", "mp"]
    base_log = HERE / "results/FIRST_FAMILY_BASE.log" if adopt_base else None
    raw = path_from_log(run("base", HERE / "assemble_experimental_family.py", args,
                            adopt_log=base_log))
    base = corrected("correct_base", raw)
    args72 = ["--xi", xi, "--degree", 10, "--layout", "log3", "--q", 72,
              "--outer", 18, "--cap", 1200, "--bridge", "mp"]
    innerraw = path_from_log(run("inner72", HERE / "assemble_experimental_family.py", args72))
    inner = corrected("correct_inner72", innerraw)
    outputraw = path_from_log(run("output48", NLO / "candidate/refine_output.py",
                                  [raw, "--special", 48, "--cap", 1200]))
    output = corrected("correct_output48", outputraw)
    dirs = {"weak10": rel(base)}
    for degree in (6, 8):
        dirs["weak" + str(degree)] = rel(derived("weak" + str(degree),
                                                "restrict_nested.py", base, degree))
    for degree in (6, 8, 10):
        dirs["strong" + str(degree)] = rel(derived("strong" + str(degree),
                                                  "collocation_reference.py", base, degree))
    for label, parent in (("inner72", inner), ("output48", output)):
        dirs[label + "_weak10"] = rel(parent)
        dirs[label + "_strong10"] = rel(derived(label + "_strong10",
                                                "collocation_reference.py", parent, 10))
    state["directories"] = dirs; save()
    for script in ("check_raw_moments.py", "check_physics.py", "check_saved_action.py"):
        run("source_" + script, NLO / "candidate" / script, [base], 120)
    import geometry_validation as gv
    point = {"xi": xi, "directories": dirs}
    for field in ("smooth", "Dterm_added"):
        for q2 in q2_values:
            name = f"{field}_Q2_{float(q2).hex()}.json"
            dest = out / name
            old = next((r for r in state["validation"] if r["field"] == field and r["Q2"] == q2), None)
            if old:
                if sha(dest) != old["sha256"]:
                    raise RuntimeError("Completed validation changed")
                continue
            if dest.exists():
                # The full JSON may have landed just before a power loss.
                try:
                    prior_result = json.loads(dest.read_text())
                except json.JSONDecodeError:
                    prior_result = None
                if prior_result is None:
                    attempts = state.setdefault("interrupted_attempts", [])
                    prior_path = out / f"{name}.interrupted.{len(attempts) + 1}.json"
                    dest.rename(prior_path)
                    attempts.append({"name": name, "file": prior_path.name, "sha256": sha(prior_path),
                                     "reason": "Incomplete validation JSON after interrupted execution"})
                    save()
                elif (float(prior_result["xi"]).hex() == float(xi).hex() and
                      float(prior_result["Q2"]).hex() == float(q2).hex() and
                      prior_result["field"] == field):
                    item = {"field": field, "Q2": q2, "file": name,
                            "sha256": sha(dest), "pass_legacy_guides": prior_result["pass"],
                            "gates": prior_result["gates"],
                            "CFF_estimated_total": prior_result["CFF_estimated_total"]}
                    state["validation"].append(item); save()
                    continue
                else:
                    raise RuntimeError("Unrecorded validation has wrong coordinate or field")
            result = gv.validate(point, q2, field, dest)
            item = {"field": field, "Q2": q2, "file": name,
                    "sha256": sha(dest), "pass_legacy_guides": result["pass"],
                    "gates": result["gates"], "CFF_estimated_total": result["CFF_estimated_total"]}
            state["validation"].append(item); save()
    state["status"] = "FAMILY_VALIDATED_WITH_CONTINUOUS_DIAGNOSTICS"
    state["operator_bytes"] = sum(p.stat().st_size for d in set(dirs.values())
                                  for p in (NLO / d).rglob("*") if p.is_file())
    save()
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi", type=float, required=True)
    parser.add_argument("--adopt-base", action="store_true")
    args = parser.parse_args()
    families = json.loads((HERE / "prepared/kernel_families.json").read_text())["families"]
    family = next((f for f in families if float(f["xi"]).hex() == float(args.xi).hex()), None)
    if family is None:
        raise ValueError("xi is not a requested experimental family")
    print(json.dumps(build(float(family["xi"]), family["Q2_GeV2"], args.adopt_base)), flush=True)
