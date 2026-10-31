"""Finish only the already running coarse case after its optimizer exits.

This watcher never starts or resumes a fit. It executes read-only diagnostics,
plots, and compact packaging only after the optimizer completion marker exists.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "runs/ideal_L0_Z1_s31"
STATUS = OUT / "UNATTENDED_STATUS.json"
OPTIMIZER_PID = 3482
ZIP = ROOT / "review_three_stage_coarse_case.zip"


def write(status, **detail):
    record = {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), **detail}
    temp = STATUS.with_suffix(".json.partial")
    temp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temp, STATUS)
    print(json.dumps(record), flush=True)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Some sandboxed launches cannot inspect even a same-user process.
        # The completion marker remains the authority; preserve the watcher.
        return True


def run(name, filename):
    log = OUT / f"{name}.log"
    write("RUNNING_" + name, log=str(log.relative_to(ROOT)))
    with log.open("a") as stream:
        completed = subprocess.run([sys.executable, str(HERE / filename)],
                                   cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   check=False)
    if completed.returncode:
        write("FAILED_" + name, returncode=completed.returncode,
              log=str(log.relative_to(ROOT)))
        raise RuntimeError(f"{name} failed; see {log}")


def package():
    include = [
        HERE / "run_coarse_case.py", HERE / "finalize_coarse_case.py",
        HERE / "plot_coarse_case.py", HERE / "summarize_coarse_case.py",
        HERE / "unattended_coarse_case.py", HERE / "score_checkpoint.py",
        HERE / "closure/solver_binding.py", HERE / "closure/numerical_domain.py",
        HERE / "closure/numerical_cache.py", HERE / "closure/problem.py",
        HERE / "closure/metrics.py", HERE / "closure/geometry.py",
        HERE / "prepared/STUDY.json", HERE / "prepared/FIT_CONFIG.json",
        HERE / "prepared/GEOMETRY.json",
        HERE / "prepared/observed/ideal_L0_Z1.npz",
        HERE / "readiness/EXECUTION_LOCK.json",
        HERE / "readiness/LOCAL_BINDING_CHECK.json",
        HERE / "inputs/accepted_experimental_banks/index.json",
        HERE / "scale_rows/index.json",
        HERE / "numerical_contractions/TASK_ideal_L0_Z1_s31.json",
        HERE / "runs/ideal_L0_Z1_s31_RUN.log",
    ]
    include += [p for p in OUT.rglob("*") if p.is_file() and not p.name.endswith(".partial")]
    temp = ZIP.with_suffix(".zip.partial")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(include)):
            archive.write(path, path.relative_to(ROOT))
    os.replace(temp, ZIP)
    write("COMPLETE", archive=str(ZIP.relative_to(ROOT)), archive_bytes=ZIP.stat().st_size)


def main():
    if ZIP.exists() and (OUT / "RESULT.json").exists():
        write("COMPLETE", archive=str(ZIP.relative_to(ROOT)), archive_bytes=ZIP.stat().st_size)
        return
    while not (OUT / "OPTIMIZER_COMPLETE.json").exists():
        journal = OUT / "ITERATIONS.jsonl"
        lines = journal.read_text().splitlines() if journal.exists() else []
        latest = json.loads(lines[-1]) if lines else None
        write("WAITING_FOR_OPTIMIZER", optimizer_pid=OPTIMIZER_PID,
              latest_iteration=latest["iteration"] if latest else 0,
              latest_J=latest["J"] if latest else None,
              last_full_checkpoint=max((n for n in (0, 20, 40, 60)
                                        if (OUT / f"STAGE{n}.pt").exists()), default=None))
        if not alive(OPTIMIZER_PID):
            write("OPTIMIZER_EXITED_WITHOUT_COMPLETION", journal=str((OUT / "ITERATIONS.jsonl").relative_to(ROOT)),
                  log=str((HERE / "runs/ideal_L0_Z1_s31_RUN.log").relative_to(ROOT)))
            return
        time.sleep(30)
    run("FINALIZE", "finalize_coarse_case.py")
    run("PLOTS", "plot_coarse_case.py")
    run("SUMMARY", "summarize_coarse_case.py")
    package()


if __name__ == "__main__":
    main()
