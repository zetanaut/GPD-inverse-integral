"""Finish the bounded central review after the sequential map batch completes."""
from __future__ import annotations

import json
import fcntl
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
STAGES = ["audit_local_caches.py", "experimental_optimizer.py", "diagnose_candidates.py", "package_review.py"]


def run():
    with (HERE / "results/FINISH.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        archive = HERE.parent / "review_experimental_ReH_central.zip"
        if archive.exists():
            with ZipFile(archive) as z:
                if z.testzip() is not None:
                    raise RuntimeError("Completed review archive failed ZIP integrity check")
            print(archive, flush=True)
            return
        batch = json.loads((HERE / "results/EXPERIMENTAL_BATCH.json").read_text())
        if batch["status"] != "ALL_FAMILIES_COMPLETE":
            raise RuntimeError("The sequential exact-family batch has not completed")
        for script in STAGES:
            log = HERE / "results" / (Path(script).stem.upper() + ".log")
            with log.open("a") as output:
                status = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE.parent,
                                        stdout=output, stderr=subprocess.STDOUT).returncode
            if status:
                raise RuntimeError(f"Review stage {script} stopped with status {status}; see {log}")
        print(archive, flush=True)


if __name__ == "__main__":
    run()
