"""One-command cache-preserving restart after an interrupted Mac session."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATUS = HERE / "results/AUTO_FINISH_STATUS.json"


def write_status(status, **fields):
    tmp = STATUS.with_suffix(".partial")
    tmp.write_text(json.dumps({"status": status, "time_unix": time.time(), **fields},
                              indent=2, sort_keys=True) + "\n")
    tmp.replace(STATUS)


def current_family_activity(batch):
    pending = next((item for item in batch["families"] if item["status"] == "RUNNING"), None)
    if pending is None:
        return None
    token = hashlib.sha256(pending["xi_hex"].encode()).hexdigest()[:16]
    folder = HERE / "families" / token
    logs = list(folder.glob("*.log"))
    recent = max((p.stat().st_mtime for p in logs), default=0)
    return {"xi_hex": pending["xi_hex"], "latest_log_age_seconds": time.time() - recent}


def run():
    if (ROOT / "review_experimental_ReH_central.zip").exists():
        return subprocess.run([sys.executable, str(HERE / "finish_experimental_run.py")], cwd=ROOT).returncode
    batch_path = HERE / "results/EXPERIMENTAL_BATCH.json"
    batch = json.loads(batch_path.read_text())
    activity = current_family_activity(batch)
    if batch["status"] == "RUNNING" and activity and activity["latest_log_age_seconds"] < 120:
        raise RuntimeError("Current family log was updated within 120 seconds; avoid a duplicate live batch: "
                           + json.dumps(activity))
    recoveries = 0
    while batch["status"] != "ALL_FAMILIES_COMPLETE":
        if batch["status"] == "BLOCKED":
            if recoveries >= 20:
                write_status("MAP_BATCH_BLOCKED", reason="Repeated bounded recoveries exceeded 20")
                raise RuntimeError("Too many bounded output48 retries; inspect saved evidence")
            recovery = subprocess.run([sys.executable, str(HERE / "recover_output48_timeout.py")],
                                      cwd=ROOT).returncode
            if recovery:
                write_status("MAP_BATCH_BLOCKED", reason="Not a verified output48 wall-cap timeout")
                raise RuntimeError("Preserved batch failure requires inspection before restart")
            recoveries += 1
            batch = json.loads(batch_path.read_text())
        write_status("RESUMED_EXACT_MAP_BATCH", current_family=current_family_activity(batch),
                     bounded_timeout_recoveries=recoveries)
        status = subprocess.run([sys.executable, str(HERE / "run_experimental_batch.py")], cwd=ROOT).returncode
        batch = json.loads(batch_path.read_text())
        if status and batch["status"] != "BLOCKED":
            write_status("MAP_BATCH_BLOCKED", returncode=status)
            raise RuntimeError("Sequential map batch stopped without a recoverable saved failure")
    write_status("FINISHING_REVIEW")
    status = subprocess.run([sys.executable, str(HERE / "finish_experimental_run.py")], cwd=ROOT).returncode
    if status:
        write_status("REVIEW_STAGE_BLOCKED", returncode=status)
        raise RuntimeError("Central fit/diagnostics/package stage stopped; inspect results logs")
    write_status("COMPLETE", archive=str(ROOT / "review_experimental_ReH_central.zip"))
    return status


if __name__ == "__main__":
    sys.exit(run())
