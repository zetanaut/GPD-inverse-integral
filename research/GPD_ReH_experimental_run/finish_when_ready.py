"""Wait for the sequential batch, then finish the authorized central review."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATUS = HERE / "results/AUTO_FINISH_STATUS.json"


def write(value):
    tmp = STATUS.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(STATUS)


def main():
    write({"status": "WAITING_FOR_EXACT_MAPS", "started_unix": time.time()})
    while True:
        batch = json.loads((HERE / "results/EXPERIMENTAL_BATCH.json").read_text())
        if batch["status"] == "ALL_FAMILIES_COMPLETE":
            break
        if batch["status"] == "BLOCKED":
            write({"status": "MAP_BATCH_BLOCKED", "batch": batch, "time_unix": time.time()})
            return
        time.sleep(45)
    write({"status": "FINISHING_REVIEW", "time_unix": time.time()})
    code = subprocess.run([sys.executable, str(HERE / "finish_experimental_run.py")],
                          cwd=HERE.parent).returncode
    write({"status": "COMPLETE" if code == 0 else "REVIEW_STAGE_BLOCKED",
           "returncode": code, "time_unix": time.time(),
           "archive": str(HERE.parent / "review_experimental_ReH_central.zip") if code == 0 else None})


if __name__ == "__main__":
    main()
