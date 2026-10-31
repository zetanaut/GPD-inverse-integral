"""Continue automatically after the already-running batch's next verified wall-cap stop."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    status_file = HERE / "results/AUTO_FINISH_STATUS.json"
    batch_file = HERE / "results/EXPERIMENTAL_BATCH.json"
    while True:
        status = json.loads(status_file.read_text())["status"]
        batch = json.loads(batch_file.read_text())["status"]
        if status == "COMPLETE":
            print("Central review completed before watcher intervention", flush=True)
            return 0
        if status == "REVIEW_STAGE_BLOCKED":
            print("Review stage blocked; preserved logs require inspection", flush=True)
            return 1
        if status == "MAP_BATCH_BLOCKED" and batch == "BLOCKED":
            with (HERE / "results/POWER_HANDOFF_AUTO_WATCH.log").open("a") as output:
                subprocess.Popen([sys.executable, str(HERE / "write_power_handoff.py"), "--watch"],
                                 cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            with (HERE / "results/AUTO_RECOVERY_CONTINUATION.log").open("a") as output:
                code = subprocess.run([sys.executable, str(HERE / "resume_after_power.py")],
                                      cwd=ROOT, stdout=output, stderr=subprocess.STDOUT).returncode
            return code
        time.sleep(45)


if __name__ == "__main__":
    sys.exit(main())
