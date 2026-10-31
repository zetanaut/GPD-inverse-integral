"""Keep a concise, atomic power-interruption handoff current during the batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TARGET = HERE / "results/POWER_INTERRUPTION_HANDOFF.md"


def current():
    batch = json.loads((HERE / "results/EXPERIMENTAL_BATCH.json").read_text())
    auto = json.loads((HERE / "results/AUTO_FINISH_STATUS.json").read_text())
    bank = json.loads((HERE / "banks/index.json").read_text())
    complete = sum(item["status"] == "COMPLETE" for item in batch["families"])
    pending = next((item for item in batch["families"] if item["status"] == "RUNNING"), None)
    stage = "none"
    recent = "none"
    row = ""
    recovery_note = "none"
    if pending:
        token = hashlib.sha256(pending["xi_hex"].encode()).hexdigest()[:16]
        folder = HERE / "families" / token
        logs = list(folder.glob("*.log"))
        if logs:
            log = max(logs, key=lambda p: p.stat().st_mtime)
            stage = log.name
            recent = str(log.relative_to(ROOT))
            lines = log.read_text(errors="replace").splitlines()
            row = next((line for line in reversed(lines) if line.startswith("ROW ")), "")
        pipeline = folder / "PIPELINE.json"
        if pipeline.exists():
            attempts = json.loads(pipeline.read_text()).get("failed_attempts", [])
            if attempts:
                last = attempts[-1]
                recovery_note = (f"{last['stage']}: {last['reason']}; "
                                 f"{last['rows_saved']}/{last['rows_total']} source-keyed rows retained; "
                                 f"evidence {last['archived_log']}")
    disk = shutil.disk_usage(ROOT)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Power interruption handoff — experimental central ReH", "",
        f"Updated: {now}", "",
        f"Batch status: **{batch['status']}**. Completed xi families: **{complete}/67**. "
        f"Exact bank pairs saved: **{len(bank['entries'])}/102**.",
        f"Automatic finisher: **{auto['status']}**. Review zip present: "
        f"**{(ROOT / 'review_experimental_ReH_central.zip').exists()}**.",
        f"Available project-volume storage: **{disk.free / 1024**3:.1f} GiB**.", "",
        "## Current family", "",
        f"Exact xi: `{pending['xi_hex'] if pending else 'none'}`. Last active stage log: `{stage}`.",
        f"Log path: `{recent}`.",
        f"Last saved row progress: `{row or 'none'}`.", "",
        f"Recovered bounded attempt: `{recovery_note}`.", "",
        "## After power returns", "",
        "From the project root, run:", "",
        "```sh",
        "python3 GPD_ReH_experimental_run/resume_after_power.py > GPD_ReH_experimental_run/results/RESUME_AFTER_POWER.log 2>&1",
        "```", "",
        "The restart command resumes the exact sequential family batch from source-keyed rows and "
        "completed PIPELINE stages, then runs the frozen two-start central fit, diagnostics, and zip packaging.",
        "It refuses to start a duplicate batch if a current family log was updated within 120 seconds. "
        "If that happens, let the live process continue or retry after it has stopped.", "",
        "Keep `GPD_NLO_completion_package/work/operators/`, `GPD_ReH_experimental_run/families/`, "
        "`banks/`, `prepared/`, and `results/` intact. An incomplete stage log is archived on restart; "
        "source-keyed row caches and completed operators are retained.", "",
        "If the batch state is `BLOCKED`, inspect `results/EXPERIMENTAL_BATCH.json` and the named "
        "family log before retrying. For the known `TimeoutError: bounded output refinement; rows retained`, "
        "the restart command archives the timeout log and reuses the exact row cache automatically. "
        "Other failures require inspection. "
        "The restart command deliberately does not erase failed evidence.", "",
        "## Review evidence", "",
        "- `results/EXPERIMENTAL_BATCH.json`: atomic family completion records.",
        "- `families/<sha256(xi.hex())[:16]>/PIPELINE.json`: atomic stage and validation records.",
        "- `banks/index.json`: accepted exact xi/Q² pairs and row hashes.",
        "- `results/EXACT_CACHE_AUDIT.json` and `EXPERIMENTAL_ROW_MAP_MANIFEST.csv`: "
        "last full audit, refreshed before the fit.",
        "- `results/BATCH_DRIVER.log` and `results/AUTO_FINISH_STATUS.json`: live progress and finisher state.",
        "- Settled source: `inputs/final_summary_by_DNN.csv`, SHA256 "
        "`8d7274db9dd7e4fcc3ab84c0230701242274e42868f9c3e66e3d61a1a6838696`.",
        "", "No source reselection, replicas, calibration, or additional GPD extraction are authorized.", "",
    ]
    value = "\n".join(lines)
    if not TARGET.exists() or TARGET.read_text() != value:
        tmp = TARGET.with_suffix(".partial.md")
        tmp.write_text(value)
        tmp.replace(TARGET)
    return {"updated": now, "families_complete": complete, "bank_pairs": len(bank["entries"]),
            "batch_status": batch["status"], "auto_status": auto["status"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        status = current()
        if not args.watch or status["auto_status"] in ("COMPLETE", "REVIEW_STAGE_BLOCKED"):
            print(json.dumps(status), flush=True)
            break
        time.sleep(45)
