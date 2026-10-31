"""Resume a bounded output48 timeout from its saved source-keyed rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import argparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NLO = ROOT / "GPD_NLO_completion_package"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, data):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def run(xi=None):
    batch_file = HERE / "results/EXPERIMENTAL_BATCH.json"
    batch = json.loads(batch_file.read_text())
    failed_batch = [item for item in batch["families"] if item["status"] == "FAILED"]
    if len(failed_batch) != 1:
        raise RuntimeError("Expected exactly one failed family")
    xi = float(failed_batch[0]["xi"]) if xi is None else float(xi)
    token = hashlib.sha256(xi.hex().encode()).hexdigest()[:16]
    folder = HERE / "families" / token
    pipeline_file = folder / "PIPELINE.json"
    pipeline = json.loads(pipeline_file.read_text())
    if pipeline["xi_hex"] != xi.hex() or pipeline["status"] != "BLOCKED":
        raise RuntimeError("Unexpected blocked family identity/status")
    failed = pipeline["steps"][-1]
    log = folder / failed["log"]
    if (failed["name"] != "output48" or failed["returncode"] != 1 or
            sha(log) != failed["log_sha256"] or
            "TimeoutError: bounded output refinement; rows retained" not in log.read_text()):
        raise RuntimeError("Failure is not the known bounded output48 timeout")
    base_line = next(x for x in (folder / "base.log").read_text().splitlines() if x.startswith("COMPLETE "))
    base_key = Path(base_line[9:]).name
    found = []
    for progress in (NLO / "work/operators").glob("*/PROGRESS.json"):
        key = json.loads((progress.parent / "KEY.json").read_text())
        if (float(key.get("basis", {}).get("xi", -1)).hex() == xi.hex() and
                key.get("parent_key") == base_key and key.get("special_output_n") == 48 and
                key.get("cap_seconds") == 1200.0):
            found.append(progress)
    if len(found) != 1:
        raise RuntimeError("Expected exactly one source-keyed output48 row cache")
    progress = json.loads(found[0].read_text())
    rows = len(progress["rows"])
    if not 0 < rows < 258 or len(list(found[0].parent.glob("row_*.npz"))) != rows:
        raise RuntimeError("Saved output48 rows are incomplete or inconsistent")
    archived = folder / f"output48.timeout.{len(pipeline.get('failed_attempts', [])) + 1}.log"
    if archived.exists():
        raise RuntimeError("Timeout evidence already archived")
    log.rename(archived)
    pipeline.setdefault("failed_attempts", []).append({
        "stage": "output48", "reason": "Source script 1200-second wall cap; rows retained",
        "original_step": failed, "archived_log": archived.name, "archived_log_sha256": sha(archived),
        "row_cache": str(found[0].parent.relative_to(ROOT)), "row_cache_key": progress["key"],
        "rows_saved": rows, "rows_total": 258,
        "retry": "Same source, parent, special=48 and cap=1200; exact key/rows reused; fresh wall window",
    })
    pipeline["steps"].pop()
    pipeline["status"] = "IN_PROGRESS"
    atomic(pipeline_file, pipeline)
    record = next(x for x in batch["families"] if x["xi_hex"] == xi.hex())
    if batch["status"] != "BLOCKED" or record["status"] != "FAILED":
        raise RuntimeError("Batch state changed during bounded recovery")
    record.setdefault("failed_attempts", []).append({k: record[k] for k in ("status", "returncode", "log")})
    record["status"] = "RUNNING"
    record.pop("returncode", None)
    record.pop("log", None)
    batch["status"] = "RUNNING"
    atomic(batch_file, batch)
    return {"status": "READY_TO_RESUME", "xi_hex": xi.hex(), "rows_reused": rows,
            "row_cache_key": progress["key"], "timeout_log": str(archived.relative_to(ROOT))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi", type=float)
    args = parser.parse_args()
    print(json.dumps(run(args.xi), indent=2), flush=True)
