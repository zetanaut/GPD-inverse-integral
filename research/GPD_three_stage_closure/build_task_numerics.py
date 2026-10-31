"""Sequentially extend numerical contractions at one registered task's exact scales."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
from closure.numerical_cache import build_one, digest
from closure.solver_binding import task_by_id


def run(task_id):
    _, task = task_by_id(task_id)
    observations = ROOT / "prepared/observed" / f"{task['branch']}_{task['truth']}.npz"
    with np.load(observations, allow_pickle=False) as z:
        pairs = list(dict.fromkeys((float(x), float(q)) for x, q in zip(z["xi"], z["Q2"])))
    out = ROOT / "numerical_contractions" / f"TASK_{task_id}.json"
    completed = []
    if out.exists():
        prior = json.loads(out.read_text())
        if prior["task"] != task_id or prior["observations_sha256"] != digest(observations):
            raise RuntimeError("Existing task numerical index changed")
        if len(prior["pairs"]) != len(pairs):
            raise RuntimeError("Existing task numerical index has the wrong pair count")
        for item, (xi, q2) in zip(prior["pairs"], pairs):
            path = ROOT.parent / item["path"]
            if (item["xi_hex"], item["Q2_hex"]) != (xi.hex(), q2.hex()) or not path.is_file() or digest(path) != item["sha256"]:
                raise RuntimeError("Existing exact numerical contraction changed")
        return prior
    for xi, q2 in pairs:
        begin = time.monotonic()
        item = build_one(xi, q2)
        path = Path(item["path"])
        completed.append({"xi_hex": xi.hex(), "Q2_hex": q2.hex(),
                          "status": item["status"],
                          "path": str(path.relative_to(ROOT.parent)),
                          "sha256": digest(path),
                          "seconds": time.monotonic() - begin})
        print(json.dumps(completed[-1]), flush=True)
    record = {"task": task_id, "observations_sha256": digest(observations),
              "pairs": completed, "status": "ALL_EXACT_NUMERICAL_CONTRACTIONS_READY"}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.partial")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, out)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    print(json.dumps({"task": run(args.task)["task"], "status": "READY"}))
