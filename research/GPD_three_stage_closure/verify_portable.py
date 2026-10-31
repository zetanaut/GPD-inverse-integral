"""Verify copied portable inputs before any destination fitting."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from closure.bank import RowBank


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    manifest_path = HERE / "readiness/PORTABLE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    lock = json.loads((HERE / "readiness/EXECUTION_LOCK.json").read_text())
    if digest(manifest_path) != lock["portable_manifest_sha256"]:
        raise RuntimeError("Execution lock and portable manifest differ")
    bad = []
    for item in manifest["included_files"]:
        path = ROOT / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            bad.append(item["path"])
    if bad:
        raise RuntimeError(f"{len(bad)} portable files missing or changed: {bad[:10]}")
    for name, expected in lock["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("Frozen binding source changed: " + name)
    bank = RowBank(HERE / "inputs/accepted_experimental_banks", HERE / "scale_rows")
    if len(bank.records) != 319:
        raise RuntimeError("Exact-row bank count changed")
    observed = list((HERE / "prepared/observed").glob("*.npz"))
    if len(observed) != 12:
        raise RuntimeError("Observed case count changed")
    return {"status": "PORTABLE_HASHES_AND_EXACT_ROWS_VERIFIED", "files": len(manifest["included_files"]),
            "families": len(manifest["basis_generator_families"]), "row_pairs": len(bank.records),
            "observed_cases": len(observed)}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
