"""Package the completed central review while keeping large native caches local."""
from __future__ import annotations

import difflib
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "review_experimental_ReH_central.zip"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prerequisites():
    batch = json.loads((HERE / "results/EXPERIMENTAL_BATCH.json").read_text())
    optimizer = json.loads((HERE / "results/central_optimization/OPTIMIZATION_COMPLETE.json").read_text())
    diagnostic = json.loads((HERE / "results/CENTRAL_DIAGNOSTICS.json").read_text())
    audit = json.loads((HERE / "results/EXACT_CACHE_AUDIT.json").read_text())
    if batch["status"] != "ALL_FAMILIES_COMPLETE" or len(batch["families"]) != 67:
        raise RuntimeError("Exact xi family batch incomplete")
    if audit["missing_exact_pairs"] or audit["exact_validated_bank_hits"] != 102:
        raise RuntimeError("Exact row bank audit incomplete")
    if optimizer["status"] != "optimization_complete_not_validated":
        raise RuntimeError("Both central starts incomplete")
    if diagnostic["status"] not in ("FINITE_CENTRAL_APPROXIMATION", "ABSTAIN_NONFINITE"):
        raise RuntimeError("Central diagnostics incomplete")
    return batch, optimizer, diagnostic, audit


def local_archives(batch):
    operators = {}
    contractions = {}
    for item in batch["families"]:
        state = json.loads((HERE / item["family_pipeline"]).read_text())
        for path in state["directories"].values():
            if path in operators:
                continue
            base = ROOT / "GPD_NLO_completion_package" / path
            operators[path] = {name: sha(base / name) for name in ("KEY.json", "operators.npz", "RESULT.json")}
        path = HERE / item["contractions"]
        contractions[item["xi_hex"]] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                                          "bytes": path.stat().st_size}
    return {"role": "Local numerical archives omitted from review zip", "operators": operators,
            "contractions": contractions}


def adapter_diff():
    original = ROOT / "completion_runs/NLO_final_geometry/source/assemble_missing.py"
    adapted = HERE / "assemble_experimental_family.py"
    a = original.read_text().splitlines(keepends=True)
    b = adapted.read_text().splitlines(keepends=True)
    parts = list(difflib.unified_diff(a, b, fromfile=str(original.relative_to(ROOT)),
                                     tofile=str(adapted.relative_to(ROOT))))
    for name in ("audit_local_caches.py", "prepare_experimental_family.py",
                 "export_experimental_rows.py", "build_experimental_contractions.py",
                 "run_experimental_batch.py", "experimental_domain.py",
                 "experimental_optimizer.py", "diagnose_candidates.py", "finish_experimental_run.py",
                 "finish_when_ready.py", "resume_after_power.py", "write_power_handoff.py",
                 "recover_output48_timeout.py", "watch_bounded_recovery.py",
                 "package_review.py"):
        path = HERE / name
        parts.extend(difflib.unified_diff([], path.read_text().splitlines(keepends=True),
                                          fromfile="/dev/null", tofile=str(path.relative_to(ROOT))))
    (HERE / "results/ADAPTER_DIFF.patch").write_text("".join(parts))


def report(batch, optimizer, diagnostic, audit, archive):
    family = [json.loads((HERE / x["family_pipeline"]).read_text()) for x in batch["families"]]
    total_wall = sum(sum(step["seconds"] for step in x["steps"]) for x in family)
    total_contractions = sum(__import__("torch").load(HERE / x["contractions"],
                                                   weights_only=True, map_location="cpu")["seconds"]
                             for x in batch["families"])
    source = json.loads((HERE / "prepared/INPUT_SUMMARY.json").read_text())
    details = {"source_sha256": source["source_sha256"], "selected_rows": source["n_selected"],
               "cut_rows_preserved": source["n_excluded"], "xi_families": len(family),
               "exact_map_pairs": audit["exact_validated_bank_hits"],
               "backward_pairs": source["n_backward_map_requests"],
               "native_family_wall_seconds_sum": total_wall,
               "contraction_wall_seconds_sum": total_contractions,
               "optimization_seconds": optimizer["optimization_seconds"],
               "diagnostics_seconds": diagnostic["seconds_diagnostics"],
               "diagnostic_status": diagnostic["status"],
               "selected": diagnostic["selected"],
               "measured_resource_plan": json.loads((HERE / "results/MEASURED_RESOURCE_PLAN.json").read_text()),
               "local_archives_manifest_sha256": sha(archive),
               "interpretation": "Conditional central ReH approximation. No calibrated uncertainty intervals or ImH observations."}
    write_json(HERE / "results/SCIENTIFIC_REVIEW_SUMMARY.json", details)
    summary = diagnostic["selected"]
    lines = ["# Experimental central ReH review", "",
             f"Source SHA256: `{source['source_sha256']}`. Selected rows: {source['n_selected']}; cut rows retained: {source['n_excluded']}.",
             f"Exact validated maps: {audit['exact_validated_bank_hits']} in {len(family)} xi families; {source['n_backward_map_requests']} below Q0².",
             f"Central diagnostics: **{diagnostic['status']}**."]
    if summary:
        lines += [f"Selected finite candidate: seed {summary['seed']}, stage {summary['stage']}, J={summary['J']:.12g}.",
                  f"Quality flags: {', '.join(summary['quality_flags']) if summary['quality_flags'] else 'none recorded'}."]
    lines += ["", "Diagnostics and every candidate remain in `results/central_diagnostics/`.",
              "Quoted source errors enter once as the ReH data sigma. ImH is an unfitted prediction.",
              "Big NLO operators and numerical contractions remain in the local project; their hashes are in `results/LOCAL_NUMERICAL_ARCHIVES.json`.",
              "", "Replay in this project environment:", "",
              "```sh", "python3 GPD_ReH_experimental_run/audit_local_caches.py",
              "python3 GPD_ReH_experimental_run/experimental_optimizer.py",
              "python3 GPD_ReH_experimental_run/diagnose_candidates.py", "```", ""]
    (HERE / "results/FINAL_REVIEW.md").write_text("\n".join(lines))


def include(path):
    rel = path.relative_to(HERE)
    if rel == Path("results/AUTO_FINISH_STATUS.json"):
        return False
    if any(part in ("__pycache__", ".pytest_cache") for part in rel.parts):
        return False
    if path.suffix in (".partial", ".pyc") or ".partial" in path.name or "failed_format" in path.name:
        return False
    if rel.parts[0] == "families":
        return path.suffix == ".json" and path.name != "CONTRACTIONS.json"
    return True


def package():
    if OUT.exists():
        raise FileExistsError("Preserve completed review archive")
    batch, optimizer, diagnostic, audit = prerequisites()
    archive = HERE / "results/LOCAL_NUMERICAL_ARCHIVES.json"
    write_json(archive, local_archives(batch))
    adapter_diff()
    report(batch, optimizer, diagnostic, audit, archive)
    temp = OUT.with_suffix(".partial.zip")
    if temp.exists():
        raise FileExistsError("Prior partial review archive requires inspection")
    with ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=6) as zipfile:
        for path in sorted(HERE.rglob("*")):
            if path.is_file() and include(path):
                zipfile.write(path, Path(HERE.name) / path.relative_to(HERE))
    temp.replace(OUT)
    return {"path": str(OUT), "sha256": sha(OUT), "bytes": OUT.stat().st_size}


if __name__ == "__main__":
    print(json.dumps(package(), indent=2), flush=True)
