"""Bind experimental rows to exact local NLO cache identities, without substitution."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from gpd_reh.bank import PHYSICS, read_index
from gpd_reh.inputs import coordinate_key

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OPERATORS = ROOT / "GPD_NLO_completion_package/work/operators"
RESULTS = HERE / "results"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def records(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def validate_hit(directory, entry):
    sidecar = json.loads((directory / entry["validation_file"]).read_text())
    meta = sidecar["source_metadata"]
    if meta["physics"] != "fixed nf3 MSbar NLO" or meta["Q02"] != 2.0 or \
       meta["alpha0"] != PHYSICS["alpha0"] or meta["running"] != "two-loop" or \
       meta["input_channels"] != PHYSICS["input_channels"] or \
       meta["output_components"] != PHYSICS["output_components"]:
        raise ValueError("Cached row physics/convention changed")
    source_files = {
        "ordered_evolution_sha256": ROOT / "GPD_NLO_completion_package/src/ordered_evolution.py",
        "forward_sha256": ROOT / "GPD_NLO_completion_package/candidate/forward.py",
        "hard_sha256": ROOT / "GPD_NLO_completion_package/frozen/python/nlo_hard.py",
        "geometry_validation_sha256": ROOT / "completion_runs/NLO_final_geometry/source/geometry_validation.py",
    }
    for name, path in source_files.items():
        if meta["source"][name] != sha(path):
            raise ValueError("Cached row source hash changed: " + name)
    pair = sidecar["records"]
    if set(pair) != {"weak10", "strong10"}:
        raise ValueError("Candidate/reference pair missing")
    for method, rec in pair.items():
        if float(rec["xi"]).hex() != float(entry["xi"]).hex() or \
           float(rec["Q2"]).hex() != float(entry["Q2"]).hex() or rec["method"] != method:
            raise ValueError("Cached pair coordinate/method changed")
        operator = ROOT / "GPD_NLO_completion_package" / rec["directory"]
        if sha(operator / "KEY.json") != rec["operator_key_sha256"] or \
           sha(operator / "operators.npz") != rec["operator_sha256"]:
            raise ValueError("Cached pair discretization changed")
        token = hashlib.sha256(float(rec["xi"]).hex().encode()).hexdigest()[:16]
        family = HERE / "families" / token
        state = json.loads((family / "PIPELINE.json").read_text())
        if (state["status"] != "FAMILY_VALIDATED_WITH_CONTINUOUS_DIAGNOSTICS" or
                state["xi_hex"] != float(rec["xi"]).hex() or
                rec["directory"] != state["directories"][method]):
            raise ValueError("Cached family source/coordinate binding changed")
        validations = {v["field"]: v for v in state["validation"] if float(v["Q2"]).hex() == float(rec["Q2"]).hex()}
        if set(validations) != {"smooth", "Dterm_added"}:
            raise ValueError("Cached family validation missing")
        for field, hash_key in (("smooth", "smooth_validation_sha256"),
                                ("Dterm_added", "Dterm_validation_sha256")):
            info = validations[field]
            if sha(family / info["file"]) != info["sha256"] or info["sha256"] != rec[hash_key]:
                raise ValueError("Cached validation diagnostic changed")


def main():
    source = HERE / "inputs/final_summary_by_DNN.csv"
    contract = json.loads((HERE / "CENTRAL_CONTRACT.json").read_text())
    source_hash = sha(source)
    if source_hash != contract["input"]["sha256"]:
        raise ValueError("Settled experimental source changed")
    requests = records(HERE / "prepared/map_requests.csv")
    req_by_id = {r["map_id"]: r for r in requests}
    if len(requests) != 102 or len(req_by_id) != 102:
        raise ValueError("Exact map roster changed")
    bank_dir = HERE / "banks"
    bank = read_index(bank_dir)
    bank_by_key = {coordinate_key(e["xi"], e["Q2"]): e for e in bank["entries"]}
    operator_by_xi = defaultdict(list)
    key_count = 0
    for path in sorted(ROOT.rglob("KEY.json")):
        key = json.loads(path.read_text())
        xi = key.get("basis", key).get("xi")
        if xi is None:
            continue
        key_count += 1
        matrix = path.parent / "operators.npz"
        result = path.parent / "RESULT.json"
        if not matrix.exists() or not result.exists():
            continue
        outcome = json.loads(result.read_text())
        if not outcome.get("finite", False):
            continue
        if outcome.get("operator_sha256") and outcome["operator_sha256"] != sha(matrix):
            raise ValueError("Changed cached operator: " + str(matrix))
        operator_by_xi[float(xi).hex()].append(path)
    rows = records(HERE / "prepared/source_rows_all.csv")
    manifest = []
    for row in rows:
        selected = row["paper_cut_pass"] == "1"
        request = req_by_id[row["map_id"]] if selected else None
        if selected and (float(row["xi"]).hex() != request["xi_hex"] or
                         float(row["Q2_GeV2"]).hex() != request["Q2_hex"]):
            raise ValueError("Row/map coordinate mismatch: " + row["row_id"])
        key = coordinate_key(row["xi"], row["Q2_GeV2"]) if selected else None
        hit = bank_by_key.get(key) if selected else None
        smooth = Dterm = smooth_sha = Dterm_sha = ""
        if hit:
            validate_hit(bank_dir, hit)
            token = hashlib.sha256(float(row["xi"]).hex().encode()).hexdigest()[:16]
            state = json.loads((HERE / "families" / token / "PIPELINE.json").read_text())
            for item in state["validation"]:
                if float(item["Q2"]).hex() == float(row["Q2_GeV2"]).hex():
                    path = str((Path("families") / token / item["file"]))
                    if item["field"] == "smooth": smooth, smooth_sha = path, item["sha256"]
                    if item["field"] == "Dterm_added": Dterm, Dterm_sha = path, item["sha256"]
        status = ("EXCLUDED_BY_SOURCE_CUT" if not selected else
                  "EXACT_VALIDATED_BANK_HIT" if hit else
                  "EXACT_BANK_MISSING_SAME_XI_OPERATOR_UNVERIFIED" if operator_by_xi.get(float(row["xi"]).hex()) else
                  "EXACT_BANK_AND_GENERATOR_MISSING")
        manifest.append({
            **row,
            "source_file_sha256": source_hash,
            "xi_hex": float(row["xi"]).hex(),
            "Q2_hex": float(row["Q2_GeV2"]).hex(),
            "map_direction": request["direction"] if request else "",
            "cache_status": status,
            "bank_file": hit["file"] if hit else "",
            "bank_sha256": hit["sha256"] if hit else "",
            "validation_file": hit["validation_file"] if hit else "",
            "validation_sha256": hit["validation_sha256"] if hit else "",
            "smooth_validation_diagnostics": smooth,
            "smooth_validation_sha256": smooth_sha,
            "Dterm_validation_diagnostics": Dterm,
            "Dterm_validation_sha256": Dterm_sha,
            "source_key": hit["source_key"] if hit else "",
            "same_xi_operator_key_files_unverified": ";".join(str(p.relative_to(ROOT)) for p in operator_by_xi.get(float(row["xi"]).hex(), [])) if selected else "",
        })
    if len(manifest) != 189 or sum(r["paper_cut_pass"] == "1" for r in manifest) != 160:
        raise ValueError("Experimental row roster changed")
    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / "EXPERIMENTAL_ROW_MAP_MANIFEST.csv"
    with dest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(manifest)
    requested_keys = {coordinate_key(r["xi"], r["Q2_GeV2"]) for r in requests}
    matched = requested_keys & bank_by_key.keys()
    xi_requested = {r["xi_hex"] for r in requests}
    summary = {
        "status": "READY_FOR_CENTRAL_FIT" if len(matched) == 102 else "EXACT_MAPS_REQUIRED",
        "experimental_source_sha256": source_hash,
        "experimental_rows": 189,
        "selected_rows": 160,
        "excluded_rows_preserved": 29,
        "required_exact_pairs": 102,
        "exact_validated_bank_hits": len(matched),
        "missing_exact_pairs": 102 - len(matched),
        "required_exact_xi_families": len(xi_requested),
        "same_xi_operator_families_unverified": len(xi_requested & operator_by_xi.keys()),
        "operator_key_files_scanned": key_count,
        "cached_operator_xi_hex": sorted(operator_by_xi),
        "physics_contract": PHYSICS,
        "cache_match_rule": "Only validated bank pairs count as hits: exact binary xi/Q2, row hash, source key, NLO MSbar nf3 two-loop alpha0/Q02, channels and numerical provenance. Same-xi operator keys are inventoried without a reuse claim; source-keyed builders verify the full recipe.",
        "row_manifest": str(dest.relative_to(HERE)),
        "row_manifest_sha256": sha(dest),
    }
    (RESULTS / "EXACT_CACHE_AUDIT.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("status", "exact_validated_bank_hits", "missing_exact_pairs", "same_xi_operator_families_unverified", "operator_key_files_scanned")}))


if __name__ == "__main__":
    main()
