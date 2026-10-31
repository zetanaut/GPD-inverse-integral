"""Measured first-case resource/science summary; dense-case ratios labelled proxies."""
from __future__ import annotations

import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/"runs/ideal_L0_Z1_s31"


def main():
    result=json.loads((OUT/"RESULT.json").read_text())
    scores=json.loads((OUT/"SCORES.json").read_text())
    optimization=result["optimization"]
    journal=[json.loads(line) for line in (OUT/"ITERATIONS.jsonl").read_text().splitlines()]
    complete_wall=(OUT/"RESULT.json").stat().st_mtime-(OUT/"RUN_LOCK.json").stat().st_mtime
    measured={"case":"ideal_L0_Z1_s31","status":result["status"],
              "total_wall_seconds_from_run_lock_to_result":complete_wall,
              "optimizer_seconds":optimization["wall_seconds_since_process_start"],
              "Adam400_seconds":optimization["stage_times"]["Adam400_seconds"],
              "restoration_seconds":optimization["stage_times"]["restoration_seconds"],
              "solver_seconds":optimization["stage_times"]["solver_seconds"],
              "postfit_validation_and_scoring_seconds":result["validation_and_scoring_seconds"],
              "peak_RSS_bytes":max(result["peak_RSS_bytes"],optimization["peak_RSS_bytes"]),
              "solver_iterations":optimization["solver_iterations"],
              "accepted_outer_steps":optimization["solver_counts"]["accepted_steps"],
              "rejected_trials":optimization["solver_counts"]["trial_rejections"],
              "solver_status":optimization["solver_status"],
              "oracle_timers_seconds":optimization["oracle_timers"],
              "binding_stats":optimization["binding_stats"],
              "linear_algebra_seconds_overlaps_subproblem_time":optimization["linear_algebra_seconds"],
              "iteration_seconds_last":journal[-1]["seconds_since_solver_start"] if journal else None}
    scorer=scores["field"];cff=scores["CFF"]
    science={"scored_checkpoint":result["scored_checkpoint"],
             "selected_stage":result["selected"]["stage"] if result["selected"] else None,
             "charge_relative_L2":scorer["charge_relative_L2"],
             "charge_absolute_RMS":scorer["charge_absolute_RMS"],
             "fit_CFF_truth_error":cff["ideal_L0"]["truth_error"],
             "holdout_CFF_truth_error":cff["surface_check"]["truth_error"],
             "all_candidate_outcomes":[{"stage":x["stage"],"J":x["J"],"pass":x["pass"],
                                        "constraint_max":x["constraint_max"],
                                        "moments_pass":x["moments"]["pass"],
                                        "full_operator_pass":x["full_operator"]["pass"]}
                                       for x in result["validations"]]}
    summary={"measured":measured,"coarse_science":science,
             "denser_lattice_estimates":{"kind":"row-count proxies only, not measured runtime predictions",
               "L0_positions":45,"L1_positions":225,"L2_positions":1377,
               "L1_to_L0_row_ratio":5.,"L2_to_L0_row_ratio":1377/45,
               "warning":"active constraint count, dense linear algebra, trials and validation need not scale linearly"}}
    (OUT/"RESOURCE_AND_SCIENCE.json").write_text(json.dumps(summary,indent=2)+"\n")
    label="selected" if result["selected"] else "diagnostic, unselected"
    lines=["# Z1 / ideal L0 / seed31: one complete finite-budget case","",
           f"Status: **{result['status']}**. The scored checkpoint is {label}; the 400-step Mac timing checkpoint was never used.","",
           "## Truth/reconstruction","",
           "See `plots/truth_reconstruction.png` for x, xi and t slices, `plots/cff_fit_and_holdout.png` for complex CFFs, and `plots/iteration_history.png` for finite-budget behavior.","",
           f"Charge-H+ relative L2 on the common field mesh: all {scorer['charge_relative_L2']['all']:.6g}, ERBL {scorer['charge_relative_L2']['ERBL']:.6g}, DGLAP {scorer['charge_relative_L2']['DGLAP']:.6g}.",
           f"Fit CFF Re/Im relative L2: {cff['ideal_L0']['truth_error']['relative_L2']}. Disjoint-surface Re/Im relative L2: {cff['surface_check']['truth_error']['relative_L2']}.","",
           "## Measured Mac resources","",
           f"Wall time from run lock to completed result: {complete_wall:.1f} s. Peak RSS: {measured['peak_RSS_bytes']/2**30:.2f} GiB.",
           f"Adam {measured['Adam400_seconds']:.1f} s; restoration {measured['restoration_seconds']:.1f} s; solver {measured['solver_seconds']:.1f} s; post-fit validation and scoring {measured['postfit_validation_and_scoring_seconds']:.1f} s.",
           f"Solver: {measured['solver_iterations']} iterations, {measured['accepted_outer_steps']} accepted, {measured['rejected_trials']} rejected trials; termination `{measured['solver_status']}`.",
           "Oracle and subproblem sub-timers are in `RESOURCE_AND_SCIENCE.json`; some are nested within solver time and must not be added to it.","",
           "## Denser lattices","",
           "L1 has 225 positions (5 times L0); L2 has 1,377 (30.6 times L0). These are row-count proxies, **not measured time estimates**. Active constraints, linear algebra and trial counts can change nonlinearly. No L1 or L2 fit ran.","",
           "All candidate failures and numerical flags are retained in `RESULT.json`, candidate `CHECK.json` files, and the full operator row reports. One seed and one finite lattice do not establish continuum closure or an uncertainty interval.",""]
    (OUT/"REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"status":result["status"],"wall_seconds":complete_wall,
                      "peak_RSS_GiB":measured["peak_RSS_bytes"]/2**30}))


if __name__=="__main__":main()
