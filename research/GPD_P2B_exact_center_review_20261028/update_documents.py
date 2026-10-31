"""Refresh six editable coordination files; preserve prior state and frozen evidence."""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
REVIEW = OUT.name
SNAP = ROOT/'handoff_history/2026-10-28/before_P2B_exact_center_review'
FILES = ['README.md','HANDOFF.md','HANDOFF_STATUS.json','PLAN.md','WORKER_PROMPT.md','DOCUMENT_INDEX.md']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, obj):
    Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def replace_one(text, old, new):
    assert text.count(old)==1, old[:80]
    return text.replace(old,new,1)

def main():
    assert not SNAP.exists(), 'Do not overwrite a prior coordination snapshot'
    evidence=json.loads((OUT/'REVIEW_EVIDENCE.json').read_text())
    assert evidence['status']=='PASS'
    SNAP.mkdir(parents=True)
    before={}
    for name in FILES:
        shutil.copy2(ROOT/name,SNAP/name)
        before[name]=sha(ROOT/name)
    protected=[ROOT/'review_GPD_P2B_exact_center_evaluation_v1.zip',ROOT/'GPD_P2B_preparation_v3.zip',
               ROOT/'review_GPD_P2A_preservation_results_v1.zip',ROOT/'GPD_P2B_preparation_v3/MANIFEST.json',
               ROOT/'GPD_P2B_preparation_v3/DESIGN.json']
    fixed={str(p.relative_to(ROOT)):sha(p) for p in protected}
    report=f'[{"independent P2B review"}]({REVIEW}/REVIEW.txt)'
    nextlink=f'[next action]({REVIEW}/NEXT_ACTION.txt)'
    notice=(f'Updated 28 October 2026. **P2B exact-center preservation is complete and passes all four frozen gates.** '
            f'The {report} accepts the saved result; noisy development remains unreleased. '
            f'The returned CPU acceleration/qualification evidence is incomplete; recover the omitted existing records before completing the next execution package. '
            f'See {nextlink}. P2A remains `STOP_PRESERVATION_FAILED`.')
    design=('The [v1 design](GPD_P2B_preparation_v3/SCIENTIFIC_DESIGN.txt) retains twelve x points on [.02,.8], '
            'three stored channels at xi=t=0 and Q0^2=2 GeV^2, the frozen correlated synthetic covariance and '
            'J_joint=J_P1+chi2_PDF/160 with lambda_stab=0. These are compatible Z1 synthetic inputs, not a real proton PDF set. '
            'The original FiLM, NLO maps, finite moment conditions, permitted coefficients, original CFF sigmas and scientific budgets remain unchanged.')
    outcome=('The selected stage60 gives observed H+ error 0.469477%, absolute A/B errors 0.676725%/0.251647%, '
             'and PDF whitened RMS 0.000795771: all four gates pass. Exact-baseline errors are 0.920287%, '
             '0.484767% and 0.869098%. Wider-chart H+ error improves from 0.750171% to 0.535259%; '
             'the worst slice and all three wider-chart internal-channel errors worsen. This is a clean-input preservation success, '
             'not independent evidence of noisy accuracy or calibrated uncertainty.')
    provenance=('The review verifies all 112 payload hashes, 51 result-retention entries and 26 staged parent identities, '
                'and replays the baseline plus four P2B saved fields. Maximum selected-metric discrepancy is 4.57e-14. '
                'No full NLO/constraint/optimizer replay was performed. The CPU acceleration overlay, its existing parity/performance '
                'review and the parent readiness/submission records are hash-bound but omitted from this return. '
                'The reported full-model/derivative qualification and backend choice therefore remain unaudited locally.')
    costs=('Parent22081808 completed 400 Adam and 60 accepted outer updates (121 solver evaluations), then paused before finalization. '
           'Evaluation22117473 completed without additional optimization. Combined job elapsed is 10:03:25; parent batch peak RSS is 14.59 GiB. '
           'This was CPU execution with one field/rank thread and eight solver BLAS threads, no GPU. Norm subproblems account for 91.15% '
           'of recorded solver time. No matched speedup or complete prior-profiling cost is established by the packet.')
    nextpara=(f'Next, recover the omitted existing execution evidence and prepare the single conditional half-noise development case under '
              f'the unchanged design; see {nextlink}. Reuse precision160_Z1_a050_s31 and its baseline, retain original sigmas and fresh seed31, '
              'and archive the prescribed PCG64 seed2026102601 PDF draw only when that case is released. Advance only if observed-slice '
              'squared H+ error improves and neither absolute target error worsens beyond the frozen tolerance. '
              'No new fit or noise draw is released here. Do not repeat the completed exact-center jobs.')

    (ROOT/'README.md').write_text('# GPD reconstruction project\n\n'+notice+'\n\n'+outcome+'\n\n'+
        f'Start with the {report}, [numerical evidence]({REVIEW}/REVIEW_EVIDENCE.json), '
        f'[packet verification]({REVIEW}/PACKET_VERIFICATION.json) and '
        '[returned ZIP](review_GPD_P2B_exact_center_evaluation_v1.zip). The earlier v3 preparation remains frozen provenance; its pending-fit wording is historical.\n\n'+
        design+'\n\n'+provenance+'\n\n'+costs+'\n\n'+nextpara+'\n\n'+
        'P2A remains closed after failure of all three preservation gates; its independent-noise assessment remains stopped. '
        'P1 stays closed. Experimental continuations/remaining replicas are paused; manuscript and categorical architecture work remain deferred.\n\n'+
        f'Use [HANDOFF.md](HANDOFF.md), [PLAN.md](PLAN.md), [WORKER_PROMPT.md](WORKER_PROMPT.md), '
        f'[status](HANDOFF_STATUS.json) and [document index](DOCUMENT_INDEX.md). '
        f'[Prior coordination](handoff_history/2026-10-28/before_P2B_exact_center_review/) is preserved; '
        f'[update receipt]({REVIEW}/UPDATE_RECEIPT.json) records identities. Frozen scientific evidence remains unchanged.\n')

    text=(SNAP/'HANDOFF.md').read_text()
    text=re.sub(r'Current compute task:.*?\n',notice+'\n',text,count=1)
    text=replace_one(text,'Updated 26 October 2026. Project root:','Updated 28 October 2026. Project root:')
    text=re.sub(r'## Active P2B preparation\n.*?(?=## Completed P2A disposition)',
        '## Completed P2B exact-center review\n\n'+outcome+'\n\n'+provenance+'\n\n'+costs+'\n\n'+nextpara+'\n\n',text,flags=re.S)
    text=re.sub(r'## Separate P2B preparation\n.*?(?=## Completed bridge and earlier evidence)',
        '## P2B design and next comparison\n\n'+design+'\n\n'+nextpara+'\n\n'+
        'Baseline versus PDF-only remains primary. The failed P2A penalty is not included. Finite-x information does not resolve the known same-forward D-term pair. '
        'Real PDFs, independent CFF/PDF noise pairs and other truth families remain later work.\n\n',text,flags=re.S)
    text=text.replace('P2B now has a separate prospective preparation package.','P2B has a separate design and has now passed its exact-center control.')
    text=text.replace('This review confirms the returned completed preservation result and updates coordination.',
                      'The 28 October review confirms the completed P2B exact-center result and updates current coordination; the P2A review remains preserved above.')
    (ROOT/'HANDOFF.md').write_text(text)

    text=(SNAP/'PLAN.md').read_text()
    text=re.sub(r'Current compute task:.*?\n',notice+'\n',text,count=1)
    text=replace_one(text,'| P2B forward-PDF information | V3 GPU preparation; scientific design v1 unchanged | Implement GPU/hybrid backend, compare full-step performance, then execution-driver review |',
        '| P2B forward-PDF information | Exact-center COMPLETE, four gates PASS; independent saved-field replay PASS | Recover omitted execution evidence, then prepare one conditional half-noise development case |')
    text=text.replace('## Current evidence and next decision','## Completed P2A evidence')
    text=text.replace('Three new fits are complete (two development, one preservation);','Three P2A fits are complete (two development, one preservation);')
    text=re.sub(r'## P2B design\n.*?(?=## Completed resource record)',
        '## P2B exact-center result and next comparison\n\n'+outcome+'\n\n'+design+'\n\n'+provenance+'\n\n'+nextpara+'\n\n'+
        '## Completed P2B resources\n\n'+costs+'\n\n',text,flags=re.S)
    text=text.replace('## Completed resource record','## Completed P2A resource record')
    text=text.replace('P2B qualification is a separate preparation task.','P2B has its own completed exact-center record above.')
    (ROOT/'PLAN.md').write_text(text)

    (ROOT/'WORKER_PROMPT.md').write_text('# Current worker handoff: reviewed P2B exact-center result\n\n'+notice+'\n\n'+
        f'Read the {report}, [verification]({REVIEW}/PACKET_VERIFICATION.json), {nextlink}, '
        '[HANDOFF.md](HANDOFF.md), [PLAN.md](PLAN.md) and the '
        '[frozen design](GPD_P2B_preparation_v3/SCIENTIFIC_DESIGN.txt).\n\n'+outcome+'\n\n'+
        'Preserve the original stage60 selection and all recorded failure/success history. Parent22081808 and evaluation22117473 are completed work, '
        'not queued tasks. The fit is finite-budget feasible; no terminal KKT certificate exists.\n\n'+provenance+'\n\n'+nextpara+'\n\n'+
        'The current exact-center driver implements only that completed case. A future development driver must use its own prospective identity, '
        'the frozen draw and cumulative continuation budgets. It must not warm-start from this successful fit or reuse old exact-center submission permissions.\n\n'+
        'P2A remains STOP_PRESERVATION_FAILED. Do not retune or revive its assessment. P1 stays closed; experimental/replica pauses and manuscript/categorical deferrals remain. '
        'This local review submits no cluster job, draws no noise and changes no frozen scientific source.\n')

    text=(SNAP/'DOCUMENT_INDEX.md').read_text()
    text=text[text.index('## Current guidance'):]
    intro=('# Current and historical project documents\n\n'+notice+'\n\n## Latest P2B evidence\n\n'+
        f'- [Return ZIP](review_GPD_P2B_exact_center_evaluation_v1.zip), {report}, '
        f'[replayed scores]({REVIEW}/REVIEW_EVIDENCE.json), [verification]({REVIEW}/PACKET_VERIFICATION.json), {nextlink}.\n'+
        f'- [Current coordination receipt]({REVIEW}/UPDATE_RECEIPT.json) and [previous snapshot](handoff_history/2026-10-28/before_P2B_exact_center_review/).\n'+
        '- [Frozen scientific design](GPD_P2B_preparation_v3/SCIENTIFIC_DESIGN.txt) and [program perspective after P2A](GPD_program_review_20261026/PROGRAM_REVIEW.txt).\n\n'+
        'One P2B exact-center fit is complete. Noisy development remains unreleased. V3 preparation/launch instructions are historical; '
        'the returned result used CPU guarded threading/cache. Missing acceleration and qualification records limit local execution review; no GPU speedup is established.\n\n')
    text=text.replace('Latest evidence: completed eligible result, all three preservation gates fail;','Completed P2A evidence: eligible result, all three preservation gates fail;')
    text=text.replace('The new return SHA-256 is','The P2A preservation return SHA-256 is')
    text=text.replace('Current coordination identities are in GPD_P2B_GPU_preparation_20261026/UPDATE_RECEIPT.json;',
                      f'Current coordination identities are in {REVIEW}/UPDATE_RECEIPT.json;')
    (ROOT/'DOCUMENT_INDEX.md').write_text(intro+text)

    status=json.loads((SNAP/'HANDOFF_STATUS.json').read_text())
    history_keys=['phase','scope','latest_review','next_approach','P2B','latest_preparation','documentation_update','artifact_identity_scope']
    status['history']['before_P2B_exact_center_review_2026_10_28']={k:copy.deepcopy(status[k]) for k in history_keys}
    status['updated']='2026-10-28'
    status['updated_at']=datetime.datetime.now().astimezone().isoformat()
    status['phase']='P2B_EXACT_CENTER_PASS_REVIEWED_DEVELOPMENT_UNRELEASED'
    status['scope']='Accepted saved exact-center P2B result; execution qualification evidence incomplete locally; P2A remains stopped'
    status['latest_prior_editable_notes']=str(SNAP.relative_to(ROOT))
    status['latest_review']=dict(date='2026-10-28',status='ACCEPT_EXACT_CENTER_PRESERVATION',package=REVIEW,
        report=f'{REVIEW}/REVIEW.txt',evidence=f'{REVIEW}/REVIEW_EVIDENCE.json',
        verification=f'{REVIEW}/PACKET_VERIFICATION.json',incoming_archive='review_GPD_P2B_exact_center_evaluation_v1.zip',
        independent_saved_field_stages=5,max_selected_metric_absolute_difference=evidence['max_replayed_metric_absolute_difference'],
        NLO_replay=False,optimizer_replay=False,raw_cluster_preflight_replay=False,new_scientific_fits_run_locally=0)
    status['next_approach']=dict(name='P2B single conditional half-noise development comparison',
        status='EXACT_CENTER_PASSED_RECOVER_EXECUTION_EVIDENCE_THEN_PREPARE_DEVELOPMENT',
        contract='GPD_P2B_preparation_v3/SCIENTIFIC_DESIGN.txt',next_task=f'{REVIEW}/NEXT_ACTION.txt',
        scientific_fits_released=False,queued_tasks=[],cluster_contacted=False,development_released=False,
        exact_center_fit_completed=True,new_scientific_noise_draws=0)
    old=status['P2B']
    status['P2B']={k:copy.deepcopy(old[k]) for k in ['benchmark','comparison','lambda_stab','PDF_set','PDF_set_scope',
        'x_interval','uncertainty_model','prospective_joint_fit_ceiling','scientific_design_version_unchanged']}
    status['P2B'].update(date='2026-10-28',status='EXACT_CENTER_PASS_REVIEWED',new_joint_fits=1,
        completed_exact_center_fits=1,development_fits_completed=0,development_released=False,
        new_scientific_noise_draws=0,scientific_package='GPD_P2B_preparation_v3',result_archive='review_GPD_P2B_exact_center_evaluation_v1.zip',
        result_archive_sha256=evidence['archive_sha256'],review=f'{REVIEW}/REVIEW.txt',next_action=f'{REVIEW}/NEXT_ACTION.txt',
        selected_stage=60,reported_eligibility='FINITE_BUDGET_FEASIBLE',
        observed_Hplus_relative_L2=evidence['selected']['observed']['all'],
        target_absolute_relative_errors=[abs(v) for v in evidence['selected']['target_signed_relative_error']],
        PDF_whitened_RMS=evidence['selected']['boundary']['whitened_RMS'],gates=evidence['selected']['gates'],
        continuation=evidence['continuation'],costs=evidence['costs'],backend='cpu_guarded',gpus_used_by_fit=0,
        full_GPU_backend_implemented=None,full_pipeline_speedup_independently_verified=False,
        execution_evidence_gap='Acceleration overlay/manifest, performance/parity and parent readiness/submission records omitted',
        pending=['recover_existing_execution_and_qualification_evidence','prepare_and_review_single_half_noise_development_case'],
        terminal_convergence_certificate=False,cluster_contacted_locally=False,existing_frozen_scientific_sources_modified=False)
    status['latest_preparation']['status']='HISTORICAL_V3_PREPARATION_SUPERSEDED_BY_EXACT_CENTER_RESULT'
    status['documentation_update']=dict(scientific_runs=0,new_noise_draws=0,new_kernels=0,cluster_contacts=0,manuscript_edits=0,
        previous_snapshot=str(SNAP.relative_to(ROOT)),receipt=f'{REVIEW}/UPDATE_RECEIPT.json',validation=f'{REVIEW}/VALIDATION.json',
        frozen_evidence_modified=False)
    status['artifact_identity_scope']=f'Historical identities describe their frozen snapshots. Current coordination identities: {REVIEW}/UPDATE_RECEIPT.json.'
    status['artifact_identities']['review_GPD_P2B_exact_center_evaluation_v1.zip']=dict(sha256=evidence['archive_sha256'],
        bytes=(ROOT/'review_GPD_P2B_exact_center_evaluation_v1.zip').stat().st_size,role='frozen returned exact-center evidence')
    write(ROOT/'HANDOFF_STATUS.json',status)
    assert all(sha(ROOT/rel)==h for rel,h in fixed.items())
    after={name:sha(ROOT/name) for name in FILES}
    write(OUT/'UPDATE_RECEIPT.json',dict(date='2026-10-28',snapshot=str(SNAP.relative_to(ROOT)),before=before,after=after,
        protected_artifact_sha256=fixed,scope='Six editable root coordination files; no frozen scientific evidence modified',
        new_scientific_fits=0,new_noise_draws=0,cluster_contacts=0))
    print('Updated six coordination files; original snapshot and protected hashes retained.')

if __name__=='__main__':
    main()
