"""One frozen half-noise development case; assessment never releases another fit."""
import hashlib
import json
import math
import os
from pathlib import Path

ROOT=Path(os.environ.get('GPD_P2B_PROJECT_ROOT','/sfs/gpfs/tardis/home/dmk9m/GPD')).resolve()
CAPSULE=ROOT/'GPD_P1_Rivanna'
PACKET=ROOT/'GPD_P2B_preparation_v3'
ACCELERATION=ROOT/'P2B_acceleration_a02'
PACKAGE=Path(__file__).resolve().parent
BUNDLE=PACKAGE.parent
CASE='GPD_P2B_half_development_s31'
INPUT_TASK='precision160_Z1_a050_s31'
TRUE_TARGETS=[0.3619108615320205,0.1105532776465300]
BUDGETS=dict(Adam_updates=400,outer_iterations=60,solver_evaluations=12000)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def configuration():
    p=os.environ.get('GPD_P2B_RUN_SPEC')
    if not p:raise RuntimeError('Explicit development run spec required')
    return read(p)

def binding(case):
    if case!=CASE:raise ValueError('Only the frozen half-noise development case is implemented')
    cfg=configuration()
    observed=CAPSULE/'project/GPD_operating_range_plan/prepared'/f'{INPUT_TASK}.npz'
    if sha(observed)!=sha(BUNDLE/'inputs/CFF_HALF.npz'):raise RuntimeError('Existing half-CFF bytes differ')
    return dict(case=case,input_task=INPUT_TASK,phase='half_noise_development',seed=31,
        observed_sha256=sha(observed),PDF_input_sha256=sha(BUNDLE/'inputs/FORWARD_NOISY.npz'),
        PDF_noise_receipt_sha256=sha(BUNDLE/'inputs/NOISE_RECEIPT.json'),
        initial_file_sha256=sha(CAPSULE/'reference/FRESH_SEED31.npy'),
        capsule_manifest_sha256=sha(CAPSULE/'MANIFEST.json'),packet_manifest_sha256=sha(PACKET/'MANIFEST.json'),
        acceleration_manifest_sha256=sha(ACCELERATION/'MANIFEST.json'),driver_manifest_sha256=sha(PACKAGE/'MANIFEST.json'),
        transfer_manifest_sha256=sha(BUNDLE/'MANIFEST.json'),baseline_sha256=sha(BUNDLE/'evidence/baseline/BASELINE.json'),
        backend=cfg['backend'],threads=cfg['threads'],solver_blas_threads=cfg['solver_blas_threads'],versions=cfg['versions'],
        dtype='float64',parameters=3867,residuals=6046,lambda_stab=0.,adam_updates=400,max_outer_iterations=60,
        max_evaluations=12000,initial_radius=.01,selection_stages=[0,20,40,60,'actual_terminal'])

def check_restart(state,identity,phase):
    if state.get('run_lock_sha256')!=identity:raise RuntimeError('Checkpoint identity mismatch')
    if phase=='adam':
        n=state['completed_updates']
        if type(n) is not int or not 0<=n<=400 or state['remaining_updates']!=400-n:
            raise RuntimeError('Adam budget mismatch')
    elif phase=='solver':
        n=state['next_iteration'];evaluations=state['oracle_counts']['evaluations']
        if (type(n) is not int or not 1<=n<=60 or len(state['history'])!=n
                or state['remaining_outer_iterations']!=60-n or type(evaluations) is not int
                or not 0<=evaluations<=12000):raise RuntimeError('Solver budget/history mismatch')
    else:raise ValueError('Unknown phase')

def tolerance(a,b):return 1e-10+1e-6*max(abs(a),abs(b))

def assess_values(eligible,observed,targets,pdf_rms,baseline):
    if not eligible:return dict(decision='STOP_INELIGIBLE',assessment_released=False)
    if not baseline['eligible']:raise RuntimeError('Matched baseline is ineligible')
    values=[observed,pdf_rms,baseline['observed_Hplus_relative_L2']]+list(targets)+list(baseline['target_absolute_relative_errors'])
    if len(targets)!=2 or not all(math.isfinite(x) for x in values) or min(values[:3])<0:
        raise ValueError('Nonfinite/invalid scores')
    squared=observed**2;base_squared=baseline['observed_Hplus_squared_relative_L2']
    errors=[abs((v-t)/t) for v,t in zip(targets,TRUE_TARGETS)]
    limits=[tolerance(a,b) for a,b in zip(errors,baseline['target_absolute_relative_errors'])]
    gates=dict(observed_Hplus_squared_improves=base_squared-squared>tolerance(squared,base_squared),
        target_A_not_worse=errors[0]-baseline['target_absolute_relative_errors'][0]<=limits[0],
        target_B_not_worse=errors[1]-baseline['target_absolute_relative_errors'][1]<=limits[1])
    return dict(decision='DEVELOPMENT_PASS_REVIEW_REQUIRED' if all(gates.values()) else 'STOP_DEVELOPMENT_FAILED',
        gates=gates,assessment_released=False,observed_Hplus_relative_L2=observed,observed_Hplus_squared_relative_L2=squared,
        baseline_observed_Hplus_squared_relative_L2=base_squared,target_absolute_relative_errors=errors,
        baseline_target_absolute_relative_errors=baseline['target_absolute_relative_errors'],
        tolerances=dict(observed_squared=tolerance(squared,base_squared),targets=limits),
        PDF_whitened_RMS=pdf_rms,PDF_RMS_is_diagnostic_not_gate=True,
        scope='Frozen development comparison; no 2 percent exact-control gates and no new PDF RMS gate; no automatic independent assessment')
