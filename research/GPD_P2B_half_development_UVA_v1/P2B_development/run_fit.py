"""P2B half-noise development driver; unchanged P1 optimizers, selection and CPU finalizer."""
import argparse
import fcntl
import gc
import inspect
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import time
import traceback

from p2b_policy import CAPSULE, PACKET, ACCELERATION, INPUT_TASK, binding, check_restart, sha, read, configuration

STOP = False

class Pause(Exception):
    pass

def stop_requested(*unused):
    global STOP
    STOP = True

def run(case, out, resume=False):
    # Called only after launch.py's shared validation and allocated guard.
    expected = binding(case)
    if resume:
        if not out.is_dir():
            raise RuntimeError('Resume requires an existing run directory')
    else:
        out.mkdir(parents=False, exist_ok=False)
        if (out.parent / 'preflight').is_dir():
            shutil.copytree(out.parent / 'preflight', out / 'preflight')
    if (out.parent / 'ATTEMPT_ENTRY.json').is_file():
        shutil.copy2(out.parent / 'ATTEMPT_ENTRY.json', out / 'ATTEMPT_ENTRY.json')
    with (out / 'RUNNING.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume and read(out / 'RUN_LOCK.json') != expected:
            raise RuntimeError('Resume requires an identical existing run lock')
        return _run_locked(case, out, resume, expected)

def _run_locked(case, out, resume, expected):
    sys.path.insert(0, str(CAPSULE / 'runtime'))
    sys.path.insert(0, str(PACKET / 'source'))
    sys.path.insert(0, str(ACCELERATION))
    import core
    core.setup()
    import torch
    from pdf_information import PDFProblem
    from development_information import information
    from acceleration_common import threads
    from model_cache import ExactModelCache
    from closure.solver_binding import adam400, restored_start, oracle_at, isolated_solver, accepted
    continuation = core.load_module('p2b_original_resume', CAPSULE / 'runtime/resume_support.py')
    lock = out / 'RUN_LOCK.json'
    if not resume:
        core.write(lock, expected)
    lock_hash = sha(lock)
    if (out / 'EXECUTION_FAILURE.json').exists():
        raise RuntimeError('Scientific execution failure requires review, not automatic resume')
    if (out / 'RESULT.json').exists():
        raise RuntimeError('Case already complete')
    signal.signal(signal.SIGUSR1, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    for name, phase in (('ADAM_RESUME.pt', 'adam'), ('SOLVER_RESUME.pt', 'solver')):
        if (out / name).exists():
            check_restart(torch.load(out / name, weights_only=False), lock_hash, phase)
    for name in ('ADAM400.pt', 'STAGE0.pt'):
        if (out / name).exists():
            if torch.load(out / name, weights_only=True).get('run_lock_sha256') != lock_hash:
                raise RuntimeError('Stage checkpoint identity mismatch: ' + name)
    config = configuration()
    if config["backend"] not in ("cpu1", "cpu_guarded"):
        raise RuntimeError("This development driver qualifies CPU execution only")
    limiter = threads(1)
    base, domain = core.problem(INPUT_TASK)
    problem = PDFProblem(base, information(base))
    limiter = threads(1)
    core.write(out / ('ENVIRONMENT_' + os.environ['SLURM_JOB_ID'] + '.json'), core.environment())
    core.check_scientific_bindings()

    def checkpoint(flat, stage, path):
        with torch.no_grad():
            names = {k: v.detach().cpu().numpy().tolist() for k, v in problem.named_components(flat).items()}
            obj = {'flat': flat.detach().clone(), 'state_dict': {k: v.detach().clone() for k, v in problem.parameters(flat).items()},
                   'seed': 31, 'stage': stage, 'J': names['J_joint'], 'named_components': names,
                   'components_schema': ['J_CFF', 'J_poly', 'J_PDF'], 'run_lock_sha256': lock_hash,
                   'lambda_stab': 0.0, 'PDF_input': 'FORWARD_NOISY.npz'}
        core.save(path, obj)
        return dict(stage=stage, J=obj['J'], path=path.name, sha256=sha(path))

    def adam_save(path, obj):
        obj['run_lock_sha256'] = lock_hash
        core.save(path, obj)
        if STOP:
            raise Pause('Adam update complete; optimizer and counters saved')
    continuation.atomic_torch = adam_save
    begin = time.monotonic()
    try:
        if not (out / 'OPTIMIZER_COMPLETE.json').exists():
            timing = {}
            if (out / 'ADAM400.pt').exists():
                adam = torch.load(out / 'ADAM400.pt', weights_only=True)['flat']
                timing['Adam400_seconds'] = read(out / 'ADAM400.json')['seconds']
            else:
                adam, seconds = continuation.run_adam(problem, adam400, out / 'ADAM_RESUME.pt')
                row = checkpoint(adam, 'Adam400', out / 'ADAM400.pt')
                timing['Adam400_seconds'] = seconds
                core.write(out / 'ADAM400.json', {'checkpoint': row, 'seconds': seconds})
            if STOP:
                raise Pause('Adam complete; restoration pending')
            if (out / 'STAGE0.pt').exists():
                stage0 = torch.load(out / 'STAGE0.pt', weights_only=True)['flat']
                timing['restoration_seconds'] = read(out / 'RESTORATION.json')['seconds']
            else:
                t0 = time.monotonic()
                core.write(out / 'RESTORATION_PHASE.json', {
                    'status': 'IN_PROGRESS', 'run_lock_sha256': lock_hash,
                    'source': 'ADAM400.pt', 'source_sha256': sha(out / 'ADAM400.pt'),
                    'restart': 'Replay unchanged deterministic restoration from completed Adam; do not reset Adam or solver budgets.'})
                choice, details = restored_start(problem, domain, adam)
                elapsed = time.monotonic() - t0
                if choice is None:
                    core.write(out / 'RESTORATION_FAILURE.json', {'candidates': details, 'seconds': elapsed})
                    raise RuntimeError('Original restoration returned no eligible start')
                value, name, stage0 = choice
                row = checkpoint(stage0, 0, out / 'STAGE0.pt')
                timing['restoration_seconds'] = elapsed
                core.write(out / 'RESTORATION.json', {'selected': name, 'selected_J': value, 'candidates': details, 'checkpoint': row, 'seconds': elapsed})
                core.write(out / 'RESTORATION_PHASE.json', {'status': 'COMPLETE', 'run_lock_sha256': lock_hash,
                                                          'checkpoint_sha256': row['sha256']})
            if STOP:
                raise Pause('Restoration complete; solver pending')
            resume_path = out / 'SOLVER_RESUME.pt'
            saved = torch.load(resume_path, weights_only=False) if resume_path.exists() else None
            x0 = stage0 if saved is None else torch.as_tensor(saved['x'], dtype=torch.float64)
            oracle = oracle_at(problem, domain, x0)
            solver, oracle, stats, settings = isolated_solver(oracle)
            limiter = threads(1)
            coordinate_cache = None
            if config["backend"] == "cpu_guarded":
                coordinate_cache = ExactModelCache(solver.model_coordinates)
                solver.model_coordinates = coordinate_cache
                from selective_threads import install
                install(solver)
                if saved:
                    coordinate_cache.stats.update(saved.get('coordinate_cache_stats', {}))
            if saved:
                stats.update(saved['binding_stats'])
            assert settings['max_iterations'] == 60 and settings['max_evaluations'] == 12000 and settings['initial_radius'] == .01

            def record(a, rec, trial):
                frame = inspect.currentframe().f_back.f_locals
                snapshot = continuation.solver_snapshot(frame, oracle, stats)
                snapshot['run_lock_sha256'] = lock_hash
                snapshot['remaining_outer_iterations'] = 60 - snapshot['next_iteration']
                snapshot['coordinate_cache_stats'] = dict(coordinate_cache.stats) if coordinate_cache is not None else None
                iteration = rec['iteration'] + 1
                core.save(out / 'LATEST_STATE.pt', {'flat': torch.as_tensor(a).clone(), 'completed_updates': iteration,
                                                  'operational_only': True, 'run_lock_sha256': lock_hash})
                if iteration in (20, 40, 60):
                    checkpoint(torch.as_tensor(a, dtype=torch.float64), iteration, out / ('STAGE%d.pt' % iteration))
                core.save(resume_path, snapshot)
                row = {'iteration': iteration, 'J_before_step': rec['J'], 'accepted': rec.get('accepted'),
                       'radius_end': rec.get('radius_end'), 'peak_RSS_bytes': core.rss_bytes(), 'solver_seconds': snapshot['seconds']}
                with (out / 'ITERATIONS.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n'); f.flush(); os.fsync(f.fileno())
                print(json.dumps(row), flush=True)
                if STOP:
                    raise Pause('Full outer-boundary state saved')

            last, diag = continuation.run_solver(solver, oracle, accepted.old.kkt, record, settings, saved)
            terminal_stage = len(diag['history'])
            terminal = out / ('STAGE%d.pt' % terminal_stage)
            if not terminal.exists():
                checkpoint(torch.as_tensor(last, dtype=torch.float64), terminal_stage, terminal)
            core.write(out / 'SOLVER_DIAGNOSTICS.json', diag)
            candidates = []
            for stage in dict.fromkeys((0, 20, 40, 60, terminal_stage)):
                path = out / ('STAGE%d.pt' % stage)
                if path.exists():
                    ck = torch.load(path, weights_only=True)
                    if ck.get('run_lock_sha256') != lock_hash:
                        raise RuntimeError('Candidate checkpoint identity mismatch')
                    candidates.append(dict(stage=stage, J=ck['J'], path=path.name, sha256=sha(path)))
            candidates.sort(key=lambda r: (r['J'], 31, r['stage']))
            timing['solver_seconds'] = diag['seconds']
            complete = {'status': 'FINITE_BUDGET_OPTIMIZER_COMPLETE_AWAITING_DIAGNOSTICS', 'task': case,
                        'run_lock_sha256': lock_hash, 'lambda_stab': 0.0, 'stage_times': timing,
                        'solver_status': diag['status'], 'solver_iterations': terminal_stage,
                        'candidates_by_supplied_J': candidates, 'binding_stats': stats,
                        'peak_RSS_bytes': core.rss_bytes(), 'last_process_elapsed_seconds': time.monotonic() - begin,
                        'oracle_counts': diag.get('oracle_counts', {}), 'oracle_timers': diag.get('oracle_timers', {}),
                        'linear_algebra_seconds': diag.get('linear_algebra_seconds', 0), 'solver_counts': diag.get('counts', {})}
            complete.update(coordinate_cache_stats=dict(coordinate_cache.stats) if coordinate_cache is not None else None,
                            field_and_rank_threads=1, solver_blas_threads=config['solver_blas_threads'])
            core.write(out / 'OPTIMIZER_COMPLETE.json', complete)
            oracle.cached = None
            del oracle, solver
            gc.collect()
        else:
            complete = read(out / 'OPTIMIZER_COMPLETE.json')
            if complete.get('run_lock_sha256') != lock_hash:
                raise RuntimeError('Optimizer-complete identity mismatch')
        if STOP:
            raise Pause('Optimization complete; validation pending')
        finalizer = core.load_module('p2b_original_finalize', CAPSULE / 'runtime/finalize.py')
        def finalize_write(path, obj):
            if Path(path).name == 'RESULT.json':
                ck = torch.load(out / obj['scored_checkpoint'], weights_only=True)
                with torch.no_grad():
                    names = {k: v.detach().cpu().numpy().tolist() for k, v in problem.named_components(ck['flat']).items()}
                obj.update(named_components=names, components_schema=['J_CFF', 'J_poly', 'J_PDF'],
                           lambda_stab=0.0, PDF_whitened_RMS=(names["chi2_PDF"]/36)**.5, run_lock_sha256=lock_hash,
                           execution_job_id=os.environ['SLURM_JOB_ID'],
                           interpretation='Frozen P2B half-noise development; J_joint/eligibility selection precedes truth scoring; independent assessment requires a later design and review')
                from score_forward import score
                boundary = score(problem, ck['flat'], out)
                core.write(out / 'FORWARD_SCORES.json', boundary)
                obj.update(forward_scores_sha256=sha(out/'FORWARD_SCORES.json'),
                           forward_scoring_seconds=boundary['seconds'])
            core.write(path, obj)
        finalizer.write = finalize_write
        finalizer.finalize(case, problem, domain, out, complete)
        print('CASE_COMPLETE', case, flush=True)
    except Pause as error:
        core.write(out / 'INTERRUPTED.json', {'status': 'PAUSED_NOT_A_RESULT', 'reason': str(error), 'run_lock_sha256': lock_hash})
        return 75
    except Exception as error:
        core.write(out / 'EXECUTION_FAILURE.json', {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        raise
    return 0
