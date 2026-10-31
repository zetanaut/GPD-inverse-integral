"""Only a clean, terminally retained same-fit pause can continue automatically."""
import shutil
from pathlib import Path
from p2b_policy import read,sha,check_restart

def verify_parent(path,expected_lock=None):
    path=Path(path).resolve()
    receipt=read(path/'TERMINAL_RETENTION.json')
    if receipt.get('verified') is not True:raise RuntimeError('Unverified parent retention')
    for row in receipt['files']:
        rel=Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts:raise RuntimeError('Unsafe retained path')
        f=path/rel
        if f.is_symlink() or not f.is_file() or f.stat().st_size!=row['bytes'] or sha(f)!=row['sha256']:
            raise RuntimeError('Changed parent artifact: '+str(rel))
    supervisors=list(path.glob('SUPERVISOR_*.json'))
    if len(supervisors)!=1:raise RuntimeError('Exactly one terminal parent supervisor required')
    supervisor=read(supervisors[0])
    if (supervisor['child_exit']!=75 or not supervisor['terminal_retention_verified'] or any(supervisor.get(k) for k in
        ['resource_stop','monitor_error','cleanup_error','terminal_retention_error','provisional_copy_errors'])):
        raise RuntimeError('Only a clean planned pause can continue; failures require review')
    if (path/'RESULT.json').exists() or (path/'EXECUTION_FAILURE.json').exists():
        raise RuntimeError('Completed/failed fit cannot be resumed automatically')
    if not (path/'INTERRUPTED.json').is_file():raise RuntimeError('Clean boundary pause record required')
    lock=read(path/'RUN_LOCK.json');identity=sha(path/'RUN_LOCK.json')
    if expected_lock is not None and lock!=expected_lock:raise RuntimeError('Continuation scientific identity changed')
    if read(path/'INTERRUPTED.json')['run_lock_sha256']!=identity:raise RuntimeError('Pause identity changed')
    import numpy as np
    import torch
    def load(name):
        with torch.serialization.safe_globals([np.core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype('float64'))]):
            return torch.load(path/name,map_location='cpu',weights_only=True)
    adam=load('ADAM_RESUME.pt');check_restart(adam,identity,'adam')
    progress=dict(Adam_updates=adam['completed_updates'],outer_iterations=0,solver_evaluations=0)
    if (path/'SOLVER_RESUME.pt').exists():
        state=load('SOLVER_RESUME.pt');check_restart(state,identity,'solver')
        latest=load('LATEST_STATE.pt')
        journal=[__import__('json').loads(s) for s in (path/'ITERATIONS.jsonl').read_text().splitlines() if s.strip()]
        n=state['next_iteration']
        if (adam['completed_updates']!=400 or latest['completed_updates']!=n or latest['run_lock_sha256']!=identity
                or not np.array_equal(np.asarray(latest['flat']),state['x']) or len(journal)!=n
                or [r['iteration'] for r in journal]!=list(range(1,n+1))
                or any(r['accepted']!=h['accepted'] for r,h in zip(journal,state['history']))):
            raise RuntimeError('Incoherent terminal solver boundary')
        progress.update(outer_iterations=n,solver_evaluations=state['oracle_counts']['evaluations'])
        for stage in (20,40,60):
            if stage<=n:
                ck=load(f'STAGE{stage}.pt')
                if ck['run_lock_sha256']!=identity:raise RuntimeError('Stage identity mismatch')
                if stage==n and not np.array_equal(np.asarray(ck['flat']),state['x']):raise RuntimeError('Stage/terminal mismatch')
    start=read(path/'ATTEMPT_ENTRY.json').get('starting_progress',dict(Adam_updates=0,outer_iterations=0,solver_evaluations=0))
    if (progress['Adam_updates'],progress['outer_iterations']) <= (start['Adam_updates'],start['outer_iterations']):
        raise RuntimeError('No completed-update progress; no automatic retry')
    return dict(root=str(path),retention_sha256=sha(path/'TERMINAL_RETENTION.json'),
                supervisor_file=supervisors[0].name,supervisor_sha256=sha(supervisors[0]),
                run_lock_sha256=identity,progress=progress)

def stage_parent(parent,out,expected_lock):
    checked=verify_parent(parent['root'],expected_lock)
    if checked!=parent:raise RuntimeError('Parent changed since continuation preparation')
    source=Path(parent['root']);out.mkdir(parents=True,exist_ok=False)
    history=out/'parent_attempts';history.mkdir()
    label=parent['supervisor_file'].removeprefix('SUPERVISOR_').removesuffix('.json')
    archived=history/label
    shutil.copytree(source,archived,ignore=shutil.ignore_patterns('provisional','*.partial','*.copying','RUNNING.lock','parent_attempts'))
    if (source/'parent_attempts').exists():
        for entry in (source/'parent_attempts').iterdir():shutil.copytree(entry,history/entry.name)
    for entry in source.iterdir():
        if entry.is_file() and (entry.suffix=='.pt' or entry.name in
            ['RUN_LOCK.json','ADAM400.json','RESTORATION.json','RESTORATION_PHASE.json','ITERATIONS.jsonl',
             'OPTIMIZER_COMPLETE.json','SOLVER_DIAGNOSTICS.json']):
            shutil.copy2(entry,out/entry.name)
            if sha(out/entry.name)!=sha(entry):raise RuntimeError('Staging copy mismatch')
