"""Owned-process watchdog, signal forwarding, and durable checkpoint retention."""
import argparse
import fcntl
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from launch import validate
from p2b_policy import PACKAGE, PACKET, sha

COPY_INTERVAL = 300
COPY_JOIN_SECONDS = 60
POLL_SECONDS = 1

def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def retain(source, target, terminal=False):
    target.mkdir(parents=True, exist_ok=True)
    records = []
    if source.exists():
        for p in sorted(source.rglob('*')):
            if not p.is_file() or p.name.endswith('.partial') or p.name == 'RUNNING.lock':
                continue
            dst = target / p.relative_to(source)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + '.copying')
            shutil.copy2(p, tmp)
            os.replace(tmp, dst)
            if terminal:
                h = sha(p)
                if sha(dst) != h:
                    raise RuntimeError('Terminal retention hash mismatch: ' + str(p))
                records.append({'path': str(p.relative_to(source)), 'sha256': h, 'bytes': dst.stat().st_size})
    if terminal:
        atomic_json(target / 'TERMINAL_RETENTION.json', {'verified': True, 'source': str(source), 'files': records})
    else:
        atomic_json(target / 'PROVISIONAL.json', {
            'coherent_restart_verified': False, 'source': str(source),
            'purpose': 'Live best-effort copies; never resume without separate identity/coherence review.'})

class CopyWorker:
    """A stuck shared-filesystem copy must not stall the resource watchdog."""
    def __init__(self, source, target, terminal=False):
        self.error = None
        def copy():
            try:
                retain(source, target, terminal)
            except Exception as error:
                self.error = repr(error)
        self.thread = threading.Thread(target=copy, daemon=True)
        self.thread.start()

    def finish(self):
        self.thread.join(COPY_JOIN_SECONDS)
        if self.thread.is_alive():
            return 'Retention did not finish within the bounded join interval'
        return self.error

def signal_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass

def live_group_members(pgid):
    """Linux process-group members which can still write (excluding zombies)."""
    live = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in ('Z', 'X'):
                live.append(int(proc.name))
        except (OSError, ValueError, IndexError):
            continue
    return live

def stop_owned_group(child):
    # The leader may already have exited while descendants keep writing.
    signal_group(child.pid, signal.SIGKILL)
    child.wait(timeout=5)
    deadline = time.monotonic() + 5
    while live_group_members(child.pid):
        if time.monotonic() >= deadline:
            raise RuntimeError('Owned process group still has live writers')
        time.sleep(.05)

def claim_case(spec):
    path = Path(spec['case_registry']) / 'RUNNING.lock'
    handle = path.open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        handle.close()
        raise
    return handle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--approval', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    spec = validate(args.spec, args.case, args.resume, args.approval, allocated=True)
    case_guard = claim_case(spec)  # held for the entire supervised attempt
    source = Path(spec['output_root']) / args.case
    target = Path(spec['persistent_root']) / args.case
    sys.path.insert(0, str(PACKET / 'source'))
    from resource_watch import memory, tree_rss
    resources = spec['resources']
    reserve = resources['reserve_GiB'] * 2**30
    initial = memory()
    cap = min(resources['tree_rss_cap_GiB'] * 2**30, initial['available_bytes'] - reserve)
    if cap <= 0:
        raise RuntimeError('Insufficient free memory reserve')
    child = None
    signal_received = None
    def forward(signum, unused_frame):
        nonlocal signal_received
        signal_received = signum
        if child is not None:
            signal_group(child.pid, signum)
    signal.signal(signal.SIGUSR1, forward)
    signal.signal(signal.SIGTERM, forward)
    command = [sys.executable, '-B', str(PACKAGE / 'launch.py'), '--spec', str(args.spec),
               '--case', args.case, '--approval', str(args.approval)]
    if args.resume:
        command.append('--resume')
    start = time.monotonic()
    child = subprocess.Popen(command, start_new_session=True)
    if signal_received is not None:
        signal_group(child.pid, signal_received)
    peak = 0
    reason = None
    last_copy = start
    copier = None
    copy_errors = []
    monitor_error = None
    cleanup_error = None
    terminal_error = None
    try:
        while child.poll() is None:
            elapsed = time.monotonic() - start
            peak = max(peak, tree_rss(child.pid))
            mem = memory()
            if peak > cap:
                reason = 'TREE_RSS_LIMIT'
            elif mem['available_bytes'] < reserve:
                reason = 'AVAILABLE_RESERVE_LIMIT'
            elif mem['swap_used_bytes'] - initial['swap_used_bytes'] >= 64 * 2**20:
                reason = 'NEW_SWAP_GROWTH'
            if reason:
                signal_group(child.pid, signal.SIGTERM)
                # Resource pressure cannot wait for a potentially long outer step.
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    signal_group(child.pid, signal.SIGKILL)
                break
            if elapsed >= resources['stop_after_seconds'] and signal_received is None:
                forward(signal.SIGUSR1, None)
            if time.monotonic() - last_copy >= COPY_INTERVAL and (copier is None or not copier.thread.is_alive()):
                if copier is not None and copier.error:
                    copy_errors.append(copier.error)
                copier = CopyWorker(source, target / 'provisional')
                last_copy = time.monotonic()
            time.sleep(POLL_SECONDS)
    except Exception as error:
        monitor_error = repr(error)
    finally:
        try:
            stop_owned_group(child)
        except Exception as error:
            cleanup_error = repr(error)
        code = child.poll()
        if copier is not None:
            error = copier.finish()
            if error:
                copy_errors.append(error)
        if cleanup_error is None:
            terminal_error = CopyWorker(source, target, terminal=True).finish()
        else:
            terminal_error = 'Terminal retention not attempted while writers may remain'
        receipt = {'job_id': os.environ['SLURM_JOB_ID'], 'case': args.case, 'child_exit': code,
                   'resource_stop': reason, 'signal_received': signal_received,
                   'monitor_error': monitor_error, 'cleanup_error': cleanup_error,
                   'provisional_copy_errors': copy_errors, 'terminal_retention_error': terminal_error,
                   'terminal_retention_verified': cleanup_error is None and terminal_error is None,
                   'elapsed_seconds': time.monotonic() - start, 'peak_tree_rss_bytes': peak,
                   'rss_cap_bytes': cap, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        atomic_json(target / ('SUPERVISOR_' + os.environ['SLURM_JOB_ID'] + '.json'), receipt)
    if reason or monitor_error or cleanup_error or terminal_error or code is None:
        return 74
    return code if code >= 0 else 128 - code

if __name__ == '__main__':
    sys.exit(main())
