"""External, single-child-tree watchdog; no scientific imports in the parent."""
import argparse, datetime, json, os, re, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def command(args):
    return subprocess.check_output(args, text=True).strip()

def memory():
    if sys.platform == 'darwin':
        raw = command(['/usr/bin/vm_stat'])
        page = int(re.search(r'page size of (\d+) bytes', raw).group(1))
        counts = {k.strip(): int(v) for k, v in re.findall(r'^([^:\n]+):\s+(\d+)\.', raw, re.M)}
        available = page * sum(counts[k] for k in ('Pages free', 'Pages inactive', 'Pages speculative'))
        swap = command(['/usr/sbin/sysctl', '-n', 'vm.swapusage'])
        used = int(float(re.search(r'used = ([\d.]+)M', swap).group(1)) * 2**20)
        pressure = command(['/usr/bin/memory_pressure', '-Q'])
        pressure_pct = int(re.search(r'memory free percentage: (\d+)%', pressure).group(1))
        return dict(available_bytes=available, available_method='(free+inactive+speculative)*page_size; reclaimability uncertain; purgeable not added twice', swap_used_bytes=used, pressure_free_percent=pressure_pct, vm_stat=raw, swap_raw=swap, pressure_raw=pressure)
    vals = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, rest = line.split(':', 1); vals[key] = int(rest.split()[0])*1024
    return dict(available_bytes=vals['MemAvailable'], available_method='Linux MemAvailable', swap_used_bytes=vals['SwapTotal']-vals['SwapFree'], pressure_free_percent=None)

def tree_rss(pid):
    rows = [tuple(map(int, line.split())) for line in command(['ps','-axo','pid=,ppid=,rss=']).splitlines() if line.strip()]
    owned = {pid}
    while True:
        extra = {p for p, parent, rss in rows if parent in owned}
        if extra <= owned: break
        owned |= extra
    return sum(rss*1024 for p, parent, rss in rows if p in owned)

def run(args):
    cfg = json.loads(args.config.read_text()); limits=cfg['resource_limits']
    ledger = json.loads(args.ledger.read_text()) if args.ledger.exists() else []
    used = sum(v['elapsed_seconds'] for v in ledger)
    if any(v.get('active') for v in ledger): raise RuntimeError('An active numerical process is recorded; inspect it, do not start another')
    if used >= limits['cumulative_seconds']: raise RuntimeError('Cumulative numerical budget exhausted')
    before=memory(); cap=limits['tree_rss_bytes']; reserve=limits['reserve_bytes']
    rec=dict(label=args.label, command=args.command, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), before=before, elapsed_seconds=0, active=False)
    if before['available_bytes']-reserve <= 0 or (before['pressure_free_percent'] is not None and before['pressure_free_percent'] < limits['minimum_pressure_free_percent']):
        rec.update(reason='PRECHECK_MEMORY_PRESSURE', returncode=None, peak_tree_rss_bytes=0)
        ledger.append(rec);write(args.ledger,ledger);return 2
    effective_cap=min(cap,before['available_bytes']-reserve)
    budget=min(limits['per_process_seconds'],limits['cumulative_seconds']-used)
    rec.update(effective_rss_cap_bytes=effective_cap,external_wall_cap_seconds=budget,active=True)
    ledger.append(rec);write(args.ledger,ledger)
    args.log.parent.mkdir(parents=True,exist_ok=True)
    t0=time.monotonic(); peak=0; reason=None; samples=[]
    with args.log.open('w') as log:
        child=subprocess.Popen(args.command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        rec['pid']=child.pid;write(args.ledger,ledger)
        try:
            while child.poll() is None:
                elapsed=time.monotonic()-t0;rss=tree_rss(child.pid);peak=max(peak,rss)
                snapshot=memory()
                samples.append(dict(seconds=elapsed,rss_bytes=rss,available_bytes=snapshot['available_bytes'],swap_used_bytes=snapshot['swap_used_bytes'],pressure_free_percent=snapshot['pressure_free_percent']))
                if elapsed >= budget: reason='EXTERNAL_WALL_LIMIT'
                elif rss > effective_cap: reason='TREE_RSS_LIMIT'
                elif snapshot['swap_used_bytes']-before['swap_used_bytes'] >= limits['material_swap_growth_bytes']:reason='NEW_SWAP_GROWTH'
                elif snapshot['available_bytes'] < reserve:reason='AVAILABLE_RESERVE_LIMIT'
                elif snapshot['pressure_free_percent'] is not None and snapshot['pressure_free_percent'] < limits['minimum_pressure_free_percent']:reason='MEMORY_PRESSURE'
                if reason:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=1)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                    break
                time.sleep(.5)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGKILL)
            child.wait()
            rec.update(active=False,returncode=child.returncode,reason=reason or ('COMPLETE' if child.returncode==0 else 'CHILD_FAILURE'),elapsed_seconds=time.monotonic()-t0,peak_tree_rss_bytes=peak,after=memory(),samples=samples)
            write(args.ledger,ledger)
    print(json.dumps({k:v for k,v in rec.items() if k not in ('before','after','samples')},indent=2),flush=True)
    return 0 if rec['reason']=='COMPLETE' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'PREFLIGHT_CONFIG.json');p.add_argument('--ledger',type=Path,default=ROOT/'results/RESOURCE_LEDGER.json');p.add_argument('--label',required=True);p.add_argument('--log',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();a.command=a.command[1:] if a.command[:1]==['--'] else a.command
    import fcntl
    a.ledger.parent.mkdir(parents=True,exist_ok=True)
    with a.ledger.with_suffix('.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sys.exit(run(a))
