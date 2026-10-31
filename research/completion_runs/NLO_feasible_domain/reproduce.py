"""One-command, isolated read-only replay of BOTH returned candidates.
Copies/hardlinks this packet into a fresh temporary project; never fits.
"""
import json,os,shutil,subprocess,sys,tempfile,hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
def materialize(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(src,dst)
    except OSError:shutil.copy2(src,dst)

def main():
    with tempfile.TemporaryDirectory(prefix='NLO_feasible_domain_replay_') as name:
        root=Path(name);task=root/'completion_runs/NLO_feasible_domain'
        task.mkdir(parents=True)
        freeze=json.loads((HERE/'FREEZE.json').read_text())
        for rel,h in freeze['files'].items():
            assert hashlib.sha256((HERE/rel).read_bytes()).hexdigest()==h,rel
            materialize(HERE/rel,task/rel)
        materialize(HERE/'FREEZE.json',task/'FREEZE.json')
        index=json.loads((HERE/'SNAPSHOT_INDEX.json').read_text())
        external=json.loads((HERE/'EXTERNAL_HASHES.json').read_text())
        for rel,h in external.items():assert index[rel]['sha256']==h,rel
        for rel,rec in index.items():
            src=HERE/rec['snapshot']
            assert hashlib.sha256(src.read_bytes()).hexdigest()==rec['sha256'],rel
            # Absolute environment-source names are provenance, never destinations.
            if not Path(rel).is_absolute():materialize(src,root/rel)
        materialize(HERE/'replay_saved.py',task/'replay_saved.py')
        for label in (31,43):
            for filename in ('final.pt','FIRST_ORDER.npz','RESULT.json'):
                rel=f'start{label}/{filename}';materialize(HERE/rel,task/rel)
        result=subprocess.run([sys.executable,str(task/'replay_saved.py')],capture_output=True,text=True)
        sys.stdout.write(result.stdout);sys.stderr.write(result.stderr)
        if result.returncode:raise SystemExit(result.returncode)

if __name__=='__main__':main()
