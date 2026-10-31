"""Verify the supplied worker packet and construct an isolated replay workspace.

No fitting, downloads, kernel construction, or modification of worker evidence.
Usage: python prepare_audit.py --packet /path/review_NLO_feasible_domain.zip
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,tempfile,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def safe_rel(name):
    p=Path(name)
    require(not p.is_absolute() and '..' not in p.parts,f'Unsafe relative path: {name}')
    return p

def verify(source,work):
    task=source/'completion_runs/NLO_feasible_domain'
    require(task.is_dir(),'Missing completion_runs/NLO_feasible_domain')
    def link(p,q):
        q.parent.mkdir(parents=True,exist_ok=True)
        if q.exists():
            require(sha(p)==sha(q),f'Conflicting workspace file {q}')
            return
        try:os.link(p,q)
        except OSError:shutil.copy2(p,q)
    manifest=json.loads((task/'MANIFEST.json').read_text())['files']
    for rec in manifest:
        p=source/safe_rel(rec['path'])
        require(p.is_file() and p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'],f'Manifest mismatch {p}')
    freeze=json.loads((task/'FREEZE.json').read_text())['files']
    for rel,h in freeze.items():require(sha(task/safe_rel(rel))==h,f'Freeze mismatch {rel}')
    index=json.loads((task/'SNAPSHOT_INDEX.json').read_text())
    external=json.loads((task/'EXTERNAL_HASHES.json').read_text())
    for rel,h in external.items():require(index[rel]['sha256']==h,f'External mismatch {rel}')
    for rel,rec in index.items():
        p=task/safe_rel(rec['snapshot']);require(sha(p)==rec['sha256'],f'Snapshot mismatch {rel}')
        if not Path(rel).is_absolute():link(p,work/safe_rel(rel))
    for p in task.rglob('*'):
        if p.is_file() and 'external_snapshot' not in p.parts:
            link(p,work/'completion_runs/NLO_feasible_domain'/p.relative_to(task))
    return {'manifest_files':len(manifest),'freeze_files':len(freeze),'snapshot_files':len(index),
            'external_reference_hashes':len(external),'pass':True,'workspace':str(work.resolve())}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    args=parser.parse_args();work=HERE/'replay_workspace'
    with tempfile.TemporaryDirectory(prefix='gpd_packet_') as tmp:
        if args.packet.is_dir():source=args.packet.resolve()
        else:
            source=Path(tmp)
            with zipfile.ZipFile(args.packet) as z:
                require(sum(i.file_size for i in z.infolist())<2_000_000_000,'Archive larger than expected')
                for info in z.infolist():
                    safe_rel(info.filename)
                    require((info.external_attr>>16)&0o170000 != 0o120000,'Archive symlink rejected')
                z.extractall(source)
        result=verify(source,work)
    (HERE/'results').mkdir(exist_ok=True)
    (HERE/'results/PACKET_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
