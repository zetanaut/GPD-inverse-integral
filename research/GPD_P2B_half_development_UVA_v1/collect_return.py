"""Collect the frozen package and every claimed attempt, including failures/costs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
BUNDLE=Path(__file__).resolve().parent
sys.path.insert(0,str(BUNDLE/'P2B_development'))
from p2b_policy import CASE,read,sha
from verify_transfer import verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--accounting',type=Path,required=True)
    p.add_argument('--assessment',type=Path);p.add_argument('--extra',type=Path,action='append',default=[]);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();spec=read(a.spec);verify(BUNDLE,spec['transfer_manifest_sha256'])
    if a.out.exists():raise RuntimeError('Return archive already exists')
    members={}
    def add(path,name):
        if path.is_symlink() or not path.is_file():raise RuntimeError('Regular, non-symlink file required: '+str(path))
        if name in members:raise RuntimeError('Duplicate return path: '+name)
        members[name]=path
    def tree(root,prefix):
        if not root.is_dir():return False
        for path in sorted(root.rglob('*')):
            rel=path.relative_to(root)
            if any(s in ['provisional','__pycache__'] for s in rel.parts) or path.name in ['RUNNING.lock','PREPARE.lock'] or path.name.endswith(('.partial','.copying')):continue
            if path.is_symlink():raise RuntimeError('Unexpected return symlink: '+str(path))
            if path.is_file():add(path,prefix+'/'+str(rel))
        return True
    tree(BUNDLE,'transfer')
    registry=Path(spec['case_registry']);claim=read(registry/'CLAIM.json')
    if Path(claim['latest_spec']).resolve()!=a.spec.resolve():raise RuntimeError('Collect from latest attempt spec')
    attempts=[];cursor=claim
    while cursor:
        path=Path(cursor['latest_spec']);s=read(path)
        if sha(path)!=cursor['latest_spec_sha256'] or s['case']!=CASE:raise RuntimeError('Claimed attempt changed')
        control=path.parent;tree(control,'control/'+control.name)
        retained=Path(s['persistent_root']);present=tree(retained,'results/'+retained.name)
        attempts.append(dict(tag=control.name,job_id=cursor['submitted_job_id'],spec_sha256=sha(path),persistent_root_present=present))
        cursor=cursor['previous']
    add(registry/'CLAIM.json','CASE_CLAIM.json');add(a.accounting,'Slurm_accounting.txt')
    if a.assessment:add(a.assessment,'ASSESSMENT.json')
    for index,path in enumerate(a.extra):
        prefix=f'extras/{index:02d}_{path.name}'
        if path.is_dir():tree(path,prefix)
        else:add(path,prefix)
    total=sum(path.stat().st_size for path in members.values())
    if total>256*2**20:raise RuntimeError('Return exceeds 256 MiB; inspect contents before packaging; never omit failures or silently truncate')
    report=dict(case=CASE,attempts=attempts,assessment_included=bool(a.assessment),independent_assessment_released=False,
        payload_bytes=total,scope='All claimed controls and retained results; prior exact/P1 evidence and fixed inputs are in transfer. No maps or environment duplication.')
    rows=[]
    prefix='review_GPD_P2B_half_development_v1/'
    with zipfile.ZipFile(a.out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(members.items()):
            digest=sha(path);size=path.stat().st_size;z.write(path,prefix+name)
            if digest!=sha(path) or size!=path.stat().st_size:raise RuntimeError('A source changed while archiving; wait for terminal quiescence')
            rows.append(dict(path=name,bytes=size,sha256=digest))
        encoded=(json.dumps(report,indent=2)+'\n').encode();z.writestr(prefix+'RETURN_SUMMARY.json',encoded)
        rows.append(dict(path='RETURN_SUMMARY.json',bytes=len(encoded),sha256=hashlib.sha256(encoded).hexdigest()))
        z.writestr(prefix+'MANIFEST.json',json.dumps(dict(files=rows),indent=2)+'\n')
    with zipfile.ZipFile(a.out) as z:
        for row in rows:
            data=z.read(prefix+row['path'])
            if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise RuntimeError('Return ZIP verification failed')
    digest=sha(a.out);a.out.with_suffix(a.out.suffix+'.sha256').write_text(digest+'  '+a.out.name+'\n')
    print(json.dumps(dict(archive=str(a.out),sha256=digest,bytes=a.out.stat().st_size,files=len(rows),attempts=len(attempts))))

if __name__=='__main__':main()
