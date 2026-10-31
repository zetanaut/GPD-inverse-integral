"""Archive a completed, verified two-continuation record."""
from pathlib import Path
import hashlib,json,zipfile,difflib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def digest(data):return hashlib.sha256(data).hexdigest()

def main():
    assert json.loads((HERE/'VALIDATION.json').read_text())['pass']
    assert json.loads((HERE/'logs/replay.json').read_text())['pass']
    target=ROOT/'review_NLO_constrained_solver.zip'
    if target.exists():raise FileExistsError('Refuse to overwrite an existing result archive')
    diff=''
    for p in sorted(HERE.glob('*.py')):
        diff+=''.join(difflib.unified_diff([],p.read_text().splitlines(True),fromfile='/dev/null',tofile=str(p.relative_to(ROOT))))
    (HERE/'SOURCE.diff').write_text(diff)
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and p not in (HERE/'MANIFEST.json',HERE/'PACKAGE_VALIDATION.json') and p.suffix not in ('.pyc','.tmp'))
    records=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())} for p in files]
    (HERE/'MANIFEST.json').write_text(json.dumps({'hash':'SHA256','self_excluded':['MANIFEST.json','PACKAGE_VALIDATION.json'],'files':records},indent=2)+'\n')
    files.append(HERE/'MANIFEST.json')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
        z.writestr('SUMMARY.md',(HERE/'SUMMARY.md').read_text())
        z.writestr('README.md','Extract with project paths preserved. The complete record is in completion_runs/NLO_constrained_solver/. Its README.md gives the no-fit replay command and exact immutable prior-packet dependency.\n')
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for r in records:
            data=z.read(r['path']);assert len(data)==r['bytes'] and digest(data)==r['sha256'],r['path']
        for rel,h in json.loads((HERE/'FREEZE.json').read_text())['files'].items():
            assert digest(z.read(str((HERE/rel).relative_to(ROOT))))==h,rel
        validation={'payload_files':len(records),'payload_hashes_pass':True,'payload_CRC_pass':True,
                    'expected_final_members':len(z.namelist())+1,'scope':'Payload verified before this record is appended; final archive CRC checked separately'}
    (HERE/'PACKAGE_VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
    with zipfile.ZipFile(target,'a',zipfile.ZIP_DEFLATED,compresslevel=6) as z:z.write(HERE/'PACKAGE_VALIDATION.json',str((HERE/'PACKAGE_VALIDATION.json').relative_to(ROOT)))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None and len(z.namelist())==validation['expected_final_members']
    h=digest(target.read_bytes());target.with_suffix('.zip.sha256').write_text(h+'  '+target.name+'\n')
    print(json.dumps({'archive':str(target),'bytes':target.stat().st_size,'sha256':h,'members':validation['expected_final_members'],'CRC_pass':True},indent=2))

if __name__=='__main__':main()
