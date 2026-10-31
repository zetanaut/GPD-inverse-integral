"""Package and verify completed evidence. No scientific calculations."""
from pathlib import Path
import json,hashlib,zipfile,difflib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha_bytes(data):return hashlib.sha256(data).hexdigest()

def main():
    assert json.loads((HERE/'VALIDATION.json').read_text())['pass']
    assert json.loads((HERE/'logs/reproduction.json').read_text())['pass']
    archive=ROOT/'review_NLO_feasible_domain.zip'
    if archive.exists():raise FileExistsError('Do not overwrite an existing result archive')
    diff=''
    for p in sorted(HERE.glob('*.py')):
        diff+=''.join(difflib.unified_diff([],p.read_text().splitlines(True),fromfile='/dev/null',tofile=str(p.relative_to(ROOT))))
    (HERE/'SOURCE.diff').write_text(diff)
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and p not in (HERE/'MANIFEST.json',HERE/'PACKAGE_VALIDATION.json') and p.suffix not in ('.tmp','.pyc'))
    records=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha_bytes(p.read_bytes())} for p in files]
    manifest={'hash':'SHA256','scope':'Complete evidence payload; this manifest and the archive-validation record are self-excluded','files':records}
    (HERE/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files.append(HERE/'MANIFEST.json')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
        # Small root-level navigation copies; scientific payload keeps project paths.
        z.writestr('SUMMARY.md',(HERE/'SUMMARY.md').read_text())
        z.writestr('README.md','Extract with paths preserved. The complete packet is in completion_runs/NLO_feasible_domain/. Read SUMMARY.md, then that folder\'s README.md for the isolated, no-fit reproduction command.\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))
        for r in records:
            data=z.read(r['path']);assert len(data)==r['bytes'] and sha_bytes(data)==r['sha256'],r['path']
        for rec in json.loads((HERE/'SNAPSHOT_INDEX.json').read_text()).values():
            member=str((HERE/rec['snapshot']).relative_to(ROOT))
            assert sha_bytes(z.read(member))==rec['sha256'],member
        for rel,h in json.loads((HERE/'FREEZE.json').read_text())['files'].items():
            member=str((HERE/rel).relative_to(ROOT))
            assert sha_bytes(z.read(member))==h,member
        assert z.read('SUMMARY.md')==(HERE/'SUMMARY.md').read_bytes()
        validation={'payload_files_verified':len(records),'manifest_sha256':sha_bytes((HERE/'MANIFEST.json').read_bytes()),
                    'payload_hashes_match':True,'payload_CRC_pass':True,'unique_members':True,
                    'expected_final_members':len(z.namelist())+1,
                    'scope':'Payload verified before appending this record; final archive CRC and SHA256 recorded alongside the archive'}
    (HERE/'PACKAGE_VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
    with zipfile.ZipFile(archive,'a',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.write(HERE/'PACKAGE_VALIDATION.json',str((HERE/'PACKAGE_VALIDATION.json').relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist())==validation['expected_final_members']
    digest=sha_bytes(archive.read_bytes())
    archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest,
                      'members':validation['expected_final_members'],'final_CRC_pass':True},indent=2))

if __name__=='__main__':main()
