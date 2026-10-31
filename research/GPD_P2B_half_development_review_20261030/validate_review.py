"""Verify final review/coordination identities without running a fit."""
import ast,hashlib,json,re,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def ident(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
r=read(OUT/'UPDATE_RECEIPT.json');i=read(OUT/'INTAKE.json');e=read(OUT/'REVIEW_EVIDENCE.json');a=read(OUT/'PACKET_VERIFICATION.json')
for name,x in r['before'].items():assert ident(ROOT/r['snapshot']/name)==x,name
for name,x in r['after'].items():assert ident(ROOT/name)==x,name
for name,x in r['protected_files'].items():assert ident(ROOT/name)==x,name
assert ident(ROOT/i['archive'])==dict(bytes=i['bytes'],sha256=i['sha256'])
packet=ROOT/i['extracted_root'];manifest=read(packet/'MANIFEST.json')['files']
for row in manifest:assert ident(packet/row['path'])=={k:row[k] for k in ['bytes','sha256']},row['path']
with zipfile.ZipFile(ROOT/i['archive']) as z:
 assert z.testzip() is None
 for info in z.infolist():
  if not info.is_dir():
   assert z.read(info)==(ROOT/info.filename).read_bytes(),info.filename
assert len(manifest)==166 and a['status']=='PASS' and e['scientific_decision']=='STOP_DEVELOPMENT_FAILED'
assert e['comparisons']['gates']==dict(observed_Hplus_squared_improves=True,target_A_not_worse=True,target_B_not_worse=False)
s=read(ROOT/'HANDOFF_STATUS.json');old=read(ROOT/r['snapshot']/'HANDOFF_STATUS.json')
for key in ['P1','P2A_preflight','P2A_development','P2A_preservation','P2A_assessment','paused','deferred']:assert old[key]==s[key],key
assert not s['next_approach']['scientific_fits_released'] and not s['next_approach']['cluster_submission_pending']
assert s['P2B']['new_joint_fits']==2 and s['P2B']['development_fits_completed']==1 and s['P2B']['pending']==[]
assert s['P2B']['new_scientific_noise_draws']==1 and s['documentation_update']['new_noise_draws']==0
assert not s['P2B']['independent_assessment_released']
for k,v in s['P2B']['exact_center'].items():
 if k!='status':assert old['P2B'][k]==v,k
links=set();new_review_links=set()
for name in r['after']:
 if not name.endswith('.md'):continue
 text=(ROOT/name).read_text()
 assert 'STOP_DEVELOPMENT_FAILED' in text
 for stale in ['Next, execute the one released','cluster execution pending','scientific result is pending','package ready | Execute']:
  assert stale not in text,(name,stale)
 for target in re.findall(r'\]\((<[^>]+>|[^)]+)\)',text):
  target=target.strip('<>')
  if '://' in target or target.startswith('#'):continue
  target=target.split('#')[0];links.add(target)
  if target.startswith(OUT.name+'/'):new_review_links.add(target)
missing=[t for t in sorted(links) if not (ROOT/t).exists() and t!=OUT.name+'/VALIDATION.json']
assert not missing,missing
for p in OUT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
validation=dict(status='PASS',date='2026-10-30',archive_identity_and_CRC=True,archive_and_extracted_bytes_match=True,payload_files_verified=166,protected_files_unchanged=len(r['protected_files']),coordination_files_verified=len(r['after']),snapshots_verified=len(r['before']),local_document_links_verified=len(links),new_review_links_verified=len(new_review_links),old_scientific_dispositions_unchanged=True,exact_center_metrics_preserved_under_explicit_case=True,current_development_status='STOP_DEVELOPMENT_FAILED',independent_assessment_released=False,no_pending_cluster_task=True,scientific_noise_draws_this_review=0,scientific_fits_this_review=0,cluster_contacts_this_review=0,proposal_pdf_unchanged=True,full_NLO_or_constraint_replay=False,max_independent_metric_discrepancy=e['max_replayed_metric_absolute_difference'])
write(OUT/'VALIDATION.json',validation)
files=[dict(path=str(p.relative_to(OUT)),**ident(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts]
write(OUT/'MANIFEST.json',dict(date='2026-10-30',scope='Independent local review only; returned packet and prior sources unchanged',files=files))
print(json.dumps(validation))
print('Review manifest:',len(files),'files')
