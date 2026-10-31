"""Update live coordination after independent review; snapshot all changed roots."""
import copy, datetime, hashlib, json, re, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
REVIEW=OUT.name
SNAP=ROOT/'handoff_history/2026-10-30/before_P2B_half_development_review'
FILES=['README.md','HANDOFF.md','PLAN.md','WORKER_PROMPT.md','DOCUMENT_INDEX.md','HANDOFF_STATUS.json']
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def identity(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert not SNAP.exists(),'Do not overwrite the pre-review snapshot'
protected={}
for directory in ['review_GPD_P2B_half_development_v1','GPD_P2B_half_development_UVA_v1','GPD_P2B_preparation_v3','GPD_P2B_exact_center_review_20261028','GPD_P2B_execution_evidence_review_20261028','GPD_P2_reconstruction_plan']:
 for p in (ROOT/directory).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:protected[str(p.relative_to(ROOT))]=identity(p)
for name in ['review_GPD_P2B_half_development_v1.zip','GPD_P2B_half_development_UVA_v1.zip','review_GPD_P2B_exact_center_evaluation_v1.zip','output/pdf/GPD_AI_Methods_Proposal_Note.pdf']:
 protected[name]=identity(ROOT/name)
SNAP.mkdir(parents=True)
before={}
for name in FILES:
 shutil.copy2(ROOT/name,SNAP/name);before[name]=identity(ROOT/name)
old=read(SNAP/'HANDOFF_STATUS.json');status=copy.deepcopy(old)
evidence=read(OUT/'REVIEW_EVIDENCE.json');intake=read(OUT/'INTAKE.json');audit=read(OUT/'PACKET_VERIFICATION.json')
lead=f'''Updated 30 October 2026. **P2B half-noise development is complete and fails the frozen development comparison: `STOP_DEVELOPMENT_FAILED`.** The selected stage 60 improves observed-slice H+ and target A, but target B worsens beyond the permitted tolerance. Independent assessment remains unreleased. Read the [independent review]({REVIEW}/REVIEW.txt), [replayed evidence]({REVIEW}/REVIEW_EVIDENCE.json) and [packet verification]({REVIEW}/PACKET_VERIFICATION.json).

Job 22157569 completed in one attempt with 400 Adam updates and 60 accepted outer updates. All 166 returned payload hashes and the original 54-file transfer verify; independent replay of the baseline and four P2B saved fields agrees within 5.66e-14. The execution evidence is sufficient for this review. No local fit, new noise draw or cluster operation occurred.

The [exact-center review](GPD_P2B_exact_center_review_20261028/REVIEW.txt) still accepts all four clean-input gates. The [prior execution review](GPD_P2B_execution_evidence_review_20261028/REVIEW.txt) retains its bounded CPU qualification. The noisy result is mixed: better forward anchoring and some targets do not establish uniformly better off-forward reconstruction.'''
comparison='''| Relative error | Reused half-noise baseline | P2B stage 60 |
|---|---:|---:|
| Observed-slice H+ L2 | 11.46345% | 10.13364% |
| Absolute target A | 4.98962% | 3.54036% |
| Absolute target B | 1.28139% | 3.23636% |
| Wider-chart H+ L2 | 28.61272% | 38.13437% |

Squared observed-slice H+ error improves 21.85518%. B's absolute relative error increases by 0.01954977 against a tolerance of 3.24636e-8. PDF whitened RMS improves from 3.03181 to 0.67139; it is diagnostic, not a development gate. Neither the exact-center 2% thresholds nor truth-based checkpoint reselection applies here.'''
disposition=f'''Close this frozen comparison at `STOP_DEVELOPMENT_FAILED`; retain the selected stage 60 and all evidence. There is no pending fit, retry or continuation. Independent assessment remains unreleased. The [next-action note]({REVIEW}/NEXT_ACTION.txt) identifies diagnosis using existing states and a prospective design as the next scientific decision, without releasing new computation. The original launch [ZIP](GPD_P2B_half_development_UVA_v1.zip) and [agent prompt](GPD_P2B_half_development_UVA_v1/UVA_AGENT_PROMPT.txt) are completed execution provenance.'''
resources='''Job 22157569 used 8 CPUs, 64 GiB and no GPU for 7:12:10 elapsed (TotalCPU 8:18:06; batch MaxRSS 14.55238 GiB). All 37 allocated fixture tests passed without skips, with four separate host submission tests and seven reported adapter tests. The run used 199 solver evaluations; 81.43% of solver time was spent in norm subproblems. It is finite-budget feasible, with no terminal convergence certificate. The different inputs and trajectories prevent interpreting its duration versus the exact-center fit as an acceleration benchmark.'''
preserved='''P1 stays closed. P2A remains `STOP_PRESERVATION_FAILED`, with its assessment stopped. Experimental continuations and remaining GPD replicas stay paused; manuscript and categorical architecture work remain deferred.'''
proposal='''The [proposal methods note](output/pdf/GPD_AI_Methods_Proposal_Note.pdf) is a preserved 29 October snapshot. Its clean-control result remains valid; its statement that noisy development is pending is superseded by this review. A current proposal account should describe the mixed noisy result and failed B safeguard.'''
(ROOT/'README.md').write_text(f'''# GPD reconstruction project

{lead}

{comparison}

{disposition}

{resources}

{proposal}

{preserved}

Use [HANDOFF.md](HANDOFF.md), [PLAN.md](PLAN.md), [WORKER_PROMPT.md](WORKER_PROMPT.md), [status](HANDOFF_STATUS.json) and [document index](DOCUMENT_INDEX.md). The [pre-review snapshot]({SNAP.relative_to(ROOT)}/) and [update receipt]({REVIEW}/UPDATE_RECEIPT.json) preserve coordination history. Frozen sources, returned evidence and prior reviews retain their original bytes.
''')
(ROOT/'WORKER_PROMPT.md').write_text(f'''# Current worker handoff: completed P2B development review

{lead}

{comparison}

{disposition}

Use the [review]({REVIEW}/REVIEW.txt) and [next-action note]({REVIEW}/NEXT_ACTION.txt) as current coordination. The launch package, frozen design, saved noise and returned run retain their original identities. Do not reinterpret an archived launch prompt as a current queued task. No new strength, covariance, seed, checkpoint substitution or extra budget follows from this review.

{proposal}

{preserved}
''')
# Preserve historical sections while replacing stale live release instructions.
old_lead=(SNAP/'README.md').read_text().split('\n\n')[1:4]
for name in ['HANDOFF.md','PLAN.md','DOCUMENT_INDEX.md']:
 text=(SNAP/name).read_text()
 for paragraph in old_lead:
  assert paragraph in text,(name,paragraph[:50])
  text=text.replace(paragraph,'',1)
 text=re.sub(r'Next, execute the one released half-noise development case[^\n]*',disposition,text)
 text=text.replace('Updated 29 October 2026. Project root:','Updated 30 October 2026. Project root:')
 text=text.replace('## P2B design and next comparison','## P2B frozen design and completed comparison')
 text=text.replace('## P2B exact-center result and next comparison','## Completed P2B exact-center control')
 text=text.replace('| P2B forward-PDF information | Exact-center/execution reviews PASS; one half-noise development package ready | Execute one fit, then review the frozen development comparison |',
                   '| P2B forward-PDF information | Exact-center PASS; noisy development STOP_DEVELOPMENT_FAILED | Preserve mixed result; independent assessment unreleased |')
 text=text.replace('This update packages one authorized development fit and one saved PDF draw; no cluster job or scientific fit was launched locally. No other campaign or manuscript work is released.',
                   'This update records the completed development review. The one saved PDF draw was reused unchanged; no fit, new noise or cluster operation occurred locally. No other campaign or manuscript work is released.')
 text=text.replace('The 28 October review confirms the completed P2B exact-center result and updates current coordination; the P2A review remains preserved above.',
                   'The 30 October review confirms the completed P2B development failure; the exact-center and P2A reviews remain preserved above.')
 text=text.replace('GPD_P2B_half_development_packaging_20261029/UPDATE_RECEIPT.json',REVIEW+'/UPDATE_RECEIPT.json')
 if name=='DOCUMENT_INDEX.md':
  text=text.replace('## Latest P2B evidence\n\n',f'''## Latest P2B evidence

- [Development return ZIP](review_GPD_P2B_half_development_v1.zip), [independent development review]({REVIEW}/REVIEW.txt), [replayed scores]({REVIEW}/REVIEW_EVIDENCE.json), [verification]({REVIEW}/PACKET_VERIFICATION.json) and [next action]({REVIEW}/NEXT_ACTION.txt). Completed run; B safeguard fails; assessment unreleased.
- [Current coordination snapshot]({SNAP.relative_to(ROOT)}/) and [validation]({REVIEW}/VALIDATION.json).
''')
  text=text.replace('[Current development ZIP]','[Completed development launch ZIP]')
  text=text.replace('One case authorized, fixed noise included, cluster execution pending.','Completed one-case execution provenance; the original fixed noise was reused.')
  text=text.replace('One P2B exact-center fit is complete. The one half-noise development case is released; its scientific result is pending.',
                    'Both prescribed P2B fits are complete: exact-center preservation passes; half-noise development fails its B safeguard. Independent assessment is unreleased.')
  text=text.replace('Current P2B development release; completed P1/P2A history and assessment boundaries','Completed P2B comparison; P1/P2A history and assessment boundaries')
  text=text.replace('Current one-case Rivanna handoff and preserved boundaries','Completed development disposition and preserved boundaries')
  text=text.replace('Completed preservation failure, review scope, continuation/costs and preserved earlier evidence','Completed P2B development and earlier evidence, review scopes and costs')
  insert=lead+'\n\n'+proposal
 else:
  insert=lead+'\n\n## Completed P2B half-noise development\n\n'+comparison+'\n\n'+resources+'\n\n'+disposition+'\n\n'+proposal
 first,rest=text.split('\n',1)
 text=first+'\n\n'+insert+'\n\n'+rest.lstrip()
 text=re.sub(r'\n{3,}','\n\n',text)
 (ROOT/name).write_text(text.rstrip()+'\n')
# Keep exact-center numbers under an explicit case label; latest case is development.
status['history']['before_P2B_half_development_review_2026_10_30']=dict(snapshot=str(SNAP.relative_to(ROOT)),phase=old['phase'],next_approach=old['next_approach'],P2B=old['P2B'],latest_review=old['latest_review'],latest_execution_review=old['latest_execution_review'],documentation_update=old['documentation_update'])
status.update(updated='2026-10-30',updated_at=datetime.datetime.now().astimezone().isoformat(),phase='P2B_HALF_DEVELOPMENT_COMPLETE_FAILED_ASSESSMENT_UNRELEASED',scope='Two prescribed P2B fits complete: exact-center preservation accepted; half-noise development B safeguard fails; independent assessment unreleased',latest_prior_editable_notes=str(SNAP.relative_to(ROOT)))
status['next_approach']=dict(name='P2B completed comparison and future-design decision',status='STOP_DEVELOPMENT_FAILED',contract='GPD_P2B_half_development_UVA_v1/SCIENTIFIC_CONTRACT.txt',next_task=REVIEW+'/NEXT_ACTION.txt',scientific_fits_released=False,queued_tasks=[],cluster_contacted=False,development_released=True,development_release_scope='Historical authorization fulfilled by one completed fit',exact_center_fit_completed=True,development_fit_completed=True,new_scientific_noise_draws=0,total_development_noise_draws=1,cluster_submission_pending=False,independent_assessment_released=False)
p=status['P2B']
casekeys=['result_archive','result_archive_sha256','review','selected_stage','reported_eligibility','observed_Hplus_relative_L2','target_absolute_relative_errors','PDF_whitened_RMS','gates','continuation','costs','terminal_convergence_certificate']
p['exact_center']={k:p.pop(k) for k in casekeys}
p['exact_center']['status']='ACCEPT_EXACT_CENTER_PRESERVATION'
s=evidence['selected'];b=evidence['baseline']
p['development']=dict(status='STOP_DEVELOPMENT_FAILED',result_archive=intake['archive'],result_archive_sha256=intake['sha256'],review=REVIEW+'/REVIEW.txt',selected_stage=s['stage'],reported_eligibility='FINITE_BUDGET_FEASIBLE',observed_Hplus_relative_L2=s['observed']['all'],target_absolute_relative_errors=[abs(x) for x in s['target_signed_relative_error']],PDF_whitened_RMS=s['boundary']['whitened_RMS'],wider_chart_Hplus_relative_L2=s['wider_chart']['charge_relative_L2']['all'],baseline_observed_Hplus_relative_L2=b['observed']['all'],baseline_target_absolute_relative_errors=[abs(x) for x in b['target_signed_relative_error']],baseline_wider_chart_Hplus_relative_L2=b['wider_chart']['charge_relative_L2']['all'],comparisons=evidence['comparisons'],continuation=evidence['trajectory'],costs=evidence['costs'],terminal_convergence_certificate=False)
p.update(date='2026-10-30',status='EXACT_CENTER_ACCEPTED_DEVELOPMENT_FAILED',new_joint_fits=2,development_fits_completed=1,review=REVIEW+'/REVIEW.txt',exact_center_review=old['P2B']['review'],result_archive=intake['archive'],result_archive_sha256=intake['sha256'],next_action=REVIEW+'/NEXT_ACTION.txt',pending=[],independent_assessment_released=False)
status['latest_review']=dict(date='2026-10-30',status='STOP_DEVELOPMENT_FAILED',package=REVIEW,report=REVIEW+'/REVIEW.txt',evidence=REVIEW+'/REVIEW_EVIDENCE.json',verification=REVIEW+'/PACKET_VERIFICATION.json',incoming_archive=intake['archive'],independent_saved_field_stages=5,max_replayed_metric_absolute_difference=evidence['max_replayed_metric_absolute_difference'],NLO_replay=False,optimizer_replay=False,raw_cluster_preflight_replay=False,new_scientific_fits_run_locally=0)
status['latest_preparation']['status']='COMPLETED_EXECUTION_PROVENANCE'
status['latest_execution_review']=dict(status='PASS_COMPLETED_HALF_DEVELOPMENT_EXECUTION',report=REVIEW+'/REVIEW.txt',audit=REVIEW+'/PACKET_VERIFICATION.json',incoming_archive=intake['archive'],incoming_archive_sha256=intake['sha256'],payload_files_verified=166,transfer_files_identical_to_sent=54,terminal_retention_files_verified=44,remote_environment_checks_reported_only=96,no_new_fit_or_noise_or_cluster_operation=True)
status['documentation_update']=dict(scientific_runs=0,new_noise_draws=0,new_kernels=0,cluster_contacts=0,cluster_jobs_submitted=0,manuscript_edits=0,previous_snapshot=str(SNAP.relative_to(ROOT)),receipt=REVIEW+'/UPDATE_RECEIPT.json',validation=REVIEW+'/VALIDATION.json',frozen_evidence_modified=False)
status['artifact_identity_scope']='Historical identities describe their snapshots. Current coordination identities: '+REVIEW+'/UPDATE_RECEIPT.json.'
status['artifact_identities'][intake['archive']]=identity(ROOT/intake['archive'])
write(ROOT/'HANDOFF_STATUS.json',status)
for key in ['P1','P2A_preflight','P2A_development','P2A_preservation','P2A_assessment','paused','deferred']:assert old[key]==status[key],key
for name,value in protected.items():assert identity(ROOT/name)==value,name
write(OUT/'UPDATE_RECEIPT.json',dict(date='2026-10-30',scope='Independent review and current root coordination only',snapshot=str(SNAP.relative_to(ROOT)),before=before,after={name:identity(ROOT/name) for name in FILES},protected_files=protected,other_scientific_dispositions_unchanged=True,proposal_pdf_unchanged=True,review_decision='STOP_DEVELOPMENT_FAILED'))
print(json.dumps(dict(updated=FILES,snapshot=str(SNAP.relative_to(ROOT)),protected_files_unchanged=len(protected))))
