"""Read-only independent intake/source/execution audit. No cluster calls or fitting."""
import ast,csv,hashlib,io,json,re
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
PACKET=ROOT/'review_GPD_P2B_half_development_v1';TRANSFER=PACKET/'transfer';SENT=ROOT/'GPD_P2B_half_development_UVA_v1'
TAG='GPD_P2B_half_development_20261029_a01';CASE='GPD_P2B_half_development_s31';JOB='22157569'
CONTROL=PACKET/'control'/TAG;RESULT=PACKET/'results'/TAG/CASE
CAP=ROOT/'review_GPD_P1_advice/reference_capsule';PREP=ROOT/'GPD_P2B_preparation_v3';OLD=ROOT/'review_GPD_P2B_execution_evidence_addendum_v1/originals'
REMOTE='/sfs/gpfs/tardis/home/dmk9m/GPD/'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,d):(OUT/name).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def verify(base,rows):
 for row in rows:
  rel=Path(row['path']);p=base/rel
  assert not rel.is_absolute() and '..' not in rel.parts and not p.is_symlink()
  assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
 return len(rows)
def mapped(path):
 if not path.startswith(REMOTE):return None
 rel=path[len(REMOTE):]
 for prefix,base in [('GPD_P2B_half_development_UVA_v1/',TRANSFER),('GPD_P1_Rivanna/',CAP),('GPD_P2B_preparation_v3/',PREP),('P2B_acceleration_a02/',OLD/'P2B_acceleration_a02'),('control/'+TAG+'/',CONTROL),('control/GPD_P2B_half_development_intake_20261029_a01/',PACKET/'extras/01_GPD_P2B_half_development_intake_20261029_a01')]:
  if rel.startswith(prefix):
   p=base/rel[len(prefix):]
   return p if p.is_file() else None
 p=OLD/rel
 return p if p.is_file() else None

def main():
 payload=verify(PACKET,read(PACKET/'MANIFEST.json')['files'])
 actual={str(p.relative_to(PACKET)) for p in PACKET.rglob('*') if p.is_file()}
 assert actual=={r['path'] for r in read(PACKET/'MANIFEST.json')['files']}|{'MANIFEST.json'}
 transfer_count=verify(TRANSFER,read(TRANSFER/'MANIFEST.json')['files'])
 original_receipt=read(ROOT/'GPD_P2B_half_development_UVA_v1_RECEIPT.json')
 assert sha(TRANSFER/'MANIFEST.json')==sha(SENT/'MANIFEST.json')==original_receipt['manifest_sha256']
 for p in TRANSFER.rglob('*'):
  if p.is_file():assert sha(p)==sha(SENT/p.relative_to(TRANSFER))
 driver_count=verify(TRANSFER/'P2B_development',read(TRANSFER/'P2B_development/MANIFEST.json')['files'])
 prep_count=verify(PREP,read(PREP/'MANIFEST.json')['files'])
 acceleration_count=verify(OLD/'P2B_acceleration_a02',read(OLD/'P2B_acceleration_a02/MANIFEST.json')['files'])
 spec,lock=read(CONTROL/'RUN_SPEC.json'),read(RESULT/'RUN_LOCK.json');deps=read(TRANSFER/'DEPENDENCIES.json')
 for label,base in [('capsule',CAP),('packet',PREP),('acceleration',OLD/'P2B_acceleration_a02'),('driver',TRANSFER/'P2B_development'),('transfer',TRANSFER)]:
  assert sha(base/'MANIFEST.json')==spec[label+'_manifest_sha256']==lock[label+'_manifest_sha256']
 for key in ['files','versions','python','performance_review']:assert spec[key]==deps[key]
 assert spec['backend']==lock['backend']=='cpu_guarded' and spec['threads']==lock['threads']==1 and spec['solver_blas_threads']==lock['solver_blas_threads']==8
 assert spec['scientific_budgets']==dict(Adam_updates=400,outer_iterations=60,solver_evaluations=12000)
 assert spec['case']==lock['case']==CASE and lock['input_task']=='precision160_Z1_a050_s31' and lock['seed']==31
 assert (lock['parameters'],lock['residuals'],lock['lambda_stab'],lock['dtype'])==(3867,6046,0.0,'float64')
 assert spec['resume_parent'] is None and spec['development_released'] and not spec['independent_assessment_released']
 resources=spec['resources'];assert (resources['account'],resources['cpus_per_task'],resources['memory_GiB'],resources['gpus'],resources['wall_seconds'])==('spinquest_standard',8,64,0,43200)
 assert spec['resources']['stop_after_seconds']==36000
 verified_pins,remote_only={},{}
 for path,digest in spec['files'].items():
  local=mapped(path)
  if local:
   assert sha(local)==digest,path;verified_pins[path]=dict(local=str(local.relative_to(ROOT)),sha256=digest)
  else:remote_only[path]=digest
 # Verify the exact local scientific modules before saved-field replay imports them.
 original={r['path']:r for r in read(CAP/'MANIFEST.json')['files']}
 source_rels=['project/GPD_three_stage_closure/'+s for s in ['vendor/gpd_reh/model.py','vendor/gpd_reh/reference_model.py','closure/truths.py','closure/geometry.py','closure/metrics.py','prepared/FIT_CONFIG.json','prepared/GEOMETRY.json']]
 source_rels+=['reference/FRESH_SEED31.npy','runtime/core.py','runtime/finalize.py','runtime/resume_support.py','project/GPD_operating_range_plan/prepared/precision160_Z1_a050_s31.npz']
 source_pins={}
 for name in source_rels:
  p=CAP/name;assert p.stat().st_size==original[name]['bytes'] and sha(p)==original[name]['sha256'];source_pins[name]=sha(p)
 assert sha(CAP/source_rels[-1])==sha(TRANSFER/'inputs/CFF_HALF.npz')==lock['observed_sha256']
 assert sha(CAP/'reference/FRESH_SEED31.npy')==lock['initial_file_sha256']
 for path,key in [('inputs/FORWARD_NOISY.npz','PDF_input_sha256'),('inputs/NOISE_RECEIPT.json','PDF_noise_receipt_sha256'),('evidence/baseline/BASELINE.json','baseline_sha256')]:assert sha(TRANSFER/path)==lock[key]
 baseline=read(TRANSFER/'evidence/baseline/BASELINE.json')
 original_baseline=ROOT/'completion_runs/P1_Rivanna_received_20261017/P1_results/precision160_Z1_a050_s31'
 for name,digest in baseline['files'].items():assert sha(TRANSFER/'evidence/baseline'/name)==sha(original_baseline/name)==digest
 result,scores=read(RESULT/'RESULT.json'),read(RESULT/'SCORES.json');selected=result['selected'];forward=read(RESULT/'FORWARD_SCORES.json')
 assert result['status']=='FINITE_BUDGET_FEASIBLE' and selected['pass'] and selected['constraints_finite'] and selected['constraint_exceedances']==0
 assert selected['moments']['pass'] and selected['full_operator']['pass'] and selected['constraint_max']<=0
 assert sha(RESULT/'final.pt')==sha(RESULT/'STAGE60.pt')==result['scored_checkpoint_sha256']
 for name,key in [('RUN_LOCK.json','run_lock_sha256'),('SCORES.json','scores_sha256'),('FORWARD_SCORES.json','forward_scores_sha256')]:assert sha(RESULT/name)==result[key]
 assert sha(RESULT/'SELECTION.json')==scores['selection_sha256']==forward['selection_sha256']
 for row in selected['full_operator']['coordinate_selected_rows']:assert row['pass'] and sha(RESULT/row['file'])==row['sha256']
 retention=read(RESULT/'TERMINAL_RETENTION.json');assert retention['verified'];retained_count=verify(RESULT,retention['files'])
 supervisor=read(RESULT/f'SUPERVISOR_{JOB}.json')
 assert supervisor['child_exit']==0 and supervisor['terminal_retention_verified'] and not any(supervisor[k] for k in ['resource_stop','signal_received','monitor_error','cleanup_error','provisional_copy_errors','terminal_retention_error'])
 assert not any((RESULT/f).exists() for f in ['INTERRUPTED.json','EXECUTION_FAILURE.json','QUALIFICATION_FAILED.json'])
 entry=read(RESULT/'ATTEMPT_ENTRY.json');assert entry['spec_sha256']==sha(CONTROL/'RUN_SPEC.json') and not entry['resume'] and entry['starting_progress']==dict(Adam_updates=0,outer_iterations=0,solver_evaluations=0)
 env=read(RESULT/f'ENVIRONMENT_{JOB}.json')
 assert env['python'].split()[0]==spec['versions']['python'] and env['executable']==spec['python'] and env['torch_threads']==1
 for k in ['numpy','scipy','torch']:assert env[k]==spec['versions'][k]
 assert env['slurm']['SLURM_JOB_ID']==JOB and env['slurm']['SLURM_CPUS_PER_TASK']=='8' and env['slurm']['SLURM_MEM_PER_NODE']=='65536'
 qualification=read(RESULT/'preflight/QUALIFICATION.json');assert qualification['status']=='PASS'
 counts={'test_policy.py':10,'test_forward_adapter.py':2,'test_continuation.py':4,'test_operations.py':7,'test_launch.py':14}
 for group in qualification['groups']:
  name=group['script'];assert group['exit_code']==0 and sha(TRANSFER/'P2B_development'/name)==group['source_sha256']
  for stream in ['stdout','stderr']:assert sha(RESULT/'preflight'/(name+'.'+stream))==group[stream+'_sha256']
  log=(RESULT/'preflight'/(name+'.stderr')).read_text();assert f'Ran {counts[name]} tests' in log and '\nOK\n' in log and 'skipped' not in log
 correction=read(CONTROL/'QUALIFICATION_COUNT_CLARIFICATION.json');assert correction['allocated_group_counts']==counts and correction['observed_allocated_tests']==sum(counts.values())==37
 assert sha(CONTROL/'HPC_STARTUP_VERIFIED_a02.json')==correction['observed_evidence_sha256']
 assert sha(CONTROL/'scientific_readiness.json')==correction['earlier_scientific_readiness_sha256']
 assert sha(CONTROL/'coordinator_exact_ack.json')==correction['earlier_ACK_sha256']
 # Inspect external submission adapter without executing it or contacting Slurm.
 adapter=ast.parse((CONTROL/'submission_adapter/sbatch').read_text())
 cfg=next(ast.literal_eval(s.value) for s in adapter.body if isinstance(s,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in s.targets))
 command=read(CONTROL/'HPC_SUBMISSION_COMMAND.json');review=read(CONTROL/'HPC_REVIEW.json');ack=read(CONTROL/'coordinator_exact_ack.json')
 assert cfg['pins']==command['pins'];assert command['actual_command']==review['adapter_review']['actual_exec_command']==ack['actual_scheduler_command']
 assert command['actual_command']==[cfg['real_sbatch'],'--account=spinquest_standard','--parsable','--job-name=GPD_P2B_half_development',cfg['script']]
 assert ack['status']=='ACK_EXACT_ONCE_ONLY_SUBMISSION' and ack['authorized_submissions_now']==1 and not ack['independent_assessment_released']
 assert ack['spec_sha256']==sha(CONTROL/'RUN_SPEC.json') and ack['script_sha256']==sha(CONTROL/'RUN.sbatch') and ack['hpc_review_sha256']==sha(CONTROL/'HPC_REVIEW.json')
 control_pins={}
 for block in [cfg['pins'],review['pins'],ack['reviewed_pins']]:
  for path,digest in block.items():
   local=mapped(path);assert local is not None,path;assert sha(local)==digest,path;control_pins[path]=digest
 prep=read(CONTROL/'PREPARATION.json');assert prep['spec_sha256']==sha(CONTROL/'RUN_SPEC.json') and prep['submission_script_sha256']==sha(CONTROL/'RUN.sbatch')
 permit=read(CONTROL/'RUN_PERMIT.json');assert permit==dict(authorized=True,case=CASE,spec_sha256=sha(CONTROL/'RUN_SPEC.json'))
 submitted=read(CONTROL/'SUBMISSION.json');assert submitted['job_id']==JOB and submitted['returncode']==0
 claim=read(PACKET/'CASE_CLAIM.json');assert claim['submitted_job_id']==JOB and claim['previous'] is None and claim['latest_spec_sha256']==sha(CONTROL/'RUN_SPEC.json')
 assert len(read(PACKET/'RETURN_SUMMARY.json')['attempts'])==1
 assert read(PACKET/'ASSESSMENT.json')==read(CONTROL/'ASSESSMENT.json')
 lines=(PACKET/'Slurm_accounting.txt').read_text().splitlines();rows=list(csv.DictReader(io.StringIO('\n'.join(s for s in lines if s and not s.startswith('#'))),delimiter='|'))
 job=next(x for x in rows if x['JobID']==JOB);batch=next(x for x in rows if x['JobID']==JOB+'.batch')
 assert all(x['State']=='COMPLETED' and x['ExitCode']=='0:0' for x in rows)
 assert job['Account']=='spinquest_standard' and job['AllocCPUS']=='8' and 'gpu' not in job['AllocTRES']
 def seconds(s):
  parts=[float(v) for v in s.split(':')];return sum(v*60**i for i,v in enumerate(parts[::-1]))
 wall,cpu=seconds(job['Elapsed']),seconds(job['TotalCPU'])
 opt=read(RESULT/'OPTIMIZER_COMPLETE.json');assert opt==result['optimization']
 costs=dict(job=JOB,elapsed_seconds=wall,TotalCPU_seconds=cpu,allocated_CPU_core_hours=8*wall/3600,CPU_efficiency_fraction=cpu/(8*wall),batch_MaxRSS_GiB=int(batch['MaxRSS'][:-1])/1024**2,
  peak_process_RSS_bytes=opt['peak_RSS_bytes'],peak_tree_RSS_bytes=supervisor['peak_tree_rss_bytes'],solver_seconds=opt['stage_times']['solver_seconds'],norm_subproblem_seconds=opt['binding_stats']['norm_subproblem_seconds'],
  norm_subproblem_fraction=opt['binding_stats']['norm_subproblem_seconds']/opt['stage_times']['solver_seconds'],gpus=0,continuations=0,accounting_steps=[x['JobID'] for x in rows],extern_row_returned=False,
  scope='One fit; no earlier profiling/exact-control costs included. Different inputs/trajectories prevent whole-fit acceleration comparison.')
 write('PACKET_VERIFICATION.json',dict(status='PASS',payload_files_verified=payload,transfer_files_identical_to_sent=transfer_count,driver_files_verified=driver_count,prep_files_verified=prep_count,acceleration_files_verified=acceleration_count,
  terminal_retention_files_verified=retained_count,original_scoring_source_pins=source_pins,baseline_bytes_match_original=True,scientific_input_and_draw_identity=True,
  direct_dependency_pins_verified_locally=verified_pins,remote_environment_pins_reported_only=remote_only,submission_control_pins_verified=control_pins,
  operational_adapter='Reviewed source adds fixed account and GPD job name only; exact once-only marker/pins retained. Real sbatch binary identity is remote-reported, not locally rehashed.',
  qualified_test_counts=counts,allocated_tests=37,allocated_skips=0,host_submission_tests_reported=4,external_adapter_tests_reported=7,fixture_tests_rerun_locally=False,
  single_clean_execution=True,costs=costs,limitations=['Full NLO maps, constraint derivatives, operator checks and optimizer trajectory were not recomputed locally.','Environment and Slurm binary provenance rely on pinned receiver records.','37 allocated plus4 host tests correct earlier41-allocated wording; no missing fixture group.']))
 print(json.dumps(dict(status='PASS',payloads=payload,transfer=transfer_count,retention=retained_count,allocated_tests=37,dependency_pins_local=len(verified_pins),dependency_pins_remote_only=len(remote_only),costs=costs),indent=2))
if __name__=='__main__':main()
