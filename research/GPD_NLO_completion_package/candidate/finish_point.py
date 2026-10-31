"""Execute the predeclared bounded controls for one constructed frozen point."""
from pathlib import Path
import sys,json,argparse,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('raw',type=Path);ap.add_argument('--q2',type=float,required=True);a=ap.parse_args();a.raw=a.raw.resolve();xi=json.loads((a.raw/'KEY.json').read_text())['basis']['xi'];tag=f'point_{xi:g}';wd=ROOT/'work'/tag;wd.mkdir(exist_ok=True);state={'xi':xi,'Q2':a.q2,'raw':str(a.raw.relative_to(ROOT)),'stages':[]};start=time.monotonic()
 def run(script,*args,label=None):
  name=label or script.removesuffix('.py');log=wd/(name+'.log');cmd=[sys.executable,str(ROOT/'candidate'/script),*map(str,args)];t=time.monotonic()
  try:
   with log.open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1300)
  except Exception as ex:
   state['failure']={'script':script,'error':repr(ex),'log':str(log.relative_to(ROOT))};(wd/'PIPELINE.json').write_text(json.dumps(state,indent=2)+'\n');raise
  text=log.read_text();state['stages'].append({'script':script,'command':cmd,'log':str(log.relative_to(ROOT)),'seconds':time.monotonic()-t});(wd/'PIPELINE.json').write_text(json.dumps(state,indent=2)+'\n');print('COMPLETE_STAGE',name,flush=True);return text
 base=Path(run('correct_qg_rows.py',a.raw,label='correct_base').strip());state['base']=str(base.relative_to(ROOT));run('point_suite.py',base,'--q2',a.q2,label='base_suite')
 for script in ['check_api.py','check_structure.py','hard_reference.py','check_evolved_moments.py','check_raw_moments.py','check_physics.py','check_saved_action.py','check_ns_charge.py']:
  args=['--q2',a.q2] if script in ['check_api.py','check_structure.py','hard_reference.py','check_evolved_moments.py'] else [];run(script,base,*args)
 raw72=Path(run('assemble_point.py','--xi',xi,'--degree',10,'--layout','log3','--q',72,'--outer',18,'--cap',1200,'--bridge','mp',label='inner72_build').strip().split('COMPLETE ')[-1]);q72=Path(run('correct_qg_rows.py',raw72,label='correct_inner72').strip());run('point_suite.py',q72,'--q2',a.q2,label='inner72_suite');run('compare_suites.py',base,q72,'--q2',a.q2,'--out',wd/'SOURCE_QUADRATURE.json',label='source_comparison')
 probe=run('refine_output.py',a.raw,'--special',48,'--cap',1200,'--benchmark',label='output48_benchmark');outpath=Path(probe.strip().split('COMPLETE ')[-1]);benchmark=json.loads((outpath/'BENCHMARK.json').read_text());plan={'mechanism':'Only crossover output quadrature24 to48; same basis, source inner48 and unchanged outer rows reused. Analytic QG subtraction follows.','benchmark':benchmark,'measured_pilot_refinement_seconds':532.102,'forecast_seconds':[400,1000],'finite_cap_seconds':1200,'row_checkpoints':True};(wd/'OUTPUT_STRATEGY.json').write_text(json.dumps(plan,indent=2)+'\n')
 raw48=Path(run('refine_output.py',a.raw,'--special',48,'--cap',1200,label='output48_build').strip().split('COMPLETE ')[-1]);out48=Path(run('correct_qg_rows.py',raw48,label='correct_output48').strip());run('point_suite.py',out48,'--q2',a.q2,label='output48_suite');run('compare_suites.py',base,out48,'--q2',a.q2,'--out',wd/'OUTPUT_QUADRATURE.json',label='output_comparison');run('budget_suite.py',base/f'POINT_Q2_{a.q2:g}.json',wd/'SOURCE_QUADRATURE.json',wd/'OUTPUT_QUADRATURE.json','--out',wd/'BUDGET.json',label='budget')
 state.update(base=str(base.relative_to(ROOT)),inner72=str(q72.relative_to(ROOT)),output48=str(out48.relative_to(ROOT)),seconds=time.monotonic()-start,status='completed measurements');(wd/'PIPELINE.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state,indent=2))
if __name__=='__main__':main()
