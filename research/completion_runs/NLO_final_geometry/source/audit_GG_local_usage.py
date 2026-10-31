"""Audit saved fallback certificates and row lineage without evaluating kernels."""
from pathlib import Path
import json,hashlib,math
PREP=Path(__file__).resolve().parents[1];ROOT=PREP.parents[1];NLO=ROOT/'GPD_NLO_completion_package'
ADDON=ROOT/'GPD_final_geometry_GG_local_repair'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 state=read(PREP/'geometry/xi0.505/GG_LOCAL_PIPELINE.json');assert state['status']=='FAMILY_VALIDATED'
 build=read(ADDON/'ACTIVE_BUILD.json');verification=read(ADDON/'LOCAL_VERIFICATION.json');assert verification['pass'] and verification['build_key']==build['key']
 identity={'build_key':build['key'],'binary_sha256':build['binary_sha256'],
           'algorithm_sha256':sha(ADDON/'gg_local.py'),'activation_sha256':sha(ADDON/'activation.py')}
 assert sha(build['library'])==identity['binary_sha256']
 banks={}
 for name in ('base','inner72','output48'):
  step=next(s for s in state['steps'] if s['name']==name and s['returncode']==0)
  text=(PREP/'geometry/xi0.505'/step['log']).read_text()
  path=Path([line[9:] for line in text.splitlines() if line.startswith('COMPLETE ')][-1])
  key=read(path/'KEY.json');assert key['GG_local_fallback']==identity
  banks[name]=(path,key)
 rows=[]
 for p in sorted((ADDON/'fallback_cache').glob('*.json')):
  item=read(p);assert item['key']['recipe']==identity
  ref=item['reference'];assert ref['xi']==.505 and ref['nf']==3 and ref['node_counts']==[24,48,96]
  assert math.isfinite(item['value']) and item['value']==ref['local']
  assert all(math.isfinite(v) for pair in ref['integrals'] for v in pair)
  assert all(error<=limit for error,limit in zip(ref['max_successive_change'],ref['limits']))
  uses={name:[i for i,x in enumerate(key['output_coordinates']) if float(x).hex()==item['key']['x_hex']] for name,(_,key) in banks.items()}
  assert any(uses.values()),'Fallback coordinate outside the constructed family'
  rows.append({'cache_file':str(p.relative_to(ROOT)),'sha256':sha(p),'x':ref['x'],'xi':ref['xi'],
               'local':ref['local'],'max_successive_change':ref['max_successive_change'],'limits':ref['limits'],
               'raw_bank_row_indices':uses,'original_error':item['original_error']})
 parent=NLO/state['parent'];assert len(state['parent_row_hashes'])==111
 for name,value in state['parent_row_hashes'].items():
  assert sha(parent/name)==value and sha(banks['base'][0]/name)==value
 result={'pass':True,'scope':'Saved certificate/hash inspection only; no native evaluation',
         'fallback_identity':identity,'unique_fallback_coordinates':len(rows),'records':rows,
         'original_111_rows_bitwise_reused':True,'old_failure_sha256':sha(parent/'FAILURE.json'),
         'banks':{name:{'path':str(path.relative_to(ROOT)),'KEY_sha256':sha(path/'KEY.json'),'operators_sha256':sha(path/'operators.npz')} for name,(path,key) in banks.items()}}
 (PREP/'results/GG_LOCAL_USAGE.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'pass':True,'unique_fallback_coordinates':len(rows),'original_rows_reused':111},indent=2))
if __name__=='__main__':main()
