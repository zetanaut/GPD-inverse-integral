"""Generate isolated output48 continuation; reuse all successful captured rows."""
from pathlib import Path
import ast,json,sys,hashlib
HERE=Path(__file__).resolve().parent
SOURCE_SHA='c3815e79354a6f10d139817f20dab519fb1435838d6918224a36b948d00428e3'
OLD_OUTPUT='3469941c41e7591df44efe185683c5039e6ce759f832b125bdb02a56f29ff21d'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def once(s,a,b):
    if s.count(a)!=1:raise ValueError('Unexpected source anchor '+a[:70])
    return s.replace(a,b,1)

def transform(s):
    s=once(s,"REPAIR=PREP.parents[1]/'GPD_final_geometry_GG_local_repair'","REPAIR=PREP.parents[1]/'GPD_GG_coordinate_repair'")
    s=once(s,'from activation import wrap_source,identity','from coordinate_activation import wrap_source,identity')
    s=once(s,"ap.add_argument('--benchmark',action='store_true');","ap.add_argument('--reuse-output',type=Path,required=True);ap.add_argument('--benchmark',action='store_true');")
    s=once(s,"assert old.get('GG_local_fallback')==identity(), 'Parent fallback signature differs'", "assert old.get('GG_local_fallback')==identity()['parent'], 'Parent fallback signature differs'")
    s=once(s,'sig=dict(old);',"sig=dict(old);sig['GG_coordinate_fallback']=identity();")
    anchor=" key=hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest();"
    addition=''' a.reuse_output=a.reuse_output.resolve()
 reuse=json.loads((a.reuse_output/'KEY.json').read_text())
 if a.reuse_output.name!='3469941c41e7591df44efe185683c5039e6ce759f832b125bdb02a56f29ff21d':raise ValueError('Unexpected incomplete output bank')
 for k in ('basis','inner_n','local_n','outer_n','source_build_key','fast_qg_key','prescription','runtime_repair_record_sha256','method_sources','GG_local_fallback','output_coordinates','weights','special_output_n','parent_key'):
  if reuse.get(k)!=sig.get(k):raise ValueError('Incompatible saved output field: '+k)
 if reuse.get('output_refinement_source')!='c3815e79354a6f10d139817f20dab519fb1435838d6918224a36b948d00428e3':raise ValueError('Unexpected parent source')
 saved_rows={p.name:digest(p) for p in sorted(a.reuse_output.glob('row_????.npz'))}
 if len(saved_rows)!=120:raise ValueError('Expected unchanged 120 finite rows in incomplete bank')
 sig['reused_output']={'key':a.reuse_output.name,'key_sha256':digest(a.reuse_output/'KEY.json'),'rows':saved_rows,'scope':'Nested fallback preserves every previous successful local evaluation'}
'''
    s=once(s,anchor,addition+anchor)
    s=once(s,"  if not dest.exists() and z.hex() in previous:",'''  if not dest.exists() and dest.name in saved_rows:
   src=a.reuse_output/dest.name
   if digest(src)!=saved_rows[dest.name]:raise ValueError('Saved output row changed')
   shutil.copy2(src,dest);entry['reused_output_sha256']=saved_rows[dest.name]
  if not dest.exists() and z.hex() in previous:''')
    ast.parse(s);return s

def main(project):
    project=Path(project).resolve();p=project/'completion_runs/NLO_final_geometry/source/refine_output_GG_local.py'
    if sha(p)!=SOURCE_SHA:raise ValueError('Original refiner changed')
    s=transform(p.read_text());dest=p.with_name('refine_output_GG_coordinates.py')
    if dest.exists() and dest.read_text()!=s:raise ValueError('Refuse to overwrite different new adapter')
    dest.write_text(s)
    result={'source':str(p),'original_sha256':sha(p),'new_file':str(dest),'sha256':sha(dest),'scope':'No row construction; new adapter only'}
    (HERE/'ASSEMBLY_ADAPTER.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':print(json.dumps(main(sys.argv[1] if len(sys.argv)>1 else HERE.parent),indent=2))
