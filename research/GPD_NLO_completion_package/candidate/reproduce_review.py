"""Replay all packaged full-support forward fixtures; no kernel construction."""
from pathlib import Path
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candidate'))
from forward import load,initial,forward,jacobian
from hard_reference import reference
from point_suite import compare

def main():
 manifest=ROOT/'REVIEW_MANIFEST.json'
 if manifest.exists():
  for name,r in json.loads(manifest.read_text()).items():
   if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=r['sha256']:raise RuntimeError('Review hash mismatch '+name)
 result=json.loads((ROOT/'work/RESULT.json').read_text());rows=[]
 for point in result['acceptance_fixtures']:
  p=json.loads((ROOT/point['point_evidence']).read_text());q2=p['Q2'];bud=json.loads((ROOT/point['budget_evidence']).read_text())
  for truth,dt in [('smooth',False),('Dterm_added',True)]:
   vals={};fields={}
   for label in ['weak10','strong10']:
    directory=ROOT/p['directories'][label];b,*_=load(directory);v=initial(b,dt);c,h,_=forward(directory,q2,v);r=p['results'][label][truth];expected=complex(*r['CFF']);replay=abs(c-expected)
    if replay>1e-9:raise RuntimeError('Saved forward replay changed')
    with np.load(ROOT/r['native_field']) as saved:
     if np.max(abs(h-saved['evolved']))>1e-10:raise RuntimeError('Saved native field replay changed')
    fields[label]=(b,h,c)
    href,diag=reference(b,h,q2,1e-9)
    if diag['warnings'] or abs(c-href)>1e-7:raise RuntimeError('Independent continuum coefficient check failed')
    j,_=jacobian(directory,q2)
    if np.max(abs(j@v-np.array([c.real,c.imag])))>1e-8:raise RuntimeError('Forward/adjoint map changed')
    vals[label]=c
   err=np.abs([float((vals['weak10']-vals['strong10']).real),float((vals['weak10']-vals['strong10']).imag)]);lim=1e-4*np.maximum(1,np.abs([vals['strong10'].real,vals['strong10'].imag]))
   if not np.all(err<=lim) or not all(bud['fixtures'][truth]['CFF_component_pass']):raise RuntimeError('Frozen component acceptance failed')
   fc=compare(fields['weak10'],fields['strong10'],q2)
   if not all(fc['field_channel_pass']):raise RuntimeError('Physical field L2 failed')
   rows.append({'field_relative_L2':fc['field_relative_L2'],'xi':p['xi'],'Q2':q2,'truth':truth,'CFF_candidate_reference_difference':err.tolist(),'status':'passed saved-operator replay and independent hard integral'})
 print(json.dumps({'status':'passed','fixtures':rows,'no_kernel_construction':True,'scope':'Reproduction of saved operator milestone; source-build and independent action evidence are separately hashed in the review.'},indent=2))
if __name__=='__main__':main()
