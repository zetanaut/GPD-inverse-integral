"""Saved-source coupled smooth/D-term forward validation at one frozen point.
This produces measurements, not an automatic scientific certificate.
"""
from pathlib import Path
import sys,json,argparse,subprocess,hashlib,time
import numpy as np
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from forward import load,initial,evaluate_field,hard_row
from src.ordered_evolution import evolve
from frozen.python.nlo_forward import evolve_field

def components(z):return [float(np.real(z)),float(np.imag(z))]
def derive(script,parent,d):return Path(subprocess.check_output([sys.executable,str(ROOT/'candidate'/script),str(parent),'--degree',str(d)],cwd=ROOT,text=True).strip())
def compare(first,second,q2):
 b,h,c=first;bb,hh,cc=second;edges=np.unique(np.r_[b.edges,bb.edges]);norms={}
 for n in [32,64,128]:
  z,w=leggauss(n);x=np.concatenate([(l+r)/2+(r-l)/2*z for l,r in zip(edges[:-1],edges[1:])]);ww=np.concatenate([(r-l)/2*w for l,r in zip(edges[:-1],edges[1:])]);f=evaluate_field(b,h,x);ref=evaluate_field(bb,hh,x);den=np.sqrt(ref**2@ww);err=np.sqrt((f-ref)**2@ww);norms[n]={'relative':(err/np.maximum(1e-15,den)).tolist(),'absolute_full_support':(np.sqrt(2)*err).tolist(),'reference_full_support':(np.sqrt(2)*den).tolist()}
 delta=np.abs(np.array(components(c-cc)));return {'CFF_difference':components(c-cc),'CFF_absolute_error':delta.tolist(),'CFF_component_threshold':(1e-4*np.maximum(1,np.abs(components(cc)))).tolist(),'CFF_component_pass':(delta<=1e-4*np.maximum(1,np.abs(components(cc)))).tolist(),'field_relative_L2':dict(zip(['Sigma','T_rescaled','Hg'],norms[128]['relative'])),'field_channel_pass':(np.array(norms[128]['relative'])<=1e-3).tolist(),'L2_quadrature':norms,'parity':'positive-half physical integration; full-support norms use exact odd/even parity'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('parent',type=Path);ap.add_argument('--q2',type=float,required=True);a=ap.parse_args();a.parent=a.parent.resolve();start=time.monotonic();dirs={'weak10':a.parent}
 for d in [6,8]:dirs[f'weak{d}']=derive('restrict_nested.py',a.parent,d)
 for d in [6,8,10]:dirs[f'strong{d}']=derive('collocation_reference.py',a.parent,d)
 data={};results={};hard={};gradients={};dterm={}
 for label,directory in dirs.items():
  b,k0,k1,key=load(directory);results[label]={};data[label]={}
  if b.degree not in hard:hard[b.degree]={n:hard_row(b,a.q2,n)[0] for n in [48,96,192]}
  rows=hard[b.degree]
  for truth,dt in [('smooth',False),('Dterm_added',True)]:
   v=initial(b,dt);h,diag=evolve(v,k0,k1,a.q2,rtol=1e-11,atol=1e-13);fname=f'{truth}_Q2_{a.q2:g}.npz';np.savez_compressed(directory/fname,initial=v,evolved=h,nodes=b.nodes);c=rows[192]@h;data[label][truth]=(b,h,c)
   tight,_=evolve(v,k0,k1,a.q2,rtol=1e-13,atol=1e-15);rk=evolve_field(v,k0,k1,a.q2,steps=128)
   results[label][truth]={'CFF':components(c),'native_field':str((directory/fname).relative_to(ROOT)),'field_sha256':hashlib.sha256((directory/fname).read_bytes()).hexdigest(),'hard_48_96_component_change':np.abs(components((rows[96]-rows[48])@h)).tolist(),'hard_96_192_component_change':np.abs(components((rows[192]-rows[96])@h)).tolist(),'ordered_tight_CFF_change':np.abs(components(rows[192]@(h-tight))).tolist(),'independent_RK4_CFF_change':np.abs(components(rows[192]@(h-rk))).tolist(),'ordered_field_max_abs':float(max(np.max(abs(h-tight)),np.max(abs(h-rk)))),'evolution':diag}
  if label in ['weak10','strong10']:
   v=initial(b);h=data[label]['smooth'][1];adj,_=evolve(rows[192],k0,k1,a.q2,rtol=1e-12,atol=1e-14,adjoint=True);np.savez_compressed(directory/f'jacobian_Q2_{a.q2:g}.npz',real=adj.real,imag=adj.imag)
   grad=[];rng=np.random.default_rng(731);nq=len(b.qids)
   for name,sl in [('Sigma',slice(0,nq)),('T_rescaled',slice(nq,2*nq)),('Hg',slice(2*nq,len(v)))]:
    direction=np.zeros_like(v);direction[sl]=rng.normal(size=len(direction[sl]));direction/=np.linalg.norm(direction)
    for eps in [1e-5,1e-6]:
     hp,_=evolve(v+eps*direction,k0,k1,a.q2,rtol=1e-12,atol=1e-14);hm,_=evolve(v-eps*direction,k0,k1,a.q2,rtol=1e-12,atol=1e-14);fd=rows[192]@(hp-hm)/(2*eps);grad.append({'channel':name,'step':eps,'analytic':components(adj@direction),'finite_difference':components(fd),'absolute_error':float(abs(fd-adj@direction))})
   gradients[label]={'bilinear_adjoint_error':float(abs(adj@v-rows[192]@h)),'directions':grad}
   dv=initial(b,True)-v;dh,_=evolve(dv,k0,k1,a.q2,rtol=1e-12,atol=1e-14);diff=data[label]['Dterm_added'][1]-h;dterm[label]={'coefficients':{'Sigma_qD':.09,'T_qD':-.02,'Hg_gD':.03},'linearity_field_max_abs':float(np.max(abs(diff-dh))),'linearity_CFF_absolute_error':float(abs(rows[192]@(diff-dh))),'evolved_Dterm_CFF':components(rows[192]@dh),'method':'evolved with coupled fields and convolved once; no separate subtraction constant'}
  print('SAVED',label,flush=True)
 comparisons={}
 for truth in ['smooth','Dterm_added']:
  comparisons[truth]={}
  for one,two in [('weak6','weak8'),('weak8','weak10'),('strong6','strong8'),('strong8','strong10'),('weak10','strong10')]:comparisons[truth][one+'_'+two]=compare(data[one][truth],data[two][truth],a.q2)
 result={'xi':b.xi,'Q02':2.,'Q2':a.q2,'directories':{k:str(v.relative_to(ROOT)) for k,v in dirs.items()},'results':results,'comparisons':comparisons,'derivatives':gradients,'Dterm_once':dterm,'seconds':time.monotonic()-start,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'status':'measurements; independent controls and cumulative error budget required'};(a.parent/f'POINT_Q2_{a.q2:g}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({t:{k:v['CFF_difference'] for k,v in comp.items()} for t,comp in comparisons.items()},indent=2))
if __name__=='__main__':main()
