"""Independent published spin-2 evolution and D-term support on saved fields."""
from pathlib import Path
import sys,json,argparse,numpy as np
from scipy.integrate import solve_ivp
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candidate'))
from forward import load,evaluate_field
def moment(b,h):
 x,w=b.outputs(96);f=evaluate_field(b,h,x);return np.array([(w*x)@f[0],w@f[2],(w*x)@f[1]])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('parent',type=Path);ap.add_argument('--q2',type=float,required=True);a=ap.parse_args();point=json.loads((a.parent/f'POINT_Q2_{a.q2:g}.json').read_text());rows=[];support={};m0=np.array([[-16/9,1],[16/9,-1]]);m1=np.array([[-2000/243,611/108],[2000/243,-611/108]])
 def rhs(t,v):
  z=v[0];qg=(2*z*m0+4*z*z*m1)@v[1:3];ns=(2*z*(-16/9)+4*z*z*(-2360/243))*v[3];return np.r_[-9*z*z-64*z**3,qg,ns]
 for label,directory in point['directories'].items():
  b,*_=load(ROOT/directory);fields={}
  for truth in ['smooth','Dterm_added']:
   with np.load(ROOT/point['results'][label][truth]['native_field']) as f:hi=f['initial'].copy();hf=f['evolved'].copy()
   before=moment(b,hi);after=moment(b,hf);sol=solve_ivp(rhs,[0,np.log(a.q2/2.)],np.r_[.35691594851514724/(4*np.pi),before],rtol=1e-12,atol=1e-14,method='DOP853');ref=sol.y[1:,-1];rows.append({'method':label,'truth':truth,'initial_moments':before.tolist(),'evolved_moments':after.tolist(),'independent_evolved_reference':ref.tolist(),'absolute_errors':abs(after-ref).tolist(),'relative_errors':(abs(after-ref)/np.maximum(1e-15,abs(ref))).tolist(),'total_momentum_change':float(sum(after[:2])-sum(before[:2]))});fields[truth]=hf
  if label in ['weak10','strong10']:
   d=fields['Dterm_added']-fields['smooth'];z,w=leggauss(96);x=(1+b.xi)/2+(1-b.xi)/2*z;f=evaluate_field(b,d,x);support[label]={'DGLAP_field_max_absolute':np.max(abs(f),axis=1).tolist(),'DGLAP_field_L2_absolute':np.sqrt((f*f)@(w*(1-b.xi)/2)).tolist(),'Dterm_CFF_imaginary':float(point['results'][label]['Dterm_added']['CFF'][1]-point['results'][label]['smooth']['CFF'][1]),'analytic_DGLAP_and_imaginary_reference':0.}
 result={'xi':b.xi,'Q2':a.q2,'moment_order':['Mq=int_0^1 x Sigma','Mg=int_0^1 Hg','MT=int_0^1 x T_rescaled'],'source':'https://arxiv.org/pdf/1012.4409 Eqs7,8 Table1','independence':'Solve published local-operator moment ODE directly with two-loop running, independent of field generators/projections. No conservation enforced on fields.','rows':rows,'Dterm_support':support};(a.parent/'EVOLVED_MOMENTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
