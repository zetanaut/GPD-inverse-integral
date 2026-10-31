from pathlib import Path
import sys,json,argparse
import numpy as np
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candidate'))
from forward import load,evaluate_field,hard_row

def main():
 ap=argparse.ArgumentParser();ap.add_argument('first',type=Path);ap.add_argument('second',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();bs=[];hs=[]
 for d in [a.first,a.second]:
  b,*_=load(d);bs.append(b)
  with np.load(d/'smooth_field.npz') as z:hs.append(z['evolved'].copy())
 edges=np.unique(np.r_[bs[0].edges,bs[1].edges]);z,w=leggauss(32);x=np.concatenate([(l+r)/2+(r-l)/2*z for l,r in zip(edges[:-1],edges[1:])]);ww=np.concatenate([(r-l)/2*w for l,r in zip(edges[:-1],edges[1:])]);fs=[evaluate_field(b,h,x) for b,h in zip(bs,hs)]
 errs=np.sqrt(((fs[0]-fs[1])**2)@ww)/np.maximum(1e-15,np.sqrt(fs[1]**2@ww));cff=[hard_row(b,2.3,96)[0]@h for b,h in zip(bs,hs)];delta=cff[0]-cff[1]
 # Each region's contribution to the squared field discrepancy is saved.
 region=[]
 for l,r in zip(edges[:-1],edges[1:]):
  mask=(x>l)&(x<r);region.append({'left':float(l),'right':float(r),'squared_L2':(((fs[0][:,mask]-fs[1][:,mask])**2)@ww[mask]).tolist()})
 result={'first':str(a.first),'second':str(a.second),'field_relative_L2':dict(zip(['Sigma','T_rescaled','Hg'],map(float,errs))),'cff_first':[float(cff[0].real),float(cff[0].imag)],'cff_second':[float(cff[1].real),float(cff[1].imag)],'CFF_difference':[float(delta.real),float(delta.imag)],'regions':region,'status':'inter-representation diagnostic; no independent reference claim'}
 a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
