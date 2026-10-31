"""Independent spin-2 identities on unprojected analytic source actions.
Separates source quadrature from endpoint-constrained field projection.
"""
from pathlib import Path
import sys,json,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candidate'))
from forward import initial
from basis_factory import Elements

def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();key=json.loads((a.directory/'KEY.json').read_text());bs=key['basis'];b=Elements(bs['xi'],bs['degree'],bs['layout']);v=initial(b);nq=len(b.qids);x=np.array(key['output_coordinates']);w=np.array(key['weights']);actions=[]
 for i in range(len(x)):
  f=a.directory/f'row_{i:04}.npz'
  if not f.is_file():raise RuntimeError('Complete saved rows required; no source evaluation in this check')
  with np.load(f) as z:actions.append({k:z[k].copy() for k in z.files})
 mq=float((w*x)@b.basis(x,'odd')@v[:nq]);mg=float(w@b.basis(x,'even')@v[2*nq:]);mt=float((w*x)@b.basis(x,'odd')@v[nq:2*nq]);out=[]
 for order in [0,1]:
  A=16/9 if order==0 else 2000/243;B=1 if order==0 else 611/108;N=16/9 if order==0 else 2360/243
  for sector,coeff,weight,ref in [('QQ_singlet',v[:nq],w*x,-A*mq),('QG',v[2*nq:],w,B*mg),('GQ',v[:nq],w,A*mq),('GG',v[2*nq:],w,-B*mg),('NS_plus',v[nq:2*nq],w*x,-N*mt)]:
   # QG outputs quarks: its moment has an x weight.
   if sector=='QG':weight=w*x
   values=np.array([r[f'K{order}_{sector}']@coeff for r in actions]);got=float(weight@values);out.append({'order':order,'sector':sector,'action_moment':got,'independent_reference':ref,'absolute_error':abs(got-ref)})
 result={'operator_key':a.directory.name,'rows':out,'source':'https://arxiv.org/pdf/1012.4409 Eqs7,8 Table1','scope':'unprojected continuous analytic-input source actions; full support; independently published spin2 anomalous dimensions'}
 (a.directory/'RAW_MOMENT_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
