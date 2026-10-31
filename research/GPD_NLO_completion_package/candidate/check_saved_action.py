"""Contract saved source rows with exact polynomial input; compare independent
rational K0 action controlling retained-order NLO factorization. No new kernels.
"""
from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candidate'))
from basis_factory import Elements
from analytic_scale_control import action
from forward import initial

def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();key=json.loads((a.directory/'KEY.json').read_text());bs=key['basis'];b=Elements(bs['xi'],bs['degree'],bs['layout']);v=initial(b);nq=len(b.qids);rows=[]
 for i,x in enumerate(key['output_coordinates']):
  f=a.directory/f'row_{i:04}.npz'
  if not f.exists():continue
  with np.load(f) as z:val=z['K0_QQ_singlet']@v[:nq]+z['K0_NS_plus']@v[nq:2*nq]+z['K0_QG']@v[2*nq:]
  ref=float(action(x,b.xi));rows.append({'i':i,'x':x,'action':float(val),'independent_reference':ref,'absolute_error':float(abs(val-ref))})
 result={'rows':rows,'evaluated':len(rows),'expected':len(key['output_coordinates']),'max_absolute_error':max(r['absolute_error'] for r in rows),'reference_source_sha256':hashlib.sha256((ROOT/'candidate/analytic_scale_control.py').read_bytes()).hexdigest(),'scope':'analytic full-support K0 quark/gluon action inside NLO normalization control; no evolved-reference claim'}
 (a.directory/'SAVED_ACTION_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='rows'})
if __name__=='__main__':main()
