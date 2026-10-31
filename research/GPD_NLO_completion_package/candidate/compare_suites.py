"""Compare saved forward fixtures; no source construction or evolution."""
from pathlib import Path
import sys,argparse,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candidate'))
from forward import load
from point_suite import compare
def main():
 ap=argparse.ArgumentParser();ap.add_argument('first',type=Path);ap.add_argument('second',type=Path);ap.add_argument('--q2',type=float,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();points=[json.loads((d/f'POINT_Q2_{a.q2:g}.json').read_text()) for d in [a.first,a.second]];out={}
 for label in ['weak10','strong10']:
  out[label]={}
  for truth in ['smooth','Dterm_added']:
   data=[]
   for p in points:
    b,*_=load(ROOT/p['directories'][label]);r=p['results'][label][truth]
    with np.load(ROOT/r['native_field']) as f:h=f['evolved'].copy()
    data.append((b,h,complex(*r['CFF'])))
   out[label][truth]=compare(*data,a.q2)
 result={'first':str(a.first),'second':str(a.second),'Q2':a.q2,'comparisons':out};a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({l:{t:r['CFF_difference'] for t,r in v.items()} for l,v in out.items()},indent=2))
if __name__=='__main__':main()
