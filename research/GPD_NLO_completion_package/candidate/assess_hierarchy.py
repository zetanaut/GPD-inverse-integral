"""No source evaluations: compare weak and strong discretizations from a bank.
This driver derives nested operators, saves native fields, then compares them.
"""
from pathlib import Path
import argparse,subprocess,sys,json,os
ROOT=Path(__file__).resolve().parents[1]
def run(script,*args):
 p=subprocess.run([sys.executable,str(ROOT/'candidate'/script),*map(str,args)],cwd=ROOT,text=True,capture_output=True,check=True);return p.stdout

def main():
 ap=argparse.ArgumentParser();ap.add_argument('parent',type=Path);a=ap.parse_args();a.parent=a.parent.resolve();out={'weak10':a.parent};logs={}
 for d in [6,8]:
  out[f'weak{d}']=Path(run('restrict_nested.py',a.parent,'--degree',d).strip())
 for d in [6,8,10]:
  out[f'strong{d}']=Path(run('collocation_reference.py',a.parent,'--degree',d).strip())
 for label,directory in out.items():
  logs[label]={}
  for script in ['check_pilot.py','check_api.py','check_physics.py']:logs[label][script]=run(script,directory)
 comparisons={}
 pairs=[('weak6','weak8'),('weak8','weak10'),('strong6','strong8'),('strong8','strong10'),('weak8','strong10'),('weak10','strong10')]
 for x,y in pairs:
  path=a.parent/f'COMPARE_{x}_{y}.json';run('compare_fields.py',out[x],out[y],'--out',path);comparisons[x+'_'+y]=json.loads(path.read_text())
 run('check_raw_moments.py',a.parent);run('check_saved_action.py',a.parent)
 result={'directories':{k:str(v.relative_to(ROOT)) for k,v in out.items()},'comparisons':comparisons,'independence':'Strong QR-selected collocation versus weak Galerkin; same kernel point functions and source quadrature. Source equations/actions, published moments, hard coefficients and finite parts have separate controls.'};(a.parent/'HIERARCHY.json').write_text(json.dumps(result,indent=2)+'\n');(a.parent/'HIERARCHY_LOGS.json').write_text(json.dumps(logs,indent=2)+'\n');print(json.dumps({k:{'CFF_difference':v['CFF_difference'],'fields':v['field_relative_L2']} for k,v in comparisons.items()},indent=2))
if __name__=='__main__':main()
