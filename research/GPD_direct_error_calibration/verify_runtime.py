"""Verify existing dependencies only. Never downloads, builds or installs."""
import argparse,hashlib,json,os
from pathlib import Path

def verify(root):
 root=Path(root).resolve();lock=json.loads((Path(__file__).parent/'RUNTIME_LOCK.json').read_text());missing=[];changed=[]
 for rel,expected in lock['files'].items():
  p=root/rel
  if not p.is_file():missing.append(rel)
  elif hashlib.sha256(p.read_bytes()).hexdigest()!=expected:changed.append(rel)
 result={'pass':not missing and not changed,'checked':len(lock['files']),'missing':missing,'changed':changed}
 if not result['pass']:raise RuntimeError(json.dumps(result))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default=os.environ.get('GPD_REVIEW_ROOT'));a=p.parse_args()
 if not a.root:raise ValueError('Set GPD_REVIEW_ROOT to existing project')
 print(json.dumps(verify(a.root),indent=2))
