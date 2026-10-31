"""Small result packet. No runtime copy, training, or expensive replay."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parent
files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts and p.suffix not in ('.zip','.pyc') and not p.name.startswith('ACTIVE_PROCESS')]
manifest={str(p.relative_to(ROOT)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files if p.name!='MANIFEST.json'}
(ROOT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(ROOT.parent/'review_NLO_direct_error_smoke.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in files:
  z.write(p,'GPD_direct_error_calibration/'+str(p.relative_to(ROOT)))
 z.write(ROOT/'MANIFEST.json','GPD_direct_error_calibration/MANIFEST.json')
print(ROOT.parent/'review_NLO_direct_error_smoke.zip')
