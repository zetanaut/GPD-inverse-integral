"""Reuse only finite local callbacks with an exact source/x/xi/order/sector key."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
class CachedSource:
 def __init__(self,source,key):
  self.source=source;self.key=key;self.directory=ROOT/'work/local_cache'/key;self.directory.mkdir(parents=True,exist_ok=True)
  fixture=ROOT/'work/action/MP_LOCAL_CHECKS.json'
  if fixture.exists():
   data=json.loads(fixture.read_text())
   if data['source_key']==key:
    for row in data['rows']:
     if 'source_local' in row:
      self.save(1,row['sector'],row['x'],.135,row['source_local'],{'source_fixture':str(fixture.relative_to(ROOT)),'sha256':hashlib.sha256(fixture.read_bytes()).hexdigest()})
 def path(self,order,sector,x,xi):
  k={'source':self.key,'x':float(x).hex(),'xi':float(xi).hex(),'order':order,'sector':sector};h=hashlib.sha256(json.dumps(k,sort_keys=True).encode()).hexdigest();return self.directory/(h+'.json'),k
 def save(self,order,sector,x,xi,value,origin):
  p,k=self.path(order,sector,x,xi)
  if not p.exists():p.write_text(json.dumps({'key':k,'value':value,'origin':origin},indent=2)+'\n')
 def evaluate(self,order,sector,x,xi,y,include_local=False):
  if not include_local:return self.source.evaluate(order,sector,x,xi,y,False)
  dest,key=self.path(order,sector,x,xi)
  if dest.exists():
   data=json.loads(dest.read_text());assert data['key']==key;r,p,_=self.source.evaluate(order,sector,x,xi,y,False);return r,p,data['value']
  r,p,local=self.source.evaluate(order,sector,x,xi,y,True);self.save(order,sector,x,xi,local,{'method':'original error-controlled callback; finite checked by SourceParts'});return r,p,local
