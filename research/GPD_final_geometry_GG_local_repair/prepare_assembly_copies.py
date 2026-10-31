"""Create two isolated assembly adapters; no library, row or original source is edited."""
from pathlib import Path
import ast,hashlib,json,sys
HERE=Path(__file__).resolve().parent
ASM_SHA='184886bca924be7e55ffe6ad93eecfb2bf2ed1ba13238d770f6c72e8e190de84'
REF_SHA='fde1e85b5aa433aa7a16cdb23c0445aeda56fc6196347e7ea06ae8846c1aa11b'

def replace_one(s,old,new):
    if s.count(old)!=1:raise ValueError('Source anchor missing/duplicated: '+old[:60])
    return s.replace(old,new,1)

def assembly_copy(s):
    s=replace_one(s,'from project_enriched import project',
       "from project_enriched import project\nREPAIR=PREP.parents[1]/'GPD_final_geometry_GG_local_repair'\nsys.path.insert(0,str(REPAIR))\nfrom activation import wrap_source,identity")
    s=replace_one(s,"p=CachedSource(p,meta['key']);qmeta", "p=wrap_source(CachedSource(p,meta['key']));qmeta")
    s=replace_one(s,"reused={}\n", "sig['GG_local_fallback']=identity()\n reused={}\n")
    start=s.index("  guard=json.loads((ROOT/'work/action/FAST_BLOCK_CHECK.json').read_text())")
    end=s.index("  for f in sorted(a.reuse_parent.glob('row_????.npz')):",start)
    s=s[:start]+'''  # Passthrough proof: every prior SUCCESS is returned unchanged by the
  # conditional fallback; only the formerly nonfinite GG-local branch changes.
  if parent['fast_qg_key']!=sig['fast_qg_key']:raise ValueError('Changed fast QG')
  if parent['method_sources']!=sig['method_sources']:raise ValueError('Changed physical method')
  if parent.get('runtime_repair_record_sha256')!=sig['runtime_repair_record_sha256']:raise ValueError('Runtime repair changed')
  if parent.get('assembly_adapter_sha256')!='184886bca924be7e55ffe6ad93eecfb2bf2ed1ba13238d770f6c72e8e190de84':raise ValueError('Unexpected parent assembly')
  if parent.get('GG_local_fallback') is not None:raise ValueError('Only original-success rows may be inherited')
''' + s[end:]
    s=s.replace('Only QG guard arithmetic changes; 360 points and two complete source blocks identical. All other methods, geometry, prescriptions and local callbacks are the same.',
      'Original finite native/cached calls are returned unchanged. Only previously rejected NLO GG local calls use the source-keyed independent paired evaluator. Physical kernels and every inherited row byte are unchanged.')
    return s

def refinement_copy(s):
    s=replace_one(s,"ROOT=Path(__file__).resolve().parents[1];", "PREP=Path(__file__).resolve().parents[1];ROOT=PREP.parents[1]/'GPD_NLO_completion_package';")
    s=replace_one(s,'from project_enriched import project',
        "from project_enriched import project\nREPAIR=PREP.parents[1]/'GPD_final_geometry_GG_local_repair'\nsys.path.insert(0,str(REPAIR))\nfrom activation import wrap_source,identity")
    s=replace_one(s,"p=FastQG(CachedSource(SourceParts(ROOT/'work/mp_guard_bridge'/old['source_build_key']/'parts.dylib'),old['source_build_key']),", "p=FastQG(wrap_source(CachedSource(SourceParts(ROOT/'work/mp_guard_bridge'/old['source_build_key']/'parts.dylib'),old['source_build_key'])),")
    s=replace_one(s,'sig=dict(old);',"assert old.get('GG_local_fallback')==identity(), 'Parent fallback signature differs'\n sig=dict(old);")
    return s

def main(project):
    root=Path(project).resolve();src=root/'completion_runs/NLO_final_geometry/source'
    files=[(src/'assemble_missing.py',ASM_SHA,src/'assemble_GG_local.py',assembly_copy),
           (root/'GPD_NLO_completion_package/candidate/refine_output.py',REF_SHA,src/'refine_output_GG_local.py',refinement_copy)]
    report=[]
    for old,h,new,transform in files:
        raw=old.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Unexpected source '+str(old))
        output=transform(raw.decode());ast.parse(output)
        if new.exists() and new.read_text()!=output:raise ValueError('Refuse to replace different adapter '+str(new))
        new.write_text(output)
        report.append({'source':str(old),'original_sha256':h,'new_file':str(new),'new_sha256':hashlib.sha256(output.encode()).hexdigest()})
    (HERE/'ASSEMBLY_COPIES.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
if __name__=='__main__':print(json.dumps(main(sys.argv[1] if len(sys.argv)>1 else HERE.parent),indent=2))
