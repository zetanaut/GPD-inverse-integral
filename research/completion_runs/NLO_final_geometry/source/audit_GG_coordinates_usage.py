"""Audit saved certificates, inherited rows and completed family pointers only."""
from pathlib import Path
import json,hashlib,math
PREP=Path(__file__).resolve().parents[1];ROOT=PREP.parents[1];NLO=ROOT/'GPD_NLO_completion_package'
OLD=ROOT/'GPD_final_geometry_GG_local_repair';NEW=ROOT/'GPD_GG_coordinate_repair'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    assert not p.exists(),'Preserve completed audit'
    p.write_text(json.dumps(v,indent=2)+'\n')
def main():
    state=read(PREP/'geometry/xi0.505/GG_COORDINATE_PIPELINE.json');assert state['status']=='FAMILY_VALIDATED'
    old_state=read(PREP/'geometry/xi0.505/GG_LOCAL_PIPELINE.json');assert old_state['status']=='BLOCKED'
    b0=read(OLD/'ACTIVE_BUILD.json');b1=read(NEW/'ACTIVE_BUILD.json')
    parent={'build_key':b0['key'],'binary_sha256':b0['binary_sha256'],'algorithm_sha256':sha(OLD/'gg_local.py'),'activation_sha256':sha(OLD/'activation.py')}
    current={'build_key':b1['key'],'binary_sha256':b1['binary_sha256'],'algorithm_sha256':sha(NEW/'mapped_local.py'),'activation_sha256':sha(NEW/'coordinate_activation.py'),'parent':parent}
    for addon,b in ((OLD,b0),(NEW,b1)):
        v=read(addon/'LOCAL_VERIFICATION.json');assert v['pass'] and v['build_key']==b['key'] and sha(b['library'])==b['binary_sha256']
    step=next(s for s in old_state['steps'] if s['name']=='inner72' and s['returncode']==0)
    text=(PREP/'geometry/xi0.505'/step['log']).read_text()
    inner=Path([line[9:] for line in text.splitlines() if line.startswith('COMPLETE ')][-1])
    banks={'base':NLO/state['raw'],'inner72':inner,'output48':NLO/state['output48_raw']}
    keys={name:read(p/'KEY.json') for name,p in banks.items()}
    for k in keys.values():assert k['GG_local_fallback']==parent
    assert keys['output48']['GG_coordinate_fallback']==current
    records=[]
    for addon,identity in ((OLD,parent),(NEW,current)):
        for p in sorted((addon/'fallback_cache').glob('*.json')):
            item=read(p);assert item['key']['recipe']==identity
            ref=item['reference'];assert ref['xi']==.505 and ref['nf']==3 and ref['node_counts']==[24,48,96]
            assert item['value']==ref['local'] and math.isfinite(item['value'])
            assert all(math.isfinite(v) for pair in ref['integrals'] for v in pair)
            assert all(math.isfinite(v) and v<=limit for v,limit in zip(ref['max_successive_change'],ref['limits']))
            uses={name:[i for i,x in enumerate(k['output_coordinates']) if float(x).hex()==item['key']['x_hex']] for name,k in keys.items()}
            if addon==NEW:uses={name:indices if name=='output48' else [] for name,indices in uses.items()}
            assert any(uses.values())
            records.append({'file':str(p.relative_to(ROOT)),'sha256':sha(p),'addon':addon.name,'x':ref['x'],'xi':ref['xi'],
                            'local':ref['local'],'max_successive_change':ref['max_successive_change'],'limits':ref['limits'],
                            'raw_bank_row_indices':uses,'original_error':item['original_error']})
    for n,h in state['parent_row_hashes'].items():assert sha(NLO/state['parent']/n)==h and sha(banks['base']/n)==h
    for n,h in state['inherited_output_rows'].items():assert sha(NLO/state['reused_output']/n)==h and sha(banks['output48']/n)==h
    for name,h in state['binding'].items():assert sha(ROOT/name)==h
    result={'pass':True,'scope':'Saved certificate and hash audit only; no numerical evaluation',
            'fallback_records':records,'original_111_base_rows_preserved':True,'original_120_output_rows_preserved':True,
            'fallback_identities':{'parent':parent,'coordinate':current},
            'banks':{name:{'path':str(p.relative_to(ROOT)),'KEY_sha256':sha(p/'KEY.json'),'operators_sha256':sha(p/'operators.npz')} for name,p in banks.items()}}
    write(PREP/'results/GG_COORDINATE_USAGE.json',result)
    pointers=[]
    for xi in (.215,.155,.305,.365,.505):
        p=PREP/f'geometry/xi{xi}'/('GG_COORDINATE_PIPELINE.json' if xi==.505 else 'PIPELINE.json');s=read(p)
        assert s['status'] in ('FIRST_FAMILY_VALIDATED','FAMILY_VALIDATED') and len(s['validation'])==6 and all(v['pass'] for v in s['validation'])
        pointers.append({'xi':xi,'pipeline':str(p.relative_to(ROOT)),'sha256':sha(p),'status':s['status'],'directories':s['directories']})
    write(PREP/'geometry/FAMILY_POINTERS.json',{'families':pointers,'repair_lineage':{
        'original_pipeline':{'path':str((PREP/'geometry/xi0.505/PIPELINE.json').relative_to(ROOT)),'status':'BLOCKED'},
        'GG_local_pipeline':{'path':str((PREP/'geometry/xi0.505/GG_LOCAL_PIPELINE.json').relative_to(ROOT)),'status':'BLOCKED'},
        'coordinate_pipeline':'FAMILY_VALIDATED','original_failures_preserved':True,'old_output_bank':state['reused_output'],'new_output_bank':state['output48_raw']}})
    print(json.dumps({'pass':True,'fallback_certificates':len(records),'base_rows_inherited':111,'output_rows_inherited':120,'families':5},indent=2))
if __name__=='__main__':main()
