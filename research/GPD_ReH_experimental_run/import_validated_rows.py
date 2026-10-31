#!/usr/bin/env python3
"""Import source-keyed paired adjoint rows from the accepted row-export format.

No nearest-kinematics substitution, matrix construction, or new validation claim.
For new coordinates the existing NLO worker first exports/validates corresponding
weak10/strong10 rows with the same metadata contract.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from gpd_reh.inputs import digest,atomic_json,coordinate_key
from gpd_reh.bank import PHYSICS,read_index


def import_rows(source,directory):
    source=Path(source);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    existing=read_index(directory);entries={coordinate_key(e['xi'],e['Q2']):e for e in existing['entries']}
    with np.load(source,allow_pickle=False) as f:
        meta=json.loads(str(f['metadata_json'].item()))
        if meta.get('Q02')!=2. or meta.get('alpha0')!=PHYSICS['alpha0'] or meta.get('physics')!='fixed nf3 MSbar NLO':
            raise ValueError('Source export has a different perturbative prescription')
        if meta.get('input_channels')!=PHYSICS['input_channels'] or meta.get('output_components')!=PHYSICS['output_components']:
            raise ValueError('Source export channel/sign convention mismatch')
        pairs={}
        for r in meta['rows']:
            if r['method'] not in ('weak10','strong10'):continue
            key=coordinate_key(r['xi'],r['Q2']);method=r['method']
            if method in pairs.setdefault(key,{}):
                old=pairs[key][method]
                for suffix in ('_x','_weights'):
                    if not np.array_equal(f[old['prefix']+suffix],f[r['prefix']+suffix]):raise ValueError('Conflicting duplicate row')
            pairs[key][method]=r
        added=[]
        for key,pair in pairs.items():
            if set(pair)!= {'weak10','strong10'}:continue
            first=pair['weak10'];token=hashlib.sha256((digest(source)+'|'+key).encode()).hexdigest()[:20]
            name='row_'+token+'.npz';vname='row_'+token+'_provenance.json'
            arrays={'xi':np.array(first['xi']), 'Q2':np.array(first['Q2'])}
            for src,dst in (('weak10','nominal'),('strong10','reference')):
                rec=pair[src];x=f[rec['prefix']+'_x'].copy();w=f[rec['prefix']+'_weights'].copy()
                if w.shape!=(2,len(x),3):raise ValueError('Unexpected original row shape')
                arrays[dst+'_x']=x;arrays[dst+'_weights']=w.transpose(0,2,1)
            target=directory/name
            if key in entries:
                # No overwrite: compare arrays, even if compressed ZIP timestamps differ.
                with np.load(directory/entries[key]['file'],allow_pickle=False) as old:
                    if set(old.files)!=set(arrays) or any(not np.array_equal(old[k],v) for k,v in arrays.items()):
                        raise ValueError('Different existing row: preserve it under a separate bank directory')
                continue
            np.savez_compressed(target,**arrays)
            atomic_json(directory/vname,{'role':'inherited accepted adjoint-row evidence; not a new full-forward check',
                'parent_file':source.name,'parent_sha256':digest(source),'source_metadata':meta,
                'records':pair,'new_fit_validation_required':True})
            rec={'xi':float(first['xi']),'Q2':float(first['Q2']),'file':name,'sha256':digest(target),
                 'source_key':digest(source)+':'+key,'validation_file':vname,'validation_sha256':digest(directory/vname)}
            entries[key]=rec;added.append(key)
    atomic_json(directory/'index.json',{'physics':PHYSICS,'entries':list(entries.values())})
    return {'added':len(added),'total_pairs':len(entries),'parent_sha256':digest(source)}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(import_rows(a.source,a.out),indent=2))
