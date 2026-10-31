"""Structural audit of saved final maps; no model, optimization, truth or noise."""
from pathlib import Path
import json,hashlib
import fit_final_geometry as f
a=f.a;np,torch=a.np,a.torch

def verify():
 geometry=json.loads((f.PREP/'FINAL_GEOMETRY.json').read_text())
 bank=f.load_bank();old=a.original_bank()
 expected=[(xi,q) for xi in geometry['fit_xi'] for q in geometry['fit_Q2']]
 assert [(p['xi'],p['Q2']) for p in bank.points]==expected and bank.n_scalar==30
 with np.load(f.PREP/'inputs/fit_bank.npz',allow_pickle=False) as z:
  assert tuple(z['row_ids'])==bank.ids
 reused=0
 for point in old.points:
  new=next(p for p in bank.points if (p['xi'],p['Q2'])==(point['xi'],point['Q2']))
  assert np.array_equal(new['x'],point['x']) and np.array_equal(new['rows'],point['rows'])
  reused+=1
 assert reused==9
 data=torch.load(f.PREP/'inputs/contractions.pt',weights_only=True)
 wanted=expected+[(xi,q) for xi in geometry['prediction_diagnostic_xi'] for q in geometry['prediction_diagnostic_Q2']]
 assert [(p['xi'],p['Q2']) for p in data['points']]==wanted
 domain=a.Domain(data['points'],data['spin2']);assert len(domain.names)==len(set(domain.names))==265
 for point in data['points']:
  q=str(point['Q2'])
  for label,entry in point['data'].items():
   for value in (entry['U'][q],entry['hard'][q],entry['hard_delta'][q],entry['M']):assert torch.isfinite(value).all()
 points=json.loads((f.PREP/'inputs/GEOMETRY_POINTS.json').read_text())
 assert len(points)==8 and sum(p['role']=='likelihood' for p in points)==5
 result={'pass':True,'scalar_likelihood_rows':30,'scalar_diagnostic_rows':18,'check_points':24,
         'constraint_count':265,'existing_complex_fit_maps_bitwise_reused':9,
         'explicit_row_ids_verified':True,'all_saved_maps_finite':True,
         'fit_bank_sha256':f.sha(f.PREP/'inputs/fit_bank.npz'),
         'contractions_sha256':f.sha(f.PREP/'inputs/contractions.pt'),
         'model_evaluations':0,'noise_draws':0,'final_truth_evaluations':0}
 (f.PREP/'results/CONSTRUCTED_GEOMETRY_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':verify()
