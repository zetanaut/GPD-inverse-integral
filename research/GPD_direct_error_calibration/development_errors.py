"""POST-FIT diagnostic on already exposed development truths, never calibration.
Formula copied exactly from the locked pilot/neural_pilot.py:truth_values.
Does not feed targets, truth labels, or errors back into the estimator.
"""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent/'results'
def truth(x,xi,dterm):
 h=np.array([2*x*(1-x*x)**2*(1+.2*xi*xi),.5*x*(1-x*x)**2*(1-.1*xi*xi),.8*(1-x*x)**2*(1+.1*xi*xi)])
 if dterm:
  z=x/xi;e=np.where(abs(z)<1,(1-z*z)**2,0.);h+=np.array([.09*z*e,-.02*z*e,.03*xi*e])
 return h
out={}
for ident,dterm in [('d000',False),('d001',True)]:
 r=json.loads((R/f'development_{ident}/RESULT.json').read_text());vals=[]
 for edges,power,norm in [([.2,.275,.4],0,.2),([0,.275,1],1,1)]:
  z,w=np.polynomial.legendre.leggauss(192);value=0.
  for a,b in zip(edges[:-1],edges[1:]):
   x=(a+b)/2+(b-a)*z/2;h=truth(x,.275,dterm)
   value+=np.sum((b-a)*w/2*x**power*(2/9)*(h[0]+h[1]))/norm
  vals.append(float(value))
 fit=r['selected']['targets'];out[ident]={'true_targets':vals,'fitted_targets':fit,'signed_error':(np.array(fit)-vals).tolist(),'phase':'development','never_used_for_selection':True}
(R/'POSTFIT_DEVELOPMENT_ERRORS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
