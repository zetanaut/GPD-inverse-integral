"""Post-selection boundary/support diagnostics; never imported by the fitter."""
import json
import time
from pathlib import Path


def score(problem, flat, out):
    out=Path(out)
    selection=out/'SELECTION.json'
    if not selection.is_file():
        raise RuntimeError('Seal objective/eligibility selection before forward truth scoring')
    from p2b_policy import sha
    choice=json.loads(selection.read_text())
    if choice.get('scored') is None:
        raise RuntimeError('No sealed scored candidate')
    import numpy as np
    import torch
    from closure.truths import Truth
    start=time.monotonic();truth=Truth('Z1')
    def field(x):
        with torch.no_grad():
            return problem.field(flat,torch.tensor(x[None],dtype=torch.float64),
                                 torch.zeros(1,dtype=torch.float64),torch.zeros(1,dtype=torch.float64))[0].numpy().T
    info=problem.information
    with torch.no_grad():
        prediction=info.prediction(flat).numpy().reshape(12,3)
        observed=info.observed.numpy().reshape(12,3)
        whitened=(info.residual(flat)*160**.5).numpy()
    # A scoring grid is not a new observation and does not add fitting residuals.
    grid=np.linspace(.02,.8,201)
    pred,central=field(grid),truth.values(grid,0.,0.)
    relative=np.sqrt(np.sum((pred-central)**2,axis=0)/np.sum(central**2,axis=0))
    regions={}
    for name,lo,hi in [('low_x_tail',0.,.02),('observed_support',.02,.8),('high_x_tail',.8,1.)]:
        orders={}
        for order in (80,160):
            z,w=np.polynomial.legendre.leggauss(order);x=(lo+hi)/2+(hi-lo)*z/2;weights=w*(hi-lo)/2
            f,tr=field(x),truth.values(x,0.,0.)
            h=(2/9)*(f[:,0]+f[:,1]);ht=(2/9)*(tr[:,0]+tr[:,1])
            orders[str(order)]={'forward_Hplus_first_moment':float(weights@(x*h)),
                                'truth_forward_Hplus_first_moment':float(weights@(x*ht)),
                                'integrated_squared_Hplus_error':float(weights@((h-ht)**2)),
                                'integrated_squared_truth_Hplus':float(weights@(ht**2))}
        regions[name]=orders
    return {'selection_sha256':sha(selection),'role':'Post-selection scoring only; no estimator input or checkpoint selection',
            'scale':{'xi':0.,'t_GeV2':0.,'Q2_GeV2':2.},'channels':['Sigma','T_rescaled','Hg=xg'],
            'observed_samples':{'x':info.x.numpy().reshape(-1).tolist(),'prediction':prediction.tolist(),
                'observed':observed.tolist(),'whitened_residual':whitened.tolist(),'chi2_PDF':float(whitened@whitened),
                'whitened_RMS':float(np.sqrt(np.mean(whitened**2)))},
            'supported_grid':{'points':201,'channel_relative_L2':relative.tolist(),
                              'x':grid.tolist(),'prediction':pred.tolist(),'truth':central.tolist()},
            'support_and_tails':regions,'seconds':time.monotonic()-start,
            'limitations':'Finite sampled boundary information; tails and same-forward D-term ambiguity remain. No continuum uncertainty band or extra residuals.'}
