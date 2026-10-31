"""Fixed, already generated PDF draw; no RNG or truth generator in the fitter."""
from p2b_policy import BUNDLE,read,sha

def validate_inputs():
    import numpy as np
    meta=read(BUNDLE/'inputs/NOISE_RECEIPT.json')
    for name,key in [('CENTRAL_FORWARD.npz','central_sha256'),('PDF_NOISE_Z.npy','noise_z_sha256'),('FORWARD_NOISY.npz','input_sha256')]:
        if sha(BUNDLE/'inputs'/name)!=meta[key]:raise RuntimeError('Frozen PDF input changed: '+name)
    if (meta['seed'],meta['bit_generator'],meta['standard_normal_variates'],meta['ordering'])!=(2026102601,'PCG64',36,'x-major/channel-minor'):
        raise RuntimeError('Noise law/seed/order drift')
    with np.load(BUNDLE/'inputs/CENTRAL_FORWARD.npz',allow_pickle=False) as f:
        x,central,cov=[f[k].copy() for k in ['x','observed','covariance']]
    with np.load(BUNDLE/'inputs/FORWARD_NOISY.npz',allow_pickle=False) as f:
        nx,y,nc=[f[k].copy() for k in ['x','observed','covariance']]
    z=np.load(BUNDLE/'inputs/PDF_NOISE_Z.npy',allow_pickle=False)
    if (not np.array_equal(nx,x) or not np.array_equal(cov,nc) or z.shape!=(36,)
            or not np.isfinite(y).all() or not np.isfinite(z).all()
            or not np.allclose(y.reshape(-1)-central.reshape(-1),np.linalg.cholesky(cov)@z,rtol=1e-12,atol=2e-15)):
        raise RuntimeError('Saved PDF draw/covariance identity failed')
    return x,y,cov

def information(base):
    from pdf_information import ForwardInformation
    return ForwardInformation(base,*validate_inputs())
