"""Plots available before fitting; never claims reconstruction."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from closure.truths import Truth
out=ROOT/'handoff_plots';out.mkdir(exist_ok=True)
x=np.linspace(-.999,.999,501)
fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
for i,name in enumerate(('Z1','Z2')):
 tr=Truth(name)
 for j,channel in enumerate(('Sigma','T rescaled','xg')):
  ax=axes[i,j]
  for t in (-.21,-.363,-.8):ax.plot(x,tr.values(x,.275,t)[:,j],label=f't={t}')
  ax.set_title(f'{name}: {channel}, xi=0.275');ax.grid(alpha=.2)
  if i==1:ax.set_xlabel('x')
  if j==0:ax.set_ylabel('Reference-scale H+')
axes[0,2].legend(fontsize=8)
fig.suptitle('Analytic truths only — no fitted reconstruction exists')
fig.tight_layout();fig.savefig(out/'truth_slices.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(10,6))
for i,name in enumerate(('Z1','Z2')):
 with np.load(ROOT/f'prepared/evaluation_only/{name}_experimental.npz') as z:
  ref=z['mean_reference'];nom=z['mean_nominal']
 with np.load(ROOT/f'prepared/evaluation_only/{name}_noise.npz') as z:noise=z['noise'];sig=z['sigma']
 for j,comp in enumerate(('Re','Im')):
  ax=axes[i,j];ax.plot(nom[:,j]-ref[:,j],label='nominal - reference',lw=1)
  if j==0:ax.plot(noise,label='one fixed noise draw',lw=1,alpha=.8)
  ax.set_title(f'{name}: {comp} at 160 experimental coordinates')
  ax.grid(alpha=.2);ax.set_xlabel('Input row index');ax.set_ylabel('CFF units')
  ax.legend(fontsize=8)
fig.suptitle('Input numerical and noise differences — no fit')
fig.tight_layout();fig.savefig(out/'numerics_and_noise.png',dpi=150);plt.close(fig)
print(json.dumps({'plots':[str(p) for p in sorted(out.glob('*.png'))]}))
