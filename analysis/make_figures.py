"""Regenerate publication figures from bundled, immutable saved-result data."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',type=Path,default=ROOT/'data')
p.add_argument('--out',type=Path,default=ROOT.parent/'outputs/analysis/figures')
a=p.parse_args()
D=a.data; O=a.out; O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
 'mathtext.fontset':'stix','font.size':9,'axes.labelsize':10,'axes.titlesize':9,
 'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
 'lines.linewidth':1.4,'pdf.fonttype':42,'savefig.dpi':300})
BLUE='#20639b'; ORANGE='#d05a20'; PURPLE='#8a4f9d'; GREEN='#278265'
def finish(fig,name):
 for ax in fig.axes:
  if hasattr(ax,'_colorbar') and ax._colorbar is not None:
   ax._colorbar.locator=MaxNLocator(nbins=4)
   ax._colorbar.formatter=FormatStrFormatter('%.3g')
   ax._colorbar.update_ticks()
 fig.savefig(O/(name+'.pdf'),bbox_inches='tight',pad_inches=.035)
 fig.savefig(O/(name+'.png'),bbox_inches='tight',pad_inches=.035)
 plt.close(fig)
def panel(ax,label): ax.text(.025,.96,label,transform=ax.transAxes,va='top',fontsize=10)
def base(ax,xlabel,ylabel):
 ax.set_xlabel(xlabel); ax.set_ylabel(ylabel);ax.grid(alpha=.16,lw=.5)

z=np.load(D/'ideal_L0_Z1_s31.npz'); s=np.load(D/'sampled_surfaces.npz')
fig,axs=plt.subplots(2,2,figsize=(6.95,5.35),layout='constrained')
ax=axs[0,0];mask=z['x']>=0
for j,(t,c) in enumerate(zip([-.38,-.8,-1.2],[BLUE,ORANGE,GREEN])):
 ax.plot(z['x'][mask],z['x_truth'][j,mask],color=c,label=rf'$t={t:g}$')
 ax.plot(z['x'][mask],z['x_fit'][j,mask],color=c,ls='--')
ax.legend(loc='upper right',title=r'$t$ in GeV$^2$',title_fontsize=8)
ax.set_title(r'$\xi=0.275$: solid truth, dashed reconstruction')
base(ax,r'$x$',r'$H^{(+)}$');panel(ax,'(a)')
ax=axs[0,1];m=z['xi_slice']<=.42
ax.plot(z['xi_slice'][m],z['xi_truth'][m],color='black',label='Truth')
ax.plot(z['xi_slice'][m],z['xi_fit'][m],color=BLUE,ls='--',label='Reconstruction')
ax.set_title(r'$x=0.5,\ t=-0.8\ {\rm GeV}^2$');ax.legend(loc='lower left')
base(ax,r'$\xi$',r'$H^{(+)}$');panel(ax,'(b)')
ax=axs[1,0];ax.plot(z['t_slice'],z['t_truth'],color='black');ax.plot(z['t_slice'],z['t_fit'],color=BLUE,ls='--')
ax.set_title(r'$x=0.5,\ \xi=0.275$');base(ax,r'$t\ [{\rm GeV}^2]$',r'$H^{(+)}$');panel(ax,'(c)')
ax=axs[1,1]
for j,c,lab in [(0,BLUE,r'$\mathrm{Re}\,\mathcal{H}$'),(1,ORANGE,r'$\mathrm{Im}\,\mathcal{H}$')]:
 ax.scatter(z['cff_holdout_truth'][:,j],z['cff_holdout_pred'][:,j],s=6,alpha=.5,color=c,label=lab,rasterized=True)
lo=min(z['cff_holdout_truth'].min(),z['cff_holdout_pred'].min());hi=max(z['cff_holdout_truth'].max(),z['cff_holdout_pred'].max())
ax.plot([lo,hi],[lo,hi],color='.35',lw=.8,ls=':');ax.legend(loc='lower right')
ax.set_title('512 disjoint kinematic positions');base(ax,'True CFF','Reconstructed CFF');panel(ax,'(d)')
finish(fig,'fig01_exact_closure')

fig,axs=plt.subplots(1,3,figsize=(6.95,2.65),layout='constrained')
vmax=max(s['truth'].max(),s['ideal'].max());lev=np.linspace(0,vmax,25)
for ax,key,title in zip(axs[:2],['truth','ideal'],['(a) Truth','(b) Reconstruction']):
 im=ax.contourf(s['x'],s['t'],s[key],levels=lev,cmap='viridis');ax.set_title(title)
 ax.axvline(.275,color='white',ls=':',lw=.8);ax.set_xlabel(r'$x$')
axs[0].set_ylabel(r'$t\ [{\rm GeV}^2]$');fig.colorbar(im,ax=list(axs[:2]),location='bottom',shrink=.75,pad=.05,label=r'$H^{(+)}$')
delta=s['ideal']-s['truth'];lim=np.max(abs(delta));ax=axs[2]
im=ax.contourf(s['x'],s['t'],delta,levels=np.linspace(-lim,lim,25),cmap='RdBu_r');ax.set_title('(c) Reconstruction − truth');ax.set_xlabel(r'$x$');ax.axvline(.275,color='.35',ls=':',lw=.8)
fig.colorbar(im,ax=ax,location='bottom',shrink=.85,pad=.05,label=r'$\Delta H^{(+)}$')
finish(fig,'fig02_exact_surface')

fig,axs=plt.subplots(2,3,figsize=(6.95,4.65),layout='constrained')
for row,seed in enumerate([31,43]):
 e=np.load(D/f'exp_Re_exact_Z1_s{seed}.npz');n=np.load(D/f'exp_Re_noisy_Z1_s{seed}.npz');m=e['x']>=0
 for j,t in enumerate([-.38,-.8,-1.2]):
  ax=axs[row,j];ax.plot(e['x'][m],e['x_truth'][j,m],color='black',label='Truth')
  ax.plot(e['x'][m],e['x_fit'][j,m],color=BLUE,ls='--',label='Exact input')
  ax.plot(n['x'][m],n['x_fit'][j,m],color=ORANGE,ls='-.',label='Noisy input')
  ax.axvline(.275,color='.65',lw=.6,ls=':');ax.set_title(rf'$t={t:g}\ {{\rm GeV}}^2$, seed {seed}')
  base(ax,r'$x$',r'$H^{(+)}$' if j==0 else '');panel(ax,f'({chr(97+row*3+j)})')
axs[0,0].legend(loc='upper right',fontsize=7)
finish(fig,'fig03_exact_noisy')

f=pd.read_csv(D/'experimental_family.csv');nom=pd.read_csv(D/'experimental_nominal.csv')
anchors=list(f.anchor_row_id.unique())
fig,axs=plt.subplots(2,3,figsize=(6.95,4.8),layout='constrained')
for j,(ax,anchor) in enumerate(zip(axs.flat,anchors)):
 q=f[(f.anchor_row_id==anchor)&(f.budget==20)&(f.x>=0)]
 for case,color,ls,label in [('ensemble_mean',BLUE,'-','Fit to replica-mean CFF'),('replica_000',ORANGE,'--','Fit to CFF replica 000'),('replica_001',PURPLE,'-.','Fit to CFF replica 001')]:
  v=q[q.case==case].sort_values('x');ax.plot(v.x,v.Hcharge_plus,color=color,ls=ls,label=label)
 v=nom[(nom.anchor_row_id==anchor)&(nom.state=='central13')&(nom.x>=0)].sort_values('x')
 ax.plot(v.x,v.Hcharge_plus,color='.5',ls=':',label='Fit to local CFF values')
 xi=q.iloc[0].xi;t=q.iloc[0].t_GeV2
 ax.axvline(xi,color='.65',lw=.6,ls=':');ax.set_title(rf'$\xi={xi:.3f},\ t={t:g}\ {{\rm GeV}}^2$')
 base(ax,r'$x$',r'$H^{(+)}$' if j%3==0 else '');panel(ax,f'({chr(97+j)})')
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=8)
finish(fig,'fig04_experimental_family')

fig,axs=plt.subplots(1,2,figsize=(6.95,3.1),layout='constrained')
im=axs[0].contourf(s['x'],s['t'],s['mean'],levels=24,cmap='viridis');axs[0].set_title('(a) GPD fitted to replica-mean CFF')
fig.colorbar(im,ax=axs[0],label=r'$H^{(+)}$')
diff=s['r001']-s['mean'];lim=np.max(abs(diff))
im=axs[1].contourf(s['x'],s['t'],diff,levels=np.linspace(-lim,lim,25),cmap='RdBu_r');axs[1].set_title('(b) Replica 001 GPD minus panel (a)')
fig.colorbar(im,ax=axs[1],label=r'$\Delta H^{(+)}$')
for ax in axs:ax.set_xlabel(r'$x$');ax.set_ylabel(r'$t\ [{\rm GeV}^2]$');ax.axvline(.275,color='white',ls=':',lw=.8)
finish(fig,'fig05_experimental_surface')

b=pd.read_csv(D/'budget_stages.csv');h=pd.read_csv(D/'budget_response.csv')
fig,axs=plt.subplots(1,2,figsize=(6.95,2.8),layout='constrained')
v=b[b.case=='ensemble_mean'].sort_values('budget');axs[0].plot(v.budget,v.mean_RMS,'o-',color=BLUE,ms=3)
base(axs[0],'Additional outer iterations',r'RMS $[(\mathcal{H}_{\rm GPD}-\overline{\mathcal{H}})/\sigma]$');panel(axs[0],'(a)')
for case,c,label in [('replica_000',ORANGE,'CFF replica 000'),('replica_001',PURPLE,'CFF replica 001')]:
 v=h[h.case==case].sort_values('budget');axs[1].plot(v.budget,v.late_rms,'o-',color=c,label=label,ms=4)
base(axs[1],'Additional outer iterations',r'RMS $(H^{(+)}_r-H^{(+)}_{\rm mean})$');panel(axs[1],'(b)');axs[1].legend(loc='upper left',bbox_to_anchor=(.03,.87))
for ax in axs:ax.set_xticks([5,10,15,20]);ax.set_xlim(4,21)
finish(fig,'fig06_budget_response')
print('Wrote six PDF figures and six PNG previews.')
