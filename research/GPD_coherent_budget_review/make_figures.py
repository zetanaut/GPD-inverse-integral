"""Figures from saved, independently replayed scalar summaries. No training."""
from pathlib import Path
import csv,os
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(name):
    with (ROOT/'results'/name).open(newline='') as f:return list(csv.DictReader(f))

def main():
    out=ROOT/'figures';out.mkdir(parents=True,exist_ok=True)
    rows=read('HPLUS_RESPONSE.csv')
    fig,ax=plt.subplots(figsize=(7.4,4.7))
    for case,label in [('replica_000','r000 minus matched mean'),('replica_001','r001 minus matched mean')]:
        r=[r for r in rows if r['case']==case]
        ax.plot([int(a['budget']) for a in r],[float(a['late_rms']) for a in r],marker='o',label=label)
    ax.set(xlabel='Cumulative continuation steps',ylabel=r'Sampled RMS of $H_r^+-H_{\mathrm{mean}}^+$',
           title='The same coherent inputs give budget-dependent GPD responses',xticks=[5,10,20])
    ax.legend();fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(out/f'Hplus_response_budget.{ext}',dpi=160)
    plt.close(fig)
    rows=read('STAGES.csv');fig,ax=plt.subplots(figsize=(7.4,4.7))
    for case,label in [('ensemble_mean','Mean input'),('replica_000','r000 input'),('replica_001','r001 input')]:
        r=[r for r in rows if r['case']==case]
        ax.plot([int(a['budget']) for a in r],[float(a['own_RMS']) for a in r],label=label)
    ax.set(xlabel='Cumulative continuation steps',ylabel='RMS residual to own input / original quoted error',
           title='Input reproduction improves, without demonstrated convergence',xticks=[5,10,20])
    ax.legend();fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(out/f'ReH_residual_budget.{ext}',dpi=160)
    plt.close(fig)
    print('Created two saved-result diagnostic figures (PNG/SVG). No training or error band.')
if __name__=='__main__':main()
