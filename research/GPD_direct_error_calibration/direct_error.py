"""Joint target-error rank calibration, with strict binding and explicit failures.

This does not fit a GPD or supply a physical prior. A fixed point estimator is
calibrated on NEW (observations, truth) pairs and tested on exchangeable NEW pairs.
Common maximum of fixed-stratum cutoffs is conservative within those strata.
No claim is made for arbitrary truth functions or experimental domain shift.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
from math import ceil, inf, isfinite, sqrt
import json
from typing import Mapping, Sequence

@dataclass(frozen=True)
class Contract:
    estimator_hash: str
    design_hash: str
    targets: tuple[str, ...] = ('charge_bin_mean','charge_x_moment')
    scales: tuple[float, ...] = (.05,.05)
    alpha: str = '0.10'
    strata: tuple[str, ...] = ('A','B')
    calibration_count: int = 80
    failure_override: str = 'any_failure_infinite'

    def __post_init__(self):
        if not self.estimator_hash or not self.design_hash: raise ValueError('Missing estimator/design binding')
        if len(self.targets)!=len(self.scales) or not self.targets or len(set(self.targets))!=len(self.targets): raise ValueError('Invalid target specification')
        if any(not isfinite(s) or s<=0 for s in self.scales): raise ValueError('Scales must be fixed, finite and positive')
        if not 0<Fraction(self.alpha)<1: raise ValueError('Invalid alpha')
        if not isinstance(self.calibration_count,int) or isinstance(self.calibration_count,bool) or self.calibration_count<1:raise ValueError('Invalid count')
        if not self.strata or len(set(self.strata))!=len(self.strata):raise ValueError('Invalid strata')
        if self.failure_override!='any_failure_infinite':raise ValueError('This version retains the conservative failure override')

    @property
    def hash(self) -> str:
        return sha256(json.dumps(asdict(self),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

@dataclass(frozen=True)
class Fit:
    dataset_id: str
    input_hash: str
    contract_hash: str
    values: tuple[float,...] | None
    failure: str | None = None

    def checked(self,c:Contract) -> None:
        if not self.dataset_id or not self.input_hash:raise ValueError('Missing dataset identity')
        if self.contract_hash!=c.hash:raise ValueError('Different estimator, geometry, covariance, targets or prescription')
        if self.failure:
            if self.values is not None:raise ValueError('Failure must not masquerade as a finite fit')
        elif self.values is None or len(self.values)!=len(c.targets) or any(not isfinite(x) for x in self.values):
            raise ValueError('Finite fit required unless explicitly failed')

@dataclass(frozen=True)
class Pair:
    fit: Fit
    stratum: str
    truth: tuple[float,...]
    phase: str

@dataclass(frozen=True)
class Cutoff:
    contract_hash: str
    q: float | None # None means +infinity, never a silently removed result.
    per_stratum: dict
    calibration_ids: tuple[str,...]
    calibration_inputs: tuple[str,...]
    scores_hash: str

    def __post_init__(self):
        if self.q is not None and (not isfinite(self.q) or self.q<0):raise ValueError('Invalid cutoff')
        if len(set(self.calibration_ids))!=len(self.calibration_ids):raise ValueError('Duplicate calibration IDs')


def rank(m:int,alpha:str) -> int:
    if not isinstance(m,int) or isinstance(m,bool) or m<1:raise ValueError('No calibration data')
    if not 0<Fraction(str(alpha))<1:raise ValueError('Invalid alpha')
    return ceil((m+1)*(1-Fraction(str(alpha))))


def joint_score(c:Contract,p:Pair) -> float:
    p.fit.checked(c)
    if p.stratum not in c.strata:raise ValueError('Unregistered stratum')
    if len(p.truth)!=len(c.targets) or any(not isfinite(x) for x in p.truth):raise ValueError('Invalid true targets')
    if p.fit.failure:return inf
    value=max(abs(a-b)/s for a,b,s in zip(p.fit.values,p.truth,c.scales))
    if not isfinite(value):raise OverflowError('Finite target inputs produced a nonfinite score')
    return value


def freeze_cutoff(c:Contract,pairs:Sequence[Pair]) -> Cutoff:
    if any(p.phase!='calibration' for p in pairs):raise ValueError('Development/test data cannot calibrate')
    ids=[p.fit.dataset_id for p in pairs];inputs=[p.fit.input_hash for p in pairs]
    if len(set(ids))!=len(ids) or len(set(inputs))!=len(inputs):raise ValueError('Duplicated calibration dataset/input')
    groups={s:[] for s in c.strata}
    for p in pairs:
        if p.stratum not in groups:raise ValueError('Unregistered stratum: '+str(p.stratum))
        groups[p.stratum].append((p.fit.dataset_id,joint_score(c,p)))
    per={};all_scores=[]
    for s in c.strata:
        g=sorted(groups[s]);m=len(g)
        if m!=c.calibration_count:raise ValueError(f'Expected {c.calibration_count} pairs in {s}, got {m}')
        k=rank(m,c.alpha);scores=sorted(x[1] for x in g);failures=sum(x==inf for x in scores)
        q=inf if failures or k>m else scores[k-1]
        per[s]={'n':m,'rank':k,'failures':failures,'q':None if q==inf else q,'infinite':q==inf}
        all_scores.extend([{'id':i,'stratum':s,'score':None if x==inf else x,'failed':x==inf} for i,x in g])
    q=inf if any(d['infinite'] for d in per.values()) else max(d['q'] for d in per.values())
    h=sha256(json.dumps(all_scores,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
    return Cutoff(c.hash,None if q==inf else q,per,tuple(sorted(ids)),tuple(sorted(inputs)),h)


def _unknown(s):raise ValueError('Unregistered stratum: '+str(s))


def intervals(c:Contract,q:Cutoff,fit:Fit) -> dict:
    """Truth-blind deployment. This function cannot see a truth or its stratum."""
    fit.checked(c)
    if q.contract_hash!=c.hash:raise ValueError('Cutoff binding mismatch')
    if fit.dataset_id in q.calibration_ids or fit.input_hash in q.calibration_inputs:raise ValueError('Calibration/test reuse')
    infinite=fit.failure is not None or q.q is None
    reason='fit_failure' if fit.failure else 'calibration_infinite' if q.q is None else 'finite'
    if infinite:
        bounds={n:None for n in c.targets} # JSON-safe full real line.
    else:
        bounds={n:[x-q.q*s,x+q.q*s] for n,x,s in zip(c.targets,fit.values,c.scales)}
    return {'status':reason,'bounds':bounds,'null_bound_means':'full_real_line','contract_hash':c.hash,'dataset_id':fit.dataset_id}


def wilson(k:int,n:int,z:float=1.959963984540054) -> tuple[float,float]:
    if n<=0 or k<0 or k>n:raise ValueError('Invalid binomial counts')
    p=k/n;d=1+z*z/n;mid=(p+z*z/(2*n))/d;half=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0.,mid-half),min(1.,mid+half)


def evaluate(c:Contract,q:Cutoff,pairs:Sequence[Pair],expected_per_stratum:int=240) -> dict:
    """Audit known test truth only AFTER producing each truth-blind interval."""
    ids=[p.fit.dataset_id for p in pairs];inputs=[p.fit.input_hash for p in pairs]
    if len(set(ids))!=len(ids) or len(set(inputs))!=len(inputs):raise ValueError('Duplicated test data')
    groups={s:[] for s in c.strata}
    for p in pairs:
        if p.phase!='test':raise ValueError('Not untouched test data')
        if p.stratum not in groups:raise ValueError('Unregistered test stratum')
        out=intervals(c,q,p.fit) # No truth or stratum passed to inference.
        joint_score(c,p) # validate truth shape; score not used to select interval.
        covered=[out['bounds'][n] is None or out['bounds'][n][0]<=v<=out['bounds'][n][1] for n,v in zip(c.targets,p.truth)]
        groups[p.stratum].append((p,out,covered))
    summary={}
    for s,rows in groups.items():
        n=len(rows)
        if n!=expected_per_stratum:raise ValueError(f'Expected {expected_per_stratum} tests in {s}, got {n}')
        k=sum(all(r[2]) for r in rows);fail=sum(r[0].fit.failure is not None for r in rows)
        finite=sum(r[1]['status']=='finite' for r in rows)
        # Conditional finite-output coverage is diagnostic, not rank-guaranteed.
        kfinite=sum(all(r[2]) for r in rows if r[1]['status']=='finite')
        lower=wilson(k,n,z=1.6448536269514722)[0]
        summary[s]={'n':n,'joint_covered':k,'joint_rate':k/n,'wilson95':wilson(k,n),
                    'one_sided95_lower':lower,'failed_fits':fail,'finite_outputs':finite,
                    'finite_only_covered':kfinite,'finite_only_rate':None if not finite else kfinite/finite,
                    'marginal_covered':{t:sum(r[2][j] for r in rows) for j,t in enumerate(c.targets)},
                    'finite_widths':None if q.q is None else {t:2*q.q*s0 for t,s0 in zip(c.targets,c.scales)},
                    'coverage_gate':lower>=.85,'failure_gate':fail/n<=.05,
                    'finite_reporting_gate':finite/n>=.95}
    return {'strata':summary,'pass':all(x['coverage_gate'] and x['failure_gate'] and x['finite_reporting_gate'] for x in summary.values()),
            'scope':'two-target simultaneous test under registered simulation strata, not experimental/global-QCD validity'}
