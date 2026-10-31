"""Prospective extended-real rank calibration. Not fitted on development data.

An abstention contributes +infinity at that row, not an automatic infinite
cutoff for the whole stratum. Missing rows are errors, not abstentions.
Requires a frozen estimator and exchangeable calibration/test scores.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Sequence

@dataclass(frozen=True)
class Score:
    dataset_id: str
    value: float
    status: str = 'finite'
    phase: str = 'calibration'
    def __post_init__(self):
        if not self.dataset_id or self.phase != 'calibration':
            raise ValueError('Named, fresh calibration rows only; development is not calibration')
        if self.status not in ('finite','abstain'):
            raise ValueError('Unknown/missing status is not a valid score')
        if self.status == 'finite' and (not math.isfinite(self.value) or self.value < 0):
            raise ValueError('A finite error score must be nonnegative and finite')
        if self.status == 'abstain' and self.value != math.inf:
            raise ValueError('An explicit abstention must contribute +infinity')

def rank_index(n: int, alpha: float = .1) -> int:
    if isinstance(n,bool) or not isinstance(n,int) or n < 1:
        raise ValueError('Positive integer sample size required')
    a=Fraction(str(alpha))
    if not 0 < a < 1: raise ValueError('0 < alpha < 1 required')
    f=(n+1)*(1-a)
    return (f.numerator+f.denominator-1)//f.denominator

def cutoff(scores: Sequence[Score], expected_n: int, alpha: float = .1) -> dict:
    if len(scores) != expected_n: raise ValueError('Incomplete/extra calibration rows')
    if len({s.dataset_id for s in scores}) != len(scores): raise ValueError('Duplicate row')
    k=rank_index(expected_n,alpha)
    q=math.inf if k>expected_n else sorted(s.value for s in scores)[k-1]
    return {'n':expected_n,'rank':k,'alpha':alpha,'infinite':math.isinf(q),
            'q':None if math.isinf(q) else q,
            'abstentions':sum(s.status=='abstain' for s in scores),
            'scope':'marginal rank statement under frozen-rule exchangeability; no finite-output-conditional claim'}

def common_cutoff(strata: dict[str,Sequence[Score]], expected_n: int, alpha: float=.1) -> dict:
    if not strata: raise ValueError('No strata')
    all_ids=[s.dataset_id for v in strata.values() for s in v]
    if len(set(all_ids))!=len(all_ids):raise ValueError('Duplicate ID across strata')
    details={k:cutoff(v,expected_n,alpha) for k,v in strata.items()}
    infinite=any(v['infinite'] for v in details.values())
    return {'strata':details,'infinite':infinite,'q':None if infinite else max(v['q'] for v in details.values())}

def error_score(estimate, truth, scales=(.05,.05)) -> float:
    if len(estimate)!=len(truth) or len(estimate)!=len(scales) or not scales:raise ValueError('Shape mismatch')
    values=tuple(estimate)+tuple(truth)+tuple(scales)
    if not all(math.isfinite(float(v)) for v in values) or any(s<=0 for s in scales):raise ValueError('Invalid values/scales')
    return max(abs(e-t)/s for e,t,s in zip(estimate,truth,scales))

def rectangle(estimate, result:dict, scales=(.05,.05), *, abstain=False):
    estimate=tuple(estimate);scales=tuple(scales)
    if len(estimate)!=len(scales) or not all(math.isfinite(float(v)) for v in estimate+scales) or any(s<=0 for s in scales):
        raise ValueError('Invalid target values/scales')
    if abstain or result['infinite']:
        return {'unbounded':True,'intervals':None,'abstention':abstain}
    q=float(result['q'])
    if not math.isfinite(q) or q<0:raise ValueError('Invalid cutoff')
    return {'unbounded':False,'intervals':[[e-q*s,e+q*s] for e,s in zip(estimate,scales)],'abstention':False}
