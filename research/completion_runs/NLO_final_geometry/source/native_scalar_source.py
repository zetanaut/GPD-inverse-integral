"""Operational scalar calls to the same captured native source ABI.

The installed macOS process reproducibly crashes in vector NS-minus calls.
Every quadrature coordinate is retained. No mathematical expression, precision,
quadrature, guard threshold or local callback is replaced.
"""
from pathlib import Path
import sys,os,numpy as np
ROOT=Path(os.environ.get('GPD_REVIEW_ROOT',Path(__file__).resolve().parents[3]))/'GPD_NLO_completion_package'
sys.path.insert(0,str(ROOT))
from src.source_parts import SourceParts as Native

class SourceParts(Native):
    def evaluate(self,order,sector,x,xi,y,include_local=False):
        y=np.ascontiguousarray(np.atleast_1d(y),dtype=np.float64)
        if len(y)==1:return super().evaluate(order,sector,x,xi,y,include_local)
        reg=np.empty_like(y);pls=np.empty_like(y);local=None
        for i,value in enumerate(y):
            r,p,l=super().evaluate(order,sector,x,xi,[float(value)],include_local and i==0)
            reg[i]=r[0];pls[i]=p[0]
            if i==0:local=l
        return reg,pls,local
