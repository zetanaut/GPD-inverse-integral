"""Independent analytic K0 action used ONLY for the NLO factorization-log control.
R2 Eqs. (73)-(79), (15): polynomial input is integrated as rational primitives.
No source bridge, matrix, interpolated field, evolution or LO campaign is used.
QG crossover limit includes the finite endpoint-layer contribution nf*g(xi)/(2xi).
"""
from pathlib import Path
import sys,json
import mpmath as mp
import numpy as np
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candidate'));sys.path.insert(0,str(ROOT/'frozen/python'))
from nlo_forward import alpha_two_loop
from forward import hard_row,initial
from spectral import Elements
mp.mp.dps=40

def mul(a,b):
 c=[mp.mpf(0)]*(len(a)+len(b)-1)
 for i,v in enumerate(a):
  for j,w in enumerate(b):c[i+j]+=v*w
 return c

def ratint(poly,c,power,l,r):
 shifted=[sum(poly[j]*mp.binomial(j,k)*c**(j-k) for j in range(k,len(poly))) for k in range(len(poly))]
 total=mp.mpf(0)
 for k,co in enumerate(shifted):
  n=k-power+1
  v=mp.log(abs((r-c)/(l-c))) if n==0 else ((r-c)**n-(l-c)**n)/n
  total+=co*v
 return total

def action(x,xi):
 x=mp.mpf(x);xi=mp.mpf(xi);one=mp.mpf(1);cf=mp.mpf(4)/3;nf=3
 aq=2*(1+mp.mpf('.2')*xi**2)+mp.mpf('.5')*(1-mp.mpf('.1')*xi**2)
 ag=mp.mpf('.8')*(1+mp.mpf('.1')*xi**2)
 G=[1-2*x*x+x**4,-2*x+x**3,-2+x*x,x,one]
 first=(x+xi)*ratint(mul([2*xi-x,one],G),-xi,1,x,one)
 if x==xi:
  second=0;local=cf*(mp.mpf('1.5')+mp.log((1-xi)/(2*xi)))
 elif x<xi:
  second=-(xi-x)*ratint(mul([-x-2*xi,one],G),xi,1,-one,x)
  local=cf*(mp.mpf('1.5')+mp.log((1-x*x)/(1+xi))+((x-xi)*mp.log(xi-x)-(x+xi)*mp.log(x+xi))/(2*xi))
 else:
  second=(xi-x)*ratint(mul([-x-2*xi,one],G),xi,1,x,one)
  local=cf*(mp.mpf('1.5')+2*mp.log(1-x)+((x-xi)*mp.log((x-xi)*(1+xi))-(x+xi)*mp.log((x+xi)*(1-xi)))/(2*xi))
 qq=aq*(cf/(2*xi)*(first+second)+local*x*(1-x*x)**2)
 gp=[one,0,-2,0,one]
 gfirst=-nf*(x+xi)/(2*xi**3)*ratint(mul([xi*(xi-2*x),xi-x],gp),-xi,2,x,one)
 if x==xi:gsecond=nf/(2*xi)*(1-xi*xi)**2
 elif x<xi:gsecond=nf*(xi-x)/(2*xi**3)*ratint(mul([xi*(2*x+xi),-(xi+x)],gp),xi,2,-one,x)
 else:gsecond=-nf*(xi-x)/(2*xi**3)*ratint(mul([xi*(2*x+xi),-(xi+x)],gp),xi,2,x,one)
 return qq+ag*(gfirst+gsecond)

def main():
 xi=.135;q2=2.3;cross=float(action(xi,xi));alpha=alpha_two_loop(q2);rows=[]
 for eps in [2e-8,2e-10]:
  f=lambda x:float(action(x,xi));fn=lambda x:(f(x)-cross)/(x-xi)+f(x)/(x+xi)
  left=quad(fn,0,xi,epsabs=eps,epsrel=eps,limit=200);right=quad(fn,xi,1,epsabs=eps,epsrel=eps,limit=200)
  c=alpha/(2*np.pi)*2/9*(left[0]+right[0]+cross*np.log((1-xi)/xi)-1j*np.pi*cross)
  rows.append({'eps':eps,'CFF':[c.real,c.imag],'quadrature_error_estimate':alpha/(2*np.pi)*2/9*(left[1]+right[1])})
 b=Elements(xi,6,'graded');v=initial(b);h=(hard_row(b,q2,96,1.)[0]-hard_row(b,q2,96,0.)[0])@v
 samples=[{'x':x,'K0_Sigma_plus_T':float(action(x,xi))} for x in [.03,.09,.134,xi-1e-8,xi,xi+1e-8,.136,.19,.67]]
 result={'scope':'independent analytic full-support K0 quark action for NLO scale-log consistency only','crossover_action':cross,'samples':samples,'direct_integrals':rows,'accepted_hard_log':[h.real,h.imag],'residual':[c.real-h.real,c.imag-h.imag],'mp_dps':mp.mp.dps}
 (ROOT/'work/action/ANALYTIC_SCALE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
