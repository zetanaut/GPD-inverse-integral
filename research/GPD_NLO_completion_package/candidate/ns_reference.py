"""Separate mpmath transcription of R2 Eq.177 and Eq.174 QQ auxiliaries.
Point values only; no evolution or finite-part certification is asserted.
"""
import mpmath as m
from gg_reference import al,l2,lp,li

def f(x,y):return x/y*(1+1/(y-x))
def h(x,y):return 2*f(1-x,1-y)*lp(1-x,y)-2*f(x,y)*(li(x)+li(1-y))
def hb(x,y):
 a=f(x,y);b=f(1-x,1-y)
 return (a-b)*(2*li(1-x/y)+l2(y))+2*a*(li(1-y)-lp(x,y))+2*b*li(1-x)

def half(x,y,sgn,nf):
 xb=1-x;yb=1-y;cf=m.mpf(4)/3;ca=m.mpf(3);beta=-11+2*m.mpf(nf)/3;z2=m.zeta(2)
 rho=m.sign(y) if 0<x/y<1 else 0;erbl=1 if 0<x<1 else 0
 if rho==0 and erbl==0:return m.mpf(0)
 v=f(x,y);b=f(xb,yb)
 p1=cf**2*((m.mpf(4)/3-2*z2)*v+3*x/y-(3*v/2-x/(2*yb))*al(x/y)-(v-b)*lp(x/y,1-x/y)+(v+x/(2*yb))*l2(x/y))
 p2=-cf*beta/2*(m.mpf(5)/3*v+x/y+v*al(x/y))
 p3=-cf*(cf-ca/2)*(m.mpf(4)/3*v+2*x/y+h(x,y)-sgn*hb(xb,y))
 p4=-x/(2*yb)*(al(x)+l2(x)-2*lp(x,xb))*cf**2
 return rho*(p1+p2+p3)+erbl*p4

def kernel(x,y,xi,sgn=1,nf=3):
 x,y,xi=map(m.mpf,(x,y,xi));X=(1+x/xi)/2;Y=(1+y/xi)/2
 return (half(X,Y,sgn,nf)+half(1-X,1-Y,sgn,nf))/(2*xi)
