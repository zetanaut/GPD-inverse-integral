"""Independent mpmath transcription of BFM v1 Eq.185 and auxiliaries 74-78,174.
Real continuations: Re log(z-i0), Re log(z-i0)^2, and real dilogarithm.
Provides point values, not a claim of full-support evolution validation.
"""
import mpmath as m

def al(z):return m.log(abs(z))
def l2(z):return al(z)**2-(m.pi**2 if z<0 else 0)
def lp(a,b):return al(a)*al(b)-(m.pi**2 if a<0 and b<0 else 0)
def li(z):return m.re(m.polylog(2,z))
def fc(x,y):return x*x/(y*y)*(2*(1-x)*y+y-x)
def fv(x,y):return x*x/(y*y)*(2+1/(y-x))+2*fc(x,y)
def h(x,y):return 2*fv(1-x,1-y)*lp(1-x,y)-2*fv(x,y)*(li(x)+li(1-y))
def hb(x,y):
 f=fv(x,y);b=fv(1-x,1-y)
 return (f-b)*(2*li(1-x/y)+l2(y))+2*f*(li(1-y)-lp(x,y))+2*b*li(1-x)

def half(x,y,nf=3):
 xb=1-x;yb=1-y;ca=m.mpf(3);cf=m.mpf(4)/3;tf=m.mpf('.5');beta=-11+2*m.mpf(nf)/3
 rho=m.sign(y) if 0<x/y<1 else m.mpf(0)
 erbl=1 if 0<x<1 else 0
 f=fv(x,y);b=fv(xb,yb);c=fc(x,y)
 regular_ca=((m.mpf(2)/3-2*m.zeta(2))*f-c/2+x*(4+5*x)/(4*y*y)-x*(10-19*x+16*x*x)/y+x*(6-13*x+8*x*x)/yb
  +(2*x*(1-2*xb*(x*y+xb*yb))/(y*yb)+x*(2-3*x-8*xb*xb)/(yb*yb))*al(x/y)
  +(f+2*x*x/(yb*yb))*l2(x/y)-(f-b)*lp(x/y,1-x/y)-h(x,y)/2-hb(xb,y)/2)
 erbl_ca=(-2*x*x/(yb*yb)*l2(x)+(xb*(x*x+xb*xb-2*xb*(1+2*x)*yb)+yb)/(yb*yb)*lp(x,xb)
  +((3*x*(x*x+xb*xb)*(yb-y)-4*x*x*y)/(y*yb)-x*(2-3*x-8*xb*xb)/(yb*yb))*al(x))
 regular_beta=(-m.mpf(5)/3*f-m.mpf(13)/3*c-m.mpf(11)/2*x*x/(y*y)+x*(x+yb)/(y*y*yb)+x*x/(yb*yb)*al(y)+xb*xb/(y*y)*al(xb))
 regular_nf=(-m.mpf(20)/3*c-12*x*x/(y*y)+2*x*(2+y+3*x*y-4*y*y)/(y*y*yb)-2*x*(x-y+2*x*y)/(y*yb*yb)*al(x/y)-x*x/(yb*yb)*l2(x/y))
 erbl_nf=-2*x*(1-3*x-2*x*yb)/(yb*yb)*al(x)+x*x/(yb*yb)*l2(x)
 return ca*ca*(rho*regular_ca+erbl*erbl_ca)+ca*beta/2*rho*regular_beta+cf*nf*tf*(rho*regular_nf+erbl*erbl_nf)

def kernel(x,y,xi,nf=3):
 x,y,xi=map(m.mpf,(x,y,xi));X=(1+x/xi)/2;Y=(1+y/xi)/2
 return (half(X,Y,nf)+half(1-X,1-Y,nf))/(2*xi)
