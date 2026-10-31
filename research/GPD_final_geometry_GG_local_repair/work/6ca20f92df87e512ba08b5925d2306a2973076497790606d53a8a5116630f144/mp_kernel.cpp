// Guarded arithmetic for BFM Eqs.177,185. Uses installed MPFR, no formula tuning.
#include <mpfr.h>
#include <cmath>
#include <stdexcept>
#ifndef GPD_MP_BITS
#define GPD_MP_BITS 192
#endif
struct R {
 mpfr_t v;
 R(double d=0) {mpfr_init2(v,GPD_MP_BITS);mpfr_set_d(v,d,MPFR_RNDN);}
 R(const R& a) {mpfr_init2(v,GPD_MP_BITS);mpfr_set(v,a.v,MPFR_RNDN);}
 R& operator=(const R& a){mpfr_set(v,a.v,MPFR_RNDN);return *this;}
 ~R(){mpfr_clear(v);}
 double as_double() const{return mpfr_get_d(v,MPFR_RNDN);}
};
#define BIN(OP,F) R operator OP(const R&a,const R&b){R c;F(c.v,a.v,b.v,MPFR_RNDN);return c;}
BIN(+,mpfr_add) BIN(-,mpfr_sub) BIN(*,mpfr_mul) BIN(/,mpfr_div)
#undef BIN
R operator-(const R&a){R c;mpfr_neg(c.v,a.v,MPFR_RNDN);return c;}
bool operator<(const R&a,const R&b){return mpfr_cmp(a.v,b.v)<0;}
bool operator>(const R&a,const R&b){return b<a;}
bool operator==(const R&a,const R&b){return mpfr_equal_p(a.v,b.v);}
R log(const R&a){R c;mpfr_log(c.v,a.v,MPFR_RNDN);return c;}
R abs(const R&a){R c;mpfr_abs(c.v,a.v,MPFR_RNDN);return c;}
R pi(){R c;mpfr_const_pi(c.v,MPFR_RNDN);return c;}
R sq(const R&a){return a*a;}
R al(const R&a){return log(abs(a));}
R l2(const R&a){return sq(al(a))-(a<R(0)?sq(pi()):R(0));}
R lp(const R&a,const R&b){return al(a)*al(b)-((a<R(0)&&b<R(0))?sq(pi()):R(0));}
R li(const R&z){
 // MPFR implements the real part of the principal dilogarithm on the real axis.
 R c;mpfr_li2(c.v,z.v,MPFR_RNDN);return c;
}
R rho(const R&x,const R&y){R q=x/y;return (R(0)<q&&q<R(1))?(y<R(0)?R(-1):R(1)):R(0);}
R erbl(const R&x){return R(0)<x&&x<R(1)?R(1):R(0);}
R gf(const R&x,const R&y){return sq(x)/sq(y)*(2+R(1)/(y-x))+2*sq(x)/sq(y)*(2*(1-x)*y+y-x);}
R gc(const R&x,const R&y){return sq(x)/sq(y)*(2*(1-x)*y+y-x);}
R gh(const R&x,const R&y){return 2*gf(1-x,1-y)*lp(1-x,y)-2*gf(x,y)*(li(x)+li(1-y));}
R ghb(const R&x,const R&y){R f=gf(x,y),b=gf(1-x,1-y);return (f-b)*(2*li(1-x/y)+l2(y))+2*f*(li(1-y)-lp(x,y))+2*b*li(1-x);}
R gg_half(const R&x,const R&y,int part){
 R xb=1-x,yb=1-y,ca=3,cf=R(4)/3,tf=R(1)/2,r=rho(x,y),e=erbl(x);
 if(r==R(0)&&e==R(0))return 0;
 R f=gf(x,y),b=gf(xb,yb),c=gc(x,y);
 R rb=-R(5)/3*f-R(13)/3*c-R(11)/2*sq(x)/sq(y)+x*(x+yb)/(sq(y)*yb)+sq(x)/sq(yb)*al(y)+sq(xb)/sq(y)*al(xb);
 if(part==1){
  R rn=-R(20)/3*c-12*sq(x)/sq(y)+2*x*(2+y+3*x*y-4*sq(y))/(sq(y)*yb)-2*x*(x-y+2*x*y)/(y*sq(yb))*al(x/y)-sq(x)/sq(yb)*l2(x/y);
  R en=-2*x*(1-3*x-2*x*yb)/sq(yb)*al(x)+sq(x)/sq(yb)*l2(x);
  return ca/3*r*rb+cf*tf*(r*rn+e*en);
 }
 R ra=(R(2)/3-sq(pi())/3)*f-c/2+x*(4+5*x)/(4*sq(y))-x*(10-19*x+16*sq(x))/y+x*(6-13*x+8*sq(x))/yb
 +(2*x*(1-2*xb*(x*y+xb*yb))/(y*yb)+x*(2-3*x-8*sq(xb))/sq(yb))*al(x/y)
 +(f+2*sq(x)/sq(yb))*l2(x/y)-(f-b)*lp(x/y,1-x/y)-gh(x,y)/2-ghb(xb,y)/2;
 R ea=-2*sq(x)/sq(yb)*l2(x)+(xb*(sq(x)+sq(xb)-2*xb*(1+2*x)*yb)+yb)/sq(yb)*lp(x,xb)
 +((3*x*(sq(x)+sq(xb))*(yb-y)-4*sq(x)*y)/(y*yb)-x*(2-3*x-8*sq(xb))/sq(yb))*al(x);
 return sq(ca)*(r*ra+e*ea)-ca*R(11)/2*r*rb;
}
R qf(const R&x,const R&y){return x/y*(1+R(1)/(y-x));}
R qh(const R&x,const R&y){return 2*qf(1-x,1-y)*lp(1-x,y)-2*qf(x,y)*(li(x)+li(1-y));}
R qhb(const R&x,const R&y){R a=qf(x,y),b=qf(1-x,1-y);return (a-b)*(2*li(1-x/y)+l2(y))+2*a*(li(1-y)-lp(x,y))+2*b*li(1-x);}
R ns_half(const R&x,const R&y,int sgn,int part){
 R xb=1-x,yb=1-y,cf=R(4)/3,ca=3,r=rho(x,y),e=erbl(x);
 if(r==R(0)&&e==R(0))return 0;
 R f=qf(x,y),b=qf(xb,yb),beta=(part==1?R(2)/3:R(-11));
 R p2=-cf*beta/2*(R(5)/3*f+x/y+f*al(x/y));
 if(part==1)return r*p2;
 R p1=sq(cf)*((R(4)/3-sq(pi())/3)*f+3*x/y-(3*f/2-x/(2*yb))*al(x/y)-(f-b)*lp(x/y,1-x/y)+(f+x/(2*yb))*l2(x/y));
 R p3=-cf*(cf-ca/2)*(R(4)/3*f+2*x/y+qh(x,y)-sgn*qhb(xb,y));
 R p4=-x/(2*yb)*(al(x)+l2(x)-2*lp(x,xb))*sq(cf);
 return r*(p1+p2+p3)+e*p4;
}
extern "C" double mp_gg_guard(double x,double y,double xi,int part){
 R c=xi,X=(1+R(x)/c)/2,Y=(1+R(y)/c)/2;
 return ((gg_half(X,Y,part)+gg_half(1-X,1-Y,part))/(2*c)).as_double();
}
extern "C" double mp_ns_guard(double x,double y,double xi,int sgn,int part){
 R c=xi,X=(1+R(x)/c)/2,Y=(1+R(y)/c)/2;
 return ((ns_half(X,Y,sgn,part)+ns_half(1-X,1-Y,sgn,part))/(2*c)).as_double();
}
