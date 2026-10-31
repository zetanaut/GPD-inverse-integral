// Independent numerical accumulation of the same captured GG source expression.
#include "mp_kernel.cpp"
R full(const R&x,const R&y,const R&c,int part){R X=(1+x/c)/2,Y=(1+y/c)/2;return (gg_half(X,Y,part)+gg_half(1-X,1-Y,part))/(2*c);}
extern "C" void gg_local_reference(double x,double xi,int n,const double* ys,double pole,int paired,double* out){
 R X=x,c=xi,P=pole;
 for(int j=0;j<n;j++){
  R t=ys[j],yp=(paired?P+t:t),ym=P-t;
  for(int part=0;part<2;part++){
   R v=full(X,yp,c,part)-full(yp,X,c,part);
   if(paired)v=v+full(X,ym,c,part)-full(ym,X,c,part);
   out[2*j+part]=v.as_double();
  }
 }
}
