#include "mp_kernel.cpp"
R pown(const R&a,int n){R b;mpfr_pow_si(b.v,a.v,n,MPFR_RNDN);return b;}
R qg_f_a(const R&x,const R&y){
 R cf=R(4)/3,ca=3,tf=R(1)/2,zeta2=sq(pi())/6;
 R f=(x/pown(y,2));
 return f;
}
R qg_f_c(const R&x,const R&y){
 R cf=R(4)/3,ca=3,tf=R(1)/2,zeta2=sq(pi())/6;
 R xbar=(R(1.0)-x);
 R f=((x/pown(y,2))*(((R(2.0)*xbar)*y)-x));
 return f;
}
R qg_h_v(const R&x,const R&y){
 R cf=R(4)/3,ca=3,tf=R(1)/2,zeta2=sq(pi())/6;
 R xbar=(R(1.0)-x);
 R ybar=(R(1.0)-y);
 R qgfv=((-qg_f_a(x,y))-(R(2.0)*qg_f_c(x,y)));
 R qgfvbar=((-qg_f_a(xbar,ybar))-(R(2.0)*qg_f_c(xbar,ybar)));
 R h=((((-R(2.0))*qgfvbar)*lp(xbar,y))-((R(2.0)*qgfv)*(li(x)+li(ybar))));
 return h;
}
R qg_hbar_v(const R&x,const R&y){
 R cf=R(4)/3,ca=3,tf=R(1)/2,zeta2=sq(pi())/6;
 R xbar=(R(1.0)-x);
 R ybar=(R(1.0)-y);
 R qgfv=((-qg_f_a(x,y))-(R(2.0)*qg_f_c(x,y)));
 R qgfvbar=((-qg_f_a(xbar,ybar))-(R(2.0)*qg_f_c(xbar,ybar)));
 R h=((((qgfvbar+qgfv)*((R(2.0)*li((R(1.0)-(x/y))))+l2(y)))+((R(2.0)*qgfv)*(li(ybar)-lp(x,y))))-((R(2.0)*qgfvbar)*li(xbar)));
 return h;
}
R kv1_qg_half(const R&x,const R&y){
 if(rho(x,y)==R(0)&&erbl(x)==R(0))return 0;
 R cf=R(4)/3,ca=3,tf=R(1)/2,zeta2=sq(pi())/6;
 R k=R(0.0);
 R xbar=(R(1.0)-x);
 R ybar=(R(1.0)-y);
 R qgfv=((-qg_f_a(x,y))-(R(2.0)*qg_f_c(x,y)));
 R qgfa=(-qg_f_a(x,y));
 R qgfbarv=((-qg_f_a(xbar,ybar))-(R(2.0)*qg_f_c(xbar,ybar)));
 R qgfbara=(-qg_f_a(xbar,ybar));
 R qghv=qg_h_v(x,y);
 R qghbarv=qg_hbar_v(xbar,y);
 R piece1=((tf*cf)*((((((((R(2.0)*(R(5.0)-(R(2.0)*zeta2)))*qgfv)-(R(4.0)*qgfa))+(x/(y*ybar)))+((R(2.0)*((qgfv+qgfbarv)+((((R(2.0)*x)-(R(2.0)*y))+(x*y))/((R(2.0)*y)*pown(ybar,2)))))*al((x/y))))-((R(2.0)*((qgfv+qgfbarv)+(R(1.0)/(y*ybar))))*al((R(1.0)-(x/y)))))+(qgfbara*l2((x/y))))+((qgfv+qgfbarv)*l2(((y/x)-R(1.0))))));
 R piece2=((tf*ca)*(((((((((((R(6.0)*x)*(R(3.0)-(R(4.0)*x)))/ybar)-(((R(2.0)*x)*(R(11.0)-(R(16.0)*x)))/y))+(((R(4.0)*x)*(R(1.0)-(R(3.0)*x)))/pown(y,2)))-((R(2.0)*(((qgfv-(R(5.0)*qgfbarv))+((R(5.0)-(R(7.0)*x))/pown(ybar,2)))+(((R(2.0)*x)*((R(3.0)-(R(2.0)*x))-((R(12.0)*xbar)*y)))/(y*ybar))))*al((x/y))))+((R(2.0)*((qgfv+qgfbarv)+(R(1.0)/(y*ybar))))*al((R(1.0)-(x/y)))))+(((qgfv-qgfbarv)+((R(1.0)-(R(4.0)*x))/pown(ybar,2)))*l2((x/y))))-((qgfv+qgfbarv)*pown(al((R(1.0)-(x/y))),2)))-qghv)+qghbarv));
 R piece3=((tf*cf)*(((((-R(2.0))*(((qgfv+qgfbarv)+qgfbara)+((x*(y+(R(4.0)*pown(ybar,2))))/((R(2.0)*y)*pown(ybar,2)))))*al(x))-((R(2.0)*qgfv)*lp(xbar,x)))+(((qgfv-qgfbarv)-qgfbara)*l2(x))));
 R piece4=((tf*ca)*(((R(2.0)*(((qgfv-(R(5.0)*qgfbarv))+((R(5.0)-(R(7.0)*x))/pown(ybar,2)))+((x*(((R(5.0)-(R(6.0)*x))-(R(22.0)*y))+((R(28.0)*x)*y)))/(y*ybar))))*al(x))-(((qgfv-qgfbarv)+((R(1.0)-(R(4.0)*x))/pown(ybar,2)))*l2(x))));
 piece1=(piece1*rho(x,y));
 piece2=(piece2*rho(x,y));
 piece3=(piece3*erbl(x));
 piece4=(piece4*erbl(x));
 k=(((piece1+piece2)+piece3)+piece4);
 return k;
}
extern "C" double mp_qg_guard(double x,double y,double xi){
 R c=xi,X=(1+R(x)/c)/2,Y=(1+R(y)/c)/2;
 return ((kv1_qg_half(X,Y)-kv1_qg_half(1-X,1-Y))/(4*c*c)).as_double();
}
extern "C" void mp_qg_vector(int n,double x,double xi,const double*y,double*out){
 for(int j=0;j<n;j++)out[j]=mp_qg_guard(x,y[j],xi);
}
