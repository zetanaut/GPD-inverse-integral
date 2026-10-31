// Coordinate-preserving integration of the UNCHANGED captured GG expression.
// All physical boundaries, offsets, Jacobians and cancellation remain MPFR until
// a complete mapped panel contribution is returned. Input x and xi remain the
// exact supplied binary64 numbers, not decimal-rounded surrogates.
#include "mp_kernel.cpp"
#include <algorithm>
#include <vector>
#include <limits>

R full(const R& x,const R& y,const R& c,int part){
 R X=(1+x/c)/2,Y=(1+y/c)/2;
 return (gg_half(X,Y,part)+gg_half(1-X,1-Y,part))/(2*c);
}
R difference(const R&x,const R&y,const R&c,int part){return full(x,y,c,part)-full(y,x,c,part);}
struct Panel{R a,b;int paired;Panel(R aa,R bb,int pp):a(aa),b(bb),paired(pp){}};
R smaller(const R&a,const R&b){return b<a?b:a;}
std::vector<Panel> make_plan(double xx,double cc){
 R x=xx,c=cc,z=abs(x);
 std::vector<R> cuts={-R(1),-z,-c,R(0),c,z,R(1)};
 std::sort(cuts.begin(),cuts.end(),[](const R&a,const R&b){return a<b;});
 std::vector<R> poles={-z,z},radii;
 for(const R&p:poles){
  auto it=std::find_if(cuts.begin(),cuts.end(),[&](const R&a){return a==p;});
  auto j=it-cuts.begin();radii.push_back(R(.45)*smaller(p-cuts[j-1],cuts[j+1]-p));
 }
 for(int j=0;j<2;j++){cuts.push_back(poles[j]-radii[j]);cuts.push_back(poles[j]+radii[j]);}
 std::sort(cuts.begin(),cuts.end(),[](const R&a,const R&b){return a<b;});
 R scale=R(.25)*smaller(smaller(abs(x-c),abs(x+c)),smaller(z,1-z));
 std::vector<Panel> out;
 auto half_panels=[&](const R&a,const R&b){R mid=(a+b)/2;out.emplace_back(a,mid,0);out.emplace_back(b,mid,0);};
 for(size_t i=0;i+1<cuts.size();i++){
  R a=cuts[i],b=cuts[i+1],mid=(a+b)/2;bool excluded=false;
  for(int j=0;j<2;j++)if(abs(mid-poles[j])<radii[j])excluded=true;
  if(excluded)continue;
  R delta=(b-a)/2;
  while(delta>scale){half_panels(a+delta/2,a+delta);half_panels(b-delta,b-delta/2);delta=delta/2;}
  half_panels(a,a+delta);half_panels(b-delta,b);
 }
 for(int j=0;j<2;j++)out.emplace_back(poles[j],radii[j],1);
 return out;
}
R mapped(const R&x,const R&c,const Panel&p,const R&u,int part,int round_coordinates){
 R y,jac,value;
 if(p.paired){
  R t=p.b*u*u,yp=p.a+t,ym=p.a-t;
  if(round_coordinates){yp=R(yp.as_double());ym=R(ym.as_double());}
  jac=2*p.b*u;value=difference(x,yp,c,part)+difference(x,ym,c,part);
 }else{
  y=p.a+(p.b-p.a)*u*u;jac=2*abs(p.b-p.a)*u;
  if(round_coordinates)y=R(y.as_double());
  value=difference(x,y,c,part);
 }
 return jac*value;
}
bool valid(double x,double c){return std::isfinite(x)&&std::isfinite(c)&&c>0&&c<1&&std::abs(x)>0&&std::abs(x)<1&&std::min({std::abs(x),std::abs(std::abs(x)-c),1-std::abs(x)})>64*std::numeric_limits<double>::epsilon();}
extern "C" int gg_local_panel_count(double x,double c){return valid(x,c)?int(make_plan(x,c).size()):-1;}
extern "C" int gg_local_panel_values(double xx,double cc,int ip,int n,const double*u,double*out){
 if(!valid(xx,cc)||n<1)return 1;auto plan=make_plan(xx,cc);if(ip<0||ip>=int(plan.size()))return 2;
 R x=xx,c=cc;
 for(int j=0;j<n;j++){
  if(!(u[j]>0&&u[j]<1))return 3;
  for(int k=0;k<2;k++){R v=mapped(x,c,plan[ip],R(u[j]),k,0);if(!mpfr_number_p(v.v))return 4;out[2*j+k]=v.as_double();}
 }return 0;
}
extern "C" int gg_local_integral_mapped(double xx,double cc,int n,const double*u,const double*w,int round_coordinates,double*out){
 if(!valid(xx,cc)||n<1||round_coordinates<0||round_coordinates>1)return 1;
 auto plan=make_plan(xx,cc);R x=xx,c=cc;R result[2]={R(0),R(0)};
 for(const Panel&p:plan)for(int j=0;j<n;j++){
  if(!(u[j]>0&&u[j]<1)||!std::isfinite(w[j])||w[j]<=0)return 3;
  for(int k=0;k<2;k++){R v=mapped(x,c,p,R(u[j]),k,round_coordinates);if(!mpfr_number_p(v.v))return 4;result[k]=result[k]+R(w[j])*v;}
 }
 for(int k=0;k<2;k++)out[k]=result[k].as_double();return 0;
}
// Deterministic integration-domain check, including both halves of paired PV regions.
extern "C" int gg_local_domain_check(double x,double c,double*out){
 if(!valid(x,c))return 1;auto plan=make_plan(x,c);R total=0,moment=0;
 for(auto&p:plan){if(p.paired){total=total+2*p.b;moment=moment+2*p.a*p.b;}
 else{R width=abs(p.b-p.a);total=total+width;moment=moment+width*(p.a+p.b)/2;}}
 out[0]=total.as_double();out[1]=moment.as_double();return 0;
}
// Existing direct/paired point ABI retained for source-expression checks.
extern "C" void gg_local_reference(double x,double xi,int n,const double*ys,double pole,int paired,double*out){
 R X=x,c=xi,P=pole;
 for(int j=0;j<n;j++){R t=ys[j],yp=(paired?P+t:t),ym=P-t;
  for(int k=0;k<2;k++){R v=difference(X,yp,c,k);if(paired)v=v+difference(X,ym,c,k);out[2*j+k]=v.as_double();}}
}
