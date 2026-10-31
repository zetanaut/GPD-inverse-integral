"""Exact arithmetic-expression translation of captured QG guard to MPFR.
Original Fortran remains untouched; each number is formed at MP precision.
"""
from pathlib import Path
import re,ast,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def extract(text,name):return re.search(r'^    function '+name+r'\(.*?^    end function '+name+r'\s*$',text,re.M|re.S|re.I).group(0)
def expr(n):
 if isinstance(n,ast.Constant):return 'R('+repr(n.value)+')'
 if isinstance(n,ast.Name):return n.id
 if isinstance(n,ast.UnaryOp):return '(-'+expr(n.operand)+')' if isinstance(n.op,ast.USub) else expr(n.operand)
 if isinstance(n,ast.BinOp):
  if isinstance(n.op,ast.Pow):return 'pown('+expr(n.left)+','+str(n.right.value)+')'
  return '('+expr(n.left)+{ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/'}[type(n.op)]+expr(n.right)+')'
 if isinstance(n,ast.Call):
  name={'abslog':'al','log2':'l2','logprod':'lp','dilog':'li','rho_step':'rho','erbl_step':'erbl'}.get(n.func.id,n.func.id)
  return name+'('+','.join(expr(v) for v in n.args)+')'
 raise TypeError(ast.dump(n))
common=(ROOT/'frozen/fortran/kernels/kernels_common.f90').read_text();nlo=(ROOT/'frozen/fortran/kernels/kernels_nlo.f90').read_text();out=['#include "mp_kernel.cpp"','R pown(const R&a,int n){R b;mpfr_pow_si(b.v,a.v,n,MPFR_RNDN);return b;}'];record={}
for name in ['QG_f_a','QG_f_c','QG_h_V','QG_hbar_V','KV1_qG_half']:
 source=extract(nlo if name.startswith('KV') else common,name);record[name]=hashlib.sha256(source.encode()).hexdigest();body=[];pending=''
 for line in source.splitlines():
  code=line.split('!')[0].strip().lower()
  if not code or '::' in code or code.startswith('function') or code.startswith('end function'):continue
  code=code.replace('_dp','').replace('&','').strip();pending+=' '+code
  if line.split('!')[0].rstrip().endswith('&'):continue
  body.append(pending.strip());pending=''
 out.append('R '+name.lower()+'(const R&x,const R&y){')
 if name.startswith('KV'):out.append(' if(rho(x,y)==R(0)&&erbl(x)==R(0))return 0;')
 out.append(' R cf=R(4)/3,ca=3,tf=R(1)/2,zeta2=sq(pi())/6;');seen={'x','y','cf','ca','tf','zeta2'};last=None
 for code in body:
  lhs,rhs=code.split('=',1);lhs=lhs.strip();decl='' if lhs in seen else 'R ';seen.add(lhs);out.append(' '+decl+lhs+'='+expr(ast.parse(rhs.strip(),mode='eval').body)+';');last=lhs
 result=re.search(r'result\((\w+)\)',source,re.I).group(1).lower();out.append(' return '+result+';\n}')
out.append('''extern "C" double mp_qg_guard(double x,double y,double xi){
 R c=xi,X=(1+R(x)/c)/2,Y=(1+R(y)/c)/2;
 return ((kv1_qg_half(X,Y)-kv1_qg_half(1-X,1-Y))/(4*c*c)).as_double();
}
extern "C" void mp_qg_vector(int n,double x,double xi,const double*y,double*out){
 for(int j=0;j<n;j++)out[j]=mp_qg_guard(x,y[j],xi);
}''')
(ROOT/'candidate/mp_qg.cpp').write_text('\n'.join(out)+'\n');(ROOT/'work/action/QG_TRANSLATION.json').write_text(json.dumps({'method':'AST arithmetic-expression translation; constants promoted; published support unchanged','captured_function_hashes':record},indent=2)+'\n')
