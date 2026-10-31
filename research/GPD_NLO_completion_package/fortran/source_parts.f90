! Diagnostic ABI only: uses captured kernels WITHOUT modifying their formulas.
! This is a source-term adapter, NOT independent validation of QCD expressions.
module source_parts_bridge
use iso_c_binding
use kernels_lo
use kernels_nlo
implicit none
contains
subroutine source_parts(order,sector,x,xi,n,y,reg,pls,local,want_local) bind(C,name='source_parts')
 integer(c_int),value::order,sector,n,want_local
 real(c_double),value::x,xi
 real(c_double),intent(in)::y(n)
 real(c_double),intent(out)::reg(n),pls(n),local
 integer::j
 reg=0._c_double;pls=0._c_double;local=0._c_double
 do j=1,n
  if(order==0)then
   select case(sector)
   case(0,1,2)
    pls(j)=K0_qq_pls(x,y(j),xi)
   case(3)
    reg(j)=3._c_double*(KA0_qG_reg(x,y(j),xi)+KVmA0_qG_reg(x,y(j),xi))
   case(4)
    reg(j)=KA0_Gq_reg(x,y(j),xi)+KVmA0_Gq_reg(x,y(j),xi)
   case(5)
    reg(j)=KVmA0_GG_reg(x,y(j),xi)
    pls(j)=KA0_GG_pls(x,y(j),xi)
   end select
  else
   select case(sector)
   case(0,2)
    pls(j)=KV1_NSp_pls(x,y(j),xi)+3._c_double*KV1_NSp_pls_nfl(x,y(j),xi)
    if(sector==2)reg(j)=3._c_double*KV1_qq_reg(x,y(j),xi)
   case(1)
    pls(j)=KV1_NSm_pls(x,y(j),xi)+3._c_double*KV1_NSm_pls_nfl(x,y(j),xi)
   case(3)
    reg(j)=3._c_double*KV1_qG_reg(x,y(j),xi)
   case(4)
    reg(j)=KV1_Gq_reg(x,y(j),xi)+3._c_double*KV1_Gq_reg_nfl(x,y(j),xi)
   case(5)
    pls(j)=KV1_GG_pls(x,y(j),xi)+3._c_double*KV1_GG_pls_nfl(x,y(j),xi)
   end select
  endif
 enddo
 if(want_local==0)return
 if(order==0)then
  select case(sector)
  case(0,1,2);local=K0_qq_cst(x,xi)
  case(5);local=KA0_GG_cst(x,xi)+3._c_double*KA0_GG_nfl(x,xi)
  end select
 else
  select case(sector)
  case(0,2);local=KV1_NSp_cst(x,xi)+3._c_double*KV1_NSp_cst_nfl(x,xi)
  case(1);local=KV1_NSm_cst(x,xi)+3._c_double*KV1_NSm_cst_nfl(x,xi)
  case(5);local=KV1_GG_cst(x,xi)+3._c_double*KV1_GG_cst_nfl(x,xi)
  end select
 endif
end subroutine
end module
