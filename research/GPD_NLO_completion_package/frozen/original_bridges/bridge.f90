module diagnostic_bridge
use iso_c_binding
use matevo
use gridspace
use kernels_nlo
use alpha_qcd,only:get_alpha_QCD
use constants,only:pi
implicit none
contains
subroutine build_raw(nx,xi,gt,x,k0,k1) bind(C,name='build_raw')
integer(c_int),value::nx,gt
real(c_double),value::xi
real(c_double),intent(out)::x(nx),k0(nx,nx,6),k1(nx,nx,6)
real(c_double)::lo(nx,nx,1),hi(nx,nx,1),a
integer::i,s
call make_kernels(nx,1,[xi],gt)
a=get_alpha_QCD(2._c_double)/(2._c_double*pi)
do i=1,nx
x(i)=push_forward(real(2*i-1,c_double)/nx-1._c_double,xi,nx,gt)
enddo
do s=1,6
select case(s)
case(1)
lo=kernel_V_QQ(2._c_double,nx,1,3,.false.,1)
hi=kernel_V_QQ(2._c_double,nx,1,3,.true.,1)
case(2)
lo=kernel_V_QQ(2._c_double,nx,1,3,.false.,-1)
hi=kernel_V_QQ(2._c_double,nx,1,3,.true.,-1)
case(3)
lo=kernel_V_QQ(2._c_double,nx,1,3,.false.,0)
hi=kernel_V_QQ(2._c_double,nx,1,3,.true.,0)
case(4)
lo=kernel_V_QG(2._c_double,nx,1,3,.false.)
hi=kernel_V_QG(2._c_double,nx,1,3,.true.)
case(5)
lo=kernel_V_GQ(2._c_double,nx,1,3,.false.)
hi=kernel_V_GQ(2._c_double,nx,1,3,.true.)
case(6)
lo=kernel_V_GG(2._c_double,nx,1,3,.false.)
hi=kernel_V_GG(2._c_double,nx,1,3,.true.)
end select
k0(:,:,s)=lo(:,:,1)/a
k1(:,:,s)=(hi(:,:,1)-lo(:,:,1))/(a*a)
enddo
end subroutine
subroutine point_k1(x,y,xi,vals) bind(C,name='point_k1')
real(c_double),value::x,y,xi
real(c_double),intent(out)::vals(6)
vals(1)=KV1_NSp_pls(x,y,xi)+3._c_double*KV1_NSp_pls_nfl(x,y,xi)
vals(2)=KV1_NSm_pls(x,y,xi)+3._c_double*KV1_NSm_pls_nfl(x,y,xi)
vals(3)=vals(1)+3._c_double*KV1_qq_reg(x,y,xi)
vals(4)=3._c_double*KV1_qG_reg(x,y,xi)
vals(5)=KV1_Gq_reg(x,y,xi)+3._c_double*KV1_Gq_reg_nfl(x,y,xi)
vals(6)=KV1_GG_pls(x,y,xi)+3._c_double*KV1_GG_pls_nfl(x,y,xi)
end subroutine
end module
