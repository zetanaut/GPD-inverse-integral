module repair_test_bridge
use iso_c_binding
use integration
use qg_wide
use kernels_nlo
use, intrinsic :: ieee_arithmetic
implicit none
contains
subroutine qg_points(n,x,y,xi,out,wide) bind(C)
 integer(c_int),value::n,wide
 real(c_double),value::x,xi
 real(c_double),intent(in)::y(n)
 real(c_double),intent(out)::out(n)
 integer::i
 do i=1,n
 if(wide==1)then
   out(i)=qg_guard_value(x,y(i),xi)
 else
   out(i)=KV1_qG_reg(x,y(i),xi)
 endif
 enddo
end subroutine
subroutine quad_test(mode,limit,value,error,okay) bind(C)
 integer(c_int),value::mode,limit
 real(c_double),intent(out)::value,error
 integer(c_int),intent(out)::okay
 type(quadrature_context)::ctx
 logical::ok
 ctx%n_pixels=mode
 call controlled_panel(test_function,0._c_double,1._c_double,ctx,.false., &
   & 1e-10_c_double,1e-10_c_double,limit,value,error,ok)
 okay=0
 if(ok)okay=1
end subroutine
function test_function(y,ctx) result(value)
 real(c_double),intent(in)::y
 type(quadrature_context),intent(in)::ctx
 real(c_double)::value
 select case(ctx%n_pixels)
 case(0)
 value=1._c_double
 case(1)
 value=y**8
 case(2)
 value=log(y)
 case(3)
 value=log(y)**2
 case(4)
 value=1._c_double/(1._c_double+((y-0.5_c_double)/0.0002_c_double)**2)
 case default
 value=ieee_value(0._c_double,ieee_quiet_nan)
 end select
end function
subroutine wide_arithmetic_points(n,ain,bin,out) bind(C)
 use software_wide
 integer(c_int),value::n
 real(c_double),intent(in)::ain(4,n),bin(4,n)
 real(c_double),intent(out)::out(4,6,n)
 type(wide_real)::a,b,c
 integer::i,j
 do i=1,n
  a=wide_real(ain(1,i),ain(2,i),ain(3,i),ain(4,i));b=wide_real(bin(1,i),bin(2,i),bin(3,i),bin(4,i))
  do j=1,6
   select case(j)
   case(1);c=a+b
   case(2);c=a-b
   case(3);c=a*b
   case(4);c=a/b
   case(5);c=log(abs(a))
   case(6);c=a**3
   end select
   out(:,j,i)=[c%hi,c%lo,c%tail,c%last]
  enddo
 enddo
end subroutine
end module
