program raw_cli
use iso_c_binding
use diagnostic_bridge, only: build_raw,point_k1
implicit none
integer(c_int)::n,grid
real(c_double)::xi,t0,t1,v(6),vc(6)
real(c_double),allocatable::x(:),k0(:,:,:),k1(:,:,:)
character(4096)::arg,out
call get_command_argument(1,arg);read(arg,*)n
call get_command_argument(2,arg);read(arg,*)xi
call get_command_argument(3,arg);read(arg,*)grid
call get_command_argument(4,out)
allocate(x(n),k0(n,n,6),k1(n,n,6))
call cpu_time(t0)
call build_raw(n,xi,grid,x,k0,k1)
call point_k1(xi,(1._c_double+xi)/2,xi,v)
call point_k1(.999_c_double*xi,(1._c_double+xi)/2,xi,vc)
call cpu_time(t1)
open(unit=99,file=trim(out),status='replace',access='stream',form='unformatted')
write(99)x,k0,k1,v,vc
close(99)
print *,'CPU_SECONDS',t1-t0
end program
