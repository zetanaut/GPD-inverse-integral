module mp_guard
use iso_c_binding
implicit none
interface
 function mp_gg_guard(x,y,xi,part) bind(C,name='mp_gg_guard') result(v)
 import c_double,c_int
 real(c_double),value::x,y,xi
 integer(c_int),value::part
 real(c_double)::v
 end function
 function mp_ns_guard(x,y,xi,sgn,part) bind(C,name='mp_ns_guard') result(v)
 import c_double,c_int
 real(c_double),value::x,y,xi
 integer(c_int),value::sgn,part
 real(c_double)::v
 end function
end interface
end module
