program precision_probe
use software_wide
implicit none
type(wide_real)::a,b,c
if(radix(1._dp)/=2.or.digits(1._dp)/=53)error stop 'IEEE binary64 required'
a=wide(1);b=wide(2._dp**(-100));c=wide(2._dp**(-200))
if(((a+b+c)-a)-b/=c)error stop 'software-wide cancellation preflight failed'
if(abs(to_double(log(wide(2)))-log(2._dp))>1e-16_dp)error stop 'software-wide logarithm preflight failed'
print *, 'SOFTWARE_WIDE_OK binary64_components=4'
end program
