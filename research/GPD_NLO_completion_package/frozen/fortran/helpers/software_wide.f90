! Portable software four-component, using IEEE binary64 only.
! Error-free sums/products follow Dekker/Knuth algorithms. No fast-math or FMA
! contraction: compile with -fno-fast-math -ffp-contract=off.
! Scope: normalized finite operands used by qg_wide. Not a full IEEE datatype.
! Four nonoverlapping binary64 components (~63 decimal significant digits).
! No extended exponent range. Underflow/overflow outside the tested kernel
! domain is not supported. No external library or native REAL128 required.
! Independent arithmetic/transcendental fixtures are required before kernels.
module software_wide
  use iso_fortran_env, only: real64
  implicit none
  private
  integer,parameter,public :: dp=real64
  type,public :: wide_real
    real(dp) :: hi=0._dp,lo=0._dp,tail=0._dp,last=0._dp
  end type
  type(wide_real),parameter :: one=wide_real(1._dp,0._dp), zero=wide_real(0._dp,0._dp)
  type(wide_real),parameter :: ln2=wide_real(6.931471805599452862e-01_dp, 2.319046813846299558e-17_dp, 5.707708438416212066e-34_dp, -3.582432210601811423e-50_dp)
  public :: wide, to_double, operator(+),operator(-),operator(*),operator(/),operator(**)
  public :: operator(==),operator(/=),operator(<),operator(>),operator(<=),operator(>=)
  public :: abs,log,sign,max,epsilon,tiny
  interface wide
    module procedure from_double,from_integer
  end interface
  interface operator(+)
    module procedure add
  end interface
  interface operator(-)
    module procedure sub,neg
  end interface
  interface operator(*)
    module procedure mul
  end interface
  interface operator(/)
    module procedure div
  end interface
  interface operator(**)
    module procedure power
  end interface
  interface operator(==)
    module procedure eq
  end interface
  interface operator(/=)
    module procedure ne
  end interface
  interface operator(<)
    module procedure lt
  end interface
  interface operator(>)
    module procedure gt
  end interface
  interface operator(<=)
    module procedure le
  end interface
  interface operator(>=)
    module procedure ge
  end interface
  interface abs
    module procedure wide_abs
  end interface
  interface log
    module procedure wide_log
  end interface
  interface sign
    module procedure wide_sign
  end interface
  interface max
    module procedure wide_max
  end interface
  interface epsilon
    module procedure wide_epsilon
  end interface
  interface tiny
    module procedure wide_tiny
  end interface
contains
  elemental function from_double(x) result(a)
    real(dp),intent(in)::x
    type(wide_real)::a
    a=wide_real(x,0._dp)
  end function
  elemental function from_integer(x) result(a)
    integer,intent(in)::x
    type(wide_real)::a
    a=wide_real(real(x,dp),0._dp)
  end function
  elemental function to_double(a) result(x)
    type(wide_real),intent(in)::a
    real(dp)::x
    x=((a%hi+a%lo)+a%tail)+a%last
  end function
  pure subroutine two_sum(a,b,s,e)
    real(dp),intent(in)::a,b
    real(dp),intent(out)::s,e
    real(dp)::v
    s=a+b;v=s-a;e=(a-(s-v))+(b-v)
  end subroutine
  pure subroutine quick_sum(a,b,s,e)
    real(dp),intent(in)::a,b
    real(dp),intent(out)::s,e
    s=a+b;e=b-(s-a)
  end subroutine
  pure subroutine two_product(a,b,p,e)
    real(dp),intent(in)::a,b
    real(dp),intent(out)::p,e
    real(dp)::ca,cb,ah,al,bh,bl
    real(dp),parameter::splitter=134217729._dp
    p=a*b;ca=splitter*a;cb=splitter*b
    ah=ca-(ca-a);al=a-ah;bh=cb-(cb-b);bl=b-bh
    e=((ah*bh-p)+ah*bl+al*bh)+al*bl
  end subroutine
  ! Nonoverlapping expansions, ascending magnitude internally. Operations
  ! accumulate error-free transforms before retaining the four largest limbs.
  pure function from_terms(terms) result(a)
    real(dp),intent(in)::terms(:)
    type(wide_real)::a
    real(dp)::e(48),f(48),h(48),q,qn,rr
    integer::i,j,n,k,bottom,top
    e=0._dp;n=0
    do i=1,size(terms)
      q=terms(i);k=0
      do j=1,n
        call two_sum(q,e(j),qn,rr)
        if(rr/=0._dp)then
          k=k+1;f(k)=rr
        endif
        q=qn
      enddo
      if(q/=0._dp.or.k==0)then
        k=k+1;f(k)=q
      endif
      n=k;e(1:n)=f(1:n)
    enddo
    ! Compress the accumulated expansion before truncation.
    q=e(n);bottom=n
    do i=n-1,1,-1
      call two_sum(q,e(i),qn,rr)
      if(rr/=0._dp)then
        h(bottom)=qn;bottom=bottom-1;q=rr
      else
        q=qn
      endif
    enddo
    h(bottom)=q;top=0
    do i=bottom+1,n
      call two_sum(h(i),q,qn,rr)
      if(rr/=0._dp)then
        top=top+1;e(top)=rr
      endif
      q=qn
    enddo
    top=top+1;e(top)=q
    a=wide_real(e(top),0._dp,0._dp)
    if(top>1)a%lo=e(top-1)
    if(top>2)a%tail=e(top-2)
    if(top>3)a%last=e(top-3)
  end function
  elemental function add(a,b) result(c)
    type(wide_real),intent(in)::a,b
    type(wide_real)::c
    c=from_terms([a%last,a%tail,a%lo,a%hi,b%last,b%tail,b%lo,b%hi])
  end function
  elemental function neg(a) result(c)
    type(wide_real),intent(in)::a
    type(wide_real)::c
    c=wide_real(-a%hi,-a%lo,-a%tail,-a%last)
  end function
  elemental function sub(a,b) result(c)
    type(wide_real),intent(in)::a,b
    type(wide_real)::c
    c=add(a,neg(b))
  end function
  elemental function mul(a,b) result(c)
    type(wide_real),intent(in)::a,b
    type(wide_real)::c
    real(dp)::av(4),bv(4),terms(32)
    integer::i,j,k
    av=[a%last,a%tail,a%lo,a%hi];bv=[b%last,b%tail,b%lo,b%hi];k=0
    do i=1,4
      do j=1,4
        call two_product(av(i),bv(j),terms(k+2),terms(k+1));k=k+2
      enddo
    enddo
    c=from_terms(terms)
  end function
  elemental function div(a,b) result(c)
    type(wide_real),intent(in)::a,b
    type(wide_real)::c,r
    real(dp)::q1,q2,q3,q4,q5
    q1=a%hi/b%hi;r=a-b*wide(q1)
    q2=r%hi/b%hi;r=r-b*wide(q2)
    q3=r%hi/b%hi;r=r-b*wide(q3)
    q4=r%hi/b%hi;r=r-b*wide(q4)
    q5=r%hi/b%hi;c=from_terms([q5,q4,q3,q2,q1])
  end function
  elemental function power(a,n) result(c)
    type(wide_real),intent(in)::a
    integer,intent(in)::n
    type(wide_real)::c,b
    integer::j
    c=one;b=a;j=abs(n)
    do while(j>0)
      if(mod(j,2)==1)c=c*b
      j=j/2
      if(j>0)b=b*b
    enddo
    if(n<0)c=one/c
  end function
  elemental logical function eq(a,b)
    type(wide_real),intent(in)::a,b
    eq=a%hi==b%hi.and.a%lo==b%lo.and.a%tail==b%tail.and.a%last==b%last
  end function
  elemental logical function ne(a,b)
    type(wide_real),intent(in)::a,b
    ne=.not.eq(a,b)
  end function
  elemental logical function lt(a,b)
    type(wide_real),intent(in)::a,b
    lt=a%hi<b%hi.or.(a%hi==b%hi.and.(a%lo<b%lo.or.(a%lo==b%lo.and.(a%tail<b%tail.or.(a%tail==b%tail.and.a%last<b%last)))))
  end function
  elemental logical function gt(a,b)
    type(wide_real),intent(in)::a,b
    gt=lt(b,a)
  end function
  elemental logical function le(a,b)
    type(wide_real),intent(in)::a,b
    le=.not.gt(a,b)
  end function
  elemental logical function ge(a,b)
    type(wide_real),intent(in)::a,b
    ge=.not.lt(a,b)
  end function
  elemental function wide_abs(a) result(c)
    type(wide_real),intent(in)::a
    type(wide_real)::c
    c=a
    if(a<zero)c=-a
  end function
  elemental function wide_sign(a,b) result(c)
    type(wide_real),intent(in)::a,b
    type(wide_real)::c
    c=abs(a)
    if(b<zero)c=-c
  end function
  elemental function wide_max(a,b) result(c)
    type(wide_real),intent(in)::a,b
    type(wide_real)::c
    c=a
    if(b>a)c=b
  end function
  elemental function wide_epsilon(a) result(c)
    type(wide_real),intent(in)::a
    type(wide_real)::c
    c=wide(2._dp**(-208))
  end function
  elemental function wide_tiny(a) result(c)
    type(wide_real),intent(in)::a
    type(wide_real)::c
    c=wide(tiny(1._dp))
  end function
  function wide_log(a) result(c)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    type(wide_real),intent(in)::a
    type(wide_real)::c,m,z,z2,term,inc,total
    integer::ex,j
    if(.not.ieee_is_finite(a%hi).or.a<=zero)error stop 'wide log outside finite positive domain'
    ex=exponent(a%hi)
    m=wide_real(scale(a%hi,-ex),scale(a%lo,-ex),scale(a%tail,-ex),scale(a%last,-ex))
    if(m%hi<0.7071067811865475244_dp)then
      m=wide_real(2._dp*m%hi,2._dp*m%lo,2._dp*m%tail,2._dp*m%last);ex=ex-1
    endif
    z=(m-one)/(m+one);z2=z*z;term=z;total=z
    do j=1,64
      term=term*z2;inc=term/wide(2*j+1);total=total+inc
      if(abs(inc%hi)<=1.e-66_dp*max(abs(total%hi),tiny(1._dp)))exit
    enddo
    if(j>64)error stop 'wide log series failed'
    c=wide(2)*total+wide(ex)*ln2
  end function
end module software_wide
