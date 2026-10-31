! specfun.f90
!
! part of the package tiktaalik for GPD evolution
!
! An amalgamation of special functions, taken from pre-existing Fotran 90 codes
! or copied from implementations in other languages. Authors of the original
! codes are attributed above the routines.

module specfun
  implicit none
  private

  integer,  parameter, private :: dp = kind(1d0)
  real(dp), parameter, private :: pi = acos(-1.0_dp)

  public :: dilog

  contains

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! Dilog algorithm due to Alexander Voigt, taken from 2201.01678
    ! Fortran90 version by Adam Freese, March 29, 2024

    recursive function dilog(z) result(L2)
        real(dp), intent(in) :: z
        real(dp) :: L2
        !
        if(z == -1.0_dp) then
          L2 = -pi**2/12.
        elseif(z == 0.0_dp) then
          L2 = 0.0_dp
        elseif(z == 0.5_dp) then
          L2 = pi**2/12. - (log(2.))**2/2.
        elseif(z == 1.0_dp) then
          L2 = pi**2/6.
        elseif(z == 2.0_dp) then
          L2 = pi**2/4.
        elseif(z < -1.0_dp) then
          L2 = log(1.-z)*(0.5*log(1.-z) - log(-z)) - pi**2/6. + dilog(1./(1.-z))
        elseif(z < 0.0_dp) then
          L2 = -dilog(z/(z-1.)) - 0.5*(log(1.-z))**2
        elseif(z > 2.0_dp) then
          L2 = -dilog(1./z) + pi**2/3. - 0.5*(log(z))**2
        elseif(z > 1.0_dp) then
          L2 = pi**2/6. - log(z)*( log(1.-1./z) + 0.5*log(z) ) + dilog(1.-1./z)
        elseif(z > 0.5_dp) then
          L2 = -dilog(1.-z) + pi**2/6. - log(z)*log(1.-z)
        else
          L2 = dilog_half(z)
        endif
    end function dilog

    function dilog_half(x) result(L2)
        real(dp), intent(in) :: x
        real(dp) :: L2
        !
        real(dp) :: px, qx
        real(dp), parameter :: p(0:6) = [ &
            & 0.9999999999999999502e+0_dp, -2.6883926818565423430e+0_dp, &
            & 2.6477222699473109692e+0_dp, -1.1538559607887416355e+0_dp, &
            & 2.0886077795020607837e-1_dp, -1.0859777134152463084e-2_dp, &
            & 0.0_dp ]
        real(dp), parameter :: q(0:6) =  [ &
            & 1.0000000000000000000e+0_dp, -2.9383926818565635485e+0_dp, &
            & 3.2712093293018635389e+0_dp, -1.7076702173954289421e+0_dp, &
            & 4.1596017228400603836e-1_dp, -3.9801343754084482956e-2_dp, &
            & 8.2743668974466659035e-4_dp ]
        real(dp) :: xpow(0:6)
        xpow(0) = 1.0_dp
        xpow(1) = x
        xpow(2) = x**2
        xpow(3) = x*xpow(2)
        xpow(4) = xpow(2)**2
        xpow(5) = xpow(4)*x
        xpow(6) = xpow(3)**2
        px = sum(xpow*p)
        qx = sum(xpow*q)
        L2 = x*px/qx
    end function dilog_half

end module specfun
