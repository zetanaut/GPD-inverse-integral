! constants.f90
!
! by Adam Freese
! part of the package tiktaalik for GPD evolution
!
! Code written in 2017, added to tiktaalik package in March 2024.

module constants

  implicit none
  public

  integer, parameter, private :: dp = kind(1d0)

  ! Casimir invariants
  real(dp), public, parameter :: CF = 4.0_dp/3.0_dp
  real(dp), public, parameter :: CA = 3.0_dp
  real(dp), public, parameter :: TF = 1.0_dp/2.0_dp

  ! Mathematical constants
  real(dp), public, parameter :: pi = acos(-1.0_dp)
  real(dp), public, parameter :: zeta2 = pi**2 / 6.0_dp
  real(dp), public, parameter :: zeta3 = 1.2020569031_dp
  real(dp), public, parameter :: zeta4 = pi**4 / 90.0_dp

  ! Current quark masses for effective quark number thresholds
  real(dp), public, parameter :: cMass  = 1.29_dp
  real(dp), public, parameter :: bMass  = 4.18_dp
  real(dp), public, parameter :: tMass  = 172.44_dp
  real(dp), public, parameter :: cMass2 = cMass**2
  real(dp), public, parameter :: bMass2 = bMass**2
  real(dp), public, parameter :: tMass2 = tMass**2

  ! QCD running coupling
  real(dp), public, parameter :: alphaMZ = 0.118_dp
  real(dp), public, parameter :: MZ = 91.1876_dp
  real(dp), public, parameter :: MZ2 = MZ**2
  real(dp), public, parameter, dimension(3:6) :: &
       & LambdaQCD = [ 0.332_dp, 0.292_dp, 0.210_dp, 0.089_dp ]

end module constants
