! alpha_qcd.f90
!
! by Adam Freese
! part of the package tiktaalik for GPD evolution
!
! Created March 5, 2024.
! Based on Python code written by Nobuo Sato.

module alpha_qcd
  use constants

  implicit none
  public

  integer,  parameter, private :: dp = kind(1d0)

  contains

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! QCD alpha function

    function get_alpha_qcd(Q2) result(alpha)
        ! For now, just assume norder=1
        real(dp), intent(in) :: Q2
        !integer, intent(in) :: norder
        real(dp) :: alpha
        !
        real(dp) :: alpha0
        if(Q2==MZ2) then
          alpha = alphaMZ
        elseif(Q2==tMass2) then
          alpha = alpha_mt2()
        elseif(Q2==bMass2) then
          alpha = alpha_mb2()
        elseif(Q2==cMass2) then
          alpha = alpha_mc2()
        elseif(Q2 > tMass2) then
          alpha0 = alpha_mt2()
          alpha = evolve_alpha_fixed_nfl(alpha0, tMass2, Q2, 6, 17, 1)
        elseif(Q2 > bMass2) then
          alpha0 = alpha_mb2()
          alpha = evolve_alpha_fixed_nfl(alpha0, bMass2, Q2, 5, 17, 1)
        elseif(Q2 > cMass2) then
          alpha0 = alpha_mc2()
          alpha = evolve_alpha_fixed_nfl(alpha0, cMass2, Q2, 4, 17, 1)
        else
          alpha0 = alpha_mc2()
          alpha = evolve_alpha_fixed_nfl(alpha0, cMass2, Q2, 3, 17, 1)
        endif
    end function get_alpha_qcd

    function evolve_alpha_fixed_nfl(alpha0, Q20, Q2, nfl, nsteps, norder) result(alpha)
        real(dp), intent(in) :: alpha0, Q20, Q2
        integer,  intent(in) :: nfl, nsteps, norder
        real(dp) :: alpha
        ! RK4
        real(dp) :: t, h, k1, k2, k3, k4, Q2c
        integer :: i
        ! Initialize
        t = 0.0_dp
        h = log(Q2/Q20) / real(nsteps)
        alpha = alpha0
        Q2c = Q20
        do i=1, nsteps, 1
          k1 = get_beta_qcd(alpha,         nfl, norder)
          k2 = get_beta_qcd(alpha+h*k1/2., nfl, norder)
          k3 = get_beta_qcd(alpha+h*k2/2., nfl, norder)
          k4 = get_beta_qcd(alpha+h*k3   , nfl, norder)
          alpha = alpha + h/6.*(k1+2.*k2+2.*k3+k4)
          Q2c = Q2c * exp(h)
        enddo
    end function evolve_alpha_fixed_nfl

    function alpha_mt2() result(alpha)
        real(dp) :: alpha
        !
        alpha = evolve_alpha_fixed_nfl(alphaMZ, MZ2, tMass2, 5, 17, 1)
    end function alpha_mt2

    function alpha_mb2() result(alpha)
        real(dp) :: alpha
        !
        alpha = evolve_alpha_fixed_nfl(alphaMZ, MZ2, bMass2, 5, 17, 1)
    end function alpha_mb2

    function alpha_mc2() result(alpha)
        real(dp) :: alpha
        !
        real(dp) :: alphaMB
        alphaMB = alpha_mb2()
        alpha = evolve_alpha_fixed_nfl(alphaMB, bMass2, cMass2, 4, 17, 1)
    end function alpha_mc2

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! QCD beta function

    function get_beta_qcd(alpha, nfl, norder) result(beta)
        real(dp), intent(in) :: alpha
        integer,  intent(in) :: nfl, norder
        real(dp) :: beta
        !
        real(dp) :: beta0, beta1, beta2
        beta0 = ( 11.*CA - 4.*real(nfl)*TF ) / (12.*pi)
        beta1 = ( 17.*CA**2 - real(nfl)*TF*(10.*CA+6.*CF) ) / (24.*pi**2)
        beta2 = ( 2857.-5033./9.*real(nfl)+325/27.*real(nfl)**2 ) / (128.*pi**3)
        beta = beta0
        if(norder >= 1) beta = beta + alpha*beta1
        if(norder >= 2) beta = beta + alpha**2*beta2
        beta = -beta * alpha**2
    end function get_beta_qcd

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! Effective number of flavors

    function get_neff(Q2) result(nfl)
        real(dp), intent(in) :: Q2
        integer :: nfl
        !
        nfl = 3
        if(Q2 >= tMass2) then
          nfl = 6
        elseif(Q2 >= bMass2) then
          nfl = 5
        elseif(Q2 >= cMass2) then
          nfl = 4
        endif
    end function get_neff

end module alpha_qcd
