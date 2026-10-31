! kernels_lo.f90
!
! by Adam Freese
! part of the package tiktaalik for GPD evolution
!
! Created June 26, 2024
! Consolidates and replaces code from kernels_v_lo.f90 and kernels_a_lo.f90,
! created before April 4 and on June 25, respectively.
!
! All kernels are taken from:
!   Belitsky, Fruend and Mueller
!   Nuclear Physics B 574 (2000) 347-406
!   Belitsky:1999hf
!   arxiv:hep-ph/9912379
!
! About this module:
!
! 1.) This file essentially encodes Eqs. (73) - (75) of BFM, with the auxiliary
! equations (76) - (78) contained in the kernels_common module.
! Since the V-type (helicity-indepdnent) kernels are equal to the A-type
! (helicity-dependent) kernels plus a regular function, and since regular
! functions are easier and faster to integrate, the fastest implementation for
! the matrix method is to create the A-type kernel and a V-A kernel, and then to
! add these to get the V-type kernel.
!
! 2.) Stuff independent of the number of active flavors (nfl) and linear in nfl
! have been separated.

module kernels_lo
  use constants,      only: CF, CA, TF, pi
  use kernels_common

  implicit none
  private

  integer,  parameter, private :: dp = kind(1d0)

  public :: K0_qq_pls, K0_qq_cst, &
      & KA0_qG_reg, KA0_Gq_reg, KVmA0_qG_reg, KVmA0_Gq_reg, &
      & KA0_GG_pls, KA0_GG_cst, KA0_GG_nfl, KVmA0_GG_reg

  contains

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! qq kernel pieces
    ! These are the same for V and A

    function K0_qq_pls(x, y, xi) result(K)
        real(dp), intent(in) :: x, y, xi
        real(dp) :: K
        !
        real(dp) :: X1, X2, Y1, Y2
        X1 = 0.5*(1.+x/xi)
        X2 = 0.5*(1.-x/xi)
        Y1 = 0.5*(1.+y/xi)
        Y2 = 0.5*(1.-y/xi)
        ! Eq. (74)
        K = (qq_f_a(X1,Y1) + qq_f_b(X1,Y1))*rho_step(X1,Y1) &
            &  + (qq_f_a(X2,Y2) + qq_f_b(X2,Y2))*rho_step(X2,Y2)
        K = CF*0.5*K/xi
    end function K0_qq_pls

    function K0_qq_cst(x, xi) result(K)
        real(dp), intent(in) :: x, xi
        real(dp) :: K
        !
        if(x > xi) then
          K = 1.5 + 2.*log(1.-x) + 0.5*((x-xi)*log((x-xi)*(1.+xi)) - (x+xi)*log((x+xi)*(1.-xi)))/xi
        elseif(x < xi .and. x > -xi) then
          K = 1.5 + log((1.-x**2)/(1.+xi)) + 0.5*((x-xi)*log(xi-x) - (x+xi)*log(x+xi))/xi
        elseif(x < -xi) then
          K = 1.5 + 2.*log(1.+x) + 0.5*((-x-xi)*log((-x-xi)*(1.+xi)) - (-x+xi)*log((-x+xi)*(1.-xi)))/xi
        else
          ! Continuous x -> +/-xi limit of either adjoining branch.
          K = 1.5_dp + log((1._dp-xi)/(2._dp*xi))
        endif
        K = K*CF
    end function K0_qq_cst

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! qG kernel pieces

    function KA0_qG_reg(x, y, xi) result(K)
        ! NOTICE:
        ! There's a missing factor nfl here.
        ! The kernel should be multplied by nfl from outside.
        real(dp), intent(in) :: x, y, xi
        real(dp) :: K
        !
        real(dp) :: X1, X2, Y1, Y2
        X1 = 0.5*(1.+x/xi)
        X2 = 0.5*(1.-x/xi)
        Y1 = 0.5*(1.+y/xi)
        Y2 = 0.5*(1.-y/xi)
        ! Eqs. (74), (75)
        K = - qG_f_a(X1,Y1)*rho_step(X1,Y1) + qG_f_a(X2,Y2)*rho_step(X2,Y2)
        K = 2.*TF*0.5*K/xi
        K = K / (2.*xi) ! Eq. (15) of BFM
    end function KA0_qG_reg

    function KVmA0_qG_reg(x, y, xi) result(K)
        ! NOTICE:
        ! There's a missing factor nfl here.
        ! The kernel should be multplied by nfl from outside.
        real(dp), intent(in) :: x, y, xi
        real(dp) :: K
        !
        real(dp) :: X1, X2, Y1, Y2
        X1 = 0.5*(1.+x/xi)
        X2 = 0.5*(1.-x/xi)
        Y1 = 0.5*(1.+y/xi)
        Y2 = 0.5*(1.-y/xi)
        ! Eqs. (74), (75)
        K = - 2.*qG_f_c(X1,Y1)*rho_step(X1,Y1) + 2.*qG_f_c(X2,Y2)*rho_step(X2,Y2)
        K = 2.*TF*0.5*K/xi
        K = K / (2.*xi) ! Eq. (15) of BFM
    end function KVmA0_qG_reg

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! Gq kernel pieces

    function KA0_Gq_reg(x, y, xi) result(K)
        real(dp), intent(in) :: x, y, xi
        real(dp) :: K
        !
        real(dp) :: X1, X2, Y1, Y2
        X1 = 0.5*(1.+x/xi)
        X2 = 0.5*(1.-x/xi)
        Y1 = 0.5*(1.+y/xi)
        Y2 = 0.5*(1.-y/xi)
        ! Eqs. (74), (75)
        K = Gq_f_a(X1,Y1)*rho_step(X1,Y1) - Gq_f_a(X2,Y2)*rho_step(X2,Y2)
        K = CF*0.5*K/xi
        K = K * (2.*xi) ! Eq. (15) of BFM
    end function KA0_Gq_reg

    function KVmA0_Gq_reg(x, y, xi) result(K)
        real(dp), intent(in) :: x, y, xi
        real(dp) :: K
        !
        real(dp) :: X1, X2, Y1, Y2
        X1 = 0.5*(1.+x/xi)
        X2 = 0.5*(1.-x/xi)
        Y1 = 0.5*(1.+y/xi)
        Y2 = 0.5*(1.-y/xi)
        ! Eqs. (74), (75)
        K = 2.*Gq_f_c(X1,Y1)*rho_step(X1,Y1) - 2.*Gq_f_c(X2,Y2)*rho_step(X2,Y2)
        K = CF*0.5*K/xi
        K = K * (2.*xi) ! Eq. (15) of BFM
    end function KVmA0_Gq_reg

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! GG kernel pieces

    function KA0_GG_pls(x, y, xi) result(K)
        real(dp), intent(in) :: x, y, xi
        real(dp) :: K
        !
        real(dp) :: X1, X2, Y1, Y2
        X1 = 0.5*(1.+x/xi)
        X2 = 0.5*(1.-x/xi)
        Y1 = 0.5*(1.+y/xi)
        Y2 = 0.5*(1.-y/xi)
        ! Eqs. (74), (75)
        K = (2.*GG_f_a(X1,Y1) + GG_f_b(X1,Y1))*rho_step(X1,Y1) &
            & + (2.*GG_f_a(X2,Y2) + GG_f_b(X2,Y2))*rho_step(X2,Y2)
        K = CA*0.5*K/xi
    end function KA0_GG_pls

    function KA0_GG_cst(x, xi) result(K)
        ! NOTICE:
        ! This is missing a piece proportional to nfl,
        ! which is instead in KA0_GG_nfl.
        real(dp), intent(in) :: x, xi
        real(dp) :: K
        !
        real(dp) :: beta0
        ! Constant from plus prescription
        K = 0.0_dp
        if(x > xi) then
          K = -2*x*(1.-x)/(1.-xi**2) + log((1.-x)**2/(1.-xi**2))
        elseif(x < -xi) then
          K = 2*x*(1.+x)/(1.-xi**2) + log((1.+x)**2/(1.-xi**2))
        else
          K = -2.*x**2/(1.+xi)/xi + log((1.-x**2)/(1.+xi)**2)
        endif
        K = K*CA
        ! beta0, but without the nfl term
        beta0 = - 11./3.*CA
        ! Second constant from Eq. (73)
        K = K - beta0/2.
        ! Third constant to get right forward limit, maybe related to Eq. (B7)?
        !K = K - 7./3.*CA
        ! ... decided to bake this into the if-else structures above; cancelled a 7/3 there.
    end function KA0_GG_cst

    function KA0_GG_nfl(x, xi) result(K)
        ! NOTICE:
        ! Missing an overall factor nfl, which should be multplied from outside.
        ! This is a delta-type kerenl.
        real(dp), intent(in) :: x, xi
        real(dp) :: K
        !
        real(dp) :: beta0
        ! Constant from Eq. (73), nfl term
        beta0 = 4./3.*TF
        K = - beta0/2.
    end function KA0_GG_nfl

    function KVmA0_GG_reg(x, y, xi) result(K)
        real(dp), intent(in) :: x, y, xi
        real(dp) :: K
        !
        real(dp) :: X1, X2, Y1, Y2
        X1 = 0.5*(1.+x/xi)
        X2 = 0.5*(1.-x/xi)
        Y1 = 0.5*(1.+y/xi)
        Y2 = 0.5*(1.-y/xi)
        ! Eqs. (74), (75)
        K = 2.*GG_f_c(X1,Y1)*rho_step(X1,Y1) + 2.*GG_f_c(X2,Y2)*rho_step(X2,Y2)
        K = CA*0.5*K/xi
    end function KVmA0_GG_reg

end module kernels_lo
