! convolution.f90
!
! by Adam Freese
! part of the package tiktaalik for GPD evolution
!
! this comment added April 4, 2024, but parts of the the code were written over
! the course of 2024.

module convolution
  use gridspace,   only: push_forward
  use integration, only: integrate, quadrature_context, kernel3_interface
  use pixelation,  only: interpixel

  implicit none
  private

  integer,  parameter, private :: dp = kind(1d0)

  public :: pixel_conv

  contains

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! Pixelated shifts

    function pixel_conv(Kreg, Kpls, Kcst, xi, N, i, j, grid_type, nlo_quad) result(shift)
        real(dp), external   :: Kreg, Kpls, Kcst
        real(dp), intent(in) :: xi
        integer,  intent(in) :: N, i, j, grid_type
        real(dp) :: shift
        logical,intent(in),optional :: nlo_quad
        !
        shift = 0.0_dp
        shift = shift + pixel_conv_reg(Kreg, xi, N, i, j, grid_type, nlo_quad)
        shift = shift + pixel_conv_pls(Kpls, xi, N, i, j, grid_type, nlo_quad)
        shift = shift + pixel_conv_cst(Kcst, xi, N, i, j, grid_type)
    end function pixel_conv

    function pixel_conv_reg(kernel, xi, n_pixels, i, j, grid_type, nlo_quad) result(shift)
        procedure(kernel3_interface) :: kernel
        real(dp), intent(in) :: xi
        integer,  intent(in) :: n_pixels, i, j, grid_type
        real(dp) :: shift
        logical,intent(in),optional :: nlo_quad
        !
        real(dp) :: x, eta
        eta = real(2*i-1,dp)/real(n_pixels,dp) - 1.
        x = push_forward(eta, xi, n_pixels, grid_type)
        shift = integrate(pixel_conv_reg_integrand, x, xi, n_pixels, grid_type, j, .true., kernel, controlled=nlo_quad)
        return
    end function pixel_conv_reg

          function pixel_conv_reg_integrand(y,ctx) result(intd)
              real(dp), intent(in) :: y
              type(quadrature_context), intent(in) :: ctx
              real(dp) :: intd
              !
              real(dp) :: fy
              fy = interpixel(ctx%n_pixels, ctx%pixel_j, y, ctx%xi, ctx%grid_type)
              if(fy == 0.0_dp) then
                intd = 0.0_dp
                return
              endif
              intd = ctx%kernel(ctx%x,y,ctx%xi)*fy
          end function pixel_conv_reg_integrand


    function pixel_conv_pls(kernel, xi, n_pixels, i, j, grid_type, nlo_quad) result(shift)
        procedure(kernel3_interface) :: kernel
        real(dp), intent(in) :: xi
        integer,  intent(in) :: n_pixels, i, j, grid_type
        real(dp) :: shift
        logical,intent(in),optional :: nlo_quad
        !
        real(dp) :: x, eta
        eta = real(2*i-1,dp)/real(n_pixels,dp) - 1.
        x = push_forward(eta, xi, n_pixels, grid_type)
        shift = integrate(pixel_conv_pls_integrand, x, xi, n_pixels, grid_type, pixel_j=j, kernel=kernel, controlled=nlo_quad)
        return
    end function pixel_conv_pls

          function pixel_conv_pls_integrand(y,ctx) result(intd)
              real(dp), intent(in) :: y
              type(quadrature_context), intent(in) :: ctx
              real(dp) :: intd
              !
              real(dp) :: fy, fx
              fy = interpixel(ctx%n_pixels, ctx%pixel_j, y, ctx%xi, ctx%grid_type)
              fx = interpixel(ctx%n_pixels, ctx%pixel_j, ctx%x, ctx%xi, ctx%grid_type)
              intd = ctx%kernel(ctx%x,y,ctx%xi)*(fy-fx)
          end function pixel_conv_pls_integrand


    function pixel_conv_cst(kernel, xi, n_pixels, i, j, grid_type) result(shift)
        real(dp), external   :: kernel
        real(dp), intent(in) :: xi
        integer,  intent(in) :: n_pixels, i, j, grid_type
        real(dp) :: shift
        !
        real(dp) :: x, fx, eta
        shift = 0.0_dp
        if(i==j) then
          eta = real(2*i-1,dp)/real(n_pixels,dp) - 1.
          x = push_forward(eta, xi, n_pixels, grid_type)
          fx = interpixel(n_pixels, j, x, xi, grid_type)
          shift = kernel(x,xi) * fx
        endif
    end function pixel_conv_cst

end module convolution
