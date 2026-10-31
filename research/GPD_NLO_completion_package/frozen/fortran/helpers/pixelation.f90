! pixelation.f90
!
! by Adam Freese
! part of the package tiktaalik for GPD evolution
!
! this comment added April 4, 2024, but parts of the the code were written over
! the course of 2024.

module pixelation
  use gridspace, only: pull_back

  implicit none
  private

  integer, parameter :: dp = kind(1d0)

  ! Interpolation stuff
  integer :: N_order_cache
  integer :: Nx_cache
  real(dp), dimension(:), allocatable :: w_table

  public :: interpixel, initialize_lagrange_weights

  contains

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! The interpixel

    function interpixel(n_pixels, i, x, xi, grid_type) result(f)
        ! Gives a function from polynomial interpolation of a pixel.
        ! One *MUST* call initialize_lagrange_weights before this!
        integer,  intent(in) :: n_pixels, i, grid_type
        real(dp), intent(in) :: x, xi
        real(dp) :: f
        !
        real(dp) :: eta
        ! Interpolate in the pulled back eta space
        eta = pull_back(x, xi, n_pixels, grid_type)
        call interpolate_lagrange_pixel(i, eta, f, grid_type)
    end function interpixel

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! Initialization routines

    subroutine initialize_lagrange_weights(nx, norder)
        integer, intent(in) :: nx, norder
        !
        real(dp) :: dx
        integer :: i, j
        ! Keep track of nx and norder values
        N_order_cache = norder
        Nx_cache = nx
        ! Assume a linear spacing
        dx = 2. / real(nx,dp)
        ! Fill the weight array
        if(allocated(w_table)) deallocate(w_table)
        allocate(w_table(norder))
        do i=1, norder, 1
          w_table(i) = 1.0_dp
          do j=1, norder, 1
            if(i.ne.j) w_table(i) = w_table(i) / (dx*real(i-j,dp))
          end do
        end do
    end subroutine initialize_lagrange_weights

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! Lagrange interpolation
    ! optimized for linearly spaced x grid

    subroutine interpolate_lagrange_pixel(ix, x, f, grid_type)
        ! Piecewise polynomial interpolation from connecting Newton polynomials.
        real(dp), intent(in)  :: x
        integer,  intent(in)  :: ix, grid_type
        real(dp), intent(out) :: f
        !
        integer :: iLoc, iStart, iEnd, region_low, region_high, cross_low, cross_high
        real(dp) :: x0
        if(grid_type == 4) then
          iLoc = min(max(int(floor(0.5_dp*(real(Nx_cache,dp)*(x+1._dp)+1._dp))),1),Nx_cache)
        else
          iLoc = locate(x, Nx_cache)
        endif
        region_low = 1
        region_high = Nx_cache
        if((grid_type == 3 .or. grid_type == 4) .and. mod(Nx_cache,4) == 2) then
          ! The crossover is a native node on the 4m+2 smooth grid. Its
          ! continuous value is shared, while stencils on either side remain
          ! in their own ERBL/DGLAP region, as in a piecewise spectral grid.
          cross_low = (Nx_cache+2)/4
          cross_high = Nx_cache+1-cross_low
          if(x < -0.5_dp) then
            region_high = cross_low
          elseif(x > 0.5_dp) then
            region_low = cross_high
          else
            region_low = cross_low
            region_high = cross_high
          endif
        endif
        iStart = min(max(iLoc-(N_order_cache-1)/2, region_low), &
                     & region_high+1-N_order_cache)
        iEnd   = iStart + N_order_cache - 1
        if(ix < iStart .or. ix > iEnd) then
          f = 0.0_dp
          return
        endif
        x0 = -1. + real(2*iStart-1,dp)/real(Nx_cache,dp)
        f = lagrange_basis(x, x0, ix-iStart+1)
    end subroutine interpolate_lagrange_pixel

    function lagrange_basis(x, x0, i) result(f)
        real(dp), intent(in) :: x, x0
        integer,  intent(in) :: i
        real(dp) :: f
        !
        real(dp) :: dx, lfact(n_order_cache)
        integer :: j
        dx = 2. / real(nx_cache,dp)
        do j=1, n_order_cache, 1
          lfact(j) = x - x0 - dx*real(j-1,dp)
        end do
        lfact(i) = 1.
        f = w_table(i) * product(lfact)
    end function lagrange_basis

    function locate(x, nx) result(iloc)
        real(dp), intent(in) :: x
        integer,  intent(in) :: nx
        integer :: iloc
        !
        iloc = floor( 0.5*(real(nx,dp)*(x+1.)+1.) )
        if(iloc < 1) iloc = 1
        if(iloc > nx) iloc = nx
    end function locate

end module pixelation
