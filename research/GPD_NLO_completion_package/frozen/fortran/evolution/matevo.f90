! matevo.f90
!
! by Adam Freese
! part of the package tiktaalik for GPD evolution
!
! Created June 6, 2024

module matevo
  use alpha_qcd,  only: get_alpha_QCD, get_neff
  use constants,  only: pi, TF
  use gridspace,  only: push_forward
  use convolution
  use kernels_common
  use kernels_lo
  use kernels_nlo
  use pixelation, only: initialize_lagrange_weights

  implicit none
  private

  integer,  parameter, private :: dp = kind(1d0)
  !real(dp), parameter, private :: pi = acos(-1.0_dp)

  integer, parameter, private :: nfl_min = 3
  integer, parameter, private :: nfl_max = 5

  ! Used for caching
  integer  :: nx_cache = 0, nxi_cache = 0, nQ2_cache = 0
  logical  :: l_nlo_cache = .false.

  real(dp), allocatable, dimension(:) :: xi_cache

  ! Zero kernel is just nx by nx
  real(dp), allocatable, dimension(:,:) :: K_zero, K_zero_2

  ! In kernels, indices are for (x,y,xi,nfl)
  ! For singlet/gluon, the first two indices go up to 2*nx,
  ! with the first nx values being singlet quark and the last nx being gluon.
  ! LO
  real(dp), allocatable, dimension(:,:,:,:) :: K_NS_0, KV_SG_0, KA_SG_0
  ! NLO
  real(dp), allocatable, dimension(:,:,:,:) :: KV_NS_1p, KV_NS_1m, KV_SG_1, KA_SG_1

  ! In evolution matrices, indices are for (x,y,xi,Q2)
  ! For singlet/gluon, the first two indices go up to 2*nx,
  ! with the first nx values being singlet quark and the last nx being gluon.
  real(dp), allocatable, dimension(:,:,:,:) :: M_NS_pls, M_NS_min, MV_SG, MA_SG

  public :: make_kernels, make_matrices, &
      & evomat_V_NS, evomat_V_SG, evomat_A_NS, evomat_A_SG, &
      & kernel_V_QQ, kernel_V_QG, kernel_V_GQ, kernel_V_GG, &
      & kernel_A_QQ, kernel_A_QG, kernel_A_GQ, kernel_A_GG, &
      & get_nx, get_nxi, get_nQ2, get_lnlo

  contains

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! Public methods to build the kernels and evolution matrices.
    ! IMPORTANT:
    ! - make_kernels **MUST** be called before any matrices can be made!
    !   and also before any routines to grab kernels are called.
    ! - make_matrices **MUST** be called before any routines to grab the
    !   evolution matrices are called.

    subroutine make_kernels(nx, nxi, xi_array, grid_type)
        ! Public method to initialize the kernel matrices
        ! This **MUST** be called before any evolution matrix routines!
        integer,  intent(in) :: nx, nxi, grid_type
        real(dp), intent(in) :: xi_array(nxi)
        ! Everything is deallocated first as a safety measure
        call deallocate_all()
        ! Initialize Lagrange weights
        call initialize_lagrange_weights(nx, 6)
        ! Initialize the 2D grids
        call initialize_2D(nx, nxi, xi_array)
        ! Initialize the kernel matrices
        call make_kernels_NS_0(nx, nxi, grid_type)
        call make_kernels_NS_1(nx, nxi, grid_type)
        call make_kernels_SG_0(nx, nxi, grid_type)
        call make_kernels_SG_1(nx, nxi, grid_type)
    end subroutine make_kernels

    subroutine make_matrices(nQ2, Q2_array, l_nlo)
        ! Public method to initialize the evolution matrices
        ! This **MUST** be called before anything to grab the evolution matrices!
        integer,  intent(in) :: nQ2
        real(dp), intent(in) :: Q2_array(nQ2)
        logical,  intent(in) :: l_nlo
        ! Keep track of whether we're doing NLO
        l_nlo_cache = l_nlo
        ! Keep track of the number of Q2 points
        nQ2_cache = nQ2
        ! Make evolution matrices
        call make_evomat_NS(nx_cache, nxi_cache, nQ2, Q2_array, l_nlo)
        call make_evomat_V_SG(nx_cache, nxi_cache, nQ2, Q2_array, l_nlo)
        call make_evomat_A_SG(nx_cache, nxi_cache, nQ2, Q2_array, l_nlo)
    end subroutine make_matrices

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! Methods to tell Python what the currently cached grid sizes are

    function get_nx() result(nx)
        integer :: nx
        nx = nx_cache
    end function get_nx

    function get_nxi() result(nxi)
        integer :: nxi
        nxi = nxi_cache
    end function get_nxi

    function get_nQ2() result(nQ2)
        integer :: nQ2
        nQ2 = nQ2_cache
    end function get_nQ2

    function get_lnlo() result(l_nlo)
        logical :: l_nlo
        l_nlo = l_nlo_cache
    end function get_lnlo

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! Public methods to return kenrel matrices

    function kernel_V_QQ(Q2, nx, nxi, nfl, l_nlo, i_ns_type) result(K)
        real(dp), intent(in) :: Q2
        integer,  intent(in) :: nx, nxi, nfl, i_ns_type
        logical,  intent(in) :: l_nlo
        real(dp), dimension(nx,nx,nxi) :: K
        real(dp) :: al2pi
        al2pi = get_alpha_QCD(Q2) / (2.*pi)
        K = al2pi * K_NS_0(:,:,:,nfl)
        if(l_nlo) then
          select case(i_ns_type)
          case(1)
            K = K + al2pi**2 * KV_NS_1p(:,:,:,nfl)
          case(-1)
            K = K + al2pi**2 * KV_NS_1m(:,:,:,nfl)
          case(0)
            K = K + al2pi**2 * KV_SG_1(1:nx,1:nx,:,nfl)
          end select
        endif
    end function kernel_V_QQ

    function kernel_V_QG(Q2, nx, nxi, nfl, l_nlo) result(K)
        real(dp), intent(in) :: Q2
        integer,  intent(in) :: nx, nxi, nfl
        logical,  intent(in) :: l_nlo
        real(dp), dimension(nx,nx,nxi) :: K
        real(dp) :: al2pi
        al2pi = get_alpha_QCD(Q2) / (2.*pi)
        K = al2pi * KV_SG_0(1:nx,nx+1:2*nx,:,nfl)
        if(l_nlo) then
          K = K + al2pi**2 * KV_SG_1(1:nx,nx+1:2*nx,:,nfl)
        endif
    end function kernel_V_QG

    function kernel_V_GQ(Q2, nx, nxi, nfl, l_nlo) result(K)
        real(dp), intent(in) :: Q2
        integer,  intent(in) :: nx, nxi, nfl
        logical,  intent(in) :: l_nlo
        real(dp), dimension(nx,nx,nxi) :: K
        real(dp) :: al2pi
        al2pi = get_alpha_QCD(Q2) / (2.*pi)
        K = al2pi * KV_SG_0(nx+1:2*nx,1:nx,:,nfl)
        if(l_nlo) then
          K = K + al2pi**2 * KV_SG_1(nx+1:2*nx,1:nx,:,nfl)
        endif
    end function kernel_V_GQ

    function kernel_V_GG(Q2, nx, nxi, nfl, l_nlo) result(K)
        real(dp), intent(in) :: Q2
        integer,  intent(in) :: nx, nxi, nfl
        logical,  intent(in) :: l_nlo
        real(dp), dimension(nx,nx,nxi) :: K
        real(dp) :: al2pi
        al2pi = get_alpha_QCD(Q2) / (2.*pi)
        K = al2pi * KV_SG_0(nx+1:2*nx,nx+1:2*nx,:,nfl)
        if(l_nlo) then
          K = K + al2pi**2 * KV_SG_1(nx+1:2*nx,nx+1:2*nx,:,nfl)
        endif
    end function kernel_V_GG

    function kernel_A_QQ(Q2, nx, nxi, nfl, l_nlo, i_ns_type) result(K)
        ! NS kernels are the same as for V, except + and - are swapped.
        real(dp), intent(in) :: Q2
        integer,  intent(in) :: nx, nxi, nfl, i_ns_type
        logical,  intent(in) :: l_nlo
        real(dp), dimension(nx,nx,nxi) :: K
        real(dp) :: al2pi
        al2pi = get_alpha_QCD(Q2) / (2.*pi)
        K = al2pi * K_NS_0(:,:,:,nfl)
        if(l_nlo) then
          select case(i_ns_type)
          case(1)
            K = K + al2pi**2 * KV_NS_1m(:,:,:,nfl)
          case(-1)
            K = K + al2pi**2 * KV_NS_1p(:,:,:,nfl)
          case(0)
            K = K + al2pi**2 * KA_SG_1(1:nx,1:nx,:,nfl)
          end select
        endif
    end function kernel_A_QQ

    function kernel_A_QG(Q2, nx, nxi, nfl, l_nlo) result(K)
        real(dp), intent(in) :: Q2
        integer,  intent(in) :: nx, nxi, nfl
        logical,  intent(in) :: l_nlo
        real(dp), dimension(nx,nx,nxi) :: K
        real(dp) :: al2pi
        al2pi = get_alpha_QCD(Q2) / (2.*pi)
        K = al2pi * KA_SG_0(1:nx,nx+1:2*nx,:,nfl)
        if(l_nlo) then
          K = K + al2pi**2 * KA_SG_1(1:nx,nx+1:2*nx,:,nfl)
        endif
    end function kernel_A_QG

    function kernel_A_GQ(Q2, nx, nxi, nfl, l_nlo) result(K)
        real(dp), intent(in) :: Q2
        integer,  intent(in) :: nx, nxi, nfl
        logical,  intent(in) :: l_nlo
        real(dp), dimension(nx,nx,nxi) :: K
        real(dp) :: al2pi
        al2pi = get_alpha_QCD(Q2) / (2.*pi)
        K = al2pi * KA_SG_0(nx+1:2*nx,1:nx,:,nfl)
        if(l_nlo) then
          K = K + al2pi**2 * KA_SG_1(nx+1:2*nx,1:nx,:,nfl)
        endif
    end function kernel_A_GQ

    function kernel_A_GG(Q2, nx, nxi, nfl, l_nlo) result(K)
        real(dp), intent(in) :: Q2
        integer,  intent(in) :: nx, nxi, nfl
        logical,  intent(in) :: l_nlo
        real(dp), dimension(nx,nx,nxi) :: K
        real(dp) :: al2pi
        al2pi = get_alpha_QCD(Q2) / (2.*pi)
        K = al2pi * KA_SG_0(nx+1:2*nx,nx+1:2*nx,:,nfl)
        if(l_nlo) then
          K = K + al2pi**2 * KA_SG_1(nx+1:2*nx,nx+1:2*nx,:,nfl)
        endif
    end function kernel_A_GG

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! Public methods to return evolution matrices

    function evomat_V_NS(nx, nxi, nQ2, nstype) result(M)
        integer,  intent(in) :: nx, nxi, nQ2, nstype
        real(dp), dimension(nx, nx, nxi, nQ2) :: M
        select case(nstype)
        case(1)
          M = M_NS_pls
        case(-1)
          M = M_NS_min
        end select
    end function evomat_V_NS

    function evomat_V_SG(nx, nxi, nQ2) result(M)
        integer,  intent(in) :: nx, nxi, nQ2
        real(dp), dimension(2*nx, 2*nx, nxi, nQ2) :: M
        M = MV_SG
    end function evomat_V_SG

    function evomat_A_NS(nx, nxi, nQ2, nstype) result(M)
        integer,  intent(in) :: nx, nxi, nQ2, nstype
        real(dp), dimension(nx, nx, nxi, nQ2) :: M
        select case(nstype)
        case(1)
          M = M_NS_min
        case(-1)
          M = M_NS_pls
        end select
    end function evomat_A_NS

    function evomat_A_SG(nx, nxi, nQ2) result(M)
        integer,  intent(in) :: nx, nxi, nQ2
        real(dp), dimension(2*nx, 2*nx, nxi, nQ2) :: M
        M = MA_SG
    end function evomat_A_SG

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! Methods to make kernel matrices

    subroutine make_kernels_NS_0(nx, nxi, grid_type)
        integer, intent(in) :: nx, nxi, grid_type
        integer :: ix, iy, iz!, nfl
        if(allocated(K_NS_0)) deallocate(K_NS_0)
        allocate(K_NS_0(nx,nx,nxi,nfl_min:nfl_max))
        !$OMP PARALLEL DO
        do ix=1, nx, 1
          do iy=1, nx, 1
            do iz=1, nxi, 1
              ! At leading order, no nfl dependence in QQ kernel.
              K_NS_0(ix,iy,iz,:) = pixel_conv(zero_func, K0_qq_pls, K0_qq_cst, xi_cache(iz), nx, ix, iy, grid_type)
            end do
          end do
        end do
        !$OMP END PARALLEL DO
    end subroutine make_kernels_NS_0

    subroutine make_kernels_NS_1(nx, nxi, grid_type)
        integer, intent(in) :: nx, nxi, grid_type
        if(allocated(KV_NS_1p)) deallocate(KV_NS_1p)
        if(allocated(KV_NS_1m)) deallocate(KV_NS_1m)
        allocate(KV_NS_1p(nx,nx,nxi,nfl_min:nfl_max))
        allocate(KV_NS_1m(nx,nx,nxi,nfl_min:nfl_max))
        call make_one_nlo_ns_kernel(nx, nxi, grid_type, &
            & zero_func, KV1_NSp_pls, KV1_NSp_cst, zero_func, KV1_NSp_pls_nfl, KV1_NSp_cst_nfl, &
            & KV_NS_1p)
        call make_one_nlo_ns_kernel(nx, nxi, grid_type, &
            & zero_func, KV1_NSm_pls, KV1_NSm_cst, zero_func, KV1_NSm_pls_nfl, KV1_NSm_cst_nfl, &
            & KV_NS_1m)
    end subroutine make_kernels_NS_1

    subroutine make_one_nlo_ns_kernel(nx, nxi, grid_type, freg0, fpls0, fcst0, freg1, fpls1, fcst1, kernel)
        integer, intent(in)   :: nx, nxi, grid_type
        real(dp), intent(out) :: kernel(nx,nx,nxi,nfl_min:nfl_max)
        real(dp), external :: freg0, fpls0, fcst0, freg1, fpls1, fcst1
        integer :: ix, iy, iz, nfl
        real(dp), allocatable, dimension(:,:,:,:) :: k0, k1
        allocate(k0(nx,nx,nxi,nfl_min:nfl_max))
        allocate(k1(nx,nx,nxi,nfl_min:nfl_max))
        !$OMP PARALLEL DO
        do ix=1, nx, 1
          do iy=1, nx, 1
            do iz=1, nxi, 1
              k0(ix,iy,iz,:) = pixel_conv(freg0, fpls0, fcst0, xi_cache(iz), nx, ix, iy, grid_type, nlo_quad=.true.)
              k1(ix,iy,iz,:) = pixel_conv(freg1, fpls1, fcst1, xi_cache(iz), nx, ix, iy, grid_type, nlo_quad=.true.)
            end do
          end do
        end do
        !$OMP END PARALLEL DO
        do nfl=nfl_min, nfl_max, 1
          k1(:,:,:,nfl) = k1(:,:,:,nfl) * real(nfl,dp)
        end do
        kernel = k0 + k1
        deallocate(k0, k1)
    end subroutine make_one_nlo_ns_kernel

    subroutine make_kernels_SG_0(nx, nxi, grid_type)
        integer, intent(in) :: nx, nxi, grid_type
        integer :: ix, iy, iz, nfl
        real(dp) :: eta, xnode
        real(dp), dimension(:,:,:), allocatable :: qq_nfl_0, qG_nfl_1, Gq_nfl_0, GG_nfl_0, GG_nfl_1
        if(allocated(KA_SG_0)) deallocate(KA_SG_0)
        if(allocated(KV_SG_0)) deallocate(KV_SG_0)
        allocate(KA_SG_0(2*nx,2*nx,nxi,nfl_min:nfl_max))
        allocate(KV_SG_0(2*nx,2*nx,nxi,nfl_min:nfl_max))
        ! Temporary sub-matrix arrays
        allocate(qq_nfl_0(nx,nx,nxi))
        allocate(qG_nfl_1(nx,nx,nxi))
        allocate(Gq_nfl_0(nx,nx,nxi))
        allocate(GG_nfl_0(nx,nx,nxi))
        allocate(GG_nfl_1(nx,nx,nxi))
        KA_SG_0 = 0.0_dp
        KV_SG_0 = 0.0_dp
        ! First, build up the A-type kernel
        !$OMP PARALLEL DO
        do ix=1, nx, 1
          do iy=1, nx, 1
            do iz=1, nxi, 1
              qG_nfl_1(ix,iy,iz) = pixel_conv(KA0_qG_reg, zero_func,  zero_func,  xi_cache(iz), nx, ix, iy, grid_type)
              Gq_nfl_0(ix,iy,iz) = pixel_conv(KA0_Gq_reg, zero_func,  zero_func,  xi_cache(iz), nx, ix, iy, grid_type)
              GG_nfl_0(ix,iy,iz) = pixel_conv(zero_func,  KA0_GG_pls, KA0_GG_cst, xi_cache(iz), nx, ix, iy, grid_type)
              GG_nfl_1(ix,iy,iz) = pixel_conv(zero_func,  zero_func,  KA0_GG_nfl, xi_cache(iz), nx, ix, iy, grid_type)
            end do
          end do
        end do
        !$OMP END PARALLEL DO
        ! For the QQ block, just copy over the NS kernel, since it's the same at LO.
        ! The NS kernel matrix is built before this method is called so this is safe.
        KA_SG_0(1:nx,1:nx,:,:) = K_NS_0(:,:,:,:)
        do nfl=nfl_min, nfl_max, 1
          !!KA_SG_0(1:nx,     1:nx,     :,nfl) = qq_nfl_0(:,:,:)
          KA_SG_0(1:nx,     nx+1:2*nx,:,nfl) = real(nfl,dp)*qG_nfl_1(:,:,:)
          KA_SG_0(nx+1:2*nx,1:nx,     :,nfl) = Gq_nfl_0(:,:,:)
          KA_SG_0(nx+1:2*nx,nx+1:2*nx,:,nfl) = GG_nfl_0(:,:,:) + real(nfl,dp)*GG_nfl_1(:,:,:)
        end do
        ! Next, the extra pieces for the V-type kernel
        !$OMP PARALLEL DO
        do ix=1, nx, 1
          do iy=1, nx, 1
            do iz=1, nxi, 1
              qG_nfl_1(ix,iy,iz) = pixel_conv(KVmA0_qG_reg, zero_func, zero_func, xi_cache(iz), nx, ix, iy, grid_type)
              Gq_nfl_0(ix,iy,iz) = pixel_conv(KVmA0_Gq_reg, zero_func, zero_func, xi_cache(iz), nx, ix, iy, grid_type)
              GG_nfl_0(ix,iy,iz) = pixel_conv(KVmA0_GG_reg, zero_func, zero_func, xi_cache(iz), nx, ix, iy, grid_type)
            end do
          end do
        end do
        !$OMP END PARALLEL DO
        KV_SG_0(1:nx,1:nx,:,:) = 0.0_dp
        do nfl=nfl_min, nfl_max, 1
          KV_SG_0(1:nx,     nx+1:2*nx,:,nfl) = real(nfl,dp)*qG_nfl_1(:,:,:)
          KV_SG_0(nx+1:2*nx,1:nx,     :,nfl) = Gq_nfl_0(:,:,:)
          KV_SG_0(nx+1:2*nx,nx+1:2*nx,:,nfl) = GG_nfl_0(:,:,:)
        end do
        ! And add the extra pieces to the A-type kernel
        KV_SG_0 = KV_SG_0 + KA_SG_0
        if(grid_type == 3 .or. grid_type == 4) then
          ! At an exactly collocated crossover the shrinking QG support has
          ! a finite nonuniform limit. For x=+/-xi, t=(xi-y)/(xi-x) gives
          ! +/- nfl*TF/xi*G(xi). Fixed-node quadrature cannot see this contact.
          do iz=1,nxi
            do ix=1,nx
              eta = real(2*ix-1,dp)/real(nx,dp) - 1.
              xnode = push_forward(eta,xi_cache(iz),nx,grid_type)
              if(abs(abs(xnode)-xi_cache(iz)) < 1e-12_dp) then
                do nfl=nfl_min,nfl_max
                  KV_SG_0(ix,nx+ix,iz,nfl) = KV_SG_0(ix,nx+ix,iz,nfl) &
                      & + sign(1._dp,xnode)*real(nfl,dp)*TF/xi_cache(iz)
                end do
              endif
            end do
          end do
        endif
        ! And we're done
        deallocate(qq_nfl_0)
        deallocate(qG_nfl_1)
        deallocate(Gq_nfl_0)
        deallocate(GG_nfl_0)
        deallocate(GG_nfl_1)
    end subroutine make_kernels_SG_0

    subroutine make_kernels_SG_1(nx, nxi, grid_type)
        integer, intent(in) :: nx, nxi, grid_type
        integer :: ix, iy, iz, nfl
        real(dp), dimension(:,:,:), allocatable :: qq_nfl_1, qG_nfl_1, Gq_nfl_0, Gq_nfl_1, GG_nfl_0, GG_nfl_1
        if(allocated(KA_SG_1)) deallocate(KA_SG_1)
        if(allocated(KV_SG_1)) deallocate(KV_SG_1)
        allocate(KA_SG_1(2*nx,2*nx,nxi,nfl_min:nfl_max))
        allocate(KV_SG_1(2*nx,2*nx,nxi,nfl_min:nfl_max))
        allocate(qq_nfl_1(nx,nx,nxi))
        allocate(qG_nfl_1(nx,nx,nxi))
        allocate(Gq_nfl_0(nx,nx,nxi))
        allocate(Gq_nfl_1(nx,nx,nxi))
        allocate(GG_nfl_0(nx,nx,nxi))
        allocate(GG_nfl_1(nx,nx,nxi))
        KA_SG_1 = 0.0_dp
        KV_SG_1 = 0.0_dp
        ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !  V-type kernel first
        !$OMP PARALLEL DO
        do ix=1, nx, 1
          do iy=1, nx, 1
            do iz=1, nxi, 1
              qq_nfl_1(ix,iy,iz) = pixel_conv(KV1_qq_reg,     zero_func,      zero_func,      xi_cache(iz), nx, ix, iy, grid_type, nlo_quad=.true.)
              qG_nfl_1(ix,iy,iz) = pixel_conv(KV1_qG_reg,     zero_func,      zero_func,      xi_cache(iz), nx, ix, iy, grid_type, nlo_quad=.true.)
              Gq_nfl_0(ix,iy,iz) = pixel_conv(KV1_Gq_reg,     zero_func,      zero_func,      xi_cache(iz), nx, ix, iy, grid_type, nlo_quad=.true.)
              Gq_nfl_1(ix,iy,iz) = pixel_conv(KV1_Gq_reg_nfl, zero_func,      zero_func,      xi_cache(iz), nx, ix, iy, grid_type, nlo_quad=.true.)
              GG_nfl_0(ix,iy,iz) = pixel_conv(zero_func,      KV1_GG_pls,     KV1_GG_cst,     xi_cache(iz), nx, ix, iy, grid_type, nlo_quad=.true.)
              GG_nfl_1(ix,iy,iz) = pixel_conv(zero_func,      KV1_GG_pls_nfl, KV1_GG_cst_nfl, xi_cache(iz), nx, ix, iy, grid_type, nlo_quad=.true.)
            end do
          end do
        end do
        ! Add nfl-independent and nfl-linear terms as needed.
        do nfl=nfl_min, nfl_max, 1
          KV_SG_1(1:nx,     1:nx,:,     nfl) = real(nfl,dp)*qq_nfl_1(:,:,:)
          KV_SG_1(1:nx,     nx+1:2*nx,:,nfl) = real(nfl,dp)*qG_nfl_1(:,:,:)
          KV_SG_1(nx+1:2*nx,1:nx,     :,nfl) = Gq_nfl_0(:,:,:) + real(nfl,dp)*Gq_nfl_1(:,:,:)
          KV_SG_1(nx+1:2*nx,nx+1:2*nx,:,nfl) = GG_nfl_0(:,:,:) + real(nfl,dp)*GG_nfl_1(:,:,:)
        end do
        ! For the QQ block, the plus-type NS kernel is a contribution, so we need to add it.
        ! (See Eq. (182).)
        ! The NS kernel matrix is built before this method is called so this is safe.
        KV_SG_1(1:nx,1:nx,:,:) = KV_SG_1(1:nx,1:nx,:,:) + KV_NS_1p(:,:,:,:)
        ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !  A-type kernel next
        !$OMP PARALLEL DO
        do ix=1, nx, 1
          do iy=1, nx, 1
            do iz=1, nxi, 1
              qq_nfl_1(ix,iy,iz) = pixel_conv(KA1_qq_reg,     zero_func,      zero_func,      xi_cache(iz), nx, ix, iy, grid_type)
              qG_nfl_1(ix,iy,iz) = pixel_conv(KA1_qG_reg,     zero_func,      zero_func,      xi_cache(iz), nx, ix, iy, grid_type)
              Gq_nfl_0(ix,iy,iz) = pixel_conv(KA1_Gq_reg,     zero_func,      zero_func,      xi_cache(iz), nx, ix, iy, grid_type)
              Gq_nfl_1(ix,iy,iz) = pixel_conv(KA1_Gq_reg_nfl, zero_func,      zero_func,      xi_cache(iz), nx, ix, iy, grid_type)
              GG_nfl_0(ix,iy,iz) = pixel_conv(zero_func,      KA1_GG_pls,     KA1_GG_cst,     xi_cache(iz), nx, ix, iy, grid_type)
              GG_nfl_1(ix,iy,iz) = pixel_conv(zero_func,      KA1_GG_pls_nfl, KA1_GG_cst_nfl, xi_cache(iz), nx, ix, iy, grid_type)
            end do
          end do
        end do
        ! Add nfl-independent and nfl-linear terms as needed.
        do nfl=nfl_min, nfl_max, 1
          KA_SG_1(1:nx,     1:nx,:,     nfl) = real(nfl,dp)*qq_nfl_1(:,:,:)
          KA_SG_1(1:nx,     nx+1:2*nx,:,nfl) = real(nfl,dp)*qG_nfl_1(:,:,:)
          KA_SG_1(nx+1:2*nx,1:nx,     :,nfl) = Gq_nfl_0(:,:,:) + real(nfl,dp)*Gq_nfl_1(:,:,:)
          KA_SG_1(nx+1:2*nx,nx+1:2*nx,:,nfl) = GG_nfl_0(:,:,:) + real(nfl,dp)*GG_nfl_1(:,:,:)
        end do
        ! For the QQ block, the minus-type NS kernel is a contribution, so we need to add it.
        ! (See Eq. (178).)
        ! The NS kernel matrix is built before this method is called so this is safe.
        KA_SG_1(1:nx,1:nx,:,:) = KA_SG_1(1:nx,1:nx,:,:) + KV_NS_1m(:,:,:,:)
        ! And we're done
        deallocate(qq_nfl_1)
        deallocate(qG_nfl_1)
        deallocate(Gq_nfl_0)
        deallocate(Gq_nfl_1)
        deallocate(GG_nfl_0)
        deallocate(GG_nfl_1)
    end subroutine make_kernels_SG_1

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! Methods to make evolution matrices

    subroutine make_evomat_NS(nx, nxi, nQ2, Q2_array, l_nlo)
        ! Just one non-singlet evolution matrix at leading order.
        ! I'll need to break this into multiple routines,
        ! for both V-type and A-type, but also plus-type and minus-type,
        ! after going to NLO. But I'll burn that bridge when I come to it.
        integer,  intent(in) :: nx, nxi, nQ2
        real(dp), intent(in) :: Q2_array(nQ2)
        logical,  intent(in) :: l_nlo
        real(dp), dimension(nx,nx) :: idnx
        integer :: ix, ixi, iQ2, nfl
        if(allocated(M_NS_pls)) deallocate(M_NS_pls)
        if(allocated(M_NS_min)) deallocate(M_NS_min)
        allocate(M_NS_pls(nx,nx,nxi,nQ2))
        allocate(M_NS_min(nx,nx,nxi,nQ2))
        ! Build an identity matrix
        idnx = 0.0_dp
        do ix=1, nx, 1
          idnx(ix,ix) = 1.0_dp
        end do
        ! Identity for no evolution at initial scale
        M_NS_pls = 0.0_dp
        M_NS_min = 0.0_dp
        do ixi=1, nxi, 1
          M_NS_pls(:,:,ixi,1) = idnx
          M_NS_min(:,:,ixi,1) = idnx
        end do
        ! Build evolution matrices for all other scales
        do iQ2=2, nQ2, 1
          !$OMP PARALLEL DO
          do ixi=1, nxi, 1
            ! First, an evolution matrix for prior Q2 step to current Q2 step
            nfl = get_neff(Q2_array(iQ2-1))
            M_NS_pls(:,:,ixi,iQ2) = idnx + &
                & rk4_NS(nx, nxi, Q2_array(iQ2-1), Q2_array(iQ2), K_NS_0(:,:,ixi,nfl), KV_NS_1p(:,:,ixi,nfl))
            M_NS_min(:,:,ixi,iQ2) = idnx + &
                & rk4_NS(nx, nxi, Q2_array(iQ2-1), Q2_array(iQ2), K_NS_0(:,:,ixi,nfl), KV_NS_1m(:,:,ixi,nfl))
            ! Then matrix multiplication to turn into matrix from initial Q2 to current Q2
            M_NS_pls(:,:,ixi,iQ2) = matmul(M_NS_pls(:,:,ixi,iQ2), M_NS_pls(:,:,ixi,iQ2-1))
            M_NS_min(:,:,ixi,iQ2) = matmul(M_NS_min(:,:,ixi,iQ2), M_NS_min(:,:,ixi,iQ2-1))
          end do
          !$OMP END PARALLEL DO
        end do
    end subroutine make_evomat_NS

    subroutine make_evomat_V_SG(nx, nxi, nQ2, Q2_array, l_nlo)
        integer,  intent(in) :: nx, nxi, nQ2
        real(dp), intent(in) :: Q2_array(nQ2)
        logical,  intent(in) :: l_nlo
        real(dp), dimension(2*nx,2*nx) :: id2nx
        integer :: ix, ixi, iQ2, nfl
        if(allocated(MV_SG)) deallocate(MV_SG)
        allocate(MV_SG(2*nx,2*nx,nxi,nQ2))
        ! Build an identity matrix
        id2nx = 0.0_dp
        do ix=1, 2*nx, 1
          id2nx(ix,ix) = 1.0_dp
        end do
        ! Identity for no evolution at initial scale
        MV_SG = 0.0_dp
        do ixi=1, nxi, 1
          MV_SG(:,:,ixi,1) = id2nx
        end do
        ! Build evolution matrices for all other scales
        do iQ2=2, nQ2, 1
          !$OMP PARALLEL DO
          do ixi=1, nxi, 1
            ! First, an evolution matrix for prior Q2 step to current Q2 step
            nfl = get_neff(Q2_array(iQ2-1))
            MV_SG(:,:,ixi,iQ2) = id2nx + &
                & rk4_SG(nx, nxi, Q2_array(iQ2-1), Q2_array(iQ2), &
                & KV_SG_0(:,:,ixi,nfl), KV_SG_1(:,:,ixi,nfl))
            ! Then matrix multiplication to turn into matrix from initial Q2 to current Q2
            MV_SG(:,:,ixi,iQ2) = matmul(MV_SG(:,:,ixi,iQ2), MV_SG(:,:,ixi,iQ2-1))
          end do
          !$OMP END PARALLEL DO
        end do
    end subroutine make_evomat_V_SG

    subroutine make_evomat_A_SG(nx, nxi, nQ2, Q2_array, l_nlo)
        integer,  intent(in) :: nx, nxi, nQ2
        real(dp), intent(in) :: Q2_array(nQ2)
        logical,  intent(in) :: l_nlo
        real(dp), dimension(2*nx,2*nx) :: id2nx
        integer :: ix, ixi, iQ2, nfl
        if(allocated(MA_SG)) deallocate(MA_SG)
        allocate(MA_SG(2*nx,2*nx,nxi,nQ2))
        ! Build an identity matrix
        id2nx = 0.0_dp
        do ix=1, 2*nx, 1
          id2nx(ix,ix) = 1.0_dp
        end do
        ! Identity for no evolution at initial scale
        MA_SG = 0.0_dp
        do ixi=1, nxi, 1
          MA_SG(:,:,ixi,1) = id2nx
        end do
        ! Build evolution matrices for all other scales
        do iQ2=2, nQ2, 1
          !$OMP PARALLEL DO
          do ixi=1, nxi, 1
            ! First, an evolution matrix for prior Q2 step to current Q2 step
            nfl = get_neff(Q2_array(iQ2-1))
            MA_SG(:,:,ixi,iQ2) = id2nx + &
                & rk4_SG(nx, nxi, Q2_array(iQ2-1), Q2_array(iQ2), &
                & KA_SG_0(:,:,ixi,nfl), KA_SG_1(:,:,ixi,nfl))
            ! Then matrix multiplication to turn into matrix from initial Q2 to current Q2
            MA_SG(:,:,ixi,iQ2) = matmul(MA_SG(:,:,ixi,iQ2), MA_SG(:,:,ixi,iQ2-1))
          end do
          !$OMP END PARALLEL DO
        end do
    end subroutine make_evomat_A_SG

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! RK4

    function rk4_NS(nx, nxi, Q2i, Q2f, K0, K1) result(K)
    !subroutine rk4_NS(nx, nxi, Q2i, Q2f, K0, K1, K)
        integer,  intent(in) :: nx, nxi
        real(dp), intent(in) :: Q2i, Q2f, K0(nx,nx), K1(nx,nx)
        !real(dp), intent(out) :: K(nx,nx)
        real(dp) :: K(nx,nx)
        !
        !real(dp), dimension(nx,nx) :: W1, W2, W3, W4, M_ini, M_mid, M_end
        real(dp), dimension(:,:), allocatable :: W1, W2, W3, W4, M_ini, M_mid, M_end
        real(dp) :: h, Q2m, a_ini, a_mid, a_end
        allocate(W1(nx,nx), W2(nx,nx), W3(nx,nx), W4(nx,nx), M_ini(nx,nx), M_mid(nx,nx), M_end(nx,nx))
        h = log(Q2f/Q2i)
        Q2m = sqrt(Q2i*Q2f) ! geometric mean for midpoint
        !
        a_ini = get_alpha_QCD(Q2i) / (2.*pi)
        a_mid = get_alpha_QCD(Q2m) / (2.*pi)
        a_end = get_alpha_QCD(Q2f) / (2.*pi)
        ! The three matrices over each subinterval
        M_ini = a_ini*K0
        M_mid = a_mid*K0
        M_end = a_end*K0
        ! Possible NLO corrections
        if(l_nlo_cache) then
          M_ini = M_ini + a_ini**2*K1
          M_mid = M_mid + a_mid**2*K1
          M_end = M_end + a_end**2*K1
        endif
        ! The RK4 k-values
        W1 = M_ini
        W2 = M_mid + 0.5*h*matmul(M_mid,W1)
        W3 = M_mid + 0.5*h*matmul(M_mid,W2)
        W4 = M_end + h*matmul(M_end,W3)
        ! And the final formula for the shift
        K = h/6.*(W1 + 2.*W2 + 2.*W3 + W4)
        deallocate(W1, W2, W3, W4, M_ini, M_mid, M_end)
    end function rk4_NS
    !end subroutine rk4_NS

    function rk4_SG(nx, nxi, Q2i, Q2f, K0, K1) result(K)
    !subroutine rk4_SG(nx, nxi, Q2i, Q2f, K0, K1, K)
        integer,  intent(in) :: nx, nxi
        real(dp), intent(in) :: Q2i, Q2f, K0(2*nx,2*nx), K1(2*nx,2*nx)
        !real(dp), intent(out) :: :: K(2*nx,2*nx)
        real(dp) :: K(2*nx,2*nx)
        !
        !real(dp), dimension(2*nx,2*nx) :: W1, W2, W3, W4, M_ini, M_mid, M_end
        real(dp), dimension(:,:), allocatable :: W1, W2, W3, W4, M_ini, M_mid, M_end
        real(dp) :: h, Q2m, a_ini, a_mid, a_end
        allocate(W1(2*nx,2*nx), W2(2*nx,2*nx), W3(2*nx,2*nx), W4(2*nx,2*nx), M_ini(2*nx,2*nx), M_mid(2*nx,2*nx), M_end(2*nx,2*nx))
        h = log(Q2f/Q2i)
        Q2m = sqrt(Q2i*Q2f) ! geometric mean for midpoint
        !
        a_ini = get_alpha_QCD(Q2i) / (2.*pi)
        a_mid = get_alpha_QCD(Q2m) / (2.*pi)
        a_end = get_alpha_QCD(Q2f) / (2.*pi)
        ! The three matrices over each subinterval
        M_ini = a_ini*K0
        M_mid = a_mid*K0
        M_end = a_end*K0
        ! Possible NLO corrections
        if(l_nlo_cache) then
          M_ini = M_ini + a_ini**2*K1
          M_mid = M_mid + a_mid**2*K1
          M_end = M_end + a_end**2*K1
        endif
        ! The RK4 k-values
        W1 = M_ini
        W2 = M_mid + 0.5*h*matmul(M_mid,W1)
        W3 = M_mid + 0.5*h*matmul(M_mid,W2)
        W4 = M_end + h*matmul(M_end,W3)
        ! And the final formula for the shift
        K = h/6.*(W1 + 2.*W2 + 2.*W3 + W4)
        deallocate(W1, W2, W3, W4, M_ini, M_mid, M_end)
    end function rk4_SG
    !end subroutine rk4_SG

    ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ! Methods to initialize some cached arrays

    subroutine initialize_2D(nx, nxi, xi_grid)
        integer, intent(in)  :: nx, nxi
        real(dp), intent(in) :: xi_grid(nxi)
        nx_cache  = nx
        nxi_cache = nxi
        if(allocated(xi_cache)) deallocate(xi_cache)
        allocate(xi_cache(nxi))
        xi_cache = xi_grid
        ! To help with singular behavior ...
        if(xi_cache(1) < 1e-6_dp) xi_cache(1) = 1e-6_dp
        if(xi_cache(nxi) > 0.999999_dp) xi_cache(nxi) = 0.999999_dp
        if(allocated(K_zero)) deallocate(K_zero)
        allocate(K_zero(nx,nx))
        K_zero = 0.0_dp
        if(allocated(K_zero_2)) deallocate(K_zero_2)
        allocate(K_zero_2(2*nx,2*nx))
        K_zero_2 = 0.0_dp
    end subroutine initialize_2D

    subroutine deallocate_all()
        if(allocated(K_NS_0))  deallocate(K_NS_0)
        if(allocated(KV_SG_0)) deallocate(KV_SG_0)
        if(allocated(M_NS_pls))   deallocate(M_NS_pls)
        if(allocated(MV_SG))   deallocate(MV_SG)
        if(allocated(KA_SG_0)) deallocate(KA_SG_0)
        if(allocated(MA_SG))   deallocate(MA_SG)
        nQ2_cache = 0
    end subroutine deallocate_all

end module matevo
