! Same GG expressions, evaluated with portable four-component arithmetic.
! No native REAL128 requirement. No coefficients/support factors changed.
! This is a finite-domain numerical guard, independently checked against mpmath.
module gg_wide
  use software_wide
  implicit none
  private
  type(wide_real),parameter::pi=wide_real(3.141592653589793116e+00_dp, 1.224646799147353207e-16_dp, -2.994769809718339666e-33_dp, 1.112454220863365282e-49_dp)
  type(wide_real),parameter::zeta2=wide_real(1.644934066848226406e+00_dp, 3.040672350398476161e-17_dp, -2.000604926952525193e-33_dp, 5.808625690353289816e-51_dp)
  type(wide_real),parameter::CF=wide_real(1.333333333333333259e+00_dp, 7.401486830834376525e-17_dp, 4.108650548026102925e-33_dp, 2.280759219278673491e-49_dp)
  type(wide_real),parameter::CA=wide_real(3._dp,0._dp),TF=wide_real(.5_dp,0._dp)
  public::gg_guard_value
contains
  function gg_guard_value(x,y,xi,nfl_part) result(v)
    real(dp),intent(in)::x,y,xi
    integer,intent(in)::nfl_part
    real(dp)::v
    type(wide_real)::a,b,c,X1,Y1,X2,Y2
    a=wide(x);b=wide(y);c=wide(xi)
    X1=(wide(1._dp)+a/c)/wide(2._dp);X2=(wide(1._dp)-a/c)/wide(2._dp)
    Y1=(wide(1._dp)+b/c)/wide(2._dp);Y2=(wide(1._dp)-b/c)/wide(2._dp)
    if(nfl_part==0)then
      v=to_double((KV1_GG_half(X1,Y1)+KV1_GG_half(X2,Y2))/(wide(2._dp)*c))
    else
      v=to_double((KV1_GG_half_nfl(X1,Y1)+KV1_GG_half_nfl(X2,Y2))/(wide(2._dp)*c))
    endif
  end function
  recursive function dilog(z) result(v)
    type(wide_real),intent(in)::z
    type(wide_real)::v,term,add
    integer::k
    if(z==-wide(1._dp))then
      v=-pi*pi/wide(12._dp)
    elseif(z==wide(0._dp))then
      v=wide(0._dp)
    elseif(z==wide(1._dp))then
      v=pi*pi/wide(6._dp)
    elseif(z<-wide(1._dp))then
      v=log(wide(1._dp)-z)*(wide(0.5_dp)*log(wide(1._dp)-z)-log(-z))-pi*pi/wide(6._dp)+dilog(wide(1._dp)/(wide(1._dp)-z))
    elseif(z<wide(0._dp))then
      v=-dilog(z/(z-wide(1._dp)))-wide(0.5_dp)*log(wide(1._dp)-z)**2
    elseif(z>wide(2._dp))then
      v=-dilog(wide(1._dp)/z)+pi*pi/wide(3._dp)-wide(0.5_dp)*log(z)**2
    elseif(z>wide(1._dp))then
      v=pi*pi/wide(6._dp)-log(z)*(log(wide(1._dp)-wide(1._dp)/z)+wide(0.5_dp)*log(z))+dilog(wide(1._dp)-wide(1._dp)/z)
    elseif(z>wide(0.5_dp))then
      v=-dilog(wide(1._dp)-z)+pi*pi/wide(6._dp)-log(z)*log(wide(1._dp)-z)
    else
      v=wide(0._dp);term=z
      do k=1,256
        add=term/wide(k)**2;v=v+add
        if(abs(add)<=epsilon(v)*max(abs(v),tiny(v)))return
        term=term*z
      enddo
      error stop 'wide dilogarithm series failed'
    endif
  end function
    function theta_step(X) result(theta)
        type(wide_real), intent(in) :: X
        type(wide_real) :: theta
        theta = wide(0.0_dp)
        if(X > wide(0.0_dp)) then
          theta = wide(1.0_dp)
        elseif(X==wide(0.0_dp)) then
          theta = wide(0.5_dp)
        endif
    end function theta_step

    function rho_step(X, Y) result(rho)
        type(wide_real), intent(in) :: X, Y
        type(wide_real) :: rho
        !
        rho = theta_step(wide(1._dp)-X/Y)*theta_step(X/Y)*sign(wide(1.0_dp),Y)
    end function rho_step

    function erbl_step(X) result(theta)
        ! Check that we're in the ERBL region
        type(wide_real), intent(in) :: X
        type(wide_real) :: theta
        !
        theta = theta_step(X)*theta_step(wide(1.0_dp)-X)
    end function erbl_step

    function abslog(x) result(y)
        ! log of absolute value
        ! Convenient for removing imaginary parts
        type(wide_real), intent(in) :: x
        type(wide_real) :: y
        !
        y = log(abs(x))
    end function abslog

    function log2(x) result(y)
        ! Square of log
        ! Removes imaginary part, accounts for -pi**2 in real part
        type(wide_real), intent(in) :: x
        type(wide_real) :: y
        !
        y = log(abs(x))**2
        if(x < wide(0.0_dp)) y = y - pi**2
    end function log2

    function logprod(x1, x2) result(y)
        ! Product of two logs
        ! Removes imaginary part, accounts for -pi**2 in real part
        type(wide_real), intent(in) :: x1, x2
        type(wide_real) :: y
        !
        y = log(abs(x1)) * log(abs(x2))
        if(x1 < wide(0.0_dp) .and. x2 < wide(0.0_dp)) y = y - pi**2
    end function logprod

    function GG_f_a(X,Y) result(f)
        ! Eq. (76)
        type(wide_real), intent(in) :: X, Y
        type(wide_real) :: f
        !
        f = X**2/Y**2
    end function GG_f_a
    function GG_f_b(X,Y) result(f)
        ! Eq. (76)
        type(wide_real), intent(in) :: X, Y
        type(wide_real) :: f
        !
        f = X**2/Y**2 / (Y-X)
    end function GG_f_b
    function GG_f_c(X,Y) result(f)
        ! Eq. (77)
        type(wide_real), intent(in) :: X, Y
        type(wide_real) :: f
        !
        type(wide_real) :: Xbar
        Xbar = wide(1.0_dp) - X
        f = X**2/Y**2 * (wide(2.0_dp)*Xbar*Y + Y - X)
    end function GG_f_c
    function GG_h_V(X,Y) result(h)
        ! Eq. (174)
        type(wide_real), intent(in) :: X, Y
        type(wide_real) :: h
        !
        type(wide_real) :: Xbar, Ybar, GGf, GGfbar
        Xbar = wide(1.0_dp) - X
        Ybar = wide(1.0_dp) - Y
        ! Eqs. (74) and (75)
        GGf    = wide(2._dp)*GG_f_a(X   ,Y   ) + GG_f_b(X   ,Y   ) + wide(2._dp)*GG_f_c(X   ,Y   )
        GGfbar = wide(2._dp)*GG_f_a(Xbar,Ybar) + GG_f_b(Xbar,Ybar) + wide(2._dp)*GG_f_c(Xbar,Ybar)
        h = wide(2._dp)*GGfbar*logprod(Xbar,Y) - wide(2._dp)*GGf*(dilog(X) + dilog(Ybar))
    end function GG_h_V
    function GG_hbar_V(X,Y) result(h)
        ! Eq. (174)
        type(wide_real), intent(in) :: X, Y
        type(wide_real) :: h
        !
        type(wide_real) :: Xbar, Ybar, GGf, GGfbar
        Xbar = wide(1.0_dp) - X
        Ybar = wide(1.0_dp) - Y
        GGf    = wide(2._dp)*GG_f_a(X   ,Y   ) + GG_f_b(X   ,Y   ) + wide(2._dp)*GG_f_c(X   ,Y   )
        GGfbar = wide(2._dp)*GG_f_a(Xbar,Ybar) + GG_f_b(Xbar,Ybar) + wide(2._dp)*GG_f_c(Xbar,Ybar)
        h = (-GGfbar + GGf)*(wide(2._dp)*dilog(wide(1._dp)-X/Y) + log2(Y)) &
            & + wide(2._dp)*GGf*(dilog(Ybar) - logprod(X,Y)) &
            & + wide(2._dp)*GGfbar*dilog(Xbar)
    end function GG_hbar_V
    function KV1_GG_half(X, Y) result(K)
        ! Eq. (185)
        type(wide_real), intent(in) :: X, Y
        type(wide_real) :: K
        !
        type(wide_real) :: piece1, piece2, piece3, piece4, piece5
        type(wide_real) :: Xbar, Ybar, beta0
        type(wide_real) :: GGfV, GGfbarV, GGfc, GGhV, GGhbarV
        Xbar = wide(1.0_dp) - X
        Ybar = wide(1.0_dp) - Y
        K = wide(0.0_dp)
        beta0 = -wide(11._dp)/wide(3._dp)*CA
        ! Terms that contribute
        GGfV    = wide(2._dp)*GG_f_a(X   ,Y   ) + GG_f_b(X   ,Y   ) + wide(2._dp)*GG_f_C(X   ,Y   )
        GGfbarV = wide(2._dp)*GG_f_a(Xbar,Ybar) + GG_f_b(Xbar,Ybar) + wide(2._dp)*GG_f_C(Xbar,Ybar)
        GGfc    = GG_f_c(X,Y)
        GGhV    = GG_h_V(X,Y)
        GGhbarV = GG_hbar_V(Xbar,Y)
        ! pieces 1, 2, 3 have support in both regions
        piece1 = CA**2*( &
            & (wide(2._dp)/wide(3._dp)-wide(2._dp)*zeta2)*GGfV - wide(0.5_dp)*GGfc + X*(wide(4._dp)+wide(5._dp)*X)/(wide(4._dp)*Y**2) &
            & - X*(wide(10._dp)-wide(19._dp)*X+wide(16._dp)*X**2)/Y + X*(wide(6._dp)-wide(13._dp)*X+wide(8._dp)*X**2)/Ybar &
            & + (wide(2._dp)*X*(wide(1._dp)-wide(2._dp)*Xbar*(X*Y+Xbar*Ybar))/(Y*Ybar) + X*(wide(2._dp)-wide(3._dp)*X-wide(8._dp)*Xbar**2)/Ybar**2)*abslog(X/Y) &
            & + (GGfV + wide(2._dp)*X**2/Ybar**2)*log2(X/Y) &
            & - (GGfV-GGfbarV)*logprod(X/Y,wide(1._dp)-X/Y) &
            & -wide(0.5_dp)*GGhV - wide(0.5_dp)*GGhbarV &
            & )
        piece2 = wide(0.5_dp)*CA*beta0*( &
            & - wide(5._dp)/wide(3._dp)*GGfV - wide(13._dp)/wide(3._dp)*GGfc - wide(11._dp)/wide(2._dp)*X**2/Y**2 + X*(X+Ybar)/(Y**2*Ybar) &
            & + X**2/Ybar**2*abslog(Y) + Xbar**2/Y**2*abslog(Xbar) &
            & )
        ! pieces 4, 5 have support only in the ERBL region
        piece4 = CA**2*( &
            & - wide(2._dp)*X**2/Ybar**2*log2(X) &
            & + (Xbar*(X**2+Xbar**2-wide(2._dp)*Xbar*(wide(1._dp)+wide(2._dp)*X)*Ybar) + Ybar)/Ybar**2*logprod(X,Xbar) &
            & + ((wide(3._dp)*X*(X**2+Xbar**2)*(Ybar-Y)-wide(4._dp)*X**2*Y)/(Y*Ybar) - X*(wide(2._dp)-wide(3._dp)*X-wide(8._dp)*Xbar**2)/Ybar**2)*abslog(X) &
            & )
        ! Factor in the support regions
        piece1 = piece1*rho_step(X,Y)
        piece2 = piece2*rho_step(X,Y)
        piece3 = wide(0.0_dp) ! linear in nfl
        piece4 = piece4*erbl_step(X)
        piece5 = wide(0.0_dp) ! linear in nfl
        K = piece1 + piece2 + piece3 + piece4 + piece5
    end function KV1_GG_half
    function KV1_GG_half_nfl(X, Y) result(K)
        ! Eq. (185)
        type(wide_real), intent(in) :: X, Y
        type(wide_real) :: K
        !
        type(wide_real) :: piece1, piece2, piece3, piece4, piece5
        type(wide_real) :: Xbar, Ybar, beta0
        type(wide_real) :: GGfV, GGfbarV, GGfc!, GGhV, GGhbarV
        Xbar = wide(1.0_dp) - X
        Ybar = wide(1.0_dp) - Y
        K = wide(0.0_dp)
        beta0 = wide(4._dp)/wide(3._dp)*TF
        ! Terms that contribute
        GGfV    = wide(2._dp)*GG_f_a(X   ,Y   ) + GG_f_b(X   ,Y   ) + wide(2._dp)*GG_f_C(X   ,Y   )
        GGfbarV = wide(2._dp)*GG_f_a(Xbar,Ybar) + GG_f_b(Xbar,Ybar) + wide(2._dp)*GG_f_C(Xbar,Ybar)
        GGfc    = GG_f_c(X,Y)
        ! pieces 1, 2, 3 have support in both regions
        piece2 = wide(0.5_dp)*CA*beta0*( &
            & - wide(5._dp)/wide(3._dp)*GGfV - wide(13._dp)/wide(3._dp)*GGfc - wide(11._dp)/wide(2._dp)*X**2/Y**2 + X*(X+Ybar)/(Y**2*Ybar) &
            & + X**2/Ybar**2*abslog(Y) + Xbar**2/Y**2*abslog(Xbar) &
            & )
        piece3 = CF*TF*( &
            & -wide(20._dp)/wide(3._dp)*GGfc - wide(12._dp)*X**2/Y**2 + wide(2._dp)*X*(wide(2._dp)+Y+wide(3._dp)*X*Y-wide(4._dp)*Y**2)/(Y**2*Ybar) &
            & - wide(2._dp)*X*(X-Y+wide(2._dp)*X*Y)/(Y*Ybar**2)*abslog(X/Y) - X**2/Ybar**2*log2(X/Y) &
            & )
        ! pieces 4, 5 have support only in the ERBL region
        piece5 = CF*TF*( &
            & -wide(2._dp)*X*(wide(1._dp)-wide(3._dp)*X-wide(2._dp)*X*Ybar)/Ybar**2*abslog(X) + X**2/Ybar**2*log2(X) &
            & )
        ! Factor in the support regions
        piece1 = wide(0.0_dp) ! nfl-independent
        piece2 = piece2*rho_step(X,Y)
        piece3 = piece3*rho_step(X,Y)
        piece4 = wide(0.0_dp) ! nfl-independent
        piece5 = piece5*erbl_step(X)
        K = piece1 + piece2 + piece3 + piece4 + piece5
    end function KV1_GG_half_nfl
end module gg_wide
