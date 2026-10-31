! Same QG expressions, evaluated with portable four-component arithmetic.
! No native REAL128 requirement. No coefficients/support factors changed.
! This is a finite-domain numerical guard, independently checked against mpmath.
module qg_wide
  use software_wide
  implicit none
  private
  type(wide_real),parameter::pi=wide_real(3.141592653589793116e+00_dp, 1.224646799147353207e-16_dp, -2.994769809718339666e-33_dp, 1.112454220863365282e-49_dp)
  type(wide_real),parameter::zeta2=wide_real(1.644934066848226406e+00_dp, 3.040672350398476161e-17_dp, -2.000604926952525193e-33_dp, 5.808625690353289816e-51_dp)
  type(wide_real),parameter::CF=wide_real(1.333333333333333259e+00_dp, 7.401486830834376525e-17_dp, 4.108650548026102925e-33_dp, 2.280759219278673491e-49_dp)
  type(wide_real),parameter::CA=wide_real(3._dp,0._dp),TF=wide_real(.5_dp,0._dp)
  public::qg_guard_value
contains
  function qg_guard_value(x,y,xi) result(v)
    real(dp),intent(in)::x,y,xi
    real(dp)::v
    type(wide_real)::a,b,c,X1,Y1,X2,Y2
    a=wide(x);b=wide(y);c=wide(xi)
    X1=(wide(1._dp)+a/c)/wide(2._dp);X2=(wide(1._dp)-a/c)/wide(2._dp)
    Y1=(wide(1._dp)+b/c)/wide(2._dp);Y2=(wide(1._dp)-b/c)/wide(2._dp)
    v=to_double((KV1_qG_half(X1,Y1)-KV1_qG_half(X2,Y2))/(wide(4._dp)*c*c))
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

    function QG_f_a(X,Y) result(f)
        ! Eq. (76)
        type(wide_real), intent(in) :: X, Y
        type(wide_real) :: f
        !
        f = X/Y**2
    end function QG_f_a

    function QG_f_c(X,Y) result(f)
        ! Eq. (78)
        type(wide_real), intent(in) :: X, Y
        type(wide_real) :: f
        !
        type(wide_real) :: Xbar
        Xbar = wide(1.0_dp) - X
        f = X/Y**2 * (wide(2.0_dp)*Xbar*Y - X)
    end function QG_f_c

    function QG_h_V(X,Y) result(h)
        ! Eq. (174)
        type(wide_real), intent(in) :: X, Y
        type(wide_real) :: h
        !
        type(wide_real) :: Xbar, Ybar, QGfV, QGfVbar
        Xbar = wide(1.0_dp) - X
        Ybar = wide(1.0_dp) - Y
        ! Eqs. (74), (75)
        QGfV    = -QG_f_a(X   ,Y)    - wide(2._dp)*QG_f_c(X   ,Y   )
        QGfVbar = -QG_f_a(Xbar,Ybar) - wide(2._dp)*QG_f_c(Xbar,Ybar)
        h = -wide(2._dp)*QGfVbar*logprod(Xbar,Y) - wide(2._dp)*QGfV*(dilog(X) + dilog(Ybar))
    end function QG_h_V

    function QG_hbar_V(X,Y) result(h)
        ! Eq. (174)
        type(wide_real), intent(in) :: X, Y
        type(wide_real) :: h
        !
        type(wide_real) :: Xbar, Ybar, QGfV, QGfVbar
        Xbar = wide(1.0_dp) - X
        Ybar = wide(1.0_dp) - Y
        ! Eqs. (74), (75)
        QGfV    = -QG_f_a(X   ,Y)    - wide(2._dp)*QG_f_c(X   ,Y   )
        QGfVbar = -QG_f_a(Xbar,Ybar) - wide(2._dp)*QG_f_c(Xbar,Ybar)
        h = (QGfVbar + QGfV)*(wide(2._dp)*dilog(wide(1._dp)-X/Y) + log2(Y)) &
            & + wide(2._dp)*QGfV*(dilog(Ybar) - logprod(X,Y)) &
            & - wide(2._dp)*QGfVbar*dilog(Xbar)
    end function QG_hbar_V

    function KV1_qG_half(X, Y) result(K)
        ! Eq. (183)
        type(wide_real), intent(in) :: X, Y
        type(wide_real) :: K
        !
        type(wide_real) :: piece1, piece2, piece3, piece4
        type(wide_real) :: QGfV, QGfA, QGfbarV, QGfbarA, QGhV, QGhbarV
        type(wide_real) :: Xbar, Ybar
        K = wide(0.0_dp)
        Xbar = wide(1.0_dp) - X
        Ybar = wide(1.0_dp) - Y
        ! Terms that contribute
        QGfV    = -QG_f_a(X,Y) - wide(2._dp)*QG_f_c(X,Y)
        QGfA    = -QG_f_a(X,Y)
        QGfbarV = -QG_f_a(Xbar,Ybar) - wide(2._dp)*QG_f_c(Xbar,Ybar)
        QGfbarA = -QG_f_a(Xbar,Ybar)
        QGhV    = QG_h_V(X,Y)
        QGhbarV = QG_hbar_V(Xbar,Y)
        ! piece 1 and piece 2 have support in both regions
        piece1 = TF*CF*( &
            & wide(2._dp)*(wide(5._dp)-wide(2._dp)*zeta2)*QGfV - wide(4._dp)*QGfA + X/(Y*Ybar) &
            & + wide(2._dp)*(QGfV + QGfbarV + (wide(2._dp)*X-wide(2._dp)*Y+X*Y)/(wide(2._dp)*Y*Ybar**2))*abslog(X/Y) &
            & - wide(2._dp)*(QGfV + QGfbarV + wide(1._dp)/(Y*Ybar))*abslog(wide(1._dp)-X/Y) &
            & + QGfbarA*log2(X/Y) + (QGfV+QGfbarV)*log2(Y/X-wide(1._dp)) &
            & )
        piece2 = TF*CA*( &
            & wide(6._dp)*X*(wide(3._dp)-wide(4._dp)*X)/Ybar - wide(2._dp)*X*(wide(11._dp)-wide(16._dp)*X)/Y + wide(4._dp)*X*(wide(1._dp)-wide(3._dp)*X)/Y**2 &
            & - wide(2._dp)*(QGfV - wide(5._dp)*QGfbarV + (wide(5._dp)-wide(7._dp)*X)/Ybar**2 + wide(2._dp)*X*(wide(3._dp)-wide(2._dp)*X-wide(12._dp)*Xbar*Y)/(Y*Ybar))*abslog(X/Y) &
            & + wide(2._dp)*(QGfV + QGfbarV + wide(1._dp)/(Y*Ybar))*abslog(wide(1._dp)-X/Y) &
            & + (QGfV - QGfbarV + (wide(1._dp)-wide(4._dp)*X)/Ybar**2)*log2(X/Y) &
            & - (QGfV + QGfbarV)*abslog(wide(1._dp)-X/Y)**2 - QGhV + QGhbarV &
            & )
        ! piece 3 and piece 4 have support in ERBL region only
        piece3 = TF*CF*( &
            & - wide(2._dp)*(QGfV + QGfbarV + QGfbarA + X*(Y+wide(4._dp)*Ybar**2)/(wide(2._dp)*Y*Ybar**2))*abslog(X) &
            & - wide(2._dp)*qGfV*logprod(Xbar,X) &
            & + (QGfV - QGfbarV - QGfbarA)*log2(X) &
            & )
        piece4 = TF*CA*( &
            & wide(2._dp)*(QGfV - wide(5._dp)*QGfbarV + (wide(5._dp)-wide(7._dp)*X)/Ybar**2 + X*(wide(5._dp)-wide(6._dp)*X-wide(22._dp)*Y+wide(28._dp)*X*Y)/(Y*Ybar))*abslog(X) &
            & - (QGfV - QGfbarV + (wide(1._dp)-wide(4._dp)*X)/Ybar**2)*log2(X) &
            & )
        ! Factor in the support regions
        piece1 = piece1*rho_step(X,Y)
        piece2 = piece2*rho_step(X,Y)
        piece3 = piece3*erbl_step(X)
        piece4 = piece4*erbl_step(X)
        K = piece1 + piece2 + piece3 + piece4
    end function KV1_qG_half
end module qg_wide
