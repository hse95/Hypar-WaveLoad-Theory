"""
Sanity tests against exact solutions.

    python -m src.tests

1) Theory: a uniformly displaced wall (f = d) must give p1 = 2 rho g A cosh k(z+h)/cosh kh * (i k d).
2) Theory: modal-quadrature lateral force and roll moment must equal the closed forms (Eqs. 15-16).
3) FE: flat wall must reproduce the standing-wave pressure; displaced wall must give p0*exp(i k d).
"""
import numpy as np
from .theory import FirstOrderSolution, closed_form_twist
from .fem import WarpedWallFEM
from .geometry import twisted


def main():
    kh, B = 1.0, 2.0
    z = np.linspace(-1, 0, 11)
    # 1) uniform shift, first-order theory
    s = FirstOrderSolution(kh, B, lambda Y, Z: 1.0 + 0 * Y, N=10, M=60)
    exact = 2 * np.cosh(kh * (z + 1)) / np.cosh(kh) * 1j * kh
    print(f"[theory] uniform shift, max |p1 - exact|      = {np.max(np.abs(s.p1(np.array([0.3]), z)[0] - exact)):.2e}")
    # 2) closed forms vs modal quadrature (twisted panel)
    L = FirstOrderSolution(kh, B, twisted(0.1), N=60, M=60).loads(zref=-0.5)
    cf = closed_form_twist(kh, B, 0.1, -0.5)
    print(f"[theory] Fy closed form vs quadrature, rel. diff = {abs(L['Fy'] - cf['Fy']) / abs(cf['Fy']):.2e}")
    print(f"[theory] Mx closed form vs quadrature, rel. diff = {abs(L['Mx'] - cf['Mx']) / abs(cf['Mx']):.2e}")
    # 3) FE exact solutions
    for d in [0.0, 0.2]:
        fe = WarpedWallFEM(kh, B, lambda Y, Z: d + 0 * Y, nx=64, ny=12, nz=16).solve()
        Yw, Zw, Pw = fe.wall_pressure()
        ex = 2 * np.cosh(kh * (Zw + 1)) / np.cosh(kh) * np.exp(1j * kh * d)
        print(f"[FE]     wall displaced by d = {d:.1f}h, max rel. pressure error = {np.max(np.abs(Pw - ex)) / np.max(np.abs(ex)):.2%}")


if __name__ == "__main__":
    main()
