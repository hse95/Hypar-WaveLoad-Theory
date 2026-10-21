"""
First-order (small-warp) linear wave theory for a vertical wall whose face is
x = g(y, z) (|g| small), in water of depth h, inside a channel of width B with
reflecting side walls (equivalently: an infinite breakwater made of identical
panels mirrored about every panel edge). Regular waves of amplitude A arrive
normally from x -> -infinity.

Non-dimensionalisation used throughout: rho = grav = A = h = 1.
  pressure ~ rho*g*A, force ~ rho*g*A*h^2, moment ~ rho*g*A*h^3.
Time dependence exp(-i*omega*t); pressure p = i*omega*rho*phi.

Zeroth order (flat wall at x = 0): standing wave
    phi0 = Phi_A cos(kx) Z0(z),  Z0 = cosh k(z+h),  Phi_A = -2 i g A / (omega cosh kh)
    p0(wall) = 2 rho g A cosh k(z+h)/cosh kh.
First order: Laplace + free-surface + bottom + side-wall + radiation conditions, with
    phi1_x(0, y, z) = d/dz( g * phi0_z )      (phi0 independent of y)
solved by expansion in transverse modes Y_n = cos(n pi (y + B/2)/B) and
vertical modes Z_m (m = 0 propagating, m >= 1 evanescent).
"""
import numpy as np
from scipy.optimize import brentq

RHO = 1.0
GRAV = 1.0
H = 1.0


def omega_of_k(k, h=H):
    return np.sqrt(GRAV * k * np.tanh(k * h))


def evanescent_roots(k, M, h=H):
    """kappa_m, m=1..M, roots of  K = -kappa tan(kappa h)  with K = k tanh(kh)."""
    K = k * np.tanh(k * h)
    roots = []
    for m in range(1, M + 1):
        a = (m - 0.5) * np.pi / h + 1e-12
        b = m * np.pi / h - 1e-12
        f = lambda q: K + q * np.tan(q * h)
        roots.append(brentq(f, a, b, xtol=1e-14, maxiter=500))
    return np.array(roots)


def gauss_composite(breaks, npts):
    """Composite Gauss-Legendre nodes/weights over intervals defined by breakpoints."""
    xg, wg = np.polynomial.legendre.leggauss(npts)
    X, W = [], []
    for a, b in zip(breaks[:-1], breaks[1:]):
        X.append(0.5 * (b - a) * xg + 0.5 * (a + b))
        W.append(0.5 * (b - a) * wg)
    return np.concatenate(X), np.concatenate(W)


class FirstOrderSolution:
    """
    Parameters
    ----------
    kh : float          non-dimensional wavenumber (h = 1)
    B  : float          channel (panel) width / h
    gfun : callable     g(Y, Z) -> warp offset (vectorised), dimensionless (/h)
    zbreaks : list      z breakpoints in [-1, 0] where g or g_z is non-smooth
    ybreaks : list      y breakpoints in [-B/2, B/2] where g or g_y is non-smooth
    N, M : int          number of transverse (0..N) and evanescent (1..M) modes
    """

    def __init__(self, kh, B, gfun, zbreaks=(-1.0, 0.0), ybreaks=None,
                 N=80, M=80, nq=120):
        self.k = kh / H
        self.B = B
        self.omega = omega_of_k(self.k)
        self.N, self.M = N, M
        self.gfun = gfun
        k = self.k
        self.kappa = evanescent_roots(k, M)
        # quadrature grids
        if ybreaks is None:
            ybreaks = [-B / 2, 0.0, B / 2]
        self.yq, self.wy = gauss_composite(np.asarray(ybreaks, float), nq)
        self.zq, self.wz = gauss_composite(np.asarray(sorted(zbreaks), float), nq)
        # vertical modes and derivatives on z grid: shape (M+1, nz)
        zh = self.zq + H
        Z = [np.cosh(k * zh)]
        dZ = [k * np.sinh(k * zh)]
        for q in self.kappa:
            Z.append(np.cos(q * zh))
            dZ.append(-q * np.sin(q * zh))
        self.Zq = np.array(Z)
        self.dZq = np.array(dZ)
        # mode norms N_m = int Z_m^2 dz
        Nm = [0.5 * (H + np.sinh(2 * k * H) / (2 * k))]
        for q in self.kappa:
            Nm.append(0.5 * (H + np.sin(2 * q * H) / (2 * q)))
        self.Nm = np.array(Nm)
        # integrals int Z_m dz and int z Z_m dz (analytic)
        IZ = [np.sinh(k * H) / k]
        IzZ = [(-np.cosh(k * H) + 1.0) / k**2]  # int_{-h}^0 z cosh k(z+h) dz = (1 - cosh kh)/k^2
        for q in self.kappa:
            IZ.append(np.sin(q * H) / q)
            IzZ.append((np.cos(q * H) - 1.0) / q**2)  # int_{-h}^0 z cos q(z+h) dz
        self.IZ, self.IzZ = np.array(IZ), np.array(IzZ)
        # transverse modes on y grid: shape (N+1, ny)
        n = np.arange(N + 1)
        self.lam = n * np.pi / B
        self.Yq = np.cos(np.outer(self.lam, self.yq + B / 2))
        self.Ynorm = np.where(n == 0, B, B / 2)
        # int y Y_n dy (analytic):  0 for even n>=0, -2B^2/(n^2 pi^2) for odd n
        self.IyY = np.where(n % 2 == 1, -2 * B**2 / (np.maximum(n, 1)**2 * np.pi**2), 0.0)
        # base potential amplitude at the wall: phi0(0,z) = PhiA * cosh k(z+h)
        self.PhiA = -2j * GRAV / (self.omega * np.cosh(k * H))
        self._solve()

    # -------------------------------------------------------------------------
    def _solve(self):
        k = self.k
        YY, ZZ = np.meshgrid(self.yq, self.zq, indexing='ij')     # (ny, nz)
        G = self.gfun(YY, ZZ)                                       # warp on grid
        G_top = self.gfun(self.yq, np.zeros_like(self.yq))          # g(y, 0)
        phi0z = self.PhiA * k * np.sinh(k * (self.zq + H))          # phi0_z(0,z)
        phi0z_top = self.PhiA * k * np.sinh(k * H)
        # int_{-h}^0 d/dz(g phi0_z) Z_m dz = g(y,0) phi0_z(0) Z_m(0) - int g phi0_z Z_m' dz
        Ztop = self.Zq[:, -1] if np.isclose(self.zq[-1], 0) else None
        Z0vals = np.array([np.cosh(k * H)] + [np.cos(q * H) for q in self.kappa])  # Z_m(0)
        bulk = (G * phi0z[None, :] * self.wz[None, :]) @ self.dZq.T                 # (ny, M+1)
        Gym = G_top[:, None] * phi0z_top * Z0vals[None, :] - bulk                    # (ny, M+1)
        # project on transverse modes
        f = (self.Yq * self.wy[None, :]) @ Gym                                       # (N+1, M+1)
        f = f / self.Ynorm[:, None] / self.Nm[None, :]
        # beta_nm: phi1 ~ exp(beta x), outgoing / decaying as x -> -inf
        beta = np.zeros((self.N + 1, self.M + 1), complex)
        for i, lam in enumerate(self.lam):
            if k > lam:
                beta[i, 0] = -1j * np.sqrt(k**2 - lam**2)
            else:
                beta[i, 0] = np.sqrt(lam**2 - k**2)
            beta[i, 1:] = np.sqrt(self.kappa**2 + lam**2)
        self.f, self.beta = f, beta
        self.C = f / beta            # phi1 = sum C_nm Y_n Z_m exp(beta x)
        # store tilted-normal ingredients
        self.G = G
        self.p0z = 2 * RHO * GRAV * np.cosh(k * (self.zq + H)) / np.cosh(k * H)

    # -------------------------------------------------------------------------
    def p1(self, y, z):
        """First-order wall pressure p1(y,z) (complex amplitude)."""
        y = np.atleast_1d(y); z = np.atleast_1d(z)
        Yn = np.cos(np.outer(self.lam, y + self.B / 2))            # (N+1, ny)
        zh = z + H
        Zm = np.vstack([np.cosh(self.k * zh)] + [np.cos(q * zh) for q in self.kappa])
        phi1 = Yn.T @ self.C @ Zm                                   # (ny, nz)
        return 1j * self.omega * RHO * phi1

    def p0(self, z):
        return 2 * RHO * GRAV * np.cosh(self.k * (z + H)) / np.cosh(self.k * H)

    def loads(self, zref=0.0, dgdy=None, dgdz=None):
        """
        First-order load increments (complex amplitudes) per panel of width B.
        Tilted-normal parts use numerical derivatives of g on the quadrature grid
        unless analytic derivative functions are supplied.
        Moments are about the point (0, 0, zref).
        """
        k, B = self.k, self.B
        iw = 1j * self.omega * RHO
        # --- scattered-field (p1) contributions ---
        Fx1 = iw * np.sum(self.C[0, :] * B * self.IZ)
        My1 = iw * np.sum(self.C[0, :] * B * (self.IzZ - zref * self.IZ))
        Mz1 = -iw * np.sum((self.IyY[:, None] * self.C) * self.IZ[None, :])
        # --- tilted-normal contributions with p0 ---
        YY, ZZ = np.meshgrid(self.yq, self.zq, indexing='ij')
        if dgdy is None or dgdz is None:
            d = 1e-6
            gy = (self.gfun(YY + d, ZZ) - self.gfun(YY - d, ZZ)) / (2 * d)
            gz = (self.gfun(YY, ZZ + d) - self.gfun(YY, ZZ - d)) / (2 * d)
        else:
            gy, gz = dgdy(YY, ZZ), dgdz(YY, ZZ)
        W = np.outer(self.wy, self.wz)
        P0 = self.p0z[None, :]
        Fy1 = -np.sum(W * P0 * gy)
        Fz1 = -np.sum(W * P0 * gz)
        Mx1 = np.sum(W * P0 * ((ZZ - zref) * gy - YY * gz))
        return dict(Fx=Fx1, Fy=Fy1, Fz=Fz1, Mx=Mx1, My=My1, Mz=Mz1)

    def base_loads(self, zref=0.0):
        k, B = self.k, self.B
        P0 = 2 * np.tanh(k * H) / k                                # int p0 dz
        zp = -np.tanh(k * H / 2) / k
        return dict(Fx=B * P0, My=B * P0 * (zp - zref), P0=P0, zp=zp)


# -----------------------------------------------------------------------------
# Closed-form results (twisted, "straight-edged" hypar  g = c y (z - zc), full depth)
# -----------------------------------------------------------------------------
def closed_form_twist(kh, B, c, zc, zref=None):
    """Lateral force and roll moment per panel, non-dimensional (h=1)."""
    k = kh
    if zref is None:
        zref = zc
    P0 = 2 * np.tanh(k) / k
    zp = -np.tanh(k / 2) / k
    Fy = c * B * P0 * (zc - zp)
    # int p0 (z - zref)(z - zc) dz  with p0 = 2 cosh k(z+1)/cosh k
    # use int cosh k(z+h) (z-a)(z-b) dz = I2 - (a+b) I1 + ab I0
    I0 = np.sinh(k) / k
    I1 = (1 - np.cosh(k)) / k**2
    I2 = (2 * np.sinh(k) - 2 * k) / k**3 + 0.0  # int z^2 cosh k(z+1) dz from -1..0
    # (from sympy: (k^2 a^2 sinh - 2k(h - a cosh + a) + 2 sinh)/k^3 with a=0 -> (2 sinh k - 2k)/k^3)
    Iab = I2 - (zref + zc) * I1 + zref * zc * I0
    Mx = c * (B * 2 * Iab / np.cosh(k) - B**3 / 12 * P0)
    return dict(Fy=Fy, Mx=Mx, P0=P0, zp=zp)
