"""
Independent verification solver: EXACT linear potential-flow problem (the warp is
NOT linearised) for a vertical wall with face x = g(y,z) at the end of a channel
of width B (reflecting side walls), depth h, regular waves incident from -x.

Discretisation: isoparametric trilinear (Q1) hexahedral finite elements on a
structured mesh mapped so that the wall face x~ = 0 coincides with x = g(y,z):
    x = x~ + g(y,z) * w(x~),   w(x~) = (1 + x~/L)^2,   -L <= x~ <= 0,
so the mapping vanishes (with zero slope) on the truncation boundary x = -L.
Radiation condition: exact modal Dirichlet-to-Neumann (DtN) operator on x = -L
(transverse cosine modes x vertical eigenmodes), with the incident wave imposed.

Weak form:  int grad(phi).grad(v) dV - K int_FS phi v dS + int_Gamma phi_x v dS = 0
Non-dimensional: rho = g = A = h = 1.
"""
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from .theory import evanescent_roots, omega_of_k

# ---------- reference Q1 element data ----------
_LOC = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                 [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
_S = 2 * _LOC - 1                                      # node signs in [-1,1]^3
_g = 1 / np.sqrt(3)
_GP3 = np.array([[a, b, c] for a in (-_g, _g) for b in (-_g, _g) for c in (-_g, _g)])
_GP2 = np.array([[a, b] for a in (-_g, _g) for b in (-_g, _g)])


def _dN3(xi):
    """Derivatives of the 8 trilinear shape functions at point xi, shape (8,3)."""
    s = _S
    d = np.zeros((8, 3))
    d[:, 0] = s[:, 0] * (1 + s[:, 1] * xi[1]) * (1 + s[:, 2] * xi[2]) / 8
    d[:, 1] = s[:, 1] * (1 + s[:, 0] * xi[0]) * (1 + s[:, 2] * xi[2]) / 8
    d[:, 2] = s[:, 2] * (1 + s[:, 0] * xi[0]) * (1 + s[:, 1] * xi[1]) / 8
    return d


_Q2 = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])


def _N2(xi):
    return (1 + _Q2[:, 0] * xi[0]) * (1 + _Q2[:, 1] * xi[1]) / 4


def _dN2(xi):
    d = np.zeros((4, 2))
    d[:, 0] = _Q2[:, 0] * (1 + _Q2[:, 1] * xi[1]) / 4
    d[:, 1] = _Q2[:, 1] * (1 + _Q2[:, 0] * xi[0]) / 4
    return d


class WarpedWallFEM:
    def __init__(self, kh, B, gfun, nx=64, ny=24, nz=16, L=None,
                 n_modes_y=None, n_modes_z=None, zgrade=1.0, xgrade=1.0):
        self.k = kh
        self.omega = omega_of_k(kh)
        self.K = self.omega**2
        self.B = B
        lam = 2 * np.pi / kh
        self.L = L if L is not None else lam
        self.gfun = gfun
        self.nx, self.ny, self.nz = nx, ny, nz
        sx = np.linspace(0, 1, nx + 1)
        xt = -self.L * (1 - sx)**xgrade      # xgrade>1 clusters nodes at the wall
        yv = np.linspace(-B / 2, B / 2, ny + 1)
        s = np.linspace(0, 1, nz + 1)
        zv = -1 + (1 - (1 - s)**zgrade) if zgrade != 1.0 else -1 + s  # optional grading to surface
        zv[0], zv[-1] = -1.0, 0.0
        self.xt, self.yv, self.zv = xt, yv, zv
        XT, YY, ZZ = np.meshgrid(xt, yv, zv, indexing='ij')
        wblend = (1 + XT / self.L)**2
        self.X = XT + gfun(YY, ZZ) * wblend
        self.Y, self.Z = YY, ZZ
        self.nn = (nx + 1) * (ny + 1) * (nz + 1)
        self.nmy = n_modes_y if n_modes_y is not None else ny // 2
        self.nmz = n_modes_z if n_modes_z is not None else nz // 2
        self._assemble()

    def nid(self, i, j, k):
        return (i * (self.ny + 1) + j) * (self.nz + 1) + k

    def _assemble(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        I, J, Kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
        I, J, Kk = I.ravel(), J.ravel(), Kk.ravel()
        conn = np.stack([self.nid(I + a, J + b, Kk + c) for a, b, c in _LOC], axis=1)  # (E,8)
        coords = np.stack([self.X.ravel(), self.Y.ravel(), self.Z.ravel()], axis=1)
        xe = coords[conn]                                                               # (E,8,3)
        Ke = np.zeros((conn.shape[0], 8, 8))
        for gp in _GP3:
            dN = _dN3(gp)                                   # (8,3)
            Jm = np.einsum('eai,aj->eij', xe, dN)           # J_ij = dx_i/dxi_j
            detJ = np.linalg.det(Jm)
            Jinv = np.linalg.inv(Jm)
            G = np.einsum('aj,eji->eai', dN, Jinv)           # dN/dx  (E,8,3)
            Ke += np.einsum('eai,ebi->eab', G, G) * detJ[:, None, None]
        rows = np.repeat(conn, 8, axis=1).ravel()
        cols = np.tile(conn, (1, 8)).ravel()
        self.Ks = sps.coo_matrix((Ke.ravel(), (rows, cols)), shape=(self.nn, self.nn)).tocsr()

        # free-surface mass (k = nz), quads in physical (x, y)
        I2, J2 = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        I2, J2 = I2.ravel(), J2.ravel()
        q = [(0, 0), (1, 0), (1, 1), (0, 1)]
        cfs = np.stack([self.nid(I2 + a, J2 + b, nz) for a, b in q], axis=1)
        xy = np.stack([self.X.ravel()[cfs], self.Y.ravel()[cfs]], axis=2)                # (F,4,2)
        Me = np.zeros((cfs.shape[0], 4, 4))
        for gp in _GP2:
            N = _N2(gp); dN = _dN2(gp)
            Jm = np.einsum('fai,aj->fij', xy, dN)
            detJ = np.abs(np.linalg.det(Jm))
            Me += np.einsum('a,b->ab', N, N)[None] * detJ[:, None, None]
        rows = np.repeat(cfs, 4, axis=1).ravel(); cols = np.tile(cfs, (1, 4)).ravel()
        self.Mfs = sps.coo_matrix((Me.ravel(), (rows, cols)), shape=(self.nn, self.nn)).tocsr()

        # truncation boundary Gamma (i = 0): quads in (y, z), plane x = -L
        Jg, Kg = np.meshgrid(np.arange(ny), np.arange(nz), indexing='ij')
        Jg, Kg = Jg.ravel(), Kg.ravel()
        cg = np.stack([self.nid(0, Jg + a, Kg + b) for a, b in q], axis=1)
        yz = np.stack([self.Y.ravel()[cg], self.Z.ravel()[cg]], axis=2)
        Mg = np.zeros((cg.shape[0], 4, 4))
        for gp in _GP2:
            N = _N2(gp); dN = _dN2(gp)
            Jm = np.einsum('fai,aj->fij', yz, dN)
            detJ = np.abs(np.linalg.det(Jm))
            Mg += np.einsum('a,b->ab', N, N)[None] * detJ[:, None, None]
        # local numbering of Gamma nodes
        gnodes = np.array([self.nid(0, j, kk) for j in range(ny + 1) for kk in range(nz + 1)])
        loc = -np.ones(self.nn, int); loc[gnodes] = np.arange(len(gnodes))
        rows = np.repeat(loc[cg], 4, axis=1).ravel(); cols = np.tile(loc[cg], (1, 4)).ravel()
        MG = sps.coo_matrix((Mg.ravel(), (rows, cols)), shape=(len(gnodes),) * 2).toarray()
        self.gnodes, self.MG = gnodes, MG
        yg = self.Y.ravel()[gnodes]; zg = self.Z.ravel()[gnodes]
        # modal DtN on Gamma
        k = self.k
        kap = evanescent_roots(k, self.nmz)
        Psi, beta = [], []
        for n in range(self.nmy + 1):
            lam = n * np.pi / self.B
            Yn = np.cos(lam * (yg + self.B / 2))
            for m in range(self.nmz + 1):
                if m == 0:
                    Zm = np.cosh(k * (zg + 1)) / np.cosh(k)
                    b = -1j * np.sqrt(k**2 - lam**2 + 0j) if k > lam else np.sqrt(lam**2 - k**2)
                else:
                    Zm = np.cos(kap[m - 1] * (zg + 1))
                    b = np.sqrt(kap[m - 1]**2 + lam**2)
                Psi.append(Yn * Zm); beta.append(b)
        Psi = np.array(Psi).T                                  # (ng, nmodes)
        beta = np.array(beta)
        Gm = Psi.T @ MG @ Psi
        # DtN (weak):  int phi_x v = MG Psi diag(beta) Gm^{-1} Psi^T MG phi
        self.Dg = MG @ Psi @ np.diag(beta) @ np.linalg.solve(Gm, Psi.T @ MG)
        # incident wave on Gamma: phi_I = (-i g A/omega) cosh k(z+h)/cosh kh exp(i k x)
        self.phiI_g = (-1j / self.omega) * np.cosh(k * (zg + 1)) / np.cosh(k) * np.exp(-1j * k * self.L)

    def solve(self):
        D = sps.coo_matrix(self.Dg)
        rows = self.gnodes[D.row]; cols = self.gnodes[D.col]
        Dfull = sps.coo_matrix((D.data, (rows, cols)), shape=(self.nn, self.nn)).tocsr()
        A = (self.Ks - self.K * self.Mfs).astype(complex) + Dfull
        rhs = np.zeros(self.nn, complex)
        rhs_g = -1j * self.k * (self.MG @ self.phiI_g) + self.Dg @ self.phiI_g
        rhs[self.gnodes] = rhs_g
        lu = spla.splu(A.tocsc(), permc_spec='COLAMD')
        self.phi = lu.solve(rhs)
        self.p = 1j * self.omega * self.phi        # pressure amplitude (rho = 1)
        return self

    # ---------- wall-face loads by exact surface integration ----------
    def wall_loads(self, zref=0.0, ng=3):
        nx, ny, nz = self.nx, self.ny, self.nz
        xg, wg = np.polynomial.legendre.leggauss(ng)
        F = np.zeros(3, complex); M = np.zeros(3, complex)
        X, Y, Z, P = self.X, self.Y, self.Z, self.p.reshape(self.X.shape)
        for j in range(ny):
            for kk in range(nz):
                idx = [(j, kk), (j + 1, kk), (j + 1, kk + 1), (j, kk + 1)]
                r = np.array([[X[nx, a, b], Y[nx, a, b], Z[nx, a, b]] for a, b in idx])   # (4,3)
                pv = np.array([P[nx, a, b] for a, b in idx])
                for xa, wa in zip(xg, wg):
                    for xb, wb in zip(xg, wg):
                        N = _N2((xa, xb)); dN = _dN2((xa, xb))
                        rp = N @ r
                        t1 = dN[:, 0] @ r; t2 = dN[:, 1] @ r          # d/d(eta_y), d/d(eta_z)
                        ndA = np.cross(t1, t2)                          # points to +x for this ordering
                        if ndA[0] < 0:
                            ndA = -ndA
                        pp = N @ pv
                        dF = pp * ndA * wa * wb
                        F += dF
                        M += np.cross(rp - np.array([0, 0, zref]), dF)
        return dict(Fx=F[0], Fy=F[1], Fz=F[2], Mx=M[0], My=M[1], Mz=M[2])

    def wall_pressure(self):
        P = self.p.reshape(self.X.shape)
        return self.Y[-1], self.Z[-1], P[-1]
