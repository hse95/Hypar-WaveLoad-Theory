"""
Verification of the first-order theory against the independent FE solution of the exact
linear problem (Table 1 and Section 5 of the paper). First-order FE coefficients are obtained
by central differences in the warp amplitude.

    python -m src.verify --case=twist --mesh=medium

Available cases: twist, umbrella, umbrella-square, cutoff
Each FE solve takes ~5-40 s on one CPU core (two solves per coefficient set).
"""
import argparse, json, os, time
import numpy as np
from .theory import FirstOrderSolution
from .fem import WarpedWallFEM
from .geometry import twisted, umbrella, umbrella_avg

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MESHES = {"twist": {"coarse": (48, 16, 12), "medium": (64, 24, 16)},
          "umbrella": {"coarse": (48, 16, 16), "medium": (64, 24, 24)}}


def fe_derivative(kh, B, make, eps, mesh, zref, **kw):
    """Central difference d(loads)/d(amplitude) from two FE solves at +eps and -eps."""
    Lp = WarpedWallFEM(kh, B, make(+eps), *mesh, **kw).solve().wall_loads(zref=zref)
    Lm = WarpedWallFEM(kh, B, make(-eps), *mesh, **kw).solve().wall_loads(zref=zref)
    return {k: (Lp[k] - Lm[k]) / (2 * eps) for k in Lp}


def report(fe, th, keys):
    for k in keys:
        rel = abs(fe[k] - th[k]) / abs(th[k]) * 100
        print(f"   {k}:  FE = {complex(fe[k]):.5f}   theory = {complex(th[k]):.5f}   diff = {rel:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="twist", choices=["twist", "umbrella", "umbrella-square", "cutoff"])
    ap.add_argument("--mesh", default="medium", choices=["coarse", "medium"])
    ap.add_argument("--save", action="store_true", help="save cutoff results to data/twist_cutoff.json")
    a = ap.parse_args()
    t0 = time.time()
    if a.case == "twist":          # Table 1, twisted panel: kh = 1, B = 2h, zc = -h/2, moments about zc
        th = FirstOrderSolution(1.0, 2.0, twisted(1.0), N=80, M=80).loads(zref=-0.5)
        fe = fe_derivative(1.0, 2.0, twisted, 0.02, MESHES["twist"][a.mesh], -0.5)
        report(fe, th, ["Fy", "Mx", "Mz"])
        print(f"   symmetric loads (must vanish): |dFx|={abs(fe['Fx']):.1e} |dFz|={abs(fe['Fz']):.1e} |dMy|={abs(fe['My']):.1e}")
    elif a.case == "umbrella":     # Table 1, umbrella: kh = 1, B = h, D = h/2, moments about the SWL
        zb = [-1, -0.5, -0.25, 0]
        th = FirstOrderSolution(1.0, 1.0, umbrella(1.0, 1.0), zbreaks=zb, N=80, M=120, nq=200).loads(zref=0.0)
        fe = fe_derivative(1.0, 1.0, lambda w: umbrella(w, 1.0), 0.01, MESHES["umbrella"][a.mesh], 0.0)
        fe2 = fe_derivative(1.0, 1.0, lambda w: umbrella_avg(w, 1.0), 0.01, MESHES["umbrella"][a.mesh], 0.0)
        report(fe, th, ["Fx", "Fz", "My"])
        print(f"   equivalent-profile check: 3D vs width-averaged 2D FE, |dFx| diff = "
              f"{abs(fe['Fx'] - fe2['Fx']) / abs(fe['Fx']) * 100:.3f}%")
    elif a.case == "umbrella-square":   # square umbrella panels B = D = h/2, graded mesh, kh = 1 and 2.5
        zb = [-1, -0.5, -0.25, 0]
        for kh in [1.0, 2.5]:
            th = FirstOrderSolution(kh, 0.5, umbrella(1.0, 0.5), zbreaks=zb, N=40, M=120, nq=200).loads(zref=0.0)
            fe = fe_derivative(kh, 0.5, lambda w: umbrella(w, 0.5), 0.01, MESHES["umbrella"][a.mesh], 0.0, xgrade=1.5)
            print(f" kh = {kh}"); report(fe, th, ["Fx", "Fz", "My"])
    elif a.case == "cutoff":       # yaw moment approaching the first transverse cutoff, B = 2h
        rec = []
        for r in [0.5, 0.7, 0.85, 0.92, 0.96]:
            kh = r * np.pi / 2.0
            th = FirstOrderSolution(kh, 2.0, twisted(1.0), N=80, M=80).loads(zref=-0.5)
            fe = fe_derivative(kh, 2.0, twisted, 0.02, MESHES["twist"][a.mesh], -0.5)
            print(f" kB/pi = {r}"); report(fe, th, ["Mz", "Fy"])
            rec.append(dict(r=r, kh=kh, fem={k: [float(np.real(v)), float(np.imag(v))] for k, v in fe.items()},
                            theory={k: [float(np.real(v)), float(np.imag(v))] for k, v in th.items()}))
        if a.save:
            json.dump(rec, open(os.path.join(DATA, "twist_cutoff.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
