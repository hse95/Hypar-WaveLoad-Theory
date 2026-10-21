"""
Finite-warp sweeps with the FE solver of the exact linear problem (Fig. 5 of the paper).
Results are merged into data/twist_finite.json or data/umbrella_finite.json.

    python -m src.sweeps --case=twist --values 0 0.1 0.2 0.4
    python -m src.sweeps --case=umbrella --values 0 0.05 0.1 -0.1

twist    : kh = 1, B = 2h, zc = -h/2, values are the twist tau = c*h, moments about zc
umbrella : kh = 1, B = D = h/2, top edge at the SWL, values are the vertex offset w/h, moments about the SWL
"""
import argparse, json, os, time
import numpy as np
from .fem import WarpedWallFEM
from .geometry import twisted, umbrella

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="twist", choices=["twist", "umbrella"])
    ap.add_argument("--values", nargs="+", type=float, required=True)
    a = ap.parse_args()
    fn = os.path.join(DATA, f"{a.case}_finite.json")
    out = json.load(open(fn))
    for v in a.values:
        t0 = time.time()
        if a.case == "twist":
            L = WarpedWallFEM(1.0, 2.0, twisted(v), 64, 24, 16).solve().wall_loads(zref=-0.5)
        else:
            L = WarpedWallFEM(1.0, 0.5, umbrella(v, 0.5), 64, 24, 24, xgrade=1.5).solve().wall_loads(zref=0.0)
        out["fem"][str(float(v))] = {k: [float(np.real(x)), float(np.imag(x))] for k, x in L.items()}
        json.dump(out, open(fn, "w"), indent=1)          # save after every solve
        print(f"{a.case} {v:+.3f}: |Fx|={abs(L['Fx']):.5f} |Fy|={abs(L['Fy']):.5f} |Fz|={abs(L['Fz']):.5f} "
              f"|Mz|={abs(L['Mz']):.5f}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
