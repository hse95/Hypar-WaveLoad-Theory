"""
Face-warp definitions x = f(y, z) used in the paper (non-dimensional, h = 1).

twisted(c, zc)            : straight-edged (twisted) hypar panel, f = c*y*(z - zc)
umbrella(w, B, D, z0)     : Candela-type four-tympan umbrella panel of width B and face height D,
                            vertex offset w at elevation z0 (= -D/2 when the top edge is at the SWL)
umbrella_avg(w, B, D, z0) : width-averaged (2D chevron) profile of the umbrella, used to check the
                            equivalent-profile theorem
"""
import numpy as np


def twisted(c, zc=-0.5):
    """Twisted panel; c is the twist rate (tau = c*h with h = 1)."""
    return lambda Y, Z: c * Y * (Z - zc)


def umbrella(w, B, D=0.5, z0=-0.25):
    """Umbrella panel; zero warp below the face (z < -D) and on all panel edges."""
    def f(Y, Z):
        inside = (Z >= -D) & (Z <= 0)
        return np.where(inside, w * (1 - 2 * np.abs(Y) / B) * (1 - 2 * np.abs(Z - z0) / D), 0.0)
    return f


def umbrella_avg(w, B, D=0.5, z0=-0.25):
    """Width average of umbrella(): half the vertex offset, uniform across the panel."""
    def f(Y, Z):
        inside = (Z >= -D) & (Z <= 0)
        return np.where(inside, 0.5 * w * (1 - 2 * np.abs(Z - z0) / D), 0.0) + 0 * Y
    return f
