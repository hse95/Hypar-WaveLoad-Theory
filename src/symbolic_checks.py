"""
Symbolic checks for the first-order (small-warp) wave-load theory.
(1) First-order wall boundary condition  phi1_x = div_perp( g grad_perp phi0 )
(2) Closed-form pressure centroid of the standing-wave pressure on a vertical wall
(3) Closed-form second moment used in the roll-moment formula
"""
import sympy as sp

x, y, z, eps = sp.symbols('x y z epsilon', real=True)
k, h = sp.symbols('k h', positive=True)

# ---------- (1) boundary-condition derivation, for a generic smooth warp g(y,z) ----------
g = sp.Function('g')(y, z)
phi0 = sp.Function('phi0')(x, y, z)
phi1 = sp.Function('phi1')(x, y, z)
phi = phi0 + eps * phi1
# exact no-flux condition on the surface x = eps*g(y,z):  phi_x - eps*g_y*phi_y - eps*g_z*phi_z = 0
N = sp.diff(phi, x) - eps * sp.diff(g, y) * sp.diff(phi, y) - eps * sp.diff(g, z) * sp.diff(phi, z)
N_on_wall = N.subs(x, eps * g)
first_order = sp.diff(N_on_wall, eps).subs(eps, 0)          # O(eps) coefficient
first_order = sp.simplify(first_order.doit())
print("O(eps) wall condition (must vanish):\n ", first_order)

# Now impose phi0_x = 0 on x=0 and Laplace: phi0_xx = -(phi0_yy + phi0_zz)
# Claimed result: phi1_x(0,y,z) = d/dy(g*phi0_y) + d/dz(g*phi0_z)
claimed = sp.diff(g * sp.diff(phi0, y), y) + sp.diff(g * sp.diff(phi0, z), z)
expr = first_order.subs(x, 0)
# Replace phi1_x by the claimed expression and phi0_xx by -(phi0_yy+phi0_zz)
P1x = sp.Derivative(phi1, x).subs(x, 0)
P0xx = sp.Derivative(phi0, (x, 2)).subs(x, 0)
expr_sub = expr.subs(P1x, claimed.subs(x, 0)).subs(P0xx, -(sp.diff(phi0, y, 2) + sp.diff(phi0, z, 2)).subs(x, 0))
print("Residual after substituting claimed BC and Laplace (should be 0):", sp.simplify(expr_sub))

# Concrete check with an actual Laplace solution (standing wave) and a hypar warp
c, zc = sp.symbols('c z_c', real=True)
phi0_sw = sp.cosh(k*(z+h)) * sp.cos(k*x)            # satisfies Laplace and phi0_x(0)=0
assert sp.simplify(sp.diff(phi0_sw, x, 2) + sp.diff(phi0_sw, z, 2)) == 0
g_h = c*y*(z - zc)
rhs = sp.diff(g_h*sp.diff(phi0_sw, z), z).subs(x, 0)
print("Twisted-hypar wall forcing  d/dz(g phi0_z) =", sp.simplify(rhs))

# ---------- (2) pressure centroid of p0 ~ cosh k(z+h) over -h<z<0 ----------
p0 = sp.cosh(k*(z+h))
P0 = sp.integrate(p0, (z, -h, 0))
M1 = sp.integrate(p0*z, (z, -h, 0))
zp = sp.simplify(M1/P0)
claimed_zp = -sp.tanh(k*h/2)/k
print("z_p (sympy) =", zp, "; difference from -tanh(kh/2)/k:",
      sp.simplify(sp.expand_trig((zp - claimed_zp).rewrite(sp.exp))))
for kh in [0.1, 0.5, 1, 2, 5]:
    v = float((zp - claimed_zp).subs({k: kh, h: 1}))
    print(f"   kh={kh}: numeric diff = {v:.2e}")
print("shallow limit z_p/h:", sp.limit((claimed_zp/h).subs(k, sp.Symbol('q', positive=True)/h), sp.Symbol('q', positive=True), 0))

# ---------- (3) second moment about z_c ----------
M2 = sp.simplify(sp.integrate(p0*(z - zc)**2, (z, -h, 0)))
print("int cosh k(z+h) (z-zc)^2 dz =", M2)
