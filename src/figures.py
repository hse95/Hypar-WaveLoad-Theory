"""
Regenerates every figure of the paper (into figures/) and the worked-example numbers
(data/fig_numbers.json) from the theory and the FE results stored in data/.

    python -m src.figures
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, FIG = os.path.join(ROOT, 'data'), os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)
import numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .theory import FirstOrderSolution, closed_form_twist, omega_of_k
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 9,
                     'axes.linewidth': 0.7, 'lines.linewidth': 1.3, 'legend.fontsize': 7.5,
                     'legend.frameon': False, 'figure.dpi': 150})
T = json.load(open(os.path.join(DATA, 'twist_finite.json'))); U = json.load(open(os.path.join(DATA, 'umbrella_finite.json')))['fem']
CUT = json.load(open(os.path.join(DATA, 'twist_cutoff.json')))
cplx = lambda a: a[0] + 1j*a[1]
zc = -0.5
D, z0 = 0.5, -0.25
def umb(w, B):
    return lambda Y, Z: np.where((Z >= -D) & (Z <= 0), w*(1-2*np.abs(Y)/B)*(1-2*np.abs(Z-z0)/D), 0.0)
tw = lambda c: (lambda Y, Z: c*Y*(Z - zc))
out = {}
# ------------------------------------------------------------------ Fig 1: geometry (face-offset maps)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[2.2, 1]))
zz = np.linspace(-1, 0, 201)
# (a) twisted panels, B = 2h, mirrored about every panel edge (array <-> channel equivalence)
B1 = 2.0; yy = np.linspace(-1.5*B1, 1.5*B1, 601)
Y, Z = np.meshgrid(yy, zz)
yloc = (Y + B1/2) % (2*B1) - B1/2                  # fold into one mirrored period
yfold = np.where(yloc <= B1/2, yloc, B1 - yloc)
sign = np.where(yloc <= B1/2, 1.0, 1.0)
G = yfold*(Z - zc)/((B1/2)*0.5)                    # normalised so corner offsets = +/-1
c0 = axs[0].contourf(Y, Z, G, 21, cmap='coolwarm', vmin=-1, vmax=1)
for e in np.arange(-1.5, 1.6, 1.0)*B1: axs[0].axvline(e, c='k', lw=0.6)
axs[0].axhline(zc, c='k', ls='--', lw=0.6); axs[0].text(-2.95, zc+0.03, 'twist axis $z=z_c$', fontsize=7)
axs[0].annotate('', xy=(-1.0, 0.0), xytext=(1.0, 0.0), arrowprops=dict(arrowstyle='<->', lw=0.7))
axs[0].text(-0.15, 0.04, '$B$', fontsize=8)
plt.colorbar(c0, ax=axs[0], label=r'face offset $f/\Delta$', ticks=[-1, 0, 1])
axs[0].set_xlabel('$y/h$'); axs[0].set_ylabel('$z/h$'); axs[0].set_ylim(-1, 0.12)
axs[0].set_title(r'(a) twisted panels: $f=c\,y\,(z-z_c)$, mirrored at panel edges', fontsize=8.5)
# (b) Candela-type umbrella panels, B = D = h/2, top edge at SWL
B2 = 0.5; yy2 = np.linspace(-1.5*B2, 1.5*B2, 301); Y2, Z2 = np.meshgrid(yy2, zz)
yl2 = (Y2 + B2/2) % B2 - B2/2
G2 = np.where((Z2 >= -D), (1-2*np.abs(yl2)/B2)*(1-2*np.abs(Z2-z0)/D), 0.0)
c1 = axs[1].contourf(Y2, Z2, G2, 21, cmap='coolwarm', vmin=-1, vmax=1)
for e in np.arange(-1.5, 1.6, 1.0)*B2: axs[1].axvline(e, c='k', lw=0.6)
axs[1].axhline(-D, c='k', lw=0.6); axs[1].plot(0, z0, 'k+', ms=6)
plt.colorbar(c1, ax=axs[1], label=r'face offset $f/w$', ticks=[-1, 0, 1])
axs[1].set_xlabel('$y/h$'); axs[1].set_ylabel('$z/h$'); axs[1].set_aspect('equal')
axs[1].set_title('(b) umbrella panels ($B=D=h/2$)', fontsize=8.5)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig1_geometry.pdf')); plt.close()

# ------------------------------------------------------------------ Fig 2: closed forms
kh = np.linspace(0.05, 5, 300)
zp = -np.tanh(kh/2)/kh
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
axs[0].plot(kh, -zp, 'k'); axs[0].axhline(0.5, ls=':', c='gray'); axs[0].plot(kh[kh > 1], 1/kh[kh > 1], ls='--', c='gray')
axs[0].text(2.6, 0.53, 'shallow limit $h/2$', fontsize=7, color='gray'); axs[0].text(1.5, 0.72, 'deep limit $1/k$', fontsize=7, color='gray')
axs[0].set_xlabel('$kh$'); axs[0].set_ylabel('$-z_p/h$'); axs[0].set_title('(a) pressure-centroid depth', fontsize=8.5)
for zcv in [0, -0.25, -0.5, -0.75, -1.0]:
    axs[1].plot(kh, zcv - zp, label=f'$z_c/h={zcv:g}$')
axs[1].axhline(0, c='k', lw=0.5); axs[1].set_xlabel('$kh$'); axs[1].set_ylabel(r'$C_{Fy}=F_y/(\tau F_{x0})$')
axs[1].legend(ncol=1, loc='upper right'); axs[1].set_title('(b) lateral force', fontsize=8.5)
for Bv in [0.5, 1.0, 2.0]:
    cmx = [closed_form_twist(k_, Bv, 1.0, -0.5)['Mx']/(Bv*2*np.tanh(k_)/k_) for k_ in kh]
    axs[2].plot(kh, cmx, label=f'$B/h={Bv:g}$')
axs[2].axhline(0, c='k', lw=0.5); axs[2].set_xlabel('$kh$'); axs[2].set_ylabel(r'$C_{Mx}=M_x/(\tau F_{x0} h)$')
axs[2].legend(); axs[2].set_title('(c) roll moment ($z_c=-h/2$)', fontsize=8.5)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig2_closed_forms.pdf')); plt.close()

# ------------------------------------------------------------------ Fig 3: frequency response
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
for Bv, col in zip([1.0, 2.0, 4.0], ['C0', 'C1', 'C2']):
    r = np.concatenate([np.linspace(0.15, 3.6, 260)])
    r = r[np.min(np.abs(r[:, None] - np.array([1, 3])[None, :]), axis=1) > 0.004]
    cmz = []
    for rr in r:
        k_ = rr*np.pi/Bv
        s = FirstOrderSolution(k_, Bv, tw(1.0), N=24, M=30, nq=60)
        cmz.append(abs(s.loads(zref=zc)['Mz'])/(Bv*2*np.tanh(k_)/k_*Bv))
    axs[0].semilogy(r, cmz, color=col, label=f'$B/h={Bv:g}$')
fr = [d['r'] for d in CUT] + [2/np.pi]
fm = [abs(cplx(d['fem']['Mz']))/(2.0*2*np.tanh(d['kh'])/d['kh']*2.0) for d in CUT]
fm.append(abs(-0.44925)/(2*2*np.tanh(1.0)*2.0))
axs[0].semilogy(fr, fm, 'ko', ms=3.5, mfc='none', label='FEM, $B/h=2$')
for v in [1, 3]: axs[0].axvline(v, c='gray', ls=':', lw=0.8)
axs[0].set_xlabel(r'$kB/\pi$'); axs[0].set_ylabel(r'$|C_{Mz}|=|M_z|/(\tau F_{x0} B)$')
axs[0].set_title('(a) yaw moment, twisted panels ($z_c=-h/2$)', fontsize=8.5); axs[0].legend(loc='lower right', ncol=1)
kh2 = np.linspace(0.1, 5, 120); re, im, fz, my = [], [], [], []
for k_ in kh2:
    s = FirstOrderSolution(k_, 0.5, umb(1.0, 0.5), zbreaks=[-1, -D, z0, 0], N=8, M=80, nq=100)
    L = s.loads(zref=0.0); Fx0 = 0.5*2*np.tanh(k_)/k_
    re.append(L['Fx'].real/Fx0/k_); im.append(L['Fx'].imag/Fx0/k_); fz.append(L['Fz']/Fx0/k_)
axs[1].plot(kh2, re, label=r'Re$(\delta F_x)/F_{x0}$ (amplitude)')
axs[1].plot(kh2, im, '--', label=r'Im$(\delta F_x)/F_{x0}$ (phase)')
axs[1].plot(kh2, fz, '-.', label=r'$F_z/F_{x0}$')
axs[1].axhline(0, c='k', lw=0.5); axs[1].set_xlabel('$kh$'); axs[1].set_ylabel(r'coefficient per unit $kw$')
axs[1].set_title('(b) umbrella panels ($B=D=h/2$)', fontsize=8.5); axs[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig3_frequency.pdf')); plt.close()

# ------------------------------------------------------------------ Fig 4: pressure maps
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
s = FirstOrderSolution(1.0, 2.0, tw(1.0), N=80, M=80, nq=160)
yy = np.linspace(-1, 1, 121); zz = np.linspace(-1, 0, 81)
P = s.p1(yy, zz).real.T/2.0
im0 = axs[0].contourf(yy, zz, P, 21, cmap='RdBu_r', vmin=-np.abs(P).max(), vmax=np.abs(P).max())
plt.colorbar(im0, ax=axs[0], label=r'Re$(p_1)/(\tau\, 2\rho g A)$')
axs[0].set_xlabel('$y/h$'); axs[0].set_ylabel('$z/h$'); axs[0].set_title('(a) twisted panel, $B=2h$, $kh=1$', fontsize=8.5)
axs[0].set_aspect('equal')
s2 = FirstOrderSolution(1.0, 0.5, umb(1.0, 0.5), zbreaks=[-1, -D, z0, 0], N=80, M=160, nq=200)
yy2 = np.linspace(-0.25, 0.25, 81); zz2 = np.linspace(-0.75, 0, 91)
P2 = s2.p1(yy2, zz2).real.T/2.0
im1 = axs[1].contourf(yy2, zz2, P2, 21, cmap='RdBu_r', vmin=-np.abs(P2).max(), vmax=np.abs(P2).max())
plt.colorbar(im1, ax=axs[1], label=r'Re$(p_1)/((w/h)\, 2\rho g A)$')
axs[1].axhline(-D, c='k', lw=0.6, ls='--'); axs[1].set_aspect('equal')
axs[1].set_xlabel('$y/h$'); axs[1].set_ylabel('$z/h$'); axs[1].set_title('(b) umbrella panel, $B=D=h/2$, $kh=1$', fontsize=8.5)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig4_pressure_maps.pdf')); plt.close()
out['p1_twist_maxabs'] = float(np.abs(P).max()); out['p1_umb_maxabs'] = float(np.abs(P2).max())
out['p1_umb_minmax'] = [float(P2.min()), float(P2.max())]

# ------------------------------------------------------------------ Fig 5: finite warp
th = {k: cplx(v) for k, v in T['theory_per_unit'].items()}
tf = {float(c): {k: cplx(v) for k, v in d.items()} for c, d in T['fem'].items()}
cs = np.array(sorted(c for c in tf if c >= 0)); fem = [tf[c] for c in cs]
Fx0f = abs(fem[0]['Fx'])
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
for key, mk in [('Fy', 'o'), ('Mx', 's'), ('Mz', '^')]:
    ratio = [abs(f[key])/abs(th[key]*c) for f, c in zip(fem[1:], cs[1:])]
    axs[0].plot(cs[1:], ratio, mk+'-', ms=3.5, label={'Fy': '$F_y$', 'Mx': '$M_x$', 'Mz': '$M_z$'}[key])
    out[f'twist_ratio_{key}'] = dict(zip([float(c) for c in cs[1:]], [float(x) for x in ratio]))
axs[0].axhline(1, c='k', lw=0.6); axs[0].set_xlabel(r'$\tau = ch$'); axs[0].set_ylabel('FEM / first-order theory')
axs[0].set_title('(a) twisted: antisymmetric loads', fontsize=8.5); axs[0].legend()
dFx = np.array([(abs(f['Fx'])-Fx0f)/Fx0f*100 for f in fem]); Fz = np.array([abs(f['Fz'])/Fx0f*100 for f in fem])
Fyp = np.array([abs(f['Fy'])/Fx0f*100 for f in fem])
a_fx = np.polyfit(cs[:4]**2, dFx[:4], 1)[0]; a_fz = np.polyfit(cs[:4]**2, Fz[:4], 1)[0]
cc = np.linspace(0, 1, 50)
axs[1].plot(cs, dFx, 'o', ms=3.5, label=r'$(|F_x|-F_{x0})/F_{x0}$'); axs[1].plot(cc, a_fx*cc**2, 'C0', lw=0.8)
axs[1].plot(cs, Fz, 's', ms=3.5, label=r'$|F_z|/F_{x0}$'); axs[1].plot(cc, a_fz*cc**2, 'C1', lw=0.8)
axs[1].plot(cs, Fyp, '^', ms=3.5, label=r'$|F_y|/F_{x0}$'); axs[1].plot(cc, abs(th['Fy'])/Fx0f*100*cc, 'C2', lw=0.8)
axs[1].set_xlabel(r'$\tau = ch$'); axs[1].set_ylabel('% of $F_{x0}$'); axs[1].legend()
axs[1].set_title('(b) twisted: symmetric loads', fontsize=8.5)
out['twist_quad_coeffs_pct'] = dict(Fx=float(a_fx), Fz=float(a_fz))
out['twist_Fx_plus_minus_06'] = [abs(tf[0.6]['Fx']), abs(tf[-0.6]['Fx'])] if -0.6 in tf and 0.6 in tf else None
ws = sorted(float(w) for w in U); Uf = {float(w): {k: cplx(v) for k, v in d.items()} for w, d in U.items()}
Fx0u = abs(Uf[0.0]['Fx']); thu = FirstOrderSolution(1.0, 0.5, umb(1.0, 0.5), zbreaks=[-1, -D, z0, 0], N=40, M=120, nq=200).loads(zref=0.0)
wsa = np.array(ws)
axs[2].plot(wsa*4, [(abs(Uf[w]['Fx'])-Fx0u)/Fx0u*100 for w in ws], 'o', ms=3.5, label=r'$(|F_x|-F_{x0})/F_{x0}$')
axs[2].plot(wsa*4, [Uf[w]['Fz'].real/Fx0u*100 for w in ws], 's', ms=3.5, label=r'$F_z/F_{x0}$')
wl = np.linspace(-0.25, 0.25, 20)
axs[2].plot(wl*4, thu['Fx'].real/Fx0u*100*wl, 'C0', lw=0.8); axs[2].plot(wl*4, thu['Fz']/Fx0u*100*wl, 'C1', lw=0.8)
axs[2].axhline(0, c='k', lw=0.5); axs[2].set_xlabel(r'edge-to-vertex slope $2w/B$'); axs[2].set_ylabel('% of $F_{x0}$')
axs[2].set_title('(c) umbrella ($B=D=h/2$)', fontsize=8.5); axs[2].legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig5_finite_warp.pdf')); plt.close()
out['umb_Fx_change_pct'] = {w: float((abs(Uf[w]['Fx'])-Fx0u)/Fx0u*100) for w in ws}
out['umb_Fz_pct'] = {w: float(Uf[w]['Fz'].real/Fx0u*100) for w in ws}
out['umb_theory_per_w'] = {k: [float(np.real(v)), float(np.imag(v))] for k, v in thu.items()}

# ------------------------------------------------------------------ worked example (dimensional)
rho, g, A, h = 1025.0, 9.81, 0.9, 10.0
def kh_of_T(Tp):
    w = 2*np.pi/Tp; k = w**2/g
    for _ in range(200): k = w**2/(g*np.tanh(k*h))
    return k*h
Fs, Ms = rho*g*A*h**2/1e3, rho*g*A*h**3/1e3   # kN, kN m
rows = []
for Bm, Delta in [(10.0, 0.5), (20.0, 0.5)]:
    Bh = Bm/h; tau = Delta/h/(Bh/4)
    kc = np.pi/Bm; Tc = 2*np.pi/np.sqrt(g*kc*np.tanh(kc*h))
    for Tp in [4.0, 5.0, 6.0]:
        k_ = kh_of_T(Tp); s = FirstOrderSolution(k_, Bh, tw(tau), N=60, M=60, nq=120)
        L = s.loads(zref=zc); Fx0 = Bh*2*np.tanh(k_)/k_
        rows.append(dict(B=Bm, Delta=Delta, tau=tau, T=Tp, kh=k_, Tc=Tc, kB_pi=k_*Bh/np.pi,
                         Fx0=Fx0*Fs, Fy=abs(L['Fy'])*Fs, Mx=abs(L['Mx'])*Ms, Mz=abs(L['Mz'])*Ms,
                         zp=-np.tanh(k_/2)/k_*h))
out['worked'] = rows
out['umb_fig3b'] = dict(kh=[float(v) for v in kh2[::10]], re=[float(v) for v in re[::10]], im=[float(v) for v in im[::10]], fz=[float(v) for v in fz[::10]])
json.dump(out, open(os.path.join(DATA, 'fig_numbers.json'), 'w'), indent=1, default=float)
for r_ in rows: print({k: round(v, 3) for k, v in r_.items()})
print({k: v for k, v in out.items() if k != 'worked'})
