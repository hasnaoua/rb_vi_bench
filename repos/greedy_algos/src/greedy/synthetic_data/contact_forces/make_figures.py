"""
Verification figures for the generated contact-force datasets.
  fig 1  membrane: lambda(mu) fields over omega_hat for representative mu
  fig 2  hertz:    lambda(mu) contact-pressure profiles vs abscissa
  fig 3  CPG-readiness: W-weighted singular values + greedy/POD projection error
"""
import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

OUT = "/mnt/user-data/outputs"
plt.rcParams.update({"font.size": 10, "axes.grid": False})


def wpod_error(Lam, W):
    """Relative W-norm POD projection error e(R) = ||Lam - P_R Lam||_W / ||Lam||_W."""
    L = la.cholesky(W + 1e-12 * np.eye(W.shape[0]), lower=False)
    G = L @ Lam                                   # isometry to Euclidean
    U, s, _ = la.svd(G, full_matrices=False)
    tot = (s ** 2).sum()
    err = [np.sqrt(max(tot - (s[:R] ** 2).sum(), 0) / tot) for R in range(len(s) + 1)]
    return np.array(err), s


# ----------------------------------------------------------------- fig 1 membrane
d = np.load(f"{OUT}/membrane_contact_forces.npz")
Lam, W, mus, xy = d["Lambda"], d["W_gram"], d["mu_samples"], d["node_coords"]
tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
# pick 4 representative parameters: base, small radius, shifted centre x/y
def find(mu1, mu2, mu3):
    return np.argmin(np.sum((mus - np.array([mu1, mu2, mu3])) ** 2, axis=1))
picks = [(find(1.0, 0.0, 0.0),  "mu=(1.0, 0, 0)"),
         (find(0.8, 0.0, 0.0),  "mu=(0.8, 0, 0)  smaller radius"),
         (find(1.0, 0.05, 0.05),"mu=(1.0, .05, .05)  shifted centre"),
         (find(1.2, -0.05, 0.05),"mu=(1.2, -.05, .05)")]
vmax = max(Lam[:, p].max() for p, _ in picks)
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True)
tc = None
for ax, (p, ttl) in zip(axes, picks):
    tc = ax.tricontourf(tri, Lam[:, p], levels=25, cmap="magma", vmin=0, vmax=vmax)
    ax.set_aspect("equal"); ax.set_title(ttl, fontsize=9)
    ax.set_xlim(-0.36, 0.36); ax.set_ylim(-0.36, 0.36)
assert tc is not None
fig.colorbar(tc, ax=axes, shrink=0.8, label=r"contact force  $\lambda(\mu)$")
fig.suptitle("Membrane obstacle (Sec 5.1): contact-force field over $\\hat\\omega$ "
             "— peak and support shift with $\\mu$", fontsize=11)
fig.savefig(f"{OUT}/fig1_membrane_contact_forces.png", dpi=130)
plt.close(fig)

# -------------------------------------------------------------------- fig 2 hertz
d = np.load(f"{OUT}/hertz_contact_forces.npz")
LamH, WH, muH, xarc = d["Lambda"], d["W_gram"], d["mu_samples"], d["contact_abscissa"]
fig, ax = plt.subplots(figsize=(7, 4.4), constrained_layout=True)
cmap = plt.cm.viridis
for mu in [0.7, 0.85, 1.0, 1.15, 1.3]:
    j = int(np.argmin(np.abs(muH - mu)))
    ax.plot(xarc, LamH[:, j], "-o", ms=2.5, color=cmap((mu - 0.7) / 0.6),
            label=f"$R_2=\\mu={muH[j]:.3f}$")
ax.set_xlabel("contact abscissa $x$ along $\\Gamma^c_1$")
ax.set_ylabel(r"contact pressure  $\lambda(\mu)$")
ax.set_title("Hertz contact (Sec 5.2): contact-force profile vs body-2 radius $\\mu$")
ax.legend(frameon=False, fontsize=9); ax.set_xlim(0, 0.32)
fig.savefig(f"{OUT}/fig2_hertz_contact_forces.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------- fig 3 CPG-readiness
errM, sM = wpod_error(Lam, W)
errH, sH = wpod_error(LamH, WH)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
a1.semilogy(range(1, len(sM) + 1), sM / sM[0], "o-", ms=3, label="membrane (885×125)")
a1.semilogy(range(1, len(sH) + 1), sH / sH[0], "s-", ms=3, label="Hertz (47×81)")
a1.set_xlabel("mode index"); a1.set_ylabel("normalised singular value")
a1.set_title("$W$-weighted singular spectrum of $\\{\\lambda(\\mu_p)\\}$")
a1.set_xlim(0, 40); a1.legend(frameon=False); a1.grid(True, which="both", alpha=.3)
a2.semilogy(range(len(errM)), errM, "o-", ms=3, label="membrane")
a2.semilogy(range(len(errH)), errH, "s-", ms=3, label="Hertz")
a2.set_xlabel("reduced dimension $R$")
a2.set_ylabel("rel. projection error $e(R)$ in $\\|\\cdot\\|_W$")
a2.set_title("POD projection error — snapshots are compressible")
a2.set_xlim(0, 40); a2.set_ylim(1e-4, 1.2); a2.legend(frameon=False)
a2.grid(True, which="both", alpha=.3)
fig.savefig(f"{OUT}/fig3_cpg_readiness.png", dpi=130)
plt.close(fig)

print("figures written:")
for f in ["fig1_membrane_contact_forces.png", "fig2_hertz_contact_forces.png",
          "fig3_cpg_readiness.png"]:
    print("  ", f)
print(f"membrane e(R): R=5 -> {errM[5]:.2e}, R=10 -> {errM[10]:.2e}, R=20 -> {errM[20]:.2e}")
print(f"hertz    e(R): R=3 -> {errH[3]:.2e}, R=6 -> {errH[6]:.2e}, R=10 -> {errH[10]:.2e}")
