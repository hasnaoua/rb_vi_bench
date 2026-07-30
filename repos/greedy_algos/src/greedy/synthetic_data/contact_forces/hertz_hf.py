"""
High-fidelity solver for the HERTZ CONTACT problem between two half-disks
(Niakh-Drouet-Ehrlacher-Ern, ESAIM:M2AN 2022, Section 5.2).

Two bodies, plane-strain linear elasticity, frictionless unilateral contact.
  * Body 1 (upper): quarter disk, radius R1 = 1     (parameter-INDEPENDENT)
  * Body 2 (lower): quarter disk, radius R2 = mu    (mu in D = [0.7, 1.3])
  * Initial gap gamma0 at the contact point; imposed displacement -/+ d on the
    outer straight edges pushes the bodies together.
  * E = 15, nu = 0.35, gamma0 = 1e-3, d = 0.09.

By symmetry only quarter disks are meshed; the symmetry axis is x = 0 (u_x = 0).

Small-deformation simplification used in the paper (Sec 5.2): the contact normal is
taken CONSTANT, n = e_y.  The non-interpenetration condition then reduces to a
*vertical gap* constraint along the interface:

        (u2 - u1) . e_y  <=  g(x; mu)  = y1(x) - y2(x)

with  y1(x) = yc1 - sqrt(R1^2 - x^2)   (lower arc of body 1),
      y2(x) =        sqrt(R2^2 - x^2)   (upper arc of body 2, centre at origin),
so no closest-point search is needed.

The "contact force" is the Lagrange multiplier lambda(mu) >= 0 = contact pressure,
defined on body-1's contact arc Gamma^c_1.  Because body 1 is parameter-independent,
this dual space W is FIXED (identical DOF ordering for every mu) -- exactly what
CPG / mCPG require.  Only the constraint operator and the gap change with mu, which
is the genuine "parameter-dependent constraint" setting studied in the paper.

lambda is recovered via the CONDENSED DUAL QP (multiplier of the contact inequality),
using the L2/mass-consistent cone condition so lambda is a proper P1 pressure field
whose W-norm is the 1-D arc mass matrix M_c.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized
from cvxopt import matrix as cvxmat, spmatrix, solvers

solvers.options["show_progress"] = False

R1 = 1.0
E, NU = 15.0, 0.35
GAMMA0 = 1e-3
D_IMP = 0.09
X_CONTACT = 0.45          # potential-contact half-width (x <= X_CONTACT on arc 1)


# ------------------------------------------------------------------- quarter mesh
def quarter_disk(R, center, sign, nr, na):
    """
    Structured polar mesh of a quarter disk.
      position = center + (r cos phi,  sign * r sin phi),  phi in [0, pi/2].
      sign = -1 : lower-right quarter (body 1);  +1 : upper-right (body 2).
    Returns coords (Nx2), tris (Mx3), and boolean edge masks.
    Node layout: node 0 = centre; then rings k=1..nr each with (na+1) nodes.
    """
    # graded angles: cluster nodes near phi = pi/2 (the contact point, x = 0)
    s = np.linspace(0, 1, na + 1)
    phis = (np.pi / 2) * (1 - (1 - s) ** 1.4)
    coords = [center.copy()]
    for k in range(1, nr + 1):
        r = R * k / nr
        for phi in phis:
            coords.append(center + np.array([r * np.cos(phi), sign * r * np.sin(phi)]))
    coords = np.array(coords)

    def nid(k, j):                       # node id of ring k (>=1), angle j
        return 1 + (k - 1) * (na + 1) + j

    tris = []
    for j in range(na):                  # fan around centre (ring 0 -> ring 1)
        tris.append([0, nid(1, j), nid(1, j + 1)])
    for k in range(1, nr):               # quads between ring k and k+1
        for j in range(na):
            a, b = nid(k, j), nid(k, j + 1)
            c, d = nid(k + 1, j + 1), nid(k + 1, j)
            tris.append([a, b, c]); tris.append([a, c, d])
    tris = np.array(tris, int)

    # edge / arc masks
    N = coords.shape[0]
    on_axis = np.zeros(N, bool)          # x = 0  (phi = pi/2)  -> symmetry
    on_flat = np.zeros(N, bool)          # phi = 0  -> outer straight edge (top/bot)
    on_arc = np.zeros(N, bool)           # r = R
    on_axis[0] = True; on_flat[0] = True                       # centre on both
    for k in range(1, nr + 1):
        on_axis[nid(k, na)] = True       # phi = pi/2
        on_flat[nid(k, 0)] = True        # phi = 0
    for j in range(na + 1):
        on_arc[nid(nr, j)] = True
    return coords, tris, on_axis, on_flat, on_arc


# --------------------------------------------------------------- elasticity element
def _D_planestrain():
    lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    mu = E / (2 * (1 + NU))
    return np.array([[lam + 2 * mu, lam, 0],
                     [lam, lam + 2 * mu, 0],
                     [0, 0, mu]]), lam, mu


def assemble_elasticity(coords, tris):
    """Global plane-strain stiffness; DOF ordering (ux,uy) per node."""
    N = coords.shape[0]
    D, _, _ = _D_planestrain()
    rows, cols, vals = [], [], []
    for t in tris:
        p = coords[t]
        x1, y1 = p[0]; x2, y2 = p[1]; x3, y3 = p[2]
        detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = 0.5 * abs(detJ)
        if area < 1e-14:
            continue
        b = np.array([y2 - y3, y3 - y1, y1 - y2]) / detJ
        c = np.array([x3 - x2, x1 - x3, x2 - x1]) / detJ
        B = np.zeros((3, 6))
        for a in range(3):
            B[0, 2 * a] = b[a]
            B[1, 2 * a + 1] = c[a]
            B[2, 2 * a] = c[a]; B[2, 2 * a + 1] = b[a]
        Ke = area * B.T @ D @ B
        dofs = np.array([[2 * n, 2 * n + 1] for n in t]).ravel()
        for a in range(6):
            for bb in range(6):
                rows.append(dofs[a]); cols.append(dofs[bb]); vals.append(Ke[a, bb])
    return sp.csr_matrix((vals, (rows, cols)), shape=(2 * N, 2 * N))


def arc_mass_1d(x_sorted):
    """P1 mass matrix on a 1-D line with node abscissas x_sorted (contact arc)."""
    n = len(x_sorted)
    M = np.zeros((n, n))
    for e in range(n - 1):
        h = abs(x_sorted[e + 1] - x_sorted[e])
        M[e, e] += h / 3; M[e + 1, e + 1] += h / 3
        M[e, e + 1] += h / 6; M[e + 1, e] += h / 6
    return M


# --------------------------------------------------------------------------- solver
class HertzHF:
    def __init__(self, nr=18, na=60):
        self.nr, self.na = nr, na
        # body 1 fixed
        self.yc1 = None                                    # set per mu (depends on R2)
        (self.c1, self.t1, self.ax1, self.fl1, self.arc1) = \
            quarter_disk(R1, np.array([0.0, 0.0]), -1, nr, na)   # placeholder centre
        # body-1 contact nodes: arc nodes with small x (potential contact)
        arc1_ids = np.where(self.arc1)[0]
        x1 = self.c1[arc1_ids, 0]
        sel = np.where(x1 <= X_CONTACT)[0]
        order = np.argsort(x1[sel])                        # increasing x (0 -> edge)
        self.cn_local = arc1_ids[sel][order]               # body-1 node ids (in b1 mesh)
        self.cn_x = x1[sel][order]                         # their abscissa
        self.n_c = self.cn_local.size
        self.Mc = arc_mass_1d(self.cn_x)                   # W-Gram (L2 on the arc)
        self.K1 = assemble_elasticity(self.c1, self.t1)    # body-1 stiffness (fixed)
        self.N1 = self.c1.shape[0]

    # ---- geometry helpers
    def _place_body1(self, R2):
        yc1 = R1 + R2 + GAMMA0
        c1 = self.c1.copy(); c1[:, 1] += yc1               # shift body 1 up
        return c1, yc1

    def solve(self, mu):
        R2 = float(mu)
        c1, yc1 = self._place_body1(R2)
        c2, t2, ax2, fl2, arc2 = quarter_disk(R2, np.array([0.0, 0.0]), +1,
                                              self.nr, self.na)
        K2 = assemble_elasticity(c2, t2)
        N2 = c2.shape[0]

        # ---- global assembly: body1 dofs [0:2N1), body2 dofs [2N1:2N1+2N2)
        N = self.N1 + N2
        K = sp.bmat([[self.K1, None], [None, K2]], format="csr")

        def d1(n, comp): return 2 * n + comp                    # body1 dof
        def d2(n, comp): return 2 * (self.N1 + n) + comp        # body2 dof

        # ---- Dirichlet BCs
        pdofs, pvals = [], []
        for n in np.where(self.ax1)[0]:                         # sym axis body1: ux=0
            pdofs.append(d1(n, 0)); pvals.append(0.0)
        for n in np.where(self.fl1)[0]:                         # top body1: uy=-d
            pdofs.append(d1(n, 1)); pvals.append(-D_IMP)
        for n in np.where(ax2)[0]:                              # sym axis body2: ux=0
            pdofs.append(d2(n, 0)); pvals.append(0.0)
        for n in np.where(fl2)[0]:                              # bottom body2: uy=+d
            pdofs.append(d2(n, 1)); pvals.append(+D_IMP)
        pdofs = np.array(pdofs); pvals = np.array(pvals)
        # deduplicate (corner nodes appear twice)
        uniq = {}
        for dof, val in zip(pdofs, pvals):
            uniq[dof] = val            # last wins; ux=0 / uy=val never conflict (diff comp)
        pdofs = np.array(list(uniq.keys())); pvals = np.array(list(uniq.values()))
        alld = np.arange(2 * N)
        fmask = np.ones(2 * N, bool); fmask[pdofs] = False
        fdofs = alld[fmask]
        fpos = -np.ones(2 * N, int); fpos[fdofs] = np.arange(fdofs.size)

        up = np.zeros(2 * N); up[pdofs] = pvals
        Kff = K[fdofs][:, fdofs].tocsc()
        f_eff = -np.asarray((K[fdofs][:, pdofs] @ up[pdofs])).ravel()   # -K_fp u_p

        # ---- contact operator B: u_free -> Delta_i = u2y(x_i) - u1y_i
        # body-2 arc nodes and their abscissa for interpolation of u2y(x)
        arc2_ids = np.where(arc2)[0]
        x2 = c2[arc2_ids, 0]
        o2 = np.argsort(x2); arc2_ids = arc2_ids[o2]; x2 = x2[o2]
        Brows, Bcols, Bvals = [], [], []
        gap = np.zeros(self.n_c)
        for i in range(self.n_c):
            xi = self.cn_x[i]
            n1 = self.cn_local[i]
            # -u1y : body-1 contact node uy (free)
            j = fpos[d1(n1, 1)]
            if j >= 0:
                Brows.append(i); Bcols.append(j); Bvals.append(-1.0)
            # +u2y interpolated at xi along body-2 arc
            k = np.searchsorted(x2, xi)
            k = min(max(k, 1), len(x2) - 1)
            xL, xR = x2[k - 1], x2[k]
            w = 0.0 if xR == xL else (xi - xL) / (xR - xL)
            for node, wt in [(arc2_ids[k - 1], 1 - w), (arc2_ids[k], w)]:
                jj = fpos[d2(node, 1)]
                if jj >= 0:
                    Brows.append(i); Bcols.append(jj); Bvals.append(wt)
            # gap g(x) = y1(x) - y2(x)
            y1 = yc1 - np.sqrt(max(R1 ** 2 - xi ** 2, 0.0))
            y2 = np.sqrt(max(R2 ** 2 - xi ** 2, 0.0))
            gap[i] = y1 - y2
        B = sp.csr_matrix((Bvals, (Brows, Bcols)),
                          shape=(self.n_c, fdofs.size))

        # ---- condensed dual QP (L2 cone: M_c (B u - gap) <= 0)
        #   C = M_c B ,  d = M_c gap
        #   A = C Kff^{-1} C^T ,  b = C Kff^{-1} f_eff - d
        Ksolve = factorized(Kff)
        Bt = B.toarray().T                                   # (nfree x n_c)
        KinvBt = np.column_stack([Ksolve(Bt[:, i]) for i in range(self.n_c)])
        BKB = B @ KinvBt                                     # B Kff^{-1} B^T
        A = self.Mc @ BKB @ self.Mc
        A = 0.5 * (A + A.T)
        Kinv_feff = Ksolve(f_eff)
        b = self.Mc @ (B @ Kinv_feff - gap)

        P = cvxmat(A + 1e-10 * np.eye(self.n_c))
        q = cvxmat(-b)
        G = spmatrix(-1.0, range(self.n_c), range(self.n_c))
        h = cvxmat(0.0, (self.n_c, 1))
        sol = solvers.qp(P, q, G, h)
        lam = np.asarray(sol["x"]).ravel()
        lam[lam < 0] = 0.0
        return lam, self.cn_x, gap


def training_grid():
    """The 81-point training set from the paper (Sec 5.2): 0.7 + 0.0075 i, i=0..80."""
    return np.array([0.7 + 0.0075 * i for i in range(81)])


if __name__ == "__main__":
    hf = HertzHF(nr=18, na=60)
    print(f"body-1 dofs = {2*hf.N1}, contact dual DOFs n_c = {hf.n_c}")
    lam, x, gap = hf.solve(0.9)
    print(f"mu=0.9  max lambda = {lam.max():.4f}  active = {(lam>1e-6*lam.max()).sum()}/{hf.n_c}")
    print("contact edge (last x with lam>1% max):",
          x[lam > 1e-2 * lam.max()].max() if lam.max() > 0 else None)
