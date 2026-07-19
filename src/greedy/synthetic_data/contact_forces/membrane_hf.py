"""
High-fidelity solver for the MEMBRANE OBSTACLE problem (Niakh-Drouet-Ehrlacher-Ern,
ESAIM:M2AN 2022, Section 5.1).

    -Delta u = l(mu)      in  Omega = (-1/2, 1/2)^2      (side A = 1)
          u >= psi(mu)    in  omega_hat  (fixed reference obstacle region)
          u  = 0          on  Gamma = boundary(Omega)

    psi(mu)(x,y) = -1.25 * ((x-mu2)^2 + (y-mu3)^2) / mu1^2
    l = -1                      (constant vertical load, pushes membrane down)
    mu = (mu1, mu2, mu3) in D = [0.8,1.2] x [-0.05,0.05] x [-0.05,0.05]

The "contact force" is the Lagrange multiplier lambda(mu) >= 0 enforcing u >= psi.

Key design choices (so the dataset is directly usable by CPG / mCPG):
  * A single FIXED P1 mesh -> every lambda(mu) is a vector in the SAME dual space
    (identical DOF ordering), exactly the invariant CPG/mCPG needs.
  * The obstacle constraint is imposed at the fixed set of interior nodes lying in a
    reference disk omega_hat (radius R_OBS). That node set defines W (the dual DOFs);
    dim(W) = n_c is constant. Only psi(mu) (hence the gap) changes with mu, so the
    active contact set shifts inside omega_hat -> genuine mu-dependence in the
    multiplier support, which is what makes the CPG greedy selection non-trivial.
  * lambda is recovered as the multiplier of the inequality constraints via the
    CONDENSED DUAL QP.  Because K (stiffness) and the constraint operator are FIXED
    here, the dual Hessian A = S K^{-1} S^T is assembled ONCE and reused for all mu;
    each parameter only updates the linear term.  -> very fast sweep.
  * W-inner product for CPG norms/projections = mass matrix M_c on the constraint
    nodes (makes ||.||_W an L^2-type norm), returned alongside the snapshots.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized
from cvxopt import matrix as cvxmat, spmatrix, solvers

solvers.options["show_progress"] = False

A_SIDE = 1.0            # membrane side length
HALF = A_SIDE / 2.0     # domain is (-HALF, HALF)^2
LOAD = -1.0             # constant load l
PSI_COEF = -1.25        # coefficient in the obstacle paraboloid
R_OBS = 0.35            # radius of the fixed reference obstacle region omega_hat


# ----------------------------------------------------------------------------- mesh
def structured_tri_mesh(n):
    """(n+1)x(n+1) node grid on (-HALF,HALF)^2, each square split into 2 triangles."""
    xs = np.linspace(-HALF, HALF, n + 1)
    ys = np.linspace(-HALF, HALF, n + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    coords = np.column_stack([X.ravel(), Y.ravel()])          # node id = j*(n+1)+i
    tris = []
    idx = lambda i, j: j * (n + 1) + i
    for j in range(n):
        for i in range(n):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tris.append([a, b, c])
            tris.append([a, c, d])
    return coords, np.asarray(tris, dtype=int)


# ------------------------------------------------------------------------- assembly
def assemble_K_M(coords, tris):
    """P1 stiffness K = int grad.grad and consistent mass M = int phi_i phi_j."""
    nn = coords.shape[0]
    Ke_rows, Ke_cols, Ke_vals = [], [], []
    Me_rows, Me_cols, Me_vals = [], [], []
    Mref = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], float) / 24.0
    for t in tris:
        p = coords[t]
        v1, v2 = p[1] - p[0], p[2] - p[0]
        detJ = v1[0] * v2[1] - v1[1] * v2[0]
        area = 0.5 * abs(detJ)
        # gradients of P1 shape functions
        b = np.array([p[1, 1] - p[2, 1], p[2, 1] - p[0, 1], p[0, 1] - p[1, 1]])
        c = np.array([p[2, 0] - p[1, 0], p[0, 0] - p[2, 0], p[1, 0] - p[0, 0]])
        Ke = (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)
        Me = Mref * (2.0 * area)
        for a in range(3):
            for bb in range(3):
                Ke_rows.append(t[a]); Ke_cols.append(t[bb]); Ke_vals.append(Ke[a, bb])
                Me_rows.append(t[a]); Me_cols.append(t[bb]); Me_vals.append(Me[a, bb])
    K = sp.csr_matrix((Ke_vals, (Ke_rows, Ke_cols)), shape=(nn, nn))
    M = sp.csr_matrix((Me_vals, (Me_rows, Me_cols)), shape=(nn, nn))
    return K, M


def obstacle(coords, mu):
    mu1, mu2, mu3 = mu
    return PSI_COEF * ((coords[:, 0] - mu2) ** 2 + (coords[:, 1] - mu3) ** 2) / mu1 ** 2


# --------------------------------------------------------------------------- solver
class MembraneHF:
    def __init__(self, n=48):
        self.coords, self.tris = structured_tri_mesh(n)
        self.n = n
        nn = self.coords.shape[0]

        K, M = assemble_K_M(self.coords, self.tris)

        # Dirichlet u=0 on the boundary -> keep only interior DOFs
        on_bnd = (np.abs(self.coords[:, 0]) > HALF - 1e-9) | \
                 (np.abs(self.coords[:, 1]) > HALF - 1e-9)
        self.free = np.where(~on_bnd)[0]
        fmap = -np.ones(nn, int); fmap[self.free] = np.arange(self.free.size)

        self.K = K[self.free][:, self.free].tocsc()
        self.M = M[self.free][:, self.free].tocsc()
        self.coords_free = self.coords[self.free]

        # load vector  f_i = int l * phi_i  = l * (row sums of M)
        self.f = LOAD * np.asarray(M[self.free].sum(axis=1)).ravel()

        # fixed dual space: interior nodes inside reference disk omega_hat
        r = np.linalg.norm(self.coords_free, axis=1)
        self.cnodes = np.where(r <= R_OBS)[0]          # indices into free DOFs
        self.n_c = self.cnodes.size
        self.cnode_coords = self.coords_free[self.cnodes]

        # W-Gram (mass on constraint nodes) for CPG norms/projections
        self.Mc = self.M[self.cnodes][:, self.cnodes].tocsc()

        # ---- condensed dual QP data that is CONSTANT over mu ------------------
        # primal:  min 1/2 u^T K u - f^T u   s.t.  L2 cone condition
        #          [M_c (u_c - psi_c)]_i >= 0   (from  int (psi-u) eta <= 0, eta>=0)
        #   => C u <= d  with  C = -M_c S,  d = -M_c psi_c
        # dual  :  min 1/2 z^T A z - b^T z    s.t. z >= 0
        #          A = C K^{-1} C^T = M_c (S K^{-1} S^T) M_c
        #          b = C K^{-1} f - d = M_c (psi_c - S K^{-1} f)
        # here z = lambda are the coefficients of the P1 multiplier field.
        Ksolve = factorized(self.K)                    # one sparse LU, reused
        self.Kinv_f = Ksolve(self.f)                   # K^{-1} f
        St = np.zeros((self.free.size, self.n_c))
        St[self.cnodes, np.arange(self.n_c)] = 1.0
        KinvSt = np.column_stack([Ksolve(St[:, j]) for j in range(self.n_c)])
        A0 = KinvSt[self.cnodes, :]                    # S K^{-1} S^T
        A0 = 0.5 * (A0 + A0.T)
        Mc_d = self.Mc.toarray()
        self.A = Mc_d @ A0 @ Mc_d                       # M_c A0 M_c
        self.A = 0.5 * (self.A + self.A.T)
        self._Mc_d = Mc_d
        self._Ksolve = Ksolve
        # cvxopt constant pieces
        self._P = cvxmat(self.A)
        self._G = spmatrix(-1.0, range(self.n_c), range(self.n_c))   # -I : z>=0
        self._h = cvxmat(0.0, (self.n_c, 1))

    def solve(self, mu):
        """Return (u_free, lambda_c) for one parameter value."""
        psi_c = obstacle(self.cnode_coords, mu)
        # dual linear term  b = M_c (psi_c - S K^{-1} f)
        b = self._Mc_d @ (psi_c - self.Kinv_f[self.cnodes])
        q = cvxmat(-b)
        sol = solvers.qp(self._P, q, self._G, self._h)
        lam = np.asarray(sol["x"]).ravel()
        lam[lam < 0] = 0.0
        # recover primal:  u = K^{-1}(f - C^T z) = K^{-1}(f + S^T M_c lam)
        rhs = self.f.copy()
        rhs[self.cnodes] += self._Mc_d @ lam
        u = self._Ksolve(rhs)
        return u, lam

    # -------- convenience: full nodal field (with boundary zeros) for plotting
    def to_full(self, u_free):
        full = np.zeros(self.coords.shape[0])
        full[self.free] = u_free
        return full


def training_grid():
    """The 5x5x5 = 125 point training set from the paper (Sec 5.1)."""
    mu1 = np.array([0.8 + 0.1 * i for i in range(5)])
    mu2 = np.array([-0.05 + 0.025 * i for i in range(5)])
    mu3 = np.array([-0.05 + 0.025 * i for i in range(5)])
    grid = [(a, b, c) for a in mu1 for b in mu2 for c in mu3]
    return np.array(grid)


if __name__ == "__main__":
    hf = MembraneHF(n=48)
    print(f"free DOFs = {hf.free.size},  dual DOFs n_c = {hf.n_c}")
    u, lam = hf.solve((1.0, 0.0, 0.0))
    print(f"single solve: min u = {u.min():.4e}, max lambda = {lam.max():.4e}, "
          f"active nodes = {(lam > 1e-9).sum()}/{hf.n_c}")
