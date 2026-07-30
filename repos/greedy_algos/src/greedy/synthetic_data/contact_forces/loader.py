"""
Loader + W-inner-product helpers for the contact-force datasets.

    from loader import load_dataset, wnorm, wpod_error
    ds = load_dataset("membrane_contact_forces.npz")
    Lam, W, mus = ds["Lambda"], ds["W_gram"], ds["mu_samples"]

Each column Lam[:, p] is a dual snapshot lambda(mu_p) >= 0 living in the positive
cone W+ of the fixed dual space; W ("W_gram") is the Gram matrix of the dual
W-inner product, i.e.  <a, b>_W = a^T W b  and  ||a||_W = sqrt(a^T W a).

This is exactly the input a CPG / mCPG greedy expects: a fixed-dimension snapshot
matrix whose columns are in the cone, plus the W-Gram used for cone projections and
error measurement.
"""
import numpy as np
import scipy.linalg as la


def load_dataset(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def w_inner(a, b, W):
    return a.T @ (W @ b)


def wnorm(a, W):
    return np.sqrt(np.maximum(np.einsum("i...,i...->...", a, W @ a), 0.0))


def wpod_error(Lam, W):
    """Relative W-norm POD projection error e(R), R = 0..rank. Baseline for CPG."""
    L = la.cholesky(W + 1e-12 * np.eye(W.shape[0]), lower=False)
    s = la.svdvals(L @ Lam)
    tot = (s ** 2).sum()
    return np.array([np.sqrt(max(tot - (s[:R] ** 2).sum(), 0) / tot)
                     for R in range(len(s) + 1)])


if __name__ == "__main__":
    for name in ["membrane_contact_forces.npz", "hertz_contact_forces.npz"]:
        ds = load_dataset(name)
        Lam, W = ds["Lambda"], ds["W_gram"]
        e = wpod_error(Lam, W)
        print(f"{name:32s}  Lambda {Lam.shape}  "
              f"all nonneg = {bool((Lam >= -1e-6).all())}  "
              f"e(R=5)={e[min(5,len(e)-1)]:.2e}  e(R=10)={e[min(10,len(e)-1)]:.2e}")
