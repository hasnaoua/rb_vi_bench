"""
Generate and package the contact-force (Lagrange-multiplier) datasets for both
test cases of Niakh-Drouet-Ehrlacher-Ern (ESAIM:M2AN 2022) into .npz files.

Each .npz bundles everything CPG / mCPG needs:
    Lambda      dual snapshots  (n_c x P), columns lambda(mu_p) >= 0  (in W+)
    W_gram      Gram matrix of the dual W-inner product  (n_c x n_c)
    mu_samples  the P training parameters
    + geometry (node coords / contact abscissa, gap) for plotting & context.
"""
import os
import numpy as np
import membrane_hf as mhf
import hertz_hf as hz

OUT = "/mnt/user-data/outputs"
os.makedirs(OUT, exist_ok=True)


def gen_membrane():
    hf = mhf.MembraneHF(n=48)
    grid = mhf.training_grid()                       # 125 x 3
    cache = "_Lam_membrane.npy"
    if os.path.exists(cache) and np.load(cache).shape == (hf.n_c, len(grid)):
        Lam = np.load(cache)
    else:
        Lam = np.column_stack([hf.solve(mu)[1] for mu in grid])
    np.savez_compressed(
        os.path.join(OUT, "membrane_contact_forces.npz"),
        Lambda=Lam,                                  # (885, 125)
        W_gram=hf.Mc.toarray(),                      # (885, 885) L2 mass on omega_hat
        mu_samples=grid,                             # (125, 3) = (mu1,mu2,mu3)
        node_coords=hf.cnode_coords,                 # (885, 2) dual-node positions
        R_obstacle=mhf.R_OBS, load=mhf.LOAD, domain_half=mhf.HALF,
        description="Membrane obstacle (Sec 5.1): lambda>=0 = obstacle reaction "
                    "on omega_hat; W-norm = W_gram (L2). mu=(radius,cx,cy).")
    return hf, grid, Lam


def gen_hertz():
    hf = hz.HertzHF(nr=24, na=110)
    grid = hz.training_grid()                        # 81
    cacheL, cacheX = "_Lam_hertz.npy", "_x_hertz.npy"
    if os.path.exists(cacheL) and np.load(cacheL).shape == (hf.n_c, len(grid)):
        Lam = np.load(cacheL)
        gaps = np.column_stack([hf.solve(mu)[2] for mu in grid])  # cheap recompute of gaps
    else:
        cols = [hf.solve(mu) for mu in grid]
        Lam = np.column_stack([c[0] for c in cols])
        gaps = np.column_stack([c[2] for c in cols])
    np.savez_compressed(
        os.path.join(OUT, "hertz_contact_forces.npz"),
        Lambda=Lam,                                  # (47, 81)
        W_gram=hf.Mc,                                # (47, 47) 1-D arc L2 mass
        mu_samples=grid,                             # (81,) = R2
        contact_abscissa=hf.cn_x,                    # (47,) x along Gamma^c_1
        gap=gaps,                                    # (47, 81) initial gap g(x;mu)
        R1=hz.R1, E=hz.E, nu=hz.NU, gamma0=hz.GAMMA0, d_imposed=hz.D_IMP,
        description="Hertz contact (Sec 5.2): lambda>=0 = contact pressure on "
                    "body-1 arc; W-norm = W_gram (L2). mu = R2 (body-2 radius).")
    return hf, grid, Lam


if __name__ == "__main__":
    m_hf, m_grid, m_Lam = gen_membrane()
    print(f"membrane: Lambda {m_Lam.shape}, dual DOFs {m_hf.n_c}, P={len(m_grid)}")
    h_hf, h_grid, h_Lam = gen_hertz()
    print(f"hertz   : Lambda {h_Lam.shape}, dual DOFs {h_hf.n_c}, P={len(h_grid)}")
    print("saved -> membrane_contact_forces.npz , hertz_contact_forces.npz")
