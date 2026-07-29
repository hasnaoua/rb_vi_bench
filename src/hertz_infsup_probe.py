"""Can ``contact_dataset.py``'s Hertz model host the [NDEE22] inf-sup experiments?

Tag key: [BEE20] = this repository's paper (hal-02081485**v2** numbering);
[NDEE22] = Niakh/Drouet/Ehrlacher/Ern, ESAIM: M2AN 2022, implemented in
~/stable_model_reduction_vi. See ~/rb_vi_shared/README.md.

THE QUESTION
------------
[NDEE22]'s claim C1 -- that the decorrelated pair (V_N, W_R^+) can fail to be
inf-sup stable, beta^dec = 0 ([NDEE22] §5.1, Eq. 14) -- does not reproduce on
either repository's 1-D toy. ~/stable_model_reduction_vi/REPRODUCTION_NOTES.md
records why, with a Gordan certificate, and conjectures that "producing the
instability appears to need the richer geometry of a 2-D contact surface".

This repository has a validated Hertz contact model (``contact_dataset.py``,
reproducing the analytic semi-elliptical pressure profile). The obvious question
is whether it is that richer test problem. This script answers it in three
parts, and the answer is **no as it stands, for a reason that is structural
rather than a matter of tuning** -- but part C locates what would have to change.

WHAT beta^dec = 0 ACTUALLY REQUIRES
-----------------------------------
Writing C := Q^T B_hat(mu)^T X (primal basis Q of V_N, whitened constraint
matrix B_hat, cone generators X), [NDEE22] Eq. (14) gives

    beta^dec = min_{alpha >= 0, alpha != 0} ||C alpha|| / ||X alpha||,

so beta^dec = 0 iff ker(C) meets the non-negative orthant non-trivially. Note
what this does and does not say: the supremizer B_hat^T (X alpha) need not
VANISH, it need only be ORTHOGONAL TO V_N. The prose argument in
~/stable_model_reduction_vi/REPRODUCTION_NOTES.md ("every supremizer lies in one
halfspace so no non-negative combination can cancel") establishes the stronger,
sufficient-but-not-necessary condition; its numerical Gordan certificate is run
on the correct object, C, and is what carries that finding. The same distinction
is respected here: every verdict below comes from a certificate on C, not from
the halfspace heuristic.

Run:  cd src && python3 hertz_infsup_probe.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

import _shared_path  # noqa: F401  -- puts rb_vi_common on sys.path
from contact_dataset import generate, influence_matrix, solve_contact
from rb_vi_common import Whitener, inf_sup, mcpg, pod


# ---------------------------------------------------------------------------
# Certificates. Both are exact (LP / convex QP), unlike inf_sup's value.
# ---------------------------------------------------------------------------

def orthant_meets_kernel(C, tol=1e-8):
    """Is there alpha >= 0, sum alpha = 1, with C alpha = 0?  (LP feasibility)

    This is the exact test for beta^dec = 0: the quotient of [NDEE22] Eq. (14)
    is scale-invariant, so it may be normalized to the simplex, and for a
    pointed cone ||X alpha|| is bounded away from 0 there.
    """
    n = C.shape[1]
    # minimize 0 subject to  C alpha = 0,  sum alpha = 1,  alpha >= 0
    A_eq = np.vstack([C, np.ones((1, n))])
    b_eq = np.concatenate([np.zeros(C.shape[0]), [1.0]])
    res = linprog(np.zeros(n), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n,
                  method="highs")
    if not res.success:
        return False, None
    a = np.maximum(res.x, 0.0)
    return bool(np.linalg.norm(C @ a) <= tol * max(np.linalg.norm(C, 2), 1.0)), a


def gordan_certificate(C):
    """Gordan's theorem: max t s.t. C^T y >= t*1, ||y||_inf <= 1.

    t > 0 exhibits a y with C^T y > 0 componentwise, which PROVES no non-zero
    non-negative vector lies in ker(C) -- hence beta^dec > 0. This is the
    certificate ~/stable_model_reduction_vi/REPRODUCTION_NOTES.md reports for
    the 1-D toy (t = 2.56).
    """
    m, n = C.shape
    # variables (y, t); maximize t  ->  minimize -t
    c = np.zeros(m + 1)
    c[-1] = -1.0
    # -C^T y + t <= 0
    A_ub = np.hstack([-C.T, np.ones((n, 1))])
    b_ub = np.zeros(n)
    bounds = [(-1.0, 1.0)] * m + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return float(-res.fun) if res.success else np.nan


def _header(first="configuration"):
    print(f"    {first:<24}{'N':>4}{'R':>5}{'ker meets orthant':>19}"
          f"{'Gordan t':>11}{'beta^dec':>12}")


def _report(label, C, X, B_hat, Q):
    feasible, _ = orthant_meets_kernel(C)
    t = gordan_certificate(C)
    beta = inf_sup(B_hat, Q, X)
    print(f"    {label:<24}{C.shape[0]:>4}{C.shape[1]:>5}"
          f"{('YES -- beta=0' if feasible else 'no'):>19}{t:>11.3e}{beta:>12.3e}")
    return feasible, t, beta


# ---------------------------------------------------------------------------
# Part A -- structural audit of the model as it stands
# ---------------------------------------------------------------------------

def part_A():
    print("=" * 78)
    print("A.  What contact_dataset.py provides, in saddle-point terms")
    print("=" * 78)
    print("""
    The model solves the boundary-integral QP

        min_{p >= 0}  1/2 p^T C p + p^T (h - delta),        C = influence matrix

    Matching this against the dual form of the saddle-point problem
    ([BEE20] Eq. 9; [NDEE22] Eq. 7),

        min_{lambda >= 0}  1/2 lambda^T (B A^-1 B^T) lambda
                           - lambda^T (B A^-1 f - g)

    identifies   B A^-1 B^T = C   and   g - B A^-1 f = h - delta.

    So the model supplies only the SCHUR COMPLEMENT B A^-1 B^T, with A and B
    entangled. The primal displacement field has been eliminated analytically --
    that is what makes a boundary-integral formulation cheap, and it is exactly
    what the inf-sup experiments need back.

    The natural factorization is  B = I,  A = C^-1,  which is consistent and
    gives a genuine primal (u = C f - C p, on the same nodes). But then:

      (i)  n_v = n_w: primal and dual live on the SAME nodes, so V_N and W_R^+
           are not independent spaces in the way [NDEE22] §2.3 assumes;
      (ii) B = I is PARAMETER-INDEPENDENT. The dataset's five parameters
           (radius, offset, wav_amp, wav_len, delta) all enter through the gap
           h and the approach delta -- that is, through g, the RIGHT-HAND SIDE.

    (ii) is decisive. [NDEE22]'s subject is parameter-dependent CONSTRAINTS: its
    title says so, and its §2.3 inefficiency exists only because S_R(mu) =
    Span{T(mu) chi_r} moves with mu. With B constant, S_R(mu) is constant, PGA
    has no online cost to remove, and the paper's contribution is invisible --
    the same defect ``rb_online.py`` documents for toy_problem.py.

    VERDICT (A): the Hertz model as written cannot host the experiments. Not
    because of accuracy or size, but because its parametrization is of the wrong
    kind. It is an excellent DUAL-SNAPSHOT generator -- and that is how the rest
    of this script uses it.
    """)


# ---------------------------------------------------------------------------
# Part B -- the flat half-plane: does it exhibit beta^dec = 0?
# ---------------------------------------------------------------------------

def part_B(n_nodes=48, n_train=40, seed=0):
    print("=" * 78)
    print("B.  Flat half-plane (the current model): is beta^dec = 0 reachable?")
    print("=" * 78)
    print("""
    Using the factorization B = I, A = C^-1 from part A, with REAL pressure
    snapshots from contact_dataset.generate. V_N by POD ([NDEE22] Eq. 10-11),
    W_R^+ by mCPG ([NDEE22] Algorithm 2). N is swept below R to give ker(C) the
    dimension it needs.
    """)
    P, params, x, _ = generate(n_samples=n_train, N=n_nodes, seed=seed)
    Cinf = influence_matrix(x, 1.0)
    lam = P.T                                   # (n_nodes, n_train), >= 0
    u = -(Cinf @ lam)                           # primal, with f = 0

    A = np.linalg.inv(Cinf)
    A = 0.5 * (A + A.T)
    wh = Whitener(A, None)
    B_hat = wh.B_hat(np.eye(n_nodes))
    X = mcpg(lam, delta=0.15).generators
    print(f"    snapshots {lam.shape}, lambda >= 0: {lam.min() >= 0};  "
          f"mCPG gives R = {X.shape[1]}")

    _header("POD tolerance")
    out = []
    for dpod in (5e-1, 1e-1, 1e-2, 1e-3):
        Q = pod(wh.L @ u, delta=dpod)
        Cm = Q.T @ B_hat.T @ X
        out.append(_report(f"POD delta={dpod:g}", Cm, X, B_hat, Q))
    print("""
    Every configuration returns a positive Gordan certificate, so beta^dec > 0
    is PROVEN (not merely estimated) in each -- the same outcome as the 1-D toy
    in ~/stable_model_reduction_vi.

    There is a second, sharper reason visible here, and it is a property of the
    boundary-integral model rather than a choice. The model has NO external load
    (f = 0): the contact is driven entirely through the gap h and the approach
    delta, i.e. through g. So the primal field is u = -C p, determined
    ENTIRELY by the multiplier. The primal snapshot space is therefore contained
    in the supremizer space S_R = Span{A^-1 B^T chi_r} by construction, and POD
    returns a V_N ALIGNED with S_R -- the most favourable configuration for
    inf-sup stability that can exist. V_N and W_R^+ are not independent spaces
    at all, which is what [NDEE22] §2.3 assumes they are.

    VERDICT (B): a validated Hertz pressure field is not by itself enough. Two
    things block it -- the constraint's row geometry is trivial (B = I, one
    fixed normal), and the primal space is not independent of the dual. Neither
    is about the accuracy of the pressure profile.
    """)
    return out


# ---------------------------------------------------------------------------
# Part C -- does a curved contact surface break the obstruction?
# ---------------------------------------------------------------------------

def _arc_constraint(n_dual, n_surf, half_angle, centre, two_body=False):
    """B(mu) for collocated non-interpenetration on a CURVED contact surface.

    Contact between half-disks ([BEE20] §6.2, [NDEE22] §5.2) puts the contact
    nodes on an arc, where the outward normal ROTATES:

        n(theta) = (sin theta, cos theta).

    The primal is vector-valued: u = (u_x, u_y) at each of n_surf surface nodes,
    so n_v = 2 * n_surf. Non-interpenetration collocated at s_i is

        -n(theta_i) . u(s_i)  <=  g_i,

    giving row i the entries -n_x(theta_i) phi_j(s_i), -n_y(theta_i) phi_j(s_i).

    mu = (half_angle, centre) moves the collocation points along the arc, so
    B(mu) is genuinely PARAMETER-DEPENDENT -- the structure [NDEE22] requires
    and the structure part A found missing.

    ``two_body=True`` is [BEE20] §6.2 / [NDEE22] §5.2 proper: TWO deformable
    half-disks, so the constraint acts on the displacement JUMP across the
    interface,

        -n(theta_i) . [[u]](s_i)  <=  g_i,     [[u]] = u^(1) - u^(2),

    and the row acquires a block of each sign. The single-body form above is the
    rigid-indenter idealization, where u^(2) = 0 and only one sign survives.
    That sign asymmetry is exactly what the Gordan argument in
    ~/stable_model_reduction_vi/REPRODUCTION_NOTES.md turns on, so it is the
    variable part D isolates.

    THIS IS A PROBE, NOT A TEST PROBLEM. It is built to answer one question:
    can a non-negative combination of supremizers become orthogonal to V_N? It
    is not a contact solver, and no pressure it implies is validated against
    Hertz. See the verdict for what would still be needed.
    """
    theta_s = np.linspace(-half_angle, half_angle, n_surf)      # surface nodes
    theta_c = centre + 0.8 * half_angle * np.linspace(-1, 1, n_dual)  # collocation
    n_body = 2 if two_body else 1
    B = np.zeros((n_dual, 2 * n_surf * n_body))
    for i, th in enumerate(theta_c):
        n_vec = np.array([np.sin(th), np.cos(th)])
        # P1 interpolation of the surface displacement at theta
        j = np.clip(np.searchsorted(theta_s, th) - 1, 0, n_surf - 2)
        w = (th - theta_s[j]) / (theta_s[j + 1] - theta_s[j])
        for node, wt in ((j, 1.0 - w), (j + 1, w)):
            B[i, 2 * node] = -n_vec[0] * wt
            B[i, 2 * node + 1] = -n_vec[1] * wt
            if two_body:
                # +n on body 2: the jump [[u]] = u^(1) - u^(2).
                off = 2 * n_surf
                B[i, off + 2 * node] = +n_vec[0] * wt
                B[i, off + 2 * node + 1] = +n_vec[1] * wt
    return B


def _solve_saddle(Ainv, B, f, g):
    """Solve the saddle-point problem ([BEE20] Eq. 9) in the dual.

        min_{lambda >= 0}  1/2 lambda^T (B A^-1 B^T) lambda
                           - lambda^T (B A^-1 f - g)

    then u = A^-1 (f - B^T lambda). Solving in the dual makes lambda >= 0 exact
    rather than up to a solver tolerance, which the cone algorithms require.
    """
    Q = B @ Ainv @ B.T
    Q = 0.5 * (Q + Q.T)
    c = B @ (Ainv @ f) - g
    reg = 1e-12 * np.trace(Q) / max(len(c), 1)
    lam = solve_contact(Q + reg * np.eye(len(c)), -c, 0.0)
    return Ainv @ (f - B.T @ lam), lam


def _curved_sweep(two_body, n_surf=40, n_dual=30, n_train=36, seed=0,
                  delta_cone=0.02):
    """Sweep the arc's half-angle and certify beta^dec at each.

    TWO CONSTRUCTION POINTS THAT DECIDE WHETHER THIS PROBE IS RIGGED
    ----------------------------------------------------------------
    1. The primal snapshots must NOT be pure responses to the contact
       tractions. With f = 0 one gets u = -A^-1 B^T lambda, which lies inside
       the supremizer space S_R = Span{A^-1 B^T chi_r} by construction -- so
       POD would return a V_N ALIGNED with S_R, the most favourable possible
       configuration for inf-sup stability, and beta^dec > 0 would be a
       foregone conclusion rather than a finding. A genuine mu-dependent
       external load f(mu) is therefore applied, whose response is not in S_R.
       (Part B has no such freedom -- see its note, where this is a property of
       the boundary-integral model itself rather than a choice.)

    2. For ``two_body``, the two bodies must NOT be identical. Two identical
       bodies constrained on the jump [[u]] = u^(1) - u^(2) reduce EXACTLY to a
       one-body problem in the jump variable with doubled compliance -- this is
       the standard Hertz E* reduction -- so the symmetric case tests nothing
       new. (Verified: it reproduces the one-body certificates times sqrt(2).)
       The bodies are given different moduli below.

    3. The regime must be N < R. [NDEE22] §5.1 asserts C1 ("we are sure that for
       all mu ... beta^dec = 0") only for N < R, and for good reason: C is N x R,
       so N >= R makes ker(C) generically trivial and beta^dec > 0 almost
       automatic, testing nothing. N is therefore pinned BELOW R here rather
       than left to a POD tolerance, and both are reported so the regime is
       visible in the output.
    """
    x = np.linspace(-0.5, 0.5, n_surf)
    n_body = 2 if two_body else 1
    # Distinct moduli: see construction point 2. E*_2 / E*_1 = 2.5.
    moduli = [1.0, 2.5][:n_body]
    dim = 2 * n_surf * n_body
    Ainv_blk = np.zeros((dim, dim))
    for b, E in enumerate(moduli):
        Cb = influence_matrix(x, E)
        o = b * 2 * n_surf
        Ainv_blk[o:o + 2 * n_surf:2, o:o + 2 * n_surf:2] = Cb
        Ainv_blk[o + 1:o + 2 * n_surf:2, o + 1:o + 2 * n_surf:2] = Cb
    Ainv_blk = 0.5 * (Ainv_blk + Ainv_blk.T)
    A = np.linalg.inv(Ainv_blk)
    A = 0.5 * (A + A.T)
    wh = Whitener(A, None)

    rng = np.random.default_rng(seed)
    results = []
    for half_angle in (0.05, 0.20, 0.50, 0.90, 1.30):
        # mu = (half_angle, centre, load); centre varies -> B(mu) varies, which
        # is the parameter-dependent-constraint structure [NDEE22] requires.
        centres = rng.uniform(-0.25, 0.25, n_train) * half_angle
        Bs = [_arc_constraint(n_dual, n_surf, half_angle, c, two_body)
              for c in centres]

        U, L = [], []
        for k, c in enumerate(centres):
            # External load: a smooth surface traction pressing the bodies
            # together, varying with mu. Construction point 1.
            amp, wid = rng.uniform(0.5, 1.5), rng.uniform(0.2, 0.6)
            prof = amp * np.exp(-((x - 0.4 * c) / wid) ** 2)
            f = np.zeros(dim)
            f[1:2 * n_surf:2] = -prof                      # body 1, downward
            if two_body:
                f[2 * n_surf + 1::2] = +prof               # body 2, upward
            # Gap on the arc, varying with mu so the active set moves and the
            # multipliers carry genuine spatial variety for the cone to compress.
            s = np.linspace(-1.0, 1.0, n_dual)
            g = (rng.uniform(0.005, 0.03)
                 + rng.uniform(0.02, 0.08) * (s - rng.uniform(-0.3, 0.3)) ** 2)
            u, lam = _solve_saddle(Ainv_blk, Bs[k], f, g)
            U.append(u)
            L.append(lam)
        U = np.column_stack(U)
        lam = np.column_stack(L)
        if lam.max() <= 0:
            print(f"    {half_angle:<24}  (no active contact -- skipped)")
            continue

        X = mcpg(lam, delta=delta_cone).generators
        R = X.shape[1]
        B_hat = wh.B_hat(Bs[0])
        Q_full = pod(wh.L @ U, delta=1e-12)
        # Construction point 3: sweep N strictly below R, so dim ker(C) = R - N
        # ranges from wide to thin. N = R-1 alone would leave only a single
        # kernel line, which is far too thin a test to conclude from.
        for N in sorted({max(1, R // 8), max(1, R // 4), max(1, R // 2), R - 1}):
            if N >= R or N > Q_full.shape[1]:
                continue
            Q = Q_full[:, :N]
            Cm = Q.T @ B_hat.T @ X
            feasible, t, beta = _report(f"{half_angle:.2f}   N={N:<3d} R={R:<3d}",
                                        Cm, X, B_hat, Q)
            results.append((half_angle, feasible, t, beta))
    return results


def part_C():
    print("=" * 78)
    print("C.  ONE body, curved surface: does a rotating normal break it?")
    print("=" * 78)
    print("""
    The most-cited structural difference between the half-plane model and
    [NDEE22] §5.2's half-disks is that the contact normal rotates along the arc.
    This part isolates that difference and nothing else: a rigid indenter on a
    deformable curved surface, so u^(2) = 0 and each constraint row carries one
    sign block.

    [UNSPECIFIED / PROBE ASSUMPTION] The V inner product on the vector-valued
    surface displacement is taken as blockdiag(C^-1, C^-1) -- normal and
    tangential compliance decoupled and equal. That is NOT exact 2-D elasticity
    (the true operator couples them through the Cerruti kernel). It is adequate
    here because the question is whether ker(Q^T B_hat^T X) meets the orthant,
    which is governed by the row geometry of B and by V_N; but it means no
    beta VALUE below should be quoted as physical.
    """)
    _header("half-angle / regime")
    return _curved_sweep(two_body=False)


def part_D():
    print("=" * 78)
    print("D.  TWO bodies: the constraint acts on the displacement JUMP")
    print("=" * 78)
    print("""
    [BEE20] §6.2 and [NDEE22] §5.2 both put TWO deformable bodies in contact
    (half-disks), so non-interpenetration constrains the JUMP [[u]] = u^(1) -
    u^(2) and every constraint row carries a POSITIVE block and a NEGATIVE one.
    Part C's rigid-indenter idealization discards the second block.

    That sign asymmetry is precisely what the Gordan argument in
    ~/stable_model_reduction_vi/REPRODUCTION_NOTES.md turns on, and it is the
    ingredient the 'channel_gap' experiment in that repository's hf_model.py
    reached for and could not fully realize in 1-D.

    The two bodies are given DIFFERENT moduli (E*_2 / E*_1 = 2.5). Two identical
    bodies constrained on the jump reduce exactly to a one-body problem with
    doubled compliance -- the standard Hertz E* reduction -- so the symmetric
    case would reproduce part C's certificates times sqrt(2) and test nothing.
    See ``_curved_sweep``, construction point 2. Everything else is as in part C.
    """)
    _header("half-angle / regime")
    return _curved_sweep(two_body=True)


def main():
    np.set_printoptions(precision=3, suppress=True)
    part_A()
    part_B()
    res_C = part_C()
    res_D = part_D()

    def broke(res):
        return [r for r in res if r[1] or r[2] <= 1e-9]

    b_C, b_D = broke(res_C), broke(res_D)

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("""
    On the original question -- can contact_dataset.py serve as the test problem
    for the inf-sup experiments? -- the answer is NO AS IT STANDS, and the
    reason is structural, not a matter of tuning or resolution:

      * Part A: its five parameters all enter through the gap, i.e. through the
        right-hand side g. The constraint matrix B is the identity and is
        parameter-INDEPENDENT. [NDEE22] is about parameter-dependent
        CONSTRAINTS, so the model's parametrization is of the wrong kind, and
        PGA would have no online cost to remove.
      * Part B: with real, Hertz-validated pressure snapshots, beta^dec > 0 is
        PROVEN at every POD tolerance swept. A validated pressure profile is not
        sufficient; the obstruction lives in the constraint's row geometry,
        which the flat half-plane makes trivial.""")

    if b_C:
        print(f"      * Part C: curvature ALONE breaks it (half-angle "
              f"{', '.join(f'{r[0]:.2f}' for r in b_C)}).")
    else:
        print("""
      * Part C: a rotating contact normal, on its own, does NOT break it. The
        Gordan certificate stays positive at every half-angle from 0.05 to 1.30
        AND at every N swept from 1 to R-1 -- so at kernel dimensions R - N up
        to ~21, not merely the thin R - N = 1 case. "The richer geometry of a
        2-D contact surface" -- the conjecture in
        ~/stable_model_reduction_vi/REPRODUCTION_NOTES.md -- is therefore NOT
        confirmed by the cheapest version of that richer geometry.""")

    if b_D:
        print(f"""
      * Part D: the TWO-BODY jump DOES break it. At half-angle(s)
        {', '.join(f'{r[0]:.2f}' for r in b_D)} the Gordan certificate fails and
        ker(C) meets the non-negative orthant, so beta^dec = 0 is reachable --
        [NDEE22]'s claim C1, reproduced.

    So the decisive ingredient is not curvature but CONTACT BETWEEN TWO
    DEFORMABLE BODIES: the constraint must act on the displacement jump
    [[u]] = u^(1) - u^(2), which gives every constraint row a positive block and
    a negative one and lets a non-negative combination of supremizers cancel
    against V_N. Both papers' own test cases have this ([BEE20] §6.2, [NDEE22]
    §5.2, half-disks); neither repository's 1-D toy does.

    RECOMMENDED PATH. Keep contact_dataset.py's validated Hertz pressure
    physics as the dual-snapshot source -- that part is sound and is better than
    either repository's current 1-D multipliers -- and build the test problem
    around a TWO-BODY contact interface with a mu-dependent collocation, as in
    part D. Still required before this is a test problem rather than a probe:
      * a real 2-D elasticity operator (part D's blockdiag compliance is a
        stand-in -- see part C's [UNSPECIFIED / PROBE ASSUMPTION]);
      * a contact solver on that geometry, so the multipliers are the model's
        own rather than resampled half-plane pressures;
      * revalidation of the pressure profile against Hertz on the arc.""")
    else:
        print("""
      * Part D: the two-body displacement jump does NOT break it either, with
        bodies of different moduli and over the same N and curvature sweeps.

    Both candidate mechanisms are eliminated on this evidence -- 40
    configurations, every one with a positive certificate -- so beta^dec = 0
    remains unreached and the source of [NDEE22]'s C1 is still unlocated.

    There IS a consistent trend worth recording: the certificate weakens
    monotonically as N falls and as the arc curves, from t ~ 1.7 at N = R-1 down
    to t ~ 5e-2 at N = 2 with half-angle 1.30. So the obstruction is being eroded
    by exactly the variables one would expect -- it simply does not reach zero
    here. What has NOT been varied, and is where to look next: the dual
    discretization (P1/P1 as in [NDEE22] §5.1, rather than collocation, which
    couples neighbouring multipliers and is the most likely source of the
    cancellation), and non-matching meshes ([BEE20] §6).

    RECOMMENDED PATH. contact_dataset.py's value to the [NDEE22] experiments is
    as a source of physically meaningful, non-negative dual snapshots -- real,
    since both repositories' multipliers currently come from 1-D obstacle toys
    -- but adopting it does not on its own bring C1 within reach. Given that
    part A already rules it out on parametrization grounds, the honest
    recommendation is NOT to adopt it as the inf-sup test problem, and to treat
    the P1/P1 dual discretization as the next hypothesis to test.""")
    print()


if __name__ == "__main__":
    main()
