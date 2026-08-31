# `repos/` — the four merged source repositories

Preserved as they were before the merge. Nothing in them was edited to make the benchmark
work: the benchmark absorbs their convention differences in `bench/adapters/` instead, and
the duplicate CPG/mCPG transcriptions are retained deliberately because they are the
*input* to the cross-implementation agreement metric, not an accident to be cleaned up.

| directory | paper | what `bench` takes from it |
|---|---|---|
| `rb_vi_shared/` | both | the shared algorithm library `rb_vi_common` — CPG, mCPG, cone projection, PGA |
| `rb_contact_cpg/` | `[BEE20]` | the `toy_bee20` obstacle problem and the `hertz_pressure` dataset |
| `stable_model_reduction_vi/` | `[NDEE22]` | the `obstacle_ndee22` problem — the only parameter-dependent `B(mu)` |
| `greedy_algos/` | — | CPG / mCPG / ADG as `greedy`, plus `fem_lambda`, `membrane_2d`, `hertz_2d` |

## This directory must stay flat

`rb_contact_cpg` and `stable_model_reduction_vi` each ship a `src/_shared_path.py` that
locates the shared library at `parents[2] / "rb_vi_shared"` — that is, `src/` → the repo →
**the repo's parent**. Because all four sit under this one parent, that resolution keeps
working unmodified, which is what lets the original entrypoints run untouched:

```bash
cd repos/stable_model_reduction_vi/src && ../../../.venv/bin/python run_experiments.py
```

Nesting or renaming any of the four breaks that walk. `_shared_path.py` checks
`RB_VI_SHARED` **first** and only then falls back to `parents[2]`, so a different layout is
possible — but every standalone invocation would then need that variable set.
`bench/_paths.py` sets it (via `os.environ.setdefault`) for anything run through `bench`,
which is why the harness would survive a move that the standalone entrypoints would not.

## Where the data is, and why it is not all in one place

Raw inputs are **not versioned** — each repo's own `.gitignore` says so, and they are large.

Data that belongs to a source repository stays inside it, because that repository's own
pipelines address it by a relative path and moving it would break them:

| file | size | read by |
|---|---|---|
| `greedy_algos/data/FEM_SOLS.zip` | 3.5 M | `greedy.datasets.lambda_snapshots` → `fem_lambda` |
| `greedy_algos/data/contact_forces/` | — | the 2-D `membrane_2d` / `hertz_2d` models |
| `greedy_algos/data/physics_data.txt` | 19 M | `greedy.datasets.physics_dataset` (a symlink — see below) |
| `rb_contact_cpg/data/contact_pressures.npz` | 1.3 M | `hertz_pressure` |

Data that belongs to **no** source repository lives in the monorepo's own `../data/`. That
is currently the pellet-cladding archive, which `bench.datasets` reads directly — see
`data/README.md`.

`greedy_algos/data/physics_data.txt` is a **symlink** into that archive. The two files were
byte-identical (same MD5), 19 M each, and keeping both meant the same matrix on disk twice
under two names with different parameter labels attached. The symlink keeps
`greedy.datasets.physics_dataset` working while there is only one copy of the bytes. If you
need the file standalone, `cp --dereference` it.

## Generated output inside these repos

Each repo has a `results/` that its own pipelines write and its own `.gitignore` excludes.
Only one path in there is an *input* to this benchmark:

* `greedy_algos/results/lambda/dataset/lambda_dataset.npz` — built by
  `python -m greedy.datasets.lambda_snapshots`, consumed by `bench.datasets._fem_lambda`.

Everything else under those `results/` directories is regenerable and is not read by
`bench`. `greedy_algos/results/physics/` in particular is now dead weight for the
benchmark: `bench.datasets._physics` reads the pellet-cladding archive directly and no
longer opens `physics_dataset.npz`.
