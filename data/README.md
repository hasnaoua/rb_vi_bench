# `data/` — raw inputs that belong to no source repository

Everything here is **raw input, not versioned**: large, regenerable-by-download rather than
by computation, and excluded in `.gitignore` exactly as each vendored repo excludes its own
`data/`. Cloning this repository does not give you these files; unpack or fetch them.

The split is by *ownership*, not by size or type. A dataset that a source repository's own
pipelines address by a relative path stays inside that repository, because moving it would
break the repository this monorepo promises not to edit — see `../repos/README.md` for that
table. What lands here is what the **benchmark itself** reads and no vendored pipeline
knows about.

## `3D_cladding_split/` — the pellet-cladding contact archive

Read by `bench.datasets._physics` (the `physics` key, reported as *3D Pellet-Cladding*).
Unpack `3D_cladding_split.zip` beside it, or point `RB_VI_CLADDING_SPLIT` at a copy held
elsewhere.

Eight files, 7676 contact nodes on a 76 × 101 quarter-cylinder interface:

| file | shape | what it is |
|---|---|---|
| `Contact_forces_data.txt` | 7676 × 99 | the full snapshot matrix, nodes × parameters |
| `Params_set.txt` | 99 | imposed axial displacement, mm |
| `Training_indices_set.txt` / `Training_params_set.txt` | 50 | the training half (even columns) |
| `Ptest_indices_set.txt` / `Ptest_params_set.txt` | 49 | the held-out half (odd columns) |
| `Contact_forces_train-data.txt` / `-test-data.txt` | 7676 × 50, × 49 | those columns, pre-selected |

The last two and the two `*_params_set` files are **redundant** — they are column selections
the indices already determine. They are kept and *verified* rather than deleted:
`_load_cladding_split` checks each against the full matrix on every load, because a split
that disagrees with the matrix it indexes is the one failure mode that produces a
plausible-looking test error rather than an error.

**Why this archive and not `repos/greedy_algos/data/physics_data.txt`.** The snapshot matrix
is byte-identical between them (same MD5) — what this archive adds is the two things that
file never carried: the 99 parameter values and the train/test partition. Without them
`greedy.datasets.physics_dataset` had to *guess* the parameter axis from a 96-point grid in
the problem statement, and guessed wrong by one grid step throughout. See
`bench.datasets.PHYSICS_PARAMETER_GRID`. `physics_data.txt` is now a symlink into this
directory so the bytes exist once.

The parameter grid is 99 points: 0.16–0.30 mm in steps of 0.005 (29 points), then
0.31–1.00 mm in steps of 0.01 (70). The first five are no-contact states — the imposed
displacement has not yet closed the 0.05 mm initial gap — and `Dataset` drops them as
numerically zero, leaving 94 snapshots split 47 / 47.
