# Greedy Algorithms for Model-Order Reduction

Greedy, cone-projected reduced-basis methods for contact / variational-inequality
problems (CPG, mCPG, Angular Defect Greedy), with lambda/FEM, physics-contact, and
synthetic experiment pipelines.

## Layout

```
greedy_algos/
├── pyproject.toml            installable package (src/ layout)
├── README.md
├── data/         (gitignored) raw inputs: physics_data.txt, FEM_SOLS.zip
├── docs/                      reference paper + LaTeX summary
├── results/      (gitignored) regenerated experiment artifacts
├── tests/
└── src/greedy/
    ├── core/                  reduction algorithms (numpy/scipy only)
    │   ├── reduction_common.py  shared NNLS/QP solvers, residuals, plot helpers
    │   ├── cpg.py               CPG            (class CPG)
    │   ├── mcpg.py              modified CPG   (class mCPG)
    │   └── angle_greedy.py      Angular Defect Greedy (class AngularDefectGreedy)
    ├── datasets/              dataset builders + loaders
    │   ├── lambda_snapshots.py  FEM_SOLS -> results/lambda/dataset/lambda_dataset.npz
    │   ├── primal_snapshots.py  optional primal/res snapshot extraction
    │   └── physics_dataset.py   physics_data.txt -> results/physics/dataset/physics_dataset.npz
    ├── pipelines/             runnable experiment drivers (thin CLIs)
    │   ├── lambda_cpg.py        CPG   -> results/lambda/cpg/
    │   ├── lambda_adg.py        ADG   -> results/lambda/adg/
    │   ├── lambda_mcpg.py       mCPG  -> results/lambda/mcpg/
    │   ├── lambda_compare.py    CPG/ADG/mCPG comparison -> results/lambda/comparison/
    │   ├── component_sweep.py   epsilon prefix sweep -> results/lambda/component_sweep/
    │   ├── physics_reduction.py physics CPG/ADG/mCPG -> results/physics/reduction/
    │   └── synthetic.py         Gaussian-bump synthetic sweep -> results/synthetic/
    ├── viz/                   plots, dashboards, publication figures
    │   ├── contact_force_profiles.py, single_lambda_snapshot.py
    │   ├── publication.py       journal-style physics figures
    │   ├── contact_dashboard.py interactive raw physics dashboard
    │   ├── cylinder_3d.py, diagnose_snapshots.py
    └── synthetic_data/        Gaussian-bump data generator
```

### Shared-logic hubs

Two pipeline modules also export reusable fitters imported by other pipelines
(kept in place to avoid risky cross-file surgery):

- `pipelines.lambda_cpg` — `project_snapshots_onto_basis` and lambda dataset helpers.
- `pipelines.component_sweep` — `fit_cpg_fixed_components`, `fit_angle_fixed_components`,
  `fit_mcpg_fixed_components`, reused by `pipelines.physics_reduction`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # add [qp] for the cvxopt QP solver, [dev] for pytest
```

## Running

All pipelines resolve `data/` and `results/` **relative to the current directory**,
so run them from the repository root as modules:

```bash
python -m greedy.datasets.lambda_snapshots
python -m greedy.datasets.primal_snapshots
python -m greedy.pipelines.lambda_cpg
python -m greedy.pipelines.lambda_adg
python -m greedy.pipelines.lambda_mcpg
python -m greedy.pipelines.lambda_compare
python -m greedy.pipelines.component_sweep
python -m greedy.datasets.physics_dataset
python -m greedy.pipelines.physics_reduction
python -m greedy.viz.publication --snapshot-index 2 47 84 86 89
python -m greedy.pipelines.synthetic
```

`data/physics_data.txt` and `data/FEM_SOLS.zip` must be present (they are gitignored).

## Naming

- Public method name: **Angular Defect Greedy**; short label **ADG** (`adg_*` result files).
- **Modified Cone-Projected Greedy** is labeled **mCPG** everywhere (class, CLI alias, `mcpg_*` files).

## Validation protocol

- Physics reduction uses the whole physics dataset for CPG/ADG basis construction and
  residual evaluation; no train/test split. Reports summarize full-dataset residuals.
- FEM_SOLS/lambda uses the paper parameter grid `{0.15 + 0.01 i | 0 <= i <= 30}` m,
  matched locally as `rad - 0.65` (training folders `rad=0.8` .. `rad=1.1`).
- Lambda CPG/ADG bases are fitted on training radii; all snapshots are projected onto
  the frozen basis by NNLS. Reports include separate train and test residual columns.

## Testing

```bash
python -m pytest
```
