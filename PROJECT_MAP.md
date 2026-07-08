# Greedy Algorithms Project Map

This repo now keeps source scripts at the root and regenerated artifacts under `results/`.

## Core Methods

- `CPG.py`: CPG implementation.
- `modified_angle_greedy.py`: Angular Defect Greedy implementation.
  - Main class: `AngularDefectGreedy`.
- `reduction_common.py`: shared NNLS, residual, and plotting helpers.

## Lambda / FEM_SOLS Workflow

- `merge_lambda_snapshots.py`: builds `results/lambda/dataset/lambda_dataset.npz`.
- `lambda_cpg_pipeline.py`: runs CPG and writes `results/lambda/cpg/`.
- `angle_greedy_pipeline.py`: runs ADG and writes `results/lambda/adg/`.
- `compare_reduction_methods.py`: compares CPG and ADG in `results/lambda/comparison/`.
- `component_sweep_comparison.py`: epsilon-driven prefix sweep in `results/lambda/component_sweep/`.
- `plot_contact_force_profiles.py` and `plot_single_lambda_snapshot.py`: extra lambda plots in `results/lambda/plots/`.
- `merge_primal_snapshots.py`: optional primal/res snapshot extraction in `results/lambda/primal_dataset/`.

## Physics Contact Workflow

- `physics_dataset.py`: builds `results/physics/dataset/physics_dataset.npz`.
- `physics_reduction_experiment.py`: runs CPG and ADG physics reduction in `results/physics/reduction/`.
- `physics_publication_visualization.py`: journal-style physics figures in `results/physics/publication/`.
- `contact_force_dashboard.py`: interactive raw physics dashboard in `results/physics/dashboard/`.

## Naming

- Public method name: `Angular Defect Greedy`.
- Short figure/table label: `ADG`.
- Physics result files use the short `adg_*` prefix for ADG arrays and figures.

## Validation Protocol

- Physics reduction now uses the whole physics dataset for CPG/ADG basis construction and residual evaluation; no train/test split is applied in `physics_reduction_experiment.py`.
- Physics metrics and reports summarize full-dataset residuals only.
- FEM_SOLS/lambda uses the paper parameter grid by default: `{0.15 + 0.01 i | 0 <= i <= 30}` m. In the local archive this is matched as `rad - 0.65`, so the training folders are `rad=0.8` through `rad=1.1`.
- Lambda CPG and ADG bases are fitted on the configured training radii and all lambda snapshots are projected onto the frozen basis by NNLS.
- Lambda reports and metrics include separate train and test residual columns.

## Fresh Rebuild Order

```bash
python3 merge_lambda_snapshots.py
python3 merge_primal_snapshots.py
python3 lambda_cpg_pipeline.py
python3 angle_greedy_pipeline.py
python3 compare_reduction_methods.py
python3 component_sweep_comparison.py
python3 physics_dataset.py
python3 physics_reduction_experiment.py
python3 physics_publication_visualization.py --snapshot-index 2 47 84 86 89
```
