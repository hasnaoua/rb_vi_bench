"""Where figures go.

Everything is grouped **per dataset**, because that is the unit you actually look at:
comparing methods only makes sense within one dataset, and the previous flat layout
scattered a dataset's figures across three places (``<ds>/`` for metric splits,
``cardinality_<ds>.png`` at the root, ``reconstruction/<ds>/`` in a third tree).

::

    results/figures/
    |-- _overview/
    |   `-- precision_all_datasets.png     the only cross-dataset figure
    `-- <dataset>/
        |-- panel.png                      four metrics in one grid
        |-- metrics/
        |   |-- precision.png
        |   |-- conditioning.png
        |   |-- orthogonality.png
        |   `-- offline_cost.png
        |-- decrement/
        |   |-- vs_cardinality.png         e(n+1) - e(n) against R
        |   `-- vs_tolerance.png           e(n+1) - e(n) against epsilon
        `-- reconstruction/
            |-- all_methods.png
            `-- <method>/
                |-- best.png
                `-- worst.png

``_overview`` is underscore-prefixed so it sorts above the dataset directories and is
never mistaken for one.
"""

from __future__ import annotations

from pathlib import Path


def slug(dataset: str) -> str:
    """Filesystem-safe form of a dataset name.

    Subsampled datasets carry their cap in the name (``hertz_pressure[n<=200]``), which
    is worth keeping visible -- the numbers do depend on it -- but the brackets and
    ``<=`` are awkward in a path. Dataset names are prose now ("Half-disks of Hertz"),
    so spaces and parentheses have to be handled too.
    """
    out = (dataset.replace("[", "_").replace("]", "")
           .replace("<", "").replace("=", "")
           .replace("(", "").replace(")", "")
           .replace(" ", "_").replace("-", "-"))
    # Collapse the doubled underscores a name like "X (pressure)" would otherwise leave.
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def dataset_dir(out: Path, dataset: str) -> Path:
    return out / slug(dataset)


def metrics_dir(out: Path, dataset: str) -> Path:
    return dataset_dir(out, dataset) / "metrics"


def decrement_dir(out: Path, dataset: str) -> Path:
    return dataset_dir(out, dataset) / "decrement"


def reconstruction_dir(out: Path, dataset: str) -> Path:
    return dataset_dir(out, dataset) / "reconstruction"


def method_dir(out: Path, dataset: str, method: str) -> Path:
    return reconstruction_dir(out, dataset) / method


def overview_dir(out: Path) -> Path:
    return out / "_overview"


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
