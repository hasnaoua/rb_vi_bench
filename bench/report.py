"""Turn ``results/grid.csv`` into readable tables and figures.

Deliberately thin: the CSV is the artefact, this is a view of it. Every table states
which mode it is reading, because a precision number at matched tolerance and one at
matched cardinality answer different questions and are not comparable
(see ``metrics.precision``).

Skipped cells are printed, not dropped. A method missing from a table because it was
never measured looks identical to one that scored badly, unless the skips are shown.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from . import _paths
from .tabular import fmt as _fmt, num as _num, read_rows as _load

RESULTS = _paths.ROOT / "results"


def _table(title: str, headers: list[str], rows: list[list[str]], note: str = "") -> str:
    if not rows:
        return f"\n{title}\n  (no rows)\n"
    widths = [max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    out = [f"\n{title}", "-" * len(line), line, "-" * len(line)]
    out += ["  ".join(r[i].ljust(widths[i]) for i in range(len(headers))) for r in rows]
    if note:
        out.append(f"\n  {note}")
    return "\n".join(out) + "\n"


def tolerance_table(rows: list[dict], dataset: str, delta: float) -> str:
    sel = [r for r in rows
           if r.get("dataset") == dataset and r.get("mode") == "tolerance"
           and not r.get("skip_reason") and _num(r, "delta") == delta]
    body = [[
        r["method_label"], r["paper_tag"],
        _fmt(_num(r, "R"), ".0f"),
        _fmt(_num(r, "test_max_rel_err")),
        _fmt(_num(r, "test_max_rel_err_persnap")),
        _fmt(_num(r, "gram_cond"), ".3g"),
        _fmt(_num(r, "e_orth_mean"), ".3f"),
        _fmt(_num(r, "calls_total"), ".0f"),
        _fmt(_num(r, "fit_seconds"), ".3f"),
    ] for r in sorted(sel, key=lambda r: _num(r, "R"))]
    return _table(
        f"{dataset} -- matched TOLERANCE delta={delta}  (R is the result, not an input)",
        ["method", "paper", "R", "test_max_err", "per_snap", "gram_cond", "e_orth",
         "solves", "secs"],
        body,
        "test_max_err divides by the LARGEST snapshot norm; per_snap divides each "
        "snapshot by its own -- the quantity ADG's tolerance bounds. They diverge where "
        "snapshot magnitudes spread.",
    )


def cardinality_table(rows: list[dict], dataset: str, R: int) -> str:
    sel = [r for r in rows
           if r.get("dataset") == dataset and r.get("mode") == "cardinality"
           and not r.get("skip_reason") and _num(r, "R_requested") == R]
    body = [[
        r["method_label"], r["paper_tag"],
        _fmt(_num(r, "train_max_rel_err")),
        _fmt(_num(r, "test_max_rel_err")),
        _fmt(_num(r, "test_max_rel_err_persnap")),
        _fmt(_num(r, "nn_max_violation"), ".2e"),
        _fmt(_num(r, "gram_cond"), ".3g"),
        _fmt(_num(r, "fit_seconds"), ".3f"),
    ] for r in sorted(sel, key=lambda r: _num(r, "test_max_rel_err"))]
    return _table(
        f"{dataset} -- matched CARDINALITY R={R}  (the only mode NMF and the orthant enter)",
        ["method", "paper", "train_max", "test_max", "test_per_snap", "nn_violation",
         "gram_cond", "secs"],
        body,
        "test_per_snap divides each snapshot by its OWN norm -- the quantity ADG's "
        "tolerance bounds -- while test_max divides by the largest. The orthant is a "
        "reference (widest admissible cone), not a competitor.",
    )


def infsup_table(rows: list[dict], dataset: str, delta: float) -> str:
    sel = [r for r in rows
           if r.get("dataset") == dataset and r.get("mode") == "tolerance"
           and not r.get("skip_reason") and _num(r, "delta") == delta
           and not math.isnan(_num(r, "beta_dec_min"))]
    if not sel:
        return ""
    body = [[
        r["method_label"],
        _fmt(_num(r, "beta_hf_min"), ".3e"),
        _fmt(_num(r, "beta_dec_min"), ".3e"),
        _fmt(_num(r, "beta_off_min"), ".3e"),
        _fmt(_num(r, "sigma_S_median"), ".3f"),
        _fmt(_num(r, "c_S_median"), ".3f"),
        _fmt(_num(r, "gordan_t_min"), ".3e"),
        _fmt(_num(r, "kernel_meets_orthant"), ".0f"),
    ] for r in sel]
    return _table(
        f"{dataset} -- inf-sup constants, delta={delta}",
        ["method", "beta_HF", "beta^dec", "beta^off", "sigma_S", "c_S", "gordan_t", "ker_hits"],
        body,
        "beta values are UPPER bounds (non-convex minimization over a cone). "
        "gordan_t > 0 PROVES beta^dec > 0, i.e. [NDEE22] claim C1 does not occur here.",
    )


def agreement_table(rows: list[dict]) -> str:
    body = [[
        r.get("dataset", ""), r.get("pair", ""),
        _fmt(_num(r, "R_a"), ".0f"), _fmt(_num(r, "R_b"), ".0f"),
        _fmt(_num(r, "set_max_diff"), ".2e"),
        r.get("verdict", "") or r.get("skip_reason", ""),
    ] for r in rows]
    return _table(
        "Cross-implementation agreement (matched tolerance)",
        ["dataset", "pair", "R_a", "R_b", "set_diff", "verdict"],
        body,
        "'divergent' between two CPGs is a bug; between the two mCPGs it is a finding "
        "about the [UNSPECIFIED] line-9 solver.",
    )


def skip_summary(rows: list[dict]) -> str:
    grouped: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.get("skip_reason"):
            grouped[r["skip_reason"]].add(f"{r.get('dataset','?')}/{r.get('method','?')}")
    body = [[reason, str(len(cells)), ", ".join(sorted(cells)[:4]) + ("..." if len(cells) > 4 else "")]
            for reason, cells in sorted(grouped.items(), key=lambda kv: -len(kv[1]))]
    return _table("Skipped cells (never silently dropped)", ["reason", "n", "examples"], body)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="render rb_vi_bench results")
    p.add_argument("--results", type=Path, default=RESULTS)
    p.add_argument("--out", type=Path, default=None, help="write to a file as well as stdout")
    args = p.parse_args(argv)

    grid = _load(args.results / "grid.csv")
    try:
        agree = _load(args.results / "agreement.csv")
    except FileNotFoundError:
        agree = []

    datasets = sorted({r["dataset"] for r in grid if r.get("dataset") and not r.get("skip_reason")})
    deltas = sorted({_num(r, "delta") for r in grid
                     if r.get("mode") == "tolerance" and not math.isnan(_num(r, "delta"))})
    cards = sorted({int(_num(r, "R_requested")) for r in grid
                    if r.get("mode") == "cardinality" and not math.isnan(_num(r, "R_requested"))})

    chunks = ["=" * 78, "rb_vi_bench results", "=" * 78]
    for ds in datasets:
        for d in deltas:
            chunks.append(tolerance_table(grid, ds, d))
            chunks.append(infsup_table(grid, ds, d))
        for R in cards:
            chunks.append(cardinality_table(grid, ds, R))
    if agree:
        chunks.append(agreement_table(agree))
    chunks.append(skip_summary(grid))

    text = "\n".join(c for c in chunks if c)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
