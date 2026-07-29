# `rb_vi_common` — shared machinery for two RB/variational-inequality papers

This package holds the algorithm implementations shared by two sibling
repositories, which implement a paper and its direct sequel:

| Tag | Paper | Repository |
|---|---|---|
| `[BEE20]` | A. Benaceur, A. Ern, V. Ehrlacher, *A reduced basis method for parametrized variational inequalities applied to contact mechanics*. IJNME 121(6):1170–1197, 2020. | [`~/rb_contact_cpg`](../rb_contact_cpg) |
| `[NDEE22]` | I. Niakh, G. Drouet, V. Ehrlacher, A. Ern, *Stable model reduction for linear variational inequalities with parameter-dependent constraints*. ESAIM: M2AN (2022), [doi:10.1051/m2an/2022077](https://doi.org/10.1051/m2an/2022077). | [`~/stable_model_reduction_vi`](../stable_model_reduction_vi) |

`[NDEE22]` cites `[BEE20]` as its reference **[9]**.

**No official code exists for either paper.** Both say the authors used FreeFem++
plus Python with `cvxopt`; neither links a repository. Nothing here derives from
the authors' code.

## Read this before touching a citation

Consolidation created one hazard that did not exist when each repository held a
single paper: **the two papers' numbering collides.**

- `[BEE20]` Eq. (57) and `[NDEE22]` Eq. (57) are unrelated.
- Both papers have an **Algorithm 1** and an **Algorithm 2**, denoting different
  algorithms:

  | | Algorithm 1 | Algorithm 2 |
  |---|---|---|
  | `[BEE20]` | the online stage | **CPG** |
  | `[NDEE22]` | **PGA** | **mCPG** |

Every citation in this package therefore carries a paper tag. A bare equation or
algorithm number anywhere in `rb_vi_common/` is a bug — with the one documented
exception of `reduction.py` and `pga.py`, which are `[NDEE22]` end to end and say
so in a module-level convention block (algorithm references there are tagged
anyway, since those are the collisions that actually mislead).

### `[BEE20]` numbering is preprint-v2 numbering

The `[BEE20]` transcription was made against the HAL preprint
[`hal-02081485v2`](https://hal.science/hal-02081485v2/file/contact.pdf)
(submitted 27 Mar 2019) — **not** the published IJNME article, and **not** v3
(24 Sep 2019). Numbered equations and algorithm content may differ between
versions. Every `[BEE20]` number means *v2 numbering*.

### `[BEE20]` carries a PDF-extraction hazard

The v2 PDF uses a shifted Type-1 font encoding: delimiters map onto letters
(`p`→`(`, `q`→`)`, `t`→`{`, `u`→`}`, `r`→`[`, `s`→`]`) and operators arrive as
`(cid:NN)` codes. Two collisions are **not** mechanically resolvable — `u` is
both a closing brace and the primal displacement, `P` is both `∈` and the
parameter set `𝒫`. Equations quoted in `[BEE20]`-tagged docstrings were read by
eye against surrounding prose. **Verify any `[BEE20]` equation you rely on
against the PDF.** `[NDEE22]` has no such hazard; its equations were read from
the rendered page.

## Layout

```
rb_vi_common/
  cone_projection.py   Π_{K⁺} via NNLS; the ‖·‖_Λ norm       [BEE20] §5, Eq. (57)
  cone_greedy.py       CPG ×2, mCPG, e_orth                  [BEE20] Alg. 2;
                                                             [NDEE22] Alg. 2, Rmk 4.3, Eq. (41)
  reduction.py         whitening, POD, supremizers, σ_S,
                       c_S, inf-sup constants                [NDEE22] §2–§3
  pga.py               PGA                                   [NDEE22] Alg. 1
tests/
  test_equivalence.py  cross-paper consistency checks
```

### Why there are two CPGs

`cone_projected_greedy` (`[BEE20]` Alg. 2) and `cpg` (`[NDEE22]` Rmk 4.3) are the
same algorithm read out of two papers. They differ in four conventions —
raw vs normalized generators, absolute (`[BEE20]` Eq. 58) vs relative
(`[NDEE22]` Eq. 13) tolerance, argmax over all vs over unselected parameters, and
an explicit `‖·‖_Λ` Gram matrix vs pre-whitened Euclidean coordinates.

Collapsing them into one function would mean picking one paper's convention and
silently attributing it to the other, which is exactly the anchoring these
implementations exist to preserve. Instead, both are kept and their equivalence
is **tested**: `test_two_cpg_transcriptions_agree` checks that at matched
tolerance they select the same parameters in the same order and generate the
same cone as a set, with generator matrices differing by exactly the snapshot
normalization.

`mcpg` is the primary algorithm — `[NDEE22]` §4 supersedes plain CPG with it.
`cpg` is retained as the Remark 4.3 baseline it is defined to be.

### What is deliberately *not* shared

The test problems and drivers stay in their own repositories:
`rb_contact_cpg/src/{toy_problem,contact_dataset}.py`,
`stable_model_reduction_vi/src/hf_model.py`, and both `run_*.py`. Those are
per-repository substitutes for the papers' own FEM test cases, they reproduce no
number in either paper, and merging them would blur which paper a given result
speaks to. `nmf_baseline.py` also stays in `rb_contact_cpg`: it is `[BEE20]` §6.4
and `[NDEE22]` does not use it.

## Use

Both repositories locate this package through a small shim, `src/_shared_path.py`,
which assumes the default sibling layout:

```
~/rb_vi_shared/          <- this package
~/rb_contact_cpg/
~/stable_model_reduction_vi/
```

If you move it, set `RB_VI_SHARED` to its path. Nothing needs installing and the
run instructions in each repository are unchanged.

```bash
pip install -r requirements.txt
python3 tests/test_equivalence.py
```

`numpy` + `scipy` only.
