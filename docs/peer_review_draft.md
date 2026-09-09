# Peer-review working draft — REVIEW-RBVIBENCH-001

> Private working document. Human review, policy checks, and factual verification are required. Do not submit this scaffold with unresolved placeholders. Do not make or announce an editorial decision.

## Intake record

- Reviewer capacity: `author_requested_reader`
- Peer-review model: `open`
- Declared processing plan: `local_deterministic_tools`
- Manuscript text is not embedded by the generator.
- Reconfirm conflicts, competence limits, tool use, confidentiality, and deletion or retention obligations before submission.

# Comments to authors

## Evidence-bounded summary

The submission is a benchmark of dual-cone construction methods for reduced-basis
approximation of contact-mechanics variational inequalities, comparing CPG [BEE20],
mCPG [NDEE22], an angular-defect greedy (ADG), NMF and reference controls across three
datasets, plus a new one-parameter family interpolating CPG and ADG. Evidence available
for review: source tree, three dataset loaders with recorded splits, a 960-cell sweep
(`results/gamma_study/grid.csv`), a 26-page report, and 165 passing tests. No preregistration,
analysis plan, validation split, or environment lockfile was available for review.

The design is unusually disciplined in several respects that most benchmark papers get wrong.
The substantive gaps are the absence of any uncertainty on reported quantities, the treatment of
strongly dependent (γ, R) cells as if they were counts of independent observations, and
prescriptive phrasing that outruns a three-dataset measurement.

## Strengths

- **Matched-cardinality comparison is enforced, not merely claimed.** Rankings are read at
  fixed R rather than aggregated, and Caveat 7 documents a concrete case where a median over
  R reversed a ranking. This is the correct discipline for a sweep of this shape.
- **Negative and null results are reported at equal weight.** The Half-disks null (γ inert
  below the rank cliff), the adverse cells, and the retraction of an earlier median-based
  conclusion are all in the text rather than omitted.
- **Failure modes are recorded per cell rather than dropped.** `skip_reason` is retained and
  grouped, missing values are never imputed, and the numerical ceiling bounds an axis without
  altering a datum.
- **A validity boundary is derived independently of the methods.** The numerical-rank cliff is
  computed from the snapshot spectrum before any method runs, and results past it are excluded
  from interpretation rather than reported and hedged.

## Major comments

### Major comment M1

**Prescriptive language in Sec. 7.5 outruns what a three-dataset measurement supports**

> Revised after author response. This comment originally asked for a validation split. That
> request is withdrawn: see the note below.

- Location: Sec. 7.5 ("Which γ to use", "recommended"), Sec. 7.2
- Observation: The γ study measures, on three datasets with the source papers' own splits, what
  each family member does. Sec. 7.5 then states a per-dataset recommendation in prescriptive
  form.
- Evidence or criterion: `gram_conditioning(result.generators)` uses generators only, and
  generators are selected from `dataset.train()`; the conditioning half of the trade-off
  therefore has no test dependence at all. Only `test_max_rel_err` touches the held-out set,
  and it is the approximation error the benchmark exists to measure.
- Why it matters: as a *measurement* on these datasets the analysis needs no validation split,
  and the withdrawn request would have broken comparability with [BEE20] and [NDEE22], whose
  partitions are used verbatim. What does not follow from three datasets is a recommendation
  addressed to a reader with a fourth problem, which is how "Which γ to use" reads.
- Requested action: Retitle Sec. 7.5 to something measured rather than prescriptive (e.g.
  "What γ does on each dataset") and phrase the three bullets as findings for these datasets at
  these cardinalities. No new computation and no re-partitioning.

WITHDRAWN-REQUEST NOTE: the original request for a fit/validation split inside the training set
is withdrawn. The methods are deterministic, the conditioning metric never sees the test set,
the candidate set is five fixed values rather than a searched space, and two of the three splits
are fixed by the source publications. Selection optimism in this setting is bounded by the
stability of a five-way ranking on the held-out parameters, not by a fitted-model variance term,
and it is not the dominant threat to the conclusions. The residual issue is claim scope, which
is what this comment now covers and which overlaps minor comment m4.

### Major comment M2

**no uncertainty is attached to any reported quantity**

- Location: all result tables and figures; `results.effect_sizes_uncertainty` in the audit
- Observation: Every number is a single run. The greedy methods are deterministic, so this is
  defensible for them, but three sources of variability are unquantified: NMF's initialization
  (three seeds exist as separate methods but their spread is not propagated into any comparison),
  the single train/test partition, and `--subsample 200`.
- Evidence or criterion: `nmf_s0/s1/s2` exist as separate methods, but no comparison
  propagates their spread; `--subsample 200` is applied without a repetition loop.
- Why it matters: Differences such as membrane's 1.25× vs 1.47× conditioning gain, or the
  4.7% error improvement, are reported as findings without any evidence that they exceed
  partition-to-partition variation. On a 3-dataset study these are the differences that carry
  the recommendation.
- Requested action: Report the across-seed spread for NMF wherever NMF is compared, and add a
  small resampling check — e.g. 5–10 alternative train/test partitions on membrane and
  Pellet-Cladding — reporting the range of the headline ratios. If repartitioning is impossible
  because the splits are the source papers' own, state that explicitly and confine the
  affected claims to the given partition.

### Major comment M3

**dependence between cells is not acknowledged where cells are counted**

- Location: Sec. 7.2 ("19 cells", "87 of 156", "14 of the 17 cardinalities")
- Observation: Several conclusions are stated as counts over (γ, R) cells.
- Evidence or criterion: Cells sharing a dataset share one split, and cells at adjacent R
  share nested cones by construction (K_R ⊆ K_{R+1}), so they are strongly dependent.
  The R = 14–19 band is one contiguous region reported as 19 cells.
- Why it matters: A count such as "87 of 156 are genuine trades" reads as a frequency
  estimate but describes a handful of contiguous R-regions.
- Requested action: Keep the counts but describe them as extents of contiguous R-ranges
  rather than as proportions, or state explicitly that cells are not independent.

### Major comment M4

**the analysis was not prespecified, and multiplicity is uncontrolled**

- Location: Sec. 7 generally; `analysis.prespecification`, `analysis.multiplicity`
- Observation: The γ grid, the fixed cardinalities, the exchange-rate metric and the choice
  of which metrics to foreground were all fixed during analysis, after results were seen. Caveat 9
  partially discloses this for the R choice.
- Why it matters: With ~600 comparable cells and a freely chosen summary statistic, the
  probability of finding *some* favourable structure is high. This does not invalidate the
  descriptive findings, but it does mean the study is hypothesis-generating rather than confirmatory.
- Evidence or criterion: Caveat 9 discloses post hoc choice for R only; the γ grid, the
  exchange-rate definition and the metric emphasis carry no equivalent disclosure.
- Requested action: Add one sentence in Sec. 7 or the caveats stating that Sec. 7 is
  exploratory and its recommendations are hypotheses for confirmation on held-out datasets. This
  is a labelling fix, not new work.

## Minor comments

### Minor comment m1

**endpoint equivalence is verified at a single cardinality**

- Location: Sec. 2.4, "endpoints are reproduced exactly"
- Observation: The γ=0 ≡ CPG and γ=1 ≡ adg_k0 equalities are checked at R = 12 only.
- Evidence or criterion: the equivalence check compares `selected_indices` at R = 12.
- Why it matters: a reader will take "exactly" as holding for all R; the verification
  performed is narrower than the sentence.
- Requested action: Either state "at R = 12" in the text, or run the check across R and say so.
  The claim is almost certainly true generally; the reporting should match what was verified.

### Minor comment m2

**the σ^γ bound is stated more broadly than it was tested**

- Location: Sec. 2.4
- Observation: The bound constrains the training residual. It is reported alongside a test
  violation (membrane, R = 8, γ = 0.25), which is correct, but the surrounding text can be read
  as a bound on held-out error.
- Evidence or criterion: 443/443 training cells satisfy the bound, 442/443 test cells.
- Why it matters: a bound that holds on training residuals but is read as a
  generalization bound would misstate what the family guarantees.
- Requested action: Make the train/test distinction explicit in the sentence that states the bound.

### Minor comment m3

**no environment or version record accompanies the results**

- Location: Reproduction section
- Observation: CLI commands are given, but no library versions, lockfile, or commit hash is
  recorded with `grid.csv` or the manifest. NNLS/`lsq_linear` behaviour and condition numbers near
  the rank cliff are SciPy-version sensitive.
- Evidence or criterion: `manifest.json` records datasets and parameters but no versions.
- Why it matters: condition numbers near the rank cliff and NNLS active-set outcomes are
  library-version sensitive, so the numbers may not reproduce on another machine.
- Requested action: Record SciPy/NumPy versions and a commit hash in `manifest.json`, and cite
  them in the Reproduction section.

### Minor comment m4

**"n = 3 datasets" is doing more inferential work than stated**

- Location: Sec. 7.1 and the closing paragraph of Sec. 7.5
- Observation: The claim that the norm spread σ predicts whether γ can matter is consistent
  with all three datasets, but three points with σ = 1.09, 1.39, 186.85 cannot distinguish
  "σ predicts responsiveness" from "one dataset differs from two others in many ways at once".
- Evidence or criterion: σ = 1.09, 1.39, 186.85 for the three datasets.
- Why it matters: with one high-σ dataset the rule cannot be separated from any other
  property that distinguishes Pellet-Cladding from the other two.
- Requested action: Soften to a consistency statement, or add datasets with intermediate σ.

## Methods, statistics, and reproducibility

Local audit (`audit_statistics_reproducibility.py`): 10 verified_present, 8 partly_documented,
1 missing, 3 not_applicable; gaps concentrated in design (independence, precision rationale),
analysis (prespecification, multiplicity, clustering), results (uncertainty), reproducibility
(environment, versions), interpretation (claim–evidence). Claim–evidence matrix
(`validate_claim_evidence.py`): 9 claims, status VALID_WITH_ALIGNMENT_GAPS, 1 unsupported (C6,
the γ recommendation) and 3 requiring resolution (C3, C4, C9). These are completeness and
consistency audits, not a reanalysis and not a quality score.

## Ethics, transparency, figures, tables, and citations

No human or animal subjects; no ethics approval applicable. Figures were checked for
axis/denominator consistency: the log-axis exception, the numerical ceiling, and the
shared-vs-per-snapshot denominators are documented and both denominators are reported.
Two presentation points: the T figure's empty left panel needs its in-panel explanation
retained in any resized version, and coincident series (γ=0 vs CPG) must remain visually
distinguishable — currently handled with a pale wide band, which should be preserved.

## Limitations of this review

This review is based on the local source tree, the sweep CSV and the compiled report. No
analysis was independently re-run from a clean environment, no dataset provenance was verified
against the source publications, and the vendored CPG/mCPG transcriptions were not checked
against the published algorithms. Conclusions about domain validity are out of scope and a
contact-mechanics specialist is required for dataset provenance and multiplier semantics.

## Reviewer disclosures

The reviewer is not independent of this artifact: the reviewer co-produced the γ-family
implementation, the metric and figure choices, and much of the report section under review, and
previously asserted several conclusions now under review, including a median-based selection that
was subsequently retracted. This review must therefore be treated as structured self-audit, not
independent peer review, and an independent reviewer should be obtained before any external
submission. Local deterministic tooling was used; no content left the machine.

# Confidential comments to editor

Reviewer independence is absent for the reason stated in the disclosures above; an independent
reviewer is required. Specialist review is requested for contact-mechanics dataset provenance.
