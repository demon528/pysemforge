# Changelog

## [0.2.0] - 2026-11-18

The multigroup layer is now a real estimator. Verified against lavaan 0.6-20 on R 4.2.3
(``cfa(..., group = "school", group.equal = ...)``) on the Holzinger-Swineford
data; every statistic below matches lavaan to four decimals.

| Model | chi2 | df | CFI | TLI | RMSEA |
| --- | --- | --- | --- | --- | --- |
| configural | 115.8513 | 48 | .9234 | .8851 | .0969 |
| metric | 124.0435 | 54 | .9209 | .8946 | .0928 |
| scalar | 164.1028 | 60 | .8825 | .8590 | .1074 |

Delta chi-square: metric 8.192 (df 6, p = .224); scalar 40.059 (df 6, p < .001).
Modification indices for the scalar model reproduce lavaan's partial models:
releasing x3 gives 19.5238, x7 gives 15.1544, x2 gives 7.7371.

### Added
- `multigroup.fit_multigroup`: joint ML over the stacked groups. One
  likelihood, one chi-square, equality constraints imposed by binding parameter
  entries to a shared index, degrees of freedom counted from the number of
  distinct free parameters. Supports `configural`, `metric`, `scalar` and
  `strict`, and arbitrary sets of released constraints.
- `multigroup.modification_indices`: likelihood-ratio change for releasing
  each cross-group constraint, with the group difference in the released
  parameter.
- `multigroup.partial_invariance`: releases the largest modification index,
  refits, and stops when the change in CFI relative to the preceding level is
  back within .01.
- `pysemforge.datasets`: the Holzinger-Swineford data (301 x 9 plus `school`),
  exported from lavaan so that the paper's numbers can be reproduced with one
  call.
- `SEMResult.parameters`: the estimate table (Estimate, Est. Std, Std. Err,
  z-value, p-value) returned by the backend.
- `dpi=` argument for `Model.plot` / `plotting.plot_model`, so a manuscript
  figure can be written at 600 dpi instead of the 150 dpi screen default.
- Semicolons are accepted as statement separators, so a specification can be
  written on one line; the syntax is normalised to one statement per line
  before it reaches the backend.
- Vector output is reproducible: the date stamp and the SVG id salt are
  fixed, so re-running an analysis produces a byte-identical `.pdf` or `.svg`.
- `InvarianceResult.to_dataframe()` returns one row per level with the step
  changes on the same row, which is the shape of a manuscript table.
- `PathDiagram(..., intercepts=..., free_intercepts=...)` writes indicator
  intercepts under the observed boxes, with free ones in the accent colour, so
  a partial invariance solution can be drawn directly from the fit.
- Ellipse width is computed from the axes width rather than the figure width,
  so a diagram drawn into one panel of a multi-panel figure is sized correctly.
- `examples/fig2_two_groups.py` reproduces Fig. 2 of the paper.
- The lavaan cross-validation test now locates `Rscript` itself (env var,
  `PATH`, or the standard install directories), refits a two-factor model with
  known loadings, and compares every statistic and standardised loading; it is
  no longer skipped on a machine that has R.

### Fixed
- The previous `invariance_test` fitted each group separately and averaged the
  fit indices. No cross-group constraint ever entered a likelihood, so the
  three levels produced identical statistics and the "ladder" was not nested.
- `Model.fit(group=...)` now reports the joint configural fit instead of the
  mean of the per-group fits.
- The sample covariance matrix used the unbiased (n - 1) divisor; lavaan uses
  the ML divisor (n). The difference is small for single-group CFA but changes
  the third decimal of a mean-structured multigroup model.
- `diagnostics` no longer ships placeholder `modification_indices` and
  `bootstrap` functions that returned noise instead of estimates.
- The README no longer advertises syntax that the parser accepts but the
  backend rejects (`~1` intercept statements, `==` constraints in the model
  string, `group:` blocks) and now states the current scope explicitly.
- The single-group summary printed `Iterations: 0` because the backend wrapper
  does not propagate the optimiser counter; the line is now omitted when no
  count is available instead of reporting a misleading zero.

### Packaging
- The sdist no longer bundles the previously built `.whl`/`.tar.gz` files,
  caches or loose figures (49 files, 57 kB instead of 51 files, 120 kB).
- Verified end to end in a fresh virtual environment on Python 3.13.12 with
  NumPy 2.5.5, pandas 3.1.1, SciPy 1.18.1, Matplotlib 3.11.2 and semopy 2.3.11:
  28 tests pass, including the lavaan cross-validation; the replication and
  figure scripts reproduce Table 2, Table 3 and both figures of the paper.

## [0.1.1] - 2026-11-18

Bug-fix release. Verified against lavaan 0.6-20 on `HolzingerSwineford1939`:
χ² = 85.306 (df = 24), CFI = .931, TLI = .896, RMSEA = .092 [.072, .114],
SRMR = .065, all matching lavaan to three decimals.

### Fixed — estimation
- `fit_model_semopy` imported `Model` but called `semopy.calc_stats`, raising
  `NameError` that was swallowed by a bare `except`; every fit index was then
  written as zero and the boundary correction forced CFI/TLI to 1.000.
- `Model.fit` passed the raw syntax string to the native estimator when semopy
  was absent. The estimator iterated the string character by character, the
  objective became constant, and the "solution" was all zeros (`~1e-8`) with a
  plausible-looking but meaningless χ². Syntax strings are now never handed to
  the native path, and a missing backend raises instead of returning garbage.
- `Model._prepare` derived its column order from a `set`. χ² is invariant to
  column order, so this was invisible there, but SRMR is not: it reported .280
  instead of .065. Columns now follow model-syntax order.
- SRMR was hard-coded to 0 (semopy's `calc_stats` has no SRMR column). It is now
  computed from the sample and implied covariance matrices, aligned by variable
  name, lower triangle including the diagonal.
- RMSEA confidence interval was hard-coded to [0, 0]. It is now obtained from the
  non-central χ², with the lower bound truncated at zero.

### Fixed — plotting
- Latent variables mentioned in a `~~` clause were also registered as observed,
  so each latent was drawn as an ellipse *and* a rectangle, one indicator lost
  its position, and nodes overlapped.
- The `nodes` property was defined twice; the second definition silently
  overwrote the first.
- Covariance edges were drawn as dashed paths without arrowheads. They are now
  genuine double-headed arcs, drawn above the node patches so the heads are not
  hidden, with curvature scaled to the distance spanned.
- Arrow endpoints into latent variables used the observed-node radius, so edges
  ran inside the ellipse, or stopped short of it.
- Layout gave every factor a fixed horizontal slot, so factors with different
  numbers of indicators collided. Horizontal space is now allocated per
  indicator.
- Models without latent variables raised `KeyError` during layout.
- Edge labels sat on the lines. They are now offset perpendicular to the edge
  and drawn on an opaque background.
- Coefficients were mapped to paths by positional index into `inspect()`, which
  breaks as soon as semopy orders parameters differently. They are now matched
  by `(lval, op, rval)`.
- Coefficients are read from the fitted object, so a figure can no longer
  disagree with the parameter table it accompanies.
- Latent variables were drawn as circles, so a name longer than about six
  characters spilled over the outline; they are now ellipses sized to the
  longest label, as the field's conventions require.
- Arrow heads were placed by shrinking along the edge by a single radius, which
  is only correct for a circle; endpoints are now solved on the node outline.
- Loading labels sat at the edge midpoint, against the latent outline. They are
  now placed towards the indicator end, in clear space above the box.
- Covariance arcs now leave the ovals at an angle above the horizontal, so
  several covariances meeting on one factor no longer stack their arrow heads
  at the same point.
- Vertical distance between the factor row and the indicator row was too small
  to hold a coefficient label without collision.

### Changed
- `semopy>=2.3.0` is now a declared dependency. In 0.1.0 the backend was
  optional, which meant a default install exercised the broken native estimator.

### Known issues
- `multigroup.invariance_test` fits each group separately and averages the
  results; cross-group equality constraints are not imposed on the fit, so the
  configural, metric and scalar levels can return identical statistics. The
  invariance ladder should be treated as experimental until a joint multigroup
  estimator lands.
- The native estimator's `_var_index` maps variable names into matrix columns by
  `hash(name) % p`, so its implied covariance matrix is unrelated to the data.
  It is kept only as a reference implementation and must not be used for
  inference.

## [0.1.0] - 2026-09-05

Initial MVP release.

### Added
- lavaan-compatible model syntax parser (`=~`, `~`, `~~`, `~1`, `==`, comments)
- Core estimation: ML, MLR (robust SE), WLSMV (polychoric pathway)
- Fit indices: χ², CFI, TLI, RMSEA (±90% CI), SRMR
- Native multigroup analysis
- **Measurement invariance** ladder (configural → metric → scalar) with
  Δχ² / ΔCFI decisions — `semTools` parity
- **Partial invariance** (build-up / tear-down)
- **graphviz-free path diagrams** (matplotlib backend)
- pandas / NumPy / SciPy interoperability
- Full test suite + lavaan cross-validation harness
- CI (lint, type check, tests, lavaan validation) for Python 3.10–3.13
- Release script + PyPI publishing workflow

[0.1.0]: https://github.com/your-username/pysemforge/releases/tag/v0.1.0
