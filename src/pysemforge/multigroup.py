"""Multigroup confirmatory factor analysis and measurement invariance.

The ladder is estimated by *joint* maximum likelihood over the stacked groups:
one likelihood is formed from every group's sample covariance matrix and mean
vector, and equality constraints are imposed by binding the corresponding
entries to a single free parameter.  The resulting chi-square is therefore a
single statistic with a single degrees-of-freedom count, and the nested
comparisons are ordinary likelihood-ratio tests.

Public API
----------
- :func:`fit_multigroup` -- one multigroup fit at a given constraint level.
- :func:`invariance_test` -- configural -> metric -> scalar ladder with
  Delta chi-square / Delta CFI / Delta RMSEA comparisons.
- :func:`modification_indices` -- score-type tests for releasing a single
  cross-group constraint, with the expected parameter change.
- :func:`partial_invariance` -- iterative release of the constraints with the
  largest modification indices (partial invariance search).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from .model import Model
from .result import InvarianceResult

LEVELS = ("configural", "metric", "scalar", "strict")


# --------------------------------------------------------------------- spec
@dataclass
class _Spec:
    """Measurement structure compiled from the parameter table."""

    latents: list = field(default_factory=list)
    observed: list = field(default_factory=list)
    loadings: list = field(default_factory=list)      # [(latent, indicator)]
    anchored: set = field(default_factory=set)        # {(latent, indicator)}
    residual_covs: list = field(default_factory=list)  # [(v1, v2)]


def _compile(model: Model) -> _Spec:
    """Read the measurement structure out of a parsed model specification."""
    pt = model.params_table
    spec = _Spec()

    for entry in pt:
        if entry.get("type") == "regression":
            raise NotImplementedError(
                "Multigroup estimation in pysemforge supports measurement models "
                "(=~) and residual covariances (~~). Structural paths between "
                "observed variables are not implemented for the multigroup case."
            )

    for entry in pt:
        if entry.get("type") != "loading":
            continue
        lat, ind = entry["latent"], entry["indicator"]
        if lat not in spec.latents:
            spec.latents.append(lat)
        if ind not in spec.observed:
            spec.observed.append(ind)
        spec.loadings.append((lat, ind))

    if not spec.loadings:
        raise ValueError("The model contains no measurement (=~) statements.")

    seen = set()
    for lat, ind in spec.loadings:
        if lat not in seen:
            seen.add(lat)
            spec.anchored.add((lat, ind))

    for entry in pt:
        if entry.get("type") != "covariance":
            continue
        a, b = entry["var1"], entry["var2"]
        if a in spec.latents and b in spec.latents:
            continue  # free by default
        if a in spec.observed and b in spec.observed and a != b:
            spec.residual_covs.append((a, b))

    return spec


# ------------------------------------------------------------------- design
def _build_design(spec: _Spec, n_groups: int, level: str, release=frozenset()):
    """Map every model quantity to an index in the free-parameter vector.

    ``release`` holds constraints that are allowed to differ across groups at
    this level (used by the partial-invariance search), e.g.
    ``{("intercept", "x7")}``.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")

    load_shared = level in ("metric", "scalar", "strict")
    int_shared = level in ("scalar", "strict")
    res_shared = level == "strict"

    pos: dict = {}
    counter = [0]

    def slot(tag):
        if tag not in pos:
            pos[tag] = counter[0]
            counter[0] += 1
        return pos[tag]

    load_idx = {}
    for lat, ind in spec.loadings:
        if (lat, ind) in spec.anchored:
            continue
        for g in range(n_groups):
            key = ("load", lat, ind) if load_shared else ("load", g, lat, ind)
            load_idx[(g, lat, ind)] = slot(key)

    int_idx = {}
    for g in range(n_groups):
        for v in spec.observed:
            shared = int_shared and ("intercept", v) not in release
            key = ("int", v) if shared else ("int", g, v)
            int_idx[(g, v)] = slot(key)

    resid_idx = {}
    for g in range(n_groups):
        for v in spec.observed:
            key = ("res", v) if res_shared else ("res", g, v)
            resid_idx[(g, v)] = slot(key)

    resid_cov_idx = {}
    for a, b in spec.residual_covs:
        for g in range(n_groups):
            key = ("rescov", a, b) if res_shared else ("rescov", g, a, b)
            resid_cov_idx[(g, a, b)] = slot(key)

    m = len(spec.latents)
    phi_idx = {}
    for g in range(n_groups):
        for j in range(m):
            for i in range(j, m):
                phi_idx[(g, i, j)] = slot(("phi", g, i, j))

    alpha_idx = {}
    means_free = level in ("scalar", "strict")
    for g in range(n_groups):
        for lat in spec.latents:
            alpha_idx[(g, lat)] = (
                slot(("alpha", g, lat)) if (means_free and g > 0) else None
            )

    design = {
        "level": level,
        "n_groups": n_groups,
        "n_free": counter[0],
        "load_idx": load_idx,
        "int_idx": int_idx,
        "resid_idx": resid_idx,
        "resid_cov_idx": resid_cov_idx,
        "phi_idx": phi_idx,
        "alpha_idx": alpha_idx,
        "n_obs_vars": len(spec.observed),
        "n_latents": m,
    }
    return design


def _start_values(spec, stats, design):
    """Starting values: unit-ish loadings, half the observed variance as residual."""
    x = np.zeros(design["n_free"])
    p, m = len(spec.observed), len(spec.latents)
    oi = {v: i for i, v in enumerate(spec.observed)}

    for (g, lat, ind), k in design["load_idx"].items():
        x[k] = 0.8
    for (g, v), k in design["int_idx"].items():
        x[k] = float(stats[g]["mean"][oi[v]])
    for (g, v), k in design["resid_idx"].items():
        var = max(float(stats[g]["cov"][oi[v], oi[v]]), 1e-3)
        x[k] = np.log(0.5 * var)
    for (_g, a, b), k in design["resid_cov_idx"].items():
        x[k] = 0.0
    for (g, i, j), k in design["phi_idx"].items():
        x[k] = 0.0 if i == j else 0.2  # Cholesky: identity + mild cross-loading
    return x


def _unpack(x, spec, stats, design):
    """Translate the free-parameter vector into per-group matrices."""
    p, m = len(spec.observed), len(spec.latents)
    oi = {v: i for i, v in enumerate(spec.observed)}
    li = {v: i for i, v in enumerate(spec.latents)}
    out = []

    for g in range(design["n_groups"]):
        Lam = np.zeros((p, m))
        for lat, ind in spec.loadings:
            if (lat, ind) in spec.anchored:
                Lam[oi[ind], li[lat]] = 1.0
            else:
                Lam[oi[ind], li[lat]] = x[design["load_idx"][(g, lat, ind)]]

        Theta = np.zeros((p, p))
        for v in spec.observed:
            i = oi[v]
            Theta[i, i] = np.exp(x[design["resid_idx"][(g, v)]])
        for a, b in spec.residual_covs:
            i, j = oi[a], oi[b]
            Theta[i, j] = Theta[j, i] = x[design["resid_cov_idx"][(g, a, b)]]

        L = np.zeros((m, m))
        for j in range(m):
            for i in range(j, m):
                L[i, j] = x[design["phi_idx"][(g, i, j)]]
        for j in range(m):
            L[j, j] = np.exp(L[j, j])
        Phi = L @ L.T

        tau = np.array([x[design["int_idx"][(g, v)]] for v in spec.observed])
        alpha = np.zeros(m)
        for lat in spec.latents:
            k = design["alpha_idx"][(g, lat)]
            if k is not None:
                alpha[li[lat]] = x[k]

        Sigma = Lam @ Phi @ Lam.T + Theta
        mu = tau + Lam @ alpha
        out.append({"Lambda": Lam, "Theta": Theta, "Phi": Phi,
                    "tau": tau, "alpha": alpha, "Sigma": Sigma, "mu": mu})
    return out


def _discrepancy(S, cov, mean, mu):
    """Normal-theory ML discrepancy for one group (without the -ln|S| - p part)."""
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0 or not np.isfinite(logdet):
        return np.inf
    try:
        inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return np.inf
    d = mean - mu
    val = logdet + float(np.trace(inv @ cov)) + float(d @ inv @ d)
    if not np.isfinite(val):
        return np.inf
    return val


def _objective(x, spec, stats, design):
    mats = _unpack(x, spec, stats, design)
    total = 0.0
    for g, mat in enumerate(mats):
        f = _discrepancy(mat["Sigma"], stats[g]["cov"], stats[g]["mean"], mat["mu"])
        if not np.isfinite(f):
            return 1e10
        total += stats[g]["n"] * (f - stats[g]["logdet"] - len(spec.observed))
    return total  # the chi-square itself


def _baseline(stats, p):
    """Independence-model chi-square, per group and pooled."""
    total = 0.0
    logdet = 0.0
    for st in stats:
        s = np.diag(st["cov"])
        total += st["n"] * (np.sum(np.log(s)) - st["logdet"])
        logdet += st["n"] * st["logdet"]
    return total


# -------------------------------------------------------------------- fitting
def _prepare_stats(model, data, group_col):
    spec = _compile(model)
    cols = [v for v in spec.observed if v in data.columns]
    missing = [v for v in spec.observed if v not in data.columns]
    if missing:
        raise KeyError(f"Variables missing from the data: {missing}")
    groups = sorted(data[group_col].unique().tolist())
    if len(groups) < 2:
        raise ValueError("Multigroup analysis requires at least two groups.")

    stats = []
    for g in groups:
        sub = data.loc[data[group_col] == g, cols]
        cov = np.cov(sub.to_numpy(dtype=float), rowvar=False, ddof=0)
        cov = np.atleast_2d(cov)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError(f"Sample covariance matrix for group {g!r} is singular.")
        stats.append({"n": int(len(sub)), "cov": cov,
                      "mean": sub.to_numpy(dtype=float).mean(axis=0),
                      "logdet": float(logdet)})
    return spec, cols, groups, stats


def fit_multigroup(model, data, group_col, level="configural", release=frozenset(),
                   maxiter=4000):
    """Fit one multigroup model at a given constraint level by joint ML.

    Parameters
    ----------
    level : {'configural', 'metric', 'scalar', 'strict'}
        ``configural``: free loadings, intercepts and residual variances;
        ``metric``: loadings equal across groups; ``scalar``: loadings and
        intercepts equal, factor means free in groups 2..G; ``strict``: also
        equal residual variances.
    release : set of tuple
        Constraints to exempt from the cross-group restriction, e.g.
        ``{("intercept", "x7")}``.

    Returns
    -------
    dict
        ``{'chisq', 'df', 'pvalue', 'cfi', 'tli', 'rmsea', 'n_free', 'n_obs',
        'n_groups', 'converged', 'group_chisq', 'matrices', 'design',
        'params'}``.
    """
    spec, cols, groups, stats = _prepare_stats(model, data, group_col)
    g_count = len(groups)
    p = len(cols)
    design = _build_design(spec, g_count, level, release)
    x0 = _start_values(spec, stats, design)

    res = minimize(
        _objective, x0, args=(spec, stats, design), method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 200000},
    )
    x = res.x

    mats = _unpack(x, spec, stats, design)
    chisq = float(_objective(x, spec, stats, design))
    n_total = sum(st["n"] for st in stats)
    n_elements = g_count * (p * (p + 1) // 2 + p)
    df = int(n_elements - design["n_free"])

    chi_b = _baseline(stats, p)
    df_b = g_count * p * (p - 1) // 2

    d_model = max(chisq - df, 0.0)
    d_base = max(chi_b - df_b, 1e-12)
    cfi = 1.0 - d_model / d_base if df_b > 0 else 1.0
    tli = ((chi_b / df_b) - (chisq / df)) / ((chi_b / df_b) - 1.0) if df > 0 and df_b > 0 else 1.0
    # lavaan evaluates the RMSEA of a multigroup model with the average group
    # size, not the pooled N: sqrt(max(chi2 - df, 0) / (df * N / G)).
    n_avg = n_total / g_count
    rmsea = float(np.sqrt(d_model / (df * n_avg))) if df > 0 else 0.0
    lo, hi = _rmsea_ci(chisq, df, n_avg)

    group_chisq = []
    for g, mat in enumerate(mats):
        f = _discrepancy(mat["Sigma"], stats[g]["cov"], stats[g]["mean"], mat["mu"])
        group_chisq.append(stats[g]["n"] * (f - stats[g]["logdet"] - p))

    return {
        "level": level,
        "chisq": chisq,
        "df": df,
        "pvalue": float(chi2.sf(chisq, df)) if df > 0 else 1.0,
        "cfi": float(min(max(cfi, 0.0), 1.0)),
        "tli": float(min(max(tli, 0.0), 1.0)),
        "rmsea": rmsea,
        "rmsea_lower": lo,
        "rmsea_upper": hi,
        "n_free": design["n_free"],
        "n_obs": n_total,
        "n_groups": g_count,
        "groups": groups,
        "converged": bool(res.success),
        "n_iter": int(res.nit),
        "params": x,
        "group_chisq": group_chisq,
        "matrices": mats,
        "design": design,
        "spec": spec,
        "stats": stats,
    }


def _rmsea_ci(chisq, df, n):
    """90% CI for the RMSEA via the non-central chi-square distribution."""
    if df <= 0:
        return 0.0, 0.0

    def _rmsea(lam):
        return float(np.sqrt(max(lam, 0.0) / (df * n)))

    lo_lam, hi_lam = _solve_nc(chisq, df, 0.95), _solve_nc(chisq, df, 0.05)
    return _rmsea(lo_lam), _rmsea(hi_lam)


def _solve_nc(chisq, df, target):
    """Non-centrality whose CDF at ``chisq`` equals ``target``."""
    lo, hi = 0.0, max(chisq, df) * 5 + 50
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2.cdf(chisq, df, mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------- ladder
def invariance_test(model, data, group_col, levels=("configural", "metric", "scalar")):
    """Run a configural -> metric -> scalar measurement invariance ladder.

    Parameters
    ----------
    model : Model
        Measurement model in lavaan-style syntax.
    data : pandas.DataFrame
        Observed indicators plus ``group_col``.
    group_col : str
        Column defining the groups.
    levels : tuple
        Levels to fit, in order.

    Returns
    -------
    InvarianceResult
        ``.levels`` holds the fit of each model; ``.comparisons`` holds the
        nested likelihood-ratio tests and the change-in-fit statistics.
    """
    levels = tuple(levels)
    level_fit = {}
    fits = {}
    for lvl in levels:
        fit = fit_multigroup(model, data, group_col, level=lvl)
        fits[lvl] = fit
        level_fit[lvl] = {
            k: fit[k] for k in
            ("chisq", "df", "pvalue", "cfi", "tli", "rmsea",
             "n_free", "n_obs", "n_groups", "converged")
        }
        level_fit[lvl]["rmsea_lower"] = fit["rmsea_lower"]
        level_fit[lvl]["rmsea_upper"] = fit["rmsea_upper"]

    comparisons = {}
    decisions = {}
    for prev, nxt in zip(levels, levels[1:]):
        a, b = level_fit[prev], level_fit[nxt]
        dchisq = b["chisq"] - a["chisq"]
        ddf = b["df"] - a["df"]
        pvalue = float(chi2.sf(max(dchisq, 0.0), ddf)) if ddf > 0 else float("nan")
        d_cfi = b["cfi"] - a["cfi"]
        d_rmsea = b["rmsea"] - a["rmsea"]
        # Chen (2007) / Cheung & Rensvold (2002) change-in-fit criteria.
        non_invariant = (d_cfi <= -0.010) or (d_rmsea >= 0.015)
        comparisons[f"{prev}_vs_{nxt}"] = {
            "delta_chisq": float(dchisq),
            "delta_df": int(ddf),
            "pvalue": pvalue,
            "delta_cfi": float(d_cfi),
            "delta_rmsea": float(d_rmsea),
        }
        decisions[f"{prev}_vs_{nxt}"] = (
            "NON-INVARIANT" if non_invariant else "INVARIANT"
        )

    result = InvarianceResult(level_fit, comparisons, decisions)
    # Full fit objects (including the estimated matrices) are kept aside so
    # that ``to_dataframe()`` stays tabular.
    result.fits = fits
    return result


# ------------------------------------------------- modification indices / EPC
_CONSTRAINTS = {
    "metric": lambda spec: [("loading", lat, ind) for lat, ind in spec.loadings
                            if (lat, ind) not in spec.anchored],
    "scalar": lambda spec: [("intercept", v) for v in spec.observed],
    "strict": lambda spec: [("residual", v) for v in spec.observed],
}


def modification_indices(model, data, group_col, level="scalar",
                         constraints=None, release=frozenset(), maxiter=4000):
    """Likelihood-ratio change from releasing single cross-group constraints.

    For each candidate constraint the model is refitted with that constraint
    (and only that one) released; the drop in chi-square, on one degree of
    freedom, is the modification index.  ``epc`` reports how far apart the two
    groups' estimates of the parameter become once it is allowed to differ.

    Parameters
    ----------
    release : set of tuple
        Constraints already released in the current model, so that indices are
        computed against the current solution rather than the fully restricted
        one.

    Returns
    -------
    pandas.DataFrame
        Columns ``constraint``, ``mi``, ``df``, ``pvalue``, ``epc``, sorted by
        descending modification index.
    """
    spec, _cols, _groups, _stats = _prepare_stats(model, data, group_col)
    if constraints is None:
        constraints = _CONSTRAINTS[level](spec)

    base = fit_multigroup(model, data, group_col, level=level,
                          release=frozenset(release), maxiter=maxiter)
    rows = []
    for con in constraints:
        fit = fit_multigroup(model, data, group_col, level=level,
                             release=frozenset(release) | {con}, maxiter=maxiter)
        mi = base["chisq"] - fit["chisq"]
        rows.append({
            "constraint": con,
            "mi": float(max(mi, 0.0)),
            "df": 1,
            "pvalue": float(chi2.sf(max(mi, 0.0), 1)),
            "epc": _expected_change(con, base, fit),
        })
    out = pd.DataFrame(rows).sort_values("mi", ascending=False, ignore_index=True)
    out.attrs["base_chisq"] = base["chisq"]
    out.attrs["base_df"] = base["df"]
    return out


def _expected_change(constraint, base, released):
    """Difference between the two groups' estimates of a released parameter."""
    kind = constraint[0]
    spec, design = base["spec"], base["design"]
    mats_b, mats_r = base["matrices"], released["matrices"]
    if kind == "intercept":
        i = spec.observed.index(constraint[1])
        return float(mats_r[1]["tau"][i] - mats_r[0]["tau"][i])
    if kind == "loading":
        _, lat, ind = constraint
        i, j = spec.observed.index(ind), spec.latents.index(lat)
        return float(mats_r[1]["Lambda"][i, j] - mats_r[0]["Lambda"][i, j])
    if kind == "residual":
        i = spec.observed.index(constraint[1])
        return float(mats_r[1]["Theta"][i, i] - mats_r[0]["Theta"][i, i])
    return float("nan")


def partial_invariance(model, data, group_col, level="scalar", alpha=0.05,
                       cfi_tolerance=0.010, max_release=10, maxiter=4000):
    """Partial invariance search by releasing constraints one at a time.

    Starting from the fully restricted model at ``level``, the constraint with
    the largest modification index is released and the model refitted.  The
    search stops when no remaining constraint is significant at ``alpha`` or
    when the change in CFI relative to the preceding level is back within
    ``cfi_tolerance``.  The decision to report a partial solution remains with
    the analyst: the function reports what was released and how fit recovered.

    Returns
    -------
    dict
        ``{'full': InvarianceResult, 'released': [...], 'final': {...}}``.
    """
    levels = ("configural", "metric", "scalar")
    ladder = invariance_test(model, data, group_col, levels=levels)
    if level not in levels:
        raise ValueError(f"level must be one of {levels}, got {level!r}")

    spec, _cols, _groups, _stats = _prepare_stats(model, data, group_col)
    remaining = list(_CONSTRAINTS[level](spec))
    reference_cfi = ladder.levels[levels[levels.index(level) - 1]]["cfi"]

    released = []
    current = fit_multigroup(model, data, group_col, level=level, maxiter=maxiter)

    while remaining and len(released) < max_release:
        table = modification_indices(model, data, group_col, level=level,
                                     constraints=remaining,
                                     release=released_params(released),
                                     maxiter=maxiter)
        top = table.iloc[0]
        if top["pvalue"] > alpha:
            break
        released.append({"step": level, "parameter": top["constraint"],
                         "mi": float(top["mi"]), "epc": float(top["epc"])})
        remaining = [c for c in remaining if c != top["constraint"]]
        current = fit_multigroup(model, data, group_col, level=level,
                                 release=frozenset(released_params(released)),
                                 maxiter=maxiter)
        if current["cfi"] - reference_cfi > -cfi_tolerance:
            break

    final = {
        "invariant_up_to": _last_invariant_level(ladder.decisions),
        "released_parameters": released,
        "chisq": current["chisq"],
        "df": current["df"],
        "cfi": current["cfi"],
        "rmsea": current["rmsea"],
        "reference_cfi": reference_cfi,
    }
    return {"full": ladder, "released": released, "final": final}


def released_params(released):
    """Extract the constraint tuples from a list of release records."""
    out = []
    for item in released:
        p = item["parameter"]
        if p not in out:
            out.append(p)
    return out


def _last_invariant_level(decisions: dict) -> str:
    last = "configural"
    for step, dec in decisions.items():
        if dec == "INVARIANT":
            last = step.split("_vs_")[-1]
    return last
