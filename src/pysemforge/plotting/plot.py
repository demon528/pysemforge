"""High-level plotting API: :func:`plot_model`."""

from __future__ import annotations

from typing import Dict, Optional

from .diagram import PathDiagram
from .style import default_style, publication_style


_PRESETS = {
    "default": default_style,
    "publication": publication_style,
}


def _ordered_variables(model):
    """Latent / observed names in first-appearance order (not alphabetical).

    Sorting alphabetically scrambles the layout: indicators end up grouped by
    name rather than by their parent latent, and ``x10`` sorts before ``x2``.
    """
    latents, observed = [], []
    latent_set = set(model.variables.get("latent", []))
    for entry in model.params_table:
        t = entry.get("type")
        names = []
        if t == "loading":
            for n in (entry["latent"], entry["indicator"]):
                names.append(n)
        elif t == "regression":
            for n in (entry["independent"], entry["dependent"]):
                names.append(n)
        elif t in ("covariance", "constraint"):
            for n in (entry["var1"], entry["var2"]):
                names.append(n)
        elif t == "intercept":
            names.append(entry["variable"])
        for n in names:
            if n in latent_set:
                if n not in latents:
                    latents.append(n)
            elif n in observed:
                continue
            elif n not in observed:
                observed.append(n)
    return latents, observed


def _fitted_coefficients(model, standardized):
    """Map params_table entry index -> fitted coefficient."""
    raw = getattr(model.result, "result", None) if model.result is not None else None
    if not isinstance(raw, dict):
        return {}

    # Preferred: semopy's estimate table, keyed by (lval, op, rval). Positional
    # alignment against ``params`` is meaningless because semopy also reports
    # latent and residual variances, which never appear in ``params_table``.
    table = raw.get("param_table")
    if table is not None:
        coef = _coef_from_table(model, table, standardized)
        if coef:
            return coef

    params = raw.get("params")
    if params is None:
        return {}
    from pysemforge.estim import _build_param_index

    param_of, _ = _build_param_index(model.params_table)
    coef = {}
    for i in param_of:
        k = param_of[i]
        if 0 <= k < len(params):
            coef[i] = float(params[k])
    if not coef:
        return {}

    if standardized:
        sd = _scale_factors(model, coef)
        for i in list(coef):
            e = model.params_table[i]
            t = e.get("type")
            if t == "loading":
                coef[i] *= sd.get(e["indicator"], 1.0)
            elif t == "regression":
                coef[i] *= sd.get(e["independent"], 1.0) / max(sd.get(e["dependent"], 1.0), 1e-12)
    return coef


def _coef_from_table(model, table, standardized):
    """Match semopy's estimate rows to ``params_table`` entries by name."""
    try:
        rows = table.to_dict("records")
    except Exception:
        return {}

    lookup = {}
    for r in rows:
        op = str(r.get("op", "")).strip()
        lval = str(r.get("lval", "")).strip()
        rval = str(r.get("rval", "")).strip()
        est = r.get("Estimate")
        std = r.get("Est. Std")
        try:
            est = float(est)
        except (TypeError, ValueError):
            continue
        try:
            std = float(std)
            if std != std:  # NaN
                std = None
        except (TypeError, ValueError):
            std = None
        val = std if (standardized and std is not None) else est
        lookup[(op, lval, rval)] = val

    def pick(*keys):
        for k in keys:
            if k in lookup:
                return lookup[k]
        return None

    coef = {}
    for i, e in enumerate(model.params_table):
        t = e.get("type")
        v = None
        if t == "loading":
            v = pick(
                ("~", e["indicator"], e["latent"]),
                ("=~", e["indicator"], e["latent"]),
                ("=~", e["latent"], e["indicator"]),
            )
        elif t == "regression":
            v = pick(
                ("~", e["dependent"], e["independent"]),
                ("~", e["independent"], e["dependent"]),
            )
        elif t == "covariance":
            v = pick(
                ("~~", e["var1"], e["var2"]),
                ("~~", e["var2"], e["var1"]),
            )
        if v is not None:
            coef[i] = v
    return coef


def _scale_factors(model, coef):
    """Observed-variable SDs used to standardise path coefficients.

    Latent variances are fixed at 1.0 in the implied-covariance construction,
    so a standardised loading is ``lambda * 1 / sd(indicator)`` inverted as
    ``lambda / sd(indicator) * 1``. Falls back to 1.0 (raw coefficients) when
    the training data are no longer attached to the model.
    """
    data = getattr(model, "_last_data", None)
    if data is None:
        return {}
    out = {}
    for name in model.variables.get("observed", []):
        if name in getattr(data, "columns", []):
            try:
                s = float(data[name].std())
            except Exception:
                s = 0.0
            if s > 0:
                out[name] = s
    return out


def plot_model(
    model,
    filename: Optional[str] = None,
    standardized: bool = True,
    show: bool = True,
    style: Optional[Dict] = None,
    ax=None,
    dpi: Optional[int] = None,
):
    """Draw a path diagram for a (fitted) :class:`pysemforge.model.Model`.

    Pure matplotlib -- no graphviz required.

    Parameters
    ----------
    model : pysemforge.model.Model
    filename : str, optional
        If given, render to this file (``.png``, ``.pdf``, ...).
    standardized : bool
        Divide loadings and regressions by the observed-variable SD, giving
        standardised path coefficients (requires the fitted data to still be
        attached to the model).
    show : bool
        If True (and not saving), display interactively.
    style : dict, optional
        Style override / preset.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If None, a new figure is created.
    dpi : int, optional
        Resolution used when ``filename`` is given; default 150. Use 600 for
        a manuscript figure.

    Returns
    -------
    matplotlib.figure.Figure or None
        The created figure when drawing interactively, else None.
    """
    import matplotlib

    if filename is not None or not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    st = dict(default_style)
    if style:
        if isinstance(style, str):
            preset = _PRESETS.get(style)
            if preset is None:
                raise ValueError(
                    f"Unknown style preset {style!r}. Choose from {list(_PRESETS)}."
                )
            st.update(preset)
        else:
            st.update(style)

    latents, observed = _ordered_variables(model)

    coef_by_entry = _fitted_coefficients(model, standardized)

    edges = []
    for i, entry in enumerate(model.params_table):
        if entry["type"] == "loading":
            edges.append((entry["latent"], entry["indicator"], coef_by_entry.get(i)))
        elif entry["type"] == "regression":
            edges.append((entry["independent"], entry["dependent"], coef_by_entry.get(i)))

    covariances = []
    for i, e in enumerate(model.params_table):
        if e["type"] == "covariance":
            covariances.append((e["var1"], e["var2"], coef_by_entry.get(i)))

    # A CFA syntax rarely spells out ``f1 ~~ f2``, yet every SEM engine
    # estimates those latent covariances and they belong in the diagram.
    # Pull them from the estimate table when it is available.
    table = raw_table = None
    try:
        raw_table = model.result.result.get("param_table")
    except Exception:
        raw_table = None
    if raw_table is not None:
        latent_set = set(latents)
        seen = {frozenset((a, b)) for a, b, _ in covariances}
        try:
            rows = raw_table.to_dict("records")
        except Exception:
            rows = []
        for r in rows:
            if str(r.get("op", "")).strip() != "~~":
                continue
            a, b = str(r.get("lval", "")).strip(), str(r.get("rval", "")).strip()
            if a == b or a not in latent_set or b not in latent_set:
                continue  # variances and residual covariances are not drawn
            if frozenset((a, b)) in seen:
                continue
            try:
                val = float(r.get("Est. Std") if standardized else r.get("Estimate"))
            except (TypeError, ValueError):
                continue
            covariances.append((a, b, val))
            seen.add(frozenset((a, b)))

    diagram = PathDiagram(latents, observed, edges, covariances, style=st)

    if filename is not None:
        diagram.save(filename, dpi=dpi)
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()
    diagram.draw(ax)
    if show:
        plt.show()
    return fig, ax


def _entry_signature(entry: dict) -> str:
    """Stable string signature for a parameter-table entry."""
    t = entry["type"]
    if t == "loading":
        return f"L[{entry['latent']}->{entry['indicator']}]"
    if t == "regression":
        return f"B[{entry['independent']}->{entry['dependent']}]"
    return f"{t}:{sorted(entry.items())}"
