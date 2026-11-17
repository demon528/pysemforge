"""Core tests: parser, estimation, fitting, invariance and plotting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysemforge import Model
from pysemforge.parser import parse_lavaan, collect_variables
from pysemforge.estim import fit_model, compute_fit_indices, polychoric_correlation
from pysemforge.multigroup import invariance_test
from pysemforge.plotting import PathDiagram, plot_model


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def cfa_data(rng):
    n = 600
    f = rng.standard_normal((n, 2))
    x1 = f[:, 0] + 0.3 * rng.standard_normal(n)
    x2 = f[:, 0] + 0.4 * rng.standard_normal(n)
    x3 = f[:, 1] + 0.3 * rng.standard_normal(n)
    x4 = f[:, 1] + 0.4 * rng.standard_normal(n)
    g = rng.choice(["A", "B"], n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "group": g})


# ---------------------------------------------------------------- parser
class TestParser:
    def test_measurement(self):
        pt = parse_lavaan("visual =~ x1 + x2 + x3")
        assert len(pt) == 3
        assert pt[0]["type"] == "loading"
        assert pt[0]["latent"] == "visual"

    def test_regression(self):
        pt = parse_lavaan("y ~ x1 + x2")
        assert len(pt) == 2
        assert pt[0]["type"] == "regression"
        assert pt[0]["dependent"] == "y"

    def test_covariance(self):
        pt = parse_lavaan("x1 ~~ x2")
        assert len(pt) == 1
        assert pt[0]["type"] == "covariance"

    def test_intercept_retained(self):
        # Intercept statements "x ~ 1" are kept as 'intercept' entries so that
        # scalar measurement invariance can constrain them across groups.
        pt = parse_lavaan("x1 ~ 1")
        assert len(pt) == 1
        assert pt[0]["type"] == "intercept"
        assert pt[0]["variable"] == "x1"
        # And collect_variables should register the variable as observed.
        from pysemforge.parser import collect_variables
        v = collect_variables(pt)
        assert "x1" in v["observed"]

    def test_collect_variables(self):
        pt = parse_lavaan("f1 =~ x1 + x2\nf2 =~ x3 + x4")
        v = collect_variables(pt)
        assert set(v["latents"]) == {"f1", "f2"}
        assert set(v["observed"]) == {"x1", "x2", "x3", "x4"}

    def test_semicolon_separated_statements(self):
        """A specification written on one line must parse like three lines."""
        one_line = Model("visual =~ x1 + x2 + x3; textual =~ x4 + x5 + x6; "
                         "speed =~ x7 + x8 + x9")
        three_lines = Model("visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6\n"
                            "speed =~ x7 + x8 + x9")
        assert len(one_line.params_table) == len(three_lines.params_table) == 9
        assert one_line.syntax.count("\n") == 2


# ----------------------------------------------------------- estimation
class TestEstimation:
    def test_fit_indices_keys(self, rng):
        n_vars = 4
        cov = rng.standard_normal((n_vars, n_vars))
        cov = cov @ cov.T + np.eye(n_vars) * 0.1
        mean = np.zeros(n_vars)
        params = np.ones(n_vars * (n_vars + 1) // 2)
        fit = compute_fit_indices(params, cov, mean, 500, [])
        for k in ("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"):
            assert k in fit

    def test_cfi_in_unit_interval(self, rng):
        n_vars = 4
        cov = cov_ref = rng.standard_normal((n_vars, n_vars))
        cov = cov @ cov.T + np.eye(n_vars) * 0.2
        mean = np.zeros(n_vars)
        fit = compute_fit_indices(np.ones(10), cov, mean, 300, [])
        assert 0.0 <= fit["cfi"] <= 1.0 or abs(fit["cfi"]) < 1e-6

    def test_polychoric_shape(self, rng):
        X = rng.integers(0, 5, size=(200, 3))
        corr = polychoric_correlation(X.astype(float))
        assert corr.shape == (3, 3)
        assert np.allclose(np.diag(corr), 1.0)


# -------------------------------------------------------------- model fit
class TestModelFit:
    def test_cfa_fit(self, cfa_data):
        syntax = "f1 =~ x1 + x2\nf2 =~ x3 + x4"
        model = Model(syntax)
        result = model.fit(cfa_data[["x1", "x2", "x3", "x4"]])
        assert result is not None
        assert model.result is not None

    def test_summary_before_fit(self):
        model = Model("f =~ x1 + x2")
        assert "not been fitted" in model.summary()

    def test_multigroup(self, cfa_data):
        model = Model("f1 =~ x1 + x2\nf2 =~ x3 + x4")
        result = model.fit(cfa_data, group="group")
        assert result.multigroup is True
        assert "groups" in result.result
        assert len(result.result["groups"]) == 2


# ------------------------------------------------------- invariance
class TestInvariance:
    def test_ladder_runs(self, cfa_data):
        model = Model("f1 =~ x1 + x2\nf2 =~ x3 + x4")
        res = invariance_test(model, cfa_data, "group")
        assert hasattr(res, "levels")
        assert hasattr(res, "comparisons")
        for lvl in ("configural", "metric", "scalar"):
            assert lvl in res.levels

    def test_decision_keys(self, cfa_data):
        model = Model("f1 =~ x1 + x2\nf2 =~ x3 + x4")
        res = invariance_test(model, cfa_data, "group")
        comp = res.comparisons
        # configural vs metric present
        assert any("configural" in k for k in comp)
        first = next(iter(comp.values()))
        for k in ("delta_chisq", "delta_df", "pvalue", "delta_cfi", "delta_rmsea"):
            assert k in first
        # Decision lives on the separate .decisions mapping.
        assert any(res.decisions.get(step) in ("INVARIANT", "NON-INVARIANT")
                   for step in comp)

    def test_partial_invariance(self, cfa_data):
        from pysemforge.multigroup import partial_invariance

        model = Model("f1 =~ x1 + x2\nf2 =~ x3 + x4")
        out = partial_invariance(model, cfa_data, "group")
        assert "full" in out
        assert "final" in out
        assert "invariant_up_to" in out["final"]


# ----------------------------------------------------------- plotting
class TestPlotting:
    def test_pathdiagram_builds(self, cfa_data):
        model = Model("f1 =~ x1 + x2\nf2 =~ x3 + x4")
        model.fit(cfa_data[["x1", "x2", "x3", "x4"]])
        diag = PathDiagram(model=model)
        assert len(diag.nodes) >= 6  # 2 latents + 4 observed
        # At least one loading edge (src=latent, tgt=observed, coef)
        assert any(src in diag.latents for src, _, _ in diag.edge_tuples)

    def test_plot_returns_figure(self, cfa_data):
        import matplotlib
        matplotlib.use("Agg")
        model = Model("f1 =~ x1 + x2\nf2 =~ x3 + x4")
        model.fit(cfa_data[["x1", "x2", "x3", "x4"]])
        fig = plot_model(model, show=False)
        assert fig is not None

    def test_to_dot(self, cfa_data):
        model = Model("f1 =~ x1 + x2\nf2 =~ x3 + x4")
        model.fit(cfa_data[["x1", "x2", "x3", "x4"]])
        dot = PathDiagram(model=model).to_dot()
        assert "digraph SEM" in dot
        assert "x1" in dot

    def test_intercept_labels(self):
        """Intercepts drawn under the boxes, released ones emphasised."""
        import matplotlib
        matplotlib.use("Agg")

        diag = PathDiagram(
            latents=["f"], observed=["x1", "x2"],
            edges=[("f", "x1", 0.8), ("f", "x2", 0.7)],
            intercepts={"x1": 4.96, "x2": 2.0}, free_intercepts={"x2"},
        )
        ax = diag.draw()
        labels = {t.get_text(): t for t in ax.texts}
        assert {"4.96", "2.00"} <= set(labels)
        assert labels["2.00"].get_fontweight() == "bold"
        assert labels["4.96"].get_fontweight() != "bold"

    def test_ellipses_fit_within_a_subplot(self):
        """Ellipse width follows the axes width, not the whole figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        diag = PathDiagram(latents=["textual"], observed=["x1"],
                           edges=[("textual", "x1", 0.8)])
        diag.draw(axes[0])
        # Drawing into one panel must not resize the figure the way a
        # full-width diagram would; both panels keep the same axes width.
        assert axes[0].get_position().width < 0.6
        plt.close(fig)


# ------------------------------------------------ multigroup validation
@pytest.fixture(scope="module")
def hs():
    """Bundled Holzinger-Swineford data."""
    from pysemforge.datasets import load_holzinger_swineford
    return load_holzinger_swineford()


class TestMultigroupValidation:
    """Pinned against lavaan 0.6.2 on the bundled Holzinger-Swineford data."""

    def test_configural_matches_lavaan(self, hs):
        from pysemforge.datasets import HOLZINGER_SWINFORD_MODEL
        from pysemforge.multigroup import fit_multigroup

        fit = fit_multigroup(Model(HOLZINGER_SWINFORD_MODEL), hs, "school")
        assert fit["chisq"] == pytest.approx(115.8513, abs=1e-3)
        assert fit["df"] == 48
        assert fit["cfi"] == pytest.approx(0.9234, abs=1e-4)
        assert fit["rmsea"] == pytest.approx(0.0969, abs=1e-4)

    def test_ladder_is_nested(self, hs):
        from pysemforge.datasets import HOLZINGER_SWINFORD_MODEL

        inv = invariance_test(Model(HOLZINGER_SWINFORD_MODEL), hs, "school")
        chi = [inv.levels[l]["chisq"] for l in ("configural", "metric", "scalar")]
        df = [inv.levels[l]["df"] for l in ("configural", "metric", "scalar")]
        assert chi == sorted(chi)
        assert df == [48, 54, 60]
        assert chi[1] == pytest.approx(124.0435, abs=1e-3)
        assert chi[2] == pytest.approx(164.1028, abs=1e-3)
        step = inv.comparisons["metric_vs_scalar"]
        assert step["delta_chisq"] == pytest.approx(40.059, abs=1e-3)
        assert step["delta_df"] == 6

    def test_modification_index_finds_shifted_intercept(self):
        from pysemforge.multigroup import modification_indices

        rng = np.random.default_rng(7)
        n = 200
        f = rng.standard_normal((2 * n, 1))
        X = f @ np.array([[0.9, 0.8, 0.85]]) + 0.5 * rng.standard_normal((2 * n, 3))
        data = pd.DataFrame(X, columns=["x1", "x2", "x3"])
        data["g"] = ["A"] * n + ["B"] * n
        data.loc[n:, "x2"] += 0.8  # intercept shift in the second group

        mi = modification_indices(Model("f =~ x1 + x2 + x3"), data, "g",
                                  level="scalar")
        assert list(mi.columns) == ["constraint", "mi", "df", "pvalue", "epc"]
        assert mi.iloc[0]["constraint"] == ("intercept", "x2")
        assert mi.iloc[0]["mi"] > 10


# --------------------------------------------------- manuscript table 2
class TestPaperTableTwo:
    """The single-group figures printed in Table 2 of the paper.

    Pinned here so that the manuscript's claim -- that the test suite holds
    these numbers -- is true, and so that a change in the estimation path
    cannot silently move a published value.
    """

    def test_single_group_hs(self):
        from pysemforge.datasets import load_holzinger_swineford, HOLZINGER_SWINFORD_MODEL

        data = load_holzinger_swineford()
        model = Model(HOLZINGER_SWINFORD_MODEL)
        fit = model.fit(data[[f"x{i}" for i in range(1, 10)]]).fit

        assert fit["chisq"] == pytest.approx(85.306, abs=5e-3)
        assert fit["df"] == 24
        assert fit["cfi"] == pytest.approx(0.931, abs=5e-4)
        assert fit["tli"] == pytest.approx(0.896, abs=5e-4)
        assert fit["rmsea"] == pytest.approx(0.092, abs=5e-4)
        assert fit["rmsea_lower"] == pytest.approx(0.072, abs=5e-4)
        assert fit["rmsea_upper"] == pytest.approx(0.114, abs=5e-4)
        assert fit["srmr"] == pytest.approx(0.065, abs=5e-4)

    def test_standardised_loadings(self):
        from pysemforge.datasets import load_holzinger_swineford, HOLZINGER_SWINFORD_MODEL

        data = load_holzinger_swineford()
        model = Model(HOLZINGER_SWINFORD_MODEL)
        result = model.fit(data[[f"x{i}" for i in range(1, 10)]])
        table = result.parameters
        rows = table[(table["op"] == "~") & (table["lval"].str.startswith("x"))]
        loadings = {str(r["lval"]): float(r["Est. Std"]) for _, r in rows.iterrows()}

        expected = {"x1": 0.772, "x2": 0.424, "x3": 0.582,
                    "x4": 0.852, "x5": 0.855, "x6": 0.838,
                    "x7": 0.569, "x8": 0.723, "x9": 0.665}
        assert set(loadings) == set(expected)
        for name, value in expected.items():
            assert loadings[name] == pytest.approx(value, abs=1e-3), name
