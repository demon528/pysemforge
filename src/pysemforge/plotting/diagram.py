"""Graphviz-free SEM path diagram backend (pure matplotlib).

R reference
-----------
Mirrors the visual conventions of R's ``semPlot`` / ``lavaanPlot`` without
requiring a system-level Graphviz install:

- latent variables  -> ellipses
- observed indicators -> squares (boxes)
- regressions ``y ~ x`` and loadings ``lv =~ x`` -> single-headed arrows
- covariances  ``a ~~ b`` -> double-headed dashed arrows

Two construction paths:
1. ``PathDiagram(model=fitted_model)``  (lavaanPlot-style one-liner)
2. Explicit ``PathDiagram(latents=..., observed=..., edges=..., covariances=...)``
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Internal edge representation
# ---------------------------------------------------------------------------
class _Edge:
    __slots__ = ("src", "dst", "coef")

    def __init__(self, src: str, dst: str, coef: Optional[float] = None):
        self.src = src
        self.dst = dst
        self.coef = coef

    # Make edges iterable as ``(src, dst, coef)`` so tests / callers can do
    # ``for src, dst, coef in diagram.edges``. Also expose ``.kind`` for
    # callers that inspect edge type.
    def __iter__(self):
        return iter((self.src, self.dst, self.coef))

    @property
    def kind(self) -> str:
        return "loading"  # structural edges are loadings/regressions; kind hook


# ---------------------------------------------------------------------------
# Main public class
# ---------------------------------------------------------------------------
class PathDiagram:
    """Layout and render an SEM path diagram on a matplotlib Axes.

    Parameters
    ----------
    latents : list of str, optional
        Latent (ellipse) nodes.
    observed : list of str, optional
        Observed (square) nodes.
    edges : list of (src, dst, coef), optional
        Directed paths with optional coefficient label.
    covariances : list of (a, b), optional
        Bidirectional (curved) covariance paths.
    style : dict, optional
        Style preset (see :mod:`pysemforge.plotting.style`).
    model : pysemforge.model.Model, optional
        A fitted model.  When given, ``latents``/``observed``/``edges`` are
        built automatically from its parameter table.
    """

    def __init__(
        self,
        latents: Optional[List[str]] = None,
        observed: Optional[List[str]] = None,
        edges: Optional[List[Tuple[str, str, Optional[float]]]] = None,
        covariances: Optional[List[Tuple]] = None,
        style: Optional[dict] = None,
        model: Optional[object] = None,
        intercepts: Optional[Dict[str, float]] = None,
        free_intercepts: Optional[object] = None,
    ):
        if model is not None:
            latents, observed, edges, covariances = self._from_model(model)

        # Indicator intercepts, drawn under each observed box.  ``free_intercepts``
        # names the ones that are estimated separately per group, which are
        # drawn in the accent colour so that a partial invariance solution shows
        # at a glance which indicators were released.
        self.intercepts: Dict[str, float] = dict(intercepts or {})
        self.free_intercepts = set(free_intercepts or [])

        self.latents: List[str] = list(latents or [])
        self.observed: List[str] = list(observed or [])
        # Accept both ``(a, b)`` and ``(a, b, coef)``.
        self.covariances: List[Tuple[str, str, Optional[float]]] = [
            (c[0], c[1], c[2] if len(c) > 2 else None) for c in (covariances or [])
        ]
        self.style: dict = dict(style or {})
        self._edges: List[_Edge] = [_Edge(s, d, c) for (s, d, c) in (edges or [])]

    # Public accessors ---------------------------------------------------------
    @property
    def edges(self) -> List[Tuple[str, str, Optional[float]]]:
        """Edges as ``(source, target, coefficient)`` tuples."""
        return [(e.src, e.dst, e.coef) for e in self._edges]

    # ---- model introspection -------------------------------------------------
    @staticmethod
    def _from_model(model) -> Tuple[List[str], List[str], List[Tuple[str, str, Optional[float]]], List[Tuple]]:
        """Build node/edge lists from a pysemforge Model's params_table."""
        from pysemforge.parser import parse_lavaan  # absolute import (PYTHONPATH=src)

        params = getattr(model, "params_table", []) or []
        if not params and getattr(model, "syntax", None):
            params = parse_lavaan(model.syntax)

        # Pass 1: every name on the LHS of ``=~`` is latent. Doing this first
        # prevents ``f1 ~~ f2`` from classifying f1/f2 as observed.
        latent_names = {p["latent"] for p in params if p.get("type") == "loading"}

        try:
            from pysemforge.plotting.plot import _fitted_coefficients

            coef_by_entry = _fitted_coefficients(model, standardized=False)
        except Exception:  # pragma: no cover - plotting must never hard-fail
            coef_by_entry = {}

        latents: List[str] = []
        observed: List[str] = []
        edges: List[Tuple[str, str, Optional[float]]] = []
        covariances: List[Tuple] = []
        seen_lat: set = set()
        seen_obs: set = set()

        def add_lat(n: str):
            if n not in seen_lat:
                seen_lat.add(n)
                latents.append(n)

        def add_obs(n: str):
            if n not in seen_lat and n not in latent_names and n not in seen_obs:
                seen_obs.add(n)
                observed.append(n)

        for i, p in enumerate(params):
            pt = p.get("type")
            coef = coef_by_entry.get(i)
            if pt == "loading":
                add_lat(p["latent"])
                add_obs(p["indicator"])
                edges.append((p["latent"], p["indicator"], coef))
            elif pt == "regression":
                add_obs(p["independent"])
                add_obs(p["dependent"])
                edges.append((p["independent"], p["dependent"], coef))
            elif pt == "covariance":
                add_obs(p["var1"])
                add_obs(p["var2"])
                covariances.append((p["var1"], p["var2"], coef))

        return latents, observed, edges, covariances

    # ---- convenience accessors ---------------------------------------------
    @property
    def nodes(self) -> List[dict]:
        """All nodes as ``{"name", "kind"}`` dicts (latents + observed)."""
        out = [{"name": n, "kind": "latent"} for n in self.latents]
        out += [{"name": n, "kind": "observed"} for n in self.observed]
        return out

    @property
    def edge_tuples(self) -> List[Tuple[str, str, Optional[float]]]:
        """Edges as ``(src, dst, coef)`` tuples."""
        return [(e.src, e.dst, e.coef) for e in self._edges]

    # ---- layout -------------------------------------------------------------
    def _layout(self) -> Dict[str, Tuple[float, float]]:
        """Layout: each latent sits above the centre of its own indicator block.

        Blocks are allotted horizontal space proportional to the number of
        indicators they contain. A fixed per-latent spacing makes a 3-indicator
        factor collide with a neighbouring 2-indicator factor.
        """
        pos: Dict[str, Tuple[float, float]] = {}
        if not self.latents:
            n = len(self.observed)
            for k, name in enumerate(self.observed):
                pos[name] = ((k - (n - 1) / 2) * 1.8, 0.0)
            return pos

        # Assign each observed node to its parent latent (first loading edge).
        parent: Dict[str, str] = {}
        for src, dst, _ in self.edge_tuples:
            if src in self.latents and dst in self.observed:
                parent.setdefault(dst, src)

        children: Dict[str, List[str]] = {l: [] for l in self.latents}
        orphans: List[str] = []
        for obs in self.observed:
            p = parent.get(obs)
            if p in children:
                children[p].append(obs)
            else:
                orphans.append(obs)

        gap = 1.15  # horizontal gap between adjacent factor blocks
        spacing = 1.75
        widths = [max(len(children[l]), 1) * spacing for l in self.latents]
        total = sum(widths) + gap * (len(self.latents) - 1)

        cursor = -total / 2
        for lat, w in zip(self.latents, widths):
            cx = cursor + w / 2
            pos[lat] = (cx, 2.2)
            kids = children[lat]
            for j, obs in enumerate(kids):
                pos[obs] = (cx + (j - (len(kids) - 1) / 2) * spacing, 0.0)
            cursor += w + gap

        # Orphans go on a bottom row, centred under the whole figure.
        for k, name in enumerate(orphans):
            pos[name] = ((k - (len(orphans) - 1) / 2) * spacing, -2.2)
        return pos

    # ---- rendering ----------------------------------------------------------
    def draw(self, ax=None, show: bool = False):
        """Draw the diagram onto ``ax`` (creates one if ``ax`` is None)."""
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, Rectangle

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        s = self.style
        pos = self._layout()
        show_coef = s.get("show_coefficients", True)
        edge_color = s.get("edge_color", "#333333")
        sig_color = s.get("significant_color", "#C0392B")
        fs = s.get("font_size", 10)
        r_obs = s.get("observed_radius", 0.42)
        # Latent variables are ellipses, not circles: it is the drawing
        # convention of the field, and the only shape that holds a
        # multi-character name without the label spilling over the outline.
        # The width follows the rendered text, so it has to account for the
        # font size and for how many points one data unit occupies.
        longest = max((len(n) for n in self.latents), default=4)
        span_x = max((p[0] for p in pos.values()), default=0.0) - min(
            (p[0] for p in pos.values()), default=0.0
        ) + 1.24
        # The width follows the rendered text, so it has to account for the
        # font size and for how many points one data unit occupies.  The axes
        # width is used rather than the figure width, so that a diagram drawn
        # into one panel of a multi-panel figure does not get ellipses sized
        # for the whole figure.
        fig_w_in = ax.get_figure().get_size_inches()[0]
        ax_w_in = ax.get_position().width * fig_w_in
        fig_width_pt = ax_w_in * 72.0 * 0.88
        pt_per_unit = fig_width_pt / max(span_x, 1e-6)
        text_half = 0.30 * fs * longest / pt_per_unit
        hw_lat = max(s.get("latent_half_width", 0.0), text_half + 0.14, 0.62)
        hh_lat = s.get("latent_half_height", 0.50)

        def half_size(name: str) -> Tuple[float, float]:
            return (hw_lat, hh_lat) if name in self.latents else (r_obs, r_obs)

        def boundary(name: str, toward: Tuple[float, float]) -> Tuple[float, float]:
            """Where the ray from ``name`` towards ``toward`` leaves its node.

            Solving this on the actual node outline (rather than shrinking by a
            nominal radius) keeps arrow heads on the border of an ellipse, which
            a single radius cannot describe.
            """
            hw, hh = half_size(name)
            cx, cy = pos[name]
            dx, dy = toward[0] - cx, toward[1] - cy
            t = 1.0 / max(np.hypot(dx / hw, dy / hh), 1e-9)
            return cx + t * dx, cy + t * dy


        # Directed edges (loadings + regressions)
        for src, tgt, coef in self.edge_tuples:
            if src not in pos or tgt not in pos:
                continue
            x1, y1 = pos[src]
            x2, y2 = pos[tgt]
            sx, sy = boundary(src, (x2, y2))
            ex, ey = boundary(tgt, (x1, y1))
            dx, dy = ex - sx, ey - sy
            dist = np.hypot(dx, dy) or 1.0
            color = edge_color
            if show_coef and coef is not None and abs(coef) > 0.05:
                color = sig_color
            ax.annotate(
                "", xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=s.get("edge_width", 1.4), mutation_scale=15,
                                shrinkA=0, shrinkB=0),
            )
            if show_coef and coef is not None:
                # Place the label towards the indicator end of the path, in the
                # clear space above the box, rather than at the midpoint where
                # it collides with the latent outline.
                mx, my = sx + dx * 0.62, sy + dy * 0.62
                ox, oy = -dy / dist * 0.24, dx / dist * 0.24
                ax.text(mx + ox, my + oy, f"{coef:.2f}", fontsize=fs - 1,
                        color=color, ha="center", va="center", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec="none", alpha=0.92))

        # Covariance (bidirectional curved) edges
        for a, b, coef in self.covariances:
            if a not in pos or b not in pos:
                continue
            # Normalise direction left-to-right: with a fixed rad the bulge
            # flips side depending on argument order, which sent one arc down
            # through the indicator row.
            if pos[b][0] < pos[a][0]:
                a, b = b, a
            pa, pb = pos[a], pos[b]
            span = float(np.hypot(pb[0] - pa[0], pb[1] - pa[1]))
            # Arcs leave the ovals at an angle above the horizontal instead of
            # at the point facing each other, so several covariances meeting on
            # one factor do not stack their arrow heads at the same spot.
            ang = 0.62 if span < 5.0 else 0.96
            x1 = pa[0] + hw_lat * np.cos(ang)
            y1 = pa[1] + hh_lat * np.sin(ang)
            x2 = pb[0] - hw_lat * np.cos(ang)
            y2 = pb[1] + hh_lat * np.sin(ang)
            chord = float(np.hypot(x2 - x1, y2 - y1)) or 1.0
            # Non-adjacent factors need a deeper arc, or it cuts through the
            # factor sitting between them.
            rad = -0.22 if span < 5.0 else -0.34
            ax.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="<|-|>", color=edge_color, lw=1.0,
                                linestyle="dashed", mutation_scale=16,
                                shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=5,  # above the node patches, or the heads are hidden
            )
            if show_coef and coef is not None:
                # Apex of a circular arc is offset by |rad| * chord / 2.
                apex = abs(rad) * chord / 2.0
                ax.text((x1 + x2) / 2, (y1 + y2) / 2 + apex + 0.16, f"{coef:.2f}",
                        fontsize=fs - 1, color=edge_color, ha="center", va="center",
                        zorder=6, bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                            ec="none", alpha=0.92))

        # Nodes
        latent_set = set(self.latents)
        for name in self.latents:
            x, y = pos[name]
            ax.add_patch(Ellipse((x, y), width=2 * hw_lat, height=2 * hh_lat,
                                 fc=s.get("latent_color", "#6C8EBF"), ec="k",
                                 lw=1.2, zorder=3))
            ax.text(x, y, name, ha="center", va="center", fontsize=fs, zorder=4)
        for name in self.observed:
            if name in latent_set:
                continue  # never draw a latent twice (oval + box)
            x, y = pos[name]
            ax.add_patch(Rectangle((x - r_obs, y - r_obs), 2 * r_obs, 2 * r_obs,
                                   fc=s.get("observed_color", "#FFFFFF"), ec="k", lw=1.0, zorder=3))
            ax.text(x, y, name, ha="center", va="center", fontsize=fs - 1, zorder=4)
            if name in self.intercepts:
                free = name in self.free_intercepts
                ax.text(
                    x, y - r_obs - 0.10, f"{self.intercepts[name]:.2f}",
                    ha="center", va="top",
                    fontsize=max(fs - 3, 6),
                    color=s.get("free_intercept_color", "#B03A2E") if free
                    else s.get("intercept_color", "#666666"),
                    fontweight="bold" if free else "normal",
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white",
                              ec="none", alpha=0.9),
                )

        ax.set_aspect("equal")
        ax.axis("off")
        ax.autoscale_view()

        if show and ax is None:
            plt.show()
        return ax

    def save(self, filename: str, dpi: Optional[int] = None):
        """Render to ``filename`` (any matplotlib-supported format).

        Vector backends record the current date in the file header by default;
        it is suppressed so that re-running the same analysis produces a
        byte-identical file.
        """
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        self.draw(ax)
        ext = os.path.splitext(filename)[1].lower()
        metadata = {"Date": None} if ext in (".svg", ".svgz") else (
            {"CreationDate": None} if ext == ".pdf" else None
        )
        if ext in (".svg", ".svgz"):
            # matplotlib derives element ids from a random salt; fix it so that
            # two runs of the same analysis produce identical bytes.
            matplotlib.rcParams["svg.hashsalt"] = "pysemforge"
        fig.savefig(
            filename,
            dpi=dpi or self.style.get("dpi", 150),
            bbox_inches="tight",
            metadata=metadata,
        )
        plt.close(fig)

    # ---- graphviz export ----------------------------------------------------
    def to_dot(self) -> str:
        """Return a Graphviz ``.dot`` string (for users who have graphviz)."""
        lines = ["digraph SEM {", "  rankdir=LR;"]
        for name in self.latents:
            lines.append(f'  "{name}" [shape=ellipse];')
        for name in self.observed:
            lines.append(f'  "{name}" [shape=box];')
        for src, tgt, _ in self.edge_tuples:
            lines.append(f'  "{src}" -> "{tgt}";')
        for a, b in self.covariances:
            lines.append(f'  "{a}" -> "{b}" [dir=both, style=dashed];')
        lines.append("}")
        return "\n".join(lines)
