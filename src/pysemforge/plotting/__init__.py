"""Graphviz-free path-diagram rendering (matplotlib backend).

Public API
----------
- :func:`plot_model` -- draw a fitted :class:`pysemforge.model.Model`.
- :class:`PathDiagram` -- lower-level diagram builder.
- :data:`default_style` / :data:`publication_style` -- style presets.
"""

from .plot import plot_model
from .diagram import PathDiagram
from .style import default_style, publication_style

__all__ = ["plot_model", "PathDiagram", "default_style", "publication_style"]
