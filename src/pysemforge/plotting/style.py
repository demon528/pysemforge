"""Style presets for path diagrams."""

from __future__ import annotations

default_style = {
    "latent_color": "#6C8EBF",
    "observed_color": "#FFFFFF",
    "edge_color": "#333333",
    "significant_color": "#C0392B",
    "latent_radius": 0.55,
    "observed_radius": 0.42,
    "font_size": 10,
    "edge_width": 1.4,
    "arrow_head": 0.02,
    "show_coefficients": True,
    "dpi": 150,
}


publication_style = dict(default_style, **{
    "font_size": 9,
    "edge_width": 1.0,
    "latent_color": "#E8E8E8",
    "significant_color": "#000000",
    "dpi": 300,
})
