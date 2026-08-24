"""Plotly figures for measured, inferred, and posterior-derived profiles."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import QuantityData
from .gp import PosteriorSummary


def _band(
    fig: go.Figure,
    x: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    color: str,
    label: str,
    row: int,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([high, low[::-1]]),
            fill="toself",
            fillcolor=color,
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            name=label,
            legendgroup=f"band-{row}",
            showlegend=row == 1,
        ),
        row=row,
        col=1,
    )


def _unstable_intervals(x: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    if not np.any(mask):
        return []
    half_step = 0.5 * float(np.median(np.diff(x)))
    indices = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    stops = np.r_[indices[breaks], indices[-1]]
    return [
        (max(float(x[0]), float(x[start] - half_step)), min(float(x[-1]), float(x[stop] + half_step)))
        for start, stop in zip(starts, stops)
    ]


def make_profile_figure(
    quantity: QuantityData,
    fit: PosteriorSummary,
    derived_kind: str,
) -> go.Figure:
    """Build the shared-radius profile, derivative, and scale diagnostic figure."""

    confidence_percent = round(100 * fit.confidence)
    probability_panel = False
    if derived_kind == "Scale length  1/[d(log f)/dR]":
        derived_median = fit.scale_length_median
        derived_low = fit.scale_length_low
        derived_high = fit.scale_length_high
        unstable = fit.scale_unstable
        derived_title = "Scale length  1 / [d log f/dR]"
        derived_units = "cm"
    elif derived_kind == "Log gradient  d(log f)/dR":
        derived_median = fit.inverse_scale_median
        derived_low = fit.inverse_scale_low
        derived_high = fit.inverse_scale_high
        unstable = fit.scale_unstable
        derived_title = "Logarithmic gradient  d log f/dR"
        derived_units = "cm⁻¹"
    elif derived_kind == "Gradient sign probability":
        derived_median = fit.gradient_positive_probability
        derived_low = derived_median
        derived_high = derived_median
        unstable = fit.scale_unstable
        derived_title = "Posterior probability  P(d log f/dR > 0)"
        derived_units = "probability"
        probability_panel = True
    else:
        raise ValueError(f"Unknown derived quantity {derived_kind!r}.")

    line_color = "#f97316" if quantity.key == "te" else "#2563eb"
    band_color = "rgba(249,115,22,0.20)" if quantity.key == "te" else "rgba(37,99,235,0.20)"

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.075,
        subplot_titles=(
            f"{quantity.title} posterior",
            f"Gradient  d{quantity.symbol}/dR",
            derived_title,
        ),
    )
    _band(
        fig,
        fit.radius_cm,
        fit.profile_low,
        fit.profile_high,
        band_color,
        f"{confidence_percent}% credible band",
        row=1,
    )
    fig.add_trace(
        go.Scatter(
            x=fit.radius_cm,
            y=fit.profile_median,
            mode="lines",
            line={"color": line_color, "width": 2.5},
            name="Posterior median",
            legendgroup="posterior",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=fit.observed_radius_cm,
            y=fit.observed_values,
            mode="markers",
            marker={"color": "#111827", "size": 6},
            error_y={
                "type": "data",
                "array": fit.observed_errors,
                "visible": True,
                "color": "#4b5563",
                "thickness": 1,
                "width": 2,
            },
            name="MPTS measurement",
        ),
        row=1,
        col=1,
    )
    if fit.edge_observation is not None:
        edge_r, edge_y, edge_sigma = fit.edge_observation
        fig.add_trace(
            go.Scatter(
                x=[edge_r],
                y=[edge_y],
                mode="markers",
                marker={"color": "#7c3aed", "size": 10, "symbol": "diamond"},
                error_y={"type": "data", "array": [edge_sigma], "visible": True},
                name="External edge observation",
            ),
            row=1,
            col=1,
        )

    _band(
        fig,
        fit.radius_cm,
        fit.gradient_low,
        fit.gradient_high,
        band_color,
        f"{confidence_percent}% gradient band",
        row=2,
    )
    fig.add_trace(
        go.Scatter(
            x=fit.radius_cm,
            y=fit.gradient_median,
            mode="lines",
            line={"color": line_color, "width": 2.3},
            name="Gradient median",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    if not probability_panel:
        _band(
            fig,
            fit.radius_cm,
            derived_low,
            derived_high,
            band_color,
            f"{confidence_percent}% derived band",
            row=3,
        )
    fig.add_trace(
        go.Scatter(
            x=fit.radius_cm,
            y=derived_median,
            mode="lines",
            connectgaps=False,
            line={"color": line_color, "width": 2.3},
            name=("Sign probability" if probability_panel else "Derived median"),
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    for x0, x1 in _unstable_intervals(fit.radius_cm, unstable):
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="#9ca3af",
            opacity=0.18,
            line_width=0,
            row=3,
            col=1,
        )

    for x0, x1 in _unstable_intervals(
        fit.radius_cm, fit.derivative_boundary_limited
    ):
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="#64748b",
            opacity=0.10,
            line_width=0,
            row=2,
            col=1,
        )

    fig.add_hline(y=0.0, line={"color": "#9ca3af", "width": 1, "dash": "dot"}, row=2, col=1)
    if probability_panel:
        tail = 0.5 * (1.0 - fit.confidence)
        for threshold in (tail, 1.0 - tail):
            fig.add_hline(
                y=threshold,
                line={"color": "#9ca3af", "width": 1, "dash": "dot"},
                row=3,
                col=1,
            )
        fig.update_yaxes(range=[-0.02, 1.02], tickformat=".0%", row=3, col=1)
    else:
        fig.add_hline(y=0.0, line={"color": "#9ca3af", "width": 1, "dash": "dot"}, row=3, col=1)
    fig.update_yaxes(title_text=quantity.units, row=1, col=1)
    fig.update_yaxes(title_text=f"{quantity.units}/cm", row=2, col=1)
    fig.update_yaxes(title_text=derived_units, row=3, col=1)
    fig.update_yaxes(exponentformat="e")
    fig.update_xaxes(title_text="Major radius R (cm)", row=3, col=1)
    fig.update_layout(
        height=900,
        margin={"l": 40, "r": 25, "t": 55, "b": 30},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        template="plotly_white",
    )
    return fig
