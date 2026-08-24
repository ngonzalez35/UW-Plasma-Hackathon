#!/usr/bin/env python3
"""Export reproducible figures for the project pedagogical companion PDF."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thomson_profiles.data import list_times, load_profile_slice
from thomson_profiles.gp import FitConfig, fit_profile


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "nstx-profiles.hdf5"
OUTPUT_DIR = ROOT / "output" / "pdf" / "assets"
ORANGE = "#e76623"
BAND = "#f6c3a7"
INK = "#253142"
GRID = "#d9dee7"


def _shade_mask(ax: plt.Axes, x: np.ndarray, mask: np.ndarray) -> None:
    """Shade contiguous intervals selected by a Boolean grid mask."""

    if not np.any(mask):
        return
    indices = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    stops = np.r_[indices[breaks], indices[-1]]
    half_step = 0.5 * float(np.median(np.diff(x)))
    for start, stop in zip(starts, stops):
        ax.axvspan(
            max(x[0], x[start] - half_step),
            min(x[-1], x[stop] + half_step),
            color="#9ca3af",
            alpha=0.18,
            linewidth=0,
        )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def export_default_temperature() -> None:
    times = list_times(DATA_PATH, "141687")
    time_index = int(np.argmin(np.abs(times - 0.315)))
    profile = load_profile_slice(DATA_PATH, "141687", time_index)
    quantity = profile.quantity("te")
    fit = fit_profile(
        profile.radius_cm,
        quantity.values,
        quantity.errors,
        FitConfig(
            n_samples=400,
            grid_size=220,
            hyperparameter_samples=24,
            length_quadrature_nodes=9,
            reference_scale=1.0,
            random_seed=int(profile.shot) + 2 * time_index,
        ),
    )

    fig, axes = plt.subplots(4, 1, figsize=(7.25, 8.3), sharex=True)
    radius = fit.radius_cm

    axes[0].fill_between(radius, fit.profile_low, fit.profile_high, color=BAND)
    axes[0].plot(radius, fit.profile_median, color=ORANGE, linewidth=2.0)
    axes[0].errorbar(
        fit.observed_radius_cm,
        fit.observed_values,
        yerr=fit.observed_errors,
        fmt="o",
        markersize=3.4,
        color=INK,
        ecolor="#566273",
        elinewidth=0.75,
        capsize=1.5,
        label="MPTS measurements",
    )
    axes[0].set_ylabel(r"$T_e$ (keV)")
    axes[0].set_title(
        "Shot 141687 at 0.3150 s: positive latent-log GP reconstruction",
        fontsize=11,
        color=INK,
        pad=8,
    )
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    axes[1].fill_between(radius, fit.gradient_low, fit.gradient_high, color=BAND)
    axes[1].plot(radius, fit.gradient_median, color=ORANGE, linewidth=1.8)
    axes[1].axhline(0.0, color="#77808e", linestyle=":", linewidth=0.8)
    _shade_mask(axes[1], radius, fit.derivative_boundary_limited)
    axes[1].set_ylabel(r"$dT_e/dR$ (keV/cm)")

    axes[2].fill_between(
        radius, fit.inverse_scale_low, fit.inverse_scale_high, color=BAND
    )
    axes[2].plot(radius, fit.inverse_scale_median, color=ORANGE, linewidth=1.8)
    axes[2].axhline(0.0, color="#77808e", linestyle=":", linewidth=0.8)
    _shade_mask(axes[2], radius, fit.scale_unstable)
    axes[2].set_ylabel(r"$d\log T_e/dR$ (cm$^{-1}$)")

    finite = np.isfinite(fit.scale_length_median)
    axes[3].fill_between(
        radius,
        fit.scale_length_low,
        fit.scale_length_high,
        where=finite,
        color=BAND,
    )
    axes[3].plot(radius, fit.scale_length_median, color=ORANGE, linewidth=1.8)
    axes[3].axhline(0.0, color="#77808e", linestyle=":", linewidth=0.8)
    _shade_mask(axes[3], radius, fit.scale_unstable)
    finite_values = np.concatenate(
        [
            fit.scale_length_low[np.isfinite(fit.scale_length_low)],
            fit.scale_length_high[np.isfinite(fit.scale_length_high)],
        ]
    )
    if finite_values.size:
        limit = max(20.0, float(np.percentile(np.abs(finite_values), 90)))
        axes[3].set_ylim(-1.15 * limit, 1.15 * limit)
    axes[3].set_ylabel(r"$L_{T_e,R}$ (cm)")
    axes[3].set_xlabel(r"Geometric major radius $R$ (cm)")

    for axis in axes:
        _style_axis(axis)
    fig.text(
        0.5,
        0.006,
        (
            "Orange line: posterior median; orange band: 95% pointwise credible interval; "
            "gray: derivative-limited or sign-unresolved radius."
        ),
        ha="center",
        fontsize=7.5,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.03, 0.03, 0.995, 0.995), h_pad=0.75)
    fig.savefig(
        OUTPUT_DIR / "default_temperature_posterior.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def export_model_comparison() -> None:
    categories = [
        "Negative posterior median",
        "Optimizer warnings",
        "Length at optimizer bound",
    ]
    linear = np.asarray([590, 286, 13])
    positive = np.asarray([0, 0, 0])
    y = np.arange(len(categories))
    height = 0.28

    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    bars_linear = ax.barh(
        y + height / 2,
        linear,
        height,
        color="#64748b",
        label="Linear-space GP baseline",
    )
    bars_positive = ax.barh(
        y - height / 2,
        positive,
        height,
        color=ORANGE,
        label="Positive latent-log GP",
    )
    for bars in (bars_linear, bars_positive):
        for bar in bars:
            value = int(bar.get_width())
            ax.text(
                max(value, 0) + 12,
                bar.get_y() + bar.get_height() / 2,
                str(value),
                ha="left",
                va="center",
                fontsize=7,
                color=INK,
            )
    ax.set_yticks(y, categories)
    ax.invert_yaxis()
    ax.set_xlabel("Number of fits", fontsize=8)
    ax.set_xlim(0, 660)
    ax.set_title(
        "Robustness sweep: 1,486 profile fits",
        fontsize=9,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    _style_axis(ax)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "model_robustness_comparison.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_default_temperature()
    export_model_comparison()
    print(f"Wrote pedagogical figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
