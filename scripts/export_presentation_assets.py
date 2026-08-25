"""Export slide-ready diagrams for the MPTS Bayesian-profile presentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "presentation"

UW_RED = "#C5050C"
CHARCOAL = "#333333"
MID_GRAY = "#767676"
LIGHT_GRAY = "#E7E7E7"
PALE_RED = "#FFF2F2"


def setup_canvas(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rounded_box(
    ax,
    xy,
    width,
    height,
    title,
    subtitle,
    accent=False,
    title_size=18,
    subtitle_size=15,
):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=PALE_RED if accent else "white",
        edgecolor=UW_RED if accent else MID_GRAY,
        linewidth=2.6,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.62,
        title,
        ha="center",
        va="center",
        color=CHARCOAL,
        fontsize=title_size,
        fontweight="bold",
    )
    ax.text(
        x + width / 2,
        y + height * 0.31,
        subtitle,
        ha="center",
        va="center",
        color=CHARCOAL,
        fontsize=subtitle_size,
        linespacing=1.25,
    )


def bayesian_update_flow() -> None:
    fig, ax = setup_canvas(11.2, 4.0)
    rounded_box(
        ax,
        (0.025, 0.35),
        0.25,
        0.38,
        "Kernel prior",
        "plausible smooth\nspatial profiles",
        title_size=17,
        subtitle_size=14,
    )
    ax.text(0.305, 0.54, "+", ha="center", va="center", fontsize=36, color=CHARCOAL)
    rounded_box(
        ax,
        (0.34, 0.35),
        0.25,
        0.38,
        "Measurements",
        r"$(R_i,\,y_i,\,\sigma_i)$",
        title_size=17,
        subtitle_size=16,
    )
    arrow = FancyArrowPatch(
        (0.61, 0.54),
        (0.72, 0.54),
        arrowstyle="-|>",
        mutation_scale=24,
        linewidth=2.7,
        color=UW_RED,
    )
    ax.add_patch(arrow)
    ax.text(
        0.665,
        0.72,
        "Bayesian\nconditioning",
        ha="center",
        va="center",
        fontsize=12,
        color=CHARCOAL,
    )
    rounded_box(
        ax,
        (0.74, 0.31),
        0.235,
        0.46,
        "Posterior profiles",
        r"$p\!\left(f(R)\mid\mathcal{D}\right)$" "\ncurve + credible band",
        accent=True,
        title_size=16,
        subtitle_size=14,
    )
    ax.text(
        0.465,
        0.17,
        r"Small $\sigma_i$: greater influence     $\bullet$     Large $\sigma_i$: weaker influence",
        ha="center",
        va="center",
        fontsize=16,
        color=MID_GRAY,
    )
    save(fig, "bayesian-update-flow.png")


def positive_profile_model() -> None:
    fig, ax = setup_canvas(10.8, 4.8)
    equations = [
        (r"$z(R)\sim\mathcal{GP}\!\left(\beta,k_{5/2}\right)$", "smooth latent log profile"),
        (r"$f(R)=f_{\mathrm{ref}}e^{z(R)}>0$", "positive temperature or density"),
        (r"$y_i\mid z\sim\mathcal{N}\!\left(f(R_i),\sigma_i^2\right)$", "measurement and reported uncertainty"),
    ]
    ys = [0.80, 0.50, 0.20]
    for index, ((equation, label), y) in enumerate(zip(equations, ys)):
        ax.text(0.42, y, equation, ha="center", va="center", fontsize=26, color=CHARCOAL)
        ax.text(0.76, y, label, ha="left", va="center", fontsize=16, color=MID_GRAY)
        if index < 2:
            arrow = FancyArrowPatch(
                (0.42, y - 0.10),
                (0.42, ys[index + 1] + 0.10),
                arrowstyle="-|>",
                mutation_scale=20,
                linewidth=2.4,
                color=UW_RED,
            )
            ax.add_patch(arrow)
    save(fig, "positive-profile-model.png")


def kernel_length_scale() -> None:
    distance = np.linspace(0.0, 60.0, 500)

    def matern52(ell: float) -> np.ndarray:
        scaled = np.sqrt(5.0) * distance / ell
        return (1.0 + scaled + scaled**2 / 3.0) * np.exp(-scaled)

    fig, ax = plt.subplots(figsize=(10.6, 4.2), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(distance, matern52(6.0), color=UW_RED, linewidth=3.2)
    ax.plot(distance, matern52(20.0), color=CHARCOAL, linewidth=3.2)
    ax.text(10.0, 0.34, r"small $\ell=6$ cm", color=UW_RED, fontsize=16)
    ax.text(27.0, 0.48, r"large $\ell=20$ cm", color=CHARCOAL, fontsize=16)
    ax.set_xlim(0.0, 60.0)
    ax.set_ylim(0.0, 1.04)
    ax.set_xlabel(r"Radial separation $d=|R-R'|$ (cm)", fontsize=16, color=CHARCOAL)
    ax.set_ylabel(r"Correlation $k(d)/k(0)$", fontsize=16, color=CHARCOAL)
    ax.tick_params(axis="both", labelsize=13, colors=CHARCOAL)
    ax.grid(True, color=LIGHT_GRAY, linewidth=1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color(MID_GRAY)
    fig.tight_layout()
    save(fig, "kernel-correlation-length.png")


def posterior_to_physics() -> None:
    fig, ax = setup_canvas(11.4, 4.1)
    steps = [
        (
            "Posterior profile",
            r"$f^{(s)}(R)$",
            "plausible temperature\nor density curves",
        ),
        ("Gradient", r"$f'^{(s)}(R)$", "absolute spatial\nchange"),
        (
            "Log gradient",
            r"$G_R^{(s)}=\dfrac{d\log f^{(s)}}{dR}$",
            "fractional change\nper unit radius",
        ),
        (
            "Scale length",
            r"$L_{f,R}^{(s)}=\dfrac{1}{G_R^{(s)}}$",
            "reported only when\ngradient sign is resolved",
        ),
    ]
    xs = [0.018, 0.265, 0.512, 0.759]
    width = 0.20
    for index, ((title, equation, subtitle), x) in enumerate(zip(steps, xs)):
        rounded_box(
            ax,
            (x, 0.28),
            width,
            0.50,
            title,
            equation,
            accent=index == 3,
            title_size=15,
            subtitle_size=16,
        )
        ax.text(
            x + width / 2,
            0.16,
            subtitle,
            ha="center",
            va="center",
            color=MID_GRAY,
            fontsize=11.5,
            linespacing=1.2,
        )
        if index < len(steps) - 1:
            arrow = FancyArrowPatch(
                (x + width + 0.008, 0.53),
                (xs[index + 1] - 0.008, 0.53),
                arrowstyle="-|>",
                mutation_scale=20,
                linewidth=2.4,
                color=UW_RED,
            )
            ax.add_patch(arrow)
    save(fig, "posterior-to-physics.png")


def main() -> None:
    bayesian_update_flow()
    positive_profile_model()
    kernel_length_scale()
    posterior_to_physics()


if __name__ == "__main__":
    main()
