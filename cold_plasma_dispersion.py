"""Plot the two cold-plasma (Stix) dispersion branches.

The equations use
    x = omega / omega_ce,       n^2 = g_1 or g_2,
    k = (omega/c) sqrt(n^2),    theta = angle(k, B0),
    k_parallel = k cos(theta),  k_perp = k sin(theta).

Run, for example:
    python cold_plasma_dispersion.py
    python cold_plasma_dispersion.py --density-exp 18 --B0 1 --theta 45
    python cold_plasma_dispersion.py --fixed-x 0.2 --save dispersion.png

Requires NumPy and Matplotlib.
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider


C = 299_792_458.0                 # speed of light [m/s]
E = 1.602_176_634e-19             # elementary charge [C]
M_E = 9.109_383_7139e-31          # electron mass [kg]


def stix_branches(x, theta, q, u=1.0 / 1836.15):
    """Return (g1, g2) = the two n^2 roots.

    Parameters can be scalars or broadcast-compatible NumPy arrays.  theta is
    in radians. Values with a negative discriminant are returned as NaN.
    """
    x = np.asarray(x, dtype=float)
    theta = np.asarray(theta, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        p = 1.0 - q * (1.0 + u) / x**2
        r = 1.0 - q / (x * (x - 1.0)) - u * q / (x * (x + u))
        ell = 1.0 - q / (x * (x + 1.0)) - u * q / (x * (x - u))
        s = 0.5 * (r + ell)

        cos2 = np.cos(theta) ** 2
        sin2 = np.sin(theta) ** 2
        a = p * cos2 + s * sin2
        b = r * ell * sin2 + p * s * (1.0 + cos2)
        ccoef = p * r * ell

        discriminant = b**2 - 4.0 * a * ccoef
        root = np.sqrt(np.where(discriminant >= 0.0, discriminant, np.nan))
        g1 = (b + root) / (2.0 * a)
        g2 = (b - root) / (2.0 * a)

    return g1, g2


def stix_coefficients(x, theta, q, u=1.0 / 1836.15):
    """Return P, R, L, S and A for cutoff/resonance identification."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        p = 1.0 - q * (1.0 + u) / x**2
        r = 1.0 - q / (x * (x - 1.0)) - u * q / (x * (x + u))
        ell = 1.0 - q / (x * (x + 1.0)) - u * q / (x * (x - u))
        s = 0.5 * (r + ell)
        a = p * np.cos(theta)**2 + s * np.sin(theta)**2
    return p, r, ell, s, a


def zero_crossings(x, y, max_abs=200.0):
    """Linearly locate real zero crossings without crossing singular jumps."""
    x, y = np.asarray(x), np.asarray(y)
    good = np.isfinite(x[:-1]) & np.isfinite(x[1:])
    good &= np.isfinite(y[:-1]) & np.isfinite(y[1:])
    good &= (np.abs(y[:-1]) < max_abs) & (np.abs(y[1:]) < max_abs)
    indices = np.where(good & (y[:-1] * y[1:] <= 0.0))[0]
    roots = []
    for i in indices:
        if y[i + 1] == y[i]:
            root = x[i]
        else:
            root = x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
        if np.isfinite(root) and root > 0:
            roots.append(root)
    return roots


def unique_log_roots(roots, tolerance=0.015):
    """Merge roots that are indistinguishable on a logarithmic axis."""
    answer = []
    for root in sorted(roots):
        if not answer or abs(np.log(root / answer[-1])) > tolerance:
            answer.append(root)
    return answer


def propagating_k(x, n2, omega_ce):
    """Return physical k [rad/m], masking non-propagating n^2 < 0."""
    n2 = np.asarray(n2, dtype=float)
    valid = np.isfinite(n2) & (n2 >= 0.0)
    return np.where(valid, x * omega_ce * np.sqrt(np.where(valid, n2, 0.0)) / C,
                    np.nan)


def masked_for_plot(y, percentile=99.0):
    """Hide extreme pole spikes so ordinary parts of the curves stay visible."""
    y = np.asarray(y, dtype=float).copy()
    finite = np.abs(y[np.isfinite(y)])
    if finite.size:
        limit = np.percentile(finite, percentile)
        y[np.abs(y) > limit] = np.nan
    return y


def symmetric_wave_surface(first_quadrant_angle, radial_k, percentile=97.5):
    """Return a four-quadrant surface using one consistent radial-k mask.

    Near a resonance, k tends to infinity.  The same radial cutoff must be
    applied to both Cartesian components; clipping k_parallel and k_perp
    separately creates asymmetric spikes and false connecting lines.
    """
    angle = np.asarray(first_quadrant_angle, dtype=float)
    radius = np.asarray(radial_k, dtype=float).copy()
    finite = np.isfinite(radius) & (radius >= 0.0)
    if np.any(finite):
        radial_limit = np.percentile(radius[finite], percentile)
        finite &= radius <= radial_limit

    # Break both sides of large neighbor-to-neighbor jumps. This prevents
    # Matplotlib from drawing a chord across a resonance discontinuity.
    safe = np.where(finite, np.maximum(radius, np.finfo(float).tiny), np.nan)
    ratios = np.maximum(safe[1:] / safe[:-1], safe[:-1] / safe[1:])
    jumps = np.where(np.isfinite(ratios) & (ratios > 4.0))[0]
    for index in jumps:
        finite[index:index + 2] = False
    radius[~finite] = np.nan

    # Stix A, B, and C depend on sin^2(theta) and cos^2(theta), so mirroring
    # this single quadrant is exact. NaN separators stop cross-quadrant chords.
    quadrants = (
        (radius * np.cos(angle), radius * np.sin(angle)),
        (-radius[::-1] * np.cos(angle[::-1]), radius[::-1] * np.sin(angle[::-1])),
        (-radius * np.cos(angle), -radius * np.sin(angle)),
        (radius[::-1] * np.cos(angle[::-1]), -radius[::-1] * np.sin(angle[::-1])),
    )
    x_parts, y_parts = [], []
    for x_part, y_part in quadrants:
        x_parts.extend((x_part, np.array([np.nan])))
        y_parts.extend((y_part, np.array([np.nan])))
    return np.concatenate(x_parts), np.concatenate(y_parts)


def make_figure(d=18.0, B0=1.0, u=1.0 / 1836.15, theta_deg=45.0,
                fixed_x=0.2, x_min=1e-4, x_max=3.0, points=4000):
    """Build and return the four-panel dispersion figure."""
    q = 1.028_822_055 * 10.0 ** (d - 19.0) / B0**2
    omega_ce = E * B0 / M_E
    f_ce = omega_ce / (2.0 * np.pi)
    theta = np.deg2rad(theta_deg)

    # Avoid sampling exactly at the cyclotron poles x=u and x=1.
    x = np.geomspace(x_min, x_max, points)
    pole = (np.abs(x - u) < 0.003 * u) | (np.abs(x - 1.0) < 0.001)
    x[pole] = np.nan
    g1, g2 = stix_branches(x, theta, q, u)
    k1 = propagating_k(x, g1, omega_ce)
    k2 = propagating_k(x, g2, omega_ce)
    freq_hz = x * f_ce

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax_disp, ax_locus, ax_kpar, ax_kperp = axes.flat
    colors = ("tab:blue", "tab:orange")

    # Original normalized plot: y = k c / omega_ce = x sqrt(n^2).
    for gi, color, label in zip((g1, g2), colors, ("branch 1", "branch 2")):
        K = x * np.sqrt(np.where(gi >= 0.0, gi, np.nan))
        ax_disp.plot(x, masked_for_plot(K), color=color, label=label)
    ax_disp.set(xscale="log", xlabel=r"$x=\omega/\omega_{ce}$",
                ylabel=r"$kc/\omega_{ce}$", title="Normalized dispersion")

    # Fixed frequency, scan propagation angle from parallel to perpendicular.
    angle_deg = np.linspace(0.0, 90.0, 721)
    angle = np.deg2rad(angle_deg)
    ga, gb = stix_branches(fixed_x, angle, q, u)
    for gi, color, label in zip((ga, gb), colors, ("branch 1", "branch 2")):
        kval = propagating_k(fixed_x, gi, omega_ce)
        ax_locus.plot(kval * np.cos(angle), kval * np.sin(angle),
                      color=color, label=label)
    ax_locus.set(xlabel=r"$k_\parallel$ [rad m$^{-1}$]",
                 ylabel=r"$k_\perp$ [rad m$^{-1}$]",
                 title=rf"Fixed frequency: $x={fixed_x:g}$ "
                       rf"($f={fixed_x*f_ce:.3g}$ Hz)")

    for kval, color, label in zip((k1, k2), colors, ("branch 1", "branch 2")):
        ax_kpar.plot(freq_hz, masked_for_plot(kval * np.cos(theta)),
                     color=color, label=label)
        ax_kperp.plot(freq_hz, masked_for_plot(kval * np.sin(theta)),
                      color=color, label=label)
    ax_kpar.set(xscale="log", xlabel="frequency [Hz]",
                ylabel=r"$k_\parallel$ [rad m$^{-1}$]",
                title=rf"Parallel component, $\theta={theta_deg:g}^\circ$")
    ax_kperp.set(xscale="log", xlabel="frequency [Hz]",
                 ylabel=r"$k_\perp$ [rad m$^{-1}$]",
                 title=rf"Perpendicular component, $\theta={theta_deg:g}^\circ$")

    for ax in axes.flat:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    fig.suptitle(rf"Cold-plasma Stix branches: $q={q:.5g}$, "
                 rf"$B_0={B0:g}$ T, $n_e=10^{{{d:g}}}$ m$^{{-3}}$")
    return fig


def make_interactive_figure(d=18.0, B0=1.0, u=1.0 / 1836.15,
                            theta_deg=45.0, fixed_x=0.2,
                            x_min=1e-4, x_max=3.0, points=900):
    """Four-panel GUI with fast dragging and a detailed mouse-release redraw."""
    full_points = max(300, points)
    fast_points = min(250, full_points)
    x_full = np.geomspace(x_min, x_max, full_points)
    x_fast = np.geomspace(x_min, x_max, fast_points)
    angle = np.linspace(0.0, np.pi / 2.0, 241)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.25,
                        hspace=0.38, wspace=0.30)
    ax_disp, ax_locus, ax_kpar, ax_kperp = axes.flat
    colors = ("tab:blue", "tab:orange")
    labels = ("branch 1", "branch 2")

    dispersion_lines = [ax_disp.plot([], [], color=c, label=l)[0]
                        for c, l in zip(colors, labels)]
    locus_lines = [ax_locus.plot([], [], color=c, label=l)[0]
                   for c, l in zip(colors, labels)]
    kpar_lines = [ax_kpar.plot([], [], color=c, label=l)[0]
                  for c, l in zip(colors, labels)]
    kperp_lines = [ax_kperp.plot([], [], color=c, label=l)[0]
                   for c, l in zip(colors, labels)]

    ax_disp.set(xscale="log", xlabel=r"$x=\omega/\omega_{ce}$",
                ylabel=r"$kc/\omega_{ce}$", title="Normalized dispersion")
    ax_locus.set(xlabel=r"$k_\parallel$ [rad m$^{-1}$]",
                 ylabel=r"$k_\perp$ [rad m$^{-1}$]")
    ax_kpar.set(xscale="log", xlabel="frequency [Hz]",
                ylabel=r"$k_\parallel$ [rad m$^{-1}$]")
    ax_kperp.set(xscale="log", xlabel="frequency [Hz]",
                 ylabel=r"$k_\perp$ [rad m$^{-1}$]")
    for ax in axes.flat:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best")

    slider_data = (
        ("d", 14.0, 21.0, d, 0.185),
        (r"$\log_{10} B_0$ [T]", -3.0, 1.0, np.log10(B0), 0.145),
        (r"$\theta$ [deg]", 0.0, 90.0, theta_deg, 0.105),
        (r"fixed $x$", 0.01, 3.0, fixed_x, 0.065),
    )
    sliders = []
    for label, lo, hi, initial, ypos in slider_data:
        slider_ax = fig.add_axes([0.16, ypos, 0.67, 0.022])
        sliders.append(Slider(slider_ax, label, lo, hi, valinit=initial))
    sd, slogb, stheta, sx = sliders

    def redraw(detailed=False):
        # A small frequency grid follows every cursor movement.  Releasing the
        # mouse swaps in the detailed grid without changing the selected state.
        x = x_full if detailed else x_fast
        current_d = sd.val
        current_B0 = 10.0 ** slogb.val
        current_theta_deg = stheta.val
        current_theta = np.deg2rad(current_theta_deg)
        current_fixed_x = sx.val
        q = 1.028_822_055 * 10.0 ** (current_d - 19.0) / current_B0**2
        omega_ce = E * current_B0 / M_E
        f_ce = omega_ce / (2.0 * np.pi)

        g1, g2 = stix_branches(x, current_theta, q, u)
        branches = (g1, g2)
        kvals = tuple(propagating_k(x, g, omega_ce) for g in branches)
        for line, g in zip(dispersion_lines, branches):
            K = x * np.sqrt(np.where(g >= 0.0, g, np.nan))
            line.set_data(x, masked_for_plot(K))

        ga, gb = stix_branches(current_fixed_x, angle, q, u)
        for line, g in zip(locus_lines, (ga, gb)):
            kval = propagating_k(current_fixed_x, g, omega_ce)
            line.set_data(kval * np.cos(angle), kval * np.sin(angle))

        freq_hz = x * f_ce
        for line, kval in zip(kpar_lines, kvals):
            line.set_data(freq_hz,
                          masked_for_plot(kval * np.cos(current_theta)))
        for line, kval in zip(kperp_lines, kvals):
            line.set_data(freq_hz,
                          masked_for_plot(kval * np.sin(current_theta)))

        ax_locus.set_title(rf"Fixed $x={current_fixed_x:.3g}$ "
                           rf"($f={current_fixed_x*f_ce:.3g}$ Hz)")
        ax_kpar.set_title(rf"Parallel component, "
                          rf"$\theta={current_theta_deg:.1f}^\circ$")
        ax_kperp.set_title(rf"Perpendicular component, "
                           rf"$\theta={current_theta_deg:.1f}^\circ$")
        fig.suptitle(rf"Stix branches: $q={q:.4g}$, $B_0={current_B0:.3g}$ T, "
                     rf"$n_e=10^{{{current_d:.2f}}}$ m$^{{-3}}$  "
                     f"({'detailed' if detailed else 'fast'} redraw)")
        for ax in axes.flat:
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw_idle()

    def update_fast(_value=None):
        redraw(detailed=False)

    def update_full(event):
        if event.button == 1:
            redraw(detailed=True)

    for slider in sliders:
        slider.on_changed(update_fast)
    fig.canvas.mpl_connect("button_release_event", update_full)

    reset_ax = fig.add_axes([0.86, 0.058, 0.075, 0.035])
    reset_button = Button(reset_ax, "Reset")

    def reset(_event):
        for slider in sliders:
            slider.reset()
        redraw(detailed=True)

    reset_button.on_clicked(reset)
    fig._dispersion_widgets = (*sliders, reset_button)  # keep widgets alive
    redraw(detailed=True)
    return fig


def make_stix_explorer(d=18.0, B0=1.0, u=1.0 / 1836.15,
                       theta_deg=45.0, fixed_x=0.2,
                       x_min=1e-5, x_max=3.0, points=200):
    """Build the complete, continuously updating six-panel Stix GUI."""
    full_points = max(200, points)
    fast_points = 80
    x_full = np.geomspace(x_min, x_max, full_points)
    x_fast = np.geomspace(x_min, x_max, fast_points)
    x_diagnostic = np.geomspace(x_min, x_max, 1600)
    # Calculate one exact quadrant, then mirror it with a common radial mask.
    surface_angle_full = np.linspace(0.0, np.pi / 2.0, 91)
    surface_angle_fast = np.linspace(0.0, np.pi / 2.0, 31)
    cone_angle_full = np.linspace(0.0, 2.0 * np.pi, 361)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.875, bottom=0.285,
                        hspace=0.42, wspace=0.31)
    ax_n2, ax_disp, ax_surface, ax_kpar, ax_kperp, ax_presets = axes.flat
    colors = ("tab:blue", "tab:orange")
    labels = (r"root $n_1^2$", r"root $n_2^2$")

    def empty_lines(ax):
        return [ax.plot([], [], color=color, lw=1.6, label=label)[0]
                for color, label in zip(colors, labels)]

    n2_lines = empty_lines(ax_n2)
    dispersion_lines = empty_lines(ax_disp)
    surface_lines = empty_lines(ax_surface)
    kpar_lines = empty_lines(ax_kpar)
    kperp_lines = empty_lines(ax_kperp)
    plot_axes = (ax_n2, ax_disp, ax_surface, ax_kpar, ax_kperp)
    selected_normalized_lines = tuple(
        ax.axvline(fixed_x, color="purple", ls="-.", lw=1.8,
                   alpha=0.9, zorder=7)
        for ax in (ax_n2, ax_disp)
    )
    selected_physical_lines = tuple(
        ax.axvline(1.0, color="purple", ls="-.", lw=1.8,
                   alpha=0.9, zorder=7)
        for ax in (ax_kpar, ax_kperp)
    )

    ax_n2.set(xscale="log", yscale="symlog", xlabel=r"$x=\omega/\omega_{ce}$",
              ylabel=r"index of refraction $n^2$",
              title=r"Index of refraction $n^2(\omega)$")
    ax_n2.axhline(0.0, color="0.35", lw=0.7)
    ax_disp.set(xscale="log", xlabel=r"normalized frequency $\omega/\omega_{ce}$",
                ylabel=r"normalized wave number $kc/\omega_{ce}$",
                title=r"Dispersion: $k(\omega)$")
    ax_surface.set(xlabel=r"parallel wave number $k_\parallel$ [rad m$^{-1}$]",
                   ylabel=r"perpendicular wave number $k_\perp$ [rad m$^{-1}$]",
                   title="Fixed-frequency wave-vector surface")
    ax_surface.axhline(0, color="0.5", lw=0.6)
    ax_surface.axvline(0, color="0.5", lw=0.6)
    ax_surface.set_aspect("equal", adjustable="datalim")
    ax_kpar.set(xscale="log", xlabel="frequency $f$ [Hz]",
                ylabel=r"parallel wave number $k_\parallel$ [rad m$^{-1}$]",
                title=r"$k_\parallel(f)$ at fixed angle")
    ax_kperp.set(xscale="log", xlabel="frequency $f$ [Hz]",
                 ylabel=r"perpendicular wave number $k_\perp$ [rad m$^{-1}$]",
                 title=r"$k_\perp(f)$ at fixed angle")
    ax_presets.set_axis_off()
    for ax in plot_axes:
        ax.grid(True, which="both", alpha=0.23)
        ax.legend(loc="best", fontsize=8)

    slider_data = (
        (r"density $d=\log_{10}(n_e/\mathrm{m}^{-3})$", 14.0, 21.0, d, 0.205),
        (r"field $\log_{10}(B_0/\mathrm{T})$", -3.0, 1.0,
         np.log10(B0), 0.165),
        (r"mass ratio $\log_{10}(m_e/m_i)$", -4.5, -1.0,
         np.log10(u), 0.125),
        (r"angle $\theta=\angle(\mathbf{k},\mathbf{B}_0)$ [deg]",
         0.0, 90.0, theta_deg, 0.085),
        (r"surface frequency $\log_{10}(\omega/\omega_{ce})$",
         -5.0, np.log10(3.0), np.log10(fixed_x), 0.045),
    )
    sliders = []
    for label, low, high, initial, ypos in slider_data:
        slider_ax = fig.add_axes([0.285, ypos, 0.42, 0.021])
        sliders.append(Slider(slider_ax, label, low, high, valinit=initial))
    sd, slogb, su, stheta, sx = sliders
    # The slider is logarithmic for usability, but its displayed value is the
    # physically familiar reciprocal mass ratio, e.g. 1/1836.
    su.valtext.set_text(f"1/{int(round(1.0 / u))}")
    sx.valtext.set_text(f"x={fixed_x:.3g}")
    status = fig.text(
        0.5, 0.91, "", fontsize=8.8, ha="center", va="center",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9,
              "boxstyle": "round,pad=0.25"},
    )
    annotation_artists = []

    def clear_annotations():
        while annotation_artists:
            annotation_artists.pop().remove()

    def annotate_special_frequencies(x, q, theta, mass_ratio, f_ce):
        """Mark P/R/L cutoffs and A=0 resonances on frequency plots."""
        clear_annotations()
        p, r, ell, _s, a = stix_coefficients(x, theta, q, mass_ratio)
        cutoffs = unique_log_roots(
            zero_crossings(x, p) + zero_crossings(x, r) + zero_crossings(x, ell)
        )
        resonances = unique_log_roots(zero_crossings(x, a))
        # Cyclotron poles are resonant singular frequencies too.
        resonances = unique_log_roots(resonances + [mass_ratio, 1.0])
        normalized_axes = (ax_n2, ax_disp)
        physical_axes = (ax_kpar, ax_kperp)
        for root in cutoffs:
            for ax in normalized_axes:
                annotation_artists.append(ax.axvline(
                    root, color="limegreen", ls=":", lw=2.4, alpha=0.95,
                    zorder=8))
            for ax in physical_axes:
                annotation_artists.append(ax.axvline(
                    root * f_ce, color="limegreen", ls=":", lw=2.4,
                    alpha=0.95, zorder=8))
        for root in resonances:
            if x_min <= root <= x_max:
                for ax in normalized_axes:
                    annotation_artists.append(ax.axvline(
                        root, color="crimson", ls="--", lw=2.2, alpha=0.95,
                        zorder=8))
                for ax in physical_axes:
                    annotation_artists.append(ax.axvline(
                        root * f_ce, color="crimson", ls="--", lw=2.2,
                        alpha=0.95, zorder=8))
        # A compact key avoids repeating dozens of labels on individual lines.
        status.set_text(
            f"green dotted = cutoffs (P, R, or L = 0; n² = 0)    "
            f"red dashed = resonances (A = 0 or cyclotron pole; n² → ∞)    "
            f"purple dash-dot = selected surface frequency    "
            f"cutoffs found: {len(cutoffs)}, resonances shown: {len(resonances)}"
        )

    def redraw(detailed=False):
        x = x_full if detailed else x_fast
        surface_angle = surface_angle_full if detailed else surface_angle_fast
        current_d, current_u = sd.val, 10.0 ** su.val
        su.valtext.set_text(f"1/{int(round(1.0 / current_u))}")
        current_B0 = 10.0 ** slogb.val
        current_theta_deg = stheta.val
        current_fixed_x = 10.0 ** sx.val
        sx.valtext.set_text(f"x={current_fixed_x:.3g}")
        theta = np.deg2rad(current_theta_deg)
        # Make the two exact limiting angles exact numerically as well.  This
        # prevents cos(pi/2) roundoff from appearing as a tiny physical k_parallel.
        cos_theta = 0.0 if np.isclose(current_theta_deg, 90.0) else np.cos(theta)
        sin_theta = 0.0 if np.isclose(current_theta_deg, 0.0) else np.sin(theta)
        q = 1.028_822_055 * 10.0 ** (current_d - 19.0) / current_B0**2
        omega_ce = E * current_B0 / M_E
        f_ce = omega_ce / (2.0 * np.pi)
        g1, g2 = stix_branches(x, theta, q, current_u)
        branches = (g1, g2)
        kvals = tuple(propagating_k(x, g, omega_ce) for g in branches)
        frequency = x * f_ce

        for line, g in zip(n2_lines, branches):
            line.set_data(x, masked_for_plot(g, 98.5))
        for line, g in zip(dispersion_lines, branches):
            normalized_k = x * np.sqrt(np.where(g >= 0.0, g, np.nan))
            line.set_data(x, masked_for_plot(normalized_k, 98.5))
        for line, kval in zip(kpar_lines, kvals):
            line.set_data(frequency, masked_for_plot(kval * cos_theta, 98.5))
        for line, kval in zip(kperp_lines, kvals):
            line.set_data(frequency, masked_for_plot(kval * sin_theta, 98.5))

        sg1, sg2 = stix_branches(current_fixed_x, surface_angle, q, current_u)
        for line, g in zip(surface_lines, (sg1, sg2)):
            kval = propagating_k(current_fixed_x, g, omega_ce)
            surface_x, surface_y = symmetric_wave_surface(surface_angle, kval)
            line.set_data(surface_x, surface_y)

        fixed_hz = current_fixed_x * f_ce
        for line in selected_normalized_lines:
            line.set_xdata([current_fixed_x, current_fixed_x])
        for line in selected_physical_lines:
            line.set_xdata([fixed_hz, fixed_hz])
        ax_surface.set_title(rf"Full wave-vector surface at $x={current_fixed_x:.3g}$ "
                             rf"($f={fixed_hz:.3g}$ Hz)")
        ax_kpar.set_title(rf"$k_\parallel(f)$ at $\theta={current_theta_deg:.1f}^\circ$")
        ax_kperp.set_title(rf"$k_\perp(f)$ at $\theta={current_theta_deg:.1f}^\circ$")
        fig.suptitle(rf"Cold magnetized plasma — Stix framework    "
                     rf"$B_0={current_B0:.3g}$ T, $n_e=10^{{{current_d:.2f}}}$ m$^{{-3}}$, "
                     rf"$m_e/m_i={current_u:.4g}$    "
                     f"({'200-point redraw' if detailed else '80-point live preview'})",
                     fontsize=13)
        if detailed:
            annotate_special_frequencies(
                x_diagnostic, q, theta, current_u, f_ce
            )
            # At fixed omega, A(theta)=0 defines resonance-cone asymptotes.
            # Draw their directions on the full k-space surface so clipped
            # infinite-k branches are not mistaken for missing data.
            _p, _r, _l, _s, surface_a = stix_coefficients(
                current_fixed_x, cone_angle_full, q, current_u
            )
            cone_angles = zero_crossings(cone_angle_full, surface_a, max_abs=1e6)
            surface_values = []
            for line in surface_lines:
                surface_values.extend(np.abs(line.get_xdata()[np.isfinite(line.get_xdata())]))
                surface_values.extend(np.abs(line.get_ydata()[np.isfinite(line.get_ydata())]))
            cone_length = np.percentile(surface_values, 95) if surface_values else 1.0
            for cone_angle in cone_angles:
                annotation_artists.append(ax_surface.plot(
                    [0.0, cone_length * np.cos(cone_angle)],
                    [0.0, cone_length * np.sin(cone_angle)],
                    color="crimson", ls="--", lw=2.5, alpha=0.95,
                    zorder=8
                )[0])
            if cone_angles:
                status.set_text(status.get_text() +
                                "    red rays on k-surface = resonance cones")
        for ax in plot_axes:
            ax.relim()
            ax.autoscale_view()
        # Preserve equal scaling after autoscale so the polar surface is not distorted.
        ax_surface.set_aspect("equal", adjustable="datalim")
        fig.canvas.draw_idle()

    preset_change = [False]

    def slider_changed(_value):
        if not preset_change[0]:
            redraw(False)

    for slider in sliders:
        slider.on_changed(slider_changed)
    fig.canvas.mpl_connect(
        "button_release_event", lambda event: redraw(True) if event.button == 1 else None
    )

    ax_presets.text(0.5, 1.02, "Representative wave regimes", ha="center",
                    va="top", fontsize=11, fontweight="bold",
                    transform=ax_presets.transAxes)
    ax_presets.text(0.5, 0.92,
                    "Presets choose a useful parameter region; both roots remain visible.",
                    ha="center", va="top", fontsize=8, wrap=True,
                    transform=ax_presets.transAxes)
    preset_note = ax_presets.text(0.5, 0.01, "Select a preset", ha="center",
                                  va="bottom", fontsize=8.5, wrap=True,
                                  transform=ax_presets.transAxes)

    # (density exponent, B0 [T], me/mi, theta [deg], omega/omega_ce, note)
    presets = {
        "O mode": (18.0, 1.0, 1 / 1836.15, 90.0, 0.60,
                   r"Perpendicular ordinary-mode region: $n^2=P$."),
        "X mode": (18.0, 1.0, 1 / 1836.15, 90.0, 1.15,
                   r"Propagating X mode above its high-frequency cutoff: $n^2=RL/S$."),
        "R / L": (18.0, 1.0, 1 / 1836.15, 0.0, 0.30,
                  r"Parallel circularly polarized roots: $n^2=R,L$."),
        "Whistler": (18.0, 1.0, 1 / 1836.15, 0.0, 0.08,
                     r"Parallel right-hand branch with $\omega_{ci}\ll\omega<\omega_{ce}$."),
        "Alfven": (18.0, 1.0, 1 / 1836.15, 10.0, 5.0e-5,
                   r"Deep low-frequency, nearly field-aligned shear-Alfvén region."),
        "Ion cycl.": (18.0, 1.0, 1 / 1836.15, 0.0, 4.0e-4,
                      r"Parallel left-hand branch just below $\omega_{ci}$."),
        "Lower hybrid": (18.0, 1.0, 1 / 1836.15, 90.0, 7.0e-3,
                         r"Nearly perpendicular lower-hybrid frequency region."),
        "Magnetosonic": (18.0, 1.0, 1 / 1836.15, 60.0, 1.0e-4,
                         r"Low-frequency oblique fast/compressional branch region."),
    }
    preset_buttons = []
    # Anchor the grid to the actual subplot bounds rather than the full figure.
    # It therefore remains centered when DPI, backend, or window size changes.
    preset_box = ax_presets.get_position()
    button_width = 0.42 * preset_box.width
    button_height = 0.13 * preset_box.height
    column_fractions = (0.04, 0.54)
    row_fractions = (0.68, 0.50, 0.32, 0.14)
    button_positions = tuple(
        (
            preset_box.x0 + column * preset_box.width,
            preset_box.y0 + row * preset_box.height,
        )
        for row in row_fractions
        for column in column_fractions
    )

    for (name, values), (button_x, button_y) in zip(presets.items(), button_positions):
        button_ax = fig.add_axes(
            [button_x, button_y, button_width, button_height]
        )
        button = Button(button_ax, name)
        # Replace Button's internal label with axes-relative text.  This keeps
        # every caption geometrically centered regardless of backend, DPI, or
        # the caption's length.
        button.label.set_visible(False)
        button_ax.text(
            0.5, 0.5, name,
            transform=button_ax.transAxes,
            ha="center", va="center",
            multialignment="center",
            fontsize=8.5,
            clip_on=True,
        )

        def choose_preset(_event, preset_values=values, preset_name=name):
            pd, pB0, pu, ptheta, px, note = preset_values
            preset_change[0] = True
            try:
                sd.set_val(pd)
                slogb.set_val(np.log10(pB0))
                su.set_val(np.log10(pu))
                stheta.set_val(ptheta)
                sx.set_val(np.log10(px))
            finally:
                preset_change[0] = False
            preset_note.set_text(f"{preset_name}: {note}")
            redraw(True)

        button.on_clicked(choose_preset)
        preset_buttons.append(button)

    reset_ax = fig.add_axes([0.905, 0.038, 0.07, 0.035])
    reset_button = Button(reset_ax, "Reset all")

    def reset(_event):
        preset_change[0] = True
        try:
            for slider in sliders:
                slider.reset()
        finally:
            preset_change[0] = False
        preset_note.set_text("Select a preset")
        redraw(True)

    reset_button.on_clicked(reset)
    fig._dispersion_widgets = (*sliders, *preset_buttons, reset_button)
    redraw(True)
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density-exp", type=float, default=18.0,
                        help="d in n_e=10^d m^-3 (default: 18)")
    parser.add_argument("--B0", type=float, default=1.0,
                        help="magnetic field h=B0 in tesla (default: 1)")
    parser.add_argument("--mass-ratio", type=float, default=1.0 / 1836.15,
                        help="u=m_e/m_i")
    parser.add_argument("--theta", type=float, default=45.0,
                        help="fixed angle in degrees for frequency scans")
    parser.add_argument("--fixed-x", type=float, default=0.2,
                        help="fixed omega/omega_ce for the k-space locus")
    parser.add_argument("--x-min", type=float, default=1e-5)
    parser.add_argument("--x-max", type=float, default=3.0)
    parser.add_argument("--points", type=int, default=200,
                        help="release-redraw samples (live dragging always uses 80)")
    parser.add_argument("--save", help="save figure to this file instead of showing it")
    return parser.parse_args()


def main():
    args = parse_args()
    fig = make_stix_explorer(
        args.density_exp, args.B0, args.mass_ratio, args.theta,
        args.fixed_x, args.x_min, args.x_max, args.points
    )
    if args.save:
        fig.savefig(args.save, dpi=180)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
