"""Interactive Stix dispersion and polarization explorer.

This is a polarization-focused companion to ``cold_plasma_dispersion.py``.
The original file is not modified.  Run with:

    python cold_plasma_polarization.py
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from cold_plasma_dispersion import (
    C, E, M_E, masked_for_plot, propagating_k, stix_branches,
    stix_coefficients, symmetric_wave_surface, unique_log_roots,
    zero_crossings,
)


COLORS = ("tab:blue", "tab:orange")
ROOT_LABELS = (r"root $n_1^2$", r"root $n_2^2$")


def polarization_vector(x, theta, q, u, n2):
    """Return the normalized complex electric-field eigenvector (Ex,Ey,Ez).

    Coordinates use B0 along +z and k in the x-z plane.  The vector is the
    right singular vector associated with the smallest singular value of the
    cold-plasma wave matrix.
    """
    if not np.isfinite(n2):
        return np.full(3, np.nan + 1j * np.nan)
    p, r, ell, s, _a = stix_coefficients(x, theta, q, u)
    dcoef = 0.5 * (r - ell)
    st, ct = np.sin(theta), np.cos(theta)
    matrix = np.array([
        [s - n2 * ct**2, -1j * dcoef, n2 * st * ct],
        [1j * dcoef, s - n2, 0.0],
        [n2 * st * ct, 0.0, p - n2 * st**2],
    ], dtype=complex)
    _left, _singular, vh = np.linalg.svd(matrix)
    field = vh[-1].conj()
    field /= np.linalg.norm(field)
    # Choose a stable global phase so component comparisons do not jump.
    pivot = int(np.argmax(np.abs(field)))
    field *= np.exp(-1j * np.angle(field[pivot]))
    return field


def polarization_diagnostics(field, theta):
    """Return transverse ellipse components and normalized field amplitudes."""
    khat = np.array([np.sin(theta), 0.0, np.cos(theta)])
    e1 = np.array([np.cos(theta), 0.0, -np.sin(theta)])
    e2 = np.array([0.0, 1.0, 0.0])
    if not np.all(np.isfinite(field)):
        return np.nan, np.nan, np.full(4, np.nan), "unavailable"
    transverse_1 = np.dot(e1, field)
    transverse_2 = np.dot(e2, field)
    longitudinal = np.dot(khat, field)
    amplitudes = np.array([
        abs(field[0]), abs(field[1]), abs(field[2]), abs(longitudinal)
    ])
    transverse_power = abs(transverse_1)**2 + abs(transverse_2)**2
    if transverse_power < 1e-14:
        shape = "electrostatic/longitudinal"
    else:
        major = max(abs(transverse_1), abs(transverse_2))
        minor = min(abs(transverse_1), abs(transverse_2))
        ratio = minor / major if major else 0.0
        phase = np.angle(transverse_2) - np.angle(transverse_1)
        circularity = abs(np.sin(phase)) * ratio
        if circularity > 0.9:
            shape = "nearly circular"
        elif circularity < 0.1:
            shape = "nearly linear"
        else:
            shape = "elliptical"
    return transverse_1, transverse_2, amplitudes, shape


def build_explorer(d=18.0, B0=1.0, u=1 / 1836.15, theta_deg=45.0,
                   fixed_x=0.2, x_min=1e-5, x_max=3.0, points=200):
    full_x = np.geomspace(x_min, x_max, max(200, points))
    fast_x = np.geomspace(x_min, x_max, 80)
    diagnostic_x = np.geomspace(x_min, x_max, 1600)
    surface_angle_full = np.linspace(0.0, np.pi / 2.0, 91)
    surface_angle_fast = np.linspace(0.0, np.pi / 2.0, 31)
    cone_angle_full = np.linspace(0.0, 2*np.pi, 361)

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.865, bottom=0.32,
                        hspace=0.43, wspace=0.34)
    (ax_n2, ax_disp, ax_surface, ax_pol,
     ax_kpar, ax_kperp, ax_components, ax_presets) = axes.flat
    plot_axes = (ax_n2, ax_disp, ax_surface, ax_kpar, ax_kperp)

    def root_lines(ax):
        return [ax.plot([], [], color=c, lw=1.6, label=label)[0]
                for c, label in zip(COLORS, ROOT_LABELS)]

    n2_lines = root_lines(ax_n2)
    dispersion_lines = root_lines(ax_disp)
    surface_lines = root_lines(ax_surface)
    kpar_lines = root_lines(ax_kpar)
    kperp_lines = root_lines(ax_kperp)

    ax_n2.set(xscale="log", yscale="symlog",
              xlabel=r"$x=\omega/\omega_{ce}$", ylabel=r"$n^2$",
              title=r"Index of refraction $n^2(\omega)$")
    ax_n2.axhline(0, color="0.4", lw=0.7)
    ax_disp.set(xscale="log", xlabel=r"$\omega/\omega_{ce}$",
                ylabel=r"$kc/\omega_{ce}$", title=r"Dispersion $k(\omega)$")
    ax_surface.set(xlabel=r"$k_\parallel$ [rad m$^{-1}$]",
                   ylabel=r"$k_\perp$ [rad m$^{-1}$]",
                   title="Fixed-frequency wave-vector surface")
    ax_surface.axhline(0, color="0.5", lw=0.6)
    ax_surface.axvline(0, color="0.5", lw=0.6)
    ax_surface.set_aspect("equal", adjustable="datalim")
    ax_kpar.set(xscale="log", xlabel="frequency [Hz]",
                ylabel=r"$k_\parallel$ [rad m$^{-1}$]")
    ax_kperp.set(xscale="log", xlabel="frequency [Hz]",
                 ylabel=r"$k_\perp$ [rad m$^{-1}$]")
    for ax in plot_axes:
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(fontsize=8)

    ax_pol.set_title("Transverse polarization ellipses")
    ax_pol.set_xlabel(r"$E_1$ in the $(\mathbf{k},\mathbf{B}_0)$ plane")
    ax_pol.set_ylabel(r"$E_2$ perpendicular to that plane")
    ax_pol.set_aspect("equal", adjustable="box")
    ax_pol.set_xlim(-2.5, 2.5)
    ax_pol.set_ylim(-1.25, 1.25)
    ax_pol.axhline(0, color="0.75", lw=0.7)
    ax_pol.grid(True, alpha=0.2)
    ellipse_lines = [ax_pol.plot([], [], color=c, lw=2.0)[0] for c in COLORS]
    ellipse_centers = (-1.3, 1.3)
    # Put status text in the inter-row margin, not over the ellipse grid.
    pol_box = ax_pol.get_position()
    ellipse_text = [fig.text(
        pol_box.x0 + fraction * pol_box.width,
        pol_box.y0 - 0.072,
        "", ha="center", va="center", color=color, fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": color,
              "alpha": .9, "pad": 1.5},
    ) for fraction, color in zip((0.25, 0.75), COLORS)]
    ellipse_arrows = []

    component_names = (r"$|E_x|$", r"$|E_y|$", r"$|E_z|$", r"$|E_k|$")
    component_x = np.arange(4)
    bar_width = 0.34
    bars = (
        ax_components.bar(component_x - bar_width / 2, np.zeros(4), bar_width,
                          color=COLORS[0], label=ROOT_LABELS[0]),
        ax_components.bar(component_x + bar_width / 2, np.zeros(4), bar_width,
                          color=COLORS[1], label=ROOT_LABELS[1]),
    )
    ax_components.set_xticks(component_x, component_names)
    ax_components.set_ylim(0, 1.05)
    ax_components.set_ylabel("normalized electric-field amplitude")
    ax_components.set_title("Polarization-vector components")
    ax_components.grid(True, axis="y", alpha=0.22)
    ax_components.legend(fontsize=8)

    slider_data = (
        (r"density $d=\log_{10}(n_e/\mathrm{m}^{-3})$", 14., 21., d, 0.185),
        (r"field $\log_{10}(B_0/\mathrm{T})$", -3., 1., np.log10(B0), 0.148),
        (r"mass ratio $\log_{10}(m_e/m_i)$", -4.5, -1., np.log10(u), 0.111),
        (r"angle $\theta=\angle(\mathbf{k},\mathbf{B}_0)$ [deg]",
         0., 90., theta_deg, 0.074),
        (r"polarization frequency $\log_{10}(\omega/\omega_{ce})$",
         -5., np.log10(3.), np.log10(fixed_x), 0.037),
    )
    sliders = []
    for label, low, high, initial, ypos in slider_data:
        slider_ax = fig.add_axes([0.27, ypos, 0.39, 0.02])
        sliders.append(Slider(slider_ax, label, low, high, valinit=initial))
    sd, slogb, su, stheta, sx = sliders
    su.valtext.set_text(f"1/{int(round(1/u))}")
    sx.valtext.set_text(f"x={fixed_x:.3g}")

    status = fig.text(0.5, 0.915, "", ha="center", va="center", fontsize=8.5,
                      bbox={"facecolor": "white", "edgecolor": "0.8",
                            "alpha": 0.9, "boxstyle": "round,pad=0.25"})
    annotations = []

    def clear_annotations():
        while annotations:
            annotations.pop().remove()

    def annotate_frequencies(q, theta, mass_ratio, f_ce):
        clear_annotations()
        p, r, ell, _s, a = stix_coefficients(diagnostic_x, theta, q, mass_ratio)
        cutoffs = unique_log_roots(zero_crossings(diagnostic_x, p) +
                                   zero_crossings(diagnostic_x, r) +
                                   zero_crossings(diagnostic_x, ell))
        resonances = unique_log_roots(zero_crossings(diagnostic_x, a) +
                                      [mass_ratio, 1.0])
        for root in cutoffs:
            for ax in (ax_n2, ax_disp):
                annotations.append(ax.axvline(root, color="limegreen", ls=":",
                                              lw=2.2, alpha=.95, zorder=8))
            for ax in (ax_kpar, ax_kperp):
                annotations.append(ax.axvline(root*f_ce, color="limegreen", ls=":",
                                              lw=2.2, alpha=.95, zorder=8))
        for root in resonances:
            if x_min <= root <= x_max:
                for ax in (ax_n2, ax_disp):
                    annotations.append(ax.axvline(root, color="crimson", ls="--",
                                                  lw=2.0, alpha=.95, zorder=8))
                for ax in (ax_kpar, ax_kperp):
                    annotations.append(ax.axvline(root*f_ce, color="crimson", ls="--",
                                                  lw=2.0, alpha=.95, zorder=8))
        status.set_text("green dotted = cutoffs    red dashed = resonances    "
                        "ellipses/components are evaluated at the slider frequency and angle")

    def update_polarization(x_selected, theta, q, mass_ratio):
        nonlocal ellipse_arrows
        for arrow in ellipse_arrows:
            arrow.remove()
        ellipse_arrows = []
        selected_roots = stix_branches(x_selected, theta, q, mass_ratio)
        phase = np.linspace(0, 2*np.pi, 241)
        for index, (n2, center, line, text_artist, bar_group) in enumerate(
                zip(selected_roots, ellipse_centers, ellipse_lines, ellipse_text, bars)):
            field = polarization_vector(x_selected, theta, q, mass_ratio, float(n2))
            e1, e2, amplitudes, _shape = polarization_diagnostics(field, theta)
            if np.isfinite(e1) and np.isfinite(e2):
                curve_x = np.real(e1 * np.exp(-1j*phase))
                curve_y = np.real(e2 * np.exp(-1j*phase))
                scale = max(np.max(np.hypot(curve_x, curve_y)), 1e-15)
                curve_x, curve_y = curve_x/scale + center, curve_y/scale
                line.set_data(curve_x, curve_y)
                j = 22
                arrow = ax_pol.annotate("", xy=(curve_x[j+2], curve_y[j+2]),
                                        xytext=(curve_x[j], curve_y[j]),
                                        arrowprops={"arrowstyle": "->", "color": COLORS[index],
                                                    "lw": 1.8})
                ellipse_arrows.append(arrow)
                longitudinal_fraction = amplitudes[3]**2
                propagation = "propagating" if n2 >= 0 else "evanescent"
                short_state = "prop." if propagation == "propagating" else "evan."
                text_artist.set_text(
                    rf"$n_{index+1}^2$: {short_state}"
                    + "\n"
                    + f"long. power={longitudinal_fraction:.2f}"
                )
            else:
                line.set_data([], [])
                text_artist.set_text(rf"$n_{index+1}^2$: unavailable")
            for rectangle, height in zip(bar_group, amplitudes):
                rectangle.set_height(height if np.isfinite(height) else 0.0)

    preset_change = [False]

    def redraw(detailed=False):
        x = full_x if detailed else fast_x
        surface_angle = surface_angle_full if detailed else surface_angle_fast
        current_d = sd.val
        current_B0 = 10**slogb.val
        current_u = 10**su.val
        current_theta_deg = stheta.val
        theta = np.deg2rad(current_theta_deg)
        selected_x = 10**sx.val
        su.valtext.set_text(f"1/{int(round(1/current_u))}")
        sx.valtext.set_text(f"x={selected_x:.3g}")
        q = 1.028_822_055 * 10**(current_d-19) / current_B0**2
        omega_ce = E * current_B0 / M_E
        f_ce = omega_ce / (2*np.pi)
        roots = stix_branches(x, theta, q, current_u)
        kvals = tuple(propagating_k(x, root, omega_ce) for root in roots)
        frequency = x*f_ce
        ct = 0.0 if np.isclose(current_theta_deg, 90) else np.cos(theta)
        st = 0.0 if np.isclose(current_theta_deg, 0) else np.sin(theta)

        for line, root in zip(n2_lines, roots):
            line.set_data(x, masked_for_plot(root, 98.5))
        for line, root in zip(dispersion_lines, roots):
            line.set_data(x, masked_for_plot(x*np.sqrt(np.where(root >= 0, root, np.nan)), 98.5))
        for line, kval in zip(kpar_lines, kvals):
            line.set_data(frequency, masked_for_plot(kval*ct, 98.5))
        for line, kval in zip(kperp_lines, kvals):
            line.set_data(frequency, masked_for_plot(kval*st, 98.5))

        surface_roots = stix_branches(selected_x, surface_angle, q, current_u)
        for line, root in zip(surface_lines, surface_roots):
            kval = propagating_k(selected_x, root, omega_ce)
            surface_x, surface_y = symmetric_wave_surface(surface_angle, kval)
            line.set_data(surface_x, surface_y)

        update_polarization(selected_x, theta, q, current_u)
        ax_surface.set_title(rf"Wave-vector surface at $x={selected_x:.3g}$")
        ax_kpar.set_title(rf"$k_\parallel(f)$ at $\theta={current_theta_deg:.1f}^\circ$")
        ax_kperp.set_title(rf"$k_\perp(f)$ at $\theta={current_theta_deg:.1f}^\circ$")
        ax_pol.set_title(rf"Polarization at $x={selected_x:.3g}$, "
                         rf"$\theta={current_theta_deg:.1f}^\circ$")
        fig.suptitle(rf"Cold-plasma Stix dispersion and polarization    "
                     rf"$B_0={current_B0:.3g}$ T, $n_e=10^{{{current_d:.2f}}}$ m$^{{-3}}$, "
                     rf"$m_e/m_i={current_u:.4g}$", fontsize=13)
        if detailed:
            annotate_frequencies(q, theta, current_u, f_ce)
            # A(theta)=0 gives the fixed-frequency resonance-cone directions.
            _p, _r, _l, _s, cone_a = stix_coefficients(
                selected_x, cone_angle_full, q, current_u
            )
            cone_angles = zero_crossings(cone_angle_full, cone_a, max_abs=1e6)
            finite_surface = []
            for line in surface_lines:
                finite_surface.extend(np.abs(line.get_xdata()[np.isfinite(line.get_xdata())]))
                finite_surface.extend(np.abs(line.get_ydata()[np.isfinite(line.get_ydata())]))
            cone_length = np.percentile(finite_surface, 95) if finite_surface else 1.0
            for cone_angle in cone_angles:
                annotations.append(ax_surface.plot(
                    [0, cone_length*np.cos(cone_angle)],
                    [0, cone_length*np.sin(cone_angle)],
                    color="crimson", ls="--", lw=2.2, alpha=.95, zorder=8
                )[0])
            if cone_angles:
                status.set_text(status.get_text() +
                                "    red rays = resonance-cone directions")
        for ax in plot_axes:
            ax.relim()
            ax.autoscale_view()
        ax_surface.set_aspect("equal", adjustable="datalim")
        fig.canvas.draw_idle()

    # Presets occupy their own subplot and remain anchored when resized.
    ax_presets.set_axis_off()
    ax_presets.text(.5, .98, "Representative wave regimes", ha="center", va="top",
                    fontsize=10.5, fontweight="bold", transform=ax_presets.transAxes)
    preset_note = ax_presets.text(.5, .02, "Select a preset", ha="center", va="bottom",
                                  fontsize=8, wrap=True, transform=ax_presets.transAxes)
    presets = {
        "O mode": (18., 1., 1/1836.15, 90., .60, r"$n_O^2=P$, linear $E\parallel B_0$."),
        "X mode": (18., 1., 1/1836.15, 90., 1.15, r"$n_X^2=RL/S$, above X cutoff."),
        "R / L": (18., 1., 1/1836.15, 0., .30, "Parallel circular eigenmodes."),
        "Whistler": (18., 1., 1/1836.15, 0., .08, "Right-hand low-frequency R branch."),
        "Alfven": (18., 1., 1/1836.15, 10., 5e-5, "Low-frequency shear-Alfvén region."),
        "Ion cycl.": (18., 1., 1/1836.15, 0., 4e-4, "Left-hand branch below ion cyclotron."),
        "Lower hybrid": (18., 1., 1/1836.15, 90., .007, "Nearly electrostatic perpendicular mode."),
        "Magnetosonic": (18., 1., 1/1836.15, 60., 1e-4, "Oblique fast/compressional branch."),
    }
    preset_buttons = []
    for index, (name, values) in enumerate(presets.items()):
        row, column = divmod(index, 2)
        button_ax = ax_presets.inset_axes([.04 + .50*column, .68 - .17*row, .42, .12])
        button = Button(button_ax, name)
        button.label.set_ha("center")
        button.label.set_va("center")
        button.label.set_fontsize(8)

        def select(_event, preset=values, preset_name=name):
            pd, pB, pu, pt, px, note = preset
            preset_change[0] = True
            try:
                sd.set_val(pd); slogb.set_val(np.log10(pB)); su.set_val(np.log10(pu))
                stheta.set_val(pt); sx.set_val(np.log10(px))
            finally:
                preset_change[0] = False
            preset_note.set_text(f"{preset_name}: {note}")
            redraw(True)

        button.on_clicked(select)
        preset_buttons.append(button)

    def slider_update(_value):
        if not preset_change[0]:
            redraw(False)

    for slider in sliders:
        slider.on_changed(slider_update)
    fig.canvas.mpl_connect("button_release_event",
                           lambda event: redraw(True) if event.button == 1 else None)
    reset_ax = fig.add_axes([.90, .03, .075, .035])
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
    fig._polarization_widgets = (*sliders, *preset_buttons, reset_button)
    redraw(True)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density-exp", type=float, default=18.)
    parser.add_argument("--B0", type=float, default=1.)
    parser.add_argument("--mass-ratio", type=float, default=1/1836.15)
    parser.add_argument("--theta", type=float, default=45.)
    parser.add_argument("--fixed-x", type=float, default=.2)
    parser.add_argument("--x-min", type=float, default=1e-5)
    parser.add_argument("--x-max", type=float, default=3.)
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--save")
    args = parser.parse_args()
    fig = build_explorer(args.density_exp, args.B0, args.mass_ratio, args.theta,
                         args.fixed_x, args.x_min, args.x_max, args.points)
    if args.save:
        fig.savefig(args.save, dpi=180)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
