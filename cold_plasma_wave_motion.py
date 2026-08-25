"""Animate cold-plasma plane waves and finite-bandwidth wave packets.

This is a standalone Matplotlib companion to the dispersion and polarization
explorers.  It reuses their Stix roots and electric-field eigenvectors, then
reconstructs

    E(s, t) = Re sum_j A_j e_j exp(i[k_j s - omega_j t]).

Here ``s`` is distance along k, B0 is uniform and stationary, and every
spectral component is an exact mode of the linear collisionless cold-plasma
model.  Run with::

    python cold_plasma_wave_motion.py
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons, Slider

from cold_plasma_dispersion import (C, E, M_E, propagating_k, stix_branches,
                                    stix_coefficients)
from cold_plasma_polarization import polarization_vector


COLORS = ("tab:blue", "tab:orange")


def wave_spectrum(d=18.0, B0=1.0, u=1 / 1836.15, theta_deg=0.0,
                  center_x=0.08, bandwidth=0.04, branch=0, samples=101):
    """Return a Gaussian spectrum of propagating cold-plasma eigenmodes.

    ``bandwidth`` is sigma_omega / omega0.  Invalid/evanescent samples are
    removed and the remaining quadrature weights are normalized.
    """
    if B0 <= 0 or center_x <= 0 or bandwidth <= 0:
        raise ValueError("B0, center_x, and bandwidth must be positive")
    if branch not in (0, 1):
        raise ValueError("branch must be 0 or 1")

    theta = np.deg2rad(theta_deg)
    q = 1.028_822_055 * 10.0 ** (d - 19.0) / B0**2
    omega_ce = E * B0 / M_E
    offsets = np.linspace(-4.0, 4.0, samples)
    x = center_x * (1.0 + bandwidth * offsets)
    keep_positive = x > 0.0
    x = x[keep_positive]
    offsets = offsets[keep_positive]
    n2 = np.asarray(stix_branches(x, theta, q, u)[branch], dtype=float)
    k = propagating_k(x, n2, omega_ce)
    good = np.isfinite(k) & (k > 0.0)
    if np.count_nonzero(good) < 5:
        raise ValueError("selected branch is evanescent or too close to a singularity")

    x, k, offsets = x[good], k[good], offsets[good]
    omega = x * omega_ce
    weights = np.exp(-0.5 * offsets**2)
    weights /= weights.sum()

    e1hat = np.array([np.cos(theta), 0.0, -np.sin(theta)])
    e2hat = np.array([0.0, 1.0, 0.0])
    fields = np.array([
        polarization_vector(float(xi), theta, q, u, float(ni))
        for xi, ni in zip(x, n2[good])
    ])
    # A singular vector has an arbitrary global complex phase. Align adjacent
    # samples so their packet superposition remains continuous.
    for index in range(1, len(fields)):
        overlap = np.vdot(fields[index - 1], fields[index])
        if abs(overlap) > 0.0:
            fields[index] *= np.exp(-1j * np.angle(overlap))
    e1 = fields @ e1hat
    e2 = fields @ e2hat

    center_index = int(np.argmin(np.abs(x - center_x)))
    phase_speed = omega[center_index] / k[center_index]
    if 0 < center_index < len(k) - 1:
        group_speed = ((omega[center_index + 1] - omega[center_index - 1]) /
                       (k[center_index + 1] - k[center_index - 1]))
    else:
        group_speed = np.gradient(omega, k)[center_index]

    return {
        "x": x, "omega": omega, "k": k, "weights": weights,
        "e1": e1, "e2": e2,
        "ex": fields[:, 0], "ey": fields[:, 1], "ez": fields[:, 2],
        "theta_deg": theta_deg, "center_index": center_index,
        "omega0": center_x * omega_ce, "k0": k[center_index],
        "phase_speed": phase_speed, "group_speed": group_speed,
    }


def synthesize_packet(spectrum, distance, time):
    """Reconstruct the two transverse electric-field components."""
    phase = np.exp(1j * (
        np.outer(spectrum["k"], np.asarray(distance)) -
        spectrum["omega"][:, None] * time
    ))
    weighted = spectrum["weights"][:, None] * phase
    return (np.real(np.sum(spectrum["e1"][:, None] * weighted, axis=0)),
            np.real(np.sum(spectrum["e2"][:, None] * weighted, axis=0)))


def build_wave_explorer(d=18.0, B0=1.0, u=1 / 1836.15,
                        theta_deg=0.0, center_x=0.08, bandwidth=0.04):
    """Build the desktop wave-motion explorer and return its figure."""
    fig, (ax_wave, ax_pol) = plt.subplots(1, 2, figsize=(13, 7))
    fig.subplots_adjust(left=.08, right=.97, top=.88, bottom=.34, wspace=.28)

    line_e1, = ax_wave.plot([], [], color=COLORS[0], lw=1.8, label=r"$E_1$")
    line_e2, = ax_wave.plot([], [], color=COLORS[1], lw=1.8, label=r"$E_2$")
    envelope_line, = ax_wave.plot([], [], color="0.25", ls="--", lw=1.0,
                                  label="packet envelope")
    center_line = ax_wave.axvline(0, color="tab:green", ls=":", lw=1.5,
                                  label=r"group center $v_g t$")
    phase_line = ax_wave.axvline(0, color="crimson", ls="-.", lw=1.4,
                                 label=r"phase crest $v_{ph}t$")
    ax_wave.set(xlabel=r"distance along $\mathbf{k}$ [m]",
                ylabel="normalized electric field",
                title="Exact spectral reconstruction")
    ax_wave.grid(True, alpha=.23)
    ax_wave.legend(loc="upper right", fontsize=8)

    ellipse_line, = ax_pol.plot([], [], color="purple", lw=2)
    ellipse_point, = ax_pol.plot([], [], "o", color="crimson", ms=7)
    ax_pol.axhline(0, color="0.7", lw=.7)
    ax_pol.axvline(0, color="0.7", lw=.7)
    ax_pol.annotate(r"$\mathbf{B}_0$", xy=(.88, .86), xytext=(.88, .52),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "lw": 2, "color": "black"},
                    ha="center")
    ax_pol.set(xlabel=r"$E_1$ (in the $\mathbf{k}$-$\mathbf{B}_0$ plane)",
               ylabel=r"$E_2$", title="Electric-field motion at packet center",
               xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect="equal")
    ax_pol.grid(True, alpha=.2)

    slider_specs = (
        (r"density $\log_{10}(n_e/\mathrm{m}^{-3})$", 14., 21., d, .255),
        (r"field $\log_{10}(B_0/\mathrm{T})$", -3., 1., np.log10(B0), .215),
        (r"mass ratio $\log_{10}(m_e/m_i)$", -4.5, -1., np.log10(u), .175),
        (r"angle $\theta$ [deg]", 0., 90., theta_deg, .135),
        (r"frequency $\log_{10}(\omega/\omega_{ce})$", -7., np.log10(3.),
         np.log10(center_x), .095),
        (r"fractional bandwidth $\sigma_\omega/\omega_0$", .005, .15,
         bandwidth, .055),
    )
    sliders = []
    for label, lo, hi, initial, y in slider_specs:
        sliders.append(Slider(fig.add_axes([.27, y, .31, .019]), label,
                              lo, hi, valinit=initial))
    sd, sb, su, sxangle, sfreq, sband = sliders
    su.valtext.set_text(f"1/{int(round(1 / (10**su.val)))}")

    branch_ax = fig.add_axes([.655, .105, .18, .13])
    branch_radio = RadioButtons(branch_ax, ("branch 1", "branch 2"), active=1)
    play_ax = fig.add_axes([.85, .19, .10, .042])
    reset_ax = fig.add_axes([.85, .135, .10, .042])
    play_button = Button(play_ax, "Pause")
    reset_button = Button(reset_ax, "Reset")

    preset_values = {
        "O mode": (18., 1., 1/1836.15, 90., .60, .025, 1),
        "X mode": (18., 1., 1/1836.15, 90., 1.15, .025, 1),
        "R / L": (18., 1., 1/1836.15, 0., .30, .035, 0),
        "Whistler": (18., 1., 1/1836.15, 0., .08, .05, 1),
        "Alfven": (18., 1., 1/1836.15, 10., 1e-6, .04, 1),
        "Ion cycl.": (18., 1., 1/1836.15, 0., 4e-4, .03, 1),
    }
    preset_buttons = []
    for index, (name, values) in enumerate(preset_values.items()):
        row, col = divmod(index, 3)
        button = Button(fig.add_axes([.65 + .105*col, .055 - .048*row, .095, .035]), name)
        button.label.set_fontsize(8)
        preset_buttons.append(button)

        def choose(_event, preset=values):
            pd, pB, pu, pa, px, pbw, pbranch = preset
            state["updating"] = True
            try:
                sd.set_val(pd); sb.set_val(np.log10(pB)); su.set_val(np.log10(pu))
                sxangle.set_val(pa); sfreq.set_val(np.log10(px)); sband.set_val(pbw)
                branch_radio.set_active(pbranch)
            finally:
                state["updating"] = False
            rebuild()
        button.on_clicked(choose)

    state = {"spectrum": None, "distance": None, "time": 0.0,
             "playing": True, "updating": False, "error": None}
    status = fig.text(.5, .925, "", ha="center", va="center", fontsize=9,
                      bbox={"facecolor": "white", "edgecolor": ".8", "alpha": .9})

    def rebuild(_value=None):
        if state["updating"]:
            return
        su.valtext.set_text(f"1/{int(round(1 / (10**su.val)))}")
        branch = (0 if str(branch_radio.value_selected).startswith("branch 1") else 1)
        if np.isclose(sxangle.val, 90.0):
            selected_x = 10**sfreq.val
            selected_u = 10**su.val
            selected_B0 = 10**sb.val
            selected_q = (1.028_822_055 * 10**(sd.val - 19.0) /
                          selected_B0**2)
            roots = np.asarray(stix_branches(
                selected_x, np.pi/2, selected_q, selected_u))
            pcoef = stix_coefficients(
                selected_x, np.pi/2, selected_q, selected_u)[0]
            o_index = int(np.nanargmin(np.abs(roots - pcoef)))
            mode_names = ["X mode", "X mode"]
            mode_names[o_index] = "O mode"
            for index, label in enumerate(branch_radio.labels):
                label.set_text(f"branch {index + 1} — {mode_names[index]}")
        else:
            for index, label in enumerate(branch_radio.labels):
                label.set_text(f"branch {index + 1}")
        try:
            spectrum = wave_spectrum(sd.val, 10**sb.val, 10**su.val,
                                     sxangle.val, 10**sfreq.val, sband.val, branch)
        except ValueError as exc:
            state["spectrum"] = None
            status.set_text(f"Unavailable: {exc}")
            line_e1.set_data([], []); line_e2.set_data([], [])
            fig.canvas.draw_idle()
            return
        wavelength = 2*np.pi/spectrum["k0"]
        state.update(spectrum=spectrum,
                     distance=np.linspace(-10*wavelength, 10*wavelength, 900),
                     time=0.0, error=None)
        ax_wave.set_xlim(-10*wavelength, 10*wavelength)
        ax_wave.set_ylim(-1.12, 1.12)
        phase = spectrum["phase_speed"]
        group = spectrum["group_speed"]
        ci = spectrum["center_index"]
        x_mode_view = (np.isclose(spectrum["theta_deg"], 90.0) and
                       abs(spectrum["ez"][ci]) < 0.5)
        state["polarization_components"] = (("ex", "ey") if x_mode_view
                                             else ("e1", "e2"))
        if x_mode_view:
            ax_pol.set_xlabel(r"$E_x$ (longitudinal, along $\mathbf{k}$)")
            line_e1.set_label(r"$E_x$"); line_e2.set_label(r"$E_y$")
            ax_pol.set_ylabel(r"$E_y$ (transverse)")
            ax_pol.set_title(r"X-mode electric-field motion in the $E_x$-$E_y$ plane")
        else:
            ax_pol.set_xlabel(r"$E_1$ (in the $\mathbf{k}$-$\mathbf{B}_0$ plane)")
            line_e1.set_label(r"$E_1$"); line_e2.set_label(r"$E_2$")
            ax_pol.set_ylabel(r"$E_2$")
            ax_pol.set_title("Transverse electric-field motion at packet center")
        ax_wave.legend(loc="upper right", fontsize=8)
        status.set_text(
            rf"$v_{{ph}}={phase:.3g}$ m/s    $v_g={group:.3g}$ m/s    "
            rf"$\lambda_0={wavelength:.3g}$ m    solid = exact packet; dotted = $v_g t$")
        draw_frame()

    def draw_frame():
        spectrum, distance = state["spectrum"], state["distance"]
        if spectrum is None:
            return line_e1, line_e2, envelope_line, center_line, ellipse_line, ellipse_point
        component_1, component_2 = state["polarization_components"]
        spectral_phase = np.exp(1j * (np.outer(spectrum["k"], distance) -
                                      spectrum["omega"][:, None] * state["time"]))
        analytic_1 = np.sum(spectrum["weights"][:, None] *
                            spectrum[component_1][:, None] * spectral_phase, axis=0)
        analytic_2 = np.sum(spectrum["weights"][:, None] *
                            spectrum[component_2][:, None] * spectral_phase, axis=0)
        magnitude = np.sqrt(np.abs(analytic_1)**2 + np.abs(analytic_2)**2)
        line_e1.set_data(distance, np.real(analytic_1))
        line_e2.set_data(distance, np.real(analytic_2))
        envelope_line.set_data(distance, magnitude)
        phase_position = spectrum["phase_speed"] * state["time"]
        phase_line.set_xdata([phase_position, phase_position])
        packet_center = spectrum["group_speed"] * state["time"]
        center_line.set_xdata([packet_center, packet_center])

        ci = spectrum["center_index"]
        component_1, component_2 = state["polarization_components"]
        phase = np.linspace(0, 2*np.pi, 241)
        curve1 = np.real(spectrum[component_1][ci] * np.exp(-1j*phase))
        curve2 = np.real(spectrum[component_2][ci] * np.exp(-1j*phase))
        scale = max(np.max(np.hypot(curve1, curve2)), 1e-15)
        ellipse_line.set_data(curve1/scale, curve2/scale)
        now = spectrum["omega0"] * state["time"]
        ellipse_point.set_data([
            np.real(spectrum[component_1][ci]*np.exp(-1j*now))/scale
        ], [np.real(spectrum[component_2][ci]*np.exp(-1j*now))/scale])
        fig.canvas.draw_idle()
        return line_e1, line_e2, envelope_line, center_line, ellipse_line, ellipse_point

    def animate(_frame):
        if state["playing"] and state["spectrum"] is not None:
            state["time"] += 2*np.pi/state["spectrum"]["omega0"] / 24.0
            period = 2*np.pi/state["spectrum"]["omega0"]
            if state["time"] > 8*period:
                state["time"] = 0.0
        return draw_frame()

    def toggle_play(_event):
        state["playing"] = not state["playing"]
        play_button.label.set_text("Pause" if state["playing"] else "Play")

    def reset(_event):
        state["time"] = 0.0
        draw_frame()

    for slider in sliders:
        slider.on_changed(rebuild)
    branch_radio.on_clicked(rebuild)
    play_button.on_clicked(toggle_play)
    reset_button.on_clicked(reset)
    animation = FuncAnimation(fig, animate, interval=40, blit=False, cache_frame_data=False)
    fig._wave_widgets = (*sliders, branch_radio, play_button, reset_button,
                         *preset_buttons, animation)
    rebuild()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density-exp", type=float, default=18.)
    parser.add_argument("--B0", type=float, default=1.)
    parser.add_argument("--mass-ratio", type=float, default=1/1836.15)
    parser.add_argument("--theta", type=float, default=0.)
    parser.add_argument("--fixed-x", type=float, default=.08)
    parser.add_argument("--bandwidth", type=float, default=.04)
    args = parser.parse_args()
    build_wave_explorer(args.density_exp, args.B0, args.mass_ratio,
                        args.theta, args.fixed_x, args.bandwidth)
    plt.show()


if __name__ == "__main__":
    main()
