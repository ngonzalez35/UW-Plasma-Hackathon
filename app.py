"""Interactive Bayesian reconstruction of NSTX Thomson-scattering profiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from thomson_profiles.data import list_shots, list_times, load_profile_slice
from thomson_profiles.gp import FitConfig, assess_profile_validity, fit_profile
from thomson_profiles.plotting import make_profile_figure


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "nstx-profiles.hdf5"
PEDAGOGICAL_PDF_PATH = (
    ROOT / "output" / "pdf" / "mpts_bayesian_profile_lab_pedagogical_companion.pdf"
)

st.set_page_config(page_title="MPTS Bayesian Profile Lab", page_icon="⚛️", layout="wide")


@st.cache_data(show_spinner=False)
def cached_shots(path: str) -> list[str]:
    return list_shots(path)


@st.cache_data(show_spinner=False)
def cached_times(path: str, shot: str) -> np.ndarray:
    return list_times(path, shot)


@st.cache_data(show_spinner=False)
def cached_slice(path: str, shot: str, time_index: int):
    return load_profile_slice(path, shot, time_index)


@st.cache_data(show_spinner=False)
def cached_fit(radius: np.ndarray, values: np.ndarray, errors: np.ndarray, config: FitConfig):
    return fit_profile(radius, values, errors, config)


st.title("MPTS Bayesian Profile Lab")
st.caption(
    "Positive Bayesian reconstruction of NSTX Thomson-scattering temperature and "
    "density profiles, with analytically propagated profile and gradient uncertainty."
)

st.info(
    "**What this app fits:** The HDF5 file already contains processed measurements of "
    "Tₑ, nₑ, and their uncertainties at discrete radii. This app fits those profile points; "
    "it does not fit the raw scattered-light spectrum."
)

if PEDAGOGICAL_PDF_PATH.exists():
    st.download_button(
        "Download the pedagogical project guide (PDF)",
        data=PEDAGOGICAL_PDF_PATH.read_bytes(),
        file_name=PEDAGOGICAL_PDF_PATH.name,
        mime="application/pdf",
        help=(
            "A six-page companion explaining the diagnostic data, Bayesian model, "
            "derivatives, validation, and scientific limitations."
        ),
    )
else:
    st.caption("The pedagogical project guide has not been built in this checkout.")

with st.expander("Bayesian profile model used by this app", expanded=True):
    st.markdown(
        "For either $f(R)=T_e(R)$ or $f(R)=n_e(R)$, the default model places a "
        "Gaussian-process prior on a dimensionless log profile while retaining the "
        "reported additive uncertainty in the original physical units:"
    )
    st.latex(
        r"""
        z(R)=\beta+u(R),\qquad
        u\sim\mathcal{GP}(0,k_{5/2}),\qquad
        f(R)=f_{\rm ref}e^{z(R)},\qquad
        y_i\mid z\sim\mathcal N\!\left(f_{\rm ref}e^{z(R_i)},\sigma_i^2\right).
        """
    )
    st.markdown("The stationary Matérn-$5/2$ covariance is")
    st.latex(
        r"""
        k_{5/2}(R,R')=\sigma_z^2
        \left(1+\frac{\sqrt5d}{\ell}+\frac{5d^2}{3\ell^2}\right)
        \exp\!\left(-\frac{\sqrt5d}{\ell}\right),\qquad d=|R-R'|.
        """
    )
    st.markdown(
        "The nonlinear likelihood makes the posterior non-Gaussian. The interactive "
        "implementation uses log-space quadrature over the correlation length and a Laplace "
        "approximation over the latent values, intercept, and amplitude at each node, then "
        "uses analytic kernel derivatives."
    )
    st.latex(
        r"""
        f'(R)=f(R)z'(R),\qquad
        G_R(R)=\frac{d\log f}{dR}=z'(R),\qquad
        L_{f,R}(R)=\frac{1}{z'(R)}.
        """
    )
    st.caption(
        "Bands are pointwise posterior credible intervals. A finite scale length is "
        "withheld wherever the selected posterior interval does not identify the gradient sign."
    )

with st.expander(
    "Background: how Thomson scattering produced these measurements", expanded=False
):
    st.markdown(
        "This is the upstream diagnostic stage, included for physical context. It is not "
        "evaluated by this application."
    )
    st.markdown(
        "A calibrated polychromator measures the laser light scattered into several "
        "wavelength bands. A compact detector-channel model for the NSTX-U analysis is"
    )
    st.latex(
        r"""
        \mu_j(T_e,n_e)=b_j+A E_L n_e
        \int \mathcal T_j(\lambda)\,\eta_j(\lambda)\,
        S_{\lambda}^{\rm Selden}(\lambda;T_e,\theta)\,d\lambda .
        """
    )
    st.markdown(
        r"Here $\mathcal T_j$ and $\eta_j$ are the filter transmission and detector "
        "efficiency, while the remaining geometry, throughput, and gain are collected in "
        r"$A$. The temperature and common amplitude are obtained from a channel-space fit:"
    )
    st.latex(
        r"""
        (\widehat T_e,\widehat a)
        =\arg\min_{T_e,a}\sum_j
        \frac{[I_j^{\rm obs}-\mu_j(T_e,a)]^2}{\sigma_{I,j}^2},
        \qquad
        \widehat n_e=\mathcal C_{\rm abs}(\widehat a,E_L).
        """
    )
    st.caption(
        "$E_L$ is the measured laser-pulse energy, $b_j$ is background, and "
        "$\\mathcal C_{\\rm abs}$ denotes the Rayleigh/Raman absolute-density calibration. "
        "The displayed detector equation is a compact representation of the reported NSTX-U "
        "analysis chain, not source code from the upstream diagnostic."
    )
    st.markdown(
        "Thus spectral broadening constrains $T_e$, while calibrated scattered-light "
        "amplitude constrains $n_e$. NSTX-U uses Selden-spectrum predictions, measured "
        r"filter responses, detector calibration, and a spectral $\chi^2$ fit. See the "
        "[NSTX-U MPTS analysis](https://www.osti.gov/servlets/purl/1510312) and the "
        "[general Thomson spectral-density model](https://docs.plasmapy.org/en/stable/"
        "api/plasmapy.diagnostics.thomson.spectral_density.html)."
    )

if not DATA_PATH.exists():
    st.error(f"Missing data file: {DATA_PATH}")
    st.stop()

shots = cached_shots(str(DATA_PATH))
default_shot = shots.index("141687") if "141687" in shots else 0

with st.sidebar:
    st.header("Profile")
    shot = st.selectbox("Shot", shots, index=default_shot)
    times = cached_times(str(DATA_PATH), shot)
    default_time = int(np.argmin(np.abs(times - 0.315)))
    time_index = st.select_slider(
        "Acquisition time",
        options=list(range(len(times))),
        value=default_time,
        format_func=lambda index: f"{times[index]:.4f} s",
    )
    quantity_key = st.radio(
        "Quantity",
        options=["te", "ne"],
        format_func=lambda key: (
            "Electron temperature  Tₑ" if key == "te" else "Electron density  nₑ"
        ),
    )

profile = cached_slice(str(DATA_PATH), shot, time_index)
quantity = profile.quantity(quantity_key)
validity = assess_profile_validity(quantity.values, quantity.errors)

with st.sidebar:
    st.subheader("Provisional profile validity")
    status_icon = {"Usable": "✅", "Caution": "⚠️", "Unusable": "⛔"}[validity.status]
    st.markdown(f"{status_icon} **{validity.status}** — {validity.message}")
    st.caption(
        f"Median σ/y: {validity.median_relative_uncertainty:.2f}; "
        f"resolved channels (y/σ > 2): {validity.resolved_points}/{validity.total_points}. "
        "This rule is provisional because upstream diagnostic status bits are absent."
    )
    show_unusable = st.toggle(
        "Show diagnostic fit anyway",
        value=False,
        disabled=validity.status != "Unusable",
    )

    st.divider()
    st.header("Inference model")
    model_name = st.selectbox(
        "Profile model",
        ["Positive latent-log GP", "Linear GP baseline"],
        help="The baseline is retained for sensitivity comparisons and may predict negative profiles.",
    )
    positive_model = model_name == "Positive latent-log GP"
    kernel_options = (
        ["Matern 5/2"]
        if positive_model
        else ["Matern 5/2", "RBF", "Rational quadratic"]
    )
    kernel = st.selectbox("Kernel", kernel_options)
    mean_prior = st.selectbox(
        "Mean function",
        ["Constant (weighted mean)", "Zero"],
        disabled=positive_model,
        help=(
            "The positive model instead infers a constant log-space intercept."
            if positive_model
            else None
        ),
    )
    optimize_length = st.toggle("Infer correlation length", value=True)
    length_scale = st.slider(
        "Initial correlation length (cm)" if optimize_length else "Correlation length (cm)",
        min_value=1.0,
        max_value=80.0,
        value=15.0,
        step=1.0,
    )

    st.subheader("Boundary information")
    edge_mode = st.selectbox(
        "External edge observation", ["None", "User-specified physical point"]
    )
    edge_enabled = edge_mode != "None"
    edge_radius = st.number_input(
        "Edge coordinate R (cm)",
        min_value=float(np.max(profile.radius_cm) + 0.01),
        value=float(np.max(profile.radius_cm) + 1.0),
        step=0.5,
        disabled=not edge_enabled,
    )
    edge_value = st.number_input(
        f"Assumed value ({quantity.units})",
        min_value=0.0,
        value=0.0,
        format="%.4e" if quantity_key == "ne" else "%.4f",
        disabled=not edge_enabled,
    )
    edge_uncertainty = st.number_input(
        f"Assumed uncertainty ({quantity.units})",
        min_value=float(max(np.median(quantity.errors) * 1e-3, 1e-15)),
        value=float(np.median(quantity.errors)),
        format="%.4e" if quantity_key == "ne" else "%.4f",
        disabled=not edge_enabled,
    )
    st.caption(
        "Disabled by default. Only use a distinct coordinate, value, and uncertainty supplied "
        "by equilibrium reconstruction or another physical source."
    )

    st.divider()
    st.header("Posterior")
    confidence = st.slider("Credible level", 0.80, 0.99, 0.95, 0.01)
    n_samples = st.slider("Posterior samples", 100, 1000, 400, 100)
    hyperparameter_samples = st.slider(
        "Hyperparameter-mixture draws", 8, 48, 24, 4, disabled=not positive_model
    )
    derived_kind = st.radio(
        "Bottom panel",
        [
            "Log gradient  d(log f)/dR",
            "Gradient sign probability",
            "Scale length  1/[d(log f)/dR]",
        ],
    )

    with st.expander("Diagnostic data filter"):
        filter_relative = st.toggle("Exclude very uncertain points", value=False)
        relative_limit = st.slider(
            "Maximum relative uncertainty",
            0.25,
            10.0,
            2.0,
            0.25,
            disabled=not filter_relative,
            help=(
                "Large uncertainties already reduce likelihood weight. Use this only as a "
                "sensitivity check, not as a diagnostic-validity replacement."
            ),
        )

if validity.status == "Unusable" and not show_unusable:
    st.warning(
        "The available channels do not contain sufficient evidence for a scientifically "
        "usable plasma profile at this acquisition time. Enable the diagnostic-fit override "
        "in the sidebar only to inspect the failure mode."
    )
    st.stop()

config = FitConfig(
    model=model_name,
    kernel=kernel,
    mean_prior=mean_prior,
    optimize_length_scale=optimize_length,
    length_scale_cm=length_scale,
    confidence=confidence,
    n_samples=n_samples,
    edge_mode=edge_mode,
    edge_radius_cm=edge_radius if edge_enabled else None,
    edge_value=edge_value,
    edge_uncertainty=edge_uncertainty if edge_enabled else None,
    max_relative_uncertainty=relative_limit if filter_relative else None,
    random_seed=2026 if quantity_key == "te" else 2027,
    reference_scale=1.0 if quantity_key == "te" else 1e13,
    hyperparameter_samples=hyperparameter_samples,
)

try:
    with st.spinner("Sampling the profile and analytic derivative posterior…"):
        fit = cached_fit(profile.radius_cm, quantity.values, quantity.errors, config)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

unstable_fraction = float(np.mean(fit.scale_unstable))
metric_columns = st.columns(5)
metric_columns[0].metric(f"Time — shot {shot}", f"{profile.time_s:.4f} s")
metric_columns[1].metric("Measurements used", f"{fit.n_observations} / {len(quantity.values)}")
metric_columns[2].metric("Length scale", f"{fit.fitted_length_scale_cm:.2f} cm")
metric_columns[3].metric("Scale withheld", f"{100 * unstable_fraction:.0f}%")
metric_columns[4].metric("Mixture components", str(fit.hyperparameter_modes))

figure = make_profile_figure(quantity, fit, derived_kind)
st.plotly_chart(figure, width="stretch", config={"displaylogo": False})

if derived_kind == "Scale length  1/[d(log f)/dR]":
    st.info(
        "Gray regions are intentionally undefined: either the posterior assigns appreciable "
        "probability to both gradient signs, or the radius lies within two channel spacings of "
        "a one-sided measurement boundary."
    )
elif derived_kind == "Log gradient  d(log f)/dR":
    st.info(
        "The bottom panel is d log f/dR with respect to geometric major radius R. Its sign "
        "does not represent one globally outward direction across the whole diagnostic chord. "
        "Light-gray endpoint regions identify derivatives constrained from only one side."
    )

with st.expander("Model and selected measurements"):
    st.write(f"**{fit.model_name}**")
    st.code(fit.kernel_description, language=None)
    st.caption(f"{fit.fit_score_label}: {fit.fit_score:.3f}")
    frame = pd.DataFrame(
        {
            "R (cm)": profile.radius_cm,
            quantity.title: quantity.values,
            f"σ ({quantity.units})": quantity.errors,
            "relative σ": quantity.errors / quantity.values,
        }
    )
    st.dataframe(frame, hide_index=True, width="stretch")

st.caption(
    "R is geometric major radius. Conversion to an outward flux-surface gradient or a/L "
    "requires a magnetic-equilibrium mapping and propagation of its uncertainty."
)
