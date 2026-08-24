# Thomson-scattering equations appropriate for the app

## Decision

It is scientifically relevant to show the Thomson-scattering physics, but it
must not be described as the model fitted by this application.

The HDF5 input already contains processed $T_e(R)$, $n_e(R)$, and their
uncertainties. The application fits those profile points with a positive
latent-log Gaussian process. It does **not** ingest polychromator signals or
refit a scattered-light spectrum. The NSTX-U MPTS analysis upstream of this
application calculated expected spectral-channel intensities from the Selden
Thomson spectrum, fitted temperature and amplitude by a chi-squared
minimization, and converted the calibrated amplitude to density. The NSTX-U
paper explicitly identifies spectral broadening with $T_e$, signal amplitude
with $n_e$, and the delivered outputs as $T_e,n_e$ and their errors
([Laggner et al., 2019, open manuscript](https://www.osti.gov/servlets/purl/1510312)).

Therefore the math section should have two clearly labeled stages:

1. **How Thomson scattering produced the input points (upstream diagnostic).**
2. **The Bayesian profile model used in this application.**

Do not use the heading “Thomson model used for this fit.” It would imply that
the application fits raw spectra.

## Recommended compact upstream equation

For polychromator channel (j), a defensible schematic forward model is

\[
\mu_j(T_e,n_e)
=b_j+A\,E_L\,n_e
\int
\mathcal T_j(\lambda)\,\eta_j(\lambda)\,
S_{\lambda}^{\rm TS}(\lambda;T_e,\theta,\ldots)\,d\lambda .
\]

Here

- \(\mu_j\) is the expected background-subtracted detector signal, with
  \(b_j\) included if the background has not already been subtracted;
- \(E_L\) is the measured laser-pulse energy;
- \(A\) collects the scattering volume and solid angle, Thomson cross section,
  optical throughput, gain, and absolute Rayleigh/Raman calibration;
- \(\mathcal T_j(\lambda)\) is the transmission of interference filter
  channel \(j\);
- \(\eta_j(\lambda)\) represents wavelength-dependent detector efficiency;
- \(S_{\lambda}^{\rm TS}\) is a normalized Thomson spectral shape; and
- \(\theta\) is the scattering angle.

This equation is a transparent synthesis of the actual NSTX-U acquisition
chain, not a verbatim equation from the paper. Laggner et al. state that MPTS
uses four- or six-filter polychromators with avalanche photodiodes, computes a
Selden spectrum, combines it with filter transmission, detector efficiency,
and amplification gain, fits $T_e$ and amplitude by chi-squared
minimization, and obtains $n_e$ from Rayleigh/Raman calibration
([Laggner et al., 2019](https://www.osti.gov/servlets/purl/1510312)). The
separate NSTX alignment study confirms that temperature information is in the
spectral spread whereas density requires absolute calibration and alignment
([LeBlanc and Diallo, PPPL-4948](https://bp-pub.pppl.gov/pub_report/2014/PPPL-4948.pdf)).

A compact schematic of the upstream parameter fit is

\[
\widehat T_e,\widehat a
=\underset{T_e,a}{\arg\min}\;
\sum_j
\frac{\left[I_j^{\rm obs}-\mu_j(T_e,a)\right]^2}
{\sigma_{I,j}^2},
\qquad
\widehat n_e=\mathcal C_{\rm abs}(\widehat a,E_L),
\]

where \(a\) is the fitted spectral amplitude and
\(\mathcal C_{\rm abs}\) denotes the campaign's Rayleigh/Raman absolute
calibration. This reflects the workflow reported for NSTX-U without claiming
that the app has access to its raw channel likelihood or complete calibration
model.

## Why $T_e$ changes width while $n_e$ changes amplitude

In the elementary non-collective, non-relativistic, one-Maxwellian limit, the
electron feature has the schematic form

\[
S_e(k,\omega)
\propto
\frac{1}{k v_{Te}}
\exp\!\left[
-\frac{(\omega-\mathbf k\!\cdot\!\mathbf u_e)^2}
{k^2v_{Te}^2}
\right],
\qquad
v_{Te}=\sqrt{\frac{2k_{\rm B}T_e}{m_e}}.
\]

Thus a larger $T_e$ produces a broader frequency or wavelength distribution,
while the number of scatterers supplies an overall factor proportional to
$n_e$. This Gaussian expression is useful intuition only. At fusion-plasma
temperatures, the wavelength-domain spectrum has relativistic corrections and
asymmetry; NSTX-U reports using the relativistic Selden form rather than this
Gaussian approximation. The original model is A. C. Selden, “Simple analytic
form of the relativistic Thomson scattering spectrum,” *Physics Letters A* 79,
405–406 (1980),
[DOI 10.1016/0375-9601(80)90276-5](https://doi.org/10.1016/0375-9601(80)90276-5).

## General spectral-density equation and regime caveat

If the interface has room for a “full plasma spectral density” disclosure, the
Maxwellian multi-species expression implemented by PlasmaPy is

\[
S(k,\omega)=
\sum_e \frac{2\pi}{k}
\left|1-\frac{\chi_e}{\epsilon}\right|^2
f_{e0,e}\!\left(\frac{\omega}{k}\right)
+
\sum_i \frac{2\pi}{k}\frac{Z_i^2}{\bar Z}
\left|\frac{\chi_e}{\epsilon}\right|^2
f_{i0,i}\!\left(\frac{\omega}{k}\right),
\]

with

\[
\epsilon=1+\sum_e\chi_e+\sum_i\chi_i,
\qquad
\alpha=\frac{1}{k\lambda_{De}}.
\]

The distribution functions are Maxwellian in this implementation. The
scattering parameter separates the regimes:
\(\alpha<1\) is non-collective and \(\alpha>1\) is collective. In the
collective regime, susceptibilities and plasma-wave features make both shape
and density dependence more involved than the simple “width versus amplitude”
picture. PlasmaPy documents the equation, definitions, regime criterion, and
instrument-function convolution in its official
[`spectral_density` documentation](https://docs.plasmapy.org/en/stable/api/plasmapy.diagnostics.thomson.spectral_density.html)
and provides a separate official
[spectrum-fitting notebook](https://docs.plasmapy.org/en/stable/notebooks/diagnostics/thomson_fitting.html).

For this dataset, NSTX literature describes the MPTS system as **incoherent**
Thomson scattering and identifies the Selden approximation as the analysis
model. The general collective expression is therefore useful background, not
a statement that the NSTX-U pipeline used PlasmaPy or fit collective features.

## The actual model fitted by the application

Immediately after the upstream explanation, retain the profile likelihood that
the application actually evaluates. For either
\(q\in\{T_e,n_e\}\),

\[
z_q\sim\mathcal{GP}(\beta_q,k_{5/2}),
\qquad
f_q(R)=f_{{\rm ref},q}e^{z_q(R)},
\qquad
y_{q,i}\mid z_q
\sim\mathcal N\!\left(f_q(R_i),\sigma_{q,i}^2\right).
\]

This stage reconstructs a positive spatial profile from already inferred
measurements. Its observation uncertainty is in physical units (keV or density
units); it is not the raw photon-counting or APD-channel noise model.

## Suggested UI text

> **How Thomson scattering produced these measurements (upstream)**
> NSTX-U MPTS inferred each local $T_e$ from the shape of the scattered-light
> spectrum and $n_e$ from its absolutely calibrated amplitude. The detector
> signals were integrated over several polychromator filters and compared with
> a relativistic Selden spectrum. This app starts from the resulting
> $T_e,n_e$, and uncertainty points; it does not refit the raw optical
> spectrum.

Then show only the compact channel equation

\[
\mu_j=b_j+A E_L n_e\int \mathcal T_j(\lambda)\eta_j(\lambda)
S_{\lambda}^{\rm Selden}(\lambda;T_e,\theta)\,d\lambda,
\]

followed by one sentence: “$T_e$ controls the spectral shape and broadening;
$n_e$ controls the calibrated scattered-light amplitude.” The full
susceptibility equation can live in a nested “General collective/non-collective
form” disclosure to avoid overwhelming the primary profile-reconstruction
story.

## Claims the UI should avoid

- “This application fits the Thomson scattering spectrum.”
- “The GP is the Thomson scattering model.”
- “The measured spectrum is exactly Gaussian.”
- “Density affects only amplitude” without restricting that sentence to the
  incoherent/non-collective interpretation.
- “PlasmaPy generated the NSTX-U values.” PlasmaPy is a corroborating
  authoritative implementation of the general theory, not the provenance of
  this dataset.
- Writing the general $S(k,\omega)$ equation without stating whether it is a
  spectral shape, a dynamic structure factor, or the complete detector-signal
  model. Geometry, cross section, laser energy, throughput, filter response,
  detector efficiency, calibration, and background are separate factors.

## Primary and authoritative sources

1. F. M. Laggner et al., “A scalable real-time framework for Thomson
   scattering analysis: Application to NSTX-U,” *Review of Scientific
   Instruments* **90**, 043501 (2019),
   [open DOE manuscript](https://www.osti.gov/servlets/purl/1510312),
   [DOI 10.1063/1.5088248](https://doi.org/10.1063/1.5088248).
2. B. P. LeBlanc and A. Diallo, “Alignment of the Thomson scattering
   diagnostic on NSTX,” PPPL-4948 (2013),
   [official PPPL report](https://bp-pub.pppl.gov/pub_report/2014/PPPL-4948.pdf).
3. PlasmaPy Project, [`plasmapy.diagnostics.thomson.spectral_density`](https://docs.plasmapy.org/en/stable/api/plasmapy.diagnostics.thomson.spectral_density.html),
   official stable documentation and linked source implementation.
4. PlasmaPy Project, [“Fitting Thomson Scattering Spectra”](https://docs.plasmapy.org/en/stable/notebooks/diagnostics/thomson_fitting.html),
   official stable example notebook.
5. A. C. Selden, “Simple analytic form of the relativistic Thomson scattering
   spectrum,” *Physics Letters A* **79**, 405–406 (1980),
   [DOI 10.1016/0375-9601(80)90276-5](https://doi.org/10.1016/0375-9601(80)90276-5).
