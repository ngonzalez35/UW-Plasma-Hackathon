# MPTS Bayesian Profile Lab

Streamlit application for reconstructing NSTX multipoint Thomson-scattering
(MPTS) electron-temperature and electron-density profiles with quantified
uncertainty in the profiles and their radial derivatives.

## Two different fitting stages

The NSTX-U diagnostic first fits calibrated polychromator signals to a Thomson
scattering spectrum. Schematically,

\[
\mu_j(T_e,n_e)=b_j+A E_L n_e
\int \mathcal T_j(\lambda)\eta_j(\lambda)
S_{\lambda}^{\mathrm{Selden}}(\lambda;T_e,\theta)\,d\lambda.
\]

The spectral shape and width constrain \(T_e\), while the calibrated amplitude
constrains \(n_e\). The [NSTX-U MPTS analysis
framework](https://www.osti.gov/servlets/purl/1510312) uses Selden-spectrum
predictions, measured filter responses, detector calibration, and spectral
chi-squared minimization for this upstream stage.

This repository does **not** re-fit the raw spectrum. `nstx-profiles.hdf5`
already contains the processed \(T_e\), \(n_e\), and reported uncertainties. The
application starts at the second stage: Bayesian reconstruction of continuous
spatial profiles from those discrete measurements.

The default reconstruction is a positive latent-log Gaussian process:

\[
z\sim\mathcal{GP}(\beta,k_{5/2}),\qquad
f(R)=f_{\mathrm{ref}}e^{z(R)},\qquad
y_i\mid z\sim\mathcal N(f(R_i),\sigma_i^2).
\]

This separates two scientific statements that should not be conflated: the
physical profile is positive, while the processed diagnostic uncertainty remains
additive Gaussian noise in its reported keV or cm^-3 units. The interactive
approximation performs log-space quadrature over the correlation length and a
Laplace approximation over the latent profile, intercept, and kernel amplitude
at each node. A linear-space empirical-Bayes GP is retained and labeled as an
exploratory comparison baseline.

The primary derived output is

\[
G_R=\frac{d\log f}{dR}=z'(R).
\]

Analytic Matérn-5/2 kernel derivatives produce the joint posterior for \(z\) and
\(z'\). The scale length \(L_{f,R}=1/G_R\) is secondary and is not reported at
radii where the posterior does not identify the gradient sign or within two
channel spacings of a one-sided measurement boundary. Those endpoint regions are
shaded on derivative plots. No finite value is created by clipping the reciprocal
singularity.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The default selection, shot `141687` near `0.315 s`, is a high-signal demo case.

## Test and validate it

```bash
source .venv/bin/activate
pytest -q
python scripts/validate_profiles.py
python scripts/validate_synthetic.py
```

The validation scripts write machine-readable results under `validation/`.
The app also applies a clearly labeled provisional validity check based on median
relative uncertainty and the number of channels with signal-to-uncertainty above
two. It refuses a normal reconstruction for obviously uninformative acquisitions
unless the user explicitly enables a diagnostic override. Upstream MPTS validity
flags should replace this provisional rule when they become available.

The rationale, adopted equations, compact calibration results, and remaining
scientific work are recorded in
[`validation/scientific_model_decision.md`](validation/scientific_model_decision.md).

## Data schema

For every top-level shot, `nstx-profiles.hdf5` contains:

| Path | Shape | Units |
| --- | --- | --- |
| `/{shot}/mpts/radius` | `(30,)` | cm |
| `/{shot}/mpts/time` | `(N_time,)` | s |
| `/{shot}/mpts/te`, `te_error` | `(30, N_time)` | keV |
| `/{shot}/mpts/ne`, `ne_error` | `(30, N_time)` | cm^-3 |

The coordinate is geometric machine major radius \(R\) along a whole diagnostic
chord. The app therefore reports \(df/dR\) and \(d\log f/dR\), not an outward
flux-surface gradient or \(a/L_f\). Those quantities require an equilibrium
mapping, an outward branch convention, and propagation of equilibrium uncertainty.

## Deliberate scope

- Matérn-5/2 is the only kernel in the positive production model. RBF and
  rational-quadratic kernels remain available only with the baseline.
- Correlation lengths shorter than the typical channel spacing are excluded from
  automatic inference.
- Pointwise posterior credible intervals are shown; they are not simultaneous
  bands or frequentist coverage guarantees.
- An external edge observation is disabled by default and requires a distinct
  physical coordinate, value, and uncertainty.
- Temperature, density, and time slices are fitted independently.
- A good-and-bad likelihood mixture is deferred until leave-one-channel-out checks
  demonstrate isolated, under-reported channel errors.

## Scientific basis

- Chilenski et al., [Gaussian-process profile and gradient uncertainty in fusion
  diagnostics](https://dspace.mit.edu/entities/publication/597ca610-4c74-49e7-9255-f029dbca945a).
- Kwak et al., [joint Bayesian Thomson/interferometer modeling with GP
  hyperparameter marginalization](https://pure.kaist.ac.kr/en/publications/bayesian-modelling-of-thomson-scattering-and-multichannel-interfe/).
- Solak et al., [derivative observations and analytic covariance derivatives in
  Gaussian processes](https://papers.nips.cc/paper_files/paper/2002/hash/5b8e4fd39d9786228649a8a8bec4e008-Abstract.html).
- Laggner et al., [NSTX-U MPTS geometry and analysis
  framework](https://www.osti.gov/servlets/purl/1510312).

The implementation is a hackathon-scale approximation, not a validated production
diagnostic. Its Laplace/quadrature approximation should be compared with HMC on
representative high-signal, flat, contaminated-channel, and no-plasma cases before
scientific deployment.
