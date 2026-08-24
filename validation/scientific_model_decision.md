# Scientific model decision

## Adopted model

The production default is a positive latent-log Gaussian process with an
additive Gaussian observation model in the reported physical units:

\[
z(R)=\beta+u(R),\qquad
u\sim\mathcal{GP}(0,k_{5/2}),\qquad
f(R)=f_{\mathrm{ref}}e^{z(R)},\qquad
y_i\mid z\sim\mathcal N(f(R_i),\sigma_i^2).
\]

This choice guarantees positive temperature and density without changing a
reported additive standard uncertainty into a lognormal measurement model. The
stationary Matérn-5/2 prior is retained because it supports a regular first
derivative without introducing weakly identified change points or spatially
varying length scales into a 30-channel profile.

The nonlinear posterior is approximated by log-space quadrature over the radial
correlation length. At each quadrature node, a Laplace approximation covers the
latent profile, constant log intercept, and covariance amplitude. Component
weights include the approximate Laplace evidence and the stated log-length
prior. This corrected a serious joint-MAP undercoverage failure found during
synthetic falsification.

## Derived quantities

Matérn covariance derivatives are used analytically to sample the joint
posterior of \(z(R)\) and \(z'(R)\). Samples are transformed as

\[
f'(R)=f(R)z'(R),\qquad
G_R(R)=\frac{d\log f}{dR}=z'(R),\qquad
L_{f,R}(R)=\frac{1}{z'(R)}.
\]

The logarithmic gradient is the primary normalized-gradient output. The scale
length is withheld when the selected posterior interval cannot identify the
gradient sign. It is also withheld within two channel spacings of either
measurement boundary, where derivative calibration tests showed material
one-sided endpoint bias. Those endpoint regions remain visible and shaded on
the derivative plots.

The coordinate is geometric major radius \(R\), so the app does not label the
result as an outward flux-surface gradient or \(a/L_f\). Such a conversion needs
an equilibrium mapping and branchwise sign convention.

## Data and boundary policy

The app uses all finite positive observations and their heteroscedastic
uncertainties by default. A relative-uncertainty filter remains available only
as a sensitivity check. A provisional validity gate withholds ordinary
reconstruction when the median relative uncertainty exceeds one or fewer than
six channels have \(y_i/\sigma_i>2\). The rule is explicitly provisional because
the supplied HDF5 file does not expose upstream MPTS status flags.

No boundary condition is active by default. An optional external edge
observation requires a coordinate beyond the final measurement, a value, and an
uncertainty. The former synthetic datum at the last channel was removed because
it was only a duplicate observation, not a physical boundary condition.

The empirical-Bayes linear-space GP remains available as a labeled exploratory
baseline. A good-and-bad Gaussian mixture likelihood is deferred until
leave-one-channel-out checks establish isolated channels whose supplied
uncertainties are under-reported.

## Validation evidence

The final compact repeated calibration uses 20 noise realizations for linear,
peaked, and flat positive truths at 25 synthetic measurement locations and a
60-point evaluation grid. With the same 400 posterior samples and 24
hyperparameter-mixture draws used by the app:

| Truth | Mean profile coverage | Mean reportable gradient coverage | Mean reportable log-gradient coverage |
| --- | ---: | ---: | ---: |
| Linear | 91.3% | 93.8% | 94.3% |
| Peaked | 94.1% | 96.3% | 96.2% |
| Flat | 95.2% | 100.0% | 100.0% |

The flat-profile scale length is withheld over the entire grid. All synthetic
positive-profile lower bands remain positive. These are compact hackathon
checks, not a substitute for the larger 72-cell, 100–200 repetition validation
matrix proposed by the scientific review.

A five-node, three-mixture-draw numerical sweep then completed all 1,486 bundled
temperature/density profiles without a fit exception. No positive-model posterior
median or lower credible band crossed zero; the earlier linear baseline produced
a negative posterior median in 590 profiles. The provisional validity gate labels
1,463 profiles usable, seven cautionary, and 16 unusable. The positive model is
deliberately more conservative for reciprocal scale lengths: 675 profiles have
more than 90% of their radius withheld because the sign or endpoint condition is
not identified, compared with 170 under the narrower plug-in linear baseline.

## Remaining work before scientific deployment

1. Compare the Laplace/quadrature approximation against HMC or NUTS on strong,
   flat, isolated-outlier, and no-plasma cases.
2. Obtain and consume upstream channel validity, saturation, over-range, and
   under-range flags.
3. Run leave-one-channel-out and radial-block posterior predictive checks before
   enabling a good-and-bad likelihood mixture or campaign-level uncertainty
   calibration parameters.
4. Add an equilibrium ensemble before interpreting the result as an outward
   flux-coordinate derivative.
5. Expand repeated calibration to multiple noise levels, missing-edge coverage,
   contaminated channels, and two-scale edge profiles.

## Scientific references

- M. A. Chilenski et al., [Gaussian-process profile and gradient uncertainty in
  fusion diagnostics](https://dspace.mit.edu/entities/publication/597ca610-4c74-49e7-9255-f029dbca945a).
- S. Kwak et al., [Bayesian Thomson/interferometer modeling with GP
  hyperparameter marginalization](https://pure.kaist.ac.kr/en/publications/bayesian-modelling-of-thomson-scattering-and-multichannel-interfe/).
- E. Solak et al., [analytic derivative covariance in Gaussian-process
  models](https://papers.nips.cc/paper_files/paper/2002/hash/5b8e4fd39d9786228649a8a8bec4e008-Abstract.html).
- F. M. Laggner et al., [NSTX-U MPTS geometry and analysis
  framework](https://www.osti.gov/servlets/purl/1510312).
