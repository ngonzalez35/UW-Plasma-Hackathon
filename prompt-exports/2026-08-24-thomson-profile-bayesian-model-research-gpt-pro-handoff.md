# GPT-Pro Handoff: Scientific Review of Bayesian Thomson-Profile Reconstruction

**Content mode:** math-first scientific review
**Action mode:** execute research now
**Requested outcome:** a citation-supported verdict, model comparison, and preferred mathematical model

## Objective

Perform a rigorous mathematical and plasma-diagnostic review of an existing
Bayesian method for reconstructing continuous electron-temperature and
electron-density profiles from multipoint Thomson-scattering measurements with
heteroscedastic uncertainties.

Determine whether the present empirical-Bayes Gaussian-process model is
scientifically defensible for a hackathon-quality tool. Investigate whether a
meaningfully better model exists for this exact problem. If so, select and
derive the preferred model rather than merely listing alternatives. Support the
analysis with verified citations to primary literature and authoritative
technical sources.

The requested review is about mathematics, statistics, uncertainty
quantification, and plasma-profile interpretation. It is not a software code
review.

## Execution Contract

Execute the research now. Do not return a research plan, scoping report,
timeline, search strategy, list of databases, or suggestions for future reading
as a substitute for the requested findings. Search for and verify sources in
this response, synthesize the evidence, derive the relevant mathematical
results, compare the candidate models, and deliver a completed scientific
recommendation.

Begin with substantive findings and a direct verdict. Planning details may
appear only when they explain a genuine limitation of the available evidence.

## What You Must Assume

- You do **not** have access to the source repository.
- You do **not** have the HDF5 data file or any attached files.
- You cannot inspect any local path or artifact mentioned elsewhere.
- You must reason only from the mathematical model, data summary, and numerical
  evidence included in this document.
- You may and should search external literature and authoritative documentation
  to complete the requested research.
- Do not ask the user to provide the repository before giving a verdict. State
  any conclusions that would still require confirmation from raw data, but
  complete the analysis using the evidence supplied here.

## Why This Matters

This is a University of Washington plasma hackathon project. The deliverable is
an interactive application that receives processed multipoint
Thomson-scattering profile measurements and reconstructs

\[
T_e(R), \qquad n_e(R), \qquad
\frac{dT_e}{dR}, \qquad \frac{dn_e}{dR}, \qquad
L_f(R)=\frac{f(R)}{df(R)/dR},
\]

with uncertainty bands propagated from a Bayesian posterior. The tool must be
scientifically honest, understandable in a short demonstration, and practical
on a laptop. A sophisticated model is useful only if it improves the inference
enough to justify its complexity for roughly 30 spatial measurements per time
slice.

The current prototype is already functional. The purpose of this research is
to decide whether its statistical model should be retained, modified, or
replaced before calling the result scientifically defensible.

## Diagnostic Stage and Data Meaning

The input is processed Thomson-scattering profile data, not raw scattered-light
spectra. Spectral inversion has already produced point estimates of electron
temperature and density and their reported standard uncertainties. The model
therefore addresses spatial profile reconstruction:

\[
\mathcal D_T=\{(R_i,T_{e,i},\sigma_{T,i})\}_{i=1}^{N},
\qquad
\mathcal D_n=\{(R_i,n_{e,i},\sigma_{n,i})\}_{i=1}^{N}.
\]

Do not conflate this problem with fitting the Thomson-scattered optical
spectrum itself. The requested literature review may discuss upstream
uncertainty semantics where relevant, but the primary inverse problem begins
with spatial profile points.

### Available data

- Device/data source: NSTX multipoint Thomson scattering (MPTS).
- Number of shots: 14.
- Total acquisition times: 743.
- Quantities fitted independently: two, giving 1,486 profile fits.
- Spatial observations per profile: 30.
- Coordinate: diagnostic major radius \(R\), in centimetres.
- Radius range: approximately \(27.60\) to \(156.17\ \mathrm{cm}\).
- Time range across shots: approximately \(0.015\) to \(1.248\ \mathrm{s}\).
- Temperature: \(T_e\) in keV with reported \(\sigma_T\) in keV.
- Density: \(n_e\) in \(\mathrm{cm}^{-3}\) with reported \(\sigma_n\) in
  \(\mathrm{cm}^{-3}\).
- Every stored value and reported uncertainty is finite and positive, but some
  acquisition times are effectively absent-plasma or failed-diagnostic cases
  with enormous relative uncertainty.
- The array axes are \((\text{radius},\text{time})\); one fit uses one column.

The coordinate is almost certainly geometric major radius across the diagnostic
chord, not normalized minor radius \(r/a\) or a flux coordinate such as
\(\rho_{\mathrm{pol}}\) or \(\rho_{\mathrm{tor}}\). The measurements span both
inboard and outboard sides of a peaked profile. Consequently, a signed
\(df/dR\) naturally changes sign near the profile maximum. No magnetic
equilibrium mapping is present in the supplied data.

## Symbol and Notation Glossary

| Symbol | Meaning |
| --- | --- |
| \(R_i\) | Major-radius location of measurement \(i\), in cm |
| \(y_i\) | Either measured \(T_{e,i}\) or \(n_{e,i}\) |
| \(\sigma_i\) | Reported standard uncertainty of \(y_i\) |
| \(f(R)\) | Latent continuous profile, either \(T_e(R)\) or \(n_e(R)\) |
| \(m(R)\) | GP prior mean |
| \(k_\phi(R,R')\) | Covariance kernel with hyperparameters \(\phi\) |
| \(\ell\) | Radial correlation length |
| \(\sigma_f^2\) | GP covariance amplitude |
| \(\Sigma\) | \(\operatorname{diag}(\sigma_1^2,\ldots,\sigma_N^2)\) |
| \(g(R)\) | Radial derivative \(df/dR\) |
| \(L_f(R)\) | Signed gradient scale length \(f/(df/dR)\) |
| \(L_f^{-1}(R)\) | Inverse normalized gradient \((1/f)(df/dR)\) |
| \(s\) | Index of a correlated posterior function sample |

## Current Mathematical Model

The present model is an exact conditional Gaussian process with
heteroscedastic Gaussian likelihood noise and empirical-Bayes hyperparameters.
Temperature and density are fitted independently at each time.

### Likelihood

For either profile,

\[
y_i=f(R_i)+\epsilon_i,
\qquad
\epsilon_i\sim\mathcal N(0,\sigma_i^2),
\]

so the measurement's reported variance is used directly as likelihood noise:

\[
\mathbf y\mid \mathbf f
\sim \mathcal N(\mathbf f,\Sigma),
\qquad
\Sigma=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_N^2).
\]

There is no additional fitted white-noise or model-discrepancy term.

### Functional prior

\[
f\sim\mathcal{GP}(m,k_\phi).
\]

The default mean is the inverse-variance-weighted average of the selected
profile:

\[
m(R)=\bar y_w,
\qquad
\bar y_w=
\frac{\sum_i y_i/\sigma_i^2}{\sum_i 1/\sigma_i^2}.
\]

A zero-mean option also exists.

The default covariance is Matérn-\(5/2\):

\[
k_{5/2}(R,R')=
\sigma_f^2
\left(
1+\frac{\sqrt 5 d}{\ell}+
\frac{5d^2}{3\ell^2}
\right)
\exp\!\left(-\frac{\sqrt 5 d}{\ell}\right),
\qquad d=|R-R'|.
\]

RBF and rational-quadratic alternatives are exposed to the user. Matérn-5/2
was selected because its sample paths are sufficiently differentiable for a
first derivative while permitting more local structure than an RBF kernel.

### Target and coordinate scaling

The coordinate is mapped affinely to \([0,1]\):

\[
x_i=\frac{R_i-R_{\min}}{R_{\max}-R_{\min}}.
\]

The target and reported uncertainty are scaled together:

\[
\tilde y_i=\frac{y_i-m}{s_y},
\qquad
\tilde\sigma_i=\frac{\sigma_i}{s_y},
\]

where

\[
s_y=\max\left\{
\operatorname{sd}(\mathbf y),
\operatorname{median}(|\mathbf y|),
10^{-12}
\right\}.
\]

This ensures the diagonal likelihood variance remains in the same scaled units
as the target.

### Hyperparameters

The covariance amplitude and radial correlation length are optimized by
maximizing the log marginal likelihood. The initial physical length is
\(15\ \mathrm{cm}\). In normalized coordinate units, the allowed length range
is

\[
0.005 \leq \tilde\ell \leq 2.0,
\]

equivalent for this radial domain to approximately
\(0.64\ \mathrm{cm}\leq\ell\leq257.14\ \mathrm{cm}\). The covariance-amplitude
bounds are \([10^{-3},10^3]\) in scaled target units. One optimizer restart is
used. A user can instead fix \(\ell\) manually.

This is empirical Bayes: posterior uncertainty is conditional on the optimized
hyperparameters. Uncertainty in \(\ell\), \(\sigma_f\), the prior mean, and any
model choice is not integrated out.

### Conditional posterior

At evaluation radii \(R_*\),

\[
\begin{aligned}
\boldsymbol\mu_*
&=\mathbf m_*+
K_{*X}(K_{XX}+\Sigma)^{-1}
(\mathbf y-\mathbf m_X),\\
C_*
&=K_{**}-
K_{*X}(K_{XX}+\Sigma)^{-1}K_{X*}.
\end{aligned}
\]

The displayed profile median and pointwise credible interval are estimated from
correlated posterior samples rather than using only the analytic marginal
Gaussian interval.

## Derivative and Scale-Length Propagation

The application evaluates the posterior on a uniform grid of 220 radii and
draws 400 complete posterior functions by default:

\[
f^{(s)}(R)\sim p(f\mid\mathcal D),
\qquad s=1,\ldots,400.
\]

Each sample is differentiated numerically with a second-order finite-difference
gradient on the physical \(R\) grid:

\[
g^{(s)}(R)=\frac{df^{(s)}}{dR}.
\]

Derived quantities are computed from each joint sample:

\[
L_f^{(s)}(R)=\frac{f^{(s)}(R)}{g^{(s)}(R)},
\qquad
\left(L_f^{-1}\right)^{(s)}(R)=
\frac{g^{(s)}(R)}{f^{(s)}(R)}.
\]

Pointwise medians and posterior quantiles are then taken across \(s\). This
preserves the correlation between \(f\) and \(f'\), which would be lost by
dividing independent marginal error bars.

### Weak-identification rule

The numerical small-gradient threshold is

\[
\epsilon_g=0.01\frac{s_y}{R_{\max}-R_{\min}}.
\]

A scale length is hidden at radius \(R\) if either:

1. the chosen posterior credible interval for \(g(R)\) contains zero; or
2. at least 10% of posterior samples satisfy
   \(|g^{(s)}(R)|<\epsilon_g\).

This rule is intended to show the mathematical singularity honestly. A locally
flat profile has infinite signed scale length, and samples near zero gradient
produce a heavy-tailed ratio that can change sign. The application shades such
regions as weakly identified rather than clipping the ratio to a convenient
finite range.

For the inverse normalized gradient, the analogous singularity check concerns
posterior profile samples near \(f=0\).

## Optional Edge Information and Data Filtering

The user may add a synthetic observation at the outermost radius:

\[
f(R_{\max})=f_{\mathrm{edge}}+\eta,
\qquad
\eta\sim\mathcal N(0,\sigma_{\mathrm{edge}}^2).
\]

The current conventions are

\[
\sigma_{\mathrm{edge}}=
\begin{cases}
0.20s_y, & \text{soft edge},\\
0.02s_y, & \text{strong edge}.
\end{cases}
\]

The edge value defaults to zero but is editable. Note that this synthetic point
is currently placed at the same outermost diagnostic radius as the last real
measurement, not at a separately known separatrix or wall coordinate.

An optional quality filter excludes points with

\[
\frac{\sigma_i}{y_i}>q_{\max}.
\]

It is disabled by default because the heteroscedastic likelihood should already
down-weight uncertain points. The filter exists to explore explicit validity
policies and pathological acquisition times.

## Quantitative Evidence from All Real Profiles

The default Matérn-5/2 reconstruction was run over every shot, time, and
quantity. This fast triage sweep used 80 evaluation radii and 30 posterior
samples per fit; therefore its scale-instability fractions have more Monte
Carlo variation than the live 400-sample result. Hyperparameter fitting was the
same as in the live application.

### Global results

| Diagnostic | Result |
| --- | ---: |
| Profile fits attempted | 1,486 |
| Profile fits completed numerically | 1,486 |
| Failed fits | 0 |
| Fits producing at least one optimizer convergence warning | 286 |
| Fits with correlation length at an optimization bound | 13 |
| Fits with negative posterior median somewhere in the radial domain | 590 |
| Fits with more than 90% of scale-length radii marked undefined | 170 |

The 590 negative-median cases are not restricted to obviously failed profiles.
Of the 743 density profiles, 326 have a negative median somewhere; 317 of those
have median reported relative uncertainty below 10%. Of the 743 temperature
profiles, 264 have a negative median somewhere; 248 of those have median
reported relative uncertainty below 10%. These negatives are generally local
radial regions, often near low-valued edges, but the sweep establishes that the
unconstrained linear-space GP does not preserve physical positivity.

### Density summary

| Quantity | Minimum | 5th percentile | Median | 95th percentile | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fitted \(\ell\) (cm) | 2.91 | 5.85 | 10.24 | 25.02 | 158.88 |
| Undefined scale-length fraction | 0.15 | 0.4125 | 0.8625 | 0.925 | 0.95 |
| Fraction of profile band below zero | 0 | 0 | 0.0375 | 0.09875 | 0.575 |
| Median reported relative uncertainty | 0.0404 | 0.0410 | 0.0425 | 0.0968 | 49.49 |

Density produced only two optimizer-warning profiles and no fitted length at a
bound. Nevertheless, 141 density fits had more than 90% of scale-length radii
undefined. This may reflect genuinely broad/flat density profiles, overly short
optimized length scales and derivative noise, the ratio diagnostic itself, or
some combination.

### Temperature summary

| Quantity | Minimum | 5th percentile | Median | 95th percentile | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fitted \(\ell\) (cm) | 5.55 | 8.43 | 16.51 | 76.59 | 257.14 |
| Undefined scale-length fraction | 0.1125 | 0.325 | 0.7875 | 0.90 | 1.00 |
| Fraction of profile band below zero | 0 | 0 | 0.075 | 0.15 | 1.00 |
| Median reported relative uncertainty | 0.0282 | 0.0293 | 0.0421 | 0.1503 | 86.52 |

Temperature produced 284 optimizer-warning profiles. In 282 cases the
optimized covariance amplitude reached its lower bound; 13 of those also
reached the upper length bound. A smaller subset reported abnormal L-BFGS
termination. This pattern is concentrated in weak/no-plasma profiles and may
indicate that the data are statistically consistent with the constant prior
mean, but it also raises questions about the mean model, scaling, validity
classification, and whether every stored acquisition should be fitted at all.

## Known-Function Validation

The derivative and ratio pipeline was tested on synthetic positive profiles
with known derivatives. These are single noise realizations, not a repeated-
sampling calibration study, so pointwise coverage values should not be
interpreted as frequentist coverage guarantees.

### Linear profile

For

\[
f(R)=2.5-0.08R,
\qquad
f'(R)=-0.08,
\]

using 25 observations on \([0,12]\), Gaussian noise \(\sigma=0.04\), 220 grid
points, and 400 posterior samples:

| Metric | Result |
| --- | ---: |
| Fitted correlation length | 24.0 |
| Profile RMSE | 0.01009 |
| Gradient RMSE | 0.00563 |
| Profile truth inside 95% pointwise interval | 100% of grid |
| Gradient truth inside 95% pointwise interval | 100% of grid |
| Scale-length truth inside interval on stable radii | 100% |
| Scale-length radii marked undefined | 0% |

The length reached its allowed upper bound for this nearly linear function.

### Peaked profile

For

\[
f(R)=0.12+1.25
\exp\!\left[-\left(\frac{R-6}{2.8}\right)^2\right],
\]

with 25 observations and \(\sigma=0.035\):

| Metric | Result |
| --- | ---: |
| Fitted correlation length | 4.488 |
| Profile RMSE | 0.02210 |
| Gradient RMSE | 0.02580 |
| Profile truth inside 95% pointwise interval | 92.7% of grid |
| Gradient truth inside 95% pointwise interval | 100% of grid |
| Scale-length truth inside interval on stable radii | 100% |
| Scale-length radii marked undefined | 11.8% |

The undefined region is expected near the profile maximum, where the exact
derivative is zero and signed scale length diverges.

### Flat profile

For \(f(R)=1.5\), the median absolute inferred gradient was
\(4.05\times10^{-4}\) in profile units per radial unit. The 95% derivative
interval contained zero over 100% of the grid, and 100% of the scale-length grid
was correctly marked undefined.

### Grid and posterior-sample sensitivity

Relative to the 220-grid, 400-sample peaked-profile baseline:

| Grid | Samples | Profile RMSE versus baseline | Gradient RMSE versus baseline | Undefined fraction |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 200 | 0.00203 | 0.00476 | 0.110 |
| 220 | 800 | 0.00100 | 0.00184 | 0.114 |
| 400 | 400 | 0.00168 | 0.00302 | 0.118 |

The finite-difference/sample procedure is reasonably stable at the current
resolution for this smooth test, but this does not prove calibration on sharp,
nonstationary, pedestal-like, or outlier-contaminated profiles.

## Representative Real-Data Cases

### Case A: primary temperature narrative

Shot 141687 at \(t\approx0.3150\ \mathrm{s}\):

| Quantity | Median relative uncertainty | Maximum relative uncertainty | Fitted \(\ell\) | Undefined scale-length fraction |
| --- | ---: | ---: | ---: | ---: |
| \(T_e\) | 4.94% | 24.21% | 30.98 cm | 13.75% |
| \(n_e\) | 4.47% | 14.11% | 6.29 cm | 93.75% |

The temperature rises from the inboard edge to a broad maximum near
\(R=100\)--\(110\ \mathrm{cm}\), then falls across the outboard edge. The
density is precisely measured but comparatively flat over much of the radius,
so its scale length is weakly identified. This is an important distinction:
small profile error does not imply a well-identified derivative ratio.

### Case B: both gradients comparatively well identified

Shot 138846 at \(t\approx1.0817\ \mathrm{s}\):

| Quantity | Median relative uncertainty | Maximum relative uncertainty | Fitted \(\ell\) | Undefined scale-length fraction |
| --- | ---: | ---: | ---: | ---: |
| \(T_e\) | 5.06% | 57.06% | 36.07 cm | 11.25% |
| \(n_e\) | 4.47% | 18.17% | 25.13 cm | 16.25% |

This is the strongest combined three-panel demonstration identified by the
sweep.

### Case C: mostly good channels with extreme-uncertainty outliers

Shot 141398 at \(t\approx0.3649\ \mathrm{s}\):

| Quantity | Median relative uncertainty | Maximum relative uncertainty | Fitted \(\ell\) | Undefined scale-length fraction |
| --- | ---: | ---: | ---: | ---: |
| \(T_e\) | 4.08% | 1,258% | 12.09 cm | 80.0% |
| \(n_e\) | 4.26% | 559% | 6.82 cm | 87.5% |

This case probes whether heteroscedastic Gaussian weighting is sufficient or
whether a robust likelihood, explicit invalid-channel model, or quality
classification is warranted.

### Known unusable/no-plasma case

Shot 140617 at \(t\approx1.0483\ \mathrm{s}\) has median relative uncertainty
of approximately 8,652% for temperature and 4,949% for density. The temperature
length reaches the upper bound and its scale length is undefined everywhere.
A numerical GP result exists, but it should not be presented as a scientifically
usable plasma reconstruction.

## Scientific Questions That Must Be Resolved

### 1. Is a linear-space Gaussian process adequate?

Both \(T_e\) and \(n_e\) are nonnegative physical quantities, yet a Gaussian
process over \(f\) assigns nonzero probability to negative values and produces
negative posterior medians in local regions of 590 real profiles.

Investigate and compare at least:

- the current linear-space heteroscedastic GP;
- a GP for \(z(R)=\log f(R)\), including the correct treatment or approximation
  of reported additive Gaussian uncertainty in \(f\);
- warped or transformed GPs that enforce positivity;
- likelihoods defined directly on positive data, such as lognormal, Gamma, or
  other justified choices;
- constrained-GP methods, if mature and practical enough;
- a positive parametric or semiparametric plasma-profile mean combined with a
  GP residual.

Do not recommend log-space fitting merely because it enforces positivity.
Analyze the likelihood mismatch when the diagnostic supplies approximately
Gaussian errors in linear units, especially for large relative uncertainty.
If a delta-method approximation

\[
\sigma_{\log f}\approx\frac{\sigma_f}{f}
\]

is proposed, state when it is valid and what to do when \(\sigma_f/f\) is not
small.

### 2. Is the stationary Matérn-5/2 prior suitable for whole-chord profiles?

The diagnostic covers low-gradient core regions and potentially sharp edge or
pedestal structure. A single stationary \(\ell\) may over-smooth steep regions
or under-smooth flat ones.

Investigate:

- stationary Matérn kernels with \(\nu=3/2\), \(5/2\), or other regularity;
- spatially varying or nonstationary kernels, including Gibbs-type kernels;
- change-point or additive core/edge covariance structures;
- rational-quadratic scale mixtures;
- physically motivated mean functions with GP residuals;
- spline-based Bayesian alternatives where they have stronger calibration or
  interpretability for plasma profiles.

Relate kernel differentiability to the existence and regularity of \(f'(R)\).
Clarify whether Matérn-5/2 is a defensible default when derivative inference is
central.

### 3. Should reported uncertainties be trusted as the complete noise model?

The current likelihood assumes independent Gaussian errors with known
variances. Investigate evidence for:

- correlated errors across spatial channels;
- uncertainty miscalibration or diagnostic systematics;
- outliers and invalid channels;
- Student-\(t\), contamination-mixture, or explicit outlier likelihoods;
- an additional inferred model-discrepancy/noise term;
- whether large reported \(\sigma_i\) should merely down-weight a channel or
  also trigger a validity decision.

Separate what is known generally about processed Thomson-scattering errors from
what cannot be concluded without device-specific calibration metadata.

### 4. Empirical Bayes versus hyperparameter uncertainty

Assess whether optimizing \(\ell\) and \(\sigma_f\) by marginal likelihood
materially understates uncertainty in \(f'\) and \(L_f\) when there are only 30
spatial points. Compare:

- plug-in empirical Bayes;
- Laplace or approximate integration over hyperparameters;
- full Bayesian sampling;
- finite mixtures/model averaging over kernel and length choices;
- pragmatic sensitivity bands suitable for an interactive hackathon tool.

The preferred recommendation should distinguish the scientifically ideal model
from a minimal improvement that remains fast and robust on a laptop.

### 5. Analytic derivative GP versus sampled finite differences

For differentiable kernels, derivative posterior moments can be obtained from
kernel derivatives. Determine whether analytic derivative conditioning should
replace the finite-difference derivative of sampled profiles, or whether the
current method is adequate at 220 grid points.

Address:

- bias and variance from finite differences;
- exact cross-covariance between \(f\) and \(f'\);
- posterior sampling jointly from \((f,f')\);
- numerical conditioning;
- the special treatment required by nonstationary or transformed GPs;
- how to propagate uncertainty into \(f/f'\) and \(f'/f\).

### 6. How should gradient scale length be reported?

The ratio \(L_f=f/f'\) has a singular, often heavy-tailed posterior near
\(f'=0\). Analyze whether posterior median plus pointwise quantiles is a
meaningful summary, and whether the current mask rule has a statistical basis.

Compare reporting:

\[
L_f=\frac{f}{f'},
\qquad
\frac{1}{L_f}=\frac{f'}{f},
\qquad
-\frac{d\log f}{dR},
\]

including common plasma sign conventions. Recommend a primary displayed
quantity and a principled definition of weak identification. Consider whether
the app should report a posterior probability such as

\[
P(f'(R)>0\mid\mathcal D)
\]

or a bounded diagnostic instead of a hard threshold based on an arbitrary
\(\epsilon_g\).

### 7. What boundary and coordinate assumptions are defensible?

Assess:

- placing a synthetic value at the outermost diagnostic channel versus at a
  known separatrix;
- using a physical edge-value prior without equilibrium mapping;
- imposing axis symmetry \(f'(R_{\mathrm{axis}})=0\) when the magnetic-axis
  location is known;
- fitting the full inboard-to-outboard major-radius chord versus mapping both
  sides to a common flux coordinate;
- uncertainty propagation through an equilibrium-derived coordinate mapping;
- whether a whole-chord \(R\) derivative should be labeled a profile gradient
  scale length without further qualification.

Do not assume equilibrium data are available for the present hackathon
prototype. State which scientific claims are safe in \(R\) and which require a
flux-coordinate mapping.

### 8. Is independent per-time, per-quantity fitting sufficient?

The current tool fits \(T_e\) and \(n_e\) independently and treats each time
slice separately. Assess the value and cost of:

- multi-output covariance between \(T_e\) and \(n_e\);
- spatiotemporal GPs across adjacent acquisition times;
- hierarchical priors that share length-scale information across time;
- whether those extensions are justified by the diagnostic physics or likely
  to impose false correlations.

The answer should not over-engineer the first deliverable. State clearly whether
these are required, optional, or actively inadvisable for the current scope.

## Candidate Mathematical Formulations to Evaluate

The following formulations are hypotheses for comparison, not predetermined
answers. Analyze their likelihood fidelity, derivative semantics,
identifiability, and computational burden. Modify or reject them where the
literature supports a better formulation.

### A. Present conjugate linear-space GP

\[
\begin{aligned}
y_i\mid f &\sim \mathcal N(f(R_i),\sigma_i^2),\\
f &\sim \mathcal{GP}(m,k_\phi).
\end{aligned}
\]

Advantages include exact conditioning, direct use of additive reported errors,
fast marginal-likelihood fitting, and a jointly Gaussian derivative process.
Its central defect is lack of positivity. It may also need a discrepancy term
if \(\sigma_i\) omits calibration or model uncertainty.

### B. Latent log-profile with the original linear-unit likelihood

One positivity-preserving option is

\[
\begin{aligned}
z &\sim \mathcal{GP}(m_z,k_z),\\
f(R)&=\exp z(R),\\
y_i\mid z &\sim
\mathcal N\!\left(\exp z(R_i),\sigma_i^2\right).
\end{aligned}
\]

This retains the supplied additive-error interpretation while making the
posterior non-Gaussian. It requires Laplace inference, variational inference,
Hamiltonian Monte Carlo, elliptical-slice methods, or another approximation.
Its derivative has the useful identity

\[
f'(R)=f(R)z'(R),
\qquad
\frac{f'(R)}{f(R)}=z'(R),
\qquad
L_f(R)=\frac{1}{z'(R)}.
\]

Thus the inverse normalized gradient is directly the derivative of the latent
log-profile. Determine whether this mathematical simplification and positivity
outweigh the nonconjugate inference cost and any bias near the diagnostic noise
floor.

### C. Approximate Gaussian regression on logged measurements

For small relative errors, one might use

\[
\log y_i\mid z(R_i)
\approx \mathcal N\!\left(
z(R_i),\frac{\sigma_i^2}{y_i^2}
\right).
\]

A second-order delta-method correction to the transformed mean may be relevant.
The approximation is suspect when \(\sigma_i/y_i\) is not small and becomes
unusable for nonpositive noise realizations. Quantify a defensible relative-
uncertainty regime, if one exists, and determine whether high-uncertainty points
should instead be handled with the exact nonlinear likelihood or excluded by a
diagnostic validity policy.

### D. Robust additive likelihood

An outlier-resistant alternative retains linear physical units but replaces the
Gaussian residual with, for example,

\[
y_i\mid f,\lambda_i
\sim \mathcal N\!\left(f(R_i),\frac{\sigma_i^2+\sigma_{\rm extra}^2}{\lambda_i}\right),
\qquad
\lambda_i\sim\operatorname{Gamma}\!\left(\frac{\nu}{2},\frac{\nu}{2}\right),
\]

which marginalizes to a Student-\(t\)-type likelihood. Another candidate is a
contamination mixture,

\[
p(y_i\mid f)=
(1-\pi)\,\mathcal N(f(R_i),\sigma_i^2)
+\pi\,\mathcal N(f(R_i),c^2\sigma_i^2),
\qquad c\gg1.
\]

Determine whether such models are justified when the data already supply very
large \(\sigma_i\) for dubious channels. A robust likelihood should not erase
the information encoded in the diagnostic uncertainty or silently reinterpret
valid edge measurements as outliers.

### E. Physically structured mean plus residual GP

A semiparametric model could use

\[
f(R)=h(R;\theta)+u(R),
\qquad
u\sim\mathcal{GP}(0,k_\phi),
\]

where \(h\) is a positive core-and-edge profile and \(u\) captures residual
structure. Candidate \(h\) functions could involve modified tanh, spline, or
other established plasma-profile parameterizations. Assess whether such a mean
improves extrapolation and derivative stability or instead injects unjustified
shape assumptions for profiles spanning both inboard and outboard radii.

If a positive link is required, compare additive residuals with

\[
f(R)=\exp\!\big(h_z(R;\theta)+u(R)\big).
\]

### F. Nonstationary covariance

A one-dimensional Gibbs covariance with spatially varying length
\(\ell(R)>0\) has a form such as

\[
k(R,R')=
\sigma_f^2
\sqrt{\frac{2\ell(R)\ell(R')}{\ell(R)^2+\ell(R')^2}}
\exp\!\left[
-\frac{(R-R')^2}{\ell(R)^2+\ell(R')^2}
\right].
\]

Possible parameterizations include a smooth core-to-edge transition in
\(\ell(R)\) or a latent GP for \(\log\ell(R)\). A change-point construction
could blend kernels:

\[
k(R,R')=
w(R)w(R')k_{\rm core}(R,R')+
[1-w(R)][1-w(R')]k_{\rm edge}(R,R'),
\]

with a sigmoid \(w\). Determine whether 30 observations are enough to identify
the additional structure without strong priors or equilibrium-based region
labels. Prefer a stationary model if the nonstationary alternative is not
identifiable.

### G. Hyperparameter integration or model averaging

Instead of a plug-in value \(\hat\phi\), the posterior of interest is

\[
p(f,f'\mid\mathcal D)=
\int p(f,f'\mid\mathcal D,\phi)
p(\phi\mid\mathcal D)\,d\phi.
\]

Possible approximations include a small quadrature or posterior sample set over
\(\ell\) and amplitude, a Laplace approximation around the marginal-likelihood
optimum, or discrete Bayesian model averaging across a short kernel menu. The
research should state whether conditional bands are likely to be materially too
narrow for derivatives, and identify the smallest practical integration scheme
that improves honesty.

### H. Exact derivative-GP construction

For a differentiable kernel, the joint prior is

\[
\begin{bmatrix}
\mathbf f\\
\mathbf f'
\end{bmatrix}
\sim \mathcal N\!\left(
\begin{bmatrix}
\mathbf m\\
\mathbf m'
\end{bmatrix},
\begin{bmatrix}
K_{ff} & K_{ff'}\\
K_{f'f} & K_{f'f'}
\end{bmatrix}
\right),
\]

where

\[
\begin{aligned}
[K_{ff'}]_{ij}
&=\frac{\partial}{\partial R_j'}k(R_i,R_j'),\\
[K_{f'f'}]_{ij}
&=\frac{\partial^2}{\partial R_i\partial R_j'}k(R_i,R_j').
\end{aligned}
\]

Conditioning this joint Gaussian on \(\mathbf y\) gives analytic derivative
means, variances, and profile-derivative cross-covariances. Joint samples from
\((f,f')\) can then be transformed into \(f/f'\). Compare this with numerical
differentiation of function samples in accuracy, conditioning, and ease of
extension to transformed or non-Gaussian models.

## Interpretation and Calibration Questions

The final recommendation must keep distinct three notions that are often
blurred in profile-fitting discussions:

1. **Conditional posterior uncertainty:** uncertainty in \(f\) given one
   kernel, one optimized hyperparameter vector, the reported \(\sigma_i\), and
   the assumed coordinate.
2. **Model and hyperparameter uncertainty:** uncertainty arising from kernel,
   likelihood, prior-mean, validity-policy, and hyperparameter choices.
3. **Diagnostic and equilibrium uncertainty:** calibration systematics,
   channel correlations, mapping from measurement location to flux coordinate,
   and uncertainty in the magnetic equilibrium.

The present bands represent only the first category. Explain which additional
categories the preferred hackathon model can realistically include and which
must be disclosed as limitations.

Also distinguish:

- a pointwise \(95\%\) credible interval at each radius;
- a simultaneous credible band for an entire function;
- repeated-sampling calibration or empirical coverage.

If simultaneous bands are worth exposing, give a practical construction from
posterior suprema or another justified method. If they would confuse the
hackathon scope, recommend precise language for retaining pointwise intervals.

The current validation reports the fraction of grid points whose pointwise
interval contains one synthetic truth for one noise realization. Design a
repeated-simulation study that can actually estimate calibration. It should
vary at least profile shape, correlation length, signal-to-noise ratio,
outliers, edge coverage, and diagnostic validity. State how many repetitions
and which summary metrics are sufficient for a practical decision.

## Literature and Citation Requirements

Search for and verify relevant primary sources. The review should include
claim-level citations, not a bibliography detached from the argument.

Prioritize:

1. peer-reviewed work on Gaussian-process or Bayesian fitting of tokamak plasma
   profiles and profile derivatives;
2. primary literature or authoritative reports on Thomson-scattering profile
   uncertainties and profile reconstruction;
3. foundational sources for derivative Gaussian processes, constrained or
   positive GPs, nonstationary kernels, robust GP likelihoods, and
   hyperparameter uncertainty;
4. official technical documentation only where needed to verify the exact
   behavior of a software method.

Use approximately 12--20 strong, directly relevant sources if the literature
supports that number. Quality and relevance are more important than padding.
For each central source, provide a DOI, arXiv identifier, or stable publisher or
institutional URL. Distinguish peer-reviewed results from theses, conference
papers, documentation, and general textbooks.

Potential leads that must be bibliographically verified rather than blindly
accepted include literature described as “improved profile fitting and
quantification of uncertainty in experimental plasma profiles,” Gaussian-
process profile analysis in magnetic-confinement experiments, and derivative
GP methodology. Find the actual publications, authors, dates, and results.

Do not cite generic machine-learning web articles for central scientific
claims. Do not invent bibliographic details. If a desired claim lacks a strong
source, state that limitation explicitly.

## Decision Criteria

Judge candidate models using the following criteria:

| Criterion | Question |
| --- | --- |
| Physical support | Does the model respect positivity and plausible profile behavior without hiding model mismatch? |
| Likelihood fidelity | Does it use the supplied measurement uncertainties consistently? |
| Derivative quality | Are \(f'\), its uncertainty, and correlation with \(f\) well defined and stable? |
| Ratio behavior | Does it handle \(f/f'\) honestly near zero derivative? |
| Nonstationarity | Can it represent broad cores and sharp edges without pathological smoothing? |
| Calibration | Can posterior uncertainty be validated and interpreted? |
| Robustness | Does it handle outliers, weak/no-plasma acquisitions, and optimizer boundaries? |
| Data sufficiency | Is the complexity supportable with only 30 spatial points per slice? |
| Runtime | Is interactive performance feasible on a laptop? |
| Explainability | Can a hackathon audience understand the model and controls? |
| Implementation risk | Can the recommendation be implemented and tested reliably within a short project? |

Do not declare a model superior based on sophistication alone. Penalize methods
whose additional flexibility is not identifiable from 30 points or whose
uncertainty semantics do not match the measurement process.

## Required Output Format

Return a finished research review using these sections:

1. **Verdict**
   State whether the current model is scientifically defensible, conditionally
   defensible, or unsuitable. Name the single preferred model or model family.

2. **Most Important Findings**
   Give a ranked list of the conclusions that should change the project.

3. **Assessment of the Current Mathematics**
   Evaluate the likelihood, mean, kernel, empirical-Bayes treatment,
   derivative sampling, ratio propagation, edge constraint, and coordinate.
   Explicitly distinguish sound choices from weak or arbitrary ones.

4. **Citation-Supported Model Comparison**
   Compare the current model with the serious alternatives in a table using
   the decision criteria above. Attach citations to the relevant claims.

5. **Preferred Generative Model**
   Write the complete recommended model in LaTeX: observation likelihood,
   latent function or transform, mean, covariance, hyperpriors or optimization
   treatment, boundary information, derivative posterior, and derived-gradient
   quantities. State all assumptions.

6. **Minimum Defensible Hackathon Model**
   Give the smallest set of changes that should be implemented now. Explain
   which advanced improvements can wait and why.

7. **Validation and Falsification Protocol**
   Specify concrete synthetic tests, posterior predictive checks,
   cross-validation or calibration diagnostics, profile-validity rules, and
   quantitative acceptance thresholds. Improve on the single-realization tests
   described above.

8. **Coordinate, Boundary, and Plasma-Physics Interpretation**
   State exactly what may be claimed using major radius \(R\), what requires
   equilibrium mapping, and how scale-length sign conventions should be shown.

9. **Recommended User-Facing Scientific Language**
   Provide short wording for the GUI describing credible intervals, empirical
   Bayes or full Bayes, undefined scale lengths, and the limits of \(R\)-space
   gradients.

10. **References**
    Give a verified bibliography with stable links and identifiers. Ensure that
    every important claim in the report points to the source supporting it.

## Constraints and Non-Negotiables

- Keep the review mathematical and scientific. Do not spend space reviewing
  source-code style, module structure, naming, or user-interface implementation.
- Do not silently reinterpret reported standard uncertainties as something
  else. If the likelihood needs different uncertainty semantics, derive the
  conversion and state its domain of validity.
- Do not claim that profile positivity alone proves a lognormal likelihood.
- Do not treat optimizer convergence as proof of physical validity.
- Do not divide independent marginal error bars to estimate uncertainty in
  \(f/f'\); preserve posterior dependence.
- Do not clip divergent scale lengths for visual convenience.
- Do not call pointwise Bayesian credible intervals simultaneous bands unless a
  simultaneous-band construction is actually used.
- Distinguish uncertainty conditional on optimized hyperparameters from full
  posterior uncertainty.
- Distinguish geometric major-radius gradients from flux-coordinate gradients.
- Do not require a coupled \((T_e,n_e)\) model or a spatiotemporal model unless
  the literature and identifiability arguments establish a concrete benefit.
- The preferred recommendation must remain computationally realistic for an
  interactive application with 30 observations per profile.

## Acceptance Criteria

The response is successful only if it:

- gives a direct, evidence-backed verdict on the present empirical-Bayes
  Matérn-5/2 GP;
- searches and cites verified primary or authoritative sources;
- identifies whether a demonstrably better model exists for these data;
- selects a preferred model instead of ending with an unranked menu;
- derives that model sufficiently precisely for implementation;
- treats positivity, heteroscedastic errors, nonstationarity, outliers,
  hyperparameter uncertainty, derivative inference, and ratio singularities;
- explains the major-radius versus flux-coordinate limitation;
- uses the supplied 1,486-fit and synthetic evidence rather than ignoring it;
- separates minimum hackathon changes from ideal future extensions;
- provides a quantitative validation protocol;
- does not substitute a research plan for completed findings.

## Exact Ask to GPT-Pro

Act as a senior plasma-diagnostic physicist and Bayesian statistician. Execute a
math-first scientific review of the model specified above. Search and verify
the relevant literature now, analyze the evidence, determine whether the
current method is defensible, and select a better model if the evidence supports
one. Deliver the completed citation-supported review in the required format.

Do not ask to inspect the repository. Do not return only possible avenues for
future investigation. Make the scientific decision, show the mathematics, cite
the evidence, and state the minimum defensible model that should be used for
this hackathon application.
