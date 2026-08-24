# Demonstration cases

These cases were selected with the complete 1,486-profile positive latent-log GP
robustness sweep and then rerun with the live app defaults: nine length-scale
quadrature nodes, 24 hyperparameter-mixture draws, 400 posterior samples, and a
95% pointwise credible level. “Scale withheld” includes sign-unresolved radii and
the deliberately boundary-limited endpoint regions.

## 1. Primary physics narrative — shot 141687 at 0.3150 s

This is the default app selection and the clearest temperature-profile story.
The temperature rises from the inboard edge to a broad core maximum near
`R = 100–110 cm` and falls steeply across the outboard edge.

| Quantity | Median relative uncertainty | Maximum relative uncertainty | Fitted length scale | Undefined scale-length radius |
| --- | ---: | ---: | ---: | ---: |
| `T_e` | 4.94% | 24.21% | 11.00 cm | 74.00% |
| `n_e` | 4.47% | 14.11% | 3.73 cm | 98.00% |

Talking point: a profile can be measured precisely while its normalized
gradient remains weakly identified. The broad density plateau makes
`n_e/(dn_e/dR)` singular over much of the radius; the app correctly leaves it
undefined instead of inventing a finite scale length.

## 2. Clean high-signal late-time profile — shot 141324 at 1.0316 s

Every channel has a moderate reported uncertainty, without the enormous
relative-error edge points present in many acquisitions. The temperature
gradient sign is identified over most of the measured interior; the density
profile demonstrates that precise point measurements do not guarantee a precise
normalized gradient.

| Quantity | Median relative uncertainty | Maximum relative uncertainty | Fitted length scale | Undefined scale-length radius |
| --- | ---: | ---: | ---: | ---: |
| `T_e` | 6.96% | 29.52% | 26.66 cm | 18.00% |
| `n_e` | 5.00% | 12.33% | 41.51 cm | 68.00% |

Talking point: use this case to contrast measurement precision with derivative
identifiability. The temperature scale length has substantial identified support,
while the density result remains appropriately conservative.

## 3. Noisy-channel and filtering demonstration — shot 141398 at 0.3649 s

Most channels are precise, but individual edge channels have very large
relative uncertainties. This is a useful demonstration of heteroscedastic
weighting and the optional data-quality control.

| Quantity | Median relative uncertainty | Maximum relative uncertainty | Fitted length scale | Undefined scale-length radius |
| --- | ---: | ---: | ---: | ---: |
| `T_e` | 4.08% | 1,258.34% | 7.07 cm | 93.00% |
| `n_e` | 4.26% | 558.93% | 4.54 cm | 97.00% |

Talking point: first show the default fit, then enable the maximum-relative-
uncertainty filter. The reported uncertainty already down-weights bad points;
filtering demonstrates how an explicit validity policy changes the posterior.

## Known anti-demo / guardrail case

Shot 140617 at 1.0483 s is effectively an absent or unusable plasma profile:
median relative uncertainties are 8,652% for `T_e` and 4,949% for `n_e`.
Both scale lengths are undefined over the entire radius. The provisional
validity gate classifies both quantities as `Unusable` and withholds the normal
reconstruction unless a diagnostic override is enabled. Use this to demonstrate
that numerical convergence is not evidence for a scientifically usable profile.
