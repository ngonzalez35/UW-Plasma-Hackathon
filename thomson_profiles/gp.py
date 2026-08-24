"""Gaussian-process profile inference, diagnostics, and baseline model."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RBF,
    RationalQuadratic,
)


@dataclass(frozen=True)
class FitConfig:
    """Controls for one independent temperature or density reconstruction."""

    model: str = "Positive latent-log GP"
    kernel: str = "Matern 5/2"
    mean_prior: str = "Constant (weighted mean)"
    optimize_length_scale: bool = True
    length_scale_cm: float = 15.0
    confidence: float = 0.95
    n_samples: int = 400
    grid_size: int = 220
    edge_mode: str = "None"
    edge_radius_cm: float | None = None
    edge_value: float = 0.0
    edge_uncertainty: float | None = None
    max_relative_uncertainty: float | None = None
    gradient_epsilon_fraction: float = 0.01
    weak_sample_fraction: float = 0.10
    random_seed: int = 2026
    reference_scale: float = 1.0
    hyperparameter_samples: int = 24
    length_quadrature_nodes: int = 9
    max_laplace_modes: int = 3


@dataclass(frozen=True)
class ProfileValidity:
    """Provisional evidence check for whether a profile can support inference."""

    status: str
    median_relative_uncertainty: float
    resolved_points: int
    total_points: int
    message: str


@dataclass(frozen=True)
class PosteriorSummary:
    """Posterior curves, samples, diagnostics, and the observations used."""

    radius_cm: np.ndarray
    profile_mean: np.ndarray
    profile_median: np.ndarray
    profile_low: np.ndarray
    profile_high: np.ndarray
    gradient_median: np.ndarray
    gradient_low: np.ndarray
    gradient_high: np.ndarray
    scale_length_median: np.ndarray
    scale_length_low: np.ndarray
    scale_length_high: np.ndarray
    inverse_scale_median: np.ndarray
    inverse_scale_low: np.ndarray
    inverse_scale_high: np.ndarray
    scale_unstable: np.ndarray
    inverse_scale_unstable: np.ndarray
    derivative_boundary_limited: np.ndarray
    near_zero_gradient_fraction: np.ndarray
    observed_radius_cm: np.ndarray
    observed_values: np.ndarray
    observed_errors: np.ndarray
    fitted_length_scale_cm: float
    kernel_description: str
    log_marginal_likelihood: float
    confidence: float
    edge_observation: tuple[float, float, float] | None
    model_name: str
    gradient_positive_probability: np.ndarray
    fit_score_label: str
    fit_score: float
    hyperparameter_modes: int

    @property
    def n_observations(self) -> int:
        return len(self.observed_radius_cm)


def _kernel(
    name: str,
    initial_length_scale: float,
    length_bounds: tuple[float, float] | str,
):
    amplitude = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
    if name == "Matern 5/2":
        base = Matern(
            length_scale=initial_length_scale,
            length_scale_bounds=length_bounds,
            nu=2.5,
        )
    elif name == "RBF":
        base = RBF(
            length_scale=initial_length_scale,
            length_scale_bounds=length_bounds,
        )
    elif name == "Rational quadratic":
        base = RationalQuadratic(
            length_scale=initial_length_scale,
            alpha=1.0,
            length_scale_bounds=length_bounds,
            alpha_bounds=(1e-2, 1e2),
        )
    else:
        raise ValueError(f"Unsupported kernel {name!r}.")
    return amplitude * base


def _credible_interval(samples: np.ndarray, confidence: float) -> tuple[np.ndarray, ...]:
    tail = 50.0 * (1.0 - confidence)
    return tuple(np.percentile(samples, [tail, 50.0, 100.0 - tail], axis=1))


def _nan_credible_interval(
    samples: np.ndarray, confidence: float
) -> tuple[np.ndarray, ...]:
    tail = 50.0 * (1.0 - confidence)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return tuple(
            np.nanpercentile(samples, [tail, 50.0, 100.0 - tail], axis=1)
        )


def _prior_center(y: np.ndarray, sigma: np.ndarray, prior: str) -> float:
    if prior == "Zero":
        return 0.0
    if prior == "Constant (weighted mean)":
        weights = 1.0 / np.square(sigma)
        return float(np.sum(weights * y) / np.sum(weights))
    raise ValueError(f"Unsupported mean prior {prior!r}.")


def _extract_length_scale(fitted_kernel) -> float:
    base = fitted_kernel.k2
    return float(np.asarray(base.length_scale).reshape(-1)[0])


def assess_profile_validity(values: np.ndarray, errors: np.ndarray) -> ProfileValidity:
    """Apply a transparent provisional signal-evidence gate.

    The thresholds are intentionally labeled provisional because the bundled
    file contains no upstream MPTS status bits or independently curated validity
    labels. They separate obvious absent-signal acquisitions from profiles that
    should proceed to posterior inference; they are not a replacement for
    diagnostic quality metadata.
    """

    values = np.asarray(values, dtype=float).reshape(-1)
    errors = np.asarray(errors, dtype=float).reshape(-1)
    usable = np.isfinite(values) & np.isfinite(errors) & (values > 0.0) & (errors > 0.0)
    if np.sum(usable) < 4:
        return ProfileValidity(
            status="Unusable",
            median_relative_uncertainty=float("inf"),
            resolved_points=0,
            total_points=len(values),
            message="Fewer than four finite positive measurements are available.",
        )

    relative = errors[usable] / values[usable]
    median_relative = float(np.median(relative))
    resolved = int(np.sum(values[usable] / errors[usable] > 2.0))
    total = int(np.sum(usable))
    if median_relative > 1.0 or resolved < 6:
        status = "Unusable"
        message = (
            "The acquisition has insufficient signal evidence for a scientifically "
            "usable reconstruction under the provisional validity rule."
        )
    elif median_relative > 0.5 or resolved < 10:
        status = "Caution"
        message = (
            "The acquisition is weakly resolved; interpret posterior structure and "
            "gradients cautiously."
        )
    else:
        status = "Usable"
        message = "The acquisition passes the provisional signal-evidence check."
    return ProfileValidity(
        status=status,
        median_relative_uncertainty=median_relative,
        resolved_points=resolved,
        total_points=total,
        message=message,
    )


def _fit_linear_profile(
    radius_cm: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    config: FitConfig = FitConfig(),
) -> PosteriorSummary:
    """Fit a heteroscedastic GP and propagate samples through radial derivatives.

    The target and its reported standard uncertainty are scaled together before
    fitting. This is deliberate: scikit-learn's ``alpha`` is a variance in the
    same units as the target supplied to ``fit``.
    """

    radius_cm = np.asarray(radius_cm, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    errors = np.asarray(errors, dtype=float).reshape(-1)
    if not (radius_cm.shape == values.shape == errors.shape):
        raise ValueError("radius, values, and errors must have identical shapes.")
    if not 0.0 < config.confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one.")
    if config.n_samples < 20:
        raise ValueError("At least 20 posterior samples are required.")
    if config.grid_size < 25:
        raise ValueError("grid_size must be at least 25.")

    finite_radius = radius_cm[np.isfinite(radius_cm)]
    if len(finite_radius) < 2:
        raise ValueError("At least two finite radial positions are required.")
    x_min, measured_x_max = float(np.min(finite_radius)), float(np.max(finite_radius))
    if measured_x_max - x_min <= 0.0:
        raise ValueError("At least two distinct radial positions are required.")

    mask = (
        np.isfinite(radius_cm)
        & np.isfinite(values)
        & np.isfinite(errors)
        & (values > 0.0)
        & (errors > 0.0)
    )
    if config.max_relative_uncertainty is not None:
        mask &= errors / values <= config.max_relative_uncertainty

    x = radius_cm[mask]
    y = values[mask]
    sigma = errors[mask]
    if len(x) < 4:
        raise ValueError("Fewer than four valid measurements remain after filtering.")

    order = np.argsort(x)
    x, y, sigma = x[order], y[order], sigma[order]
    observed_x = x.copy()
    observed_y = y.copy()
    observed_sigma = sigma.copy()

    prior_center = _prior_center(y, sigma, config.mean_prior)
    target_scale = max(float(np.std(y)), float(np.median(np.abs(y))), 1e-12)

    edge_observation = None
    if config.edge_mode != "None":
        if config.edge_mode != "User-specified physical point":
            raise ValueError(f"Unsupported edge mode {config.edge_mode!r}.")
        if config.edge_radius_cm is None or config.edge_radius_cm <= measured_x_max:
            raise ValueError("The edge coordinate must lie beyond the outermost measurement.")
        if config.edge_uncertainty is None or config.edge_uncertainty <= 0.0:
            raise ValueError("The edge uncertainty must be positive.")
        edge_sigma = float(config.edge_uncertainty)
        x = np.append(x, config.edge_radius_cm)
        y = np.append(y, config.edge_value)
        sigma = np.append(sigma, edge_sigma)
        edge_observation = (
            float(config.edge_radius_cm),
            float(config.edge_value),
            edge_sigma,
        )

    x_max = float(config.edge_radius_cm) if edge_observation is not None else measured_x_max
    x_span = x_max - x_min

    x_scaled = ((x - x_min) / x_span).reshape(-1, 1)
    y_scaled = (y - prior_center) / target_scale
    sigma_scaled = sigma / target_scale

    initial_length = np.clip(config.length_scale_cm / x_span, 0.005, 2.0)
    if config.optimize_length_scale:
        length_bounds: tuple[float, float] | str = (0.005, 2.0)
    else:
        length_bounds = "fixed"

    gp = GaussianProcessRegressor(
        kernel=_kernel(config.kernel, initial_length, length_bounds),
        alpha=np.square(sigma_scaled) + 1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=1 if config.optimize_length_scale else 0,
        normalize_y=False,
        random_state=config.random_seed,
    )
    gp.fit(x_scaled, y_scaled)

    grid = np.linspace(x_min, x_max, config.grid_size)
    grid_scaled = ((grid - x_min) / x_span).reshape(-1, 1)
    mean_scaled = gp.predict(grid_scaled)
    profile_mean = prior_center + target_scale * mean_scaled
    samples_scaled = gp.sample_y(
        grid_scaled,
        n_samples=config.n_samples,
        random_state=config.random_seed,
    )
    profile_samples = prior_center + target_scale * samples_scaled

    profile_low, profile_median, profile_high = _credible_interval(
        profile_samples, config.confidence
    )
    gradient_samples = np.gradient(profile_samples, grid, axis=0, edge_order=2)
    gradient_low, gradient_median, gradient_high = _credible_interval(
        gradient_samples, config.confidence
    )

    gradient_epsilon = config.gradient_epsilon_fraction * target_scale / x_span
    near_zero_gradient = np.abs(gradient_samples) < gradient_epsilon
    near_zero_gradient_fraction = np.mean(near_zero_gradient, axis=1)
    gradient_crosses_zero = (gradient_low <= 0.0) & (gradient_high >= 0.0)
    scale_unstable = gradient_crosses_zero | (
        near_zero_gradient_fraction >= config.weak_sample_fraction
    )

    safe_gradient = np.where(near_zero_gradient, np.nan, gradient_samples)
    scale_samples = profile_samples / safe_gradient
    scale_low, scale_median, scale_high = _nan_credible_interval(
        scale_samples, config.confidence
    )
    scale_low[scale_unstable] = np.nan
    scale_median[scale_unstable] = np.nan
    scale_high[scale_unstable] = np.nan

    profile_epsilon = 1e-8 * max(float(np.nanmax(np.abs(profile_samples))), target_scale)
    near_zero_profile = np.abs(profile_samples) < profile_epsilon
    safe_profile = np.where(near_zero_profile, np.nan, profile_samples)
    inverse_scale_samples = gradient_samples / safe_profile
    inverse_low, inverse_median, inverse_high = _nan_credible_interval(
        inverse_scale_samples, config.confidence
    )
    inverse_unstable = (
        (profile_low <= 0.0) & (profile_high >= 0.0)
    ) | (np.mean(near_zero_profile, axis=1) >= config.weak_sample_fraction)
    inverse_low[inverse_unstable] = np.nan
    inverse_median[inverse_unstable] = np.nan
    inverse_high[inverse_unstable] = np.nan

    typical_spacing = float(np.median(np.diff(np.unique(observed_x))))
    boundary_margin = 2.0 * typical_spacing
    boundary_limited = (grid < observed_x[0] + boundary_margin) | (
        grid > observed_x[-1] - boundary_margin
    )
    scale_unstable |= boundary_limited
    scale_low[boundary_limited] = np.nan
    scale_median[boundary_limited] = np.nan
    scale_high[boundary_limited] = np.nan

    return PosteriorSummary(
        radius_cm=grid,
        profile_mean=profile_mean,
        profile_median=profile_median,
        profile_low=profile_low,
        profile_high=profile_high,
        gradient_median=gradient_median,
        gradient_low=gradient_low,
        gradient_high=gradient_high,
        scale_length_median=scale_median,
        scale_length_low=scale_low,
        scale_length_high=scale_high,
        inverse_scale_median=inverse_median,
        inverse_scale_low=inverse_low,
        inverse_scale_high=inverse_high,
        scale_unstable=scale_unstable,
        inverse_scale_unstable=inverse_unstable,
        derivative_boundary_limited=boundary_limited,
        near_zero_gradient_fraction=near_zero_gradient_fraction,
        observed_radius_cm=observed_x,
        observed_values=observed_y,
        observed_errors=observed_sigma,
        fitted_length_scale_cm=_extract_length_scale(gp.kernel_) * x_span,
        kernel_description=str(gp.kernel_),
        log_marginal_likelihood=float(gp.log_marginal_likelihood_value_),
        confidence=config.confidence,
        edge_observation=edge_observation,
        model_name="Linear-space GP (exploratory baseline)",
        gradient_positive_probability=np.mean(gradient_samples > 0.0, axis=1),
        fit_score_label="Log marginal likelihood",
        fit_score=float(gp.log_marginal_likelihood_value_),
        hyperparameter_modes=1,
    )


def fit_profile(
    radius_cm: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    config: FitConfig = FitConfig(),
) -> PosteriorSummary:
    """Dispatch to the selected scientifically labeled reconstruction model."""

    if config.model == "Positive latent-log GP":
        from .positive_gp import fit_positive_log_profile

        return fit_positive_log_profile(radius_cm, values, errors, config)
    if config.model == "Linear GP baseline":
        return _fit_linear_profile(radius_cm, values, errors, config)
    raise ValueError(f"Unsupported inference model {config.model!r}.")
