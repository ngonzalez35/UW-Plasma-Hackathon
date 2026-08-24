"""Positive latent-log GP with additive Gaussian measurements in physical units."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import OptimizeResult, minimize
from scipy.special import logsumexp

from .gp import (
    FitConfig,
    PosteriorSummary,
    _credible_interval,
    _nan_credible_interval,
)


_SQRT5 = np.sqrt(5.0)


def _matern52(
    left: np.ndarray,
    right: np.ndarray,
    amplitude: float,
    length_scale: float,
) -> np.ndarray:
    distance = np.abs(left[:, None] - right[None, :])
    scaled = _SQRT5 * distance / length_scale
    return amplitude**2 * (1.0 + scaled + scaled**2 / 3.0) * np.exp(-scaled)


def _matern52_log_length_derivative(
    points: np.ndarray, amplitude: float, length_scale: float
) -> np.ndarray:
    distance = np.abs(points[:, None] - points[None, :])
    scaled = _SQRT5 * distance / length_scale
    return amplitude**2 * (scaled**2 + scaled**3) * np.exp(-scaled) / 3.0


@dataclass(frozen=True)
class _LaplaceMode:
    point: np.ndarray
    objective: float
    covariance: np.ndarray
    log_weight: float


class _LogPosterior:
    def __init__(
        self,
        x: np.ndarray,
        y_scaled: np.ndarray,
        sigma_scaled: np.ndarray,
        config: FitConfig,
        minimum_length: float,
        radius_span_cm: float,
    ) -> None:
        self.x = x
        self.y = y_scaled
        self.sigma = sigma_scaled
        self.variance = np.square(sigma_scaled) + 1e-12
        self.n = len(x)
        self.config = config
        self.radius_span_cm = radius_span_cm
        self.has_free_length = config.optimize_length_scale
        self.fixed_length = float(
            np.clip(config.length_scale_cm / radius_span_cm, minimum_length, 2.0)
        )
        self.minimum_length = minimum_length
        self.log_length_prior_mean = np.log(0.12)
        self.log_length_prior_std = 0.75
        self.bounds = [(-20.0, 12.0)] * self.n
        self.bounds.extend([(-12.0, 12.0), (np.log(0.03), np.log(5.0))])
        if self.has_free_length:
            self.bounds.append((np.log(minimum_length), np.log(2.0)))

    @property
    def dimension(self) -> int:
        return self.n + 2 + int(self.has_free_length)

    def unpack(self, vector: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        z = vector[: self.n]
        beta = float(vector[self.n])
        amplitude = float(np.exp(vector[self.n + 1]))
        if self.has_free_length:
            length_scale = float(np.exp(vector[self.n + 2]))
        else:
            length_scale = self.fixed_length
        return z, beta, amplitude, length_scale

    def initial_point(self, length_scale: float) -> np.ndarray:
        floor = np.maximum(0.05 * self.sigma, 1e-8)
        z = np.log(np.clip(np.maximum(self.y, floor), np.exp(-12.0), np.exp(10.0)))
        weights = 1.0 / self.variance
        beta = float(np.sum(weights * z) / np.sum(weights))
        amplitude = float(np.clip(np.std(z), 0.25, 2.0))
        values = [*z, beta, np.log(amplitude)]
        if self.has_free_length:
            values.append(np.log(np.clip(length_scale, self.minimum_length, 2.0)))
        return np.asarray(values, dtype=float)

    def objective_and_gradient(self, vector: np.ndarray) -> tuple[float, np.ndarray]:
        z, beta, amplitude, length_scale = self.unpack(vector)
        expected = np.exp(np.clip(z, -30.0, 20.0))
        residual = self.y - expected
        likelihood = 0.5 * np.sum(
            np.square(residual) / self.variance + np.log(2.0 * np.pi * self.variance)
        )
        gradient_z_likelihood = (expected - self.y) * expected / self.variance

        signal_covariance = _matern52(self.x, self.x, amplitude, length_scale)
        covariance = signal_covariance + np.eye(self.n) * 1e-7
        try:
            factor = cho_factor(covariance, lower=True, check_finite=False)
            centered = z - beta
            alpha = cho_solve(factor, centered, check_finite=False)
            inverse = cho_solve(factor, np.eye(self.n), check_finite=False)
        except np.linalg.LinAlgError:
            return 1e100, np.zeros_like(vector)

        log_determinant = 2.0 * np.sum(np.log(np.diag(factor[0])))
        gp_term = 0.5 * float(centered @ alpha) + 0.5 * log_determinant
        beta_prior = 0.5 * (beta / 2.0) ** 2
        log_amplitude = float(vector[self.n + 1])
        amplitude_prior = 0.5 * amplitude**2 - log_amplitude
        objective = likelihood + gp_term + beta_prior + amplitude_prior

        common = inverse - np.outer(alpha, alpha)
        gradient_z = gradient_z_likelihood + alpha
        gradient_beta = -float(np.sum(alpha)) + beta / 4.0
        derivative_amplitude = 2.0 * signal_covariance
        gradient_log_amplitude = (
            0.5 * float(np.sum(common * derivative_amplitude)) + amplitude**2 - 1.0
        )
        gradient_parts: list[np.ndarray | float] = [
            gradient_z,
            gradient_beta,
            gradient_log_amplitude,
        ]

        if self.has_free_length:
            log_length = float(vector[self.n + 2])
            length_prior = 0.5 * (
                (log_length - self.log_length_prior_mean) / self.log_length_prior_std
            ) ** 2
            objective += length_prior
            derivative_length = _matern52_log_length_derivative(
                self.x, amplitude, length_scale
            )
            gradient_log_length = 0.5 * float(np.sum(common * derivative_length)) + (
                log_length - self.log_length_prior_mean
            ) / self.log_length_prior_std**2
            gradient_parts.append(gradient_log_length)

        gradient = np.concatenate(
            [
                np.asarray(gradient_parts[0], dtype=float),
                np.asarray(gradient_parts[1:], dtype=float),
            ]
        )
        return float(objective), gradient


@dataclass(frozen=True)
class _LaplaceComponent:
    model: _LogPosterior
    mode: _LaplaceMode


def _finite_difference_hessian(model: _LogPosterior, point: np.ndarray) -> np.ndarray:
    dimension = len(point)
    hessian = np.empty((dimension, dimension), dtype=float)
    for column in range(dimension):
        step = 2e-4 * max(1.0, abs(float(point[column])))
        lower_bound, upper_bound = model.bounds[column]
        plus = point.copy()
        minus = point.copy()
        plus[column] = min(point[column] + step, upper_bound)
        minus[column] = max(point[column] - step, lower_bound)
        denominator = plus[column] - minus[column]
        if denominator <= 0.0:
            hessian[:, column] = 0.0
            continue
        _, gradient_plus = model.objective_and_gradient(plus)
        _, gradient_minus = model.objective_and_gradient(minus)
        hessian[:, column] = (gradient_plus - gradient_minus) / denominator
    return 0.5 * (hessian + hessian.T)


def _laplace_covariance(hessian: np.ndarray) -> tuple[np.ndarray, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    largest = max(float(np.max(eigenvalues)), 1.0)
    stabilized = np.maximum(eigenvalues, 1e-6 * largest)
    inverse_values = np.minimum(1.0 / stabilized, 9.0)
    covariance = (eigenvectors * inverse_values) @ eigenvectors.T
    return covariance, float(np.sum(np.log(stabilized)))


def _optimize_modes(model: _LogPosterior, config: FitConfig) -> list[_LaplaceMode]:
    initial_length = np.clip(
        config.length_scale_cm / model.radius_span_cm,
        model.minimum_length,
        2.0,
    )
    starts = [initial_length]
    if model.has_free_length:
        starts.extend([0.05, 0.12, 0.30, 0.70])

    results: list[OptimizeResult] = []
    for length in starts:
        result = minimize(
            model.objective_and_gradient,
            model.initial_point(length),
            method="L-BFGS-B",
            jac=True,
            bounds=model.bounds,
            options={
                "maxiter": 1200,
                "ftol": 1e-12,
                "gtol": 1e-6,
                "maxls": 100,
                "maxcor": 30,
            },
        )
        if np.isfinite(result.fun):
            results.append(result)
    if not results:
        raise ValueError("Positive latent-log posterior optimization failed.")

    results.sort(key=lambda result: float(result.fun))
    selected: list[OptimizeResult] = []
    best_objective = float(results[0].fun)
    for result in results:
        if float(result.fun) > best_objective + 25.0:
            continue
        # Latent training values can differ slightly along weak directions even
        # when the scientifically meaningful intercept/kernel mode is the same.
        signature = result.x[model.n :]
        if any(
            np.linalg.norm(signature - previous.x[model.n :])
            / np.sqrt(len(signature))
            < 0.05
            for previous in selected
        ):
            continue
        selected.append(result)
        if len(selected) >= config.max_laplace_modes:
            break
    if not selected:
        selected = [results[0]]

    modes: list[_LaplaceMode] = []
    for result in selected:
        hessian = _finite_difference_hessian(model, result.x)
        covariance, log_determinant = _laplace_covariance(hessian)
        log_weight = -float(result.fun) - 0.5 * log_determinant
        modes.append(
            _LaplaceMode(
                point=np.asarray(result.x, dtype=float),
                objective=float(result.fun),
                covariance=covariance,
                log_weight=log_weight,
            )
        )
    return modes


def _inside_bounds(vector: np.ndarray, bounds: list[tuple[float, float]]) -> bool:
    return all(lower <= value <= upper for value, (lower, upper) in zip(vector, bounds))


def _draw_parameter_points(
    components: list[_LaplaceComponent],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[_LogPosterior, np.ndarray]]:
    log_weights = np.asarray([component.mode.log_weight for component in components])
    weights = np.exp(log_weights - logsumexp(log_weights))
    assignments = rng.choice(len(components), size=count, p=weights)
    draws: list[tuple[_LogPosterior, np.ndarray]] = []
    for assignment in assignments:
        component = components[int(assignment)]
        model = component.model
        mode = component.mode
        accepted = None
        for _ in range(100):
            candidate = rng.multivariate_normal(mode.point, mode.covariance)
            if _inside_bounds(candidate, model.bounds):
                accepted = candidate
                break
        draws.append(
            (model, mode.point.copy() if accepted is None else accepted)
        )
    return draws


def _build_laplace_components(
    x: np.ndarray,
    y_scaled: np.ndarray,
    sigma_scaled: np.ndarray,
    config: FitConfig,
    minimum_length: float,
    radius_span_cm: float,
) -> list[_LaplaceComponent]:
    """Approximate length-scale marginalization by log-space quadrature.

    Optimizing the joint latent state and length scale can concentrate on a
    joint-MAP value even when the marginal length-scale posterior is broad.
    Fixed-length Laplace evidence at prior-supported quadrature nodes captures
    that volume effect and gives derivatives materially better calibration.
    """

    if not config.optimize_length_scale:
        model = _LogPosterior(
            x,
            y_scaled,
            sigma_scaled,
            config,
            minimum_length,
            radius_span_cm,
        )
        return [_LaplaceComponent(model=model, mode=_optimize_modes(model, config)[0])]

    log_prior_mean = np.log(0.12)
    log_prior_std = 0.75
    log_lower = np.log(minimum_length)
    log_upper = min(np.log(1.0), log_prior_mean + 2.5 * log_prior_std)
    if log_upper <= log_lower:
        log_nodes = np.asarray([log_lower])
    else:
        log_nodes = np.linspace(log_lower, log_upper, config.length_quadrature_nodes)

    components: list[_LaplaceComponent] = []
    for index, log_length in enumerate(log_nodes):
        length_scale = float(np.exp(log_length))
        fixed_config = replace(
            config,
            optimize_length_scale=False,
            length_scale_cm=length_scale * radius_span_cm,
            max_laplace_modes=1,
        )
        model = _LogPosterior(
            x,
            y_scaled,
            sigma_scaled,
            fixed_config,
            minimum_length,
            radius_span_cm,
        )
        mode = _optimize_modes(model, fixed_config)[0]
        length_log_prior = -0.5 * (
            (log_length - log_prior_mean) / log_prior_std
        ) ** 2
        # Trapezoidal endpoint weights approximate integration over log(ell).
        endpoint_weight = np.log(0.5) if index in {0, len(log_nodes) - 1} else 0.0
        weighted_mode = replace(
            mode,
            log_weight=mode.log_weight + length_log_prior + endpoint_weight,
        )
        components.append(_LaplaceComponent(model=model, mode=weighted_mode))
    return components


def _joint_conditional(
    model: _LogPosterior,
    parameter_point: np.ndarray,
    grid: np.ndarray,
    radius_span_cm: float,
) -> tuple[np.ndarray, np.ndarray]:
    z_train, beta, amplitude, length_scale = model.unpack(parameter_point)
    train_covariance = _matern52(model.x, model.x, amplitude, length_scale)
    train_covariance += np.eye(model.n) * 1e-7
    factor = cho_factor(train_covariance, lower=True, check_finite=False)

    profile_train = _matern52(grid, model.x, amplitude, length_scale)
    delta_grid_train = grid[:, None] - model.x[None, :]
    distance_grid_train = np.abs(delta_grid_train)
    scaled_grid_train = _SQRT5 * distance_grid_train / length_scale
    derivative_train = (
        -amplitude**2
        * (_SQRT5 / length_scale) ** 2
        * delta_grid_train
        * (1.0 + scaled_grid_train)
        * np.exp(-scaled_grid_train)
        / (3.0 * radius_span_cm)
    )
    cross = np.vstack([profile_train, derivative_train])
    solved_center = cho_solve(factor, z_train - beta, check_finite=False)
    mean = np.concatenate(
        [beta + profile_train @ solved_center, derivative_train @ solved_center]
    )

    grid_covariance = _matern52(grid, grid, amplitude, length_scale)
    delta_grid = grid[:, None] - grid[None, :]
    distance_grid = np.abs(delta_grid)
    scaled_grid = _SQRT5 * distance_grid / length_scale
    coefficient = amplitude**2 * (_SQRT5 / length_scale) ** 2 / 3.0
    profile_derivative = (
        coefficient
        * delta_grid
        * (1.0 + scaled_grid)
        * np.exp(-scaled_grid)
        / radius_span_cm
    )
    derivative_covariance = (
        coefficient
        * (1.0 + scaled_grid - scaled_grid**2)
        * np.exp(-scaled_grid)
        / radius_span_cm**2
    )
    prior_joint = np.block(
        [
            [grid_covariance, profile_derivative],
            [profile_derivative.T, derivative_covariance],
        ]
    )
    conditional = prior_joint - cross @ cho_solve(factor, cross.T, check_finite=False)
    conditional = 0.5 * (conditional + conditional.T)
    eigenvalues, eigenvectors = np.linalg.eigh(conditional)
    tolerance = 1e-10 * max(float(np.max(eigenvalues)), 1.0)
    conditional = (eigenvectors * np.maximum(eigenvalues, tolerance)) @ eigenvectors.T
    return mean, conditional


def fit_positive_log_profile(
    radius_cm: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    config: FitConfig,
) -> PosteriorSummary:
    """Fit a positive latent log profile and analytically infer its derivative."""

    if config.kernel != "Matern 5/2":
        raise ValueError("The positive latent-log model currently supports Matérn 5/2 only.")
    if config.reference_scale <= 0.0:
        raise ValueError("reference_scale must be positive.")
    if not 0.0 < config.confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one.")
    if config.n_samples < 20 or config.grid_size < 25:
        raise ValueError("Use at least 20 posterior samples and 25 grid points.")
    if config.hyperparameter_samples < 1:
        raise ValueError("hyperparameter_samples must be positive.")
    if config.optimize_length_scale and config.length_quadrature_nodes < 3:
        raise ValueError("Use at least three length-scale quadrature nodes.")

    radius_cm = np.asarray(radius_cm, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    errors = np.asarray(errors, dtype=float).reshape(-1)
    if not (radius_cm.shape == values.shape == errors.shape):
        raise ValueError("radius, values, and errors must have identical shapes.")

    finite_radius = radius_cm[np.isfinite(radius_cm)]
    if len(finite_radius) < 2:
        raise ValueError("At least two finite radial positions are required.")
    radius_min = float(np.min(finite_radius))
    measured_radius_max = float(np.max(finite_radius))
    if measured_radius_max - radius_min <= 0.0:
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
    observed_radius = radius_cm[mask]
    observed_values = values[mask]
    observed_errors = errors[mask]
    if len(observed_radius) < 4:
        raise ValueError("Fewer than four valid measurements remain after filtering.")
    order = np.argsort(observed_radius)
    observed_radius = observed_radius[order]
    observed_values = observed_values[order]
    observed_errors = observed_errors[order]

    edge_observation = None
    fit_radius = observed_radius.copy()
    fit_values = observed_values.copy()
    fit_errors = observed_errors.copy()
    if config.edge_mode != "None":
        if config.edge_mode != "User-specified physical point":
            raise ValueError(f"Unsupported edge mode {config.edge_mode!r}.")
        if (
            config.edge_radius_cm is None
            or config.edge_radius_cm <= measured_radius_max
        ):
            raise ValueError("The edge coordinate must lie beyond the outermost measurement.")
        if config.edge_uncertainty is None or config.edge_uncertainty <= 0.0:
            raise ValueError("The edge uncertainty must be positive.")
        edge_error = float(config.edge_uncertainty)
        fit_radius = np.append(fit_radius, config.edge_radius_cm)
        fit_values = np.append(fit_values, config.edge_value)
        fit_errors = np.append(fit_errors, edge_error)
        edge_observation = (
            float(config.edge_radius_cm),
            float(config.edge_value),
            edge_error,
        )

    radius_max = (
        float(config.edge_radius_cm)
        if edge_observation is not None
        else measured_radius_max
    )
    radius_span = radius_max - radius_min

    x = (fit_radius - radius_min) / radius_span
    spacing = np.diff(np.unique(np.sort((observed_radius - radius_min) / radius_span)))
    # Do not infer structure below the typical channel separation: the data do
    # not resolve such a length scale, even if the optimizer can numerically use it.
    minimum_length = max(0.01, float(np.median(spacing)))

    components = _build_laplace_components(
        x,
        fit_values / config.reference_scale,
        fit_errors / config.reference_scale,
        config,
        minimum_length,
        radius_span,
    )

    rng = np.random.default_rng(config.random_seed)
    parameter_count = min(config.hyperparameter_samples, config.n_samples)
    parameter_draws = _draw_parameter_points(components, parameter_count, rng)
    grid_radius = np.linspace(radius_min, radius_max, config.grid_size)
    grid = (grid_radius - radius_min) / radius_span
    samples_per_parameter = int(np.ceil(config.n_samples / parameter_count))
    joint_samples: list[np.ndarray] = []
    length_draws: list[float] = []
    for model, parameter_point in parameter_draws:
        mean, covariance = _joint_conditional(model, parameter_point, grid, radius_span)
        try:
            factor = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            factor = np.linalg.cholesky(covariance + np.eye(len(mean)) * 1e-8)
        normal = rng.standard_normal((len(mean), samples_per_parameter))
        joint_samples.append(mean[:, None] + factor @ normal)
        length_draws.append(model.unpack(parameter_point)[3] * radius_span)
    samples = np.concatenate(joint_samples, axis=1)[:, : config.n_samples]

    latent_samples = samples[: config.grid_size]
    log_gradient_samples = samples[config.grid_size :]
    profile_samples = config.reference_scale * np.exp(
        np.clip(latent_samples, -30.0, 20.0)
    )
    gradient_samples = profile_samples * log_gradient_samples

    profile_low, profile_median, profile_high = _credible_interval(
        profile_samples, config.confidence
    )
    gradient_low, gradient_median, gradient_high = _credible_interval(
        gradient_samples, config.confidence
    )
    inverse_low, inverse_median, inverse_high = _credible_interval(
        log_gradient_samples, config.confidence
    )
    positive_probability = np.mean(log_gradient_samples > 0.0, axis=1)
    tail_probability = 0.5 * (1.0 - config.confidence)
    sign_unstable = (positive_probability > tail_probability) & (
        positive_probability < 1.0 - tail_probability
    )
    typical_spacing_cm = float(np.median(np.diff(np.unique(observed_radius))))
    boundary_margin_cm = 2.0 * typical_spacing_cm
    boundary_limited = (
        grid_radius < observed_radius[0] + boundary_margin_cm
    ) | (grid_radius > observed_radius[-1] - boundary_margin_cm)
    scale_unstable = sign_unstable | boundary_limited
    safe_log_gradient = np.where(
        np.abs(log_gradient_samples) > 1e-14, log_gradient_samples, np.nan
    )
    scale_samples = 1.0 / safe_log_gradient
    scale_low, scale_median, scale_high = _nan_credible_interval(
        scale_samples, config.confidence
    )
    scale_low[scale_unstable] = np.nan
    scale_median[scale_unstable] = np.nan
    scale_high[scale_unstable] = np.nan

    component_log_weights = np.asarray(
        [component.mode.log_weight for component in components]
    )
    component_weights = np.exp(
        component_log_weights - logsumexp(component_log_weights)
    )
    best_component = components[int(np.argmax(component_log_weights))]
    fitted_length = float(np.median(length_draws))
    length_low, length_high = np.percentile(length_draws, [2.5, 97.5])
    material_components = int(np.sum(component_weights >= 0.01))
    return PosteriorSummary(
        radius_cm=grid_radius,
        profile_mean=np.mean(profile_samples, axis=1),
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
        inverse_scale_unstable=np.zeros(config.grid_size, dtype=bool),
        derivative_boundary_limited=boundary_limited,
        near_zero_gradient_fraction=np.zeros(config.grid_size),
        observed_radius_cm=observed_radius,
        observed_values=observed_values,
        observed_errors=observed_errors,
        fitted_length_scale_cm=fitted_length,
        kernel_description=(
            "Positive latent-log GP with Matérn-5/2 covariance; "
            f"length-scale median {fitted_length:.3g} cm "
            f"(sampled 95% range {length_low:.3g}–{length_high:.3g} cm); "
            f"{material_components} material Laplace/quadrature components"
        ),
        log_marginal_likelihood=float("nan"),
        confidence=config.confidence,
        edge_observation=edge_observation,
        model_name="Positive latent-log GP",
        gradient_positive_probability=positive_probability,
        fit_score_label="Best Laplace component log weight (relative)",
        fit_score=best_component.mode.log_weight,
        hyperparameter_modes=material_components,
    )
