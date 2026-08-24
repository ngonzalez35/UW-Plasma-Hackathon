import numpy as np

from thomson_profiles.gp import FitConfig, assess_profile_validity, fit_profile


def test_fit_produces_ordered_bands_and_physical_derivative_units():
    radius = np.linspace(0.0, 10.0, 16)
    values = 3.0 - 0.12 * radius
    errors = np.full_like(radius, 0.04)
    config = FitConfig(
        model="Linear GP baseline",
        kernel="RBF",
        optimize_length_scale=False,
        length_scale_cm=4.0,
        n_samples=80,
        grid_size=70,
        random_seed=7,
    )

    result = fit_profile(radius, values, errors, config)

    assert result.profile_median.shape == (70,)
    assert result.gradient_median.shape == (70,)
    assert np.all(result.profile_low <= result.profile_median)
    assert np.all(result.profile_median <= result.profile_high)
    assert np.median(result.gradient_median[8:-8]) < -0.08
    assert np.isclose(result.fitted_length_scale_cm, 4.0)


def test_flat_profile_marks_scale_length_as_weakly_identified():
    radius = np.linspace(0.0, 12.0, 18)
    values = np.ones_like(radius) * 2.0
    errors = np.ones_like(radius) * 0.08
    config = FitConfig(
        model="Positive latent-log GP",
        kernel="Matern 5/2",
        optimize_length_scale=False,
        length_scale_cm=5.0,
        n_samples=100,
        grid_size=60,
        hyperparameter_samples=4,
        random_seed=11,
    )

    result = fit_profile(radius, values, errors, config)

    assert np.mean(result.scale_unstable) > 0.7
    assert np.mean(np.isnan(result.scale_length_median)) > 0.7


def test_relative_uncertainty_filter_is_applied_before_fit():
    radius = np.linspace(0.0, 10.0, 10)
    values = 2.0 - 0.05 * radius
    errors = np.full_like(radius, 0.05)
    errors[-2:] = 10.0
    config = FitConfig(
        model="Positive latent-log GP",
        optimize_length_scale=False,
        n_samples=40,
        grid_size=40,
        hyperparameter_samples=2,
        max_relative_uncertainty=1.0,
    )

    result = fit_profile(radius, values, errors, config)

    assert result.n_observations == 8


def test_positive_log_model_enforces_support_and_recovers_log_gradient():
    radius = np.linspace(0.0, 10.0, 18)
    expected_log_gradient = -0.08
    values = 2.0 * np.exp(expected_log_gradient * radius)
    errors = 0.025 * values
    config = FitConfig(
        model="Positive latent-log GP",
        optimize_length_scale=False,
        length_scale_cm=4.0,
        n_samples=80,
        grid_size=60,
        hyperparameter_samples=4,
        random_seed=19,
    )

    result = fit_profile(radius, values, errors, config)

    assert np.all(result.profile_low > 0.0)
    assert np.isclose(
        np.median(result.inverse_scale_median[8:-8]),
        expected_log_gradient,
        atol=0.03,
    )
    assert np.mean(result.gradient_positive_probability[8:-8]) < 0.05
    assert result.model_name == "Positive latent-log GP"


def test_provisional_validity_gate_distinguishes_signal_from_no_signal():
    good = assess_profile_validity(np.ones(30), np.full(30, 0.1))
    weak = assess_profile_validity(np.ones(30), np.full(30, 3.0))

    assert good.status == "Usable"
    assert good.resolved_points == 30
    assert weak.status == "Unusable"


def test_external_edge_observation_requires_a_distinct_coordinate():
    radius = np.linspace(0.0, 10.0, 12)
    values = 1.5 - 0.03 * radius
    errors = np.full_like(radius, 0.05)
    config = FitConfig(
        model="Positive latent-log GP",
        optimize_length_scale=False,
        length_scale_cm=4.0,
        n_samples=40,
        grid_size=40,
        hyperparameter_samples=2,
        edge_mode="User-specified physical point",
        edge_radius_cm=11.0,
        edge_value=0.8,
        edge_uncertainty=0.1,
    )

    result = fit_profile(radius, values, errors, config)

    assert result.radius_cm[-1] == 11.0
    assert result.n_observations == len(radius)
    assert result.edge_observation == (11.0, 0.8, 0.1)


def test_automatic_length_inference_uses_laplace_quadrature_components():
    radius = np.linspace(0.0, 12.0, 20)
    values = 0.2 + np.exp(-np.square((radius - 6.0) / 3.0))
    errors = np.full_like(radius, 0.04)
    config = FitConfig(
        model="Positive latent-log GP",
        optimize_length_scale=True,
        length_quadrature_nodes=5,
        hyperparameter_samples=6,
        n_samples=60,
        grid_size=50,
        random_seed=29,
    )

    result = fit_profile(radius, values, errors, config)

    assert result.fitted_length_scale_cm > 0.0
    assert result.hyperparameter_modes >= 1
    assert "Laplace/quadrature" in result.kernel_description
    assert np.all(result.profile_low > 0.0)
