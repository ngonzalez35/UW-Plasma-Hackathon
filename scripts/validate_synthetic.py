#!/usr/bin/env python3
"""Validate sampled profile derivatives against smooth functions with known truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thomson_profiles.gp import FitConfig, fit_profile


def peaked_profile(radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = (radius - 6.0) / 2.8
    exponential = np.exp(-np.square(centered))
    profile = 0.12 + 1.25 * exponential
    gradient = -2.5 * (radius - 6.0) / (2.8**2) * exponential
    return profile, gradient


def linear_profile(radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return 2.5 - 0.08 * radius, np.full_like(radius, -0.08)


def interval_coverage(truth: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    valid = np.isfinite(truth) & np.isfinite(low) & np.isfinite(high)
    return float(np.mean((truth[valid] >= low[valid]) & (truth[valid] <= high[valid])))


def evaluate_case(
    name: str,
    truth_function,
    *,
    noise_sigma: float,
    seed: int,
    grid_size: int,
    samples: int,
) -> tuple[dict[str, object], object]:
    observed_radius = np.linspace(0.0, 12.0, 25)
    exact_observed, _ = truth_function(observed_radius)
    rng = np.random.default_rng(seed)
    errors = np.full_like(observed_radius, noise_sigma)
    observed = exact_observed + rng.normal(0.0, errors)

    fit = fit_profile(
        observed_radius,
        observed,
        errors,
        FitConfig(
            kernel="Matern 5/2",
            optimize_length_scale=True,
            length_scale_cm=3.0,
            n_samples=samples,
            grid_size=grid_size,
            random_seed=seed,
        ),
    )
    exact_profile, exact_gradient = truth_function(fit.radius_cm)
    exact_log_gradient = exact_gradient / exact_profile
    exact_scale = np.divide(
        exact_profile,
        exact_gradient,
        out=np.full_like(exact_profile, np.nan),
        where=np.abs(exact_gradient) > 0.02,
    )
    scale_mask = np.isfinite(exact_scale) & ~fit.scale_unstable
    derivative_reportable = ~fit.derivative_boundary_limited

    result = {
        "name": name,
        "grid_size": grid_size,
        "posterior_samples": samples,
        "fitted_length_scale_cm": fit.fitted_length_scale_cm,
        "profile_rmse": float(np.sqrt(np.mean(np.square(fit.profile_median - exact_profile)))),
        "gradient_rmse": float(
            np.sqrt(np.mean(np.square(fit.gradient_median - exact_gradient)))
        ),
        "log_gradient_rmse": float(
            np.sqrt(np.mean(np.square(fit.inverse_scale_median - exact_log_gradient)))
        ),
        "profile_pointwise_inclusion_fraction": interval_coverage(
            exact_profile, fit.profile_low, fit.profile_high
        ),
        "gradient_pointwise_inclusion_fraction": interval_coverage(
            exact_gradient, fit.gradient_low, fit.gradient_high
        ),
        "gradient_reportable_region_inclusion_fraction": interval_coverage(
            exact_gradient[derivative_reportable],
            fit.gradient_low[derivative_reportable],
            fit.gradient_high[derivative_reportable],
        ),
        "scale_length_pointwise_inclusion_fraction": (
            interval_coverage(
                exact_scale[scale_mask],
                fit.scale_length_low[scale_mask],
                fit.scale_length_high[scale_mask],
            )
            if np.any(scale_mask)
            else None
        ),
        "scale_unstable_fraction": float(np.mean(fit.scale_unstable)),
    }
    return result, fit


def flat_case() -> dict[str, object]:
    radius = np.linspace(0.0, 12.0, 25)
    values = np.full_like(radius, 1.5)
    errors = np.full_like(radius, 0.06)
    fit = fit_profile(
        radius,
        values,
        errors,
        FitConfig(
            optimize_length_scale=False,
            length_scale_cm=5.0,
            n_samples=400,
            grid_size=220,
            random_seed=303,
        ),
    )
    return {
        "name": "flat",
        "expected_gradient": 0.0,
        "median_absolute_inferred_gradient": float(
            np.median(np.abs(fit.gradient_median))
        ),
        "gradient_interval_contains_zero_fraction": float(
            np.mean((fit.gradient_low <= 0.0) & (fit.gradient_high >= 0.0))
        ),
        "scale_unstable_fraction": float(np.mean(fit.scale_unstable)),
    }


def repeated_calibration_case(
    name: str,
    truth_function,
    *,
    noise_sigma: float,
    repetitions: int,
) -> dict[str, object]:
    """Estimate repeated-noise pointwise calibration on a compact grid."""

    observed_radius = np.linspace(0.0, 12.0, 25)
    exact_observed, _ = truth_function(observed_radius)
    profile_coverage: list[float] = []
    gradient_coverage: list[float] = []
    gradient_reportable_coverage: list[float] = []
    log_gradient_coverage: list[float] = []
    log_gradient_reportable_coverage: list[float] = []
    profile_rmse: list[float] = []
    gradient_rmse: list[float] = []
    negative_support: list[float] = []
    for seed in range(repetitions):
        errors = np.full_like(observed_radius, noise_sigma)
        observed = exact_observed + np.random.default_rng(seed).normal(0.0, errors)
        fit = fit_profile(
            observed_radius,
            observed,
            errors,
            FitConfig(
                n_samples=400,
                grid_size=60,
                hyperparameter_samples=24,
                random_seed=seed,
            ),
        )
        exact_profile, exact_gradient = truth_function(fit.radius_cm)
        exact_log_gradient = exact_gradient / exact_profile
        profile_coverage.append(
            interval_coverage(exact_profile, fit.profile_low, fit.profile_high)
        )
        gradient_coverage.append(
            interval_coverage(exact_gradient, fit.gradient_low, fit.gradient_high)
        )
        log_gradient_coverage.append(
            interval_coverage(
                exact_log_gradient, fit.inverse_scale_low, fit.inverse_scale_high
            )
        )
        reportable = ~fit.derivative_boundary_limited
        gradient_reportable_coverage.append(
            interval_coverage(
                exact_gradient[reportable],
                fit.gradient_low[reportable],
                fit.gradient_high[reportable],
            )
        )
        log_gradient_reportable_coverage.append(
            interval_coverage(
                exact_log_gradient[reportable],
                fit.inverse_scale_low[reportable],
                fit.inverse_scale_high[reportable],
            )
        )
        profile_rmse.append(
            float(np.sqrt(np.mean(np.square(fit.profile_median - exact_profile))))
        )
        gradient_rmse.append(
            float(np.sqrt(np.mean(np.square(fit.gradient_median - exact_gradient))))
        )
        negative_support.append(float(np.mean(fit.profile_low <= 0.0)))

    return {
        "name": name,
        "repetitions": repetitions,
        "mean_profile_pointwise_coverage": float(np.mean(profile_coverage)),
        "mean_gradient_pointwise_coverage": float(np.mean(gradient_coverage)),
        "mean_gradient_reportable_region_coverage": float(
            np.mean(gradient_reportable_coverage)
        ),
        "mean_log_gradient_pointwise_coverage": float(np.mean(log_gradient_coverage)),
        "mean_log_gradient_reportable_region_coverage": float(
            np.mean(log_gradient_reportable_coverage)
        ),
        "mean_profile_rmse": float(np.mean(profile_rmse)),
        "mean_gradient_rmse": float(np.mean(gradient_rmse)),
        "mean_nonpositive_lower_band_fraction": float(np.mean(negative_support)),
    }


def sensitivity_study() -> list[dict[str, object]]:
    baseline_result, baseline = evaluate_case(
        "peaked",
        peaked_profile,
        noise_sigma=0.035,
        seed=101,
        grid_size=220,
        samples=400,
    )
    studies: list[dict[str, object]] = []
    for grid_size, samples in ((100, 200), (220, 800), (400, 400)):
        result, fit = evaluate_case(
            "peaked",
            peaked_profile,
            noise_sigma=0.035,
            seed=101,
            grid_size=grid_size,
            samples=samples,
        )
        profile_on_baseline = np.interp(
            baseline.radius_cm, fit.radius_cm, fit.profile_median
        )
        gradient_on_baseline = np.interp(
            baseline.radius_cm, fit.radius_cm, fit.gradient_median
        )
        result.update(
            {
                "profile_rmse_vs_baseline": float(
                    np.sqrt(
                        np.mean(
                            np.square(profile_on_baseline - baseline.profile_median)
                        )
                    )
                ),
                "gradient_rmse_vs_baseline": float(
                    np.sqrt(
                        np.mean(
                            np.square(gradient_on_baseline - baseline.gradient_median)
                        )
                    )
                ),
            }
        )
        studies.append(result)
    return [baseline_result, *studies]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("validation"))
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()

    linear, _ = evaluate_case(
        "linear",
        linear_profile,
        noise_sigma=0.04,
        seed=99,
        grid_size=220,
        samples=400,
    )
    sensitivity = sensitivity_study()
    results = {
        "known_function_cases": [linear, sensitivity[0], flat_case()],
        "grid_and_sample_sensitivity": sensitivity,
        "repeated_calibration": [
            repeated_calibration_case(
                "linear",
                linear_profile,
                noise_sigma=0.04,
                repetitions=args.repetitions,
            ),
            repeated_calibration_case(
                "peaked",
                peaked_profile,
                noise_sigma=0.035,
                repetitions=args.repetitions,
            ),
            repeated_calibration_case(
                "flat",
                lambda radius: (
                    np.full_like(radius, 1.5),
                    np.zeros_like(radius),
                ),
                noise_sigma=0.06,
                repetitions=args.repetitions,
            ),
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "positive_log_synthetic_validation.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
