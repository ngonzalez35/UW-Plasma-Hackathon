#!/usr/bin/env python3
"""Sweep every bundled MPTS profile through the default GP reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thomson_profiles.data import list_shots, list_times, load_profile_slice
from thomson_profiles.gp import FitConfig, assess_profile_validity, fit_profile


def percentile_summary(values: pd.Series) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return {}
    return {
        "min": float(finite.min()),
        "p05": float(finite.quantile(0.05)),
        "median": float(finite.median()),
        "p95": float(finite.quantile(0.95)),
        "max": float(finite.max()),
    }


def run_sweep(
    data_path: Path,
    samples: int,
    grid_size: int,
    hyperparameter_samples: int,
    quadrature_nodes: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for shot in list_shots(data_path):
        times = list_times(data_path, shot)
        for time_index, time_s in enumerate(times):
            profile = load_profile_slice(data_path, shot, time_index)
            for quantity_key, seed_offset in (("te", 0), ("ne", 1)):
                quantity = profile.quantity(quantity_key)
                relative_errors = quantity.errors / quantity.values
                validity = assess_profile_validity(quantity.values, quantity.errors)
                config = FitConfig(
                    n_samples=samples,
                    grid_size=grid_size,
                    hyperparameter_samples=hyperparameter_samples,
                    length_quadrature_nodes=quadrature_nodes,
                    max_laplace_modes=2,
                    reference_scale=1.0 if quantity_key == "te" else 1e13,
                    random_seed=int(shot) + 2 * time_index + seed_offset,
                )
                row: dict[str, object] = {
                    "shot": shot,
                    "time_index": time_index,
                    "time_s": float(time_s),
                    "quantity": quantity_key,
                    "success": False,
                    "error": "",
                    "median_relative_error": float(np.median(relative_errors)),
                    "max_relative_error": float(np.max(relative_errors)),
                    "validity_status": validity.status,
                    "resolved_points": validity.resolved_points,
                    "posterior_samples": samples,
                    "grid_size": grid_size,
                    "hyperparameter_samples": hyperparameter_samples,
                    "length_quadrature_nodes": quadrature_nodes,
                }
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        fit = fit_profile(
                            profile.radius_cm,
                            quantity.values,
                            quantity.errors,
                            config,
                        )
                    convergence_messages = [
                        str(item.message)
                        for item in caught
                        if issubclass(item.category, ConvergenceWarning)
                    ]
                    radius_span = float(np.ptp(profile.radius_cm))
                    length_fraction = fit.fitted_length_scale_cm / radius_span
                    row.update(
                        {
                            "success": True,
                            "n_observations": fit.n_observations,
                            "fitted_length_scale_cm": fit.fitted_length_scale_cm,
                            "length_scale_over_span": length_fraction,
                            "length_at_optimizer_bound": bool(
                                length_fraction <= 0.0051 or length_fraction >= 1.99
                            ),
                            "fit_score": fit.fit_score,
                            "fit_score_label": fit.fit_score_label,
                            "hyperparameter_modes": fit.hyperparameter_modes,
                            "negative_profile_median_fraction": float(
                                np.mean(fit.profile_median < 0.0)
                            ),
                            "profile_band_below_zero_fraction": float(
                                np.mean(fit.profile_low < 0.0)
                            ),
                            "scale_unstable_fraction": float(
                                np.mean(fit.scale_unstable)
                            ),
                            "inverse_scale_unstable_fraction": float(
                                np.mean(fit.inverse_scale_unstable)
                            ),
                            "convergence_warning_count": len(convergence_messages),
                            "convergence_warnings": " | ".join(convergence_messages),
                        }
                    )
                except Exception as exc:  # keep the sweep going and report every failure
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                if len(rows) % 100 == 0:
                    print(f"Completed {len(rows)} profile fits…", flush=True)
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, elapsed_seconds: float) -> dict[str, object]:
    successful = frame[frame["success"]].copy()
    by_quantity: dict[str, object] = {}
    for quantity, group in successful.groupby("quantity"):
        by_quantity[str(quantity)] = {
            "profiles": int(len(group)),
            "fitted_length_scale_cm": percentile_summary(
                group["fitted_length_scale_cm"]
            ),
            "scale_unstable_fraction": percentile_summary(
                group["scale_unstable_fraction"]
            ),
            "profile_band_below_zero_fraction": percentile_summary(
                group["profile_band_below_zero_fraction"]
            ),
            "median_relative_error": percentile_summary(
                group["median_relative_error"]
            ),
        }

    return {
        "data_path": str(frame.attrs.get("data_path", "")),
        "elapsed_seconds": elapsed_seconds,
        "profiles_attempted": int(len(frame)),
        "profiles_succeeded": int(frame["success"].sum()),
        "profiles_failed": int((~frame["success"]).sum()),
        "profiles_with_convergence_warning": int(
            (successful["convergence_warning_count"] > 0).sum()
        ),
        "profiles_with_length_at_optimizer_bound": int(
            successful["length_at_optimizer_bound"].sum()
        ),
        "profiles_with_negative_posterior_median": int(
            (successful["negative_profile_median_fraction"] > 0.0).sum()
        ),
        "profiles_with_over_90_percent_scale_unstable": int(
            (successful["scale_unstable_fraction"] > 0.90).sum()
        ),
        "validity_status_counts": {
            str(key): int(value)
            for key, value in frame["validity_status"].value_counts().items()
        },
        "by_quantity": by_quantity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("nstx-profiles.hdf5"))
    parser.add_argument("--output-dir", type=Path, default=Path("validation"))
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--grid-size", type=int, default=60)
    parser.add_argument("--hyperparameter-samples", type=int, default=3)
    parser.add_argument("--quadrature-nodes", type=int, default=5)
    args = parser.parse_args()

    start = perf_counter()
    frame = run_sweep(
        args.data,
        args.samples,
        args.grid_size,
        args.hyperparameter_samples,
        args.quadrature_nodes,
    )
    elapsed = perf_counter() - start
    frame.attrs["data_path"] = str(args.data.resolve())
    summary = summarize(frame, elapsed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "positive_log_profile_robustness_sweep.csv"
    json_path = args.output_dir / "positive_log_profile_robustness_summary.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
