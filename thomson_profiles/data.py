"""Read the NSTX MPTS profile schema used by the hackathon data set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class QuantityData:
    """One measured radial quantity and its pointwise standard uncertainty."""

    key: str
    symbol: str
    title: str
    units: str
    values: np.ndarray
    errors: np.ndarray


@dataclass(frozen=True)
class ProfileSlice:
    """MPTS measurements for one shot at one acquisition time."""

    shot: str
    time_index: int
    time_s: float
    radius_cm: np.ndarray
    te_keV: np.ndarray
    te_error_keV: np.ndarray
    ne_cm3: np.ndarray
    ne_error_cm3: np.ndarray

    def quantity(self, key: str) -> QuantityData:
        if key == "te":
            return QuantityData(
                key="te",
                symbol="Tₑ",
                title="Electron temperature",
                units="keV",
                values=self.te_keV,
                errors=self.te_error_keV,
            )
        if key == "ne":
            return QuantityData(
                key="ne",
                symbol="nₑ",
                title="Electron density",
                units="cm⁻³",
                values=self.ne_cm3,
                errors=self.ne_error_cm3,
            )
        raise ValueError(f"Unknown quantity {key!r}; expected 'te' or 'ne'.")


def _mpts_group(h5: h5py.File, shot: str) -> h5py.Group:
    path = f"{shot}/mpts"
    if path not in h5:
        raise ValueError(f"Shot {shot!r} has no MPTS group at /{path}.")
    return h5[path]


def list_shots(path: str | Path) -> list[str]:
    """Return numeric shot identifiers that contain an MPTS group."""

    with h5py.File(Path(path), "r") as h5:
        shots = [name for name in h5 if f"{name}/mpts" in h5]
    return sorted(shots, key=int)


def list_times(path: str | Path, shot: str | int) -> np.ndarray:
    """Return the acquisition times for one shot in seconds."""

    with h5py.File(Path(path), "r") as h5:
        group = _mpts_group(h5, str(shot))
        return np.asarray(group["time"][:], dtype=float)


def load_profile_slice(
    path: str | Path, shot: str | int, time_index: int
) -> ProfileSlice:
    """Load one radial profile, preserving the file's ``(radius, time)`` axes."""

    shot = str(shot)
    with h5py.File(Path(path), "r") as h5:
        group = _mpts_group(h5, shot)
        time = np.asarray(group["time"][:], dtype=float)
        if not 0 <= time_index < len(time):
            raise IndexError(
                f"time_index={time_index} is outside [0, {len(time) - 1}] "
                f"for shot {shot}."
            )

        radius = np.asarray(group["radius"][:], dtype=float)
        arrays = {
            name: np.asarray(group[name][:, time_index], dtype=float)
            for name in ("te", "te_error", "ne", "ne_error")
        }

    for name, values in arrays.items():
        if values.shape != radius.shape:
            raise ValueError(
                f"Dataset {shot}/mpts/{name} has a radial slice of shape "
                f"{values.shape}; expected {radius.shape}."
            )

    return ProfileSlice(
        shot=shot,
        time_index=time_index,
        time_s=float(time[time_index]),
        radius_cm=radius,
        te_keV=arrays["te"],
        te_error_keV=arrays["te_error"],
        ne_cm3=arrays["ne"],
        ne_error_cm3=arrays["ne_error"],
    )
