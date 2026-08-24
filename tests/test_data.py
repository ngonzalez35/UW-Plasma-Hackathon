from pathlib import Path

import numpy as np

from thomson_profiles.data import list_shots, list_times, load_profile_slice


DATA = Path(__file__).parents[1] / "nstx-profiles.hdf5"


def test_repository_data_exposes_expected_mpts_schema():
    shots = list_shots(DATA)
    assert "141687" in shots
    assert len(shots) == 14

    times = list_times(DATA, "141687")
    profile = load_profile_slice(DATA, "141687", int(np.argmin(abs(times - 0.315))))

    assert profile.radius_cm.shape == (30,)
    assert profile.te_keV.shape == (30,)
    assert profile.ne_cm3.shape == (30,)
    assert np.all(np.diff(profile.radius_cm) > 0)
    assert np.all(profile.te_error_keV > 0)
    assert np.all(profile.ne_error_cm3 > 0)


def test_quantity_metadata_and_unknown_key():
    profile = load_profile_slice(DATA, "141687", 0)
    assert profile.quantity("te").units == "keV"
    assert profile.quantity("ne").units == "cm⁻³"

    try:
        profile.quantity("pressure")
    except ValueError as exc:
        assert "pressure" in str(exc)
    else:
        raise AssertionError("Unknown quantity should have raised ValueError")
