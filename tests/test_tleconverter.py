import math

import pytest

from tle_io.tleconverter import TLEConverter
from tle_io.keplerelement import KeplerElements


def test_parse_extracts_expected_kepler_elements():
    lines = [
        "1 25544U 98067A   20029.54791667  .00000716  00000-0  21590-4 0  9990",
        "2 25544  51.6445  38.0185 0007417  85.4867  51.4902 15.49184748209363",
    ]

    result = TLEConverter.parse(lines)

    assert isinstance(result, KeplerElements)

    expected_inclination = math.radians(51.6445)
    expected_raan = math.radians(38.0185)
    expected_eccentricity = 0.0007417
    expected_mean_anomaly = math.radians(51.4902)
    expected_mean_motion = 15.49184748
    expected_n = expected_mean_motion * 2 * math.pi / (24 * 3600)
    expected_semi_major_axis = (KeplerElements.MU_E / expected_n**2) ** (1 / 3)

    assert result.i == pytest.approx(expected_inclination)
    assert result.raan == pytest.approx(expected_raan)
    assert result.e == pytest.approx(expected_eccentricity)
    assert result.M == pytest.approx(expected_mean_anomaly)
    assert result.n == pytest.approx(expected_n)
    assert result.a == pytest.approx(expected_semi_major_axis)
