import math

import numpy as np

from kepler.twobodyprop import TwoBodyPropagator
from tle_io.keplerelement import KeplerElements


def _make_prop():
    prop = TwoBodyPropagator()
    prop.KeplerElements = KeplerElements
    return prop


def test_equation_of_motion_matches_newtonian_gravity():
    prop = _make_prop()
    radius = 7_000_000.0
    r = np.array([radius, 0.0, 0.0])
    v = np.array([0.0, math.sqrt(KeplerElements.MU_E / radius), 0.0])
    state = np.hstack((r, v))

    result = prop.equation_of_motion(state)

    expected_accel = -KeplerElements.MU_E * r / np.linalg.norm(r) ** 3
    expected = np.hstack((v, expected_accel))

    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0.0)


def test_step_preserves_circular_orbit_over_short_step():
    prop = _make_prop()
    radius = 7_000_000.0
    mu = KeplerElements.MU_E
    angular_speed = math.sqrt(mu / radius ** 3)
    velocity_mag = math.sqrt(mu / radius)
    dt = 1.0

    state = np.array([radius, 0.0, 0.0, 0.0, velocity_mag, 0.0], dtype=float)

    next_state = prop.step(state, dt)

    angle = angular_speed * dt
    expected_r = np.array([radius * math.cos(angle), radius * math.sin(angle), 0.0])
    expected_v = np.array([-velocity_mag * math.sin(angle), velocity_mag * math.cos(angle), 0.0])

    np.testing.assert_allclose(next_state[:3], expected_r, atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(next_state[3:], expected_v, atol=1e-12, rtol=0.0)
