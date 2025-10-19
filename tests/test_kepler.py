import numpy as np

from kepler.kepler import KeplerToRV
from tle_io.keplerelement import KeplerElements


def test_rv_eci_circular_equatorial_orbit():
    a = 7000e3
    elements = KeplerElements(
        i=0.0,
        raan=0.0,
        e=0.0,
        w=0.0,
        M=0.0,
        n=np.sqrt(KeplerElements.MU_E / a**3),
        a=a,
    )

    r_eci, v_eci = KeplerToRV().rv_eci(elements)

    np.testing.assert_allclose(r_eci, np.array([a, 0.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(
        v_eci,
        np.array([0.0, np.sqrt(KeplerElements.MU_E / a), 0.0]),
        atol=1e-6,
    )


def test_rv_eci_general_inclined_eccentric_orbit():
    a = 26560e3
    e = 0.1
    elements = KeplerElements(
        i=np.deg2rad(63.4),
        raan=np.deg2rad(45.0),
        e=e,
        w=np.deg2rad(30.0),
        M=np.deg2rad(40.0),
        n=np.sqrt(KeplerElements.MU_E / a**3),
        a=a,
    )

    r_eci, v_eci = KeplerToRV().rv_eci(elements)

    np.testing.assert_allclose(
        r_eci,
        np.array([-4047156.98661689, 11226472.58091381, 21567275.45640172]),
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        v_eci,
        np.array([-3192.37799809, -2471.09015579, 1018.50143148]),
        rtol=0,
        atol=1e-6,
    )
