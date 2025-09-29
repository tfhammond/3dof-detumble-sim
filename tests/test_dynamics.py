import numpy as np

from attitude_dynamics.dynamics import Dynamics


def make_dynamics(I=None, q=None, w=None):
    if I is None:
        I = np.eye(3)
    if q is None:
        q = np.array([1.0, 0.0, 0.0, 0.0])
    if w is None:
        w = np.zeros(3)
    return Dynamics(I=I, q_IB=q.copy(), w_B=w.copy())


def test_control_torque_matches_cross_product():
    dynamics = make_dynamics()
    m_c_B = np.array([0.1, -0.2, 0.3])
    b_E_B = np.array([-0.4, 0.5, 0.6])
    expected = np.cross(m_c_B, b_E_B)

    result = dynamics.control_torque(m_c_B, b_E_B)

    np.testing.assert_allclose(result, expected)


def test_euler_moment_zero_state_zero_torque():
    inertia = np.diag([2.0, 3.0, 4.0])
    w = np.zeros(3)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    dynamics = make_dynamics(I=inertia, q=q, w=w)
    state = np.hstack((w, q))
    tau = np.zeros(3)

    derivative = dynamics.euler_moment(state, tau)

    np.testing.assert_allclose(derivative, np.zeros(7))


def test_step_constant_angular_velocity_updates_quaternion():
    inertia = np.diag([4.0, 6.0, 8.0])
    w = np.array([0.2, 0.0, 0.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    dynamics = make_dynamics(I=inertia, q=q, w=w)
    tau = np.zeros(3)
    dt = 0.5

    dynamics.step(dt, tau)

    np.testing.assert_allclose(dynamics.w_B, w, atol=1e-12)
    angle = np.linalg.norm(w) * dt
    expected_q = np.array([np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0])
    np.testing.assert_allclose(dynamics.q_IB, expected_q, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(dynamics.q_IB), 1.0, atol=1e-12)


def test_step_normalizes_quaternion_when_input_not_unit():
    dynamics = make_dynamics(q=np.array([2.0, 0.0, 0.0, 0.0]), w=np.zeros(3))
    tau = np.zeros(3)

    dynamics.step(0.0, tau)

    np.testing.assert_allclose(dynamics.q_IB, np.array([1.0, 0.0, 0.0, 0.0]))
