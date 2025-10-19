import numpy as np
from attitude_dynamics.dynamics import Dynamics


def test_dynamics_step():
        # Inertia (kg·m^2) and initial state
    I   = np.diag([0.011, 0.012, 0.013])
    q0  = np.array([1.0, 0.0, 0.0, 0.0])
    w0  = np.deg2rad([0.0, 0.0, 120.0])  # 120 deg/s about body z

    dynamics = Dynamics(I=I, q_IB=q0, w_B=w0)

    # Step forward for 10 s with no control torque yet (pure free motion)
    dt = 0.005
    for _ in range(int(10.0 / dt)):
        dynamics.step(dt=dt, tau_c_B=np.zeros(3))

    print("Final ω_B [deg/s]:", np.rad2deg(dynamics.w_B))
    print("||q||:", np.linalg.norm(dynamics.q_IB), "  kinetic V:", dynamics.kinetic_energy())

if __name__ == "__main__":
    test_dynamics_step()