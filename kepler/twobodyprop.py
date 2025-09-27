import numpy as np
from tle_io.keplerelement import KeplerElements


class TwoBodyPropagator:
    def equation_of_motion(self, state):
        r =state[:3]
        v = state[3:]
        r_norm = np.linalg.norm(r)
        a = -self.mu * r / r_norm**3
        return np.hstack((v, a))
    
    def rk4_step(self, state, dt):
        k1 = self.equation_of_motion(state)
        k2 = self.equation_of_motion(state + 0.5 * dt * k1)
        k3 = self.equation_of_motion(state + 0.5 * dt * k2)
        k4 = self.equation_of_motion(state + dt * k3)
        return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)



