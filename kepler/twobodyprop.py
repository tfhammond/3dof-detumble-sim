import numpy as np
from tle_io.keplerelement import KeplerElements


def rk4_step(f, x, dt):
    """ generic rk4 step """
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4) #array

class TwoBodyPropagator:
    def equation_of_motion(self, state):
        """state = [r(3), v(3)]
            returns x
        """
        r =state[:3]
        v = state[3:]
        r_norm = np.linalg.norm(r)
        a = -self.KeplerElements.MU_E * r / r_norm**3
        return np.hstack((v, a))
    
    def step(self, state, dt):
        """ propagate the state by dt seconds using rk4 """
        return rk4_step(self.equation_of_motion, state, dt)
    




