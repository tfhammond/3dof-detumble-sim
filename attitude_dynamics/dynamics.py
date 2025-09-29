import numpy as np

from dataclasses import dataclass

from kepler.twobodyprop import TwoBodyPropagator, rk4_step
from math_equations.math import w_matrix
from math_equations.math import normalize_quat

@dataclass
class Dynamics:
    I: np.ndarray
    q_IB : np.ndarray
    w_B : np.ndarray

    def control_torque(self, m_c_B, b_E_B):
        """Magnetic control torque in body coords (Eq 5)"""
        tau_c_B = np.cross(m_c_B, b_E_B)
        return tau_c_B
    
    def euler_moment(self, state, tau_c_B):
        """
        state = [first 3 w_B, 4 q)]


        Euler's euqation (Eq 1). No disturbances yet so torque = torque_c_B
        I*w_dot = -w x (I*w) + tau_c_B
        """
        
        w = state[:3]
        q = state[3:7]

        w_cross_Iw = np.cross(w, self.I @ w)
        wdot = np.linalg.solve(self.I, -w_cross_Iw + tau_c_B) #ax = b

        q_dot = 0.5 * (w_matrix(w) @ q) # need to implement w_matrix 

        return np.hstack((wdot, q_dot))
    
    def step(self, dt, tau_c_B): #quaternion or no
        """ propagate the state by dt seconds using rk4 """

        x = np.hstack((self.w_B, self.q_IB))

        # Rk4 
        def f(local_x):
            return self.euler_moment(local_x, tau_c_B)

        x_next = rk4_step(f, x, dt)
    
        self.w_B = x_next[:3]
        self.q_IB = normalize_quat(x_next[3:7])
