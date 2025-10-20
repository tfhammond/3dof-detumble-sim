import numpy as np

from kepler.twobodyprop import rk4_step
from math_equations.math_eqs import normalize_quat, quat_rate_scalar_first

class Dynamics:
    
    def __init__(self, I, q_IB, w_B):

        I = np.array(I, dtype=float, copy=True)
        self.I = I

        diag = np.diag(self.I)
        self._is_diag = np.allclose(self.I, np.diag(diag))

        if self._is_diag:
            self.I_diag = diag
            self.invI = None
        else:
            self.I_diag = None
            self.invI = np.linalg.inv(I) # inverse I


        self.q_IB = normalize_quat(np.asarray(q_IB, dtype=float).copy())
        self.w_B = np.asarray(w_B, dtype=float).copy()

    @staticmethod
    def control_torque(m_c_B, b_E_B):
        """Magnetic control torque in body coords (Eq 5)"""
        return np.cross(m_c_B, b_E_B)
    
    def euler_moment(self, state, tau_c_B):
        """
        state = [first 3 w_B, 4 q)]


        Euler's euqation (Eq 1). No disturbances yet so torque = torque_c_B
        I*w_dot = -w x (I*w) + tau_c_B
        """
        
        w = state[:3]
        q = state[3:7]

        if self._is_diag:
            Iw = self.I_diag * w
        else:
            Iw = self.I @ w

        w_cross_Iw = np.cross(w, Iw)
        rhs = -w_cross_Iw + tau_c_B

        if self._is_diag:
            wdot = rhs / self.I_diag
        else:
            wdot = self.invI @ rhs

        q_dot = quat_rate_scalar_first(q, w)

        xdot = np.empty(7, dtype=float)
        xdot[:3] = wdot
        xdot[3:] = q_dot
        return xdot
    
    def step(self, dt, tau_c_B): #quaternion or no
        """ propagate the state by dt seconds using rk4 """

        x0 = np.empty(7, dtype=float)
        x0[:3] = self.w_B
        x0[3:] = self.q_IB


        #x = np.hstack((self.w_B, self.q_IB))

        # Rk4 
        def f(x):
            return self.euler_moment(x, tau_c_B)

        x1 = rk4_step(f, x0, dt)
    
        self.w_B = x1[:3]
        self.q_IB = normalize_quat(x1[3:])
