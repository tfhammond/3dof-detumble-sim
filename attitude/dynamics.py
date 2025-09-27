import numpy as np

from kepler.twobodyprop import TwoBodyPropagator
#from math.math import 


class Dynamics:

    def control_torque(self, m_c_B, b_E_B):
        """Magnetic control torque in body coords (Eq 5)"""
        tau_c_B = np.cross(m_c_B, b_E_B)
        return tau_c_B
    
    def euler_moment(self, tau_c_B):
        """
        Euler's euqation (Eq 1). No disturbances yet so torque = torque_c_B
        I*w_dot = -w * (I*w) + tau_c_B
        """
        pass











        