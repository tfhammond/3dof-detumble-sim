import numpy as np
from dataclasses import dataclass

@dataclass
class BDotConfig:
    T_s : float
    duty : float # duty cycle d in greek letter thing
    m_bar : np.array
    I_min : float
    omega_orbit : float
    xi_geomag : float
    polarity: np.array

class BDotController:

    def __init__(self, cfg: BDotConfig):

        self.cfg = cfg
        self.k_bdot_star = 2.0 * cfg.omega_orbit * (1.0 + np.sin((cfg.xi_geomag))) * cfg.I_min
        self.prev_b_hat = None

    def command(self, b_E_B):
        """b_E_B [3,]
        
        
            returns m_d, t_on, direction [3,]
        """

        b_norm = np.linalg.norm(b_E_B)

        if b_norm <= 0.0:
            m_d = np.zeros(3)
            return m_d, np.zeros(3), np.zeros(3, dtype=int)
        
        b_hat = b_E_B / b_norm

        
        if self.prev_b_hat is None:
            b_hat_dot = np.zeros(3)
        else:
            b_hat_dot = (b_hat - self.prev_b_hat) / self.cfg.T_s

        self.prev_b_hat = b_hat

        m_d = -(self.k_bdot_star / b_norm) * b_hat_dot

        t_on = np.empty(3)
        direction = np.zeros(3, dtype=int)

        for i in range(3):
            if self.cfg.m_bar[i] <= 0.0:
                t_on[i] = 0.0
                direction[i] = 0
                continue
            scale = abs(m_d[i]) / self.cfg.m_bar[i]
            t_on[i] = self.cfg.duty * self.cfg.T_s * min(1.0, scale)
            if t_on[i] > 0.0:
                direction[i] = int(self.cfg.polarity[i] * np.sign(m_d[i]))
            else:
                direction[i] = 0
            
        return m_d, t_on, direction