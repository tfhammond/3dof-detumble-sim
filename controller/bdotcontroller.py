import numpy as np
from dataclasses import dataclass

#not sure about this

@dataclass
class BDotController:

    T_s : float
    duty : float # duty cycle d in greek letter thing
    m_bar : np.array
    I_min : float
    omega_orbit : float
    xi_geomag : float
    phi : float
    eps : float
    alpha : float
    polarity: np.array

    prev_b_hat: np.array

    keep_vector_p : bool #maybe
    prev_p : float
    prev_pv : np.array

    def p_scalar(self, b_hat_dot):

        increase = (self.alpha * self.T_s / 2.0) * float(np.linalg.norm(b_hat_dot))
        p = increase + (1.0 - self.alpha) * self.prev_p
        p = float(np.clip(p, 0.0, 1.0))
        self.p_prev = p
        return p
    
    def p_vector(self, b_hat_dot):

        abs_dot = np.abs(b_hat_dot)
        increase = (self.alpha * self.T_s / 2.0) * abs_dot
        if self.prev_pv is None:
            pv = increase
        else:
            pv = increase + (1.0 - self.alpha) * self.prev_pv
        
        pv = np.clip(pv, 0.0, 1.0)
        self.prev_pv = pv
        return pv




    def command(self, b_tilde_B):

        """
        b_tilde_B = magnetic field at time_k eq 21
        
        
        returns a dictionary or array (erase this when u decide)
        
        """

        b_norm = float(np.linalg.norm(b_tilde_B))

        if b_norm == 0.0:
            b_hat = np.zeros(3)
            b_hat_dot = np.zeros(3)
        else:
            b_hat = b_tilde_B / b_norm

            if b_hat_dot is None:
                b_hat_dot = np.zeros(3)
            else:
                b_hat_dot = (b_hat - self.prev_b_hat) / self.T_s # eq 22

        
       #p = 0.0 # REPLACE THIS!
        #pv = 0.0
        p = self.p_scalar(b_hat_dot) # eq 23 & 24
        pv = self.p_vector(b_hat_dot) # eq 33

        k_star = 2.0 * self.omega_orbit * (1.0 + np.sin(self.xi_geomag)) * self.I_min # eq 11
        k_bdot = k_star / (self.phi * p + self.eps) # eq 10

        # dipole eq 9

        if b_norm == 0.0:
            m_d = np.zeros(3)
        else:
            m_d = -(k_bdot / b_norm) * b_hat_dot

        

        #25 and 26 and 27
        m_abs = np.abs(m_d)

        
        #idk lol

        #check line after eq 27


                # Compute scale safely without divide-by-zero warnings
        valid = self.m_bar > 0.0                     # only use positive limits
        ratio = np.zeros_like(m_abs)                 # start with zeros
        ratio[valid] = m_abs[valid] / self.m_bar[valid]  # compute ratio where valid
        scale = np.minimum(ratio, 1.0)               # clamp to a maximum of 1.0

        t_on = self.duty * self.T_s * scale
        sgn = np.sign(m_d).astype(int)

        dir_vec = (self.polarity.astype(int) * sgn).astype(int)
        m_cmd = dir_vec * np.minimum(m_abs, self.m_bar)


        self.prev_b_hat = b_hat

        dict_return = {"m_cmd": m_cmd, "dir": dir_vec, "t_on": t_on, "p": float(p), "pv": pv, "gain": float(k_bdot)}

        return dict_return






 
        



            