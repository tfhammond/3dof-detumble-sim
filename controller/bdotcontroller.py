import numpy as np
from dataclasses import dataclass, field

#not sure about this

@dataclass
class BDotConfig:
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

    keep_vector_p : bool #maybe


class BDotController:


    def __init__(self, cfg: BDotConfig):

        self.cfg = cfg

        self.prev_b_hat = None
        self.prev_p = 0.0
        self.prev_pv = np.zeros(3)
        #self.prev_pv = np.array([self.T_s, self.T_s, self.T_s])
        #trapezoid
        self.prev_bhatdot_norm = 0.0
        self.prev_bhatdot_abs  = np.zeros(3)

        #idk
        assert np.all(self.cfg.m_bar > 0.0)
        assert self.cfg.T_s > 0.0 and 0.0 < self.cfg.duty <= 1.0
        assert self.cfg.phi > 0.0 and self.cfg.eps > 0.0
        assert 0.0 < self.cfg.alpha <= 1.0
        assert np.all(np.isin(self.cfg.polarity.astype(int), [-1, 1]))

        if not np.all(np.isin(self.cfg.polarity, [-1, 0, 1])):
            raise ValueError(
                f"Invalid polarity values {self.cfg.polarity}. "
                "Each element must be -1, 0, or 1."
            )
        if self.cfg.polarity.shape != (3,):
            raise ValueError(
                f"Polarity must be a 3-element vector; got shape {self.cfg.polarity.shape}"
            )
        print(f"[BDotController] Coil polarity set to {self.cfg.polarity}")


    def command(self, b_E_B):
        
        # normalize magnetic field 
        b_norm = float(np.linalg.norm(b_E_B))

        # compute normalized B-dot
        if b_norm <= 0.0:
            b_hat = np.zeros(3)
            b_hat_dot = np.zeros(3)
        else:
            b_hat = b_E_B / b_norm
            if self.prev_b_hat is None:
                b_hat_dot = np.zeros(3)
            else:
                b_hat_dot = (b_hat - self.prev_b_hat) / self.cfg.T_s

        #for trapezoidal p and pv
        cur_norm = float(np.linalg.norm(b_hat_dot))
        cur_abs_vec = np.abs(b_hat_dot)

        # scalar and vector "p" filters (your original version) 
        p_increase = (self.cfg.alpha * self.cfg.T_s / 2.0) * (cur_norm + self.prev_bhatdot_norm)
        p = (1.0 - self.cfg.alpha) * self.prev_p + p_increase # 23
        p = float(np.clip(p, 0.0, 1.0)) # 24

        p_v_increase = (self.cfg.alpha * self.cfg.T_s / 2.0) * (cur_abs_vec + self.prev_bhatdot_abs)
        p_v = (1.0 - self.cfg.alpha) * self.prev_pv + p_v_increase #33
        p_v = np.clip(p_v, 0.0, 1.0)

        # gain schedule 
        k_star = 2.0 * self.cfg.omega_orbit * (1.0 + np.sin(self.cfg.xi_geomag)) * self.cfg.I_min
        k_bdot = k_star / (self.cfg.phi * p + self.cfg.eps)
        #k_bdot = 0.1

        #  compute desired dipole (m_des) 
        if b_norm <= 0.0:
            m_des = np.zeros(3)
        else:
            m_des = -(k_bdot / b_norm) * b_hat_dot

        

        #  saturation, duty cycle, and direction 

        ratio = np.zeros(3, dtype=float)
        np.divide(np.abs(m_des), self.cfg.m_bar, out=ratio, where=self.cfg.m_bar>0)
        scale = np.minimum(ratio,1.0)


        #scale = np.minimum(np.abs(m_des) / self.cfg.m_bar, 1.0)
        t_on = self.cfg.duty * self.cfg.T_s * scale 
        t_on = np.clip(t_on, 0.0, self.cfg.duty * self.cfg.T_s) #eq 26

        # dir_vec = np.sign(m_des).astype(int) # eq 27
        # dir_vec = dir_vec * self.cfg.polarity 

        dir_vec = np.sign(m_des).astype(int) * self.cfg.polarity.astype(int)
        dir_vec = np.where(scale > 0.0, dir_vec, 0)
        
        m_cmd = dir_vec * self.cfg.m_bar

        # update stored values for next step 
        self.prev_b_hat = b_hat
        self.prev_p = p
        self.prev_pv = p_v

        self.prev_bhatdot_norm = cur_norm
        self.prev_bhatdot_abs = cur_abs_vec

        np.testing.assert_array_equal(np.sign(dir_vec).astype(int), dir_vec.astype(int))

        # return all useful quantities for logging 
        return {
            "p": p,
            "p_v": p_v,
            "b_hat_dot": b_hat_dot,
            "k_bdot": k_bdot,
            "m_cmd": m_cmd,
            "t_on": t_on,
            "dir": dir_vec,
            "m_des": m_des,
        }

 
        



            