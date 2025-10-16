import numpy as np

from controller.bdotcontroller import BDotController
from attitude_dynamics.dynamics import Dynamics
from magnetic_field.model import MagneticFieldModel

from simulator.config import DetumbleConfig

from datetime import timezone, timedelta
from zoneinfo import ZoneInfo

def _as_utc(dt):
    """Ensure timezone-aware UTC (IGRF expects real UTC epochs)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class DetumbleSim:
    # core detumble sim loop


    def __init__(self, orbit_step, attitude, field, mags, ctrl, cfg):

        self.orbit_step = orbit_step #orbital propagator: x -> x_next for state x = [r(3); v(3)].
        self.att = attitude
        self.field = field
        self.mags = mags # dont need?
        self.ctrl = ctrl
        self.cfg = cfg # DetumbleConfig

        self._stop_counter = 0
        self._detumbled = False

        self.log = {
            "t": [],
            "w_B": [],
            "p": [],
            "pv": [],
            "b_norm": [],
            "k_bdot": [],
            "m_cmd": [],
            "t_on": [],
            "dir": []
        }
    


    def run(self, t0, t_final, x_orbit0):

        current_time = _as_utc(t0)
        #print(current_time.timestamp())
        t_end = _as_utc(t_final)
        T_s = self.cfg.T_s
        h = self.cfg.h
        Nsub = max(1, int(np.ceil(T_s / h)))
        h = T_s / Nsub
        x_orbit = x_orbit0.copy() #do i need .copy()

        print(f"End time (Pacific) = {t_end.astimezone(ZoneInfo('America/Los_Angeles'))}")

        while current_time <= t_end and not self._detumbled:

            b_body_T = self._field_body_now(x_orbit, current_time)

            b_tilde_B = b_body_T #???

            out = self.ctrl.command(b_tilde_B)
            m_cmd = out["m_cmd"]
            t_on = out["t_on"]
            dir_vec = out["dir"]
            p = out["p"]
            pv = out["p_v"]
            k_bdot = out["k_bdot"]

            b_norm = float(np.linalg.norm(b_tilde_B))

            if np.all(pv <= self.cfg.p_bar):
                self._stop_counter += 1
            else:
                self._stop_counter = 0
            if self._stop_counter >= self.cfg.Nw:
                self._detumbled = True

            if self.cfg.log_every_sample:
                self._log_sample(current_time, self.att.w_B, p, pv, b_norm, k_bdot, m_cmd, t_on, dir_vec)

            if self._detumbled:
                break


            elapsed = 0.0
            for i in range(Nsub):
                m_c = np.zeros(3) #commanded mag dipole
                for j in range(3):
                    if elapsed < float(t_on[j]):
                        m_c[j] = float(m_cmd[j])
                    else:
                        m_c[j] = 0.0

                b_body_sub_T = self._field_body_now(x_orbit, current_time)

                tau_c_B = np.cross(m_c, b_body_sub_T)

                self.att.step(h, tau_c_B)
                x_orbit = self.orbit_step(x_orbit, h)

                current_time = current_time + timedelta(seconds=h)
                elapsed += h

        if self._detumbled:
            t_det = current_time
        else:
            t_det = None

        

        return {
            "t": np.asarray(self.log["t"], dtype="datetime64[ns]") if self.cfg.log_every_sample else np.array([], dtype="datetime64[ns]"),
            "w_B": self._maybe_array("w_B", (0, 3)),
            "p": self._maybe_array("p", (0,)),
            "pv": self._maybe_array("pv", (0, 3)),
            "b_norm": self._maybe_array("b_norm", (0,)),
            "k_bdot": self._maybe_array("k_bdot", (0,)),
            "m_cmd": self._maybe_array("m_cmd", (0, 3)),
            "t_on": self._maybe_array("t_on", (0, 3)),
            "dir": self._maybe_array("dir", (0, 3)),
            "t_detumbled": np.datetime64(t_det) if t_det is not None else None,
            "detumbled": bool(self._detumbled),
        }



    def _empty(self, shape):
        return np.empty(shape)

    def _maybe_array(self, key, shape):
        if self.cfg.log_every_sample:
            return np.asarray(self.log[key])
        return self._empty(shape)
            
    

    def _log_sample(self, t, w_B,p,pv,b_norm,k_bdot,m_cmd,t_on,dir_vec):
        """Collect per-sample metrics for analysis and validation."""
        self.log["t"].append(t)
        self.log["w_B"].append(np.asarray(w_B, dtype=float).copy())
        self.log["p"].append(float(p))
        self.log["pv"].append(np.asarray(pv, dtype=float).copy())
        self.log["b_norm"].append(float(b_norm))
        self.log["k_bdot"].append(float(k_bdot))
        self.log["m_cmd"].append(np.asarray(m_cmd, dtype=float).copy())
        self.log["t_on"].append(np.asarray(t_on, dtype=float).copy())
        self.log["dir"].append(np.asarray(dir_vec, dtype=int).copy())



    def _field_body_now(self, x_orbit, current_time):

        r_eci_m = x_orbit[:3]

        return self.field.b_body(r_eci_m, self.att.q_IB, current_time) #T``