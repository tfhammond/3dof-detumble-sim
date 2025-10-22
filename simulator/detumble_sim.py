import numpy as np

from controller.bdotcontroller import BDotController
from attitude_dynamics.dynamics import Dynamics
from magnetic_field.model import MagneticFieldModel, rotate_eci, GMSTTracker, gmst_angle_rad

from simulator.config import DetumbleConfig

from datetime import timezone, timedelta

from time import perf_counter


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
            "b_norm": [],
            "m_cmd": [],
            "t_on": [],
            "dir": [],
            "r_eci": [],   # meters, ECI
            "v_eci": [],   # m/s, ECI
            "r_norm": [],  # |r| [m]
            "v_norm": [],  # |v| [m/s]
        }
    


    def run(self, t0, t_final, x_orbit0):
        #test
        t_wall0 = perf_counter()
        field_time = 0.0
        att_time = 0.0
        orb_time = 0.0
        ctrl_calls = 0
        substeps = 0
        cmd_time = 0.0
        #test

        t0_utc = _as_utc(t0).replace(tzinfo=None)
        t_end = _as_utc(t_final).replace(tzinfo=None)

        duration = (t_end - t0_utc).total_seconds()
        T_s = float(self.cfg.T_s)
        h = float(self.cfg.h)
        Nsub = max(1, int(np.ceil(T_s / h)))
        h = T_s / Nsub

        gmst0 = gmst_angle_rad(t0_utc)
        self.field._gmst = GMSTTracker(t0_utc, gmst0)

        x_orbit = x_orbit0.copy()

        print(f"Sim Duration (s) = {duration}")

        



        t_sim = 0.0
        t_on_sum = [0.0,0.0,0.0]
        amount_of_cycles = 0

        while t_sim <= duration + 1e-12 and not self._detumbled:
            
            epoch_now = t0_utc + timedelta(seconds=t_sim)

            #test
            t1 = perf_counter()
            b_eci_T = self.field.b_eci(x_orbit[:3], epoch_now)
            #test
            field_time += perf_counter() - t1

            b_body_T = rotate_eci(self.att.q_IB, b_eci_T) #???
            b_norm = float(np.linalg.norm(b_body_T))
            #TEST
            t_cmd = perf_counter()

            out = self.ctrl.command(b_body_T)
            m_cmd = out[0]
            t_on = out[1]

            t_on_sum[0] += t_on[0]
            t_on_sum[1] += t_on[1]
            t_on_sum[2] += t_on[2]
            amount_of_cycles += 1

            dir_vec = out[2]

            #TEST
            cmd_time += perf_counter() - t_cmd
            ctrl_calls += 1

            if b_norm <= 0.0:
                m_cmd = np.zeros(3)
                t_on = np.zeros(3)
                dir_vec = np.zeros(3, dtype=int)

            if np.linalg.norm(self.att.w_B) <= self.cfg.w_stop_rad:
                self._stop_counter += 1
            else:
                self._stop_counter = 0
            if self._stop_counter >= self.cfg.Nw:
                self._detumbled = True
                break

            # scrapped eq 26 and using the average instead
            m_eff = dir_vec.astype(float) * self.ctrl.cfg.m_bar * (t_on / self.cfg.T_s)

            elapsed = 0.0
            for i in range(Nsub):
                t1 = perf_counter()
                b_body_sub_T = rotate_eci(self.att.q_IB, b_eci_T)
                field_time += perf_counter() - t1
                tau_c_B = np.cross(m_eff, b_body_sub_T)
                #test
                t2 = perf_counter()
                self.att.step(h, tau_c_B) #STEP
                #test
                att_time += perf_counter() - t2
                #test
                t3 = perf_counter()
                x_orbit = self.orbit_step(x_orbit, h)
                #test
                orb_time += perf_counter() - t3
                substeps += 1
                elapsed += h
            
            if self.cfg.log_every_sample:
                r_eci = x_orbit[:3].copy()
                v_eci = x_orbit[3:].copy()
                self._log_sample(t_sim + T_s, self.att.w_B, b_norm, m_cmd, t_on, dir_vec, r_eci, v_eci)
            t_sim += T_s


        if self._detumbled:
            t_det = t0_utc + timedelta(seconds=t_sim)
        else:
            t_det = None

        t_total = perf_counter() - t_wall0
        print(f"[Perf] RATIO={duration/t_total:.3f}im_sec={duration:.3f}  wall_sec={t_total:.3f}  "
          f"field={field_time:.3f}s  cmd={cmd_time:.3f}s  att={att_time:.3f}s  "
          f"orb={orb_time:.3f}s  ctrl_ticks={ctrl_calls} substeps={substeps}")



        return {
            "t": np.asarray(self.log["t"], dtype=float) if self.cfg.log_every_sample else np.array([], dtype=float),
            "w_B": self._maybe_array("w_B", (0, 3)),
            "b_norm": self._maybe_array("b_norm", (0,)),
            "m_cmd": self._maybe_array("m_cmd", (0, 3)),
            "t_on": self._maybe_array("t_on", (0, 3)),
            "dir": self._maybe_array("dir", (0, 3)),

            "r_eci": self._maybe_array("r_eci", (0, 3)),
            "v_eci": self._maybe_array("v_eci", (0, 3)),
            "r_norm": self._maybe_array("r_norm", (0,)),
            "v_norm": self._maybe_array("v_norm", (0,)),


            "t_detumbled": np.datetime64(t_det) if t_det is not None else None,
            "detumbled": bool(self._detumbled),
        }

    def _empty(self, shape):
        return np.empty(shape)

    def _maybe_array(self, key, shape):
        if self.cfg.log_every_sample:
            return np.asarray(self.log[key])
        return self._empty(shape)
            
    def _log_sample(self, t, w_B,b_norm,m_cmd,t_on,dir_vec, r_eci, v_eci):
        """Collect per-sample metrics for analysis and validation."""
        self.log["t"].append(t)
        self.log["w_B"].append(np.asarray(w_B, dtype=float).copy())
        self.log["b_norm"].append(float(b_norm))
        self.log["m_cmd"].append(np.asarray(m_cmd, dtype=float).copy())
        self.log["t_on"].append(np.asarray(t_on, dtype=float).copy())
        self.log["dir"].append(np.asarray(dir_vec, dtype=int).copy())

        r = np.asarray(r_eci, dtype=float).copy()
        v = np.asarray(v_eci, dtype=float).copy()
        self.log["r_eci"].append(r)
        self.log["v_eci"].append(v)
        self.log["r_norm"].append(float(np.linalg.norm(r)))
        self.log["v_norm"].append(float(np.linalg.norm(v)))