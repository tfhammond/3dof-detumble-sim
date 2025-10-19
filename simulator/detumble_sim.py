import numpy as np

from controller.bdotcontroller import BDotController
from attitude_dynamics.dynamics import Dynamics
from magnetic_field.model import MagneticFieldModel, rotate_eci

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
            "p": [],
            "pv": [],
            "b_norm": [],
            "k_bdot": [],
            "m_cmd": [],
            "t_on": [],
            "dir": []
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

        t0_utc = _as_utc(t0)
        t_end = _as_utc(t_final)

        duration = max(0.0, (t_end - t0_utc).total_seconds())
        T_s = float(self.cfg.T_s)
        h = float(self.cfg.h)
        Nsub = max(1, int(np.ceil(T_s / h)))
        h = T_s / Nsub

        x_orbit = x_orbit0.copy() # check if i need this later

        print(f"End time = {duration}")




        # K_field = 5
        # next_field_update = 0.0
        # b_body_T = None










        t_sim = 0.0

        while t_sim <= duration + 1e-12 and not self._detumbled:
            
            epoch_now = t0_utc + timedelta(seconds=t_sim)

            #test
            t1 = perf_counter()

            b_eci_T = self.field.b_eci(x_orbit[:3], epoch_now)
            #b_body_T = self._field_body_now(x_orbit, epoch_now)


            #test
            field_time += perf_counter() - t1

            #next_field_update = t_sim + K_field * T_s







            b_tilde_B = rotate_eci(self.att.q_IB, b_eci_T) #???

            #TEST
            t_cmd = perf_counter()



            out = self.ctrl.command(b_tilde_B)


            #TEST
            cmd_time += perf_counter() - t_cmd
            ctrl_calls += 1


            m_cmd = out["m_cmd"]
            t_on = out["t_on"]
            dir_vec = out["dir"]
            p = out["p"]
            pv = out["p_v"]
            k_bdot = out["k_bdot"]

            b_norm = float(np.linalg.norm(b_tilde_B))

            # if np.all(pv <= self.cfg.p_bar):
            #     self._stop_counter += 1
            # if np.all(self.att.w_B <= np.deg2rad(np.array([2.0, 2.0, 2.0]))):
            #     self._stop_counter += 1

            if np.linalg.norm(self.att.w_B) <= np.deg2rad(2.0):
                self._stop_counter += 1
            else:
                self._stop_counter = 0
            if self._stop_counter >= self.cfg.Nw:
                self._detumbled = True

            

            if self._detumbled:
                break


            elapsed = 0.0
            m_c = np.zeros(3)
            for i in range(Nsub):
                m_c.fill(0.0) #commanded mag dipole
                for j in range(3):
                    if elapsed < float(t_on[j]):
                        m_c[j] = float(m_cmd[j])
                    else:
                        m_c[j] = 0.0

                # epoch_now_sub = t0_utc + timedelta(seconds=(t_sim + elapsed))
                # b_body_sub_T = self._field_body_now(x_orbit, epoch_now_sub)

                # tau_c_B = np.cross(m_c, b_body_sub_T)

                b_body_sub_T = rotate_eci(self.att.q_IB, b_eci_T)

                tau_c_B = np.cross(m_c, b_body_sub_T)

                #test
                t2 = perf_counter()




                self.att.step(h, tau_c_B)

                #test
                att_time += perf_counter() - t2


                #test
                t3 = perf_counter()


                x_orbit = self.orbit_step(x_orbit, h)



                #test
                orb_time += perf_counter() - t3
                substeps += 1


                elapsed += h
            
            
            # if self.cfg.log_every_sample:
            #     self._log_sample(t_sim, self.att.w_B, p, pv, b_norm, k_bdot, m_cmd, t_on, dir_vec)

            if self.cfg.log_every_sample:
                self._log_sample(t_sim + T_s, self.att.w_B, p, pv, b_norm, k_bdot, m_cmd, t_on, dir_vec)
            t_sim += T_s

        

        if self._detumbled:
            t_det = t0_utc + timedelta(seconds=t_sim)
        else:
            t_det = None


        #TESTS
        t_total = perf_counter() - t_wall0
        print(f"[Perf] sim_sec={duration:.3f}  wall_sec={t_total:.3f}  "
        f"field={field_time:.3f}s  cmd={cmd_time:.3f}s  att={att_time:.3f}s  orb={orb_time:.3f}s  "
        f"ctrl_ticks={ctrl_calls} field_updates={ctrl_calls} substeps={substeps} ")
        #TESTS



        return {
            #"t": np.asarray(self.log["t"], dtype="datetime64[ns]") if self.cfg.log_every_sample else np.array([], dtype="datetime64[ns]"),
            "t": np.asarray(self.log["t"], dtype=float) if self.cfg.log_every_sample else np.array([], dtype=float),
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