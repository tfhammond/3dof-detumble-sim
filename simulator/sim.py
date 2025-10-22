import numpy as np
import matplotlib.pyplot as plt
import requests

from datetime import datetime, timezone, timedelta

from tle_io.tleload import TLELoader
from tle_io.tleconverter import TLEConverter
from tle_io.keplerelement import KeplerElements
from kepler.kepler import KeplerToRV
from kepler.twobodyprop import TwoBodyPropagator
from simulator.detumble_sim import DetumbleSim
from simulator.config import DetumbleConfig
from controller.bdotcontroller import BDotConfig, BDotController
from attitude_dynamics.dynamics import Dynamics
from magnetic_field.model import MagneticFieldModel, GMSTTracker, gmst_angle_rad, rotate_eci


iss_txt_url = "https://live.ariss.org/iss.txt"

def load_kepler_from_tle():

    # ex tle. Can use iss_txt_url if you want
    tle = "1 25544U 98067A   19178.82735530  .00002515  00000-0  49918-4 0  9997\n2 25544  51.6428 308.4904 0008116  89.4883  70.1063 15.51247238176900"
    load = TLELoader.read_lines(tle)
    kep = TLEConverter.parse(load)

    return kep


def make_orbit_stepper():
    prop = TwoBodyPropagator()
    def orbit_step(x,dt):
        return prop.step(x,dt)
    return orbit_step

def build_sim():
    #orbit intial state
    kep = load_kepler_from_tle()

    print(kep)

    r0, v0 = KeplerToRV().rv_eci(kep)
    x_orbit0 = np.hstack((r0, v0))

    I = np.diag([0.001731, 0.001726, 0.000264]) # kg*m^2
    #I =  np.diag([0.001, 0.001, 0.001]) # kg·m^2 (example)
    q_IB0 = np.array([1.0, 0.0, 0.0, 0.0]) # scalar-first quaternion
    w_B0 = np.deg2rad(np.array([180.0, 180.0, 180.0]))  # rad/s (fast tumbling start) from paper
    dyn = Dynamics(I=I, q_IB=q_IB0, w_B=w_B0)


    aware0 = datetime.now(timezone.utc)
    t0_native = aware0.replace(tzinfo=None)
    gmst0 = gmst_angle_rad(t0_native)
    gmst_tracker = GMSTTracker(t0_native, gmst0)

    #field model
    field = MagneticFieldModel(gmst_tracker)

    #controller

    T_s = 0.1 # s
    h = 0.01
    duty = 0.6
    m_bar = np.array([0.002, 0.002, 0.002]) # A*m^2
    polarity = np.array([-1, -1, -1], dtype=int)
    I_min = float(np.min(np.diag(I)))
    omega_orbit = float(kep.n)
    xi_geomag = kep.i

    ctrl_cfg = BDotConfig(
        T_s=T_s,
        duty=duty,
        m_bar=m_bar,
        I_min=I_min,
        omega_orbit=omega_orbit,
        xi_geomag=xi_geomag,
        polarity=polarity,
    )

    ctrl = BDotController(cfg=ctrl_cfg)

    cfg = DetumbleConfig(
        T_s = T_s,
        h = h,
        Nw = int(1800.0 / T_s),
        omega_max = np.deg2rad(180.0),
        log_every_sample = True,
        w_stop_rad = np.deg2rad(2.0),
    )

    orbit_step = make_orbit_stepper()
    mags = None
    sim = DetumbleSim(orbit_step=orbit_step, attitude=dyn, field=field, mags=mags, ctrl=ctrl, cfg=cfg)

    return sim, x_orbit0

def to_seconds(t_array, t0):
    """Convert numpy datetime64 or datetime objects to seconds since t0."""
    if len(t_array) == 0:
        return np.array([])
    t0_ns = np.datetime64(t0).astype("datetime64[ns]").astype(np.int64)
    t_ns = np.array(t_array, dtype="datetime64[ns]").astype(np.int64)
    return (t_ns - t0_ns) / 1e9



#Please check this
def plot_results(res, t0):
    # Extract with fallbacks (supports either return dict or sim.log form)
    t_vals = res.get("t", [])
    w_B = res.get("w_B", [])
    b_norm = res.get("b_norm", [])
    m_cmd = res.get("m_cmd", [])
    t_on = res.get("t_on", [])
    m_des = res.get("m_des", [])

    r_eci = res.get("r_eci", [])
    v_eci = res.get("v_eci", [])
    r_norm = res.get("r_norm", [])
    v_norm = res.get("v_norm", [])


    # Convert time to seconds for x-axis
    try:
        t_sec = np.asarray(t_vals, dtype=float)
    except Exception:
        try:
            t_sec = to_seconds(np.asarray(t_vals), t0)
        except Exception:
            t_sec = np.asarray([], dtype=float)

    # Prepare arrays
    w_B = np.asarray(w_B, dtype=float)        # shape (N, 3)
    b_norm = np.asarray(b_norm, dtype=float)  # shape (N,)
    m_cmd = np.asarray(m_cmd, dtype=float)    # shape (N, 3)
    t_on = np.asarray(t_on, dtype=float)      # shape (N, 3)
    m_des = np.asarray(m_des, dtype=float)

    r_eci = np.asarray(r_eci, dtype=float) if len(r_eci) else np.asarray([], dtype=float)
    v_eci = np.asarray(v_eci, dtype=float) if len(v_eci) else np.asarray([], dtype=float)
    r_norm = np.asarray(r_norm, dtype=float) if len(r_norm) else np.asarray([], dtype=float)
    v_norm = np.asarray(v_norm, dtype=float) if len(v_norm) else np.asarray([], dtype=float)

    if r_eci.size and (r_norm.size == 0 or r_norm.shape[0] != r_eci.shape[0]):
        r_norm = np.linalg.norm(r_eci, axis=1)
    if v_eci.size and (v_norm.size == 0 or v_norm.shape[0] != v_eci.shape[0]):
        v_norm = np.linalg.norm(v_eci, axis=1)

    # 1) Angular-rate norm
    plt.figure()
    if w_B.size:
        w_norm = np.linalg.norm(w_B, axis=1)
        plt.plot(t_sec, w_norm)
        plt.xlabel("Time [s]")
        plt.ylabel("||ω_B|| [rad/s]")
        plt.title("Body Rate Magnitude vs Time")
        plt.grid(True)

    # 2) Magnetic field magnitude
    plt.figure()
    if b_norm.size:
        plt.plot(t_sec, b_norm)
        plt.xlabel("Time [s]")
        plt.ylabel("||B|| [T]")
        plt.title("Magnetic Field Magnitude vs Time")
        plt.grid(True)

    # 3) Commanded dipole & duty on-times
    plt.figure()
    if m_cmd.size:
        plt.plot(t_sec, m_cmd[:, 0], label="m_cmd_x")
        plt.plot(t_sec, m_cmd[:, 1], label="m_cmd_y")
        plt.plot(t_sec, m_cmd[:, 2], label="m_cmd_z")
        plt.xlabel("Time [s]")
        plt.ylabel("Commanded Dipole [A·m²]")
        plt.title("Commanded Dipole vs Time")
        plt.grid(True)
        plt.legend()

    # 4) time on
    plt.figure()
    if t_on.size:
        plt.plot(t_sec, t_on[:, 0], label="t_on_x")
        plt.plot(t_sec, t_on[:, 1], label="t_on_y")
        plt.plot(t_sec, t_on[:, 2], label="t_on_z")
        plt.xlabel("Time [s]")
        plt.ylabel("On-time per sample [s]")
        plt.title("Duty On-time vs Time")
        plt.grid(True)
        plt.legend()

    # 5–7) Individual ω_B component plots
    if w_B.size:
        # Ensure correct shape
        if w_B.ndim != 2 or w_B.shape[1] != 3:
            raise ValueError(f"w_B expected shape (N,3), got {w_B.shape}")

        # Time alignment
        N = w_B.shape[0]
        t_plot = t_sec[:N] if t_sec.size >= N else np.pad(t_sec, (0, N - t_sec.size), 'edge')

        # X-axis
        plt.figure()
        plt.plot(t_plot, w_B[:, 0])
        plt.xlabel("Time [s]")
        plt.ylabel("ω_Bx [rad/s]")
        plt.title("Body Rate ω_Bx vs Time")
        plt.grid(True)

        # Y-axis
        plt.figure()
        plt.plot(t_plot, w_B[:, 1])
        plt.xlabel("Time [s]")
        plt.ylabel("ω_By [rad/s]")
        plt.title("Body Rate ω_By vs Time")
        plt.grid(True)

        # Z-axis
        plt.figure()
        plt.plot(t_plot, w_B[:, 2])
        plt.xlabel("Time [s]")
        plt.ylabel("ω_Bz [rad/s]")
        plt.title("Body Rate ω_Bz vs Time")
        plt.grid(True)
    
    if r_eci.size:
        if r_eci.ndim != 2 or r_eci.shape[1] != 3:
            raise ValueError(f"r_eci expected shape (N,3), got {r_eci.shape}")
        Nr = r_eci.shape[0]
        t_plot_r = t_sec[:Nr] if t_sec.size >= Nr else np.pad(t_sec, (0, Nr - t_sec.size), 'edge')

        plt.figure()
        plt.plot(t_plot_r, r_eci[:, 0], label="r_x (ECI)")
        plt.plot(t_plot_r, r_eci[:, 1], label="r_y (ECI)")
        plt.plot(t_plot_r, r_eci[:, 2], label="r_z (ECI)")
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")
        plt.title("ECI Position Components vs Time")
        plt.grid(True)
        plt.legend()

    # 9) ECI Velocity components v_x, v_y, v_z
    if v_eci.size:
        if v_eci.ndim != 2 or v_eci.shape[1] != 3:
            raise ValueError(f"v_eci expected shape (N,3), got {v_eci.shape}")
        Nv = v_eci.shape[0]
        t_plot_v = t_sec[:Nv] if t_sec.size >= Nv else np.pad(t_sec, (0, Nv - t_sec.size), 'edge')

        plt.figure()
        plt.plot(t_plot_v, v_eci[:, 0], label="v_x (ECI)")
        plt.plot(t_plot_v, v_eci[:, 1], label="v_y (ECI)")
        plt.plot(t_plot_v, v_eci[:, 2], label="v_z (ECI)")
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [m/s]")
        plt.title("ECI Velocity Components vs Time")
        plt.grid(True)
        plt.legend()

    # 10) Magnitudes |r| and |v|
    if r_norm.size or v_norm.size:
        # align times based on whichever exists
        Nmag = max(r_norm.shape[0] if r_norm.size else 0, v_norm.shape[0] if v_norm.size else 0)
        if Nmag:
            t_plot_mag = t_sec[:Nmag] if t_sec.size >= Nmag else np.pad(t_sec, (0, Nmag - t_sec.size), 'edge')

            plt.figure()
            if r_norm.size:
                plt.plot(t_plot_mag[:r_norm.shape[0]], r_norm, label="|r| (ECI)")
            if v_norm.size:
                plt.plot(t_plot_mag[:v_norm.shape[0]], v_norm, label="|v| (ECI)")
            plt.xlabel("Time [s]")
            plt.ylabel("Magnitude")
            plt.title("ECI Magnitudes |r| and |v| vs Time")
            plt.grid(True)
            plt.legend()

            # Small annotation clarifying ECI axes (useful in reviews)
            ax = plt.gca()
            txt = (
                "ECI frame: Earth-centered, non-rotating.\n"
                "Z along Earth's spin axis; X to vernal equinox; Y completes right hand."
            )
            ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9,
                    verticalalignment="bottom", horizontalalignment="left",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    plt.show()


def run():#

    aware = datetime.now(timezone.utc)
    t0 = aware.replace(tzinfo=None)

    sim, x_orbit0 = build_sim()

    #sim_duration = 5 * 60
    #sim_duration =  8 * 1.53 * 60 * 60.0
    #sim_duration = 4 * 60 * 60 #seconds 
    #sim_duration = 10000
    sim_duration = 18.5 * 1.53 * 3600.0 # n orbits? i think

    

    t_final = t0 + timedelta(seconds=sim_duration)
 
    result = sim.run(t0=t0,t_final=t_final,x_orbit0=x_orbit0)

    res = result if isinstance(result, dict) else getattr(sim, "log", {})

    plot_results(res, t0)

    t_arr = np.asarray(res.get("t", []), dtype=float)
    if t_arr.size:
        print("n_samples:", len(t_arr))
        print("t[0] [s]:", float(t_arr[0]), "  t[-1] [s]:", float(t_arr[-1]))
        print("simulated_span_seconds:", float(t_arr[-1] - t_arr[0]))


    
