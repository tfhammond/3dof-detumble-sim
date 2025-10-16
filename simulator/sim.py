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
from magnetic_field.model import MagneticFieldModel


iss_txt_url = "https://live.ariss.org/iss.txt"

def load_kepler_from_tle():

    #tle = "1 25544U 98067A   25281.52879516  .00013673  00000-0  24910-3 0  9990\n2 25544  51.6312 105.3299 0000824 223.8777 136.2147 15.49772654532744"

    tle = "1 25544U 98067A   19178.82735530  .00002515  00000-0  49918-4 0  9997\n2 25544  51.6428 308.4904 0008116  89.4883  70.1063 15.51247238176900"

    # TLE 
    #iss = requests.get(iss_txt_url)
    #load = TLELoader.read_lines(iss.text)
    load = TLELoader.read_lines(tle)
    # keplerele object
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

    #attitude & inertia (ask for the values cuz idk them)
    #I = np.diag([0.001731, 0.001726, 0.000264]) # kg*m^2
    I = np.diag([1.731e-3, 1.726e-3, 0.264e-3])
    #I = np.diag([0.001731, 0.001726, 0.000264])     # kg·m^2 (example)
    q_IB0 = np.array([1.0, 0.0, 0.0, 0.0]) # scalar-first quaternion
    # w0_deg_s = np.array([8.0, -6.0, 10.0]) # initial body rates in deg/s (example)
    # w_B0 = np.deg2rad(w0_deg_s)            # rad/s

    #w_B0 = np.deg2rad(np.array([180.0, 180.0, 180.0]))  # rad/s (fast tumbling start) from paper

    w_B0 = np.deg2rad(np.array([45.0, 45.0, 45.0]))

    dyn = Dynamics(I=I, q_IB=q_IB0, w_B=w_B0)

    #field model
    field = MagneticFieldModel()

    #controller

    #T_s = 0.1 
    T_s = 0.1 # s
    h = 0.05 #still deciding if I need this
    duty = 0.6
    m_bar = np.array([0.002, 0.002, 0.002]) # A*m^2
    #m_bar = np.array([0.15, 0.15, 0.15])
    polarity = np.array([-1, 1, -1], dtype=int)
    I_min = float(np.min(np.diag(I)))
    omega_orbit = float(kep.n) #probably should use this but lets let it cook
    #omega_orbit = 0.0011407380731989082
    #xi_geomag = np.deg2rad(60.0)
    xi_geomag = kep.i
    print(xi_geomag)


    p_bar = 8.125e-3
    phi = 4.0 / T_s
    eps = 1.0 - np.sqrt(3.0 * phi * p_bar)

    # eps = 0.1
    # phi = 2

    alpha = 1.0 / 200.0

    ctrl_cfg = BDotConfig(
        T_s=T_s,
        duty=duty,
        m_bar=m_bar,
        I_min=I_min,
        omega_orbit=omega_orbit,
        xi_geomag=xi_geomag,
        phi=phi,
        eps=eps,
        alpha=alpha,
        polarity=polarity,
        # prev_b_hat=None,
        # prev_p=0.0,
        # prev_pv=None,
        # prev_p=np.sqrt(3.0) * T_s,
        # prev_pv=np.array([T_s, T_s, T_s]),
        keep_vector_p=True,
    )

    ctrl = BDotController(cfg=ctrl_cfg)

    cfg = DetumbleConfig(
        T_s = T_s,
        h = h,
        p_bar = p_bar,
        Nw = int(1800.0 / T_s),
        omega_max = np.deg2rad(180.0),
        log_every_sample = True,
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
    p = res.get("p", [])
    pv = res.get("pv", [])
    b_norm = res.get("b_norm", [])
    k_bdot = res.get("k_bdot", [])
    m_cmd = res.get("m_cmd", [])
    t_on = res.get("t_on", [])
    m_des = res.get("m_des", [])

    # Convert time to seconds for x-axis
    try:
        t_sec = to_seconds(np.asarray(t_vals), t0)
    except Exception:
        # If t is already float seconds
        t_sec = np.asarray(t_vals, dtype=float)

    # Prepare arrays
    w_B = np.asarray(w_B, dtype=float)        # shape (N, 3)
    p = np.asarray(p, dtype=float)            # shape (N,)
    pv = np.asarray(pv, dtype=float)          # shape (N, 3)
    b_norm = np.asarray(b_norm, dtype=float)  # shape (N,)
    k_bdot = np.asarray(k_bdot, dtype=float)  # shape (N,)
    m_cmd = np.asarray(m_cmd, dtype=float)    # shape (N, 3)
    t_on = np.asarray(t_on, dtype=float)      # shape (N, 3)
    m_des = np.asarray(m_des, dtype=float)

    # 1) Angular-rate norm
    plt.figure()
    if w_B.size:
        w_norm = np.linalg.norm(w_B, axis=1)
        plt.plot(t_sec, w_norm)
        plt.xlabel("Time [s]")
        plt.ylabel("||ω_B|| [rad/s]")
        plt.title("Body Rate Magnitude vs Time")
        plt.grid(True)

    # 2) Weighted gain and p
    plt.figure()
    if k_bdot.size:
        plt.plot(t_sec, k_bdot)
        plt.xlabel("Time [s]")
        plt.ylabel("k_bdot")
        plt.title("Weighted B-dot Gain vs Time")
        plt.grid(True)

    plt.figure()
    if p.size:
        plt.plot(t_sec, p)
        plt.xlabel("Time [s]")
        plt.ylabel("p (scalar)")
        plt.title("Scalar p vs Time")
        plt.grid(True)

    # 3) p_v components
    plt.figure()
    if pv.size:
        plt.plot(t_sec, pv[:, 0], label="p_vx")
        plt.plot(t_sec, pv[:, 1], label="p_vy")
        plt.plot(t_sec, pv[:, 2], label="p_vz")
        plt.xlabel("Time [s]")
        plt.ylabel("p_v components")
        plt.title("Vector p_v vs Time")
        plt.grid(True)
        plt.legend()

    # 4) Magnetic field magnitude
    plt.figure()
    if b_norm.size:
        plt.plot(t_sec, b_norm)
        plt.xlabel("Time [s]")
        plt.ylabel("||B|| [T]")
        plt.title("Magnetic Field Magnitude vs Time")
        plt.grid(True)

    # 5) Commanded dipole & duty on-times
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

    plt.figure()
    if m_des.size:
        plt.plot(t_sec, m_des[:, 0], label="m_des_x")
        plt.plot(t_sec, m_des[:, 1], label="m_des_y")
        plt.plot(t_sec, m_des[:, 2], label="m_des_z")
        plt.xlabel("Time [s]")
        plt.ylabel("Commanded Dipole [A·m²]")
        plt.title("Commanded Dipole vs Time")
        plt.grid(True)
        plt.legend()

    plt.show()


def run():#

    sim, x_orbit0 = build_sim()

    t0 = datetime.now(timezone.utc)

    #sim_duration = 2 * 60
    #sim_duration =  2 * 1.53 * 60 * 60.0
    #sim_duration = 2.5 * 60 * 60 #seconds 
    sim_duration = 30
    #sim_duration = 12.5 * 1.53 * 3600.0 # n orbits? i think
    t_final = t0 + timedelta(seconds=sim_duration)
 
    result = sim.run(t0=t0,t_final=t_final,x_orbit0=x_orbit0)

    res = result if isinstance(result, dict) else getattr(sim, "log", {})

    plot_results(res, t0)

    print("n_samples:", len(res["t"]))
    print("t[0]:", res["t"][0], "  t[-1]:", res["t"][-1])
    print("unique_seconds_spanned:", (np.asarray(res["t"], dtype="datetime64[ns]").astype("int64")[-1] -
                                    np.asarray(res["t"], dtype="datetime64[ns]").astype("int64")[0]) / 1e9)



    
