import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import ppigrf

from math_equations.math_eqs import quat_conjugate, quat_multiply, normalize_quat


# IERS conventional Earth rotation rate (rad/s)
EARTH_OMEGA = 7.2921150e-5
TWOPI = 2.0 * np.pi

class GMSTTracker:
    """
    Keep GMST updated cheaply between known UTC-naive datetimes.
    Assumes date inputs are naive UTC (tzinfo is None).
    """
    def __init__(self, t0_utc_naive, gmst0_rad):
        self.t_ref = t0_utc_naive
        self.theta_ref = gmst0_rad  # gmst_angle_rad(t0_utc_naive)

    def theta(self, t_utc_naive):
        # Fast: linear update from the reference time
        dt = (t_utc_naive - self.t_ref).total_seconds()
        return (self.theta_ref + EARTH_OMEGA * dt) % TWOPI

    def reset(self, t_utc_naive, gmst_exact_rad):
        # Occasionally recompute exact GMST to bound any drift
        self.t_ref = t_utc_naive
        self.theta_ref = gmst_exact_rad

def utc_to_julian_date(dt_utc):
    """
    UTC datetime to a Julian Date.
    """

    year = dt_utc.year
    month = dt_utc.month
    day = dt_utc.day
    hour = dt_utc.hour
    minute = dt_utc.minute
    second = dt_utc.second
    #microsecond = dt_utc.microsecond #you dont need this cmon

    if month <= 2:
        year -= 1
        month += 12

    A = int(year / 100)
    B = 2 - A + int(A / 4)

    julian = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5

    # Add the fractional part for the time of day
    fractional_day = (hour + minute / 60 + second / 3600) / 24.0
    
    return julian + fractional_day

def gmst_angle_rad(date):

    julian = utc_to_julian_date(date)
    
    T = (julian - 2451545.0) / 36525

    theta_deg = (280.46061837
                 + 360.98564736629 * (julian - 2451545.0)
                 + 0.000387933 * T**2
                 - (T**3) / 38710000.0)

    return np.deg2rad(theta_deg % 360.0) #to rad




def about_z(theta):
    """rotation about z axis by theta. used in eci to ecef and ecef to eci"""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c,   -s, 0.0  ],
                     [s,    c,  0.0 ],
                     [0.0, 0.0, 1.0]])



def eci_to_ecef(state, theta):
    """ positive z"""
    R = about_z(theta)
    return R @ state

def ecef_to_eci(state, theta):
    """ negative z"""
    R = about_z(theta)
    return R.T @ state

def ecef_to_spherical(ecef): #returns r km,theta deg,phi deg

    x,y,z = ecef
    r = float(np.linalg.norm(ecef))
    if r == 0.0:
        return 0.0,0.0,0.0

    z_r = np.clip(z / r, -1.0, 1.0)
    theta_rad = np.arccos(z_r)
    phi_rad = np.arctan2(y, x)

    r_km = r / 1000.0

    theta = np.degrees(theta_rad)
    phi = np.degrees(phi_rad)

    return r_km, theta, phi

def spherical_to_ecef(b_r, b_theta, b_phi, theta, phi): #theta and phi in deg

   #(X = r * cos(lat) * cos(lon), Y = r * cos(lat) * sin(lon), and Z = r * sin(lat) I think)

    # https://en.wikipedia.org/wiki/Spherical_coordinate_system#Integration_and_differentiation_in_spherical_coordinates

    th = np.radians(theta)
    ph = np.radians(phi)
    sth, cth = np.sin(th), np.cos(th)
    sph, cph = np.sin(ph), np.cos(ph)

    # Unit vectors at (θ, φ)
    e_r     = np.array([ sth*cph,  sth*sph,  cth ], dtype=float)
    e_theta = np.array([ cth*cph,  cth*sph, -sth ], dtype=float)
    e_phi   = np.array([-sph,      cph,      0.0 ], dtype=float)

    b_ecef = b_r * e_r + b_theta * e_theta + b_phi * e_phi

    return b_ecef

def rotate_eci(q_ib, v_eci):

    q = normalize_quat(q_ib)

    v_q = np.array([0.0, v_eci[0], v_eci[1], v_eci[2]])

    q_conjugate = quat_conjugate(q)

    v_b = quat_multiply(quat_multiply(q, v_q), q_conjugate)

    return v_b[1:]

def eci_to_body(q_IB, v_eci):
    # q_IB is Body→Inertial stored in the state; invert to get Inertial→Body
    return rotate_eci(quat_conjugate(q_IB), v_eci)

    
class MagneticFieldModel:
    """ eci to ecef at when 

        ECI position → ECEF position → geocentric spherical coords → IGRF (spherical components) → ECEF vector → ECI vector → Body vector ?

        r_eci in meters
    """

    def __init__(self, gmsttracker):
        self._gmst = gmsttracker

    def b_eci(self, r_eci, date):
        """
        1. ECI -> ECEF using GMST(when).
        2. Convert r_ecef to (r_km, theta_deg, phi_deg).
        3. Call ppigrf.igrf_gc to get (Br, Btheta, Bphi).
        4. Convert spherical components to ECEF Cartesian.
        5. ECEF -> ECI return b_eci.
        """
        
        assert date.tzinfo is None, "b_eci expects a native UTC datetime"


        #theta = gmst_angle_rad(date)
        theta = self._gmst.theta(date)


        r_ecef = eci_to_ecef(r_eci, theta)

        r, theta_deg, phi_deg = ecef_to_spherical(r_ecef)
        if r <= 0.0:
            return np.zeros(3)
        
        #make sure I call this 
        #date_naive = date.astimezone(timezone.utc).replace(tzinfo=None) 
        
        b_r, b_theta, b_phi = ppigrf.igrf_gc(r,theta_deg,phi_deg, date)

        b_r *= 1e-9
        b_theta *= 1e-9
        b_phi *= 1e-9

        b_ecef_T = spherical_to_ecef(b_r, b_theta, b_phi, theta_deg, phi_deg)

        b_eci_T = ecef_to_eci(b_ecef_T, theta)

        return b_eci_T
    
    def b_body(self, r_eci, q_ib, date):

        b_eci_T = self.b_eci(r_eci, date)

        b_b = rotate_eci(q_ib, b_eci_T)

        return b_b



