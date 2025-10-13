import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import ppigrf

from math_equations.math_eqs import quat_conjugate, quat_multiply, normalize_quat

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
    #fractional_day = (hour + minute / 60 + second / 3600 + microsecond / 3600000000) / 24.0
    
    return julian + fractional_day

def gmst_angle_rad(date):

    julian = utc_to_julian_date(date)
    
    T = (julian - 2451545.0) / 36525

    theta = (280.46061837 + 360.98564736629 * (julian - 2451545.0) + 0.000387933 * T**2 - (T**3) / 38710000.0)

    return np.radians(theta) #to rad




def about_z(theta):
    """rotation about z axis by theta. used in eci to ecef and ecef to eci"""

    return np.array([[ np.cos(theta), np.sin(theta), 0.0],
                     [-np.sin(theta), np.cos(theta), 0.0],
                     [0.0           , 0.0          , 1.0]])



def eci_to_ecef(state, theta):
    """ positive z"""
    return about_z(theta) @ state

def ecef_to_eci(state, theta):
    """ negative z"""
    return about_z(-theta) @ state

def ecef_to_spherical(ecef): #returns r km,theta deg,phi deg

    x,y,z = ecef
    x_km, y_km, z_km = x/1000, y/1000, z/1000 #km
    r_km = float(np.linalg.norm(ecef)) / 1000 #km
    
    #0 at north pole and 90 at equator? (second quatrant)
    
    theta_rad = np.arccos(np.clip(z_km / r_km, -1.0, 1.0))  #colatitude 
    phi_rad = np.arctan2(y_km,x_km) # degrees east (same as lon)

    theta = np.degrees(theta_rad)
    phi = np.degrees(phi_rad)

    return r_km, theta, phi

def spherical_to_ecef(b_r, b_theta, b_phi, theta, phi): #theta and phi in deg

   #(X = r * cos(lat) * cos(lon), Y = r * cos(lat) * sin(lon), and Z = r * sin(lat) I think)

    # https://en.wikipedia.org/wiki/Spherical_coordinate_system#Integration_and_differentiation_in_spherical_coordinates


    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)


    #unit vectors
    e_r = np.array([np.sin(theta_rad) * np.cos(phi_rad), np.sin(theta_rad) * np.sin(phi_rad), np.cos(theta_rad)])

    e_theta = np.array([np.cos(theta_rad) * np.cos(phi_rad), np.cos(theta_rad) * np.sin(phi_rad), -np.sin(theta_rad)])
                          
    e_phi = np.array([-np.sin(phi_rad), np.cos(phi_rad), 0.0])

    b_ecef = b_r * e_r + b_theta * e_theta + b_phi * e_phi

    return b_ecef

def rotate_eci(q_ib, v_eci):

    q = normalize_quat(q_ib)

    v_q = np.array([0.0, v_eci[0], v_eci[1], v_eci[2]])

    v_b = quat_multiply(quat_multiply(q,v_q), quat_conjugate(q))

    return v_b[1:]


    
class MagneticFieldModel:
    """ eci to ecef at when 

        ECI position → ECEF position → geocentric spherical coords → IGRF (spherical components) → ECEF vector → ECI vector → Body vector ?

        r_eci in meters
    """

    def b_eci(self, r_eci, date):
        """
        1. ECI -> ECEF using GMST(when).
        2. Convert r_ecef to (r_km, theta_deg, phi_deg).
        3. Call ppigrf.igrf_gc to get (Br, Btheta, Bphi).
        4. Convert spherical components to ECEF Cartesian.
        5. ECEF -> ECI return b_eci.
        """
        
        theta = gmst_angle_rad(date)


        r_ecef = eci_to_ecef(r_eci, theta)

        r, theta_deg, phi_deg = ecef_to_spherical(r_ecef)
        if r <= 0.0:
            return np.zeros(3)
        
        date_naive = date.astimezone(timezone.utc).replace(tzinfo=None)
        
        b_r, b_theta, b_phi = ppigrf.igrf_gc(r,theta_deg,phi_deg, date_naive)

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



