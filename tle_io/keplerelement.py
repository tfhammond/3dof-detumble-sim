from dataclasses import dataclass

import numpy as np

@dataclass
class KeplerElements: 

    i: float # inclination
    raan: float # right ascension of ascending node
    e: float # eccentricity
    w: float # argument of periapsis
    M: float # mean anomaly

    n: float # mean motion (do i need n in here????)
    
    a: float # semi-major axis
    MU_E = 3.986004418e14 # m^3/s^2 standard gravitational parameter for Earth

    def __str__(self):
        return (
            f"Kepler Elements:\n"
            f"  Inclination (i): {self.i:.6f} rad\n"
            f"  RAAN (Ω): {self.raan:.6f} rad\n"
            f"  Eccentricity (e): {self.e:.6f}\n"
            f"  Argument of Periapsis (ω): {self.w:.6f} rad\n"
            f"  Mean Anomaly (M): {self.M:.6f} rad\n"
            f"  Mean Motion (n): {self.n:.6e} rad/s\n"
            f"  Semi-major Axis (a): {self.a:.3f} m"
        )