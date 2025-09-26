from dataclasses import dataclass

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