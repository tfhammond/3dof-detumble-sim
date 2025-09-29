from tle_io.keplerelement import KeplerElements
import numpy as np

DEG = np.pi/180


# TODO Write tests for parse
class TLEConverter:

    def parse(lines):
        l2 = lines[1]
        inclination_deg = float(l2[8:16])
        raan_deg = float(l2[17:25]) # Right Ascension of Ascending Node
        eccentricity = l2[26:33] 
        argp_deg = l2[34:42] # Argument of Perigee (lower omega)
        M_deg = float(l2[43:51])
        mmotion_rev = float(l2[52:63])

        i = inclination_deg * DEG
        raan = raan_deg*DEG
        e = float(f"0.{eccentricity}")
        w = argp_deg 
        m = M_deg*DEG
        n = mmotion_rev * 2*np.pi / (24*3600)  # rad/s (DO I NEED N????)
        a = (KeplerElements.MU_E / (n**2))**(1/3)
        return KeplerElements(i=i, w=w, e=e, M=m, n=n, a=a, raan=raan)
