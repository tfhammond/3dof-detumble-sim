from tle_io.keplerelement import KeplerElements
import numpy as np

DEG = np.pi/180


# TODO Write tests for parse
class TLEConverter:

    def parse(lines):
        l2 = lines[1]
        i_deg = float(l2[8:16])
        raan_deg = float(l2[17:25]) # Right Ascension of Ascending Node
        eccentricity = l2[26:33] 
        w_deg = float(l2[34:42]) # Argument of Perigee (lower omega)
        M_deg = float(l2[43:51])
        mmotion_rev = float(l2[52:63])

        i = i_deg * DEG
        raan = raan_deg * DEG
        e = float(f"0.{eccentricity}")
        w = w_deg * DEG 
        m = M_deg * DEG
        n = mmotion_rev * 2*np.pi / (24*3600.0)  # rad/s (DO I NEED N????)
        a = (KeplerElements.MU_E / (n**2))**(1/3)
        return KeplerElements(i=i, raan=raan, e=e, w=w, M=m, n=n, a=a)
