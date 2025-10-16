from tle_io.keplerelement import KeplerElements
import numpy as np

DEG = np.pi/180


# TODO Write tests for parse
class TLEConverter:

    # def parse(lines):
    #     l2 = lines[1]
    #     i_deg = float(l2[8:16])
    #     raan_deg = float(l2[17:25]) # Right Ascension of Ascending Node
    #     eccentricity = l2[26:33] 
    #     w_deg = float(l2[34:42]) # Argument of Perigee (lower omega)
    #     M_deg = float(l2[43:51])
    #     mmotion_rev = float(l2[52:63])

    #     i = i_deg * DEG
    #     raan = raan_deg * DEG
    #     e = float(f"0.{eccentricity}")
    #     w = w_deg * DEG 
    #     m = M_deg * DEG
    #     n = mmotion_rev * 2*np.pi / (24*3600.0)  # rad/s (DO I NEED N????)
    #     a = (KeplerElements.MU_E / (n**2))**(1/3)
    #     return KeplerElements(i=i, raan=raan, e=e, w=w, M=m, n=n, a=a)

    def parse(lines):
        """
        Parse a 2-line element set and return KeplerElements in SI & radians.
        Expects: lines[0] = line 1, lines[1] = line 2.
        """
        if len(lines) < 2:
            raise ValueError("Expected two TLE lines (line 1 and line 2).")
        l2 = lines[1].rstrip("\n")

        try:
            # Column slices per TLE spec (1-based -> Python 0-based half-open):
            # i: 9-16, Ω: 18-25, e: 27-33, ω: 35-42, M: 44-51, n(rev/day): 53-63
            i_deg        = float(l2[8:16].strip())
            raan_deg     = float(l2[17:25].strip())   # RAAN (Ω)
            ecc_str      = l2[26:33]                  # 7 digits, no decimal
            w_deg        = float(l2[34:42].strip())   # argument of perigee (ω)
            M_deg        = float(l2[43:51].strip())   # mean anomaly (M)
            mmotion_rev  = float(l2[52:63].strip())   # rev/day

        except Exception as ex:
            raise ValueError(f"Failed to parse TLE line 2: {ex}") from ex

        # Eccentricity: "ddddddd" -> 0.ddddddd
        e = float("0." + ecc_str.strip().zfill(7))

        # Radians
        i    = np.deg2rad(i_deg)
        raan = np.deg2rad(raan_deg)
        w    = np.deg2rad(w_deg)
        M    = np.deg2rad(M_deg)

        # Mean motion (rad/s)
        n = mmotion_rev * 2.0 * np.pi / 86400.0

        # Semi-major axis from n (consistent with KeplerElements.MU_E units: m^3/s^2)
        a = (KeplerElements.MU_E / (n**2)) ** (1.0 / 3.0)

        return KeplerElements(i=i, raan=raan, e=e, w=w, M=M, n=n, a=a)