import numpy as np
from tle_io.keplerelement import KeplerElements

def kepler_solve_E(M, e):
    """newton's method for solving inverse kepler equation"""
    #https://en.wikipedia.org/wiki/Kepler%27s_equation#Numerical_approximation_of_inverse_problem
    
    tol = 1e-12
    i_max = 50

    M = np.mod(M, 2*np.pi)


    if e < 0.8:
        E = M
    else:
        E = np.pi

    for i in range(i_max):

        f = E - e*np.sin(E) - M
        fp = 1 - e*np.cos(E)
        dE = -f / fp
        E += dE

        if abs(dE) < tol: # too much accuracy
            break
    return E





class KeplerToRV:


    def rv_eci(self, el):
        i = el.i
        raan = el.raan
        e = el.e
        w = el.w
        M = el.M
        #n = el.n           # this n or n calculated below???
        a = el.a

        n = np.sqrt(KeplerElements.MU_E / a**3)
    
        E = kepler_solve_E(M, e) 
        cosE = np.cos(E)
        sinE = np.sin(E)
        r = a * (1 - e*cosE)


        # position and velocity in perifocal frame
        r_pf = a * np.array(
            [cosE - e,
             np.sqrt(1 - e**2) * sinE,
             0.0]
        )

        v_pf = ((n * a**2) / r) * np.array(
            [-sinE,
            np.sqrt(1-e**2)*cosE,
            0.0]
        )

        # crassidis page 379-380
        R = np.array([
            [np.cos(raan)*np.cos(w) - np.sin(raan)*np.sin(w)*np.cos(i),
             -np.cos(raan)*np.sin(w) - np.sin(raan)*np.cos(w)*np.cos(i),
             np.sin(raan)*np.sin(i)],

            [np.sin(raan)*np.cos(w) + np.cos(raan)*np.sin(w)*np.cos(i),
             -np.sin(raan)*np.sin(w) + np.cos(raan)*np.cos(w)*np.cos(i),
             -np.cos(raan)*np.sin(i)],

            [np.sin(w)*np.sin(i),
             np.cos(w)*np.sin(i),
             np.cos(i)]
        ])

        r_eci = R @ r_pf
        v_eci = R @ v_pf

        return r_eci, v_eci


        



