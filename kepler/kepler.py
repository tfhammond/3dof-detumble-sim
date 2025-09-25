import numpy as np

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

    MU_E = 3.986004418e14

    def rv_eci(self, el):
        i = el.i
        omega = el.omega
        e = el.e
        loweromega = el.loweromega
        M = el.M
        n = el.n
        a = el.a
    
        E = kepler_solve_E(M, e) 
        cosE = np.cos(E)
        sinE = np.sin(E)

        



