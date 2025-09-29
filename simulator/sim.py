import numpy as np
from tle_io.tleload import TLELoader
from tle_io.tleconverter import TLEConverter
from tle_io.keplerelement import KeplerElements
from kepler.kepler import KeplerToRV
from attitude_dynamics.dynamics import Dynamics
#from math_equations.math import quat_conjugate, quat_multiply, w_matrix, normalize_quat
import requests

iss_txt_url = "https://live.ariss.org/iss.txt"

def run():

    # TLE 
    iss = requests.get(iss_txt_url)
    load = TLELoader.read_lines(iss.text)

    # keplerele object
    kep = TLEConverter.parse(load)

    r0, v0 = KeplerToRV().rv_eci(kep)

    #print(r0)
    #print(v0)

    
