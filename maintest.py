import numpy as np

from tle_io.keplerelement import KeplerElements
from tle_io.tleconverter import TLEConverter
from tle_io.tleload import TLELoader

from attitude_dynamics.dynamics import Dynamics
#import kepler.kepler

from simulator.sim import run as simrun



import requests

iss_txt_url = "https://live.ariss.org/iss.txt"

def mainTest():

    
    simrun()



mainTest()