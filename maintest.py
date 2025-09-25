from tle_io.keplerelement import KeplerElements
from tle_io.tleconverter import TLEConverter
from tle_io.tleload import TLELoader
#import kepler.kepler
from pathlib import Path

import requests

iss_txt_url = "https://live.ariss.org/iss.txt"

def mainTest():

    iss = requests.get(iss_txt_url)
    load = TLELoader.read_lines(iss.text)

    print(load)
    
    kep =TLEConverter.parse(load)
    print(kep)



mainTest()