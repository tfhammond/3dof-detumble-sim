import numpy as np
from dataclasses import dataclass

@dataclass
class DetumbleConfig:
    T_s : float # sampling time
    
    h: float #fast integrator

    p_bar : float #stop for p_v (0.2 degrees I think for the sim?)

    Nw: int #Confirmation window length in samples for the stop rule (Eq. (34)).

    omega_max : float #worst case |w| rad/s

    log_every_sample : bool #log each control sample or not
