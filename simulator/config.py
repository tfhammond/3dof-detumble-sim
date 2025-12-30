import numpy as np
from dataclasses import dataclass

@dataclass
class DetumbleConfig:
    T_s : float # sampling time
    
    h: float #fast integrator

    Nw: int #Confirmation window length in samples for the stop rule (Eq. (34)).

    omega_max : float #worst case |w| rad/s

    log_every_sample : bool #log each control sample or not
    log_stride : int = 10  # log every Nth control sample (1 = log all)

    w_stop_rad : float = np.deg2rad(2.0)
