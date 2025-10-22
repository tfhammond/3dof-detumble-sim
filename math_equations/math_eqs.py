import numpy as np

def quat_multiply(q, p):
    """ x = q x p """

    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = p

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z])

def quat_conjugate(q):
    """ q* """
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def w_matrix(w):
    """
    making the "skew symmetric matrix" OMEGA(w) required for q_dot = 1/2 * OMEGA(w) * q
    """

    wx, wy, wz = w

    w_array = np.array([[0.0, -wx, -wy, -wz],
                         [wx, 0.0, wz, -wy],
                         [wy, -wz, 0.0, wx],
                         [wz, wy, -wx, 0.0]])
    
    return w_array

def normalize_quat(q):
    n = np.linalg.norm(q)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0]) # default to no rotatio
        #return q
    return q / n

def quat_rate_scalar_first(q, w):
    """
    Quaternion derivative q̇ = 0.5 * Ω(ω) q for scalar-first q = [q0, q1, q2, q3].
    ω = [p, q, r] in body axes.
    """
    q0, q1, q2, q3 = q
    p, q_, r = w
    return 0.5 * np.array([
        -(p*q1 + q_*q2 + r*q3),
         p*q0 + r*q2 - q_*q3,
         q_*q0 - r*q1 + p*q3,
         r*q0 + q_*q1 - p*q2
    ])

