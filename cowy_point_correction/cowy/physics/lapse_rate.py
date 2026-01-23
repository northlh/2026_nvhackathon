
import numpy as np

def _hypsometric_equation(p1, p2, t1, t2):
    """Hypsometric equation for thickness between two pressure levels."""
    R = 287.0   # J/(kg*K)
    g = 9.81    # m/s^2
    t_mean = (t1 + t2) / 2.0
    return (R * t_mean / g) * np.log(p1 / p2)

def compute_elr(p_surface, t_surface, p_upper, t_upper, z_surface=2.0):
    """
    Environmental Lapse Rate (K/m) between surface and upper level.
    """
    dz = _hypsometric_equation(p_surface, p_upper, t_surface, t_upper)
    return (t_upper - t_surface) / dz
