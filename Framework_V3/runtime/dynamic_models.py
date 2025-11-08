from math import sin, pi

def _clamp(v: float, V_MIN: float, V_MAX: float) -> float:
    if V_MIN is not None:
        v = max(V_MIN, v)
    if V_MAX is not None:
        v = min(V_MAX, v)
    return v

def velocity(t: float) -> float:
    v = 4.46e-11*t**3 - 3.0e-7*t**2 + 6.34e-4*t + 1.2
    return v