from CoolProp.HumidAirProp import HAPropsSI
import math

def _hold_ramp_hold(t: float,
                    y0: float, y1: float,
                    t0: float, duration: float,
                    easing: str = "linear") -> float:
    """
    Piecewise profile: y=y0 for t<t0, then a ramp over 'duration', then y=y1 constant.
    easing: "linear" or "smoothstep" (C1-continuous, no slope kink)
    """
    if duration <= 0.0:
        return float(y0 if t < t0 else y1)

    if t <= t0:
        return float(y0)
    if t >= t0 + duration:
        return float(y1)

    s = (t - t0) / duration  # 0..1
    if easing == "linear":
        pass
    elif easing == "smoothstep":
        s = s * s * (3.0 - 2.0 * s)
    else:
        raise ValueError(f"Unknown easing='{easing}' (use 'linear' or 'smoothstep').")

    return float(y0 + (y1 - y0) * s)

def T_a_profile(t: float,
                T_before_C: float, T_after_C: float,
                t_switch: float = 120.0,
                ramp_duration_s: float = 60.0,
                easing: str = "linear") -> float:
    """Air temperature profile with a hold-ramp-hold transition."""
    return _hold_ramp_hold(t, T_before_C, T_after_C, t_switch, ramp_duration_s, easing)


def w_amb_profile(t: float,
                  T_a_C: float,
                  p_a_Pa: float,
                  RH_before: float, RH_after: float,
                  t_switch: float = 120.0,
                  ramp_duration_s: float = 60.0,
                  easing: str = "linear") -> float:
    """Ambient humidity ratio profile derived from RH hold-ramp-hold."""
    # RH profile (constant -> ramp -> constant)
    RH = _hold_ramp_hold(t, RH_before, RH_after, t_switch, ramp_duration_s, easing)
    # Small numerical safety clamp
    RH = max(0.0, min(0.999999, RH))
    w = HAPropsSI("W", "T", T_a_C + 273.15, "P", p_a_Pa, "R", RH)
    return float(w)
