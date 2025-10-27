from dataclasses import dataclass

@dataclass(frozen=True)
class CaseConfig:
    # air data
    T_a: float          # °C temperature air
    v_a: float          # m/s velocity air
    p_a: float          # Pa pressure air
    RH: float           # relative humidity air
    w_amb: float
    rho_amb: float
    v_kin: float
    lam: float
    c_p_a: float
    D_std:  float
    C: float
    isv: float

    # refrigerant data
    T_w: float

    # ice data
    rho_i: float
    h_sub: float

    # numerics
    t_end: float
    dt: float

