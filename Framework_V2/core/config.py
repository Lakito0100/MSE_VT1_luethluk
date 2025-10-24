from dataclasses import dataclass

@dataclass(frozen=True)
class CaseConfig:
    # air data
    T_a: float
    v_a: float
    p_a: float
    RH: float
    rho_amb: float
    v_kin: float
    lam: float
    c_p_a: float
    D_std:  float
    C: float

    # refrigerant data
    T_w: float

    # ice data
    rho_i: float
    h_sub: float

    # numerics
    t_end: float
    dt: float

