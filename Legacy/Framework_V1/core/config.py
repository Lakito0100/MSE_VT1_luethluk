from dataclasses import dataclass

@dataclass(frozen=True)
class CaseConfig:
    T_air: float
    T_wall: float
    u_air: float
    RH: float
    p_atm: float
    t_end: float
    dt: float
    test_ceof: float
