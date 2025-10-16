from dataclasses import dataclass

@dataclass(frozen=True)
class CaseConfig:
    T_a: float
    T_w: float
    RH: float
    p_atm: float
    t_end: float
    dt: float
    test_ceof: float
