from Framework_V1.core.config import CaseConfig

def testmodell_fx_at_t(con: CaseConfig, t: float) -> float:
    k = (con.T_a - con.T_w) * con.test_ceoff
    return k * t
