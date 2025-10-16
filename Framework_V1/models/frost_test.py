from Framework_V1.core.config import CaseConfig
import math

def testmodell_fx_at_t(con: CaseConfig, t: float) -> float:
    k = (con.T_air - con.T_wall) * con.test_ceof
    if math.log(t) < 0:
        return 0.0
    return k * math.log(t)
