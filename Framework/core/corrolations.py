import math
from dataclasses import dataclass

# Dimensionless numbers
@dataclass(frozen=True)
class DK:
    """Dimensionless-number helper formulas."""
    @staticmethod
    def Re(u: float, l: float, kv: float) -> float:
        """Return Reynolds number."""
        return u * l / kv

    @staticmethod
    def Pr(kv: float, lam: float, c_p: float, rho: float) -> float:
        """Return Prandtl number."""
        a = lam / (c_p * rho)
        return kv / a

    @staticmethod
    def Nu(alpha: float, l: float, lam: float) -> float:
        """Return Nusselt number."""
        return alpha * l / lam


@dataclass(frozen=True)
class CorLammellenRohrWU:
    """Correlation for finned-tube wall-to-fluid heat transfer."""
    @staticmethod
    def k(geom, alpha_s: float, alpha_i: float, lam_g: float) -> float:
        """Return overall heat-transfer coefficient k, related to the air-side area of one segment."""
        s_rohr = (geom.d_tube_a - geom.d_rohr_i()) / (2 * lam_g)
        w_leitung_i_a = 1/alpha_i + s_rohr
        A_a = geom.A_one_segment()                                        # air side
        A_i = geom.d_rohr_i() * math.pi * geom.l_tube() * geom.n_fin      # refrigerant side
        return (1/alpha_s + A_a/A_i * w_leitung_i_a)**(-1)
