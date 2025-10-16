from dataclasses import dataclass

# Dimensionslose Kennzahl
@dataclass(frozen=True)
class DK:
    def Re(self, u: float, l: float, kv: float) -> float:
        return u * l / kv
    def Pr(self, kv: float, lam: float, c_p: float, rho: float) -> float:
        a = lam / (c_p * rho)
        return kv / a
    def Nu(self, alpha: float, l: float, lam: float) -> float:
        return alpha * l / lam


@dataclass(frozen=True)
class CorLammellenRohrWU:
    def k(self,geom, alpha_s: float, alpha_i: float, lam_g: float) -> float:
        s_rohr = (geom.d_rohr_a - geom.d_rohr_i()) / (2 * lam_g)
        w_leitung_i_a = 1/alpha_i + s_rohr
        return (1/alpha_s + geom.A()/geom.A_i() * w_leitung_i_a)**(-1)