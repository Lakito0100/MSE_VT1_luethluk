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