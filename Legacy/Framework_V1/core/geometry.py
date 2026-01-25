import math
from dataclasses import dataclass

@dataclass(frozen=True)
class FlatPlate:
    L: float            # Charakteristische Länge

@dataclass(frozen=True)
class LammellenRohrWU:
    n_rippen: float
    l_rippen: float
    t_rippen: float
    rippen_dicke: float
    d_rohr_a: float
    rohr_dicke: float

    def l_rohr(self):
        return self.n_rippen*(self.rippen_dicke + self.t_rippen)

    def A(self):
        A_rohr_aussen = self.d_rohr_a*math.pi*self.t_rippen
        A_rippen = (self.l_rippen*self.l_rippen - self.d_rohr_a*self.d_rohr_a*math.pi) * 2
        return (A_rohr_aussen + A_rippen) * (self.n_rippen - 1)

    def d_rohr_i(self):
        return self.d_rohr_a - 2 * self.rohr_dicke

    def phi_s(self):
        lr_uber_br = 1  # Verhältnis der Breite und Höhe der Rippen
        return 1.28 * (self.l_rippen/self.d_rohr_a) * math.sqrt(lr_uber_br - 0.2)

    def phi(self):
        return (self.phi_s() - 1) * (1 + 0.35 * math.log(self.phi_s()))

    def x_rippe(self):
        gew_hoehe = self.phi() * self.d_rohr_a/2
