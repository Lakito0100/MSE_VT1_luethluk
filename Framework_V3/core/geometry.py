import math
from dataclasses import dataclass

@dataclass(frozen=True)
class FlatPlate:
    L: float            # Charakteristische Länge

@dataclass(frozen=True)
class FinnTubedHX:
    n_fin: float
    l_fin: float
    fin_thickness: float
    fin_pitch: float
    d_tube_a: float
    tube_thickness: float

    def l_rohr(self):
        return self.n_fin*(self.fin_pitch + self.fin_thickness)

    def A(self):
        A_rohr_aussen = self.d_tube_a * math.pi * self.fin_thickness / 2
        A_rippen = (self.l_fin * self.l_fin - self.d_tube_a * self.d_tube_a * math.pi) * 2
        return (A_rohr_aussen + A_rippen) * (self.n_fin - 1)

    def d_rohr_i(self):
        return self.d_tube_a - 2 * self.tube_thickness

    def phi_s(self):
        lr_uber_br = 1  # Verhältnis der Breite und Höhe der Rippen
        return 1.28 * (self.l_fin / self.d_tube_a) * math.sqrt(lr_uber_br - 0.2)

    def phi(self):
        return (self.phi_s() - 1) * (1 + 0.35 * math.log(self.phi_s()))

    def x_rippe(self):
        gew_hoehe = self.phi() * self.d_tube_a / 2
