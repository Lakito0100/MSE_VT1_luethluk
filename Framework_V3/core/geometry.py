import math
from dataclasses import dataclass

@dataclass(frozen=True)
class FlatPlate:
    L: float            # Charakteristische Länge

@dataclass(frozen=True)
class FinnTubedHX:
    n_fin: float
    l_fin: float
    h_fin: float
    fin_thickness: float
    fin_pitch: float
    d_tube_a: float
    tube_thickness: float

    def l_rohr(self):
        return self.n_fin*(self.fin_pitch + self.fin_thickness)

    def A_rohr_one_segment(self):
        return self.d_tube_a * math.pi * self.fin_pitch

    def A_fin_one_segment(self):
        return 2.0*(self.l_fin*self.h_fin - (self.d_tube_a**2)/4.0)

    def d_rohr_i(self):
        return self.d_tube_a - 2 * self.tube_thickness

    def phi(self):
        lr_uber_br = self.l_fin/self.h_fin
        phi_s = 1.28 * (self.l_fin / self.d_tube_a) * math.sqrt(lr_uber_br - 0.2)
        return (phi_s - 1) * (1 + 0.35 * math.log(phi_s))

    def x_rippe(self, alpha_f: float, lambda_f: float):
        gew_hoehe = self.phi() * self.d_tube_a / 2
        root = math.sqrt(2*alpha_f/(lambda_f*self.fin_thickness))
        return gew_hoehe*root

    def mue_finn(self, alpha_f: float, lambda_f: float):
        x = self.x_rippe(alpha_f, lambda_f)
        return math.tanh(x)/x
