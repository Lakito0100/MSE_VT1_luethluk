import math
from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class FlatPlate:
    L: float            # Charakteristische Länge

@dataclass(frozen=True)
class FinnTubedHX:
    n_seg_l: float          # Anzahl Reihen in Flussrichtung der Luft
    n_seg_r: float          # Anzahl Reihen in Flussrichtung des Kältemittel inlet
    stacks: int             # Anzahl der Stapel
    n_fin: float            # Number of fins per segment
    l_fin: float            # Length of the fins
    h_fin: float            # Height of the fins (usually h_fin=l_fin)
    fin_thickness: float    # Fin thickness
    fin_pitch_cc: float     # Fin pitch center to center
    d_tube_a: float         # Tube outer diameter
    tube_thickness: float   # Tube wall thickness
    lambda_fin: float       # Fin and tube heat conduction coefficient lambda
    rho_solid: float        # Fin and tube density
    c_solid: float          # Fin and tube heat capacity
    CP: str                 # Definition of the connection path

    def fin_gap(self):
        return self.fin_pitch_cc - self.fin_thickness

    def l_tube(self):
        return self.fin_pitch_cc + self.fin_thickness

    def A_tube_one_segment(self):
        return self.d_tube_a * math.pi * self.fin_gap() * self.n_fin

    def A_fin_one_segment(self):
        return 2.0*(self.l_fin*self.h_fin - ((self.d_tube_a**2)*math.pi)/4.0) * self.n_fin

    def A_one_segment(self):
        return self.A_tube_one_segment() + self.A_fin_one_segment()

    def d_rohr_i(self):
        return self.d_tube_a - 2 * self.tube_thickness

    def phi(self):
        lr_uber_br = self.l_fin/self.h_fin
        phi_s = 1.28 * (self.l_fin / self.d_tube_a) * math.sqrt(lr_uber_br - 0.2)
        return (phi_s - 1) * (1 + 0.35 * math.log(phi_s))

    def x_rippe(self, alpha_f: float):
        gew_hoehe = self.phi() * self.d_tube_a / 2
        root = math.sqrt(2*alpha_f/(self.lambda_fin*self.fin_thickness))
        return gew_hoehe*root

    def mue_fin(self, alpha_f: float):
        x = self.x_rippe(alpha_f)
        return math.tanh(x)/x

    def build_connection_path(self, variant: str = "row_serpentine") -> List[Tuple[int, int]]:
        """
        Erzeugt einen Connection Path (CP) über das Segment-Grid auf Basis von
        n_seg_l × n_seg_r.

        Indizes:
            (i_l, i_r) mit
            i_l in [0, n_seg_l - 1]   - Richtung Luftströmung
            i_r in [0, n_seg_r - 1]   - Richtung Kältemittelinlet

        Varianten
        ---------
        variant = "row_serpentine":
            Schlange zeilenweise, Start unten rechts.
            Für 5×5 ergibt das z.B.:
            (4,4),(4,3),(4,2),(4,1),(4,0),
            (3,0),(3,1),(3,2),(3,3),(3,4), ...

        variant = "col_serpentine":
            Schlange spaltenweise, Start unten rechts.
        """
        n_l = int(self.n_seg_l)
        n_r = int(self.n_seg_r)

        if n_l <= 0 or n_r <= 0:
            raise ValueError(f"n_seg_l und n_seg_r müssen > 0 sein (n_seg_l={n_l}, n_seg_r={n_r}).")

        variant = variant.lower()
        path: List[Tuple[int, int]] = []

        if variant in ("serpentine"):
            # Start oben rechts, zeilenweise Schlange nach links
            for row_idx_from_bottom, i_l in enumerate(range(n_l - 1, -1, -1)):
                if row_idx_from_bottom % 2 == 0:
                    # gerade "Zeile von unten": rechts -> links
                    col_range = range(n_r - 1, -1, -1)
                else:
                    # ungerade "Zeile von unten": links -> rechts
                    col_range = range(0, n_r)
                for i_r in col_range:
                    path.append((i_l, i_r))

        elif variant in ("serpentine_variant"):
            # Start unten rechts, spaltenweise Schlange nach links
            for col_idx_from_right, i_r in enumerate(range(n_r - 1, -1, -1)):
                if col_idx_from_right % 2 == 0:
                    # gerade "Spalte von rechts": unten -> oben
                    row_range = range(n_l - 1, -1, -1)
                else:
                    # ungerade "Spalte von rechts": oben -> unten
                    row_range = range(0, n_l)
                for i_l in row_range:
                    path.append((i_l, i_r))

        else:
            raise ValueError(f"Unbekannte variant='{variant}'. Unterstützt: "
                             f"'row_serpentine', 'col_serpentine'.")

        return path
