from dataclasses import dataclass

@dataclass(frozen=True)
class FlatPlate:
    L: float            # Charakteristische Länge

@dataclass(frozen=True)
class FinnedTubeHE:
    rows: float         # Anzahl Reihen
    columns: float      # Anzahl Zeilen
    d_l: float          # Distanz zw. Lamellen