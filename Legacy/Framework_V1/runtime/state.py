from dataclasses import dataclass

@dataclass
class SimState:
    t: float = 0.0
    x_frost: float = 0.0