from dataclasses import dataclass
import numpy as np

@dataclass
class SimState:
    t: float = 0.0
    fx: float = 0.0