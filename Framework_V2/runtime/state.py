from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class GridShape:
    nx: int = 100
    nr: int = 100
    ntheta: int = 90

@dataclass
class SimState:
    t: float = 0.0

    # edge domain
    s_e: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    T_e: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    rho_e: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    rho_a: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    w_e: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))

    # finn and tube domain
    s_ft: float = 0.0
    T_ft: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    rho_ft: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    w_ft: float = field(default_factory=lambda: np.zeros((0,), dtype=float))