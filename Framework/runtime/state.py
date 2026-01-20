from dataclasses import dataclass, field
import numpy as np

@dataclass
class SimState:
    """Container for simulation state fields."""
    t: float = 0.0

    # Edge domain
    s_e: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    T_e: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    rho_e: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    rho_a: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    w_e: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))

    # Fin and tube domain
    s_ft: float = 0.0
    T_ft: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    rho_ft: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    w_ft: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
