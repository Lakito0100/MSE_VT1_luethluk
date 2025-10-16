from Framework_V1.core.config import CaseConfig
from Framework_V1.core.geometry import FlatPlate
from Framework_V1.models import frost_test
from Framework_V1.runtime.simulator import Simulator
from Framework_V1.visualisation import plot

cfg = CaseConfig(
    T_a = 10.0,         # °C
    T_w = -10.0,        # °C
    RH = 0.8,           # -
    p_atm = 100000,     # Pa
    t_end = 10*60,      # s
    dt = 0.1,           # s
    test_ceof = 3e-4   # m/(s*K)
)

geom = FlatPlate(
    L = 0.1             # m
)

sim = Simulator()

results = sim.run(cfg, frost_test)
plot.plot_frostdicke(results)