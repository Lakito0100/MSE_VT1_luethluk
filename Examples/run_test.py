from Framework_V1.core.config import CaseConfig
from Framework_V1.core.geometry import FlatPlate, LammellenRohrWU
from Framework_V1.models import frost_test
from Framework_V1.runtime.simulator import Simulator
from Framework_V1.visualisation import plot

cfg = CaseConfig(
    T_air= 10.0,        # °C
    T_wall= -10.0,      # °C
    u_air= 1.0,         # m/s
    RH = 0.8,           # -
    p_atm = 100000,     # Pa
    t_end = 10*60,      # s
    dt = 0.1,           # s
    test_ceof = 3e-5    # m/(s*K)
)

geom = LammellenRohrWU(
    n_rippen= 4,           # -
    l_rippen= 0.01,          # m
    rippen_dicke= 0.002,    # m
    t_rippen= 0.01,          # m
    d_rohr_a= 0.01,          # m
    rohr_dicke= 0.002       # m
)

sim = Simulator()

plot.plot_finned_tube_side(geom)

results = sim.run(cfg, frost_test)
plot.plot_frostdicke(results)