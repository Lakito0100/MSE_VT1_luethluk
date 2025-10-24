from Framework_V2.core.config import CaseConfig
from Framework_V2.core.geometry import FinnTubedHX
from Framework_V2.models import frost_test
from Framework_V2.runtime.simulator import Simulator
from Framework_V2.visualisation import plot

cfg = CaseConfig(
    # air data
    T_a = 10,
    v_a = 1,
    p_a = 103500,
    RH = 0.8,
    w_amb = 1,
    rho_amb = 1.2,
    v_kin = 1.5e-5,
    lam = 0.025,
    c_p_a = 1000,
    D_std = 2.2e-5,
    C = 1,
    isv = 2830000,    # J/kg

    # refrigerant data
    T_w = -10,

    # ice data
    rho_i = 1,
    h_sub= 334,

    # numerics
    t_end = 60*10,
    dt = 5
)

geom = FinnTubedHX(
    n_fin = 4,           # -
    l_fin = 0.01,          # m
    d_fin = 0.002,    # m
    fin_pitch = 0.01,          # m
    d_tube_a = 0.01,          # m
    tube_thickness = 0.002       # m
)

sim = Simulator(fields=("t","x_frost"))

plot.plot_finned_tube_side(geom)

results = sim.run(cfg, frost_test)
results.to_csv("results.csv")
plot.plot_frostdicke(results)