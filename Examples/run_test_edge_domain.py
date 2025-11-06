from Framework_V2.core.config import CaseConfig, GridShape
from Framework_V2.core.geometry import FinnTubedHX
from Framework_V2.models import Frostmodell_V1
from Framework_V2.runtime.simulator import Simulator
from Framework_V2.visualisation import plot
from CoolProp.HumidAirProp import HAPropsSI



T_a = 16
T_w = -10
P = 103500
RH = 0.8
v_a = 1

cfg = CaseConfig(
    # air data
    T_a = T_a,          # °C temperature air
    v_a = v_a,          # m/s velocity air
    p_a = P,            # Pa pressure air
    RH = RH,            # relative humidity air
    w_amb = HAPropsSI('W','T',T_a+273.15,'P',P,'R', RH),          # kg/kg water vapor moisture content
    rho_amb = 1.2,      # kg/m^3 density air
    v_kin = 1.5e-5,     # m^2/s kinematic viscosity air
    lam = 0.025,        # W/mK heat conduction coefficient air
    c_p_a = 1000,       # J/kgK heat capacity air
    D_std = 2.2e-5,     # m^2/s water vapor diffusion coefficient
    C = 900,            # 1/s empirical water vapor absorbed coefficient
    isv = 2830000,      # J/kg latent heat of desublimation

    # refrigerant data
    T_w = T_w,          # °C wall temperature

    # ice data
    rho_i = 920,        # kg/m^3 frost density (solid)
    h_sub= 334,         # kJ/kg latent heat of ablimation for water vapor

    # numerics
    t_end = 60,      # s endtime
    dt = 1              # s time step
)

geom = FinnTubedHX(
    n_fin = 4,           # -
    l_fin = 0.01,          # m
    d_fin = 0.002,    # m
    fin_pitch = 0.01,          # m
    d_tube_a = 0.01,          # m
    tube_thickness = 0.002       # m
)

gs = GridShape(
    nx = 100,
    nr = 100,
    ntheta = 1#90
)

sim = Simulator(fields=("t","s_e"))

result_file = "results_test_edge.csv"

results = sim.run(cfg, geom, gs, Frostmodell_V1)
results.to_csv(result_file)
plot.plot_results(result_file, "t", "s_e", "Plot der Resultate", False, 400, True)