from pathlib import Path
import sys
import numpy as np
from CoolProp.HumidAirProp import HAPropsSI
from CoolProp.CoolProp import PropsSI
sys.path.insert(0, str(Path.cwd().parent))
from Framework.core.config import CaseConfig, GridShape, HeatPump
from Framework.core.geometry import FinnTubedHX
from Framework.models import Frostmodell_V1
from Framework.runtime.simulator_V2 import Simulator
from Framework.visualisation import plot
from Framework.runtime.recorder import ResultRecorder

sim_run = True
read_data = True

#air
T_a = 20.0
P = 101325
RH = 0.1
v_a = 1.2

#water
T_in_water = 30.0
N_cond = 22

#refrigerant
w_amb_in = HAPropsSI('W','T',T_a+273.15,'P',P,'R', RH)
rho_amb_in = 1.0 / HAPropsSI("Vha","T",T_a+273.15,"P",P,"R",RH)

refrigerant = "R134a"
#at evaportor
x_evap = 0.5
p_ref_evap = PropsSI("P", "T", T_a + 273.15, "Q", x_evap, refrigerant)
h_ref_evap = PropsSI("H", "T", T_a + 273.15, "Q", x_evap, refrigerant)

#at condenser
x_cond = 0.5
p_ref_cond = PropsSI("P", "T", T_in_water + 273.15, "Q", x_cond, refrigerant)
h_ref_cond = PropsSI("H", "T", T_in_water + 273.15, "Q", x_cond, refrigerant)

geom = FinnTubedHX(
    n_seg_l = 2,        # -
    n_seg_r = 20,        # -
    stacks = 180,         # -
    n_fin = 2,           # -
    l_fin = 0.022,          # m
    h_fin = 0.022,          # m
    fin_thickness = 0.0002,    # m
    fin_pitch_cc = 0.0032,          # m
    d_tube_a = 0.00952,          # m
    tube_thickness = 0.0005,       # m
    lambda_fin = 237,            # W/mK
    rho_solid = 2700,            # kg/m3
    c_solid = 900,               # J/kgK
    CP = "row_serpentine"
)

gs = GridShape(
    t_end = 5*60.0,      # s endtime
    dt = 1.0,           # s time step
    store_grid_every_x_it = 10,

    nx = 100,
    nr = 100,
    ntheta = 90,

    #Model Setup
    cal_steady_state = False,
    cal_air = True,
    cal_frost = False,
    cal_ref = True
)

cfg = CaseConfig(
    # air data
    m_dot = v_a*rho_amb_in*(geom.h_fin*geom.fin_gap()*geom.n_seg_r*geom.stacks), # Backup massflow if no fan is used
    T_a = T_a,          # °C temperature air
    v_a = v_a,          # m/s velocity air
    p_a = P,            # Pa pressure air
    RH = RH,            # relative humidity air
    w_amb = w_amb_in,          # kg/kg water vapor moisture content
    rho_amb = rho_amb_in,      # kg/m^3 density air
    v_kin = 1.5e-5,     # m^2/s kinematic viscosity air
    lam = 0.025,        # W/mK heat conduction coefficient air
    c_p_a = 1000,       # J/kgK heat capacity air
    D_std = 2.2e-5,     # m^2/s water vapor diffusion coefficient
    C = 900,            # 1/s empirical water vapor absorbed coefficient
    isv = 2830000,      # J/kg latent heat of desublimation

    # fan model
    use_fan = True,
    fan_dp0 = 150.0,
    fan_V0 = v_a*(geom.h_fin*geom.fin_gap()*geom.n_seg_r*geom.stacks),
    dp_clean = 40.0,


    # refrigerant data
    ref_str = refrigerant,
    T_tube = T_a,    # °C tube temperature
    T_ref = T_a,     # °C tube temperature
    p_ref = p_ref_evap,      # Pa Kältemitteldruck
    h_ref = h_ref_evap,          # J/kg spezifische Enthalpie
    m_dot_ref = 0,      # kg/s Massenstrom am Inlet (invalid)
    m_dot_ref_out = 0,      # kg/s Massenstrom am Outlet (invalid)
    x_ref = x_evap,           # Dampfqualität (0..1), NaN falls einphasig

    # ice data
    rho_i = 920,        # kg/m^3 ice density
    h_sub= 2830000,     # J/kg latent heat of ablimation for water vapor
)

HP = HeatPump(
    N_cond=N_cond,
    p_ref_cond = p_ref_cond,
    h_ref_cond = np.full(N_cond, h_ref_cond, dtype=float),
    T_wall = np.full(N_cond, T_in_water, dtype=float),
    T_water = np.full(N_cond, T_in_water, dtype=float),

    A_flow_cond = 0.01,
    dx_cond = 0.5/N_cond,
    t_plate = 0.001,
    height_cond=1.0,
    A_plate = 3.65,
    c_plate = 500,
    rho_plate = 8000.0,
    lamda_plate = 50.0,

    #water
    T_in_water = T_in_water,
    m_water = 0.5,
    c_water = 4200,
    rho_water = 997
)

result_file = "results_test"

if sim_run:
    sim = Simulator(geom,cfg,HP,result_file)
    results = sim.run(cfg, geom, gs, Frostmodell_V1)

    data = results.data
    results.to_csv(result_file,data)

if read_data:
    data = ResultRecorder.read_results_csv_json(result_file)
    results = ResultRecorder.from_jsonl(result_file)

### All Plots

snapshots = results.grid_snapshots

fig, ax, Z_sft = plot.plot_segment_field_grid(
    snapshots,
    field="v_a",
    source="cfg",
    t_idx=-1,          # letzter Zeitschritt
    title="Geschwindigkeit der Luft über alle Segmente",
    cmap="viridis"
)

fig, ax, Z_sft = plot.plot_segment_field_grid(
    snapshots,
    field="s_ft",
    source="st",       # aus SimState
    t_idx=-1,          # letzter Zeitschritt
    title="Frostdicke s_ft über alle Segmente",
    cmap="viridis"
)

fig, ax, Z_Ta = plot.plot_segment_field_grid(
    snapshots,
    field="T_a",
    source="cfg",      # aus CaseConfig
    t_idx=-1,
    title="Lufttemperatur T_a über alle Segmente",
    cmap="plasma"
)

fig, ax, Z_Ta = plot.plot_segment_field_grid(
    snapshots,
    field="w_amb",
    source="cfg",      # aus CaseConfig
    t_idx=-1,
    title="Feuchtigkeit der Luft w_amb über alle Segmente",
    cmap="viridis"
)

fig, ax, Z_Ta = plot.plot_segment_field_grid(
    snapshots,
    field="RH",
    source="cfg",      # aus CaseConfig
    t_idx=-1,
    title="relative Feuchtigkeit der Luft w_amb über alle Segmente",
    cmap="viridis"
)

fig, ax, Z_Ta = plot.plot_segment_field_grid(
    snapshots,
    field="T_tube",
    source="cfg",      # aus CaseConfig
    t_idx=-1,
    title="Temperatur der Wände über alle Segmente",
    cmap="plasma"
)

fig, ax, Z_Ta = plot.plot_segment_field_grid(
    snapshots,
    field="T_ref",
    source="cfg",      # aus CaseConfig
    t_idx=-1,
    title="Temperatur des Kältemittels über alle Segmente",
    cmap="plasma"
)

fig, ax, Z_Ta = plot.plot_segment_field_grid(
    snapshots,
    field="h_ref",
    source="cfg",      # aus CaseConfig
    t_idx=-1,
    title="Enthalpie des Kältemittels über alle Segmente",
    cmap="viridis"
)

fig, ax, Z_Ta = plot.plot_segment_field_grid(
    snapshots,
    field="p_ref",
    source="cfg",      # aus CaseConfig
    t_idx=-1,
    title="Druck vom Kältemittel über alle Segmente",
    cmap="viridis"
)

fig, ax, Z_Ta = plot.plot_segment_field_grid(
    snapshots,
    field="x_ref",
    source="cfg",      # aus CaseConfig
    t_idx=-1,
    title="X des Kältemittel über alle Segmente",
    cmap="viridis"
)

plot.plot_logph_cycles(
    ref=cfg.ref_str,
    t=data["t"],
    cycle_ph=data["cycle_ph"],
    at_time=gs.t_end,
    isotherms=True,
    grid="none"
)

plot.plot_logph_cycles(
    ref=cfg.ref_str,
    t=data["t"],
    cycle_ph=data["cycle_ph"],
    t_start=2*60.0,
    t_end=gs.t_end,
    every_s=60.0,   # alle 60 s ein Kreisprozess
    isotherms=True,
    grid="none",
    title="Kreisprozess über die Zeit"
)

plot.plot_any(kind="time vs any",
              x=data['t'], y=data['EER'],
              xlabel="Zeit [s]", ylabel="EER",
              title=f"EER", marker=None)

plot.plot_any(kind="time vs any",
              x=data['t'], y=data['COP'],
              xlabel="Zeit [s]", ylabel="COP",
              title=f"COP", marker=None)

plot.plot_any(kind="time vs any",
              x=data['t'], y=data['mean_s_ft'],
              xlabel="Zeit [s]", ylabel="Frostdicke [m]",
              title=f"Durchschnittliche Frostdicke", marker=None)

plot.plot_any(kind="time vs any",
              x=data['t'], y=data['m_dot_air'],
              xlabel="Zeit [s]", ylabel="Massenfluss [kg/s]",
              title=f"Gesamter Massenfluss der Luft", marker=None)

plot.plot_any(kind="time vs any",
              x=data['t'], y=data['v_in_air'],
              xlabel="Zeit [s]", ylabel="Geschwindigkeit [m/s]",
              title=f"Geschwindigkeit der Luft am inlet", marker=None)

plot.plot_any(kind="time vs any",
              x=data['t'], y=data['T_out_air_mean'],
              xlabel="Zeit [s]", ylabel="Temperatur [°C]",
              title=f"Gemittelte Austrittstemperatur Luft", marker=None)

plot.plot_any(kind="time vs any",
              x=data['t'], y=data['T_out_ref'],
              xlabel="Zeit [s]", ylabel="Temperatur [°C]",
              title=f"Austrittstemperatur Kältemittel", marker=None)

plot.plot_any(kind="time vs any",
              x=data['t'], y=data['p_ref_evap'],
              xlabel="Zeit [s]", ylabel="Druck [Pa]",
              title=f"Druck des Kältemittels Verdampferseite", marker=None)

plot.plot_any(kind="time vs any",
              x=data['t'], y=data['p_ref_cond'],
              xlabel="Zeit [s]", ylabel="Druck [Pa]",
              title=f"Druck des Kältemittels Kondenserseite", marker=None)

plot.plot_any(kind="time vs any",
              x=data['t'], y=data['humidity'],
              xlabel="Zeit [s]", ylabel="Feuchtegehalt [kg/kg]",
              title=f"Feuchtgehalt entlang dem Luftstrom", marker=None,
              labels=["Seg 1","Seg 2","Seg 3","Seg 4","Seg 5"])