from dataclasses import dataclass
import numpy as np

@dataclass(frozen=False)
class CaseConfig:
    """Simulation boundary conditions and material properties."""
    # Air data
    m_dot: float        # kg/s air mass flow
    T_a: float          # °C air temperature
    v_a: float          # m/s air velocity
    p_a: float          # Pa air pressure
    RH: float           # relative humidity
    w_amb: float        # kg/kg water vapor moisture content
    rho_amb: float      # kg/m^3 air density
    v_kin: float        # m^2/s kinematic viscosity of air
    lam: float          # W/mK air thermal conductivity
    c_p_a: float        # J/kgK air heat capacity
    D_std:  float       # m^2/s water vapor diffusion coefficient
    C: float            # 1/s empirical water vapor absorption coefficient
    isv: float          # J/kg latent heat of desublimation

    # Fan model
    use_fan: bool
    fan_dp0: float
    fan_V0: float
    dp_clean: float     # Pa, clean-coil pressure drop (currently not used by the fan model)

    # Refrigerant data
    ref_str: str        # refrigerant name
    T_tube: float       # °C tube temperature
    T_ref: float        # °C refrigerant temperature
    p_ref: float        # Pa refrigerant pressure
    h_ref: float        # J/kg specific enthalpy
    m_dot_ref: float    # kg/s refrigerant mass flow at the inlet
    m_dot_ref_out: float   # kg/s refrigerant mass flow at the outlet
    x_ref: float        # [-]

    # Ice data
    rho_i: float        # kg/m^3 frost density (solid)
    h_sub: float        # J/kg latent heat of sublimation for water vapor

    frost_condition: bool = False

@dataclass(frozen=False)
class GridShape:
    """Discretization and solver control parameters."""
    # Numerics
    t_end: float        # s end time
    dt: float           # s time step
    print_output_every_x_it: int = 10
    store_grid_every_x_it: int = 10  # store every x iterations

    nx: int = 100
    nr: int = 100
    ntheta: int = 90

    # cal_steady_state: bool = True
    cal_air: bool = True
    cal_frost: bool = True
    cal_ref: bool = True

    # Adaptive time step
    dt_min: float = 0.01            # s lower bound
    dt_max: float = 0.5             # s upper bound
    it_target: int = 20             # target inner refrigerant-solver steps per time step

    # Dynamic models: hold-ramp-hold profiles of the inlet air
    change_humidity: bool = True
    change_temperature: bool = True

    # Temperature profile [°C]; T_a_start=None -> CaseConfig.T_a
    T_a_start: float | None = None
    T_a_end: float = -5.0
    t_switch_T: float = 30 * 60.0   # s start of the ramp
    ramp_T: float = 60.0            # s ramp duration

    # Relative humidity profile [-]; RH_start=None -> CaseConfig.RH
    RH_start: float | None = None
    RH_end: float = 0.8
    t_switch_RH: float = 5 * 60.0   # s start of the ramp
    ramp_RH: float = 20.0           # s ramp duration

    # Window [s] around each ramp in which the time step is reduced
    dt_window_switch: float = 10.0

@dataclass(frozen=False)
class HeatPump:
    """Heat-pump (condenser and water side) configuration."""
    # Condenser
    N_cond: int             # number of condenser control volumes
    p_ref_cond: float
    h_ref_cond: np.ndarray
    T_wall: np.ndarray
    T_water: np.ndarray

    A_flow_cond: float      # cross-sectional flow area [m2]
    t_plate: float          # Plate thickness
    height_cond: float      # heat exchanger height
    length_cond: float      # heat exchanger length
    n_plates: int           # number of plates
    A_plate: float          # Condenser heat transfer area [m2]
    c_plate: float
    rho_plate: float
    lamda_plate: float

    # Water
    T_in_water: float
    m_water: float          # water mass flow
    c_water: float
    rho_water: float

    # Controller
    use_controller: bool

    # Heat flows
    Q_cond: float = 0.0
    Q_evap: float = 0.0

    # Compressor
    W_comp: float = 0.001

    def RPM(self, t):
        """Return compressor RPM ramped from zero to 1000."""
        return min(1000, 80 * t)
