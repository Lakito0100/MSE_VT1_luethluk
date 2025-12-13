from dataclasses import dataclass

@dataclass(frozen=False)
class CaseConfig:
    # air data
    m_dot: float        # kg/s air mass flow
    T_a: float          # °C temperature air
    v_a: float          # m/s velocity air
    p_a: float          # Pa pressure air
    RH: float           # relative humidity air
    w_amb: float        # kg/kg water vapor moisture content
    rho_amb: float      # kg/m^3 density air
    v_kin: float        # m^2/s kinematic viscosity air
    lam: float          # W/mK heat conduction coefficient air
    c_p_a: float        # J/kgK heat capacity air
    D_std:  float       # m^2/s water vapor diffusion coefficient
    C: float            # 1/s empirical water vapor absorbed coefficient
    isv: float          # J/kg latent heat of desublimation

    # refrigerant data
    ref_str: str        # Name of refrigerant
    T_tube: float       # °C tube temperature
    T_ref: float        # °C refrigerant temperature
    p_ref: float        # Pa Refrigerant Pressure
    h_ref: float        # J/kg specific Enthalpie
    m_dot_ref: float    # kg/s Massflow Refrigerant at the inlet and outlet
    x_ref: float        # [-]

    # ice data
    rho_i: float        # kg/m^3 frost density (solid)
    h_sub: float        # kJ/kg latent heat of ablimation for water vapor

    frost_condition = False

@dataclass(frozen=False)
class GridShape:
    # numerics
    t_end: float        # s endtime
    dt: float           # s time step
    store_grid_every_x_it: int = 10 # store every x iterations

    nx: int = 100
    nr: int = 100
    ntheta: int = 90
