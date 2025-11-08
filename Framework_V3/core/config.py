from dataclasses import dataclass

@dataclass(frozen=False)
class CaseConfig:
    # air data
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
    T_w: float          # °C wall temperature

    # ice data
    rho_i: float        # kg/m^3 frost density (solid)
    h_sub: float        # kJ/kg latent heat of ablimation for water vapor

    # numerics
    t_end: float        # s endtime
    dt: float           # s time step

@dataclass(frozen=True)
class GridShape:
    nx: int = 100
    nr: int = 100
    ntheta: int = 90
