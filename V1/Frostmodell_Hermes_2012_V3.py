# Hermes (2012) Frostmodell mit CoolProp + Matplotlib-Plot (ohne Datei-Export)

from dataclasses import dataclass
from typing import Dict
import numpy as np
import math
import matplotlib.pyplot as plt
from CoolProp.HumidAirProp import HAPropsSI
from CoolProp.CoolProp import PropsSI

# --------------------- SETUP ---------------------

@dataclass(frozen=True)
class Ambient:
    Ta_C: float
    RH: float          # 0..1 (rel. Feuchte, bezogen auf Wasser)
    ua: float          # m/s
    p_atm: float       # Pa
    L: float           # m (char. Länge)

@dataclass(frozen=True)
class Constants:
    isv: float = 2.83e6      # J/kg  (latente Wärme der Resublimation)
    a1: float = 0.277        # 1/°C  (Exponent-Slope in rho_f(Ts))
    kf0: float = 0.132       # W/mK  (kf-Intercept)
    b_kf: float = 3.0e-4     # W·m^3/(kg·K) (kf = kf0 + b*rho_f)

TW_LIST = [-4.0, -8.0, -12.0, -16.0]   # °C
SIM_TIME_MIN = 120.0
DT_SECONDS = 1.0

AMB = Ambient(Ta_C=16.0, RH=0.80, ua=1.0, p_atm=101325.0, L=0.20)
CST = Constants()

# --------------------- FROST-/LUFT-KORRELATIONEN ---------------------

def frost_density(Ts_C: float) -> float:
    """Hayashi et al. (1977): rho_f [kg/m^3] = 650 * exp(0.277 * Ts[°C])"""
    return 650.0 * math.exp(0.277 * Ts_C)

def frost_k(rho_f: float, kf0: float=CST.kf0, b: float=CST.b_kf) -> float:
    return kf0 + b * rho_f

# --------------------- ABGELEITETE PARAMETER (mit CoolProp) ---------------------

@dataclass(frozen=True)
class DerivedParams:
    xa: float
    xw: float
    x_tilde: float
    T_tilde: float
    Ja: float
    Nu: float
    k_air: float
    cp_air: float
    mu_air: float
    rho_air: float
    k_ratio_tilde: float  # ~k = kf0 / k_air

def derive_params(Ta_C: float, Tw_C: float, amb: Ambient, cst: Constants) -> DerivedParams:
    Ta_K = Ta_C + 273.15
    Tw_K = Tw_C + 273.15
    P = amb.p_atm
    RH = amb.RH

    # Umgebungszustand (bezogen auf Wasser-RH)
    xa = HAPropsSI('W',   'T', Ta_K, 'P', P, 'R', RH)
    k_a = HAPropsSI('K',  'T', Ta_K, 'P', P, 'R', RH)     # W/m/K (feuchte Luft)
    rho = HAPropsSI('D',  'T', Ta_K, 'P', P, 'R', RH)     # kg/m^3 (feuchte Luft)

    # Für μ und c_p nehmen wir trockene Luft aus PropsSI (sehr gute Näherung)
    mu  = PropsSI('V', 'T', Ta_K, 'P', P, 'Air')          # Pa*s
    cp  = PropsSI('C', 'T', Ta_K, 'P', P, 'Air')          # J/kg/K

    # Sättigungs-Feuchte am kalten Rand (über Eis, falls T_w < 0°C)
    xw = HAPropsSI('W', 'T', Tw_K, 'P', P, 'R', 1.0)    # RH wrt WATER = 100 %

    x_tilde = max(xa - xw, 1e-12)

    # dimensionslose Größen
    T_tilde = cst.a1 * (Ta_C - Tw_C)
    Ja = cp * (Ta_C - Tw_C) / (cst.isv * x_tilde)

    # Re, Pr, Nu (turbulente Platte; Hermes: Nu = 0.037 Re^0.8 Pr^0.43)
    nu_kin = mu / rho
    Re = max(amb.ua * amb.L / max(nu_kin, 1e-12), 1.0)
    Pr = max(cp * mu / max(k_a, 1e-12), 1e-8)
    Nu = 0.037 * (Re ** 0.8) * (Pr ** 0.43)

    k_ratio_tilde = CST.kf0 / k_a

    return DerivedParams(
        xa=xa, xw=xw, x_tilde=x_tilde, T_tilde=T_tilde, Ja=Ja, Nu=Nu,
        k_air=k_a, cp_air=cp, mu_air=mu, rho_air=rho, k_ratio_tilde=k_ratio_tilde
    )

# --------------------- MODELLGLEICHUNGEN ---------------------

def h_of_X(X: float, dp: DerivedParams) -> float:
    # h = Nu * (k_air/k_f) * (1 + 1/Ja) * X; k_f ≈ kf0 => k_air/k_f ≈ 1/~k
    return (dp.Nu / dp.k_ratio_tilde) * (1.0 + 1.0 / dp.Ja) * X

def Ts_from_X(X: float, Ta_C: float, Tw_C: float, dp: DerivedParams) -> float:
    # (Ts - Tw)/(Ta - Tw) = h/(1+h)
    X = max(X, 0.0)
    h = h_of_X(X, dp)
    return Tw_C + (Ta_C - Tw_C) * h / (1.0 + h)

def dXds_exact(X: float, s: float, dp: DerivedParams) -> float:
    # dX/ds = Nu*~x/~k + Nu*(1+~T)*(1+1/Ja)*X - Nu*~x*~T * [h/(1+h)] * (s/X)
    X = max(X, 1e-12)
    h = h_of_X(X, dp)
    term1 = dp.Nu * dp.x_tilde / dp.k_ratio_tilde
    term2 = dp.Nu * (1.0 + dp.T_tilde) * (1.0 + 1.0 / dp.Ja) * X
    term3 = dp.Nu * dp.x_tilde * dp.T_tilde * (h / (1.0 + h)) * (s / X)
    return term1 + term2 - term3

def X_of_s_exact(s_grid: np.ndarray, dp: DerivedParams) -> np.ndarray:
    # Heun (RK2)
    X = np.zeros_like(s_grid)
    x = 1e-12
    s_prev = s_grid[0]
    X[0] = x
    for i in range(1, len(s_grid)):
        s_cur = s_grid[i]
        ds = s_cur - s_prev
        k1 = dXds_exact(x, s_prev, dp)
        x_euler = x + ds * k1
        k2 = dXds_exact(x_euler, s_cur, dp)
        x = x + 0.5 * ds * (k1 + k2)
        X[i] = max(x, 0.0)
        s_prev = s_cur
    return X

def X_of_s_approx(s_grid: np.ndarray, dp: DerivedParams) -> np.ndarray:
    # X = ( sqrt(d1^2 + 4*d0*s) - d1 ) / 2
    d0 = dp.x_tilde * (1.0 + dp.T_tilde) * (1.0 + 1.0 / dp.Ja) / (2.0 + dp.T_tilde)
    d1 = (dp.k_ratio_tilde / dp.Nu) * (2.0 + dp.T_tilde) / ((1.0 + dp.T_tilde) * (1.0 + 1.0 / dp.Ja))
    X = (np.sqrt(d1 * d1 + 4.0 * d0 * s_grid) - d1) * 0.5
    return np.maximum(X, 0.0)

def s_to_t_minutes(s_grid: np.ndarray, X_grid: np.ndarray, Ta_C: float, Tw_C: float,
                   amb: Ambient, dp: DerivedParams) -> np.ndarray:
    # ds/dt ≈ kf0 / (rho_f * c_p * L^2)  -> dt = (rho_f * c_p * L^2 / kf0) ds
    dt_sec = np.zeros_like(s_grid)
    for i in range(1, len(s_grid)):
        X_mid = 0.5 * (X_grid[i] + X_grid[i-1])
        Ts_mid = Ts_from_X(X_mid, Ta_C, Tw_C, dp)
        rho_f = frost_density(Ts_mid)
        ds = s_grid[i] - s_grid[i-1]
        dt = (rho_f * dp.cp_air * amb.L * amb.L / CST.kf0) * ds
        dt_sec[i] = dt
    return np.cumsum(dt_sec) / 60.0

# --------------------- SIMULATION + PLOT ---------------------

def simulate_case(Tw_C: float, amb: Ambient, cst: Constants,
                  t_end_min: float, dt_s: float) -> Dict[str, np.ndarray]:
    dp = derive_params(amb.Ta_C, Tw_C, amb, cst)

    # s-Gitter durch Marsch in t (ds aus ds/dt mit rho_f von Approx.-X)
    n_steps = int(max(1, t_end_min * 60.0 // dt_s)) + 1
    s_vals = np.zeros(n_steps)
    Xa_prev = 1e-10
    for i in range(1, n_steps):
        Ts_prev = Ts_from_X(Xa_prev, amb.Ta_C, Tw_C, dp)
        rho_f_prev = frost_density(Ts_prev)
        ds = (CST.kf0 / (rho_f_prev * dp.cp_air * amb.L * amb.L)) * dt_s
        s_vals[i] = s_vals[i-1] + ds
        Xa_prev = X_of_s_approx(np.array([s_vals[i]]), dp)[0]

    X_exact = X_of_s_exact(s_vals, dp)
    X_approx = X_of_s_approx(s_vals, dp)

    t_exact_min = s_to_t_minutes(s_vals, X_exact, amb.Ta_C, Tw_C, amb, dp)
    t_approx_min = s_to_t_minutes(s_vals, X_approx, amb.Ta_C, Tw_C, amb, dp)

    xs_exact_mm = X_exact * amb.L * 1e3
    xs_approx_mm = X_approx * amb.L * 1e3

    return dict(
        t_exact_min=t_exact_min, xs_exact_mm=xs_exact_mm,
        t_approx_min=t_approx_min, xs_approx_mm=xs_approx_mm
    )

def make_plot():
    plt.figure(figsize=(8, 5))
    for Tw in TW_LIST:
        res = simulate_case(Tw, AMB, CST, SIM_TIME_MIN, DT_SECONDS)
        plt.plot(res["t_exact_min"],  res["xs_exact_mm"],  label=f"T_w={Tw:.0f}°C (exakt)")
        plt.plot(res["t_approx_min"], res["xs_approx_mm"], linestyle="--", label=f"T_w={Tw:.0f}°C (approx)")
    plt.xlabel("Zeit [min]")
    plt.ylabel("Frostdicke $x_s$ [mm]")
    plt.title("Frostwachstum — Hermes (2012) mit CoolProp (Fig. 2–ähnlich)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.xlim([0, SIM_TIME_MIN])
    plt.ylim([0, 5])
    plt.show()   # nur anzeigen

if __name__ == "__main__":
    make_plot()
