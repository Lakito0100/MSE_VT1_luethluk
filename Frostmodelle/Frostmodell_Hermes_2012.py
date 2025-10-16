from dataclasses import dataclass
import math
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.HumidAirProp import HAPropsSI

# ----------------- Basisdaten -----------------

@dataclass(frozen=True)
class Flatplate:
    L: float  # [m]

@dataclass(frozen=True)
class AirProperty:
    k_a: float   # [W/mK]
    c_p: float   # [J/kgK]
    Pr: float    # [-]
    rho: float   # [kg/m^3]
    mu: float    # [Pa*s]

@dataclass(frozen=True)
class OpPoint:
    T_a: float
    T_w: float
    w_tilde: float
    u_a: float

@dataclass(frozen=True)
class HermesParams:
    a0: float
    a1: float
    a2: float
    k_f0: float
    i_sv: float

# -------------- Psychrometrie (für w̃) ---------------

def w_from_T_RH(T_C: float, RH: float, p_atm: float = 101325.0) -> float:
    """
    Feuchteverhältnis x_a [kg/kg_trockene_Luft] aus (T, RH, p).
    """
    T_K = T_C + 273.15
    RH = max(0.0, min(1.0, RH))
    return float(HAPropsSI("W", "T", T_K, "P", p_atm, "R", RH))

def w_sat_over_ice_at_T(Tw_C: float, p_atm: float = 101325.0) -> float:
    """
    Sättigungs-Feuchteverhältnis x_w über EIS bei Wandtemperatur Tw.
    CoolProp: Frostpunkt-Eingang 'Tfp' erzwingt Sättigung über Eis.
    """
    Tw_K = Tw_C + 273.15
    return float(HAPropsSI("W", "T", Tw_K, "P", p_atm, "R", 1.0))

def compute_w_tilde(Ta_C: float, Tw_C: float, RH: float, p_atm: float = 101325.0) -> float:
    """
    w̃ = x_a - x_w  (x_w: Sättigung über Eis bei Tw)
    """
    x_a = w_from_T_RH(Ta_C, RH, p_atm)
    x_w = w_sat_over_ice_at_T(Tw_C, p_atm)
    return max(0.0, x_a - x_w)

# -------------- Strömungs-/Wärmeübergang ----------------

def Nu_Lienhard_turbulent_flat_plate(L: float, air: AirProperty, u: float) -> float:
    Re = air.rho * u * L / air.mu
    return 0.037 * (Re**0.8) * (air.Pr**0.43)

# -------------- Hermes 2012: X(s) (Eq. 34/35) ----------

def X_of_s(s: float, geom: Flatplate, air: AirProperty, op: OpPoint, hp: HermesParams) -> float:
    # Tilde-Größen
    T_tilde = hp.a1 * (op.T_a - op.T_w)
    w_tilde = op.w_tilde
    if w_tilde <= 0.0:
        return 0.0

    Nu = Nu_Lienhard_turbulent_flat_plate(geom.L, air, op.u_a)
    Ja = (air.c_p * (op.T_a - op.T_w)) / (hp.i_sv * w_tilde)
    ktilde0 = hp.k_f0 / air.k_a

    denom = (1.0 + T_tilde) * (1.0 + 1.0/Ja)
    d0 = w_tilde * (2.0 + T_tilde) / denom
    d1 = (ktilde0 / Nu) * (2.0 + T_tilde) / denom

    return max(0.0, 0.5 * (math.sqrt(d1*d1 + 4.0*d0*s) - d1))

# -------------- h(X), T_s, q_f  -------------------------
# Eq. (18): h = Nu * (k/k_f0) * (1 + 1/Ja) * X    (k/k_f0 = 1/ktilde0)
# Eq. (20): T_s = T_air - (T_air - T_wall)/(1 + h)
# Eq. (13): q_f = a0 * exp(a1 * T_s + a2)

def surface_state_from_s(s, geom, air, op, hp, b=3.0e-4, tol=1e-8, itmax=30):
    """
    Returns: X, h, Ts, qf   (alle paper-konsistent)
    - X(s): Eq. (34)/(35)
    - h(X): Eq. (18) mit k/kf, NICHT k/kf0
    - Ts:   Eq. (20)
    - qf:   Eq. (13)
    - kf:   Eq. (14) -> nur intern
    """
    # 1) X(s) aus der geschlossenen Lösung
    X  = X_of_s(s, geom, air, op, hp)
    if X <= 0.0:
        Ts = min(op.T_wall, 0.0)
        qf = hp.a0 * math.exp(hp.a1*Ts + hp.a2)
        h  = 0.0
        return X, h, Ts, qf

    Nu = Nu_Lienhard_turbulent_flat_plate(geom.L, air, op.u_a)
    Ja = (air.c_p * (op.T_air - op.T_wall)) / (hp.i_sv * op.w_tilde) if op.w_tilde > 0 else 1e12

    # Fixpunkt für kf(Ts)
    kf = hp.k_f0  # Start
    for _ in range(itmax):
        # h mit aktuellem kf: Eq. (18)
        h  = Nu * (air.k_a / max(kf, 1e-12)) * (1.0 + 1.0/Ja) * X
        # Ts: Eq. (20)
        Ts = op.T_air - (op.T_air - op.T_wall) / (1.0 + h)
        Ts = min(Ts, 0.0)  # Eisoberfläche
        # rho_f(Ts): Eq. (13)
        rho_f = hp.a0 * math.exp(hp.a1 * Ts + hp.a2)
        # neues kf: Eq. (14)
        kf_new = hp.k_f0 + b * rho_f
        if abs(kf_new - kf) <= tol * max(1.0, kf):
            kf = kf_new
            break
        kf = kf_new

    # qf am Ende (Eq. 13)
    qf = hp.a0 * math.exp(hp.a1 * Ts + hp.a2)
    return X, h, Ts, qf

# -------------- Zeitabbildung via Eq. (22) ---------------

def t_of_s(s_grid, geom, air, op, hp):
    q = np.array([surface_state_from_s(s, geom, air, op, hp)[3] for s in s_grid])
    coeff = air.c_p*(geom.L**2)/hp.k_f0
    return coeff * s_grid * q

# -------------- Kurven berechnen & plotten ---------------

if __name__ == "__main__":
    # Geometrie & Luft
    geom = Flatplate(L=0.10)  # 10 cm
    air  = AirProperty(k_a=0.026, c_p=1005.0, Pr=0.71, rho=1.2, mu=1.8e-5)

    # Randbedingungen
    T_a = 16.0
    RH  = 0.80
    u_a = 1.0

    Nu = Nu_Lienhard_turbulent_flat_plate(geom.L, air, u_a)
    print(Nu)

    # Hermes-Parameter (Paper)
    a0   = 207.0
    a1   = 0.266
    k_f0 = 0.132     # W/mK
    i_sv = 2.83e6    # J/kg

    # Zielzeit
    t_target = 120.0 * 60.0  # 120 min in s

    Tw_list = [-4.0, -8.0, -12.0, -16.0]
    plt.figure(figsize=(7.2, 4.6))

    for Tw in Tw_list:
        w_tilde = compute_w_tilde(T_a, Tw, RH)
        hp = HermesParams(a0=a0, a1=a1, a2=-0.615*Tw, k_f0=k_f0, i_sv=i_sv)
        op = OpPoint(T_a=T_a, T_w=Tw, w_tilde=w_tilde, u_a=u_a)

        # s-Grid so wählen, dass t(s_max) ~ t_target (2–3 einfache Iterationen)
        s_max = 1.0
        for _ in range(3):
            s_grid = np.linspace(0.0, s_max, 300)
            t_grid = t_of_s(s_grid, geom, air, op, hp)
            if t_grid[-1] > 1e-9:
                s_max *= (t_target / t_grid[-1])

        # Finales Grid + Größen
        s = np.linspace(0.0, s_max, 600)
        t = t_of_s(s, geom, air, op, hp)
        Xs = np.array([X_of_s(si, geom, air, op, hp) for si in s])
        x_mm = 1e3 * Xs * geom.L

        plt.plot(t/60.0, x_mm, label=f"Tw = {Tw:.0f}°C")

    plt.xlabel("Zeit [min]")
    plt.ylabel("Frostdicke $x_s$ [mm]")
    plt.title("Hermes (2012)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()