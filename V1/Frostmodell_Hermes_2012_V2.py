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
    beta: float

# -------------- Modellimplementation ---------------

def T_tilde(op: OpPoint, hp: HermesParams) -> float:
    return hp.a1 * (op.T_a - op.T_w)

def Ja(air: AirProperty, op: OpPoint, hp: HermesParams) -> float:
    return air.c_p * (op.T_a - op.T_w) / (hp.i_sv * op.w_tilde)

def k_tilde(air: AirProperty, hp: HermesParams) -> float:
    return hp.k_f0 / air.k_a

def Re(geom: Flatplate, air: AirProperty, op: OpPoint) -> float:
    return air.rho * op.u_a * geom.L / air.mu

def Nu(geom: Flatplate, air: AirProperty, op: OpPoint) -> float:
    return 0.037 * Re(geom, air, op)**0.8 * air.Pr**0.43

def X_of_s(s: float, geom: Flatplate, air: AirProperty, op: OpPoint, hp: HermesParams) -> float:

    w_tilde = op.w_tilde

    d0 = w_tilde * (2 + T_tilde(op, hp)) / ((1 + T_tilde(op, hp)) * (1 + 1/Ja(air, op, hp)))
    d1 = (k_tilde(air, hp)/Nu(geom, air, op)) * (2 + T_tilde(op, hp)) / ((1 + T_tilde(op, hp)) * (1 + 1/Ja(air, op, hp)))
    return 0.5 * (math.sqrt(d1**2 + 4.0*d0*s) - d1)

def frost_state_at_s(s: float, geom: Flatplate, air: AirProperty, op: OpPoint, hp: HermesParams,
                     Ts0: float | None = None, tol: float = 1e-6, itmax: int = 1000):

    # initial guess: weighted between Ta and Tw
    Ts = Ts0 if Ts0 is not None else (0.3*op.T_a + 0.7*op.T_w)

    Nu_val = Nu(geom, air, op)
    X = X_of_s(s, geom, air, op, hp)
    Ja_val = Ja(air, op, hp)

    for _ in range(itmax):
        rho_f = hp.a0 * math.exp(hp.a1*Ts + hp.a2)
        k_f  = hp.k_f0 + hp.beta * rho_f
        Bi  = Nu_val * X * (air.k_a / k_f)
        theta  = Bi * (1.0 + 1.0/Ja_val)
        Ts_new = op.T_a - (op.T_a - op.T_w) / (1.0 + theta)
        if abs(Ts_new - Ts) < tol:
            return Ts_new, rho_f, k_f, Bi, theta
        Ts = Ts_new

    # If not converged, still return last iterate
    raise SystemExit("Frostzustand nicht konvergiert!")

def s_of_t(t: float, geom: Flatplate, air: AirProperty, op: OpPoint, hp: HermesParams, s_old: float) -> float:
    Ts, rho_f, k_f, _, _ = frost_state_at_s(s_old, geom, air, op, hp)
    return k_f * t / (rho_f * air.c_p * geom.L**2)

def t_of_s(s: float, geom: Flatplate, air: AirProperty, op: OpPoint, hp: HermesParams) -> float:
    Ts, rho_f, k_f, _, _ = frost_state_at_s(s, geom, air, op, hp)
    return (s * rho_f * air.c_p * geom.L**2) / k_f

def Ts_of_s(s: float, geom: Flatplate, air: AirProperty, op: OpPoint, hp: HermesParams) -> float:
    Ts, *_ = frost_state_at_s(s, geom, air, op, hp)
    return Ts

# -------------- Kurven berechnen & plotten ---------------

if __name__ == "__main__":

    # Randbedingungen
    T_a = 16.0      # °C Lufttemperatur
    T_w = -8.0     # °C Wandtemperatur
    RH  = 0.8      # Relative Luftfeuchtigkeit in der Luft
    u_a = 1.0       # m/s
    p = 100000   # Pa

    w_w = HAPropsSI("W", "T", T_w+273.15, "P", p, "R", 1.0)
    w_a = HAPropsSI("W", "T", T_a+273.15, "P", p, "R", RH)
    w_tilde = w_a - w_w

    op = OpPoint(T_a=T_a, T_w=T_w, w_tilde=w_tilde, u_a=u_a)

    # Hermes-Parameter (Paper)
    a0   = 207.0
    a1   = 0.266
    k_f0 = 0.132     # W/mK
    i_sv = 2.83e6    # J/kg
    beta = 3e-4

    hp = HermesParams(a0=a0, a1=a1, a2=-0.615*T_w, k_f0=k_f0, i_sv=i_sv, beta=beta)

    # Geometrie & Luft
    geom = Flatplate(L=0.10)  # 10 cm
    k_feucht = HAPropsSI('K', 'T', T_a+273.15, 'P', p, 'R', RH)
    air  = AirProperty(k_a=k_feucht, c_p=1005.0, Pr=0.71, rho=1.2, mu=1.8e-5)

    # Zielzeit
    t_end = 120.0 * 60.0  # 120 min in s
    s_end = 30

    s_array = np.linspace(1e-9, s_end, 10000)

    X = np.array([X_of_s(s, geom, air, op, hp) for s in s_array])
    results = [frost_state_at_s(s, geom, air, op, hp) for s in s_array]
    Ts, rho_f, k_f, Bi, theta = map(np.asarray, zip(*results))

    x_s = X * geom.L * 1e3

    dt_ds = (rho_f * air.c_p * geom.L ** 2) / k_f
    t_array = np.concatenate([[0.0], np.cumsum(0.5 * (dt_ds[1:] + dt_ds[:-1]) * np.diff(s_array))])

    # X vs rech

    rech = (theta/(1 + theta)) * (s_array/X)  # elementweise

    plt.plot(X, rech, label=f"Tw = {T_w:.0f}°C")
    plt.xlabel("X")
    #plt.ylabel("")
    plt.title("Hermes (2012)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Xs

    plt.plot(t_array/60, x_s, label=f"Tw = {T_w:.0f}°C")
    plt.text(
        0.05, 0.95,
        fr"$\tilde{{\omega}} = {w_tilde:.4f}, \; \tilde{{T}} = {T_tilde(op, hp):.1f}, \; Nu = {Nu(geom, air, op):.0f}$",
        fontsize=12,
        ha='left', va='top',
        transform=plt.gca().transAxes
    )

    plt.xlabel("Zeit [min]")
    plt.ylabel("Frostdicke $x_s$ [mm]")
    plt.title("Hermes (2012)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xlim([0, t_end / 60])
    plt.show()

    # Frost Oberflächentemperatur

    plt.plot(t_array/60, Ts, label=f"Tw = {T_w:.0f}°C")
    plt.xlabel("Zeit [min]")
    plt.ylabel("Frost Oberflächentemperatur [°C]")
    plt.title("Hermes (2012)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xlim([0, t_end / 60])
    plt.show()

    # Frostdichte

    mask = t_array <= t_end
    y_vis = rho_f[mask]
    plt.plot(t_array/60, rho_f, label=f"Tw = {T_w:.0f}°C")
    plt.xlabel("Zeit [min]")
    plt.ylabel("mittlere Frostdichte [kg/m^3]")
    plt.title("Hermes (2012)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xlim([0, t_end/60])
    plt.ylim(y_vis.min()*0.98, y_vis.max()*1.02)
    plt.show()

    # Frost Wärmeleitfähigkeit

    mask = t_array <= t_end
    y_vis = k_f[mask]
    plt.plot(t_array/60, k_f, label=f"Tw = {T_w:.0f}°C")
    plt.xlabel("Zeit [min]")
    plt.ylabel("Frost Wärmeleitfähigkeit [W/mK]")
    plt.title("Hermes (2012)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xlim([0, t_end/60])
    plt.ylim(y_vis.min()*0.98, y_vis.max()*1.02)
    plt.show()

    # Biotzahl

    mask = t_array <= t_end
    y_vis = Bi[mask]
    plt.plot(t_array/60, Bi, label=f"Tw = {T_w:.0f}°C")
    plt.xlabel("Zeit [min]")
    plt.ylabel("Biotzahl")
    plt.title("Hermes (2012)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xlim([0, t_end/60])
    plt.ylim(y_vis.min()*0.98, y_vis.max()*1.02)
    plt.show()