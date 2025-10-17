from dataclasses import dataclass
import math
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.HumidAirProp import HAPropsSI
from CoolProp.CoolProp import PropsSI

# ----------------- Basisdaten -----------------

@dataclass(frozen=True)
class Flatplate:
    L: float  # [m]

@dataclass(frozen=True)
class AirProperty:
    k_a: float   # [W/mK]
    c_p: float   # [J/kgK]
    rho: float   # [kg/m^3]
    mu: float    # [Pa*s]

    def Pr(self):
        return self.c_p * self.mu / self.k_a

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
    return 0.037 * Re(geom, air, op)**0.8 * air.Pr()**0.43

def X_of_s(s: float, geom: Flatplate, air: AirProperty, op: OpPoint, hp: HermesParams) -> float:

    w_tilde = op.w_tilde

    d0 = w_tilde * (2 + T_tilde(op, hp)) / ((1 + T_tilde(op, hp)) * (1 + 1/Ja(air, op, hp)))
    d1 = (k_tilde(air, hp)/Nu(geom, air, op)) * (2 + T_tilde(op, hp)) / ((1 + T_tilde(op, hp)) * (1 + 1/Ja(air, op, hp)))
    return 0.5 * (math.sqrt(d1**2 + 4.0*d0*s) - d1)

def frost_state_at_s(s: float, geom: Flatplate, air: AirProperty, op: OpPoint, hp: HermesParams,
                     Ts0: float | None = None, tol: float = 1e-6, itmax: int = 1000):

    # initial guess: weighted between Ta and Tw
    Ts = Ts0 if Ts0 is not None else (0.5*op.T_a + 0.5*op.T_w)

    Nu_val = Nu(geom, air, op)
    X = X_of_s(s, geom, air, op, hp)
    Ja_val = Ja(air, op, hp)

    for _ in range(itmax):
        rho_f = hp.a0 * math.exp(hp.a1*(Ts) + hp.a2)
        k_f  = hp.k_f0 + hp.beta * rho_f
        Bi  = Nu_val * X * (air.k_a / k_f)
        theta  = Bi * (1.0 + 1.0/Ja_val)
        Ts_new = op.T_a - (op.T_a - op.T_w) / (1.0 + theta)
        if abs(Ts_new - Ts) < tol:
            return Ts_new, rho_f, k_f, Bi, theta
        Ts = Ts_new

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
    T_w = -4.0      # °C Wandtemperatur
    RH  = 0.8       # Relative Luftfeuchtigkeit
    u_a = 1.0       # m/s
    p = 103500      # Pa

    # Feuchte-Luft
    w_w = HAPropsSI("W", "T", T_w+273.15, "P", p, "R", 1.0)
    w_a = HAPropsSI("W", "T", T_a+273.15, "P", p, "R", RH)
    w_tilde = w_a - w_w
    op = OpPoint(T_a=T_a, T_w=T_w, w_tilde=w_tilde, u_a=u_a)

    # Hermes-Parameter
    a0   = 207.3
    a1   = 0.266
    k_f0 = 0.132     # W/mK
    i_sv = 2830000    # J/kg
    beta = 0.000313
    hp = HermesParams(a0=a0, a1=a1, a2=(-0.0615)*(T_w),
                      k_f0=k_f0, i_sv=i_sv, beta=beta)

    # Geometrie & Luft
    geom = Flatplate(L=0.1)
    k_feucht = HAPropsSI('K', 'T', T_a+273.15, 'P', p, 'R', RH)
    c_p_feucht = HAPropsSI('cp', 'T', T_a+273.15, 'P', p, 'R', RH)
    rho = PropsSI('D', 'T', T_a+273.15, 'P', p,'Air')
    mu = HAPropsSI('M', 'T', T_a+273.15, 'P', p, 'R', RH)
    air  = AirProperty(k_a=k_feucht, c_p=c_p_feucht, rho=rho, mu=mu)

    # Zielzeit (real)
    t_end = 120.0  # min

    # Dimensionslose Zeit definieren
    s0 = 1e-8
    s_end = 6
    N = 10000
    s_array = np.linspace(1e-9, s_end, N)

    # Zustände entlang s
    results = [frost_state_at_s(s, geom, air, op, hp) for s in s_array]
    Ts, rho_f, k_f, Bi, theta = map(np.asarray, zip(*results))
    X = np.array([X_of_s(s, geom, air, op, hp) for s in s_array])
    x_s = X * geom.L * 1e3  # mm

    t_array = []
    for s in s_array:
        t_array.append(t_of_s(s, geom, air, op, hp))

    t_array = np.array(t_array)

    # ---- Plots (wie gehabt) ----

    rech = (theta/(1 + theta)) * (s_array / X)
    plt.plot(X, rech, label=f"Tw = {T_w:.0f}°C")
    plt.xlabel("X [-]")
    plt.ylabel(r"$\frac{\theta}{1+\theta}\,\frac{s}{X}$ [-]")
    plt.title("Hermes (2012)")
    plt.xlim([0, 0.12])
    plt.ylim([0, 30])
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    plt.plot(t_array/60, x_s, label=f"Tw = {T_w:.0f}°C")
    plt.text(0.05, 0.95,
             fr"$\tilde{{\omega}}={w_tilde:.4f},\;\tilde{{T}}={T_tilde(op,hp):.1f},\;Nu={Nu(geom,air,op):.0f}$",
             fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)
    plt.xlabel("Zeit [min]"); plt.ylabel("Frostdicke $x_s$ [mm]")
    plt.title("Hermes (2012)")
    plt.xlim([0, 120])
    plt.ylim([0, 5])
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()


    plt.plot(t_array/60, Ts, label=f"Tw = {T_w:.0f}°C")
    plt.xlabel("Zeit [min]"); plt.ylabel("Frost-Oberflächentemperatur [°C]")
    plt.title("Hermes (2012)")
    plt.xlim([0, t_end])
    plt.ylim([T_w, T_a])
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    plt.plot(t_array/60, rho_f, label=f"Tw = {T_w:.0f}°C")
    plt.xlabel("Zeit [min]"); plt.ylabel("mittlere Frostdichte [kg/m³]")
    plt.title("Hermes (2012)")
    plt.xlim([0, t_end])
    plt.ylim([0, 318])
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    plt.plot(t_array/60, k_f, label=f"Tw = {T_w:.0f}°C")
    plt.xlabel("Zeit [min]"); plt.ylabel("Frost Wärmeleitfähigkeit [W/mK]")
    plt.title("Hermes (2012)")
    plt.xlim([0, t_end])
    #plt.ylim([0, 5])
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    plt.plot(t_array/60, Bi, label=f"Tw = {T_w:.0f}°C")
    plt.xlabel("Zeit [min]"); plt.ylabel("Biotzahl [-]")
    plt.title("Hermes (2012)")
    plt.xlim([0, t_end])
    #plt.ylim([0, 5])
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
