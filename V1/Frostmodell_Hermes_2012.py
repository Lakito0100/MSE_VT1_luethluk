
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
from typing import Protocol
import numpy as np
from matplotlib import pyplot as plt


# Core objects ------------------------------------------------------

@dataclass(frozen=True)
class Flatplate:
    L: float # [m] charakteristische Länge

@dataclass(frozen=True)
class AirProperty:
    k_a: float  # [W/mK] Wärmeleitfähigkeit
    c_p: float  # [J/kgK] specifische Wärmekapazität
    Pr: float   # [-] Prandlt Zahl
    rho: float  # [kg/m^3] Dichte
    mu: float   # [Pa*s] dynamische Viskosität

@dataclass(frozen=True)
class OperatingPoint:
    T_a: float              # [°C] Lufttemperatur
    T_w: float              # [°C] Wandtemperatur
    w_a_minus_w_w: float    # [-] Supersättigungsgrad w = w_a - w_w
    u_a: float              # [m/s] Luftgeschwindigkeit

@dataclass(frozen=True)
class HermesParams:
    a0: float # [-] aus Paper
    a1: float # [1/C°] aus Paper
    a2: float # [°C] aus Paper
    k_f0: float # [W/mK] intercept in k_f = k_f0 + b*rho_f (hier nur k_f0 nötig)
    i_sv: float # [J/kg] Latente Desublimationswärme
    beta: float # [-] aus Paper

# Strategies --------------------------------------------------------

class NusseltModel(Protocol):
    def Nu(self, geom: Flatplate, air: AirProperty, op: OperatingPoint) -> float: ...

class LienhardTurbulentFlatPlateNu:
    def Nu(self, geom, air, op) -> float:
        Re = air.rho * op.u_a * geom.L / air.mu
        return 0.037 * (Re**0.8) * (air.Pr**0.43)

# --- Psychrometrie-Helfer (RH -> humidity ratio x) -----------------
# Gültigkeitsbereiche: typische Buck/Sonntag-Approximationen; T in °C, Drücke in Pa.

def _clamp01(z: float) -> float:
    return max(0.0, min(1.0, z))

def p_sat_water(T_C: float) -> float:
    """
    Sättigungsdampfdruck über Wasser [Pa], Buck (ähnlich Sonntag).
    Guter Bereich: ca. -20..+50 °C (für <0 °C lieber Eis nehmen).
    """
    return 611.21 * math.exp((18.678 - T_C/234.5) * (T_C/(257.14 + T_C)))

def p_sat_ice(T_C: float) -> float:
    """
    Sättigungsdampfdruck über Eis [Pa], Buck/Sonntag.
    Guter Bereich: ca. -80..0 °C.
    """
    return 611.15 * math.exp((23.036 - T_C/333.7) * (T_C/(279.82 + T_C)))

def humidity_ratio_from_RH(T_C: float, RH: float, p_atm: float = 101325.0) -> float:
    """
    Feuchteverhältnis x bei Temperatur T_C und relativer Feuchte RH (0..1).
    Nutzt über Wasser für T>=0°C, über Eis für T<0°C.
    x = 0.62198 * pv / (p_atm - pv)
    """
    RH = _clamp01(RH)
    pws = p_sat_water(T_C) if T_C >= 0.0 else p_sat_ice(T_C)
    pv = RH * pws
    return 0.62198 * pv / (p_atm - pv)

def humidity_ratio_saturation(T_C: float, p_atm: float = 101325.0) -> float:
    """
    Sättigungs-Feuchteverhältnis x_sat bei T_C.
    Für T<0°C: über Eis; sonst über Wasser.
    """
    pws = p_sat_ice(T_C) if T_C < 0.0 else p_sat_water(T_C)
    return 0.62198 * pws / (p_atm - pws)

def compute_w_tilde_from_RH(Ta_C: float, Tw_C: float, RH: float, p_atm: float = 101325.0) -> float:
    """
    Berechnet w̃ = x_a - x_w (dimensionsloser Feuchteunterschied) aus
    Umgebung (Ta_C, RH) und Wand (Tw_C, Sättigung).
    Negative Werte -> 0 (keine Vereisung).
    """
    x_a = humidity_ratio_from_RH(Ta_C, RH, p_atm)
    x_w = humidity_ratio_saturation(Tw_C, p_atm)
    return max(0.0, x_a - x_w)

# Frostmodell -------------------------------------------------------

class IFrostModel(ABC):
    @abstractmethod
    def X_of_s(self, s, geom, air, op) -> float: ...

class Hermes2012Analytical(IFrostModel):
    def __init__(self, nu_model: NusseltModel, params: HermesParams):
        self.nu_model = nu_model
        self.p = params

    def Ja(self, air: AirProperty, op: OperatingPoint) -> float:
        return (air.c_p * (op.T_a - op.T_w)) / (self.p.i_sv * op.w_a_minus_w_w)

    def Nu(self, geom, air, op) -> float:
        return self.nu_model.Nu(geom, air, op)

    def X_of_s(self, s, geom, air, op) -> float:
        T_tilde = self.p.a1 * (op.T_a - op.T_w)
        w_tilde = op.w_a_minus_w_w
        Nu = self.Nu(geom, air, op)
        Ja = self.Ja(air, op)
        k_tilde = self.p.k_f0 / air.k_a

        d0 = w_tilde*(2+T_tilde) /((1+T_tilde) * (1 + 1/Ja))
        d1 = (k_tilde/Nu) * (2+T_tilde) / ((1+T_tilde) * (1 + 1/Ja))

        return max(0.0, (math.sqrt(d1*d1 + 4*d0*s) - d1) * 0.5)

    def phi(self, s: float, geom: Flatplate, air: AirProperty, op: OperatingPoint) -> float:
        X = self.X_of_s(s, geom, air, op)
        Nu = self.Nu(geom, air, op)
        ktilde0 = self.p.k_f0 / air.k_a
        Bi = (Nu * X) / ktilde0
        return Bi*(1.0 + 1.0 / self.Ja(air, op))

    def T_s(self, s: float, geom: Flatplate, air: AirProperty, op: OperatingPoint) -> float:
        Ts_raw = op.T_a - (op.T_a - op.T_w) / (1.0 + self.phi(s, geom, air, op))
        return min(Ts_raw, 0.0)

    def rho_f(self, s: float, geom: Flatplate, air: AirProperty, op: OperatingPoint) -> float:
        return self.p.a0 * math.exp(self.p.a1 * self.T_s(s, geom, air, op) + self.p.a2)

    def thickness_mm(self, s: float, geom: Flatplate, air: AirProperty, op: OperatingPoint) -> float:
        return 1e3 * self.X_of_s(s, geom, air, op) * geom.L

    def k_f(self, s: float, geom: Flatplate, air: AirProperty, op: OperatingPoint) -> float:
        return self.p.k_f0 + self.p.beta * self.rho_f(s, geom, air, op)

# Time mapping ------------------------------------------------------

class ITtimeMapper(Protocol):
    def t_of_s(self, s: float, air: AirProperty, par: HermesParams, geom: Flatplate) -> float: ...

class PhysicalTimeMapper:
    def t_of_s(self, s: float, model: Hermes2012Analytical, geom: Flatplate, air: AirProperty, op: OperatingPoint) -> float:
        rho_f = model.rho_f(s, geom, air, op)
        k_f = model.k_f(s, geom, air, op)
        return s * (rho_f * air.c_p * geom.L**2) / (k_f)

class HermesConstTimeMapper:
    # Konstante Abbildung s = alpha * t, mit alpha = k_f0 / (rho_ref * c_p * L^2).
    def __init__(self, alpha: float):
        self.alpha = alpha  # [1/s]
    def t_of_s(self, s: float) -> float:
        return s / self.alpha


# Berechnung --------------------------------------------------------

if __name__ == "__main__":
    # Geometrie und Stoffwerte
    geom = Flatplate(L=0.1)
    air = AirProperty(k_a=0.026, c_p=1005.0, Pr=0.71, rho=1.2, mu=1.8e-5)

    # Betriebszustand
    RH = 0.8
    p_atm = 101325.0

    T_a = 16.0
    T_w = -4.0
    w_tilde = compute_w_tilde_from_RH(T_a, T_w, RH, p_atm)

    op = OperatingPoint(
        T_a = T_a,             # °C
        T_w = T_w,             # °C
        w_a_minus_w_w = w_tilde, # aus Paper
        u_a = 1.0               # m/s
    )

    # Parameter des Hermes 2012 Modell
    hermes_params = HermesParams(
        a0 = 207,
        a1 = 0.266,             # 1/°C (aus 2.3)
        a2 = -0.615*op.T_w,
        k_f0 = 0.132,           # W/mK
        i_sv = 2.83e6,          # J/kg
        beta = 3.0e-4           # m^4 s^-3 K^-1
    )

    model = Hermes2012Analytical(nu_model=LienhardTurbulentFlatPlateNu(), params=hermes_params)

    T_tilde = hermes_params.a1 * (op.T_a - op.T_w)
    Ja = model.Ja(air, op)
    Nu = model.Nu(geom, air, op)
    ktilde0 = hermes_params.k_f0 / air.k_a
    print(f"w̃={op.w_a_minus_w_w:.5f}, T̃={T_tilde:.3f}, Ja={Ja:.3f}, Nu={Nu:.1f}, k̃0={ktilde0:.3f}")

    # Ausgabe
    rho_ref = 100
    alpha = hermes_params.k_f0 / (rho_ref * air.c_p * geom.L ** 2)

    time_map = HermesConstTimeMapper(alpha=alpha)

    # Zielzeit
    m = 120.0
    t_target = m * 60.0
    s_end = alpha * t_target

    # Kurven
    s = np.linspace(0.0, s_end, 400)
    t = np.array([time_map.t_of_s(ss) for ss in s])
    xs = np.array([model.thickness_mm(ss, geom, air, op) for ss in s])
    rho_f_curve = np.array([model.rho_f(ss, geom, air, op) for ss in s])

    # Plot: Frostdicke
    fig, ax = plt.subplots()
    ax.plot(t/60.0, xs)
    ax.text(
        0.9, 0.1,
        f"Ta = {op.T_a:.0f} °C\nTw = {op.T_w:.0f} °C\n RH = {RH*100} % \nua = {op.u_a:.1f} m/s",
        transform=ax.transAxes, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=1.0)
    )
    ax.set_xlabel("Zeit [min]")
    ax.set_ylabel("Frostdicke [mm]")
    ax.set_title("Frostdicke über der Zeit (Hermes 2012, Flat plate)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot: Dichte
    fig, ax = plt.subplots()
    ax.plot(t/60.0, rho_f_curve)
    ax.text(
        0.9, 0.1,
        f"Ta = {op.T_a:.0f} °C\nTw = {op.T_w:.0f} °C\n RH = {RH*100} % \nua = {op.u_a:.1f} m/s",
        transform=ax.transAxes, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=1.0)
    )
    ax.set_xlabel("Zeit [min]")
    ax.set_ylabel("Frostdichte [kg/m^3]")
    ax.set_title("Frostdichte über der Zeit (Hermes 2012, Flat plate)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()