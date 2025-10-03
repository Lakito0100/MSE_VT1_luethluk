
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


# Strategies --------------------------------------------------------

class NusseltModel(Protocol):
    def Nu(self, geom: Flatplate, air: AirProperty, op: OperatingPoint) -> float: ...

class LienhardTurbulentFlatPlateNu:
    def Nu(self, geom, air, op) -> float:
        Re = air.rho * op.u_a * geom.L / air.mu
        return 0.037 * (Re**0.8) * (air.Pr**0.43)

# Frostmodell -------------------------------------------------------

class IFrostModel(ABC):
    @abstractmethod
    def X_of_s(self, s, geom, air, op) -> float: ...

class Hermes2012Analytical(IFrostModel):
    def __init__(self, nu_model: NusseltModel, params: HermesParams):
        self.nu_model = nu_model
        self.p = params

    def Ja(self, op) -> float:
        return (air.c_p * (op.T_a - op.T_w)) / (self.p.i_sv * op.w_a_minus_w_w)

    def Nu(self, geom, air, op) -> float:
        return self.nu_model.Nu(geom, air, op)

    def X_of_s(self, s, geom, air, op) -> float:
        T_tilde = self.p.a1 * (op.T_a - op.T_w)
        k_tilde = self.p.k_f0 / air.k_a
        d0 = op.w_a_minus_w_w*(2+T_tilde) /((1+T_tilde) * (1 + 1/self.Ja(op)))
        d1 = (k_tilde/self.Nu(geom, air, op)) * (2+T_tilde) / ((1+T_tilde) * (1 + 1/self.Ja(op)))

        return max(0.0, (math.sqrt(d1*d1 + 4*d0*s) - d1) * 0.5)

    def phi(self, s) -> float:
        Bi = self.Nu(geom, air, op) * self.X_of_s(s, geom, air, op)
        return Bi*(1+1/self.Ja(op))

    def T_s(self, op, s) -> float:
        return op.T_a - (op.T_a - op.T_w)/(1 + self.phi(s))

    def rho_f(self, s) -> float:
        return self.p.a0 * math.exp(self.p.a1*self.T_s(op, s) + self.p.a2)

    def thickness_mm(self, s, geom, air, op) -> float:
        return 1e3 * self.X_of_s(s, geom, air, op) * geom.L

# Time mapping ------------------------------------------------------

class ITtimeMapper(Protocol):
    def t_of_s(self, s: float, air: AirProperty, par: HermesParams, geom: Flatplate) -> float: ...

class ConstantAlphaTimeMapper:
    def t_of_s(self, s: float, model, air: AirProperty, par: HermesParams, geom: Flatplate) -> float:
        return (s * model.rho_f(s) * air.c_p * geom.L**2) / par.k_f0

# Berechnung --------------------------------------------------------

if __name__ == "__main__":
    # Geometrie und Stoffwerte
    geom = Flatplate(L=1)
    air = AirProperty(k_a=0.026, c_p=1005.0, Pr=0.71, rho=1.2, mu=1.8e-5)

    # Betriebszustand
    op = OperatingPoint(
        T_a = 16.0,             # °C
        T_w = -16.0,             # °C
        w_a_minus_w_w = 0.0075, # aus Paper
        u_a = 1.0               # m/s
    )

    # Parameter des Hermes 2012 Modell
    hermes_params = HermesParams(
        a0 = 207,
        a1 = 0.266,             # 1/°C (aus 2.3)
        a2 = -0.615*op.T_w,
        k_f0 = 0.132,           # W/mK
        i_sv = 2.83e6           # J/kg
    )

    model = Hermes2012Analytical(nu_model=LienhardTurbulentFlatPlateNu(), params=hermes_params)

    # Zeitabbildung
    time_map = ConstantAlphaTimeMapper()

    # Ausgabe
    m = 120.0 # Anzahl Minuten
    s_start = 1.0
    s_end = 2.0
    resid = 1e3
    n = 0
    while (resid > 1e-3 and n <= 1000):
        s_end = hermes_params.k_f0 * m*60 / (model.rho_f(s_start) * air.c_p *  geom.L**2)
        resid = math.fabs((s_start - s_end)/s_start)
        s_start = s_end
        n += 1
    print(s_end, resid, n)
    s = np.linspace(0.0, s_end, 1000)
    t = np.array([time_map.t_of_s(ss, model, air, hermes_params, geom) for ss in s])
    xs = np.array([model.thickness_mm(ss, geom, air, op) for ss in s])

    # Prints

    # Plot
    fig, ax = plt.subplots()
    ax.plot(t, xs)

    ax.text(
        0.9, 0.1,
        f"Ta = {op.T_a:.0f} °C \n Tw = {op.T_w:.0f} °C \n ua = {op.u_a:.1f} m/s \n w_tilde = {op.w_a_minus_w_w}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        bbox=dict(boxstyle="square,pad=0.5", facecolor="white", edgecolor="black", alpha=1.0)
    )

    plt.xlim([0, m*60])
    #plt.ylim([0, 5])
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Frostdicke [mm]")
    ax.set_title("Frostdicke über der Zeit (Hermes 2012, Flat plate)")
    ax.grid(True)
    plt.show()