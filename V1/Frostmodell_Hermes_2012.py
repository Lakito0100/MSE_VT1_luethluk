
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
from typing import Protocol, Iterable
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
    a1: float # [1/C°] aus Paper (T = a1*(Ta - Tw))
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

    def X_of_s(self, s, geom, air, op) -> float:
        T_tilde = self.p.a1 * (op.T_a - op.T_w)
        w_tilde = op.w_a_minus_w_w
        Nu = self.nu_model.Nu(geom, air, op)
        Ja = (air.c_p * (op.T_a - op.T_w)) / (self.p.i_sv * w_tilde)
        k_tilde = self.p.k_f0 / air.k_a

        d0 = w_tilde*(2+T_tilde) /((1+T_tilde) * (1 + 1/Ja))
        d1 = (k_tilde/Nu) * (2+T_tilde) / ((1+T_tilde) * (1 + 1/Ja))

        return max(0.0, (math.sqrt(d1*d1 + 4*d0*s) - d1) * 0.5)

    def thickness_mm(self, s, geom, air, op) -> float:
        return 1e3 * self.X_of_s(s, geom, air, op) * geom.L

# Time mapping ------------------------------------------------------

class ITtimeMapper(Protocol):
    def s_of_t(self, t_seconds: float, air: AirProperty, op: OperatingPoint) -> float: ...

class ConstantAlphaTimeMapper:
    # Erste Näherung s = alpha * t
    def __init__(self, alpha: float):
        self.alpha = alpha
    def s_of_t(self, t_seconds: float, air: AirProperty, op: OperatingPoint) -> float:
        return self.alpha * t_seconds

# Berechnung --------------------------------------------------------

if __name__ == "__main__":
    # Geometrie und Stoffwerte
    geom = Flatplate(L=0.2)
    air = AirProperty(k_a=0.026, c_p=1005.0, Pr=0.71, rho=1.2, mu=1.8e-5)

    # Betriebszustand
    op = OperatingPoint(
        T_a = 16.0,             # °C
        T_w = 10.0,             # °C
        w_a_minus_w_w = 0.0075, # aus Paper
        u_a = 1.0               # m/s
    )

    # Parameter des Hermes 2012 Modell
    hermes_params = HermesParams(
        a1 = 0.266,             # 1/°C (aus 2.3)
        k_f0 = 0.132,           # W/mK
        i_sv = 2.83e6           # J/kg
    )

    model = Hermes2012Analytical(nu_model=LienhardTurbulentFlatPlateNu(), params=hermes_params)

    # Zeitabbildung
    time_map = ConstantAlphaTimeMapper(alpha=1.0e-4)

    # Ausgabe
    t = np.linspace(0.0, 3600.0*h, 1000)
    s = np.array([time_map.s_of_t(tt, air, op) for tt in t])
    xs = np.array([model.thickness_mm(ss, geom, air, op) for ss in s])

    # Plot
    plt.figure()
    plt.plot(t, xs)
    plt.xlabel("Zeit [s]")
    plt.ylabel("Frostdicke [mm]")
    plt.title("Frostdicke über der Zeit (Hermes 2012, Flat plate)")
    plt.grid(True)
    plt.show()