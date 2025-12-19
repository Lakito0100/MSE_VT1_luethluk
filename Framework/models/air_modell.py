from CoolProp.HumidAirProp import HAPropsSI
from Framework.models.Frostmodell_V1 import Frostmodell_Finn_and_Tube
import math

class Air:

    def __init__(self):
        self.ft_model = Frostmodell_Finn_and_Tube()

    def p_ws_buck_Pa(self,T_C: float) -> float:
        """Saturation vapor pressure [Pa]; Buck (1981): water for T>=0°C, ice for T<0°C."""
        if T_C >= 0.0:
            # over water, result in hPa -> Pa
            return 100.0 * 6.1121 * math.exp((18.678 - T_C / 234.5) * (T_C / (257.14 + T_C)))
        else:
            # over ice, hPa -> Pa
            return 100.0 * 6.1115 * math.exp((23.036 - T_C / 333.7) * (T_C / (279.82 + T_C)))

    def w_sat(self,T_C: float, p_Pa: float) -> float:
        pws = min(self.p_ws_buck_Pa(T_C), 0.99 * p_Pa)
        return 0.62198 * pws / (p_Pa - pws)

    def pw_from_w(self,w: float, p_Pa: float) -> float:
        """Water vapor partial pressure from humidity ratio w [kg/kg_da]."""
        return p_Pa * w / (0.62198 + max(w, 0.0))

    def h_moist_da_Jpkg(self,T_C: float, w: float) -> float:
        """Moist air enthalpy per kg dry air [J/kg_da], T in °C."""
        return 1000.0 * (1.006 * T_C + w * (2501.0 + 1.86 * T_C))

    def T_from_h_w_C(self,h_Jpkg: float, w: float) -> float:
        """Invert h = (1.006 + 1.86 w) T + 2501 w  (kJ/kg_da) for T in °C."""
        denom = 1000.0 * (1.006 + 1.86 * w)
        return (h_Jpkg - 1000.0 * 2501.0 * w) / denom

    def propagate_inplace(self,
                          cfg_in,
                          cfg_out,
                          s_frost_bevor,
                          st_seg,
                          geom,
                          m_dot_a: float,
                          Q_seg: float,
                          m_s_seg: float,
                          dt: float,
                          dp_seg: float = 0.0):

        T_in = cfg_in.T_a
        w_in = cfg_in.w_amb
        p_in = cfg_in.p_a

        p_out = p_in - dp_seg

        # Prefer dry-air mass flow because w is defined per kg dry air
        m_dot_ha = m_dot_a
        m_dot_da = m_dot_ha / (1.0 + w_in)

        # 1) moisture update
        w_out = w_in - m_s_seg / m_dot_da
        w_out = max(w_out, 0.0)

        # 2) enthalpy update (TOTAL heat Q_seg includes latent)
        h_in = self.h_moist_da_Jpkg(T_in, w_in)
        h_out = h_in - Q_seg / m_dot_da

        # 3) compute outlet temperature from (h_out, w_out)
        T_out = self.T_from_h_w_C(h_out, w_out)

        # 4) enforce saturation (numerical safety)
        #wsat_out = self.w_sat(T_out, p_out)
        #if w_out > wsat_out:
            # clamp very slightly below saturation to avoid RH=1+ε
        #    w_out = wsat_out * (1.0 - 1e-9)

            # optional (recommended): make enthalpy consistent with the clamped w_out
            # by re-solving T_out from the same h_out and the adjusted w_out:
        #    T_out = self.T_from_h_w_C(h_out, w_out)

        # 5) RH from partial pressure ratio (no CoolProp call)
        pws = self.p_ws_buck_Pa(T_out)
        pw = self.pw_from_w(w_out, p_out)
        RH_out = max(0.0, min(1.0, pw / pws))

        # 6) density (you can keep CoolProp Vha here if you want;
        # otherwise ideal-gas mixture fallback)
        # rho_out = ...

        # Update velocity with a protected free-flow area
        A = geom.h_fin * max(geom.fin_gap() - 2.0 * s_frost_bevor, 1e-9)
        R_da = 287.058
        T_K = T_out + 273.15
        rho_da = p_out / (R_da * T_K * (1.0 + 1.6078 * w_out))  # kg_dry_air / m³
        rho_moist = rho_da * (1.0 + w_out)  # kg_moist_air / m³
        v_out = m_dot_ha / (A * rho_moist)

        cfg_out.T_a = T_out
        cfg_out.w_amb = w_out
        cfg_out.p_a = p_out
        cfg_out.RH = RH_out
        cfg_out.rho_amb = rho_moist
        cfg_out.v_a = v_out

        return T_out, w_out, p_out


