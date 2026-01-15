from CoolProp.HumidAirProp import HAPropsSI
from Framework.models.Frostmodell_V1 import Frostmodell_Finn_and_Tube
import math
import numpy as np

class Air:

    def __init__(self):
        self.ft_model = Frostmodell_Finn_and_Tube()

        self._fan_initialized = False
        self._K_clean = None
        self._sigma_min = 1.0
        self._last_mdot_total = None
        self._last_dp_seg = 0.0

        self._last_rho = None
        self._last_nu = None

    def _fan_enabled(self, cfg) -> bool:
        return bool(getattr(cfg, "use_fan", False))

    def _n_parallel_air_paths(self, geom) -> int:
        n_y = geom.n_seg_r
        stacks = geom.stacks
        return max(1, n_y * stacks)

    def _sigma_from_frost(self, geom, s_frost) -> float:
        gap0 = float(geom.fin_gap())
        gap_eff = max(gap0 - 2.0 * float(s_frost), 1e-9)
        return float(np.clip(gap_eff / gap0, 0.001, 1.0))

    def _dp_sys_kays_london(self, Vdot: float, cfg, geom, sigma_blockage: float) -> float:
        """
        Systemdruckverlust Δp_sys(Vdot) nach Kays & London (wie in SONG2025Chap9, Gl. 9.40–9.44).

        Eingänge:
            Vdot            Volumenstrom [m^3/s]
            sigma_blockage  dimensionsloser Blockage-Faktor (0..1], aus Geometrieverengung (z.B. g_eff/g0)

        Benötigte/angenommene Grössen:
            - Stirnfläche A_fr ~ l_fin*h_fin*stacks
            - Minimale freie Fläche A_min wird aus einem "clean"-Querschnitt und sigma_blockage skaliert
            - Tube pitches Xt/Xl: falls nicht vorhanden -> Xt/Xl = 1
            - Ki, Ko: als Parameter aus cfg (Default 0)
        """
        if Vdot <= 0.0:
            return 0.0

        # Luftzustand (für Fan-Kennlinie reicht ein repräsentatives Dichteniveau)
        rho_in = float(getattr(cfg, "rho_amb", self._last_rho if self._last_rho is not None else 1.2))
        rho_out = float(getattr(cfg, "rho_amb", rho_in))  # ohne Modellierung der Dichteänderung: rho_out ~ rho_in
        nu = float(getattr(cfg, "v_kin", self._last_nu if self._last_nu is not None else 1.5e-5))
        mu = rho_in * nu

        # Verlustbeiwerte (falls nicht vorgegeben: 0)
        Ki = float(getattr(cfg, "Ki_in", 0.0))
        Ko = float(getattr(cfg, "Ko_out", 0.0))

        # Stirnfläche (face area)
        A_fr = float(geom.l_fin * geom.h_fin * getattr(geom, "stacks", 1))

        # Minimale freie Fläche (vereinfachtes Abbild deiner Geometrie):
        # clean: A_min_clean_seg ~ l_tube() * (h_fin - d_tube_a)
        # total: über n_seg_r und stacks skaliert
        g0 = max(float(geom.h_fin - geom.d_tube_a), 1e-9)
        A_min_clean_seg = float(geom.l_tube() * g0)

        n_cols = int(getattr(geom, "n_seg_r", 1))
        stacks = int(getattr(geom, "stacks", 1))
        A_min_clean = A_min_clean_seg * n_cols * stacks

        # Frost/Blockage
        sigma_blockage = float(np.clip(sigma_blockage, 1e-6, 1.0))
        A_min = A_min_clean * sigma_blockage

        # σ in Gl. (9.40): Verhältnis min. freie Fläche / Stirnfläche
        sigma = float(np.clip(A_min / max(A_fr, 1e-12), 1e-6, 1.0))

        # Massenstromdichte bezogen auf A_min
        mdot = rho_in * Vdot
        G_a = mdot / max(A_min, 1e-12)  # [kg/(m^2 s)]

        # Charakteristische Durchmesser / Pitches
        d = float(geom.d_tube_a)
        Dc = float(geom.d_tube_a + 2.0 * geom.fin_thickness)  # fin collar outside diameter

        Xt = float(getattr(geom, "X_t", getattr(geom, "Xt", 1.0)))
        Xl = float(getattr(geom, "X_l", getattr(geom, "Xl", Xt)))
        if Xt <= 0.0 or Xl <= 0.0:
            Xt, Xl = 1.0, 1.0

        # Reynolds-Zahl auf Dc: Re = G_a*Dc/mu
        Re_Dc = max(G_a * Dc / max(mu, 1e-12), 1.0)

        # Korrelationsexponenten (Gl. 9.42–9.44)
        Nprime = max(float(getattr(geom, "n_seg_l", 1)), 1.0)  # Tube rows ~ n_seg_l
        F1 = -0.764 + 0.739 * (Xt / Xl) + 0.177 * (d / Dc) - 0.00758 / Nprime
        F2 = -15.689 + 64.021 / math.log(Re_Dc)
        F3 = 1.696 - 15.695 / math.log(Re_Dc)

        # Reibungsbeiwert (Gl. 9.41) – mit F3 für (d/Dc)-Exponent (konsistent zu 9.44)
        f = 0.0267 * (Re_Dc ** F1) * ((Xt / Xl) ** F2) * ((d / Dc) ** F3)

        # Hydraulischer Durchmesser: Dh = 4*A_min*L'/A_wet  (über "wetted perimeter" angenähert)
        # A_wet_total aus Segmentflächen (fin+tube) skaliert; L' als Strömungslänge ~ n_rows * Xl
        n_rows = int(getattr(geom, "n_seg_l", 1))
        A_wet_seg = float(getattr(geom, "A_one_segment")())
        A_wet = A_wet_seg * n_rows * n_cols * stacks
        Lprime = max(float(n_rows * Xl), 1e-9)
        Dh = max(4.0 * A_min * Lprime / max(A_wet, 1e-12), 1e-9)

        # Dichte-Mittelterm (1/rho)_m
        inv_rho_m = 0.5 * (1.0 / max(rho_in, 1e-12) + 1.0 / max(rho_out, 1e-12))

        # Gl. (9.40)
        term_in = (1.0 - sigma ** 2 + Ki)
        term_acc = 2.0 * (rho_in / max(rho_out, 1e-12) - 1.0)
        term_fric = 4.0 * f * (Lprime / Dh) * rho_in * inv_rho_m
        term_out = (1.0 - sigma ** 2 + Ko) * (rho_in / max(rho_out, 1e-12))

        dp = (G_a ** 2) / (2.0 * rho_in) * (term_in + term_acc + term_fric - term_out)
        return max(float(dp), 0.0)

    def _solve_fan_operating_point(self, cfg, geom) -> float:
        """
        Bestimme Vdot aus Δp_fan(Vdot)=Δp_sys(Vdot) via Bisektion.
        Gibt mdot_total zurück.
        """
        dp0 = float(getattr(cfg, "fan_dp0", 0.0))
        V0 = float(getattr(cfg, "fan_V0", 0.0))
        if dp0 <= 0.0 or V0 <= 0.0:
            # Fallback: keine Fan-Daten -> mdot bleibt wie cfg.m_dot
            return float(getattr(cfg, "m_dot", 0.0))

        # Fan curve
        def dp_fan(V):
            V = max(V, 0.0)
            if V >= V0:
                return 0.0
            return dp0 * (1.0 - (V / V0) ** 2)

        sigma_blockage = float(np.clip(self._sigma_min, 1e-6, 1.0))
        #SIGMA_FLOOR = 0.10
        #sigma_blockage = max(float(sigma_blockage), SIGMA_FLOOR)

        # Residuum
        def F(V):
            return dp_fan(V) - self._dp_sys_kays_london(V, cfg, geom, sigma_blockage)

        # Bisektion im Intervall [0, V0]
        a, b = 0.0, V0
        Fa, Fb = F(a), F(b)
        if Fa <= 0.0:
            V_star = a
        elif Fb >= 0.0:
            V_star = b
        else:
            V_star = 0.5 * (a + b)
            for _ in range(50):
                V_star = 0.5 * (a + b)
                Fm = F(V_star)
                if abs(Fm) < 1e-6:
                    break
                if Fm > 0.0:
                    a = V_star
                else:
                    b = V_star

        rho = float(getattr(cfg, "rho_amb", self._last_rho if self._last_rho is not None else 1.2))
        mdot_total = rho * V_star
        #MDOT_FLOOR = 0.05  # kg/s (an deine Anlage anpassen)
        #mdot_total = max(float(mdot_total), MDOT_FLOOR)
        return float(max(mdot_total, 0.0))

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
                          Q_sens_seg: float,
                          m_s_seg: float,
                          dt: float,
                          dp_seg: float = 0.0):

        T_in = cfg_in.T_a
        w_in = cfg_in.w_amb
        p_in = cfg_in.p_a

        #if self._fan_enabled(cfg_in) and getattr(cfg_in, "fan_master", False):
        #    npar = self._n_parallel_air_paths(geom)
        #
        #    # 1) Frost -> sigma (konservativ)
        #    sigma = self._sigma_from_frost(geom, s_frost_bevor)
        #    self._sigma_min = min(self._sigma_min, sigma)
        #
        #    # 2) dp bookkeeping (aus deiner dp-Korrelation / aus dem Simulator übergeben)
        #    self._last_dp_seg = float(dp_seg)
        #
        #    # 3) Operating point: löse mdot_total aus Ventilator- und Systemkurve
        #    mdot_total = self._solve_fan_operating_point(cfg_in, geom)
        #
        #    # Gesamtstrom zurückschreiben
        #    cfg_in.m_dot = float(mdot_total)
        #
        #    # Strom pro Pfad für dieses Segment
        #    m_dot_a = float(mdot_total) / npar
        #
        #    self._last_mdot_total = float(mdot_total)

        p_out = p_in - dp_seg

        # Prefer dry-air mass flow because w is defined per kg dry air
        m_dot_ha = m_dot_a
        m_dot_da = m_dot_ha / (1.0 + w_in)

        m_s_max = 0.999999 * w_in * m_dot_da  # [kg/s] max removable vapor
        m_s_used = min(m_s_seg, m_s_max)

        # 1) moisture update
        w_out = w_in - m_s_used / m_dot_da
        if cfg_out.frost_condition:
            w_out = max(w_out, st_seg.w_ft[-1])
        else:
            w_out = max(w_out, 0.0)

        # 2) enthalpy update (TOTAL heat Q_seg includes latent)
        h_min = self.h_moist_da_Jpkg(st_seg.T_ft[-1], w_out)


        Q_tot = float(Q_sens_seg) + float(cfg_in.h_sub) * m_s_used
        h_in = self.h_moist_da_Jpkg(T_in, w_in)
        Q_max = (h_in - h_min) * m_dot_da

        Q_tot_eff = min(Q_tot, Q_max)

        h_out = h_in - Q_tot_eff / m_dot_da

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

