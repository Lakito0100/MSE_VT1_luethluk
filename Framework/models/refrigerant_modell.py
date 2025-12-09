from time import perf_counter
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from numpy.linalg import solve
from CoolProp.CoolProp import PropsSI


@dataclass
class RefGeomParams:
    A_flow: float
    dx: float
    A_inner: float
    V_wall: float
    rho_wall: float
    c_wall: float
    dp_ref_seg: float = 0.0


class Refrigerant:
    def __init__(self, geom_params: RefGeomParams,
                 connection_path=None):
        self.geom = geom_params

        if connection_path is None:
            self.connection_path = [
                (4,4),(4,3),(4,2),(4,1),(4,0),
                (3,0),(3,1),(3,2),(3,3),(3,4),
                (2,4),(2,3),(2,2),(2,1),(2,0),
                (1,0),(1,1),(1,2),(1,3),(1,4),
                (0,4),(0,3),(0,2),(0,1),(0,0)
            ]
        else:
            self.connection_path = connection_path

    # ------------------------------------------------------------------
    # Lokale Korrelation für h_int (Wärmeübergang Wand ↔ Kältemittel)
    # Aktuell nur Platzhalter: konstanter Wert 10 W/m²K
    # ------------------------------------------------------------------
    def _h_int_corr(self,
                    p_i: float,
                    h_i: float,
                    m_dot_ref: float,
                    Q_seg_i: float,
                    geom,
                    fluid: str,
                    orientation: str = "horizontal") -> float:
        """
        Interner Wärmeübergangskoeffizient h_int zwischen Rohrinnenwand und Kältemittel.

        - bestimmt zuerst die Phase (flüssig / zweiphasig / gasförmig) über CoolProp
        - zweiphasig: Shah (2022) – neue Korrelation für gesättigtes Sieden in Mini-/Makrokanälen
        - einphasig: Dittus-Boelter (turbulent) bzw. Nu=3.66 (laminar)

        Parameter
        ---------
        p_i        : Druck im Segment [Pa]
        h_i        : Enthalpie im Segment [J/kg]
        m_dot_ref  : Massenstrom durch das Rohr [kg/s]
        Q_seg_i    : Wärmestrom in das Kältemittel in diesem Segment [W] (vom Frostmodell)
        geom       : FinnTubedHX-Objekt (benutzt: d_tube_a, tube_thickness, n_seg_l, l_fin)
        fluid      : CoolProp-Fluidstring, z.B. "R134a"
        orientation: "horizontal" oder "vertical" (für Froude‐/Oberflächenspannungs-Faktor)

        Rückgabe
        --------
        h_int [W/m²K]
        """

        # ---------------- Geometrie / Wärmestrom -> q'' ---------------------------------------------------------------
        d_o = geom.d_tube_a                          # Außendurchmesser
        t_w = geom.tube_thickness
        d_i = d_o - 2.0 * t_w                        # Innendurchmesser
        if d_i <= 0:
            raise ValueError("Innendurchmesser <= 0 – prüfe d_tube_a und tube_thickness")

        # Strömungsquerschnitt und Massenfluss
        A_flow = 0.25 * np.pi * d_i**2               # m²
        G = m_dot_ref / A_flow                       # kg/(m² s)

        # Segmentlänge in Strömungsrichtung (angenommen: l_fin / n_seg_l)
        L_tot = geom.l_fin
        L_seg = L_tot / geom.n_seg_l
        A_i_seg = np.pi * d_i * L_seg                # innere Oberfläche des Segments

        # Wärmestromdichte (Vorzeichen: Betrag für Boiling Number)
        if A_i_seg <= 0:
            raise ValueError("A_i_seg <= 0 – prüfe Geometrie")
        q_flux = Q_seg_i / A_i_seg                   # W/m²
        q_flux_abs = abs(q_flux)

        # ---------------- Phase aus CoolProp bestimmen ---------------------------------------------------------------
        try:
            x = PropsSI("Q", "P", p_i, "H", h_i, fluid)   # Dampfqualität
        except ValueError:
            x = float("nan")

        # Zweiphasig: 0 < x < 1 -> Shah 2022
        if np.isfinite(x) and 0.0 < x < 1.0:
            return self._h_shah_2022_boiling(
                p=p_i,
                G=G,
                x=x,
                q_flux=q_flux_abs,
                D_i=d_i,
                fluid=fluid,
                orientation=orientation,
            )

        # Einphasig: Dittus-Boelter
        # (funktioniert sowohl für Flüssigkeit als auch für Dampf – nur die Stoffwerte ändern sich)
        T = PropsSI("T", "P", p_i, "H", h_i, fluid)
        mu = PropsSI("V", "P", p_i, "H", h_i, fluid)        # Viskosität [Pa·s]
        k = PropsSI("L", "P", p_i, "H", h_i, fluid)         # Wärmeleitfähigkeit [W/mK]
        Pr = PropsSI("PRANDTL", "P", p_i, "H", h_i, fluid)  # Prandtl-Zahl [-]

        Re = G * d_i / mu

        if Re < 2300.0:
            # laminarer Rohrfluss – konstanter Wärmestrom
            Nu = 3.66
        else:
            # Dittus-Boelter, beheiztes Fluid
            Nu = 0.023 * Re**0.8 * Pr**0.4

        h_int = Nu * k / d_i
        return h_int

    # ------------------------------------------------------------------------------------------------------------------
    # Shah (2022) – Korrelation für gesättigtes Sieden in Mini-/Makrokanälen
    # ------------------------------------------------------------------------------------------------------------------
    def _h_shah_2022_boiling(self,
                             p: float,
                             G: float,
                             x: float,
                             q_flux: float,
                             D_i: float,
                             fluid: str,
                             orientation: str = "horizontal") -> float:
        """
        Shah (2022) – neue allgemeine Korrelation für gesättigtes Sieden in Mini-/Makrokanälen. :contentReference[oaicite:1]{index=1}

        Parameter wie im Paper:
        h_TP = F_st * ψ * h_LS
        mit ψ = max(ψ0, ψ_cb, ψ_bs)
        """

        g = 9.81
        D_HYD = D_i      # glattes Rundrohr: hydraulischer & beheizter Durchmesser gleich
        D_HP = D_i

        # Sättigungseigenschaften
        rho_L = PropsSI("D", "P", p, "Q", 0, fluid)
        rho_G = PropsSI("D", "P", p, "Q", 1, fluid)
        mu_L = PropsSI("V", "P", p, "Q", 0, fluid)
        k_L = PropsSI("L", "P", p, "Q", 0, fluid)
        Pr_L = PropsSI("PRANDTL", "P", p, "Q", 0, fluid)
        sigma = PropsSI("I", "P", p, "Q", 0, fluid)        # Oberflächenspannung :contentReference[oaicite:2]{index=2}

        h_L = PropsSI("H", "P", p, "Q", 0, fluid)
        h_G = PropsSI("H", "P", p, "Q", 1, fluid)
        h_LG = h_G - h_L                                   # Verdampfungsenthalpie

        # Qualität ein bisschen clampen, um Division durch 0 zu vermeiden
        x_clamp = float(np.clip(x, 1e-4, 0.9999))

        # dimensionslose Kennzahlen (Definitionen im Paper/Nomenklatur) :contentReference[oaicite:3]{index=3}
        Bo = q_flux / (G * h_LG)                           # Boiling number
        Co = ((1.0 / x_clamp) - 1.0)**0.8 * (rho_G / rho_L)**0.5
        We_GT = G**2 * D_HYD / (rho_G * sigma)
        Fr_LT = G**2 / (rho_L**2 * g * D_HYD)

        # Einphasen-HTC der Flüssigkeit (h_LS, Eq. (9)) :contentReference[oaicite:4]{index=4}
        Re_LS = G * (1.0 - x_clamp) * D_HP / mu_L
        h_LS = 0.023 * Re_LS**0.8 * Pr_L**0.4 * (k_L / D_HP)

        # Parameter J (Eq. (5)) – mit abgeschnittener Klammer, damit J > 0 bleibt
        if orientation.lower().startswith("h"):
            if Fr_LT < 0.04:
                n = 1.0
            else:
                n = 0.0
        else:
            n = 0.0

        base = max(0.38 * Fr_LT - 0.3, 1e-6)
        J = (base**n) * Co

        # ψ0 nach neuer Korrelation (Eq. (21)/(22)) :contentReference[oaicite:5]{index=5}
        if fluid.upper() in ("CO2", "R744"):
            psi0 = 1820.0 * Bo**0.68
        else:
            psi0 = 1.0 + 560.0 * Bo**0.65

        # ψ_cb und ψ_bs (Eq. (23),(24)) :contentReference[oaicite:6]{index=6}
        psi_cb = 2.0 / (J**0.8)
        psi_bs = psi0 * (1.0 + 0.16 / (J**0.87))

        psi = max(psi0, psi_cb, psi_bs)

        # Oberflächenspannungs-Faktor F_st (Eq. (12) + Anpassung für Fr_LT<0.04)
        if orientation.lower().startswith("h") and Fr_LT < 0.04:
            F_st = 1.0
        else:
            F_st = 2.1 - 0.008 * We_GT - 110.0 * Bo
            if F_st < 1.0:
                F_st = 1.0

        h_TP = F_st * psi * h_LS
        return h_TP


    def update_all_segments(self,
                            cfg_inlet,
                            cfg_grid,
                            st_grid,
                            geom,
                            Q_seg_list,
                            time: float,
                            dt: float,):
        """
        Integriert p_i, h_i, T_w,i von t_outer bis t_outer+dt_outer.
        Benutzt BDF mit inneren Zeitschritten <= dt_inner.
        In jedem Segment wird das 2x2-System aus (Massen- und
        Energieerhaltung) nach [dp_i/dt, dh_i/dt] gelöst.
        """

        fluid = cfg_inlet.ref_str
        gp = self.geom
        path = self.connection_path
        n_seg = len(path)

        # ----------------------------------------------------------
        # 1) Massenstrom aus Eintrittszustand (konstant entlang Pfad)
        # ----------------------------------------------------------
        rho_in = PropsSI("D", "P", cfg_inlet.p_ref,
                              "H", cfg_inlet.h_ref, fluid)
        m_dot_ref = rho_in * cfg_inlet.V_dot_ref
        if m_dot_ref <= 0.0:
            raise ValueError("Refrigerant: m_dot_ref <= 0 from inlet state.")

        # ----------------------------------------------------------
        # 2) Anfangszustand y0 = [p_0..p_N-1, h_0..h_N-1, Tw_0..Tw_N-1]
        # ----------------------------------------------------------
        p0 = np.zeros(n_seg)
        h0 = np.zeros(n_seg)
        Tw0 = np.zeros(n_seg)

        for k, (ix, iy) in enumerate(path):
            cfg = cfg_grid[ix][iy]
            p0[k] = cfg.p_ref
            h0[k] = cfg.h_ref
            T_tube_C = getattr(cfg, "T_tube", cfg.T_ref)
            Tw0[k] = T_tube_C + 273.15  # °C → K

        y0 = np.concatenate([p0, h0, Tw0])

        # ----------------------------------------------------------
        # 3) RHS der ODE: berechne [dp/dt, dh/dt, dTw/dt]
        # ----------------------------------------------------------
        def rhs(t, y):
            p = y[0:n_seg]
            h = y[n_seg:2*n_seg]
            Tw = y[2*n_seg:3*n_seg]

            dpdt = np.zeros_like(p)
            dhdt = np.zeros_like(h)
            dTwdt = np.zeros_like(Tw)

            for k, (ix, iy) in enumerate(path):

                p_i = p[k]
                h_i = h[k]

                # --- Thermodynamische Größen -----------------------
                rho_i = PropsSI("D", "P", p_i, "H", h_i, fluid)

                # ∂ρ/∂p|h  und  ∂ρ/∂h|p
                try:
                    drho_dp = PropsSI("d(Dmass)/d(P)|H",
                                      "P", p_i, "H", h_i, fluid)
                except ValueError:
                    drho_dp = 0.0

                try:
                    drho_dh = PropsSI("d(Dmass)/d(Hmass)|P",
                                      "P", p_i, "H", h_i, fluid)
                except ValueError:
                    drho_dh = 0.0

                # Y(h,p) und Z(h,p)
                Y_i = h_i * drho_dh + rho_i
                Z_i = h_i * drho_dp - 1.0

                # --- Konvektiver Term: upstream-Enthalpie ----------
                if k == 0:
                    h_up = cfg_inlet.h_ref
                else:
                    h_up = h[k-1]

                # --- Wärmetransfer Wand → Kältemittel --------------
                T_ref_K = PropsSI("T", "P", p_i, "H", h_i, fluid)

                h_int_i = self._h_int_corr(
                    p_i=p_i,
                    h_i=h_i,
                    m_dot_ref=m_dot_ref,
                    Q_seg_i=Q_seg_list[ix, iy],
                    geom=geom,
                    fluid=fluid
                )

                Q_ref_i = h_int_i * gp.A_inner * (Tw[k] - T_ref_K)  # [W]
                qdot_i = Q_ref_i / (gp.A_flow * gp.dx)               # [W/m³]

                # RHS der Energiegleichung (2):
                rhs_energy = (
                    -m_dot_ref / (gp.A_flow * gp.dx) * (h_i - h_up)
                    + rho_i * qdot_i
                )

                # --- 2x2-System für [dp_i/dt, dh_i/dt] lösen --------
                #   [ drho_dp  drho_dh ] [dp/dt] = [ 0        ]
                #   [   Z_i      Y_i   ] [dh/dt]   [ rhs_energy ]
                A_loc = np.array([[drho_dp, drho_dh],
                                  [Z_i,     Y_i    ]], dtype=float)
                b_loc = np.array([0.0, rhs_energy], dtype=float)

                try:
                    dp_i_dt, dh_i_dt = solve(A_loc, b_loc)
                except np.linalg.LinAlgError as e:
                    raise RuntimeError(
                        f"Singuläres lokales System in Segment {k} bei t={t:.3f}s: {e}"
                    )

                dpdt[k] = dp_i_dt
                dhdt[k] = dh_i_dt

                # --- Wand-ODE (3): dTw/dt --------------------------
                Q_f_i = Q_seg_list[ix][iy]       # [W] von Luft/Frost in äußere Wand
                dTwdt[k] = (Q_f_i - Q_ref_i) / (gp.rho_wall * gp.c_wall * gp.V_wall)

            return np.concatenate([dpdt, dhdt, dTwdt])

        # ----------------------------------------------------------
        # 4) Integration über ein äußeres Zeitintervall
        # ----------------------------------------------------------
        t0 = time
        t1 = time + dt

        cycl_t0_ref = perf_counter()
        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0,
            method="BDF",
            rtol=1e-6,
            atol=1e-8,
            max_step=dt,
        )
        cycl_t1_ref = perf_counter()

        if not sol.success:
            raise RuntimeError(
                f"Refrigerant ODE solver failed: {sol.message}"
            )

        # ein wenig Output wie bei Finn & Tube
        n_inner = len(sol.t) - 1
        y_end = sol.y[:, -1]
        p_end = y_end[0:n_seg]
        h_end = y_end[n_seg:2*n_seg]
        Tw_end = y_end[2*n_seg:3*n_seg]

        # Beispiel: Auslass-Temperatur aus erstem Segment (Pfadanfang)
        p_out0 = p_end[0]
        h_out0 = h_end[0]
        T_out0_K = PropsSI("T", "P", p_out0, "H", h_out0, fluid)
        T_out0_C = T_out0_K - 273.15
        mean_Tw_C = np.mean(Tw_end) - 273.15

        ref_time = cycl_t1_ref - cycl_t0_ref
        print("Refrigerant Domain Inner Steps: " + str(n_inner) +
              " \t T_out_ref: " + f"{T_out0_C:.3e}" +
              " \t mean T_tube: " + f"{mean_Tw_C:.3e}" +
              " \t cycle time: " + f"{ref_time:.3f} s")

        # ----------------------------------------------------------
        # 5) Endzustand zurück in cfg_grid schreiben
        # ----------------------------------------------------------
        for k, (ix, iy) in enumerate(path):
            cfg = cfg_grid[ix][iy]

            p_out = float(p_end[k])
            h_out = float(h_end[k])

            T_out_K = PropsSI("T", "P", p_out, "H", h_out, fluid)
            T_out_C = T_out_K - 273.15

            rho_out = PropsSI("D", "P", p_out, "H", h_out, fluid)
            V_dot_out = m_dot_ref / rho_out

            try:
                x_out = PropsSI("Q", "P", p_out, "H", h_out, fluid)
            except ValueError:
                x_out = float("nan")

            cfg.p_ref = p_out
            cfg.h_ref = h_out
            cfg.T_ref = T_out_C
            cfg.V_dot_ref = V_dot_out
            cfg.x_ref = x_out
            cfg.T_tube = float(Tw_end[k] - 273.15)