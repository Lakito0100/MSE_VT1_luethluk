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
    def h_int_corr(self,
                    x:float):
        return 50.0


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

                cfg = cfg_grid[ix][iy]
                p_i = p[k]
                h_i = h[k]

                # --- Thermodynamische Größen -----------------------
                rho_i = PropsSI("D", "P", p_i, "H", h_i, fluid)

                drho_dp = PropsSI("d(Dmass)/d(P)|H",
                                      "P", p_i, "H", h_i, fluid)

                drho_dh = PropsSI("d(Dmass)/d(Hmass)|P",
                                      "P", p_i, "H", h_i, fluid)

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

                h_int_i = self.h_int_corr(
                    x=cfg.x_ref
                )

                Q_ref_i = h_int_i * gp.A_inner * (Tw[k] - T_ref_K)  # [W]
                qdot_i = Q_ref_i / (gp.A_inner * gp.dx)               # [W/m³]

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