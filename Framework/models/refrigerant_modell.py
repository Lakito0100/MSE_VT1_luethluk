from time import perf_counter
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from CoolProp.CoolProp import PropsSI


@dataclass
class RefGeomParams:
    A_flow: float
    dx: float
    A_inner: float
    V_wall: float
    rho_wall: float
    c_wall: float
    h_int: float
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

    def update_all_segments(self,
                            cfg_inlet,
                            cfg_grid,
                            st_grid,
                            Q_seg_list,
                            t_outer: float,
                            dt_outer: float,
                            dt_inner: float):
        fluid = cfg_inlet.ref_str
        gp = self.geom
        path = self.connection_path
        n_seg = len(path)

        # ---------------- mass flow ----------------
        rho_in = PropsSI("D", "P", cfg_inlet.p_ref,
                         "H", cfg_inlet.h_ref, fluid)
        m_dot_ref = rho_in * cfg_inlet.V_dot_ref
        if m_dot_ref <= 0.0:
            raise ValueError("Refrigerant: m_dot_ref <= 0 from inlet state.")

        # ---------------- initial state y0 ----------------
        h0 = np.zeros(n_seg)
        Tw0 = np.zeros(n_seg)
        for k, (ix, iy) in enumerate(path):
            cfg = cfg_grid[ix][iy]
            h0[k] = cfg.h_ref
            T_tube_C = getattr(cfg, "T_tube", cfg.T_ref)
            Tw0[k] = T_tube_C + 273.15  # °C→K
        y0 = np.concatenate([h0, Tw0])

        # ---------------- RHS (same as before) ----------------
        def rhs(t, y):
            h = y[:n_seg]
            Tw = y[n_seg:]
            dhdt = np.zeros_like(h)
            dTwdt = np.zeros_like(Tw)

            for k, (ix, iy) in enumerate(path):
                p_i = cfg_inlet.p_ref - k * gp.dp_ref_seg
                h_i = h[k]

                rho_i = PropsSI("D", "P", p_i, "H", h_i, fluid)
                try:
                    drho_dh = PropsSI("d(Dmass)/d(Hmass)|P",
                                      "P", p_i, "H", h_i, fluid)
                except ValueError:
                    drho_dh = 0.0

                Y_i = h_i * drho_dh + rho_i

                # upwind enthalpy
                if k == 0:
                    h_up = cfg_inlet.h_ref
                else:
                    h_up = h[k - 1]

                T_ref_K = PropsSI("T", "P", p_i, "H", h_i, fluid)

                Q_ref_i = gp.h_int * gp.A_inner * (Tw[k] - T_ref_K)  # W
                qdot_i = Q_ref_i / (gp.A_flow * gp.dx)  # W/m³

                dhdt[k] = (
                                  -m_dot_ref / (gp.A_flow * gp.dx) * (h_i - h_up)
                                  + rho_i * qdot_i
                          ) / Y_i

                Q_f_i = Q_seg_list[ix][iy]  # W
                dTwdt[k] = (Q_f_i - Q_ref_i) / (gp.rho_wall * gp.c_wall * gp.V_wall)

            return np.concatenate([dhdt, dTwdt])

        # ---------------- integrate one outer step ----------------
        t0 = t_outer
        t1 = t_outer + dt_outer

        t0_ref = perf_counter()
        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0,
            method="BDF",
            rtol=1e-6,
            atol=1e-8,
            max_step=dt_inner,  # internal time step control
            # no t_eval → solver chooses its own step times, we just read them
        )
        t1_ref = perf_counter()

        if not sol.success:
            raise RuntimeError(f"Refrigerant ODE solver failed: {sol.message}")

        # number of inner solver steps (minus initial point)
        n_inner = len(sol.t) - 1

        # final state
        y_end = sol.y[:, -1]
        h_end = y_end[:n_seg]
        Tw_end = y_end[n_seg:]

        # some characteristic values for the print:
        #   outlet refrigerant temperature (take first segment in path)
        h_out0 = float(h_end[0])
        p_out0 = cfg_inlet.p_ref  # minus 0*dp_seg
        T_out0_K = PropsSI("T", "P", p_out0, "H", h_out0, fluid)
        T_out0_C = T_out0_K - 273.15

        mean_Tw_C = np.mean(Tw_end) - 273.15

        # print line similar to your Finn & Tube output
        ref_time = t1_ref - t0_ref
        print("Refrigerant Domain Inner Steps: " + str(n_inner) +
              " \t T_out_ref: " + f"{T_out0_C:.3e}" +
              " \t mean T_tube: " + f"{mean_Tw_C:.3e}" +
              " \t cycle time: " + f"{ref_time:.3f} s")

        # ---------------- write final state back into cfg_grid ----------------
        for k, (ix, iy) in enumerate(path):
            cfg = cfg_grid[ix][iy]

            h_out = float(h_end[k])
            p_out = cfg_inlet.p_ref - k * gp.dp_ref_seg

            T_out_K = PropsSI("T", "P", p_out, "H", h_out, fluid)
            T_out_C = T_out_K - 273.15

            rho_out = PropsSI("D", "P", p_out, "H", h_out, fluid)
            V_dot_out = m_dot_ref / rho_out

            try:
                x_out = PropsSI("Q", "P", p_out, "H", h_out, fluid)
            except ValueError:
                x_out = float("nan")

            cfg.h_ref = h_out
            cfg.p_ref = p_out
            cfg.T_ref = T_out_C
            cfg.V_dot_ref = V_dot_out
            cfg.x_ref = x_out
            cfg.T_tube = float(Tw_end[k] - 273.15)