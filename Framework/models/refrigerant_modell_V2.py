from time import perf_counter
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp, BDF
from scipy.interpolate import interp1d
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI
from Framework.models.derivatives_of_rho import drho_dP_dH
from CoolProp.CoolProp import AbstractState, PSmass_INPUTS, HmassP_INPUTS


@dataclass
class RefGeomParams:
    """Geometric parameters for the refrigerant-side segments."""
    A_flow: float
    dx: float
    A_inner: float
    V_wall: float
    rho_wall: float
    c_wall: float
    dp_ref_seg: float = 0.0

    def __post_init__(self):
        if self.A_flow <= 0:
            raise ValueError(f"A_flow must be > 0, got {self.A_flow}")
        if self.dx <= 0:
            raise ValueError(f"dx must be > 0, got {self.dx}")
        if self.A_inner <= 0:
            raise ValueError(f"A_inner must be > 0, got {self.A_inner}")
        if self.V_wall <= 0:
            raise ValueError(f"V_wall must be > 0, got {self.V_wall}")
        if self.rho_wall <= 0:
            raise ValueError(f"rho_wall must be > 0, got {self.rho_wall}")
        if self.c_wall <= 0:
            raise ValueError(f"c_wall must be > 0, got {self.c_wall}")


class Refrigerant:
    """Refrigerant-side model including wall and condenser coupling."""
    def __init__(self,geometry, geom_params: RefGeomParams, cfg, HP, path_variant="row_serpentine"):
        self.geometry = geometry
        self.geom = geom_params
        self.fluid = cfg.ref_str
        self.AS = CP.AbstractState("HEOS", self.fluid)  # or "BICUBIC&HEOS", "TTSE&HEOS", etc.
        self.connection_path = geometry.build_connection_path(variant=path_variant)
        self._path_ix = np.array([ix for ix, iy in self.connection_path], dtype=int)
        self._path_iy = np.array([iy for ix, iy in self.connection_path], dtype=int)
        self.HP = HP
        self._bdf = None
        self._bdf_dim = None
        self._rhs_ptr = None
        self._rhs_wrapper = lambda t, y: self._rhs_ptr(t, y)
        self._init_rho_tables()

        # --- Valve PI controller state ---
        self.valve_pos = None  # last valve position [%]
        self._valve_I = 0.0  # integrator state [K*s]
        self._valve_last_t = None  # last time [s]
        self._valve_SH_filt = None  # optional filtered SH [K]

    def _init_rho_tables(self):
        out = drho_dP_dH(
            self,
            fluid=self.fluid,
            P_min=1e4, P_max=30e5, dP=1e4,
            H_min=2e5, H_max=1e6, dH=1e3,
            scheme="central",
            save_path=None,
            load_path=None,
        )
        self._rho_interp = out["rho_interp"]
        self._drho_dP_interp = out["drho_dP_interp"]
        self._drho_dH_interp = out["drho_dH_interp"]
        self._T_interp = out.get("T_interp", None)
        self._T_sat_interp = None

        # Clipping bounds (so the interpolator does not return NaN)
        self._P_min = float(out["P_vec"][0])
        self._P_max = float(out["P_vec"][-1])
        self._H_min = float(out["H_vec"][0])
        self._H_max = float(out["H_vec"][-1])

        self._P_vec = np.asarray(out["P_vec"], dtype=float)
        if self._P_vec.size:
            T_sat_vec = PropsSI("T", "P", self._P_vec, "Q", 1.0, self.fluid).astype(float)
            self._T_sat_interp = interp1d(
                self._P_vec,
                T_sat_vec,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

    def T_from_PH(self, p_i, h_i):
        p = float(np.clip(p_i, self._P_min, self._P_max))
        h = float(np.clip(h_i, self._H_min, self._H_max))
        return float(self._T_interp([[p, h]])[0])

    def rho_and_derivs(self, p_i, h_i):
        p = float(np.clip(p_i, self._P_min, self._P_max))
        h = float(np.clip(h_i, self._H_min, self._H_max))
        pt = np.array([[p, h]], dtype=float)

        rho  = float(self._rho_interp(pt)[0])
        drhodp = float(self._drho_dP_interp(pt)[0])
        drhodh = float(self._drho_dH_interp(pt)[0])

        return rho, drhodp, drhodh

    def rho_and_derivs_vec(self, P, h_vec):
        P = float(np.clip(P, self._P_min, self._P_max))
        h = np.asarray(h_vec, dtype=float)
        h = np.clip(h, self._H_min, self._H_max)

        pts = np.column_stack([np.full_like(h, P, dtype=float), h])  # shape (N,2)

        rho = self._rho_interp(pts).astype(float)
        rhoP = self._drho_dP_interp(pts).astype(float)
        rhoh = self._drho_dH_interp(pts).astype(float)

        return rho, rhoP, rhoh

    def T_from_PH_vec(self, P, h_vec):
        P = float(np.clip(P, self._P_min, self._P_max))
        h = np.asarray(h_vec, dtype=float)
        h = np.clip(h, self._H_min, self._H_max)

        if self._T_interp is None:
            # PropsSI kann Arrays; P muss gleich lang sein
            P_arr = np.full_like(h, P, dtype=float)
            return PropsSI("T", "P", P_arr, "H", h, self.fluid).astype(float)

        pts = np.column_stack([np.full_like(h, P, dtype=float), h])
        T = self._T_interp(pts).astype(float)

        return T

    # ------------------------------------------------------------------
    # Local correlation for h_int (wall ↔ refrigerant heat transfer)
    # ------------------------------------------------------------------
    def h_int_corr(self):
        return 5000.0

    def h_int_corr_cond(self):
        return 5000.0

    def h_int_corr_water(self):
        return 8000.0

    def valve_controller(self, t, P_suction=None, h_suction=None):
        """
        Simple PI controller for superheat SH at evaporator outlet.

        Input:
            t          : time [s]
            P_suction  : suction pressure at evap outlet [Pa]
            h_suction  : suction enthalpy  at evap outlet [J/kg]

        Output:
            valve position [%] in [0..100]
        """

        # -----------------------------
        # Enable condition
        # -----------------------------
        HP = self.HP


        # -----------------------------
        # controller settings
        # -----------------------------
        SH_set = 8.0  # [K] target superheat
        Kp = 0.1  # [%/K]
        Ki = 0.005  # [%/(K*s)]
        u_min = 0.0  # [%]
        u_max = 100.0  # [%]
        u0 = 20.0  # [%] bias / initial opening
        du_max_per_s = 1.0 # [%/s]
        T_sample = 0.2 # [s] Sample time for controller

        if not HP.use_controller:
            self.valve_pos = u0
            return u0

        if t <= 60:
            self.valve_pos = u0
            return float(self.valve_pos)

        # -----------------------------
        # Init state on first call
        # -----------------------------
        if self.valve_pos is None:
            self.valve_pos = float(np.clip(u0, u_min, u_max))
        if self._valve_last_t is None:
            self._valve_last_t = float(t)
            self._valve_I = 0.0
            return self.valve_pos

        # time step (avoid issues if solver calls multiple times at same t)
        t_now = float(t)
        dt = t_now - float(self._valve_last_t)
        if dt < T_sample:
            return float(self.valve_pos)

        # if measurement not available -> hold last valve position
        if P_suction is None or h_suction is None:
            self._valve_last_t = t_now
            return float(self.valve_pos)

        P_suction = float(P_suction)
        h_suction = float(h_suction)

        # -----------------------------
        # Compute SH = T_suction - T_sat(P_suction)
        # -----------------------------
        fluid = self.fluid

        # T_suction [K]
        T_suction = float(self.T_from_PH_vec(P_suction, np.array([h_suction], dtype=float))[0])

        # T_sat [K]
        T_sat = float(self._T_sat_interp(P_suction))

        SH = T_suction - T_sat  # [K] can be negative

        dt_eff = dt
        du_max = du_max_per_s * dt

        tau = 2.0  # [s] Filter Time constant
        alpha = dt_eff / (tau + dt_eff)

        if getattr(self, "_SH_filt", None) is None:
            self._SH_filt = SH

        self._SH_filt = (1 - alpha) * self._SH_filt + alpha * SH
        SH_used = self._SH_filt
        #print(f"SH_used: {SH_used}")

        # -----------------------------
        # PI control law with simple anti-windup
        # -----------------------------
        e = SH_used - SH_set  # positive => SH too high => open valve more
        #print(f"Error in controller: {e}")
        #if abs(e) < 0.1:  # [K] Deadband
        #    e = 0.0

        I_old = float(self._valve_I)

        I_new = I_old + e * dt

        u_unclamped = u0 + Kp * e + Ki * I_new
        u = float(np.clip(u_unclamped, u_min, u_max))

        # Anti-windup: if saturated and error pushes further into saturation -> freeze integrator
        if (u_unclamped > u_max and e > 0.0) or (u_unclamped < u_min and e < 0.0):
            I_new = I_old
            u_unclamped = u0 + Kp * e + Ki * I_new
            u = float(np.clip(u_unclamped, u_min, u_max))

        u = float(np.clip(u, self.valve_pos - du_max, self.valve_pos + du_max))

        limited_by_rate = abs(u - u_unclamped) > 1e-9

        if limited_by_rate:
            I_new = I_old
            u_unclamped = u0 + Kp * e + Ki * I_new
            u = float(np.clip(u_unclamped, u_min, u_max))
            u = float(np.clip(u, self.valve_pos - du_max, self.valve_pos + du_max))

        # store state
        self._valve_I = float(I_new)
        self.valve_pos = float(u)
        self._valve_last_t = t_now

        return float(self.valve_pos)

    def valve_model(self, pi, hi, po, VPos):
        dp = pi-po
        if dp <= 0.0:
            return 0.0, hi

        Kv = 0.25

        try:
            #rho = PropsSI("D", "P", pi, "H", hi, self.fluid)
            rho, _, _ = self.rho_and_derivs_vec(pi, hi)
        except Exception:
            return 0.0, hi

        if not np.isfinite(rho) or rho <= 0.0:
            return 0.0, hi

        U = VPos / 100
        m = Kv * U * np.sqrt(rho * (pi - po) * 1e-2) / 3600
        ho = hi

        return m, ho

    def compressor_model(self, pi, hi, po, RPM):
        if RPM <= 1e-6:
            return 0.0, hi
        TTSE = AbstractState("TTSE&HEOS", self.fluid)
        n = 4  # number of cylinders
        bore = 0.06  # bore [m]
        stroke = 0.042  # stroke [m]
        Vd = n * bore ** 2 * np.pi / 4 * stroke  # displacement volume

        a = np.array([-5.31166292e-02, 1.21402922e-03, 8.81226071e-05, 1.03163725e+00])
        b = np.array([9.38116126e-03, -1.52858792e-03, -4.08026601e-03, 6.31332600e-04, 6.77625196e-01])

        eta_v = a[0] * (po / pi) + a[1] * (po / pi) ** 2 + a[2] * (RPM / 60) + a[3]
        eta_is = b[0] * (po / pi) + b[1] * (po / pi) ** 2 + b[2] * (RPM / 60) + b[3] * (po / pi) * (RPM / 60) + b[4]

        eta_v = float(np.clip(eta_v, 0.05, 1.20))
        eta_is = float(np.clip(eta_is, 0.05, 0.90))

        TTSE.update(HmassP_INPUTS, hi, pi)
        s = TTSE.smass()
        # s = PropsSI("S", "P", pi, "H", hi, self.fluid)
        TTSE.update(PSmass_INPUTS, po, s)
        h_is = TTSE.hmass()
        # h_is = PropsSI('H', 'P', po, 'S', s, self.fluid)
        #h_is = h_is.reshape(po.shape)
        # rho = PropsSI("D", "P", pi, "H", hi, self.fluid)
        rho, _, _ = self.rho_and_derivs_vec(pi, hi)

        ho = hi + (h_is - hi) / eta_is
        m = RPM / 60 * Vd * eta_v * rho

        self.HP.W_comp = m * (ho-hi)
        return m, ho

    def update_all_segments(
            self,
            cfg_inlet,
            cfg_grid,
            st_grid,
            geom,
            Q_seg_list,
            time: float,
            dt: float,
    ):

        fluid = cfg_inlet.ref_str
        gp = self.geom
        HP = self.HP
        path = self.connection_path
        N = len(path)
        N_condenser = HP.N_cond

        def solve_dp_dh_condenser(P, h, T, T_in_water, h_in, m_in, m_out, m_water, Q_into_ref, Q_wall_water, V, M_water,
                                  M_wall, rho, rhoP, rhoh):

             a = V * rhoP
             b = V * rhoh
             c = V*(h*rhoP - 1.0)
             d = V*(h*rhoh + rho)

             A_mat = np.zeros((3 * N_condenser + 1, 3 * N_condenser + 1))
             b_vec = np.zeros((3 * N_condenser + 1,))

             # --- [1] Total Mass and Energy balances ---
             idx_v = np.arange(0, N_condenser + 1)
             # Mass
             A_mat[0, idx_v[0]] = np.sum(a)
             A_mat[0, idx_v[1:]] = b
             b_vec[0] = m_in - m_out
             # Energy
             A_mat[1, idx_v[0]] = np.sum(c)
             A_mat[1, idx_v[1:]] = d
             b_vec[1] = m_in * h_in - m_out * h[-1] + np.sum(Q_into_ref)

             # --- [2] Energy balances ---
             A_mat[2, 0] = c[0] - a[0] * h[0]
             A_mat[2, 1] = d[0] - b[0] * h[0]
             b_vec[2] = m_in * (h_in - h[0]) + Q_into_ref[0]
             row_e = np.arange(3, N_condenser + 1)
             A_mat[row_e, 0] = c[1:-1] - a[1:-1] * h[1:-1] - (h[1:-1] - h[:-2]) * np.cumsum(a[:N_condenser - 2])
             A_mat[row_e, np.arange(2,N_condenser)] = d[1:-1] - b[1:-1] * h[1:-1]
             b_vec[row_e] = m_in * (h[:-2] - h[1:-1]) + Q_into_ref[1:-1]
             delta_h = h[1:-1] - h[:-2]
             b_broadcast = b[np.newaxis, :N_condenser - 1]
             delta_h_broadcast = delta_h[:, np.newaxis]
             lower_tri = np.tril(np.ones((N_condenser - 2,N_condenser - 1)))
             fill_vals = delta_h_broadcast * b_broadcast * lower_tri
             A_mat[row_e[:, None], np.arange(1,N_condenser)] += fill_vals

             # --- [3] Wall Energy balances ---
             row_w = np.arange(N_condenser + 1, 2 * N_condenser + 1)
             col_w = np.arange(N_condenser + 1, 2 * N_condenser + 1)
             A_mat[row_w, col_w] = M_wall * HP.c_plate
             b_vec[row_w] = (-Q_into_ref) - Q_wall_water

             # --- [4] Secondary Fluid Energy balances ---
             row_s = np.arange(2 * N_condenser + 1, 3 * N_condenser + 1)
             col_s = np.arange(2 * N_condenser + 1, 3 * N_condenser + 1)
             A_mat[row_s, col_s] = M_water * HP.c_water
             #b_vec[row_s[:-1]] = m_water * HP.c_water * (T[1:] - T[:-1]) + Q_wall_water[:-1]
             #b_vec[row_s[-1]] = m_water * HP.c_water * (T_in_water - T[-1]) + Q_wall_water[-1]
             T_prev = np.empty_like(T)
             T_prev[0] = T_in_water
             T_prev[1:] = T[:-1]
             b_vec[row_s] = m_water * HP.c_water * (T_prev - T) + Q_wall_water

             x = np.linalg.solve(A_mat, b_vec)

             dPdt = float(x[0])
             dhdt = x[1:1 + N_condenser].copy()
             dT_walldt = x[N_condenser+1:2*N_condenser+1].copy()
             dT_waterdt = x[2*N_condenser+1:3*N_condenser+1].copy()

             return dPdt, dhdt, dT_walldt, dT_waterdt

        def solve_dp_dh_evaporator(P, h, h_in, m_in, m_out, Q_into_ref, V, rho, rhoP, rhoh):
            """
            """

            a = V * rhoP
            b = V * rhoh
            c = V*(h*rhoP - 1.0)
            d = V*(h*rhoh + rho)

            A_mat = np.zeros((N + 1, N + 1))
            b_vec = np.zeros((N + 1,))

            # --- [1] Total Mass and Energy balances ---
            idx_v = np.arange(0, N + 1)
            # Mass
            A_mat[0, idx_v[0]] = np.sum(a)
            A_mat[0, idx_v[1:]] = b
            b_vec[0] = m_in - m_out
            # Energy
            A_mat[1, idx_v[0]] = np.sum(c)
            A_mat[1, idx_v[1:]] = d
            b_vec[1] = m_in * h_in - m_out * h[-1] + np.sum(Q_into_ref)

            # --- [2] Energy balances ---
            A_mat[2, 0] = c[0] - a[0] * h[0]
            A_mat[2, 1] = d[0] - b[0] * h[0]
            b_vec[2] = m_in * (h_in - h[0]) + Q_into_ref[0]
            row_e = np.arange(3, N + 1)
            A_mat[row_e, 0] = c[1:-1] - a[1:-1] * h[1:-1] - (h[1:-1] - h[:-2]) * np.cumsum(a[:N - 2])
            A_mat[row_e, np.arange(2, N)] = d[1:-1] - b[1:-1] * h[1:-1]
            b_vec[row_e] = m_in * (h[:-2] - h[1:-1]) + Q_into_ref[1:-1]
            delta_h = h[1:-1] - h[:-2]
            b_broadcast = b[np.newaxis, :N - 1]
            delta_h_broadcast = delta_h[:, np.newaxis]
            lower_tri = np.tril(np.ones((N - 2, N - 1)))
            fill_vals = delta_h_broadcast * b_broadcast * lower_tri
            A_mat[row_e[:, None], np.arange(1, N)] += fill_vals

            x = np.linalg.solve(A_mat, b_vec)

            dPdt = float(x[0])
            dhdt = x[1:1 + N].copy()

            return dPdt, dhdt

        # ----------------------------------------------------------
        # Initial state
        # ----------------------------------------------------------
        h0 = np.zeros(N, dtype=float)
        Tw0 = np.zeros(N, dtype=float)

        h0_cond = np.zeros(N_condenser, dtype=float)
        T0_wall = np.zeros(N_condenser, dtype=float)
        T0_water = np.zeros(N_condenser, dtype=float)

        # Pressure is global: pick inlet if available, else first segment
        (ix0, iy0) = path[0]
        cfg0 = cfg_grid[ix0][iy0]
        P0 = float(cfg0.p_ref)
        P0_cond = float(HP.p_ref_cond)

        for k, (ix, iy) in enumerate(path):
            cfg = cfg_grid[ix][iy]
            h0[k] = float(cfg.h_ref)
            T_tube_C = float(cfg.T_tube)
            Tw0[k] = float(T_tube_C + 273.15)

        h0_cond = HP.h_ref_cond
        T0_wall = HP.T_wall + 273.15
        T0_water = HP.T_water + 273.15

        # State vector: [P, h_0..h_{N-1}, Tw_0..Tw_{N-1}]
        y0 = np.concatenate([np.array([P0]), h0, Tw0, np.array([P0_cond]), h0_cond, T0_wall, T0_water])

        # ----------------------------------------------------------
        # Vectorising
        # ----------------------------------------------------------

        # Evaporator constants:
        ixp = self._path_ix
        iyp = self._path_iy

        Qseg = np.asarray(Q_seg_list, dtype=float)

        V = np.full(N, gp.A_flow * gp.dx, dtype=float)
        den_wall = gp.rho_wall * gp.c_wall * gp.V_wall
        h_int_evap = float(self.h_int_corr())

        # Condenser constants:
        V_plate = HP.n_plates * HP.t_plate * HP.length_cond * HP.height_cond
        M_plate = V_plate * HP.rho_plate
        V_ref = HP.A_flow_cond * HP.length_cond
        V_cond = np.full(N_condenser, V_ref/N_condenser, dtype=float)
        M_water = np.full(N_condenser, HP.rho_water * V_ref/N_condenser, dtype=float)
        M_wall = np.full(N_condenser, M_plate/N_condenser, dtype=float)
        A_plate_seg = HP.A_plate / N_condenser
        h_int_cond = float(self.h_int_corr_cond())
        h_int_water = float(self.h_int_corr_water())

        # ----------------------------------------------------------
        # RHS
        # ----------------------------------------------------------
        def rhs(t, y):
            P = float(y[0])
            h = y[1:1 + N]
            Tw = y[1 + N:1 + 2 * N]
            P_cond = y[1 + 2*N]
            h_cond = y[2 + 2*N:2+2*N + N_condenser]
            T_wall = y[2+2*N + N_condenser:2+2*N + 2*N_condenser]
            T_water = y[2+2*N + 2*N_condenser:2+2*N + 3*N_condenser]

            m_comp, h_out_comp = self.compressor_model(P, h[-1], P_cond, HP.RPM(t))
            m_valve, h_out_valve = self.valve_model(P_cond, h_cond[-1], P, self.valve_controller(t, P_suction=P, h_suction=h[-1]))

            # --- Evaporator: vectorised ---
            rho, rhoP, rhoh = self.rho_and_derivs_vec(P, h)
            T_ref_K = self.T_from_PH_vec(P, h)

            Q_into_ref = h_int_evap * gp.A_inner * (Tw - T_ref_K)  # shape (N,)
            Q_f = Qseg[ixp, iyp]  # shape (N,)
            dTwdt = (Q_f - Q_into_ref) / den_wall  # shape (N,)

            # Solve coupled refrigerant linear system
            dPdt, dhdt = solve_dp_dh_evaporator(
                P=P,
                h=h,
                h_in=h_out_valve,
                m_in=m_valve/self.geometry.stacks,
                m_out=m_comp/self.geometry.stacks,
                Q_into_ref=Q_into_ref,
                V=V,
                rho=rho,
                rhoP=rhoP,
                rhoh=rhoh,
            )

            # --- Condenser: vectorised ---
            rho_cond, rhoP_cond, rhoh_cond = self.rho_and_derivs_vec(P_cond, h_cond)
            T_ref_K_cond = self.T_from_PH_vec(P_cond, h_cond)

            Q_into_ref_cond = h_int_cond * A_plate_seg * (T_wall - T_ref_K_cond)
            Q_wall_water = h_int_water * A_plate_seg * (T_wall - T_water)

            dPdt_cond, dhdt_cond, dTdt_wall_cond, dTdt_water_cond = solve_dp_dh_condenser(
                P=P_cond,
                h=h_cond,
                T=T_water,
                T_in_water=HP.T_in_water+273.15,
                h_in=h_out_comp,
                m_in=m_comp,
                m_out=m_valve,
                m_water=HP.m_water,
                Q_into_ref=Q_into_ref_cond,
                Q_wall_water=Q_wall_water,
                V=V_cond,
                M_water=M_water,
                M_wall=M_wall,
                rho=rho_cond,
                rhoP=rhoP_cond,
                rhoh=rhoh_cond
            )

            # Pack derivative vector
            return np.concatenate([np.array([dPdt]), dhdt, dTwdt, np.array([dPdt_cond]), dhdt_cond, dTdt_wall_cond, dTdt_water_cond])

        # ----------------------------------------------------------
        # Integrate
        # ----------------------------------------------------------
        t0 = float(time)
        t1 = float(time + dt)

        # 1) Plug in the current RHS (boundary conditions / Q_seg_list for this macro step)
        self._rhs_ptr = rhs

        # 2) Initialize solver once (or hard restart if dimension/time mismatch)
        need_restart = (
                self._bdf is None
                or self._bdf_dim != y0.size
                or abs(float(self._bdf.t) - t0) > 1e-12
        )

        # call the controller
        P_suction_0 = float(y0[0])
        h_suction_0 = float(y0[1 + N - 1])

        #VPos_hold = float(self.valve_controller(t0, P_suction=P_suction_0, h_suction=h_suction_0))
        #self.valve_pos = VPos_hold

        # Optional: restart if cfg_grid was changed externally and no longer matches solver.y
        if (not need_restart) and (np.max(np.abs(self._bdf.y - y0)) > 1e-6):
            need_restart = True
            print("\033[93mRestarting the solver...\033[0m")

        if need_restart:
            self._bdf = BDF(
                fun=self._rhs_wrapper,
                t0=t0,
                y0=y0,
                t_bound=np.inf,  # we control stopping via max_step + while-loop
                rtol=1e-6,
                atol=1e-8,
                max_step=np.inf
            )
            self._bdf_dim = y0.size

        # 3) Integrate forward to t1 without stepping past it
        cycl_t0_ref = perf_counter()
        inner_steps = 0

        # Small tolerance against float rounding
        while self._bdf.status == "running" and self._bdf.t < t1 - 1e-15:
            dt_rem = t1 - float(self._bdf.t)
            # Prevent overshoot beyond the end of the macro step
            self._bdf.max_step = dt_rem
            msg = self._bdf.step()
            inner_steps += 1
            if self._bdf.status == "failed":
                raise RuntimeError(f"Refrigerant ODE solver failed: {msg}")

        # End state (current state of the persistent solver)
        y_end = self._bdf.y.copy()
        cycl_t1_ref = perf_counter()

        # if not sol.success:
        #     raise RuntimeError(f"Refrigerant ODE solver failed: {sol.message}")

        # End state
        P_end = float(y_end[0])
        h_end = y_end[1:1 + N]
        Tw_end = y_end[1 + N:1 + 2 * N]
        P_cond_end = y_end[1 + 2 * N]
        h_cond_end = y_end[2 + 2 * N:2 + 2 * N + N_condenser]
        T_wall_end = y_end[2 + 2 * N + N_condenser:2 + 2 * N + 2 * N_condenser]
        T_water_end = y_end[2 + 2 * N + 2 * N_condenser:2 + 2 * N + 3 * N_condenser]

        # Heat flows
        T_ref_evap_K = np.array([PropsSI("T", "P", P_end, "H", float(hk), fluid) for hk in h_end], dtype=float)

        Q_evap = self.geometry.stacks * float(np.sum(self.h_int_corr() * gp.A_inner * (Tw_end - T_ref_evap_K)))  # [W]
        Q_cond = float(np.sum(self.h_int_corr_water() * (HP.A_plate / N_condenser) * (T_wall_end - T_water_end)))  # [W]

        # Console output
        n_inner = inner_steps
        ref_time = cycl_t1_ref - cycl_t0_ref

        def fmt_sh_sc(P, h, fluid, kind: str) -> str:
            """
            kind: "evap" -> report superheat (SH)
                  "cond" -> report subcooling (SC)
            """
            P = float(P)
            h = float(h)

            # actual temperature from (P,h)
            T = float(PropsSI("T", "P", P, "H", h, fluid))
            # saturation temperature at this pressure
            T_sat = float(PropsSI("T", "P", P, "Q", 0, fluid))  # same as Q=1

            # Try quality -> only defined in 2-phase region
            x = None
            try:
                x_try = float(PropsSI("Q", "P", P, "H", h, fluid))
                if np.isfinite(x_try) and 0.0 <= x_try <= 1.0:
                    x = x_try
            except Exception:
                pass

            if x is not None:
                # two-phase: no meaningful SH/SC
                return f"TP(x={x:.3f}, Tsat={T_sat - 273.15:.2f}°C, T={T - 273.15:.2f}°C)"
            else:
                # single-phase: use temperature distance to saturation
                if kind == "evap":
                    SH = T - T_sat
                    if SH >= 0:
                        return f"SH={SH:.2f}K (Tsat={T_sat - 273.15:.2f}°C, T={T - 273.15:.2f}°C)"
                    else:
                        return f"NO-SH(subcooled?)={(-SH):.2f}K (Tsat={T_sat - 273.15:.2f}°C, T={T - 273.15:.2f}°C)"
                else:  # "cond"
                    SC = T_sat - T
                    if SC >= 0:
                        return f"SC={SC:.2f}K (Tsat={T_sat - 273.15:.2f}°C, T={T - 273.15:.2f}°C)"
                    else:
                        return f"NO-SC(superheated?)={(-SC):.2f}K (Tsat={T_sat - 273.15:.2f}°C, T={T - 273.15:.2f}°C)"

        # HX exits: evap outlet is last segment (suction to compressor), cond outlet is last segment (to valve)
        evap_exit = fmt_sh_sc(P_end, h_end[-1], fluid, kind="evap")
        cond_exit = fmt_sh_sc(P_cond_end, h_cond_end[-1], fluid, kind="cond")

        valve_pos_current = self.valve_pos

        print(
            f"HP(it={n_inner}, evap_out={evap_exit}, cond_out={cond_exit}, "
            f"Q_evap={Q_evap:.1f} W, Q_cond={Q_cond:.1f} W, Valve={valve_pos_current:.1f} %, ct={ref_time:.3f} s)"
        )

        # ----------------------------------------------------------
        # Write back to cfg_grid
        # ----------------------------------------------------------
        for k, (ix, iy) in enumerate(path):
            cfg = cfg_grid[ix][iy]
            h_k = float(h_end[k])
            Tw_C = float(Tw_end[k] - 273.15)

            # Derived outputs
            T_ref_K = float(PropsSI("T", "P", P_end, "H", h_k, fluid))
            T_ref_C = T_ref_K - 273.15

            try:
                x_out = float(PropsSI("Q", "P", P_end, "H", h_k, fluid))
            except ValueError:
                x_out = float("nan")

            cfg.p_ref = P_end
            cfg.h_ref = h_k
            cfg.T_ref = T_ref_C
            cfg.x_ref = x_out
            cfg.T_tube = Tw_C

        HP.p_ref_cond = P_cond_end
        HP.h_ref_cond = h_cond_end
        HP.T_wall = T_wall_end - 273.15
        HP.T_water = T_water_end - 273.15
        HP.Q_evap = Q_evap
        HP.Q_cond = Q_cond

        return n_inner


    def reset_integrator(self):
        self._bdf = None

        self.valve_pos = None
        self._valve_I = 0.0
        self._valve_last_t = None
        self._valve_SH_filt = None
