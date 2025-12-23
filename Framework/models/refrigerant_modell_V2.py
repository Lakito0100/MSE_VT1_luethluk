from time import perf_counter
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp, BDF
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI
from Framework.models.derivatives_of_rho import drho_dP_dH


@dataclass
class RefGeomParams:
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

    def _init_rho_tables(self):
        out = drho_dP_dH(
            self,
            fluid=self.fluid,
            P_min=1e3, P_max=20e5, dP=1e4,
            H_min=2e5, H_max=1e6, dH=1e3,
            scheme="central",
            save_path=None,
            load_path=None,
        )
        self._rho_interp = out["rho_interp"]
        self._drho_dP_interp = out["drho_dP_interp"]
        self._drho_dH_interp = out["drho_dH_interp"]
        self._T_interp = out.get("T_interp", None)

        # Grenzen zum Clipping (damit Interpolator nicht NaN liefert)
        self._P_min = float(out["P_vec"][0]);
        self._P_max = float(out["P_vec"][-1])
        self._H_min = float(out["H_vec"][0]);
        self._H_max = float(out["H_vec"][-1])

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
    # Lokale Korrelation für h_int (Wärmeübergang Wand ↔ Kältemittel)
    # ------------------------------------------------------------------
    def h_int_corr(self):
        return 5000.0

    def h_int_corr_cond(self):
        return 1000.0

    def h_int_corr_water(self):
        return 1000.0

    def valve_controller(self):
        VPos = 40.0
        return VPos

    def valve_model(self, pi, hi, po, VPos):
        dp = pi-po
        if dp <= 0.0:
            return 0.0, hi

        Kv = 0.25

        try:
            rho = PropsSI("D", "P", pi, "H", hi, self.fluid)
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

        n = 4  # number of cylinders
        bore = 0.06  # bore [m]
        stroke = 0.042  # stroke [m]
        Vd = n * bore ** 2 * np.pi / 4 * stroke  # displacement volume

        a = np.array([-5.31166292e-02, 1.21402922e-03, 8.81226071e-05, 1.03163725e+00])
        b = np.array([9.38116126e-03, -1.52858792e-03, -4.08026601e-03, 6.31332600e-04, 6.77625196e-01])

        eta_v = a[0] * (po / pi) + a[1] * (po / pi) ** 2 + a[2] * (RPM / 60) + a[3]
        eta_is = b[0] * (po / pi) + b[1] * (po / pi) ** 2 + b[2] * (RPM / 60) + b[3] * (po / pi) * (RPM / 60) + b[4]

        s = PropsSI("S", "P", pi, "H", hi, self.fluid)
        h_is = PropsSI('H', 'P', po, 'S', s, self.fluid)
        #h_is = h_is.reshape(po.shape)
        rho = PropsSI("D", "P", pi, "H", hi, self.fluid)
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
             A_mat[row_w, col_w] = M_wall * self.geometry.c_solid
             b_vec[row_w] = (-Q_into_ref) - Q_wall_water

             # --- [4] Secondary Fluid Energy balances ---
             row_s = np.arange(2 * N_condenser + 1, 3 * N_condenser + 1)
             col_s = np.arange(2 * N_condenser + 1, 3 * N_condenser + 1)
             A_mat[row_s, col_s] = M_water * HP.c_water
             b_vec[row_s[:-1]] = m_water * HP.c_water * (T[1:] - T[:-1]) + Q_wall_water[:-1]
             b_vec[row_s[-1]] = m_water * HP.c_water * (T_in_water - T[-1]) + Q_wall_water[-1]

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
            T_tube_C = getattr(cfg, "T_tube", cfg.T_ref)
            Tw0[k] = float(T_tube_C + 273.15)

        for i in range(N_condenser):
            h0_cond[i] = HP.h_ref_cond[i]
            T0_wall[i] = HP.T_wall[i] + 273.15
            T0_water[i] = HP.T_water[i] + 273.15

        # State vector: [P, h_0..h_{N-1}, Tw_0..Tw_{N-1}]
        y0 = np.concatenate([np.array([P0]), h0, Tw0, np.array([P0_cond]), h0_cond, T0_wall, T0_water])

        # ----------------------------------------------------------
        # Vectorising
        # ----------------------------------------------------------

        # Verdampfer-Konstanten:
        ixp = self._path_ix
        iyp = self._path_iy

        Qseg = np.asarray(Q_seg_list, dtype=float)

        V = np.full(N, gp.A_flow * gp.dx, dtype=float)
        den_wall = gp.rho_wall * gp.c_wall * gp.V_wall
        h_int_evap = float(self.h_int_corr())

        # Kondensator-Konstanten:
        V_cond = np.full(N_condenser, HP.A_flow_cond * HP.dx_cond, dtype=float)
        M_water = np.full(N_condenser, HP.A_flow_cond * HP.dx_cond * HP.rho_water, dtype=float)
        M_wall = np.full(N_condenser, HP.t_plate * HP.dx_cond * HP.height_cond * HP.rho_plate, dtype=float)
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
            m_valve, h_out_valve = self.valve_model(P_cond, h_cond[-1], P, self.valve_controller())

            # Per-segment arrays needed for linear solve
            #V = np.full(N, gp.A_flow * gp.dx, dtype=float)
            #rho = np.zeros(N, dtype=float)
            #rhoP = np.zeros(N, dtype=float)
            #rhoh = np.zeros(N, dtype=float)
            #Q_into_ref = np.zeros(N, dtype=float)  # + into refrigerant [W]
            #dTwdt = np.zeros(N, dtype=float)
            #
            # Build property and heat-transfer terms
            #for k, (ix, iy) in enumerate(path):
            #    h_k = float(h[k])
            #
            #    # density and derivatives at (P, h_k)
            #    rho_k, drho_dP_k, drho_dh_k = self.rho_and_derivs(P, h_k) # Grid interpolator um ableitungen vektorisiert holen
            #    rho[k] = float(rho_k)
            #    rhoP[k] = float(drho_dP_k)
            #    rhoh[k] = float(drho_dh_k)
            #
            #    # Refrigerant temperature
            #    T_ref_K = self.T_from_PH(P, h_k)
            #
            #    # Internal HTC
            #    h_int_k = float(self.h_int_corr())
            #
            #    # Heat rate into refrigerant (your convention)
            #    Q_ref_k = h_int_k * gp.A_inner * (Tw[k] - T_ref_K)  # [W]
            #    Q_into_ref[k] = Q_ref_k
            #
            #    # External heat into wall (from air/frost model)
            #    Q_f_k = float(Q_seg_list[ix][iy])  # [W] into wall
            #
            #    # Wall ODE: (in - out) / (m*c)
            #    dTwdt[k] = (Q_f_k - Q_ref_k) / (gp.rho_wall * gp.c_wall * gp.V_wall)

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

            # Per-segment arrays needed for linear solve
            #V_cond = np.full(N_condenser, HP.A_flow_cond * HP.dx_cond, dtype=float)
            #rho_cond = np.zeros(N_condenser, dtype=float)
            #rhoP_cond = np.zeros(N_condenser, dtype=float)
            #rhoh_cond = np.zeros(N_condenser, dtype=float)
            #Q_into_ref_cond = np.zeros(N_condenser, dtype=float)  # + into refrigerant [W]
            #Q_wall_water = np.zeros(N_condenser, dtype=float)
            #M_water = np.full(N_condenser, HP.A_flow_cond * HP.dx_cond * HP.rho_water, dtype=float)
            #M_wall = np.full(N_condenser, HP.A_wall * HP.dx_cond * self.geometry.rho_solid, dtype=float)
            #
            ## Build property and heat-transfer terms
            #for k in range(N_condenser):
            #    h_k = float(h_cond[k])
            #
            #    # density and derivatives at (P, h_k)
            #    rho_k, drho_dP_k, drho_dh_k = self.rho_and_derivs(P_cond, h_k)  # Grid interpolator um ableitungen vektorisiert holen
            #    rho_cond[k] = float(rho_k)
            #    rhoP_cond[k] = float(drho_dP_k)
            #    rhoh_cond[k] = float(drho_dh_k)
            #
            #    # Refrigerant temperature
            #    T_ref_K = self.T_from_PH(P_cond,h_k)
            #
            #    # Internal HTC
            #    h_int_k = float(self.h_int_corr_cond())
            #    h_int_water = float(self.h_int_corr_water())
            #
            #    # Heat rate into refrigerant
            #    Q_ref_k = h_int_k * HP.A_plate/N_condenser * (T_wall[k] - T_ref_K)  # [W]
            #    Q_into_ref_cond[k] = Q_ref_k
            #
            #    # External heat into wall (from air/frost model)
            #    Q_wall_water_k = h_int_water * HP.A_plate/N_condenser * (T_wall[k] - T_water[k])  # [W] into wall
            #
            #    Q_wall_water[k] = Q_wall_water_k

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

        # 1) Aktuellen RHS "einstecken" (Randbedingungen/ Q_seg_list etc. dieser Makro-Stufe)
        self._rhs_ptr = rhs

        # 2) Solver einmalig initialisieren (oder hart neu starten, falls Dimension/Time nicht passt)
        need_restart = (
                self._bdf is None
                or self._bdf_dim != y0.size
                or abs(float(self._bdf.t) - t0) > 1e-12
        )

        # Optional: wenn cfg_grid extern geändert wurde und nicht zu solver.y passt -> restart
        if (not need_restart) and (np.max(np.abs(self._bdf.y - y0)) > 1e-6):
            need_restart = True
            print("\033[93mRestarting the solver...\033[0m")

        if need_restart:
            self._bdf = BDF(
                fun=self._rhs_wrapper,
                t0=t0,
                y0=y0,
                t_bound=np.inf,  # wir steuern das Stoppen selbst über max_step + while-loop
                rtol=1e-6,
                atol=1e-8,
                max_step=np.inf
            )
            self._bdf_dim = y0.size

        # 3) Bis t1 vorwärts integrieren, ohne über t1 hinaus zu gehen
        cycl_t0_ref = perf_counter()
        inner_steps = 0

        # kleine Toleranz gegen Float-Rundung
        while self._bdf.status == "running" and self._bdf.t < t1 - 1e-15:
            dt_rem = t1 - float(self._bdf.t)
            # verhindere Overshoot über das Ende des Makroschritts
            self._bdf.max_step = dt_rem
            msg = self._bdf.step()
            inner_steps += 1
            if self._bdf.status == "failed":
                raise RuntimeError(f"Refrigerant ODE solver failed: {msg}")

        # End state (genau der aktuelle Zustand des persistenten Solvers)
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
        #n_inner = len(sol.t) - 1
        n_inner = inner_steps
        mean_T_ref_evap_K = float(PropsSI("T", "P", P_end, "H", float(np.mean(h_end)), fluid))
        mean_T_ref_evap_C = mean_T_ref_evap_K - 273.15
        mean_Tw_C = float(np.mean(Tw_end) - 273.15)
        ref_time = cycl_t1_ref - cycl_t0_ref

        print(
            f"HP(it={n_inner}, mean T_evap={mean_T_ref_evap_C:.2f} °C, mean T_tube_evap={mean_Tw_C:.2f} °C, Q_evap={Q_evap:.2f} W, Q_cond={Q_cond:.2f} W, ct={ref_time:.3f} s)"
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
            rho_out = float(PropsSI("D", "P", P_end, "H", h_k, fluid))

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


    def reset_integrator(self):
        self._bdf = None