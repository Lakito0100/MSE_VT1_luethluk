from time import perf_counter
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp, BDF
from scipy.interpolate import interp1d
try:  # compiled helpers of RegularGridInterpolator (fallback: the interpolator itself)
    from scipy.interpolate._rgi_cython import find_indices as _find_indices, evaluate_linear_2d as _evaluate_linear_2d
except ImportError:
    _find_indices = _evaluate_linear_2d = None
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI
from Framework.models.derivatives_of_rho import drho_dP_dH, DEFAULT_TABLE_CACHE_DIR
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
    def __init__(self,geometry, geom_params: RefGeomParams, cfg, HP, path_variant="serpentine"):
        self.geometry = geometry
        self.geom = geom_params
        self.fluid = cfg.ref_str
        self.AS = CP.AbstractState("HEOS", self.fluid)  # created once, reused by compressor_model
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
            H_min=1.5e5, H_max=1e6, dH=1e3,  # 150 kJ/kg = sat. liquid at about -38.5 °C (0.55 bar)
            scheme="central",
            save_path=None,
            load_path=None,
            cache_dir=DEFAULT_TABLE_CACHE_DIR,  # reuse the CoolProp grids between runs
        )
        # Bilinear tables on the (P, h) grid: density, its derivatives and temperature
        self._interp = {"rho": out["rho_interp"], "drho_dP": out["drho_dP_interp"],
                        "drho_dH": out["drho_dH_interp"], "T": out["T_interp"]}
        self._grid = out["rho_interp"].grid
        # RegularGridInterpolator uses its compiled 2-D path only for these tables
        self._fast = {k: _find_indices is not None and v.values.ndim == 2 and v.values.dtype == np.float64
                      and v.values.flags.writeable and v.values.dtype.byteorder == "="
                      for k, v in self._interp.items()}

        # Clipping bounds (so the interpolator does not return NaN)
        self._P_min, self._P_max = float(out["P_vec"][0]), float(out["P_vec"][-1])
        self._H_min, self._H_max = float(out["H_vec"][0]), float(out["H_vec"][-1])

        self._P_vec = np.asarray(out["P_vec"], dtype=float)
        T_sat_vec = PropsSI("T", "P", self._P_vec, "Q", 1.0, self.fluid).astype(float)
        self._T_sat_interp = interp1d(self._P_vec, T_sat_vec, kind="linear",
                                      bounds_error=False, fill_value="extrapolate")

    def _lookup(self, P, h, names=("rho", "drho_dP", "drho_dH", "T")):
        """
        Table values at (P, h), clipped to the table range; P and h broadcast.
        Returns one array per name. The grid search is done once for all tables, with
        the same compiled routines RegularGridInterpolator uses (identical results).
        """
        P, h = np.broadcast_arrays(np.clip(P, self._P_min, self._P_max),
                                   np.clip(np.asarray(h, dtype=float), self._H_min, self._H_max))
        xi = np.column_stack([P.ravel(), h.ravel()]).astype(float)
        if not np.isfinite(xi).all() or not all(self._fast[n] for n in names):
            return [self._interp[n](xi).reshape(P.shape) for n in names]
        idx, dist = _find_indices(self._grid, xi.T)
        return [_evaluate_linear_2d(self._interp[n].values, idx, dist, self._grid,
                                    np.empty(len(xi))).reshape(P.shape) for n in names]

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
        du_max_per_s = 1.0  # [%/s]
        T_sample = 0.2  # [s] sample time for controller

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
        T_suction = float(self._lookup(P_suction, h_suction, ("T",))[0])

        # T_sat [K]
        T_sat = float(self._T_sat_interp(P_suction))

        SH = T_suction - T_sat  # [K] can be negative

        dt_eff = dt
        du_max = du_max_per_s * dt

        tau = 2.0  # [s] Filter Time constant
        alpha = dt_eff / (tau + dt_eff)

        if self._valve_SH_filt is None:
            self._valve_SH_filt = SH

        self._valve_SH_filt = (1 - alpha) * self._valve_SH_filt + alpha * SH
        SH_used = self._valve_SH_filt

        # -----------------------------
        # PI control law with simple anti-windup
        # -----------------------------
        e = SH_used - SH_set  # positive => SH too high => open valve more

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
            rho = float(self._lookup(pi, hi, ("rho",))[0])
        except Exception:
            return 0.0, hi

        if not np.isfinite(rho) or rho <= 0.0:
            return 0.0, hi

        U = VPos / 100
        m = Kv * U * np.sqrt(rho * (pi - po) * 1e-2) / 3600
        ho = hi

        return m, ho

    def compressor_model(self, pi, hi, po, RPM):
        """
        Return (mass flow [kg/s], outlet enthalpy [J/kg]).
        Pure function of its inputs (no state is written), so it can be called
        from the ODE right-hand side. Power: W = m * (ho - hi).
        """
        if RPM <= 1e-6:
            return 0.0, hi
        AS = self.AS
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

        AS.update(HmassP_INPUTS, float(hi), float(pi))
        s = AS.smass()
        AS.update(PSmass_INPUTS, float(po), s)
        h_is = AS.hmass()
        rho = float(self._lookup(pi, hi, ("rho",))[0])

        ho = hi + (h_is - hi) / eta_is
        m = RPM / 60 * Vd * eta_v * rho

        return float(m), float(ho)

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

        lower_tri = {n: np.tril(np.ones((n - 2, n - 1))) for n in (N, N_condenser)}

        def flow_system(M, h, h_in, m_in, m_out, Q, V, rho, rhoP, rhoh):
            """
            Mass and energy balances of a pipe with n control volumes and one pressure,
            for k states at once (h, Q, rho, ...: shape (k, n); h_in, m_in, m_out: (k,)).
            Unknowns [dP/dt, dh_0/dt, ..., dh_n-1/dt]; the internal mass flows are
            eliminated. Returns A (k, M, M) and b (k, M) with rows 0 ... n filled.
            """
            k, n = h.shape
            a, b = V * rhoP, V * rhoh
            c, d = V*(h*rhoP - 1.0), V*(h*rhoh + rho)
            A, rb = np.zeros((k, M, M)), np.zeros((k, M))
            # Total mass and energy balances
            A[:, 0, 0], A[:, 0, 1:n + 1], rb[:, 0] = np.sum(a, axis=-1), b, m_in - m_out
            A[:, 1, 0], A[:, 1, 1:n + 1] = np.sum(c, axis=-1), d
            rb[:, 1] = m_in * h_in - m_out * h[:, -1] + np.sum(Q, axis=-1)
            # Energy balances of the control volumes 0 ... n-2
            A[:, 2, 0] = c[:, 0] - a[:, 0] * h[:, 0]
            A[:, 2, 1] = d[:, 0] - b[:, 0] * h[:, 0]
            rb[:, 2] = m_in * (h_in - h[:, 0]) + Q[:, 0]
            row_e = np.arange(3, n + 1)
            A[:, row_e, 0] = c[:, 1:-1] - a[:, 1:-1] * h[:, 1:-1] - (h[:, 1:-1] - h[:, :-2]) * np.cumsum(a[:, :n - 2], axis=-1)
            A[:, row_e, np.arange(2, n)] = d[:, 1:-1] - b[:, 1:-1] * h[:, 1:-1]
            rb[:, row_e] = m_in[:, None] * (h[:, :-2] - h[:, 1:-1]) + Q[:, 1:-1]
            A[:, row_e[:, None], np.arange(1, n)] += -(h[:, 1:-1] - h[:, :-2])[:, :, None] * b[:, None, :n - 1] * lower_tri[n]
            return A, rb

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

        # Valve position: evaluated once per macro step from the state at t = time
        # and held constant during the integration (keeps rhs free of side effects)
        valve_pos_step = self.valve_controller(time, P_suction=P0, h_suction=h0[-1])

        # ----------------------------------------------------------
        # RHS
        # ----------------------------------------------------------
        def rhs(t, Y):
            """Time derivatives for k states at once (Y: shape (n_states, k), as the BDF solver passes them)."""
            Y = np.ascontiguousarray(Y.T)
            k, Nc, o = Y.shape[0], N_condenser, 1 + 2 * N
            P, h, Tw = Y[:, 0], Y[:, 1:1 + N], Y[:, 1 + N:o]
            P_cond, h_cond = Y[:, o], Y[:, o + 1:o + 1 + Nc]
            T_wall, T_water = Y[:, o + 1 + Nc:o + 1 + 2 * Nc], Y[:, o + 1 + 2 * Nc:o + 1 + 3 * Nc]

            # Compressor and valve: once per distinct input (most Jacobian columns share them)
            RPM, cache, comp, valve = HP.RPM(t), {}, np.empty((k, 2)), np.empty((k, 2))
            for j in range(k):
                key = (Y[j, 0], Y[j, N], Y[j, o], Y[j, o + Nc])
                if key not in cache:
                    cache[key] = (self.compressor_model(float(Y[j, 0]), h[j, -1], P_cond[j], RPM),
                                  self.valve_model(P_cond[j], h_cond[j, -1], float(Y[j, 0]), valve_pos_step))
                comp[j], valve[j] = cache[key]
            (m_comp, h_out_comp), (m_valve, h_out_valve) = comp.T, valve.T

            # Evaporator
            rho, rhoP, rhoh, T_ref_K = self._lookup(P[:, None], h)
            Q_into_ref = h_int_evap * gp.A_inner * (Tw - T_ref_K)
            dTwdt = (Qseg[ixp, iyp] - Q_into_ref) / den_wall
            stacks = self.geometry.stacks
            A_e, b_e = flow_system(N + 1, h, h_out_valve, m_valve / stacks, m_comp / stacks, Q_into_ref, V, rho, rhoP, rhoh)

            # Condenser: refrigerant balances plus wall and water energy balances
            rho_c, rhoP_c, rhoh_c, T_ref_K_cond = self._lookup(P_cond[:, None], h_cond)
            Q_into_ref_cond = h_int_cond * A_plate_seg * (T_wall - T_ref_K_cond)
            Q_wall_water = h_int_water * A_plate_seg * (T_wall - T_water)
            A_c, b_c = flow_system(3 * Nc + 1, h_cond, h_out_comp, m_comp, m_valve, Q_into_ref_cond, V_cond, rho_c, rhoP_c, rhoh_c)
            row_w, row_s = np.arange(Nc + 1, 2 * Nc + 1), np.arange(2 * Nc + 1, 3 * Nc + 1)
            A_c[:, row_w, row_w] = M_wall * HP.c_plate
            b_c[:, row_w] = (-Q_into_ref_cond) - Q_wall_water
            A_c[:, row_s, row_s] = M_water * HP.c_water
            T_prev = np.concatenate([np.full((k, 1), HP.T_in_water_at(t) + 273.15), T_water[:, :-1]], axis=1)
            b_c[:, row_s] = HP.m_water * HP.c_water * (T_prev - T_water) + Q_wall_water

            # [dP, dh, dTw] evaporator, [dP, dh, dT_wall, dT_water] condenser
            x_e = np.linalg.solve(A_e, b_e[..., None])[..., 0]
            x_c = np.linalg.solve(A_c, b_c[..., None])[..., 0]
            return np.concatenate([x_e, dTwdt, x_c], axis=1).T

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
                max_step=np.inf,
                vectorized=True,  # Jacobian columns in one rhs call
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

        m_comp_end, h_out_comp_end = self.compressor_model(P_end, float(h_end[-1]), P_cond_end, HP.RPM(t1))
        W_comp = m_comp_end * (h_out_comp_end - float(h_end[-1]))  # [W]

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
            T_ref_K = float(T_ref_evap_K[k])  # same PropsSI(T, P_end, h_k) as computed above
            T_ref_C = T_ref_K - 273.15

            try:
                x_out = float(PropsSI("Q", "P", P_end, "H", h_k, fluid))
            except ValueError:
                x_out = float("nan")
            if not (0.0 <= x_out <= 1.0):
                x_out = float("nan")  # single phase (CoolProp returns -1)

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
        HP.W_comp = W_comp

        return n_inner


    def reset_integrator(self):
        self._bdf = None
        self._bdf_dim = None

        self.valve_pos = None
        self._valve_I = 0.0
        self._valve_last_t = None
        self._valve_SH_filt = None
