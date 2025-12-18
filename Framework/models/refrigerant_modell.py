from time import perf_counter
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from numpy.linalg import solve
import CoolProp.CoolProp as CP
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
    def __init__(self,geometry, geom_params: RefGeomParams, cfg, path_variant="row_serpentine"):
        self.geom = geom_params
        self.fluid = cfg.ref_str
        self.AS = CP.AbstractState("HEOS", self.fluid)  # or "BICUBIC&HEOS", "TTSE&HEOS", etc.
        self.connection_path = geometry.build_connection_path(variant=path_variant)


    def rho_and_derivs(self, p_i, h_i, x_i):
        """
        Returns density and its partial derivatives wrt p and h.

        Inputs:
            p_i  [Pa]      local pressure
            h_i  [J/kg]    local specific enthalpy
            x_i  [-]       local vapour quality from your model (cfg.x_ref)
        """

        AS = self.AS

        # Update state with (P, H)
        AS.update(CP.HmassP_INPUTS, h_i, p_i)
        rho_i = AS.rhomass()

        # Decide phase via your own quality
        # two-phase if 0 < x < 1 (you can add a small tolerance if needed)
        if 0.0 < x_i < 1.0:
            # Two-phase (Thorade) derivatives
            drho_dp = AS.first_two_phase_deriv(CP.iDmass, CP.iP,     CP.iHmass)   # (∂ρ/∂p)|h
            drho_dh = AS.first_two_phase_deriv(CP.iDmass, CP.iHmass, CP.iP)       # (∂ρ/∂h)|p
        else:
            # Single-phase derivatives
            drho_dp = AS.first_partial_deriv(CP.iDmass, CP.iP,     CP.iHmass)     # (∂ρ/∂p)|h
            drho_dh = AS.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)         # (∂ρ/∂h)|p

        return rho_i, drho_dp, drho_dh

    # ------------------------------------------------------------------
    # Lokale Korrelation für h_int (Wärmeübergang Wand ↔ Kältemittel)
    # ------------------------------------------------------------------
    def h_int_corr(self,
                    x:float):
        return 5000.0

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
        """
        Integrates:
          - one global refrigerant pressure P(t)  [Pa]
          - segment enthalpies h_i(t)             [J/kg]
          - segment tube wall temperatures Tw_i(t)[K]

        At each RHS call, solves the coupled FV refrigerant equations
        for [dP/dt, dh_i/dt] AND the internal face mass flows m_faces,
        given boundary mass flows m_in and m_out.

        Sign convention:
          Q_ref_i > 0  means heat INTO the refrigerant [W]
          Wall ODE uses dTw/dt = (Q_f_i - Q_ref_i) / (m_wall*c_wall)

        NOTE:
          You must supply a boundary outlet mass flow m_out.
          If you have no compressor model yet, set m_out = m_in (fallback below).
        """

        fluid = cfg_inlet.ref_str
        gp = self.geom
        path = self.connection_path
        N = len(path)

        # ---------------------------
        # Helper: solve 2N linear system
        # Unknowns x = [dPdt, dhdt_0..dhdt_{N-1}, m_face_1..m_face_{N-1}]
        # Equations: 2 per cell (mass + energy)
        # Boundaries: m_face_0 = m_in, m_face_N = m_out
        # ---------------------------
        def solve_dp_dh_and_mfaces(P, h, h_in, m_in, m_out, Q_into_ref, V, rho, rhoP, rhoh):
            """
            Returns:
              dPdt: scalar
              dhdt: (N,) array
              m_faces: (N+1,) array of face mass flows, where:
                  m_faces[0]   = m_in
                  m_faces[j]   = internal face j, j=1..N-1
                  m_faces[N]   = m_out
            """
            # if N == 1:
            #     # 2x2 system for dPdt and dhdt only
            #     a = V[0] * rhoP[0]
            #     b = V[0] * rhoh[0]
            #     c = V[0] * (h[0] * rhoP[0] - 1.0)
            #     d = V[0] * (h[0] * rhoh[0] + rho[0])
            #
            #     rhs_m = m_in - m_out
            #     rhs_e = m_in * h_in - m_out * h[0] + Q_into_ref[0]
            #
            #     det = a * d - b * c
            #     dPdt = (rhs_m * d - b * rhs_e) / det
            #     dhdt = np.array([(a * rhs_e - rhs_m * c) / det], dtype=float)
            #     m_faces = np.array([m_in, m_out], dtype=float)
            #     return dPdt, dhdt, m_faces

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

            # n_unknown = 2 * N  # 1 + N + (N-1)
            # A = np.zeros((2 * N, n_unknown), dtype=float)
            # bvec = np.zeros((2 * N,), dtype=float)
            #
            # idx_dP = 0
            #
            # def idx_dh(k):  # k=0..N-1
            #     return 1 + k
            #
            # def idx_mface(j):  # j=1..N-1
            #     return 1 + N + (j - 1)
            #
            # for k in range(N):
            #     V_k = V[k]
            #     rho_k = rho[k]
            #     rhoP_k = rhoP[k]
            #     rhoh_k = rhoh[k]
            #     h_k = h[k]
            #     h_up = h_in if k == 0 else h[k - 1]
            #
            #     # FV coefficients
            #     a = V_k * rhoP_k
            #     bb = V_k * rhoh_k
            #     c = V_k * (h_k * rhoP_k - 1.0)
            #     d = V_k * (h_k * rhoh_k + rho_k)
            #
            #     # Face indexing:
            #     # left face is j=k   (0..N-1)  -> m_faces[k]
            #     # right face is j=k+1 (1..N)   -> m_faces[k+1]
            #     left_is_boundary = (k == 0)
            #     right_is_boundary = (k == N - 1)
            #
            #     # ---------- Mass row ----------
            #     # a dP + b dh = m_left - m_right
            #     row_m = 2 * k
            #     A[row_m, idx_dP] = a
            #     A[row_m, idx_dh(k)] = bb
            #
            #     # RHS known boundary contributions
            #     bvec[row_m] = (m_in if left_is_boundary else 0.0) - (m_out if right_is_boundary else 0.0)
            #
            #     # Unknown internal faces on LHS: -m_left + m_right
            #     if not left_is_boundary:
            #         # left internal face index is j=k (1..N-1)
            #         A[row_m, idx_mface(k)] += -1.0
            #     if not right_is_boundary:
            #         # right internal face index is j=k+1 (1..N-1)
            #         A[row_m, idx_mface(k + 1)] += +1.0
            #
            #     # ---------- Energy row ----------
            #     # c dP + d dh = m_left*h_up - m_right*h_k + Q_into_ref
            #     # => c dP + d dh - m_left*h_up + m_right*h_k = Q_into_ref
            #     row_e = 2 * k + 1
            #     A[row_e, idx_dP] = c
            #     A[row_e, idx_dh(k)] = d
            #
            #     bvec[row_e] = Q_into_ref[k]
            #     if left_is_boundary:
            #         bvec[row_e] += m_in * h_up
            #     if right_is_boundary:
            #         bvec[row_e] += -m_out * h_k
            #
            #     if not left_is_boundary:
            #         A[row_e, idx_mface(k)] += -h_up
            #     if not right_is_boundary:
            #         A[row_e, idx_mface(k + 1)] += +h_k

            x = np.linalg.solve(A_mat, b_vec)

            dPdt = float(x[0])
            dhdt = x[1:1 + N].copy()

            #m_faces = np.empty(N + 1, dtype=float)
            #m_faces[0] = m_in
            #m_faces[N] = m_out
            #m_faces[1:N] = x[1 + N:]  # m_face_1..m_face_{N-1}

            return dPdt, dhdt

        # ----------------------------------------------------------
        # Initial state
        # ----------------------------------------------------------
        h0 = np.zeros(N, dtype=float)
        Tw0 = np.zeros(N, dtype=float)

        # Pressure is global: pick inlet if available, else first segment
        (ix0, iy0) = path[0]
        cfg0 = cfg_grid[ix0][iy0]
        P0 = float(getattr(cfg_inlet, "p_ref", cfg0.p_ref))

        for k, (ix, iy) in enumerate(path):
            cfg = cfg_grid[ix][iy]
            h0[k] = float(cfg.h_ref)
            T_tube_C = getattr(cfg, "T_tube", cfg.T_ref)
            Tw0[k] = float(T_tube_C + 273.15)

        # State vector: [P, h_0..h_{N-1}, Tw_0..Tw_{N-1}]
        y0 = np.concatenate([np.array([P0]), h0, Tw0])

        # ----------------------------------------------------------
        # Boundary mass flows (must come from cycle coupling)
        # ----------------------------------------------------------
        m_in = float(cfg_inlet.m_dot_ref)

        # Best practice: provide m_out from compressor coupling.
        (ixL, iyL) = path[-1]
        cfgL = cfg_grid[ixL][iyL]
        m_out = float(cfgL.m_dot_ref_out)

        # ----------------------------------------------------------
        # RHS
        # ----------------------------------------------------------
        def rhs(t, y):
            P = float(y[0])
            h = y[1:1 + N]
            Tw = y[1 + N:1 + 2 * N]

            # Per-segment arrays needed for linear solve
            V = np.full(N, gp.A_flow * gp.dx, dtype=float)
            rho = np.zeros(N, dtype=float)
            rhoP = np.zeros(N, dtype=float)
            rhoh = np.zeros(N, dtype=float)
            Q_into_ref = np.zeros(N, dtype=float)  # + into refrigerant [W]
            dTwdt = np.zeros(N, dtype=float)

            # Build property and heat-transfer terms
            for k, (ix, iy) in enumerate(path):
                cfg = cfg_grid[ix][iy]
                h_k = float(h[k])

                # Quality for correlations (optional but useful)
                try:
                    x_k = float(PropsSI("Q", "P", P, "H", h_k, fluid))
                except ValueError:
                    x_k = float("nan")

                # density and derivatives at (P, h_k)
                rho_k, drho_dP_k, drho_dh_k = self.rho_and_derivs(P, h_k, x_k)
                rho[k] = float(rho_k)
                rhoP[k] = float(drho_dP_k)
                rhoh[k] = float(drho_dh_k)

                # Refrigerant temperature
                T_ref_K = float(PropsSI("T", "P", P, "H", h_k, fluid))

                # Internal HTC
                h_int_k = float(self.h_int_corr(x=x_k))

                # Heat rate into refrigerant (your convention)
                Q_ref_k = h_int_k * gp.A_inner * (Tw[k] - T_ref_K)  # [W]
                Q_into_ref[k] = Q_ref_k

                # External heat into wall (from air/frost model)
                Q_f_k = float(Q_seg_list[ix][iy])  # [W] into wall

                # Wall ODE: (in - out) / (m*c)
                dTwdt[k] = (Q_f_k - Q_ref_k) / (gp.rho_wall * gp.c_wall * gp.V_wall)

            # Solve coupled refrigerant linear system
            dPdt, dhdt = solve_dp_dh_and_mfaces(
                P=P,
                h=h,
                h_in=float(cfg_inlet.h_ref),
                m_in=m_in,
                m_out=m_out,
                Q_into_ref=Q_into_ref,
                V=V,
                rho=rho,
                rhoP=rhoP,
                rhoh=rhoh,
            )

            # Pack derivative vector
            return np.concatenate([np.array([dPdt]), dhdt, dTwdt])

        # ----------------------------------------------------------
        # Integrate
        # ----------------------------------------------------------
        t0 = float(time)
        t1 = float(time + dt)

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
            raise RuntimeError(f"Refrigerant ODE solver failed: {sol.message}")

        # End state
        y_end = sol.y[:, -1]
        P_end = float(y_end[0])
        h_end = y_end[1:1 + N]
        Tw_end = y_end[1 + N:1 + 2 * N]

        # Compute end-of-step internal mass flows for storing (optional but useful)
        # Re-evaluate Q and properties at end state
        #V_end = np.full(N, gp.A_flow * gp.dx, dtype=float)
        #rho_end = np.zeros(N, dtype=float)
        #rhoP_end = np.zeros(N, dtype=float)
        #rhoh_end = np.zeros(N, dtype=float)
        #Q_into_ref_end = np.zeros(N, dtype=float)

        #for k, (ix, iy) in enumerate(path):
            #h_k = float(h_end[k])
            #try:
                #x_k = float(PropsSI("Q", "P", P_end, "H", h_k, fluid))
            #except ValueError:
                #x_k = float("nan")

            #rho_k, drho_dP_k, drho_dh_k = self.rho_and_derivs(P_end, h_k, x_k)
            #rho_end[k] = float(rho_k)
            #rhoP_end[k] = float(drho_dP_k)
            #rhoh_end[k] = float(drho_dh_k)

            #T_ref_K = float(PropsSI("T", "P", P_end, "H", h_k, fluid))
            #h_int_k = float(self.h_int_corr(x=x_k))
            #Q_into_ref_end[k] = h_int_k * gp.A_inner * (Tw_end[k] - T_ref_K)

        #_dPdt_end, _dhdt_end = solve_dp_dh_and_mfaces(
            #P=P_end,
            #h=h_end,
            #h_in=float(cfg_inlet.h_ref),
            #m_in=m_in,
            #vm_out=m_out,
            #Q_into_ref=Q_into_ref_end,
            #V=V_end,
            #rho=rho_end,
            #rhoP=rhoP_end,
            #rhoh=rhoh_end,
        #)

        # Console output
        n_inner = len(sol.t) - 1
        T_out0_K = float(PropsSI("T", "P", P_end, "H", float(h_end[0]), fluid))
        T_out0_C = T_out0_K - 273.15
        mean_Tw_C = float(np.mean(Tw_end) - 273.15)
        ref_time = cycl_t1_ref - cycl_t0_ref

        print(
            "Refrigerant Domain Inner Steps: " + str(n_inner) +
            " \t T_out_ref: " + f"{T_out0_C:.3e}" +
            " \t mean T_tube: " + f"{mean_Tw_C:.3e}" +
            " \t cycle time: " + f"{ref_time:.3f} s"
        )

        # ----------------------------------------------------------
        # Write back to cfg_grid
        # ----------------------------------------------------------
        for k, (ix, iy) in enumerate(path):
            cfg = cfg_grid[ix][iy]
            h_k = float(h_end[k])
            Tw_C = float(Tw_end[k] - 273.15)

            # Cell-centered mass flow for storage: average of adjacent faces
            #m_cell = 0.5 * (m_faces_end[k] + m_faces_end[k + 1])

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
            #cfg.m_dot_ref = float(m_cell)
            cfg.x_ref = x_out
            cfg.T_tube = Tw_C