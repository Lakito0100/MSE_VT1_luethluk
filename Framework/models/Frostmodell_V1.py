import numpy as np
from scipy.linalg import get_lapack_funcs
from Framework.core.corrolations import DK

# Picard iteration of the frost solvers: the temperature update is under-relaxed
# with OMEGA_T_START and the factor is halved whenever the residual grows
# (never below OMEGA_T_MIN, the previously used fixed value).
OMEGA_T_START = 0.8
OMEGA_T_MIN = 0.1


def p_ws_Pa(T_C):
    """
    Saturation vapor pressure of water [Pa], vectorised.
    Below 273.16 K over ice (IAPWS 2011, Wagner et al.), above over liquid water
    (Wagner & Pruss). Identical to CoolProp's HAProps_Aux("p_ws") below 0 °C
    (same equation) and within 1e-4 relative above.
    """
    T = np.asarray(T_C, dtype=float) + 273.15
    out = np.empty_like(T)
    ice = T < 273.16
    if np.any(ice):
        th = T[ice] / 273.16
        out[ice] = 611.657 * np.exp((-0.212144006e2 * th**0.333333333e-2
                                     + 0.273203819e2 * th**0.120666667e1
                                     - 0.610598130e1 * th**0.170333333e1) / th)
    if np.any(~ice):
        Tw = T[~ice]
        tau = 1.0 - Tw / 647.096
        out[~ice] = 22.064e6 * np.exp(647.096 / Tw * (
            -7.85951783 * tau + 1.84408259 * tau**1.5 - 11.7866497 * tau**3
            + 22.6807411 * tau**3.5 - 15.9618719 * tau**4 + 1.80122502 * tau**7.5))
    return out


def w_sat_vec(Tf_C, p_Pa):
    """Saturation humidity ratio [kg/kg_da] at Tf_C (float or array; p_Pa may be one value per row)."""
    p_ws = p_ws_Pa(Tf_C)
    w = 0.621945 * p_ws / (np.asarray(p_Pa, dtype=float) - p_ws)
    return float(w) if np.ndim(Tf_C) == 0 else w


# Lower bound of the frost density [kg/m^3] (= initial frost density of the model).
RHO_FROST_MIN = 25.0


def frost_density(T_surface_C, T_wall_C):
    """
    Frost density [kg/m^3] after Hermes et al. (2009):
        rho = 207 * exp(0.266*T_surface - 0.0615*T_wall)
    The correlation was fitted for wall temperatures of about -16 ... -4 °C.
    Below that it predicts unphysically light frost (~5 kg/m^3 at -20 °C), which
    makes a small deposited mass grow into a very thick layer. The result is
    therefore limited to RHO_FROST_MIN.
    """
    return max(207.0 * np.exp(0.266 * T_surface_C - 0.0615 * T_wall_C), RHO_FROST_MIN)


_gtsv = get_lapack_funcs("gtsv", (np.zeros(1),))


def _solve_tridiag(lower, main, upper, b):
    """Solve A x = b for tridiagonal A (lower[i] = A[i+1, i], upper[i] = A[i, i+1]); LAPACK gtsv as in solve_banded."""
    x, info = _gtsv(lower, main, upper, b)[3:]
    if info:
        raise np.linalg.LinAlgError("singular matrix")
    return x


def _picard_rows(T, w, d, fa, fg, D_int, k_int, k_s, D_s, h, T_wall, ups, edge, tol, niter):
    """
    Picard iteration of the coupled frost temperature T and humidity ratio w (arrays (k, N))
    of k segments at once. Every segment keeps its own relaxation factor and stops at its own
    convergence, so it follows exactly the iteration it would follow alone.
    d (k, 1): node spacing; fa, fg ((k, 1) or (k, N-2)): factors of the upper / lower neighbour
    in the interior equations; D_int, k_int (k, N-2): diffusivity / conductivity of the interior nodes;
    k_s, D_s, h, T_wall (k, 1): surface conductivity / diffusivity, air-side heat transfer
    coefficient, wall temperature; ups: upstream air states (CaseConfig) of the segments.
    edge=True: no surface flux while the air is drier than saturation (edge model), otherwise
    only the air-side vapour flux is zero (fin/tube model).
    Returns T_new, w_new, the dry-air density and mass transfer coefficient of the last
    iteration, the "air drier than surface" flag, and iterations / residuals per segment.
    """
    p, cp, w_amb, T_a, h_sub, C, isv = (np.array([getattr(c, n) for c in ups], dtype=float)[:, None]
                                        for n in ("p_a", "c_p_a", "w_amb", "T_a", "h_sub", "C", "isv"))
    k, N = T.shape
    one, zero = np.ones((k, 1)), np.zeros((k, 1))
    inv_d2 = 1.0 / (d * d)
    # Temperature equation: constant matrix (the conductivity only depends on the frost density)
    main_T = np.hstack([one, -2.0 * k_int * inv_d2, k_s / d])
    upper_T, lower_T = np.hstack([zero, k_int * fa]), np.hstack([k_int * fg, -k_s / d])
    T_new, w_new, rho_a, hm = np.empty_like(T), np.empty_like(w), np.empty_like(T), np.empty((k, 1))
    neg, it = np.zeros(k, dtype=bool), np.zeros(k, dtype=int)
    omega, res_T, res_w = np.full(k, OMEGA_T_START), np.full(k, np.inf), np.full(k, np.inf)
    a = np.arange(k)  # segments still iterating
    while a.size:
        To, wo, pa = T[a], w[a], p[a]
        rho_a[a] = ra = pa / (287.058 * (To + 273.15))
        # Surface: saturation, vapour fluxes (air side minus diffusion into the frost), heat flux
        Tfs, rs, wfs = To[:, -1:], ra[:, -1:], w_sat_vec(To[:, -1:], pa)
        dw = w_amb[a] - wfs
        neg[a] = ~(dw[:, 0] >= 0.0)
        hm[a] = h[a] / (rs * cp[a])
        m_f, m_rho = hm[a] * rs * dw, D_s[a] * rs * ((wo[:, -1:] - wo[:, -2:-1]) / d[a])
        m_delta = np.where(dw >= 0.0, m_f - m_rho, 0.0) if edge else np.where(dw >= 0.0, m_f, 0.0) - m_rho
        q_tot = h[a] * (T_a[a] - Tfs) + h_sub[a] * m_delta
        # Humidity equation (depends on the dry-air density) and right-hand sides
        ri, wsi = ra[:, 1:-1], w_sat_vec(To[:, 1:-1], pa)
        A = D_int[a] * ri
        o, z = one[:a.size], zero[:a.size]
        main_w = np.hstack([-o, -2.0 * A * inv_d2[a] - C[a] * ri, o])
        upper_w, lower_w = np.hstack([o, A * fa[a]]), np.hstack([A * fg[a], z])
        b_w = np.hstack([z, -C[a] * ri * wsi, wfs])
        b_T = np.hstack([T_wall[a], -isv[a] * C[a] * ri * (wo[:, 1:-1] - wsi), q_tot])
        for i, j in enumerate(a):
            T_new[j] = _solve_tridiag(lower_T[j], main_T[j], upper_T[j], b_T[i])
            w_new[j] = _solve_tridiag(lower_w[i], main_w[i], upper_w[i], b_w[i])
        # Convergence and adaptive under-relaxation (omega_T halves if the residual grows)
        res_prev = res_T[a]
        res_T[a] = np.max(np.abs(T_new[a] - To), axis=1)
        res_w[a] = np.max(np.abs(w_new[a] - wo), axis=1)
        omega[a] = np.where(res_T[a] > res_prev, np.maximum(0.5 * omega[a], OMEGA_T_MIN), omega[a])
        om = omega[a][:, None]
        T[a] = (1 - om) * To + om * T_new[a]
        w[a] = (1 - 1.0) * wo + 1.0 * w_new[a]
        it[a] += 1
        a = a[(it[a] < niter) & ((res_T[a] > tol) | (res_w[a] > tol))]
    return T_new, w_new, rho_a, hm, neg, it, res_T, res_w


class Frostmodell_Edge:
    """Edge frost model with coupled heat/mass transfer at the fin edge."""

    @staticmethod
    def w_sat_coolprop(Tf_C, p_Pa):
        """
        Saturation humidity ratio over ice/water [kg/kg_da].
        Tf_C: float or array-like (°C), p_Pa: float (Pa). Returns float or np.ndarray.
        Same saturation pressure as CoolProp's HAProps_Aux("p_ws"), but vectorised.
        """
        return w_sat_vec(Tf_C, p_Pa)

    @staticmethod
    def Nu_edge(cfg, geom, theta):
        """Return the edge Nusselt number correlation."""
        Re_d = DK.Re(cfg.v_a, geom.fin_pitch_cc, cfg.v_kin)
        Pr = DK.Pr(cfg.v_kin, cfg.lam, cfg.c_p_a, cfg.rho_amb)
        return 0.23 * (Re_d**0.466) * (Pr**(1/3)) * (0.7 + 1.06e-4 * (theta - 90)**2)

    def h_conv(self, cfg, geom, theta):
        """Convective heat transfer coefficient at the edge."""
        Nu = self.Nu_edge(cfg, geom, theta)
        return Nu * cfg.lam / geom.fin_pitch_cc

    def h_mass(self, cfg, geom, st, theta):
        """Mass transfer coefficient based on convective heat transfer."""
        h = self.h_conv(cfg, geom, theta)
        return h / (st.rho_a_e[-1,theta] * cfg.c_p_a)

    def m_dot_f(self, cfg, geom, st, theta):
        """Convective vapor mass flux to the frost surface."""
        hm = self.h_mass(cfg, geom, st, theta)
        w_fs = st.w_e[-1, theta]
        return hm * st.rho_a_e[-1,theta] * (cfg.w_amb - w_fs)

    def m_dot_rho_f(self, cfg, geom, st, gs, theta):
        """Diffusive vapor mass flux inside frost at the interface."""
        Deff = self.D_eff(cfg, st, -1, theta)
        dr = (0.5*geom.fin_thickness + st.s_e[theta] - 0.5*geom.fin_thickness) / (gs.nr - 1)
        dwf_dr = (st.w_e[-1, theta] - st.w_e[-2, theta]) / dr
        return Deff * st.rho_a_e[-1,theta] * dwf_dr

    def m_dot_s_f(self, cfg, geom, st, gs, theta):
        """Net vapor mass flux at the frost surface."""
        return self.m_dot_f(cfg, geom, st, theta) - self.m_dot_rho_f(cfg, geom, st, gs, theta)

    @staticmethod
    def D_eff(cfg, st, r, theta):
        """Effective diffusion coefficient in frost."""
        numerator = cfg.D_std * (cfg.rho_i - st.rho_e[r,theta])
        denominator = cfg.rho_i - 0.58 * st.rho_e[r,theta]
        return numerator / denominator

    @staticmethod
    def k_eff(st, r, theta):
        """Effective thermal conductivity of the frost layer."""
        return 0.132 + 3.13e-4 * st.rho_e[r,theta] + 1.6e-7 * (st.rho_e[r,theta])**2

    @staticmethod
    def rho_a_dry_local(Tf_C, p_Pa):
        """Local dry-air density at temperature Tf_C and pressure p_Pa."""
        Tf_K = np.asarray(Tf_C) + 273.15
        R = 287.058
        return p_Pa / (R*Tf_K)

    def New_edge_state_seg_at_90(self, cfg_up, cfg, geom, st, gs, tol=1e-6, niter=1000):
        """Solve the edge frost state at theta=90° for one segment; returns (iterations, res_T, res_w)."""
        return self.New_edge_state_row_at_90([cfg_up], [cfg], geom, [st], gs, tol, niter)[0]

    def New_edge_state_row_at_90(self, cfg_ups, cfgs, geom, sts, gs, tol=1e-6, niter=1000):
        """Edge frost state at theta=90° for several segments at once; returns [(iterations, res_T, res_w)]."""
        j, idx = gs.ntheta - 1, np.arange(1, gs.nr - 1)
        r0 = 0.5 * geom.fin_thickness
        r = np.array([np.linspace(r0, r0 + float(st.s_e[j]), gs.nr) for st in sts])  # radial grid per segment
        dr = r[:, 1:2] - r[:, 0:1]
        inv_dr2, inv_2rdr = 1.0 / (dr * dr), 1.0 / (2.0 * r[:, idx] * dr)
        col = lambda vals: np.array(vals, dtype=float)[:, None]
        T_new, w_new, *_, it, res_T, res_w = _picard_rows(
            np.array([st.T_e[:, j] for st in sts]), np.array([st.w_e[:, j] for st in sts]), dr,
            inv_dr2 + inv_2rdr, inv_dr2 - inv_2rdr,
            np.array([self.D_eff(c, st, idx, j) for c, st in zip(cfg_ups, sts)]),
            np.array([self.k_eff(st, idx, j) for st in sts]),
            col([self.k_eff(st, -1, j) for st in sts]), col([self.D_eff(c, st, -1, j) for c, st in zip(cfg_ups, sts)]),
            col([self.h_conv(c, geom, j) for c in cfg_ups]), col([c.T_tube for c in cfgs]), cfg_ups, True, tol, niter)

        # Write back, then explicit update of frost density and thickness
        for n, (c_up, c, st) in enumerate(zip(cfg_ups, cfgs, sts)):
            st.T_e[:, j], st.w_e[:, j] = T_new[n], w_new[n]
            st.rho_a_e[:gs.nr, j] = self.rho_a_dry_local(T_new[n][:gs.nr], c_up.p_a)
            st.rho_e[:, j] = frost_density(st.T_e[-1, j], c.T_tube)
            st.s_e[j] = max(st.s_e[j] + (self.m_dot_s_f(c_up, geom, st, gs, j) / st.rho_e[-1, j]) * gs.dt, 1e-6)
        return list(zip(it, res_T, res_w))

class Frostmodell_Finn_and_Tube:
    """Frost model for the fin-and-tube domain."""

    def __init__(self):
        # Set by New_finn_and_tube_state_seg: True if the air-side vapour flux was
        # negative in the last call (m_f set to 0)
        self.last_negative_mass_flux = False

    @staticmethod
    def w_sat_coolprop(Tf_C, p_Pa):
        """
        Saturation humidity ratio over ice/water [kg/kg_da].
        Tf_C: float or array-like (°C), p_Pa: float (Pa). Returns float or np.ndarray.
        Same saturation pressure as CoolProp's HAProps_Aux("p_ws"), but vectorised.
        """
        return w_sat_vec(Tf_C, p_Pa)

    @staticmethod
    def rho_a_dry_local(Tf_C, p_Pa):
        """Local dry-air density at temperature Tf_C and pressure p_Pa."""
        Tf_K = np.asarray(Tf_C) + 273.15
        R = 287.058
        return p_Pa / (R * Tf_K)

    @staticmethod
    def D_eff(cfg, st, i):
        """Effective diffusion coefficient in the fin/tube frost."""
        rho_f = st.rho_ft[i]
        numerator = cfg.D_std * (cfg.rho_i - rho_f)
        denominator = cfg.rho_i - 0.58 * rho_f
        return numerator / denominator

    @staticmethod
    def k_f(st, i):
        """Effective thermal conductivity in the fin/tube frost."""
        rho_f = st.rho_ft[i]
        return 0.132 + 3.13e-4 * rho_f + 1.6e-7 * (rho_f ** 2)

    def alpha_tube(self, cfg, geom):
        """Heat transfer coefficient around the tube."""
        l = np.pi * geom.d_tube_a / 2.0
        Re = DK.Re(cfg.v_a, l, cfg.v_kin)
        Pr = 0.7
        Nu_lam = 0.664 * np.sqrt(Re) * Pr ** (1 / 3)
        Nu_turb = (0.037 * (Re ** 0.8) * Pr) / (1 + 2.443 * (Re ** -0.1) * (Pr ** (2 / 3) - 1))
        Nu = 0.3 + np.sqrt(Nu_lam ** 2 + Nu_turb ** 2)
        alpha = Nu * cfg.lam / l
        return alpha

    def h_eff(self, cfg, geom, st):
        """Effective air-side heat transfer coefficient including fin efficiency."""
        h_0 = self.alpha_tube(cfg,geom)
        mue_fin = geom.mue_fin(h_0)
        A_G = geom.A_tube_one_segment()
        A_R = geom.A_fin_one_segment()
        A = geom.A_one_segment_frost(st.s_ft)
        h_eff = h_0 * (A_G/A + mue_fin*A_R/A)
        return h_eff

    def New_finn_and_tube_state_seg(self, cfg_up, cfg, geom, st, gs, tol=1e-6, niter=1000):
        """Solve the fin-and-tube frost state for one segment; returns (iterations, res_T, res_w)."""
        it, res_T, res_w, self.last_negative_mass_flux = self.New_finn_and_tube_state_row([cfg_up], [cfg], geom, [st], gs, tol, niter)[0]
        return it, res_T, res_w

    def New_finn_and_tube_state_row(self, cfg_ups, cfgs, geom, sts, gs, tol=1e-6, niter=1000):
        """
        Fin-and-tube frost state for several segments at once.
        Returns [(iterations, res_T, res_w, negative air-side vapour flux)] per segment.
        """
        N, idx = gs.nx, np.arange(1, gs.nx - 1)
        col = lambda vals: np.array(vals, dtype=float)[:, None]
        x = [np.linspace(0.0, max(float(st.s_ft), 1e-6), N) for st in sts]  # frost grid per segment
        dx = col([xi[1] - xi[0] for xi in x])
        inv_dx2 = 1.0 / (dx * dx)
        T_new, w_new, rho_a, hm, neg, it, res_T, res_w = _picard_rows(
            np.array([st.T_ft for st in sts], dtype=float), np.array([st.w_ft for st in sts], dtype=float), dx,
            inv_dx2, inv_dx2, np.array([self.D_eff(c, st, idx) for c, st in zip(cfg_ups, sts)]),
            np.array([self.k_f(st, idx) for st in sts]), col([self.k_f(st, N - 1) for st in sts]),
            col([self.D_eff(c, st, N - 1) for c, st in zip(cfg_ups, sts)]),
            col([self.h_eff(c, geom, st) for c, st in zip(cfg_ups, sts)]), col([c.T_tube for c in cfgs]),
            cfg_ups, False, tol, niter)

        # Write back, then explicit update of frost density and thickness growth at the surface
        for n, (c_up, c, st) in enumerate(zip(cfg_ups, cfgs, sts)):
            st.T_ft, st.w_ft = T_new[n].copy(), w_new[n].copy()
            st.rho_a_ft[:] = rho_a[n]
            st.rho_ft[:] = frost_density(st.T_ft[-1], c.T_tube)
            m_rho_s = self.D_eff(c_up, st, N - 1) * self.rho_a_dry_local(st.T_ft[-1], c_up.p_a) * ((st.w_ft[-1] - st.w_ft[-2]) / dx[n, 0])
            m_f_s = 0.0 if neg[n] else hm[n, 0] * st.rho_a_ft[-1] * (c_up.w_amb - st.w_ft[-1])
            st.s_ft = max(st.s_ft + ((m_f_s - m_rho_s) / st.rho_ft[-1]) * gs.dt, 1e-6)
        return list(zip(it, res_T, res_w, neg))

    def segment_flows(self, cfg, geom, st, gs):
        """
        Flows between the air (upstream state cfg) and the frost of one segment:
        (vapour mass flow into the frost [kg/s], sensible heat flow from the air [W],
         heat flow into the wall [W], sensible heat flow for the frost-free case [W]).
        """
        h_eff = self.h_eff(cfg, geom, st)
        dx = max(float(st.s_ft), 1e-6) / (gs.nx - 1)
        Tfs, wfs = float(st.T_ft[-1]), float(st.w_ft[-1])
        rho_a_s = self.rho_a_dry_local(Tfs, cfg.p_a)
        dw = cfg.w_amb - wfs
        m_fs = h_eff / (rho_a_s * cfg.c_p_a) * rho_a_s * dw if dw >= 0.0 else 0.0  # [kg/(m² s)]
        q_sens_fs = h_eff * (cfg.T_a - Tfs)  # [W/m²]
        q_x0 = self.k_f(st, 0) * (st.T_ft[1] - st.T_ft[0]) / dx  # conduction into the wall [W/m²]
        A_seg_frost = geom.A_one_segment_frost(st.s_ft)
        return m_fs * A_seg_frost, q_sens_fs * A_seg_frost, q_x0 * geom.A_one_segment(), q_sens_fs * A_seg_frost
