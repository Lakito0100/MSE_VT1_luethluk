import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from Framework_V2.core.corrolations import DK
from CoolProp.HumidAirProp import HAPropsSI

class Frostmodell_Edge:

    @staticmethod
    def w_sat_coolprop(Tf_C: float, p_Pa: float) -> float:
        Tf_K = Tf_C + 273.15
        return HAPropsSI("W", "T", Tf_K, "P", p_Pa, "R", 1.0)

    @staticmethod
    def Nu_edge(cfg, geom, theta):
        Re_d = DK.Re(cfg.v_a, geom.fin_pitch, cfg.v_kin)
        Pr = DK.Pr(cfg.v_kin, cfg.lam, cfg.c_p_a, cfg.rho_amb)
        return 0.23 * (Re_d**0.466) * (Pr**(1/3)) * (0.7 + 1.06e-4 * (theta - 90)**2)

    def h_conv(self, cfg, geom, theta):
        Nu = self.Nu_edge(cfg, geom, theta)
        return Nu * cfg.lam / geom.fin_pitch

    def q_dot_sens_fs(self, cfg, geom, st, theta):
        T_fs = st.T_e[-1, theta]
        return self.h_conv(cfg, geom, theta) * (cfg.T_a - T_fs)

    def h_mass(self, cfg, geom, theta):
        h = self.h_conv(cfg, geom, theta)
        return h / (cfg.rho_amb * cfg.c_p_a)

    def m_dot_f(self, cfg, geom, st, theta):
        hm = self.h_mass(cfg, geom, theta)
        w_fs = st.w_e[-1, theta]
        return hm * st.rho_a[-1,theta] * (cfg.w_amb - w_fs)

    def m_dot_rho_f(self, cfg, geom, st, gs, theta):
        Deff = self.D_eff(cfg, st, -1, theta)
        dr = (0.5*geom.fin_thickness + st.s_e[theta] - 0.5*geom.fin_thickness) / (gs.nr - 1)
        dwf_dr = (st.w_e[-1, theta] - st.w_e[-2, theta]) / dr
        return Deff * st.rho_a[-1,theta] * dwf_dr

    def m_dot_s_f(self, cfg, geom, st, gs, theta):
        return self.m_dot_f(cfg, geom, st, theta) - self.m_dot_rho_f(cfg, geom, st, gs, theta)

    def q_dot_lat_fs(self, cfg, geom, st, gs, theta):
        return cfg.h_sub * self.m_dot_s_f(cfg, geom, st, gs, theta)

    def q_dot_tot_fs(self, cfg, geom, st, gs, theta):
        return self.q_dot_sens_fs(cfg, geom, st, theta) + self.q_dot_lat_fs(cfg, geom, st, gs, theta)

    @staticmethod
    def D_eff(cfg, st, r, theta):
        numerator = cfg.D_std * (cfg.rho_i - st.rho_e[r,theta])
        denominator = cfg.rho_i - 0.58 * st.rho_e[r,theta]
        return numerator / denominator

    @staticmethod
    def k_eff(st, r, theta):
        return 0.132 + 3.13e-4 * st.rho_e[r,theta] + 1.6e-7 * (st.rho_e[r,theta])**2

    @staticmethod
    def rho_a_dry_local(Tf_C, w, p_Pa):
        Tf_K = np.asarray(Tf_C) + 273.15
        w = np.asarray(w)
        Rd = 287.058
        eps = 0.62198
        return p_Pa / (Rd * Tf_K * (1.0 + w / eps))

    def New_edge_state_seg(self, cfg, geom, st, gs, tol = 1e-6, niter = 1000):
        it = 0
        res_T = res_w = np.inf
        T_f_old = np.asarray(st.T_e, dtype=float).copy()
        w_f_old = np.asarray(st.w_e, dtype=float).copy()
        T_f_new = np.empty_like(T_f_old)
        w_f_new = np.empty_like(w_f_old)

        while (it < niter) and ((res_T > tol) or (res_w > tol)):

            for j in range(gs.ntheta):
                r_start = geom.fin_thickness*0.5
                r_end = float(st.s_e[j]) + r_start
                r = np.linspace(r_start, r_end, gs.nr)
                N = len(r)

                # lokale trockene Luftdichte im Frost aktualisieren (über alle Radialknoten)
                st.rho_a[:N, j] = self.rho_a_dry_local(T_f_old[:N, j], w_f_old[:N, j], cfg.p_a)

                dr = r[1] - r[0]
                Tfs = T_f_old[-1, j]
                wfs = w_f_old[-1, j]
                hm = self.h_mass(cfg, geom, j) * cfg.rho_amb  # hm·ρ_a (vgl. 9.10–9.11)
                m_f = hm * (cfg.w_amb - wfs)  # (9.11)
                De_s = self.D_eff(cfg, st, -1, j)
                rhoa_s = st.rho_a[-1, j]
                gradw = (w_f_old[-1, j] - w_f_old[-2, j]) / dr
                m_rho = De_s * rhoa_s * gradw  # (9.12)
                m_delta = m_f - m_rho  # (9.13)

                q_sens = self.h_conv(cfg, geom, j) * (cfg.T_a - Tfs)  # (9.9)
                q_tot = q_sens + cfg.h_sub * m_delta  # (9.16)

                A_w = lil_matrix((N, N), dtype=float)
                b_w = np.zeros(N)
                A_T = lil_matrix((N, N), dtype=float)
                b_T = np.zeros(N)

                for i in range(len(r)):
                    if i == 0:
                        A_w[i,i] = -1.0
                        A_w[i,i+1] = 1.0
                        b_w[i] = 0.0

                        A_T[i,i] = 1.0
                        b_T[i] = cfg.T_w # Define T_edge ---------------------------------
                    elif i == N-1:
                        #A_w[i,i] = self.D_eff(cfg,st,i,j)*st.rho_a[i,j]/dr + self.h_mass(cfg,geom,j)
                        #A_w[i,i-1] = - self.D_eff(cfg,st,i,j)*st.rho_a[i,j]/dr
                        #b_w[i] = self.h_mass(cfg,geom,j) * cfg.w_amb * cfg.rho_amb
                        A_w[i, i] = 1.0
                        b_w[i] = self.w_sat_coolprop(T_f_old[-1, j], cfg.p_a)

                        A_T[i,i] = 1.0
                        A_T[i,i-1] = -1.0
                        b_T[i] = q_tot * dr / self.k_eff(st, -1, j)
                    else:
                        w_sat_i = self.w_sat_coolprop(T_f_old[i, j], cfg.p_a)

                        alpha_w = (2*dr*self.D_eff(cfg,st,i,j)*st.rho_a[i,j] +
                                   (self.D_eff(cfg,st,i+1,j)*st.rho_a[i,j] -
                                    self.D_eff(cfg,st,i-1,j)*st.rho_a[i,j] +
                                    self.D_eff(cfg,st,i,j)*st.rho_a[i+1,j] -
                                    self.D_eff(cfg,st,i,j)*st.rho_a[i-1,j] +
                                    4*self.D_eff(cfg,st,i,j)*st.rho_a[i,j])*r[i])/(4*(dr**2)*r[i])
                        beta_w = - cfg.C*st.rho_a[i,j] - 2*self.D_eff(cfg,st,i,j)*st.rho_a[i,j]/(dr**2)
                        gamma_w = (-2*dr*self.D_eff(cfg,st,i,j)*st.rho_a[i,j] +
                                   (-self.D_eff(cfg,st,i+1,j)*st.rho_a[i,j] +
                                    self.D_eff(cfg,st,i-1,j)*st.rho_a[i,j] -
                                    self.D_eff(cfg,st,i,j)*st.rho_a[i+1,j] +
                                    self.D_eff(cfg,st,i,j)*st.rho_a[i-1,j] +
                                    4*self.D_eff(cfg,st,i,j)*st.rho_a[i,j])*r[i])/(4*(dr**2)*r[i])

                        A_w[i,i+1] = alpha_w
                        A_w[i,i] = beta_w
                        A_w[i,i-1] = gamma_w
                        b_w[i] = -cfg.C*st.rho_a[i,j]*w_sat_i

                        alpha_T = (2*dr*self.k_eff(st,i,j) +
                                   (self.k_eff(st,i+1,j)) -
                                   self.k_eff(st,i-1,j) +
                                   4*self.k_eff(st,i,j)*r[i])/(4*(dr**2)*r[i])
                        beta_T = -2*self.k_eff(st,i,j)/(dr**2)
                        gamma_T = (-2*dr*self.k_eff(st,i,j) +
                                   (-self.k_eff(st,i+1,j)) +
                                   self.k_eff(st,i-1,j) -
                                   4*self.k_eff(st,i,j)*r[i])/(4*(dr**2)*r[i])

                        A_T[i,i+1] = alpha_T
                        A_T[i,i] = beta_T
                        A_T[i,i-1] = gamma_T
                        b_T[i] = -cfg.isv*cfg.C*st.rho_a[i,j]*(w_f_old[i,j]-w_sat_i)

                T_f_new[:, j] = spsolve(csr_matrix(A_T), b_T)
                w_f_new[:,j] = spsolve(csr_matrix(A_w), b_w)

                # To be deleted ----------------------------------------------------------------------------------------
                # ==== Diagnose für θ=j ====
                # Geometrie
                r_start = 0.5 * geom.fin_thickness
                r_end = r_start + float(st.s_e[j])
                r = np.linspace(r_start, r_end, gs.nr)
                dr = r[1] - r[0]

                # Oberfläche
                Tfs = float(T_f_new[-1, j])
                wfs = float(w_f_new[-1, j])
                wfs_sat = float(self.w_sat_coolprop(Tfs, cfg.p_a))

                h = self.h_conv(cfg, geom, j)
                hm = self.h_mass(cfg, geom, j)  # = h/(rho_amb*c_p_a)
                m_f = hm * cfg.rho_amb * (cfg.w_amb - wfs)  # Konvektion von außen
                De_s = float(self.D_eff(cfg, st, -1, j))
                rho_a_s = float(st.rho_a[-1, j])
                gradw = float((w_f_new[-1, j] - w_f_new[-2, j]) / dr)
                m_rho = De_s * rho_a_s * gradw  # Diffusion in den Frost
                m_delta = m_f - m_rho

                q_sens = h * (cfg.T_a - Tfs)
                q_lat = cfg.h_sub * m_delta
                q_tot = q_sens + q_lat

                #print(f"[θ={j}] s_e={st.s_e[j]:.3e} m  dr={dr:.3e} m")
                #print(f"  Tfs={Tfs:.3f} °C  wfs={wfs:.5e}  w_sat(Tfs)={wfs_sat:.5e}")
                #print(f"  De_s={De_s:.3e}  rho_a_s={rho_a_s:.4f}  gradw_fs={gradw:.3e}")
                #print(f"  m_f={m_f:.5e}  m_rho={m_rho:.5e}  m_delta={m_delta:.5e}  [kg/(m²·s)]")
                #print(f"  q_sens={q_sens:.1f}  q_lat={q_lat:.1f}  q_tot={q_tot:.1f}  [W/m²]")

                # Innen: einfache Konsistenzchecks
                #print(f"  Tmin/Tmax={T_f_new[:, j].min():.3f}/{T_f_new[:, j].max():.3f} °C")
                #print(f"  wmin/wmax={w_f_new[:, j].min():.5e}/{w_f_new[:, j].max():.5e}")

                # Volumenquellen (ohne r-Gewichtung; für Zylindersymmetrie optional mit r_i/r_e gewichten)
                w_sat_vec = np.array([self.w_sat_coolprop(T_f_new[i, j], cfg.p_a) for i in range(gs.nr)])
                S_m = cfg.C * st.rho_a[:gs.nr, j] * (
                            w_f_new[:, j] - w_sat_vec)  # [kg_da/(m³·s)]*kg_w/kg_da -> kg_w/(m³·s)
                S_q = - cfg.isv * S_m  # [W/m³]
                M_vol = S_m.sum() * dr
                Q_vol = S_q.sum() * dr
                #print(f"  M_vol≈{M_vol:.5e} kg/(m²·s)   Q_vol≈{Q_vol:.1f} W/m²")
                # To be deleted ----------------------------------------------------------------------------------------

            res_T = np.max(np.abs(T_f_new - T_f_old))
            res_w = np.max(np.abs(w_f_new - w_f_old))
            T_f_old = T_f_new.copy()
            w_f_old = w_f_new.copy()
            it += 1

        st.T_e = T_f_new
        st.w_e = w_f_new

        # calculate s_e and rho_f

        N, ntheta = w_f_new.shape

        for j in range(ntheta):
            for i in range(N):
                w_sat_i = self.w_sat_coolprop(st.T_e[i, j], cfg.p_a)
                source = cfg.C * st.rho_a[i,j] * (st.w_e[i, j] - w_sat_i)
                st.rho_e[i, j] = np.clip(st.rho_e[i, j] + source * cfg.dt, 1, cfg.rho_i)


        for j in range(gs.ntheta):
            rho_fs = st.rho_e[-1, j]
            m_dot_sf = self.m_dot_s_f(cfg, geom, st, gs, j)
            st.s_e[j] += (m_dot_sf / rho_fs) * cfg.dt
            st.s_e[j] = max(st.s_e[j], 1e-6)

        return it, res_T, res_w

    def New_edge_state_seg_diverg_form(self, cfg, geom, st, gs, tol=1e-6, niter=1000):
        it = 0
        rel_T = rel_w = np.inf

        # Arbeitskopien (float) der Felder
        T_f_old = np.asarray(st.T_e, dtype=float).copy()
        w_f_old = np.asarray(st.w_e, dtype=float).copy()
        T_f_new = np.empty_like(T_f_old)
        w_f_new = np.empty_like(w_f_old)

        # feste Relaxationsfaktoren (können optional adaptiv gemacht werden)
        omega_w = 0.6
        omega_T = 0.6

        # Hilfsfunktionen ---------------------------------------------
        def w_sat_clip_vec(TC_arr):
            """Sättigungs-ω(T) vektorisiert, T auf CP-Gültigkeit geclippt."""
            TK = np.asarray(TC_arr, float) + 273.15
            TK = np.clip(TK, 130.0, 372.0)  # sicher unter Siedepunkt ~1.035 bar

            def _one(TK_):
                return HAPropsSI("W", "T", float(TK_), "P", float(cfg.p_a), "R", 1.0)

            return np.vectorize(_one)(TK)

        def hmean(a, b, eps=1e-20):
            """Harmonisches Mittel (stabiler bei starken Gradienten)."""
            return 2.0 * a * b / (a + b + eps)

        # -------------------------------------------------------------

        aitken_states_w = [{'omega': 0.6, 'dprev': None} for _ in range(gs.ntheta)]
        aitken_states_T = [{'omega': 0.6, 'dprev': None} for _ in range(gs.ntheta)]

        def aitken_update(x_old, x_new, state, lo=0.15, hi=0.95):
            d = (x_new - x_old).ravel()
            if state['dprev'] is None:
                omega = state['omega']
            else:
                dd = d - state['dprev']
                denom = float(dd @ dd)
                if denom > 1e-30:
                    omega = state['omega'] * (-(float(d @ dd)) / denom)
                else:
                    omega = state['omega']
            omega = float(np.clip(omega, lo, hi))
            x_relaxed = x_old + omega * (x_new - x_old)
            state['omega'] = omega
            state['dprev'] = d.copy()
            return x_relaxed, omega

        while (it < niter) and ((rel_T > tol) or (rel_w > tol)):
            # Residuen-Basis (vor diesem Swep)
            T_prev = T_f_old.copy()
            w_prev = w_f_old.copy()

            for j in range(gs.ntheta):
                # Geometrie/Radius
                r_start = 0.5 * geom.fin_thickness
                r_end = r_start + float(st.s_e[j])
                r = np.linspace(r_start, r_end, gs.nr)
                N = len(r)
                dr = r[1] - r[0]

                # lokale trockene Luftdichte im Frost (je Iteration aktualisieren)
                st.rho_a[:N, j] = self.rho_a_dry_local(T_f_old[:N, j], w_f_old[:N, j], cfg.p_a)

                # Feldvektoren
                rho_a_vec = st.rho_a[:N, j]
                rho_f_vec = st.rho_e[:N, j]
                Deff_vec = cfg.D_std * (cfg.rho_i - rho_f_vec) / (cfg.rho_i - 0.58 * rho_f_vec)
                k_eff_vec = 0.132 + 3.13e-4 * rho_f_vec + 1.6e-7 * (rho_f_vec ** 2)

                # w_sat(T_alt) für diesen inneren Sweep
                w_sat_vec = w_sat_clip_vec(T_f_old[:N, j])

                # =============== 1) MASSEN-GLEICHUNG (ω) =================
                A_w = lil_matrix((N, N), dtype=float);
                b_w = np.zeros(N)

                Aprop_w = Deff_vec * rho_a_vec
                S_P_w = cfg.C * rho_a_vec
                S_U_w = -cfg.C * rho_a_vec * w_sat_vec

                # Innenknoten (Divergenz-Form, harmonische Flächenwerte)
                for i in range(1, N - 1):
                    r_i = r[i]
                    r_e = 0.5 * (r[i] + r[i + 1]);
                    A_e = hmean(Aprop_w[i], Aprop_w[i + 1])
                    r_wi = 0.5 * (r[i] + r[i - 1]);
                    A_wf = hmean(Aprop_w[i], Aprop_w[i - 1])
                    aE = r_e * A_e / (r_i * dr * dr)
                    aW = r_wi * A_wf / (r_i * dr * dr)
                    aP = -(aW + aE) - S_P_w[i]
                    A_w[i, i + 1] = aE
                    A_w[i, i] = aP
                    A_w[i, i - 1] = aW
                    b_w[i] = S_U_w[i]

                # Randbedingungen ω
                # Wand: Neumann dω/dr=0  -> ω1 - ω0 = 0
                A_w[0, 0] = -1.0;
                A_w[0, 1] = 1.0;
                b_w[0] = 0.0
                # Oberfläche: Dirichlet ωfs = ω_sat(Tfs_alt)
                Tfs_old = float(T_f_old[-1, j])
                A_w[N - 1, N - 1] = 1.0
                b_w[N - 1] = float(w_sat_clip_vec(Tfs_old))

                # Solve ω
                w_f_new[:, j] = spsolve(csr_matrix(A_w), b_w)
                w_f_old[:, j], omega_wj = aitken_update(w_f_old[:, j], w_f_new[:, j], aitken_states_w[j])

                # Surface-Flüsse jetzt mit FRISCHEM ω
                wfs = float(w_f_old[-1, j])
                gradw = (w_f_old[-1, j] - w_f_old[-2, j]) / dr
                h = self.h_conv(cfg, geom, j)
                hm = self.h_mass(cfg, geom, j)  # = h / (rho_amb * c_p)
                m_f = hm * cfg.rho_amb * (cfg.w_amb - wfs)  # (9.11)
                m_rho = float(Deff_vec[-1] * rho_a_vec[-1] * gradw)  # (9.12)
                m_del = m_f - m_rho  # (9.13)

                # =============== 2) ENERGIE-GLEICHUNG (T) ================
                A_T = lil_matrix((N, N), dtype=float);
                b_T = np.zeros(N)

                S_P_T = np.zeros(N)
                # Volumenquelle mit frischem w, aber w_sat(T_alt)
                S_U_T = -cfg.isv * cfg.C * rho_a_vec * (w_f_old[:N, j] - w_sat_vec)

                for i in range(1, N - 1):
                    r_i = r[i]
                    r_e = 0.5 * (r[i] + r[i + 1]);
                    A_e = hmean(k_eff_vec[i], k_eff_vec[i + 1])
                    r_wi = 0.5 * (r[i] + r[i - 1]);
                    A_wf = hmean(k_eff_vec[i], k_eff_vec[i - 1])
                    aE = r_e * A_e / (r_i * dr * dr)
                    aW = r_wi * A_wf / (r_i * dr * dr)
                    aP = -(aW + aE) - S_P_T[i]
                    A_T[i, i + 1] = aE
                    A_T[i, i] = aP
                    A_T[i, i - 1] = aW
                    b_T[i] = S_U_T[i]

                # T-Randbedingungen
                # Wand: Dirichlet T = T_w
                A_T[0, 0] = 1.0;
                b_T[0] = cfg.T_w
                # Oberfläche: k*(T_N - T_{N-1})/dr = q_tot, mit q_tot = h(Ta-Tfs_alt) + h_sub*m_delta
                q_sens = h * (cfg.T_a - Tfs_old)  # (9.9)
                q_tot = q_sens + cfg.h_sub * m_del  # (9.16)
                A_T[N - 1, N - 1] = 1.0
                A_T[N - 1, N - 2] = -1.0
                b_T[N - 1] = q_tot * dr / float(k_eff_vec[-1])

                # Solve T
                T_f_new[:, j] = spsolve(csr_matrix(A_T), b_T)
                T_f_old[:, j], omega_Tj = aitken_update(T_f_old[:, j], T_f_new[:, j], aitken_states_T[j])

            # Residuen (relativ) für den ganzen Sweep
            res_T = np.max(np.abs(T_f_old - T_prev))
            res_w = np.max(np.abs(w_f_old - w_prev))
            rng_T = max(1e-6, T_prev.max() - T_prev.min())
            rng_w = max(1e-12, w_prev.max() - w_prev.min())
            rel_T = res_T / rng_T
            rel_w = res_w / rng_w

            it += 1

        # Übernehme die konvergierten Felder in den Zustand
        st.T_e = T_f_old
        st.w_e = w_f_old

        # ---------- explizite Updates: ρ_f und s_e ----------
        N, ntheta = st.w_e.shape

        # ρ_f-Update (mit geclipptem w_sat, vektorisiert & begrenzt)
        for j in range(ntheta):
            w_sat_all = w_sat_clip_vec(st.T_e[:, j])
            source = cfg.C * st.rho_a[:N, j] * (st.w_e[:, j] - w_sat_all)
            # optional einfacher Limiter auf Δρ_f:
            # source *= 0.5
            st.rho_e[:N, j] = np.clip(st.rho_e[:N, j] + source * cfg.dt, 1.0, cfg.rho_i)

        # s_e-Update (Dickenwachstum) – nutzt deine vorhandene Funktion m_dot_s_f
        for j in range(gs.ntheta):
            rho_fs = st.rho_e[-1, j]
            m_dot_sf = self.m_dot_s_f(cfg, geom, st, gs, j)
            delta_s = (m_dot_sf / rho_fs) * cfg.dt
            st.s_e[j] += delta_s
            st.s_e[j] = max(st.s_e[j], 1e-4)  # numerischer Mindestwert (stabilere dr)

        return it, res_T, res_w
