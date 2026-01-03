import numpy as np
import math
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from Framework.core.corrolations import DK
from CoolProp.HumidAirProp import HAPropsSI, HAProps_Aux

class Frostmodell_Edge:

    @staticmethod
    def w_sat_coolprop(Tf_C, p_Pa):
        """
        Tf_C: float oder array-like (°C)
        p_Pa: float (Pa)
        Returns: float oder np.ndarray
        """
        Tf_C_arr = np.asarray(Tf_C)

        # scalar fast-path
        if Tf_C_arr.ndim == 0:
            Tf_K = float(Tf_C_arr) + 273.15
            p_ws, _units = HAProps_Aux("p_ws", Tf_K, float(p_Pa), 0.0)
            return 0.621945 * p_ws / (float(p_Pa) - p_ws)

        # vector path: loop scalars (CoolProp call is scalar-only)
        out = np.empty_like(Tf_C_arr, dtype=float)
        p = float(p_Pa)
        for i, tC in np.ndenumerate(Tf_C_arr):
            Tf_K = float(tC) + 273.15
            p_ws, _units = HAProps_Aux("p_ws", Tf_K, p, 0.0)
            out[i] = 0.621945 * p_ws / (p - p_ws)
        return out
        #return HAPropsSI("W", "T", Tf_K, "P", p_Pa, "R", 1.0)

    @staticmethod
    def Nu_edge(cfg, geom, theta):
        Re_d = DK.Re(cfg.v_a, geom.fin_pitch_cc, cfg.v_kin)
        Pr = DK.Pr(cfg.v_kin, cfg.lam, cfg.c_p_a, cfg.rho_amb)
        return 0.23 * (Re_d**0.466) * (Pr**(1/3)) * (0.7 + 1.06e-4 * (theta - 90)**2)

    def h_conv(self, cfg, geom, theta):
        Nu = self.Nu_edge(cfg, geom, theta)
        return Nu * cfg.lam / geom.fin_pitch_cc

    def q_dot_sens_fs(self, cfg, geom, st, theta):
        T_fs = st.T_e[-1, theta]
        return self.h_conv(cfg, geom, theta) * (cfg.T_a - T_fs)

    def h_mass(self, cfg, geom, st, theta):
        h = self.h_conv(cfg, geom, theta)
        return h / (st.rho_a_e[-1,theta] * cfg.c_p_a)

    def m_dot_f(self, cfg, geom, st, theta):
        hm = self.h_mass(cfg, geom, st, theta)
        w_fs = st.w_e[-1, theta]
        return hm * st.rho_a_e[-1,theta] * (cfg.w_amb - w_fs)

    def m_dot_rho_f(self, cfg, geom, st, gs, theta):
        Deff = self.D_eff(cfg, st, -1, theta)
        dr = (0.5*geom.fin_thickness + st.s_e[theta] - 0.5*geom.fin_thickness) / (gs.nr - 1)
        dwf_dr = (st.w_e[-1, theta] - st.w_e[-2, theta]) / dr
        return Deff * st.rho_a_e[-1,theta] * dwf_dr

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
    def rho_a_dry_local(Tf_C, p_Pa):
        Tf_K = np.asarray(Tf_C) + 273.15
        R = 287.058
        return p_Pa / (R*Tf_K)

    def New_edge_state_seg_at_90(self, cfg, geom, st, gs, tol=1e-6, niter=1000):
        it = 0
        res_T = res_w = np.inf
        j = gs.ntheta-1

        # Arbeitskopien
        T_f_old = np.asarray(st.T_e[:,j], dtype=float).copy()
        w_f_old = np.asarray(st.w_e[:,j], dtype=float).copy()
        T_f_new = np.empty_like(T_f_old)
        w_f_new = np.empty_like(w_f_old)

        while (it < niter) and ((res_T > tol) or (res_w > tol)):
            # Radialgitter für den Winkel 90°
            r_start = 0.5 * geom.fin_thickness
            r_end = r_start + float(st.s_e[j])
            r = np.linspace(r_start, r_end, gs.nr)
            N = len(r)
            dr = r[1] - r[0]

            # lokale trockene Luftdichte im Frost
            st.rho_a_e[:N, j] = self.rho_a_dry_local(T_f_old[:N], cfg.p_a)
            rho_a = st.rho_a_e[:N, j]

            # Randwerte an der Oberfläche
            Tfs = T_f_old[-1]
            wfs_sat = self.w_sat_coolprop(Tfs, cfg.p_a)

            # konvektiver Massenübergang
            rho_a_fs = rho_a[-1]
            h = self.h_conv(cfg, geom, j)
            hm = h / (rho_a_fs * cfg.c_p_a)
            m_f = hm * rho_a_fs * (cfg.w_amb - wfs_sat)  # (9.11)

            # diffusive Dampfmasse im Frost am Interface
            De_s = self.D_eff(cfg, st, -1, j)
            grad_w = (w_f_old[-1] - w_f_old[-2]) / dr  # dw/dr nach außen
            m_rho = De_s * rho_a_fs * grad_w  # (9.12)

            m_delta = m_f - m_rho  # (9.13)

            # Wärmeströme an der Oberfläche
            q_sens = h * (cfg.T_a - Tfs)  # (9.9)
            q_tot = q_sens + cfg.h_sub * m_delta  # (9.16)

            # Vectorized ---------------------------------------------------------------
            # --- RHS ---
            b_w = np.zeros(N, dtype=float)
            b_T = np.zeros(N, dtype=float)

            # --- diagonals ---
            lower_w = np.zeros(N - 1, dtype=float)  # A[i, i-1], i=1..N-1  -> lower_w[i-1]
            main_w = np.zeros(N, dtype=float)  # A[i, i]
            upper_w = np.zeros(N - 1, dtype=float)  # A[i, i+1], i=0..N-2  -> upper_w[i]

            lower_T = np.zeros(N - 1, dtype=float)
            main_T = np.zeros(N, dtype=float)
            upper_T = np.zeros(N - 1, dtype=float)

            # =========================
            # Boundary conditions
            # =========================

            # i = 0
            # w: Neumann dw/dr = 0 -> w1 - w0 = 0
            main_w[0] = -1.0
            upper_w[0] = 1.0
            b_w[0] = 0.0

            # T: Dirichlet T = T_w
            main_T[0] = 1.0
            b_T[0] = cfg.T_tube

            # i = N-1
            # w: Dirichlet w_fs = wfs_sat
            main_w[-1] = 1.0
            b_w[-1] = wfs_sat
            lower_w[-1] = 0.0  # last row must not couple to w_{N-2}

            # T: Neumann at surface: k (T_fs - T_{N-1})/dr = q_tot
            k_s = self.k_eff(st, -1, j)
            main_T[-1] = k_s / dr
            lower_T[-1] = -k_s / dr
            b_T[-1] = q_tot

            # =========================
            # Interior (vectorized)
            # =========================
            idx = np.arange(1, N - 1)  # 1..N-2
            ri = r[idx]
            rho = rho_a[idx]

            inv_dr2 = 1.0 / (dr * dr)
            inv_2rdr = 1.0 / (2.0 * ri * dr)

            # ---- w equation ----
            Deff = self.D_eff(cfg, st, idx, j)  # if vectorized

            Aprop = Deff * rho

            alpha_w = Aprop * (inv_dr2 + inv_2rdr)
            beta_w = -2.0 * Aprop * inv_dr2 - cfg.C * rho
            gamma_w = Aprop * (inv_dr2 - inv_2rdr)

            upper_w[idx] = alpha_w  # row i, col i+1
            main_w[idx] = beta_w
            lower_w[idx - 1] = gamma_w  # row i, col i-1 stored at i-1

            T_old_vec = T_f_old[idx]
            w_sat_i = self.w_sat_coolprop(T_old_vec, cfg.p_a)

            b_w[idx] = -cfg.C * rho * w_sat_i

            # ---- T equation ----
            k_i = self.k_eff(st, idx, j)

            alpha_T = k_i * (inv_dr2 + inv_2rdr)
            beta_T = -2.0 * k_i * inv_dr2
            gamma_T = k_i * (inv_dr2 - inv_2rdr)

            upper_T[idx] = alpha_T
            main_T[idx] = beta_T
            lower_T[idx - 1] = gamma_T

            b_T[idx] = -cfg.isv * cfg.C * rho * (w_f_old[idx] - w_sat_i)

            # =========================
            # Build CSR matrices from diagonals
            # =========================
            rows_main = np.arange(N)
            cols_main = rows_main

            rows_upper = np.arange(N - 1)
            cols_upper = rows_upper + 1

            rows_lower = np.arange(1, N)
            cols_lower = rows_lower - 1

            # w-matrix
            data_w = np.concatenate([main_w, upper_w, lower_w])
            rows_w = np.concatenate([rows_main, rows_upper, rows_lower])
            cols_w = np.concatenate([cols_main, cols_upper, cols_lower])
            A_w = csr_matrix((data_w, (rows_w, cols_w)), shape=(N, N))

            # T-matrix
            data_T = np.concatenate([main_T, upper_T, lower_T])
            rows_T = np.concatenate([rows_main, rows_upper, rows_lower])
            cols_T = np.concatenate([cols_main, cols_upper, cols_lower])
            A_T = csr_matrix((data_T, (rows_T, cols_T)), shape=(N, N))

            #for i in range(N):
            #    if i == 0:
            #        # Wand BC
            #        # w: Neumann dw/dr = 0  -> w1 - w0 = 0
            #        A_w[i, i] = -1.0
            #        A_w[i, i+1] = 1.0
            #        b_w[i] = 0.0
            #
            #        # T: Dirichlet T = T_w
            #        A_T[i, i] = 1.0
            #        b_T[i] = cfg.T_tube
            #
            #    elif i == N - 1:
            #        # Oberfläche w: Dirichlet w_fs = w_sat(T_fs)
            #        A_w[i, i] = 1.0
            #        b_w[i] = wfs_sat
            #
            #        # Oberfläche T: k (T_fs - T_{N-1}) / dr = q_tot
            #        k_s = self.k_eff(st, -1, j)
            #        A_T[i, i] = k_s / dr
            #        A_T[i, i - 1] = -k_s / dr
            #        b_T[i] = q_tot
            #
            #    else:
            #        r_i = r[i]
            #        rho_ij = rho_a[i]
            #
            #        # ---- Massen-Gleichung (w) ----
            #        Deff_ij = self.D_eff(cfg, st, i, j)
            #        Aprop = Deff_ij * rho_ij  # "D_eff * rho_a" am Knoten i
            #
            #        alpha_w = Aprop * (1.0 / dr ** 2 + 1.0 / (2.0 * r_i * dr))
            #        beta_w = -2.0 * Aprop / dr ** 2 - cfg.C * rho_ij
            #        gamma_w = Aprop * (1.0 / dr ** 2 - 1.0 / (2.0 * r_i * dr))
            #
            #        A_w[i, i + 1] = alpha_w
            #        A_w[i, i] = beta_w
            #        A_w[i, i - 1] = gamma_w
            #
            #        w_sat_i = self.w_sat_coolprop(T_f_old[i], cfg.p_a)
            #        b_w[i] = -cfg.C * rho_ij * w_sat_i
            #
            #        # ---- Energie-Gleichung (T) ----
            #        k_i = self.k_eff(st, i, j)
            #
            #        alpha_T = k_i * (1.0 / dr ** 2 + 1.0 / (2.0 * r_i * dr))
            #        beta_T = -2.0 * k_i / dr ** 2
            #        gamma_T = k_i * (1.0 / dr ** 2 - 1.0 / (2.0 * r_i * dr))
            #
            #        A_T[i, i + 1] = alpha_T
            #        A_T[i, i] = beta_T
            #        A_T[i, i - 1] = gamma_T
            #
            #        b_T[i] = -cfg.isv * cfg.C * rho_ij * (w_f_old[i] - w_sat_i)

            # lineare Systeme lösen
            T_f_new[:] = spsolve(A_T, b_T)
            w_f_new[:] = spsolve(A_w, b_w)

            # Konvergenzkriterium
            res_T = np.max(np.abs(T_f_new - T_f_old))
            res_w = np.max(np.abs(w_f_new - w_f_old))

            # Unterrelaxation aus Stabilitätsgründen

            omega_T = 0.1
            omega_w = 1.0

            T_f_old[:] = (1 - omega_T) * T_f_old + omega_T * T_f_new
            w_f_old[:] = (1 - omega_w) * w_f_old + omega_w * w_f_new

            it += 1

        # konvergierte Felder in den Zustand zurückschreiben
        st.T_e[:,j] = T_f_new
        st.w_e[:,j] = w_f_new

        # rho_a updaten
        st.rho_a_e[:gs.nr, j] = self.rho_a_dry_local(T_f_new[:gs.nr], cfg.p_a)

        # --------- Explizites Update von rho_e und s_e ---------

        # rho_e-Update
        for i in range(gs.nr):
            st.rho_e[i, j] = 207*np.exp(0.266*st.T_e[-1,j] - 0.0615*cfg.T_tube)

        # s_e-Update
        rho_fs = st.rho_e[-1, j]
        m_dot_sf = self.m_dot_s_f(cfg, geom, st, gs, j)
        st.s_e[j] += (m_dot_sf / rho_fs) * gs.dt
        st.s_e[j] = max(st.s_e[j], 1e-6)

        return it, res_T, res_w

    def New_edge_state_seg(self, cfg, geom, st, gs, tol=1e-6, niter=1000):
        it = 0
        res_T = res_w = np.inf

        # Arbeitskopien
        T_f_old = np.asarray(st.T_e, dtype=float).copy()
        w_f_old = np.asarray(st.w_e, dtype=float).copy()
        T_f_new = np.empty_like(T_f_old)
        w_f_new = np.empty_like(w_f_old)

        while (it < niter) and ((res_T > tol) or (res_w > tol)):
            for j in range(gs.ntheta):
                # Radialgitter für diesen Winkel
                r_start = 0.5 * geom.fin_thickness
                r_end = r_start + float(st.s_e[j])
                r = np.linspace(r_start, r_end, gs.nr)
                N = len(r)
                dr = r[1] - r[0]

                # lokale trockene Luftdichte im Frost
                st.rho_a_e[:N, j] = self.rho_a_dry_local(T_f_old[:N, j], cfg.p_a)
                rho_a = st.rho_a_e[:N, j]

                # Randwerte an der Oberfläche
                Tfs = T_f_old[-1, j]
                wfs_sat = self.w_sat_coolprop(Tfs, cfg.p_a)

                # konvektiver Massenübergang
                rho_a_fs = st.rho_a_e[-1,j]
                h = self.h_conv(cfg, geom, j)
                hm = h / (rho_a_fs * cfg.c_p_a)
                m_f = hm * rho_a_fs * (cfg.w_amb - wfs_sat)  # (9.11)

                if m_f >= 0.0:
                # diffusive Dampfmasse im Frost am Interface
                    De_s = self.D_eff(cfg, st, -1, j)
                    grad_w = (w_f_old[-1, j] - w_f_old[-2, j]) / dr  # dw/dr nach außen
                    m_rho = De_s * rho_a[-1] * grad_w  # (9.12)
                    m_delta = m_f - m_rho  # (9.13)
                    check_m_delta = False
                else:
                    De_s = self.D_eff(cfg, st, -1, j)
                    grad_w = (w_f_old[-1, j] - w_f_old[-2, j]) / dr  # dw/dr nach außen
                    m_rho = De_s * rho_a[-1] * grad_w  # (9.12)
                    m_delta = 0.0
                    check_m_delta = True

                # Wärmeströme an der Oberfläche
                q_sens = h * (cfg.T_a - Tfs)  # (9.9)
                q_tot = q_sens + cfg.h_sub * m_delta  # (9.16)

                # Systemmatrizen für w und T
                A_w = lil_matrix((N, N), dtype=float)
                b_w = np.zeros(N)
                A_T = lil_matrix((N, N), dtype=float)
                b_T = np.zeros(N)

                for i in range(N):
                    if i == 0:
                        # Wand BC
                        # w: Neumann dw/dr = 0  -> w1 - w0 = 0
                        A_w[i, i] = -1.0
                        A_w[i, i+1] = 1.0
                        b_w[i] = 0.0

                        # T: Dirichlet T = T_w
                        A_T[i, i] = 1.0
                        b_T[i] = cfg.T_tube

                    elif i == N - 1:
                        # Oberfläche w: Dirichlet w_fs = w_sat(T_fs)
                        A_w[i, i] = 1.0
                        b_w[i] = wfs_sat

                        # Oberfläche T: k (T_fs - T_{N-1}) / dr = q_tot
                        k_s = self.k_eff(st, -1, j)
                        A_T[i, i] = k_s / dr
                        A_T[i, i - 1] = -k_s / dr
                        b_T[i] = q_tot

                    else:
                        r_i = r[i]
                        rho_ij = rho_a[i]

                        # ---- Massen-Gleichung (w) ----
                        Deff_ij = self.D_eff(cfg, st, i, j)
                        Aprop = Deff_ij * rho_ij  # "D_eff * rho_a" am Knoten i

                        alpha_w = Aprop * (1.0 / dr ** 2 + 1.0 / (2.0 * r_i * dr))
                        beta_w = -2.0 * Aprop / dr ** 2 - cfg.C * rho_ij
                        gamma_w = Aprop * (1.0 / dr ** 2 - 1.0 / (2.0 * r_i * dr))

                        A_w[i, i + 1] = alpha_w
                        A_w[i, i] = beta_w
                        A_w[i, i - 1] = gamma_w

                        w_sat_i = self.w_sat_coolprop(T_f_old[i, j], cfg.p_a)
                        b_w[i] = -cfg.C * rho_ij * w_sat_i

                        # ---- Energie-Gleichung (T) ----
                        k_i = self.k_eff(st, i, j)

                        alpha_T = k_i * (1.0 / dr ** 2 + 1.0 / (2.0 * r_i * dr))
                        beta_T = -2.0 * k_i / dr ** 2
                        gamma_T = k_i * (1.0 / dr ** 2 - 1.0 / (2.0 * r_i * dr))

                        A_T[i, i + 1] = alpha_T
                        A_T[i, i] = beta_T
                        A_T[i, i - 1] = gamma_T

                        b_T[i] = -cfg.isv * cfg.C * rho_ij * (w_f_old[i, j] - w_sat_i)

                # lineare Systeme lösen
                T_f_new[:, j] = spsolve(csr_matrix(A_T), b_T)
                w_f_new[:, j] = spsolve(csr_matrix(A_w), b_w)

            # Konvergenzkriterium
            res_T = np.max(np.abs(T_f_new - T_f_old))
            res_w = np.max(np.abs(w_f_new - w_f_old))

            # Unterrelaxation aus Stabilitätsgründen

            omega_T = 0.1 # save stabil: 0.1
            omega_w = 1.0

            T_f_old[:] = (1 - omega_T) * T_f_old + omega_T * T_f_new
            w_f_old[:] = (1 - omega_w) * w_f_old + omega_w * w_f_new

            it += 1

        # konvergierte Felder in den Zustand zurückschreiben
        st.T_e = T_f_new
        st.w_e = w_f_new

        # rho_a updaten
        for j in  range(gs.ntheta):
            st.rho_a_e[:gs.nr, j] = self.rho_a_dry_local(T_f_new[:gs.nr, j], cfg.p_a)

        if check_m_delta:
            print(f"\033[31mNegative moisture mass flow detected, setting m_delta = 0!\033[0m")

        # --------- Explizites Update von rho_e und s_e ---------

        N, ntheta = w_f_new.shape

        # rho_e-Update
        for j in range(ntheta):
            for i in range(N):
                st.rho_e[i, j] = 207*np.exp(0.266*st.T_e[-1,j] - 0.0615*cfg.T_tube)

        # s_e-Update
        for j in range(ntheta):
            rho_fs = st.rho_e[-1, j]
            m_dot_sf = self.m_dot_s_f(cfg, geom, st, gs, j)
            st.s_e[j] += (m_dot_sf / rho_fs) * gs.dt
            st.s_e[j] = max(st.s_e[j], 1e-6)

        return it, res_T, res_w

class Frostmodell_Finn_and_Tube:

    @staticmethod
    def w_sat_coolprop(Tf_C, p_Pa):
        """
        Tf_C: float oder array-like (°C)
        p_Pa: float (Pa)
        Returns: float oder np.ndarray
        """
        Tf_C_arr = np.asarray(Tf_C)

        # scalar fast-path
        if Tf_C_arr.ndim == 0:
            Tf_K = float(Tf_C_arr) + 273.15
            p_ws, _units = HAProps_Aux("p_ws", Tf_K, float(p_Pa), 0.0)
            return 0.621945 * p_ws / (float(p_Pa) - p_ws)

        # vector path: loop scalars (CoolProp call is scalar-only)
        out = np.empty_like(Tf_C_arr, dtype=float)
        p = float(p_Pa)
        for i, tC in np.ndenumerate(Tf_C_arr):
            Tf_K = float(tC) + 273.15
            p_ws, _units = HAProps_Aux("p_ws", Tf_K, p, 0.0)
            out[i] = 0.621945 * p_ws / (p - p_ws)
        return out
        #return HAPropsSI("W", "T", Tf_K, "P", p_Pa, "R", 1.0)

    @staticmethod
    def rho_a_dry_local(Tf_C, p_Pa):
        Tf_K = np.asarray(Tf_C) + 273.15
        R = 287.058
        return p_Pa / (R * Tf_K)

    @staticmethod
    def D_eff(cfg, st, i):
        rho_f = st.rho_ft[i]
        numerator = cfg.D_std * (cfg.rho_i - rho_f)
        denominator = cfg.rho_i - 0.58 * rho_f
        return numerator / denominator

    @staticmethod
    def k_f(st, i):
        rho_f = st.rho_ft[i]
        return 0.132 + 3.13e-4 * rho_f + 1.6e-7 * (rho_f ** 2)

    def alpha_tube(self, cfg, geom):
        l = np.pi * geom.d_tube_a / 2.0
        Re = DK.Re(cfg.v_a, l, cfg.v_kin)
        Pr = 0.7
        Nu_lam = 0.664 * np.sqrt(Re) * Pr ** (1 / 3)
        Nu_turb = (0.037 * (Re ** 0.8) * Pr) / (1 + 2.443 * (Re ** -0.1) * (Pr ** (2 / 3) - 1))
        Nu = 0.3 + np.sqrt(Nu_lam ** 2 + Nu_turb ** 2)
        alpha = Nu * cfg.lam / l
        return alpha

    def h_eff(self, cfg, geom):
        h_0 = self.alpha_tube(cfg,geom)
        mue_fin = geom.mue_fin(h_0)
        A_G = geom.A_tube_one_segment()
        A_R = geom.A_fin_one_segment()
        A = geom.A_one_segment()
        h_eff = h_0 * (A_G/A + mue_fin*A_R/A)
        return h_eff

    def New_finn_and_tube_state_seg(self, cfg, geom, st, gs, tol=1e-6, niter=1000):
        it = 0
        res_T = res_w = np.inf

        # Arbeitskopien
        T_f_old = np.asarray(st.T_ft, dtype=float).copy()
        w_f_old = np.asarray(st.w_ft, dtype=float).copy()
        T_f_new = np.empty_like(T_f_old)
        w_f_new = np.empty_like(w_f_old)

        N = gs.nx

        # aktuelles Frostgitter in x-Richtung
        delta_f = max(float(st.s_ft), 1e-6)
        x = np.linspace(0.0, delta_f, N)
        dx = x[1] - x[0]

        # effektiver Luft-Wärmeübergang
        h_eff = self.h_eff(cfg, geom)


        while (it < niter) and ((res_T > tol) or (res_w > tol)):

            # lokale trockene Luftdichte im Frost
            st.rho_a_ft[:] = self.rho_a_dry_local(T_f_old, cfg.p_a)
            rho_a = st.rho_a_ft

            # Systemmatrizen
            #A_w = lil_matrix((N, N), dtype=float)
            #b_w = np.zeros(N)
            #A_T = lil_matrix((N, N), dtype=float)
            #b_T = np.zeros(N)

            # Temperatur- und Sättigungszustand an der Oberfläche
            Tfs = float(T_f_old[-1])
            wfs_sat = self.w_sat_coolprop(Tfs, cfg.p_a)

            # Massenströme (Luftseite + diffusive im Frost)
            rho_a_sf = self.rho_a_dry_local(Tfs, cfg.p_a)
            hm_eff = h_eff / (rho_a_sf * cfg.c_p_a)  # Massenübergangskoeffizient

            dw = cfg.w_amb - wfs_sat
            if dw >= 0.0:
                m_f = hm_eff * rho_a_sf * dw
                Deff_s = self.D_eff(cfg, st, N - 1)
                grad_w = (w_f_old[-1] - w_f_old[-2]) / dx
                m_rho = Deff_s * rho_a[-1] * grad_w
                m_delta = m_f - m_rho
                check_m_delta = False
            else:
                m_f = hm_eff * rho_a_sf * dw
                Deff_s = self.D_eff(cfg, st, N - 1)
                grad_w = (w_f_old[-1] - w_f_old[-2]) / dx
                m_rho = Deff_s * rho_a[-1] * grad_w
                m_delta = 0.0
                check_m_delta = True

            # Wärmeströme
            q_sens_fs = h_eff * (cfg.T_a - Tfs)
            q_lat_fs = cfg.h_sub * m_delta
            q_tot_fs = q_sens_fs + q_lat_fs

            # Temperatur an der Frosoberfläche
            h_0 = self.alpha_tube(cfg,geom)
            mue_fin = geom.mue_fin(h_0)
            #T_s_fs = cfg.T_a - (mue_fin*geom.A_fin_one_segment())*(cfg.T_a-cfg.T_tube)/(geom.A_one_segment())

            # --- RHS ---
            b_w = np.zeros(N, dtype=float)
            b_T = np.zeros(N, dtype=float)

            # --- diagonals ---
            lower_w = np.zeros(N - 1, dtype=float)  # A[i, i-1] stored at lower_w[i-1]
            main_w = np.zeros(N, dtype=float)  # A[i, i]
            upper_w = np.zeros(N - 1, dtype=float)  # A[i, i+1] stored at upper_w[i]

            lower_T = np.zeros(N - 1, dtype=float)
            main_T = np.zeros(N, dtype=float)
            upper_T = np.zeros(N - 1, dtype=float)

            inv_dx2 = 1.0 / (dx * dx)

            # =========================
            # Boundary conditions
            # =========================

            # i = 0
            # w: Neumann dw/dx = 0 -> w1 - w0 = 0
            main_w[0] = -1.0
            upper_w[0] = 1.0
            b_w[0] = 0.0

            # T: Dirichlet T = T_tube
            main_T[0] = 1.0
            b_T[0] = cfg.T_tube

            # i = N-1
            # w: Dirichlet w_fs = wfs_sat
            main_w[-1] = 1.0
            b_w[-1] = wfs_sat
            lower_w[-1] = 0.0  # last row must not couple to w_{N-2}

            # T: Neumann at surface: k (T_fs - T_{N-1})/dx = q_tot_fs
            k_eff_s = self.k_f(st, N - 1)
            main_T[-1] = k_eff_s / dx
            lower_T[-1] = -k_eff_s / dx
            b_T[-1] = q_tot_fs

            # =========================
            # Interior (vectorized)
            # =========================
            idx = np.arange(1, N - 1)  # 1..N-2
            rho = rho_a[idx]

            # Deff, k_eff vectors
            Deff = self.D_eff(cfg, st, idx)  # if vectorized

            k_eff = self.k_f(st, idx)  # if vectorized

            Aprop = Deff * rho  # Deff_i * rho_a_i

            alpha_w = Aprop * inv_dx2
            beta_w = -2.0 * Aprop * inv_dx2 - cfg.C * rho
            gamma_w = Aprop * inv_dx2  # same as alpha_w here

            upper_w[idx] = alpha_w
            main_w[idx] = beta_w
            lower_w[idx - 1] = gamma_w

            # saturation w at interior nodes
            T_old_vec = T_f_old[idx]
            w_sat_i = self.w_sat_coolprop(T_old_vec, cfg.p_a)

            b_w[idx] = -cfg.C * rho * w_sat_i

            # ---- T equation ----
            alpha_T = k_eff * inv_dx2
            beta_T = -2.0 * k_eff * inv_dx2
            gamma_T = k_eff * inv_dx2

            upper_T[idx] = alpha_T
            main_T[idx] = beta_T
            lower_T[idx - 1] = gamma_T

            b_T[idx] = -cfg.isv * cfg.C * rho * (w_f_old[idx] - w_sat_i)

            # =========================
            # Build CSR matrices (no new imports) + solve
            # =========================
            rows_main = np.arange(N)
            cols_main = rows_main

            rows_upper = np.arange(N - 1)
            cols_upper = rows_upper + 1

            rows_lower = np.arange(1, N)
            cols_lower = rows_lower - 1

            data_w = np.concatenate([main_w, upper_w, lower_w])
            rows_w = np.concatenate([rows_main, rows_upper, rows_lower])
            cols_w = np.concatenate([cols_main, cols_upper, cols_lower])
            A_w = csr_matrix((data_w, (rows_w, cols_w)), shape=(N, N))

            data_T = np.concatenate([main_T, upper_T, lower_T])
            rows_T = np.concatenate([rows_main, rows_upper, rows_lower])
            cols_T = np.concatenate([cols_main, cols_upper, cols_lower])
            A_T = csr_matrix((data_T, (rows_T, cols_T)), shape=(N, N))

            #for i in range(N):
            #    if i == 0:
            #        # x = 0: kalte Wand/Tube -> Dirichlet T = T_s_fs
            #        # w: Neumann dw/dx = 0 -> w1 - w0 = 0
            #        A_w[i, i] = -1.0
            #        A_w[i, i+1] = 1.0
            #        b_w[i] = 0.0
            #
            #        A_T[i, i] = 1.0
            #        b_T[i] = cfg.T_tube
            #
            #    elif i == N - 1:
            #        # x = δ_f: Frostoberfläche zur Luft
            #        # w: Dirichlet-Bedingung
            #        A_w[i, i] = 1.0
            #        b_w[i] = wfs_sat
            #
            #        k_eff_i = self.k_f(st, i)
            #        A_T[i, i] = k_eff_i / dx
            #        A_T[i, i-1] = -k_eff_i / dx
            #        b_T[i] = q_tot_fs
            #
            #    else:
            #        # innerer Knoten 0 < i < N-1
            #        rho_a_i = rho_a[i]
            #        Deff_i = self.D_eff(cfg, st, i)
            #        k_eff_i = self.k_f(st, i)
            #
            #        # ---- Massen-Gleichung (w) ----
            #        alpha_w = Deff_i * rho_a_i / (dx ** 2)
            #        gamma_w = Deff_i * rho_a_i / (dx ** 2)
            #        beta_w = -2.0 * Deff_i * rho_a_i / (dx ** 2) - cfg.C * rho_a_i
            #
            #        A_w[i, i + 1] = alpha_w
            #        A_w[i, i] = beta_w
            #        A_w[i, i - 1] = gamma_w
            #
            #        w_sat_i = self.w_sat_coolprop(T_f_old[i], cfg.p_a)
            #        b_w[i] = -cfg.C * rho_a_i * w_sat_i
            #
            #        # ---- Energie-Gleichung (T) ----
            #        alpha_T = k_eff_i / (dx ** 2)
            #        gamma_T = k_eff_i / (dx ** 2)
            #        beta_T = -2.0 * k_eff_i / (dx ** 2)
            #
            #        A_T[i, i + 1] = alpha_T
            #        A_T[i, i] = beta_T
            #        A_T[i, i - 1] = gamma_T
            #
            #        b_T[i] = -cfg.isv * cfg.C * rho_a_i * (w_f_old[i] - w_sat_i)

            # lineare Systeme lösen
            T_f_new[:] = spsolve(A_T, b_T)
            w_f_new[:] = spsolve(A_w, b_w)

            # Konvergenzbewertung
            res_T = np.max(np.abs(T_f_new - T_f_old))
            res_w = np.max(np.abs(w_f_new - w_f_old))

            # Unterrelaxation wie beim Edge-Modell
            omega_T = 0.1 # save stabil: 0.1
            omega_w = 1.0

            T_f_old[:] = (1 - omega_T) * T_f_old + omega_T * T_f_new
            w_f_old[:] = (1 - omega_w) * w_f_old + omega_w * w_f_new

            it += 1

        # Wärmestrohm berechnen
        Q_sens = q_sens_fs*geom.A_one_segment()
        Q_tot = Q_sens + cfg.h_sub*m_delta*geom.A_one_segment()


        # konvergierte Felder zurückschreiben
        st.T_ft = T_f_new.copy()
        st.w_ft = w_f_new.copy()

        if check_m_delta:
            print(f"\033[31mNegative moisture mass flow detected, setting m_delta = 0!\033[0m")

        for i in range(N):
            st.rho_ft[i] = 207*np.exp(0.266*st.T_ft[-1] - 0.0615*cfg.T_tube)

        # Dickenwachstum an der Oberfläche
        Tfs = st.T_ft[-1]
        wfs_sat = self.w_sat_coolprop(Tfs, cfg.p_a)
        rho_a_s = self.rho_a_dry_local(Tfs, cfg.p_a)
        Deff_s = self.D_eff(cfg, st, N - 1)
        grad_w_s = (st.w_ft[-1] - st.w_ft[-2]) / dx
        m_rho_s = Deff_s * rho_a_s * grad_w_s
        m_f_s = hm_eff * st.rho_a_ft[-1] * (cfg.w_amb - wfs_sat)
        m_delta_s = m_f_s - m_rho_s

        rho_fs = st.rho_ft[-1]
        st.s_ft += (m_delta_s / rho_fs) * gs.dt
        st.s_ft = max(st.s_ft, 1e-6)

        return it, res_T, res_w

    def _segment_surface_fluxes(self, cfg, geom, st, gs):
        """
        Berechnet die flächenspezifischen Flüsse an der Frostofffläche
        der Finn-&-Tube-Domäne dieses Segments.

        Rückgabe:
            q_tot_fs  [W/m²]       - gesamter Wärmestrom von Luft -> Frost
            m_delta   [kg/(m² s)]  - Nettomassenstrom Wasserdampf -> Frost
        """
        N = gs.nx
        if N < 2:
            return 0.0, 0.0

        # effektiver Luft-Wärmeübergang
        h_eff = self.h_eff(cfg, geom)


        # aktuelles Frostgitter in x-Richtung
        delta_f = max(float(st.s_ft), 1e-6)
        dx = delta_f / (N - 1)

        # Werte an der Frostoberfläche
        Tfs = float(st.T_ft[-1])
        wfs = float(st.w_ft[-1])

        # Sättigungszustand + Gradienten im Frost
        rho_a_s = self.rho_a_dry_local(Tfs, cfg.p_a)
        Deff_s = self.D_eff(cfg, st, N - 1)
        grad_w = (st.w_ft[-1] - st.w_ft[-2]) / dx

        # Massenflüsse (Luftseite + diffusive im Frost)
        hm_eff = h_eff / (rho_a_s * cfg.c_p_a)

        dw = cfg.w_amb - wfs
        if dw >= 0.0:
            m_fs = hm_eff * rho_a_s * dw  # [kg/(m² s)]
            m_rho = Deff_s * rho_a_s * grad_w  # [kg/(m² s)]
            m_delta = m_fs - m_rho  # [kg/(m² s)]
        else:
            m_delta = 0.0
            #print(f"\033[31mNegative moisture mass flow detected, setting m_delta = 0!\033[0m")

        m_x0 = cfg.C * st.rho_ft[0]*(st.w_ft[0]-self.w_sat_coolprop(st.T_ft[0], cfg.p_a))*dx

        # Wärmeströme
        q_sens_fs = h_eff * (cfg.T_a - Tfs)  # [W/m²]
        q_tot_fs = q_sens_fs + cfg.h_sub * m_delta  # [W/m²]
        #q_tot_fs_2 = self.k_f(st, -1) * (Tfs - st.T_ft[-2]) / dx

        #q_tot_x0 = q_sens_fs + cfg.h_sub * m_x0
        q_tot_x0_2 = self.k_f(st, 0) * (st.T_ft[1] - st.T_ft[0])/dx

        # Heat flow for steady state
        q_steady = q_sens_fs

        return q_tot_fs, q_tot_x0_2, m_delta, q_steady

    def segment_mass_flux_air_frost(self, cfg, geom, st, gs):
        """
        Integrierter Massenstrom Wasserdampf -> Frost eines Segments.

        Rückgabe:
            m_s_seg [kg/s]
        """
        q_tot_fs, q_tot_x0, m_delta, q_steady = self._segment_surface_fluxes(cfg, geom, st, gs)

        A_seg = geom.A_one_segment()
        m_s_seg = m_delta * A_seg  # [kg/s]

        return m_s_seg

    def segment_heat_flux_air_frost(self, cfg, geom, st, gs):
        """
        Integrierter Wärmestrom von der Luft in den Frost eines Segments.

        Rückgabe:
            Q_seg_fs [W]
            Q_seg_x0 [W]
        """
        q_tot_fs, q_tot_x0, m_delta, q_steady = self._segment_surface_fluxes(cfg, geom, st, gs)

        A_seg = geom.A_one_segment()
        Q_seg_fs = q_tot_fs * A_seg  # [W]
        Q_seg_x0 = q_tot_x0 * A_seg  # [W]

        # For steady state
        Q_steady = q_steady * A_seg

        return Q_seg_fs, Q_seg_x0, Q_steady