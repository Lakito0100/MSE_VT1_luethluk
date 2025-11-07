import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from dataclasses import field
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
        return hm * cfg.rho_amb * (cfg.w_amb - w_fs)

    def m_dot_rho_f(self, cfg, st, gs, theta):
        Deff = self.D_eff(cfg, st, -1, theta)
        dr = st.s_e[theta] / gs.nr
        dwf_dr = (st.w_e[-1, theta] - st.w_e[-2, theta]) / dr
        return Deff * cfg.rho_amb * dwf_dr

    def m_dot_s_f(self, cfg, geom, st, gs, theta):
        return self.m_dot_f(cfg, geom, st, theta) - self.m_dot_rho_f(cfg, st, gs, theta)

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

    def New_edge_state_seg(self, cfg, geom, st, gs, tol = 1e-6, niter = 1000):
        it = 0
        res_T = res_w = np.inf
        T_f_old = np.asarray(st.T_e, dtype=float).copy()
        w_f_old = np.asarray(st.w_e, dtype=float).copy()
        T_f_new = np.empty_like(T_f_old)
        w_f_new = np.empty_like(w_f_old)

        while (it < niter) and ((res_T > tol) or (res_w > tol)):

            for j in range(gs.ntheta):
                r_start = geom.fin_pitch*0.5
                r_end = float(st.s_e[j]) + r_start
                r = np.linspace(r_start, r_end, gs.nr)
                N = len(r)
                dr = r[1] - r[0]

                A_w = lil_matrix((N, N), dtype=float)
                b_w = np.zeros(N)
                A_T = lil_matrix((N, N), dtype=float)
                b_T = np.zeros(N)

                for i in range(len(r)):
                    if i == 0:
                        A_w[i,i] = -1
                        A_w[i,i+1] = 1
                        b_w[i] = 0

                        A_T[i,i] = 1
                        b_T[i] = cfg.T_w # Define T_edge ---------------------------------
                    elif i == N-1:
                        A_w[i,i] = self.D_eff(cfg,st,i,j)/dr + self.h_mass(cfg,geom,j)
                        A_w[i,i-1] = - self.D_eff(cfg,st,i,j)/dr
                        b_w[i] = self.h_mass(cfg,geom,j) * cfg.w_amb
                        #A_w[i, i] = 1.0
                        #b_w[i] = self.w_sat_coolprop(T_f_old[-1, j], cfg.p_a)

                        A_T[i,i] = -1
                        A_T[i,i-1] = 1
                        b_T[i] = self.q_dot_tot_fs(cfg, geom, st, gs, j) * dr / self.k_eff(st, i, j)
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

            res_T = np.max(np.abs(T_f_new - T_f_old))
            res_w = np.max(np.abs(w_f_new - w_f_old))
            T_f_old = T_f_new.copy()
            w_f_old = w_f_new.copy()
            it += 1

        st.T_e = T_f_new
        st.w_e = w_f_new

        # calculate s_e and rho_f

        N, ntheta = w_f_new.shape

        for theta in range(ntheta):
            for i in range(N):
                w_sat_i = self.w_sat_coolprop(st.T_e[i, theta], cfg.p_a)
                source = cfg.C * cfg.rho_amb * (st.w_e[i, theta] - w_sat_i)
                st.rho_e[i, theta] = np.clip(st.rho_e[i, theta] + source * cfg.dt, 1, cfg.rho_i)


        for theta in range(gs.ntheta):
            rho_fs = st.rho_e[-1, theta]
            m_dot_sf = self.m_dot_s_f(cfg, geom, st, gs, theta)
            st.s_e[theta] += m_dot_sf / rho_fs
            st.s_e[theta] = max(st.s_e[theta], 1e-6)

        return it, res_T, res_w