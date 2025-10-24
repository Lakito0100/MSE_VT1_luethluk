import numpy as np
from Framework_V2.core.corrolations import DK

class Frostmodell_Edge:

    @staticmethod
    def Nu_edge(cfg, geom, theta):
        Re_d = DK.Re(cfg.v_a, geom.fin_pitch, cfg.v_kin)
        Pr = DK.Pr(cfg.v_kin, cfg.lam, cfg.c_p_a, cfg.rho)
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
        return hm * cfg.rho_amb * (cfg.w_a - w_fs)

    def m_dot_rho_f(self, cfg, st, gs, theta):
        Deff = self.D_eff(cfg, st, -1, theta)
        dr = st.s_e[theta] / gs.nr
        dwf_dr = (cfg.w_a - st.w_e[-1, theta]) / dr
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

        while tol < res_T and tol < res_w and niter > it:

            for theta in range(gs.ntheta+1):
                r = np.linspace(geom.fin_pitch, st.s_e[:,theta], gs.nr)
                dr = r[1] - r[0]
                for i in range(len(r)):
                    if i == 0:
                        A = # Define BC A, B and C
                    elif i == len(r):
                        A = # Define BC A, B and C
                    else:
                        A = (r[i]/r[i+1] - r[i]/r[i-1]) * self.D_eff(cfg, st, r[i], theta) * st.rho_a[r[i], theta]
                        B = -4 * (dr**2) * cfg.C * st.rho_a[r[i], theta]
                        C = (r[i]/r[i+1] - r[i]/r[i-1]) * self.k_eff(st, r[i], theta)

            # Calculation for T_f_new and w_f_new

            res_T = np.max(np.abs(T_f_new -T_f_old))
            res_w = np.max(np.abs(w_f_new - w_f_old))
            T_f_old = T_f_new
            w_f_old = w_f_new
            it += 1

        st.T_e = T_f_new
        st.w_e = w_f_new
        return it, res_T, res_w


