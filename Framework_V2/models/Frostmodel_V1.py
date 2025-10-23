import numpy as np
from Framework_V2.core.corrolations import DK

class Frostmodel_Edge:

    @staticmethod
    def Nu_edge(cfg, geom, phi):
        Re_d = DK.Re(cfg.v_a, geom.fin_pitch, cfg.v_kin)
        Pr = DK.Pr(cfg.v_kin, cfg.lam, cfg.c_p_a, cfg.rho)
        return 0.23 * (Re_d**0.466) * (Pr**(1/3)) * (0.7 + 1.06e-4 * (phi - 90)**2)

    def h_conv(self, cfg, geom, phi):
        Nu = self.Nu_edge(cfg, geom, phi)
        return Nu * cfg.lam / geom.fin_pitch

    def q_dot_sens_fs(self, cfg, geom, st, phi):
        T_fs = st.T_e[-1, phi]
        return self.h_conv(cfg, geom, phi) * (cfg.T_a - T_fs)

    def h_mass(self, cfg, geom, phi):
        h = self.h_conv(cfg, geom, phi)
        return h / (cfg.rho_amb * cfg.c_p_a)

    def m_dot_f(self, cfg, geom, st, phi):
        hm = self.h_mass(cfg, geom, phi)
        w_fs = st.w_e[-1, phi]
        return hm * cfg.rho_amb * (cfg.w_a - w_fs)

    def m_dot_rho_f(self, cfg, geom, st, gs, phi):
        Deff = self.D_eff(cfg, st, -1, phi)
        dr = st.s_e[phi] / gs.nr
        dwf_dr = (cfg.w_a - st.w_e[-1, phi]) / dr
        return Deff * cfg.rho_amb * dwf_dr

    def m_dot_s_f(self, cfg, geom, st, phi):
        return self.m_dot_f(cfg, geom, st, phi) - self.m_dot_rho_f(cfg, st, phi)

    def q_dot_lat_fs(self, cfg, geom, st, phi):
        return cfg.h_sub * self.m_dot_s_f(cfg, geom, st, phi)

    def q_dot_tot_fs(self, cfg, geom, st, phi):
        return self.q_dot_sens_fs(cfg, geom, st, phi) + self.q_dot_lat_fs(cfg, geom, st, phi)


    @staticmethod
    def D_eff(cfg, st, r, phi):
        numerator = cfg.D_std * (cfg.rho_i - st.rho_e[r,phi])
        denominator = cfg.rho_i - 0.58 * st.rho_e[r,phi]
        return numerator / denominator

    @staticmethod
    def k_eff(st, r, phi):
        return 0.132 + 3.13e-4 * st.rho_e[r,phi] + 1.6e-7 * (st.rho_e[r,phi])**2

    @staticmethod
    def New_edge_state_seg(st, tol = 1e-6, niter = 1000):
        it = 0
        res_T = res_w = np.inf
        T_f_old = np.asarray(st.T_e, dtype=float).copy()
        w_f_old = np.asarray(st.w_e, dtype=float).copy()
        while tol < res_T and tol < res_w and niter > it:

            # Calculation for T_f_new and w_f_new


            res_T = np.max(np.abs(T_f_new -T_f_old))
            res_w = np.max(np.abs(w_f_new - w_f_old))
            T_f_old = T_f_new
            w_f_old = w_f_new
            it += 1

        st.T_e = T_f_new
        st.w_e = w_f_new
        return it, res_T, res_w


