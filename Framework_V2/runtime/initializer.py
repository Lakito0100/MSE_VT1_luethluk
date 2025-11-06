import numpy as np

@staticmethod
def init_fields(cfg, st, gs):

    # Initializing frost thickness
    st.s_e = np.zeros(gs.ntheta, dtype=float)
    st.s_e[:] = 1e-6
    st.s_ft = 1e-6

    # Initializing edge domain [r, theta]
    st.T_e = np.zeros((gs.nr, gs.ntheta), dtype=float)
    st.T_e[:, :] = cfg.T_w # Add calculation for finn edge temperature
    st.rho_e = np.zeros((gs.nr, gs.ntheta), dtype=float)
    st.rho_e[:, :] = 100  # Define initial density
    st.rho_a = np.zeros((gs.nr, gs.ntheta), dtype=float)
    st.rho_a[:, :] = cfg.rho_amb
    st.w_e = np.zeros((gs.nr, gs.ntheta), dtype=float)
    st.w_e[:, :] = 0.01 # water vapor moisture content

    # Initializing finn and tube domain
    st.T_ft = np.zeros(gs.nx, dtype=float)
    st.T_ft[:] = cfg.T_w # mabe not correct ????
    st.rho_ft = np.zeros(gs.nx, dtype=float)
    st.rho_ft[:] = 1 # Define initial density
    st.w_ft = np.zeros(gs.nx, dtype=float)
    st.w_ft[:] = 1 # water vapor moisture content