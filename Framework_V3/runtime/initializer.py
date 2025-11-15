import numpy as np
from Framework_V3.models.Frostmodell_V1 import Frostmodell_Edge

@staticmethod
def init_fields(cfg, st, gs):

    # Initializing frost thickness
    st.s_e = np.zeros(gs.ntheta, dtype=float)
    st.s_e[:] = 1.0e-6
    st.s_ft = 1.0e-6

    # Initializing edge domain [r, theta]
    st.T_e = np.zeros((gs.nr, gs.ntheta), dtype=float)
    st.T_e[:, :] = cfg.T_w # Add calculation for finn edge temperature
    st.rho_e = np.zeros((gs.nr, gs.ntheta), dtype=float)
    st.rho_e[:, :] = 25.0  # Define initial density
    st.w_e = np.zeros((gs.nr, gs.ntheta), dtype=float)
    st.w_e[:, :] = Frostmodell_Edge.w_sat_coolprop(cfg.T_w,cfg.p_a) # water vapor moisture content
    st.rho_a = np.zeros((gs.nr, gs.ntheta), dtype=float)
    st.rho_a[:, :] = Frostmodell_Edge.rho_a_dry_local(st.T_e[0,0],cfg.p_a)

    # Initializing finn and tube domain
    st.T_ft = np.zeros(gs.nx, dtype=float)
    st.T_ft[:] = cfg.T_w # mabe not correct ????
    st.rho_ft = np.zeros(gs.nx, dtype=float)
    st.rho_ft[:] = 25.0 # Define initial density
    st.w_ft = np.zeros(gs.nx, dtype=float)
    st.w_ft[:] = 1 # water vapor moisture content