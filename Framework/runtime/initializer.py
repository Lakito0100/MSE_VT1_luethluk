import numpy as np
from Framework.models.Frostmodell_V1 import Frostmodell_Edge, Frostmodell_Finn_and_Tube


@staticmethod
def init_fields(cfg, st, gs):

    # Initializing frost thickness
    st.s_e = np.zeros(gs.ntheta, dtype=float)
    st.s_e[:] = 1.0e-6
    st.s_ft = 1.0e-6

    # Initializing edge domain [r, theta]
    st.T_e = np.zeros((gs.nr, gs.ntheta), dtype=float)
    st.T_e[:, :] = cfg.T_tube # Add calculation for finn edge temperature
    st.rho_e = np.zeros((gs.nr, gs.ntheta), dtype=float)
    st.rho_e[:, :] = 25.0  # Define initial density
    st.w_e = np.zeros((gs.nr, gs.ntheta), dtype=float)
    st.w_e[:, :] = Frostmodell_Edge.w_sat_coolprop(cfg.T_tube,cfg.p_a) # water vapor moisture content
    st.rho_a_e = np.zeros((gs.nr, gs.ntheta), dtype=float)
    st.rho_a_e[:, :] = Frostmodell_Edge.rho_a_dry_local(st.T_e[0,0],cfg.p_a)

    # Initializing finn and tube domain
    st.T_ft = np.zeros(gs.nx, dtype=float)
    st.T_ft[:] = cfg.T_tube
    st.rho_ft = np.zeros(gs.nx, dtype=float)
    st.rho_ft[:] = 25.0 # Define initial density
    st.w_ft = np.zeros(gs.nx, dtype=float)
    st.w_ft[:] = Frostmodell_Finn_and_Tube.w_sat_coolprop(cfg.T_tube,cfg.p_a)
    st.rho_a_ft = np.zeros(gs.nx, dtype=float)
    st.rho_a_ft[:] = Frostmodell_Finn_and_Tube.rho_a_dry_local(st.T_ft, cfg.p_a)