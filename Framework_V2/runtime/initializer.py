def init_fields(self, cfg, st):

    # Initializing frost thickness
    st.s_e = 0
    st.s_ft = 0

    # Initializing edge domain [r, theta]
    st.T_e[:, :] = cfg.T_w # Add calculation for finn edge temperature
    st.rho_e[:, :] = 1  # Define initial density
    st.rho_amb[:, :] = cfg.rho_amb
    st.w_e[:, :] = 1 # water vapor moisture content

    # Initializing finn and tube domain
    st.T_ft[:, :] = cfg.T_W # mabe not correct ????
    st.rho_ft[:, :] = 1 # Define initial density
    st.w_ft[:, :] = 1 # water vapor moisture content